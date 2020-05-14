// Copyright (c) 2018 Daniel Augusto Rizzi Salvadori. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

// Package tsne implements t-Distributed Stochastic Neighbor Embedding (t-SNE), a prize-winning technique for
// dimensionality reduction particularly well suited for visualizing high-dimensional datasets.
package tsne

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

const (
	Epsilon                  = 1e-7
	GreaterThanZero          = 1e-12
	EntropyTolerance         = 1e-5
	MaxBinarySearchSteps     = 50
	InitialStandardDeviation = 1e-4
)

// TSNE is a t-Distributed Stochastic Neighbor Embedding (t-SNE) dimensionality reduction object.
type TSNE struct {
	n            int     // Number of datapoints
	dimsOut      int     // Number of dimensions in the low dimensional map
	perplexity   float64 // Perplexity target for the Gaussian kernels in high dimension
	learningRate float64 // Gradient descent learning rate
	verbose      bool    // If true, then TSNE outputs progress data to stdout
	maxIter      int     // Max number of gradient descent iterations

	P *mat.Dense // Matrix of pairwise affinities in the high dimensional space (Gaussian kernel)
	Q *mat.Dense // Matrix of pairwise affinities in the low dimensional space (t-Student kernel)
	Y *mat.Dense // The output embedding with dimsOut dimensions

	PlogP float64    // The constant portion of the KL divergence, computed only once
	dCdY  *mat.Dense // Gradient of the KL divergence with respect to the low dimensional map
}

// NewTSNE creates and returns a new t-SNE dimensionality reductor with the specified parameters.
func NewTSNE(dimensionsOut int, perplexity, learningRate float64, maxIter int, verbose bool) *TSNE {

	tsne := new(TSNE)
	tsne.dimsOut = dimensionsOut
	tsne.perplexity = perplexity
	tsne.learningRate = learningRate
	tsne.maxIter = maxIter
	tsne.verbose = verbose
	return tsne
}

// EmbedData initializes the pairwise affinity matrix P with the similarity
// probabilities calculated based on the provided data matrix and runs t-SNE.
// It returns the generated embedding.
func (tsne *TSNE) EmbedData(X mat.Matrix, stepFunc func(iter int, divergence float64, embedding mat.Matrix) bool) mat.Matrix {

	D := SquaredDistanceMatrix(X)
	return tsne.EmbedDistances(D, stepFunc)
}

// InitDistances initializes the pairwise affinity matrix P with the similarity
// probabilities calculated based on the provided (squared) distance matrix and runs t-SNE.
// It returns the generated embedding.
func (tsne *TSNE) EmbedDistances(D mat.Matrix, stepFunc func(iter int, divergence float64, embedding mat.Matrix) bool) mat.Matrix {

	// Verify that D is square
	n, d := D.Dims()
	if n != d {
		panic("squared distance matrix is not square")
	}

	tsne.n = n
	tsne.d2p(D, EntropyTolerance, tsne.perplexity)
	tsne.initSolution()
	tsne.run(stepFunc)
	return tsne.Y
}

// initSolution initializes the t-SNE solution.
func (tsne *TSNE) initSolution() {

	// Allocate the embedding matrix (result)
	tsne.Y = mat.NewDense(tsne.n, tsne.dimsOut, nil)
	tsne.Y.Apply(func(i, j int, v float64) float64 {
		return RandNormal(0, InitialStandardDeviation)
	}, tsne.Y)

	// Allocate gradient matrix
	tsne.dCdY = mat.NewDense(tsne.n, tsne.dimsOut, nil)

	// Compute and store the constant portion of the KL divergence
	PlogP := mat.NewDense(tsne.n, tsne.n, nil)
	PlogP.CloneFrom(tsne.P)
	PlogP.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, PlogP)
	PlogP.MulElem(tsne.P, PlogP)
	tsne.PlogP = mat.Sum(PlogP)
}

// d2p computes the P matrix based on a (squared) distance matrix D.
// It performs a binary search to obtain a similarity probability for each pairwise distance
// in such a way that each Gaussian kernel has the same perplexity (specified).
// D should be a squared distance matrix, it should be square and symmetric.
func (tsne *TSNE) d2p(D mat.Matrix, tol, perplexity float64) {

	// The target entropy of the gaussian kernels is the log of the target perplexity
	Htarget := math.Log(perplexity)

	// Allocate the probability matrix
	tsne.P = mat.NewDense(tsne.n, tsne.n, nil)

	dDense := mat.DenseCopyOf(D)
	// Loop over all data points
	for i := 0; i < tsne.n; i++ {
		// Print progress
		if tsne.verbose && i%500 == 0 {
			fmt.Printf("Computing P-values for point %d of %d...\n", i, tsne.n)
		}
		// Perform a binary search for the Gaussian kernel precision (beta)
		// such that the entropy (and thus the perplexity) of the distribution
		// given the data is the same for all data points
		betaMin := math.Inf(-1)
		betaMax := math.Inf(1)
		beta := float64(1) // initial value of precision
		Di := dDense.RowView(i)
		for tries := 0; tries < MaxBinarySearchSteps; tries++ {
			// Compute raw probabilities with beta precision (along with sum of all raw probabilities)
			pSum := float64(0)
			for j := 0; j < Di.Len(); j++ {
				var p float64
				if i == j {
					p = 0
				} else {
					p = math.Exp(-Di.AtVec(j) * beta)
				}
				tsne.P.Set(i, j, p)
				pSum += p
			}
			// Normalize probabilities and compute entropy H
			var H float64 // Distribution entropy
			for j := 0; j < Di.Len(); j++ {
				var p float64
				if pSum == 0 {
					p = 0
				} else {
					p = tsne.P.At(i, j) / pSum
				}
				tsne.P.Set(i, j, p)
				if p > Epsilon {
					H -= p * math.Log(p)
				}
			}
			// Adjust beta to move H closer to Htarget
			Hdiff := H - Htarget
			if Hdiff > 0 {
				// Entropy is too high (distribution too spread-out)
				// So we need to increase the precision
				betaMin = beta // Move up the bounds
				if betaMax == math.Inf(1) {
					beta = beta * 2
				} else {
					beta = (beta + betaMax) / 2
				}
			} else {
				// Entropy is too low - need to decrease precision
				betaMax = beta // Move down the bounds
				if betaMin == math.Inf(-1) {
					beta = beta / 2
				} else {
					beta = (beta + betaMin) / 2
				}
			}
			// If current entropy is within specified tolerance - we are done with this data point
			if math.Abs(Hdiff) < tol {
				break
			}
		}
	}
	// Symmetrize and normalize P
	tsne.P.Add(tsne.P, tsne.P.T())
	tsne.P.Scale(1/float64(2*tsne.n), tsne.P)
	tsne.P.Apply(func(i, j int, v float64) float64 {
		return math.Max(v, GreaterThanZero)
	}, tsne.P)
}

// run performs batch gradient descent to reduce the Kullback-Leibler divergence between P and Q,
// the high dimensional affinities and the low dimensional affinities respectively.
func (tsne *TSNE) run(stepFunc func(iter int, divergence float64, embedding mat.Matrix) bool) {

	for iter := 0; iter < tsne.maxIter; iter++ {
		// Compute KL divergence and update the gradient matrix
		divergence := tsne.costGradient(tsne.P, tsne.Y)
		// Step in the direction of negative gradient (times the learning rate)
		scaledGrad := mat.NewDense(tsne.n, tsne.dimsOut, nil)
		scaledGrad.CloneFrom(tsne.dCdY)
		scaledGrad.Scale(tsne.learningRate, scaledGrad)
		tsne.Y.Sub(tsne.Y, scaledGrad)
		// Reproject Y to have zero mean
		ymean := make([]float64, tsne.dimsOut)
		for i := 0; i < tsne.n; i++ {
			for d := 0; d < tsne.dimsOut; d++ {
				ymean[d] += tsne.Y.At(i, d)
			}
		}
		tsne.Y.Apply(func(i, j int, v float64) float64 {
			return v - ymean[j]/float64(tsne.n)
		}, tsne.Y)
		// If provided, call user step function
		if stepFunc != nil {
			stop := stepFunc(iter, divergence, tsne.Y)
			if stop {
				break
			}
		}
	}
}

// costGradient computes the Kullback-Leibler divergence between
// P and the Student-t based joint probability distribution Q.
// It also computes the gradient of the divergence with respect to the
// low-dimensional map Y (the desired output of t-SNE).
func (tsne *TSNE) costGradient(P, Y mat.Matrix) float64 {

	// Initialize divergence and gradient matrix
	var divergence float64
	n, d := Y.Dims()
	// Compute Q matrix of low dimensional affinities (unnormalized at first)
	Qu := mat.DenseCopyOf(SquaredDistanceMatrix(Y))
	Qu.Apply(func(i, j int, v float64) float64 {
		if i == j { // Clear diagonal
			return 0
		} else { // Student t-distribution
			return 1 / (1 + v)
		}
	}, Qu)
	// Normalize Q matrix
	sumQu := mat.Sum(Qu)
	Q := mat.NewDense(n, n, nil)
	Q.CloneFrom(Qu)
	Q.Scale(1/sumQu, Q)
	Q.Apply(func(i, j int, v float64) float64 {
		return math.Max(v, GreaterThanZero)
	}, Q)
	// Compute the the non-constant portion of the divergence
	logQ := mat.NewDense(n, n, nil)
	logQ.CloneFrom(Q)
	logQ.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, logQ)
	PlogQmat := mat.NewDense(n, n, nil)
	PlogQmat.MulElem(P, logQ)
	// Compute the divergence
	PlogQ := mat.Sum(PlogQmat)
	divergence = tsne.PlogP - PlogQ
	// Compute the matrix of scalar multiples for scaling the 3D difference matrix prior to squashing
	mult := mat.NewDense(n, n, nil)
	mult.Sub(P, Q)
	mult.Scale(4, mult)
	mult.MulElem(mult, Qu)
	// Compute the gradient
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			m := mult.At(r, c)
			for k := 0; k < d; k++ {
				yDiff := Y.At(r, k) - Y.At(c, k)
				orig := tsne.dCdY.At(r, k)
				tsne.dCdY.Set(r, k, orig+m*yDiff)
			}
		}
	}
	return divergence
}

//
// Utility functions
//

// RandNormal samples from a Gaussian distribution
// with the specified mean and standard deviation.
func RandNormal(mu, std float64) float64 {

	return mu + rand.NormFloat64()*std
}

// Diagonal returns the diagonal elements of the specified matrix as a Vector.
func Diagonal(a mat.Matrix) mat.Vector {

	n, d := a.Dims()
	if n != d {
		panic("matrix is not square")
	}
	dense := mat.DenseCopyOf(a)
	diagData := make([]float64, n)
	for i := 0; i < n; i++ {
		diagData[i] = dense.At(i, i)
	}
	return mat.NewVecDense(n, diagData)
}

// SquaredDistanceMatrix computes the squared distance matrix for row vectors in X.
// Returns a matrix where the {i, j}-th element is the squared euclidean distance between the i-th and j-th rows in X.
//
// D(x, y)^2 = ∥y – x∥^2 = x'x + y'y – 2 x'y
//
func SquaredDistanceMatrix(X mat.Matrix) mat.Matrix {

	n, _ := X.Dims()
	// Allocate the distance matrix
	D := mat.NewDense(n, n, nil)
	// Multiply X transpose by X (to obtain the x'y term)
	xy := mat.NewDense(n, n, nil)
	xy.Mul(X, X.T())
	diagVec := Diagonal(xy) // Obtain the diagonal before scaling
	xy.Scale(-2, xy)
	// Compute the x'x and the y'y terms
	xx := mat.NewDense(n, n, nil)
	yy := mat.NewDense(n, n, nil)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			xx.Set(r, c, diagVec.AtVec(r))
			yy.Set(r, c, diagVec.AtVec(c))
		}
	}
	// Compute the final sum: x'x + y'y – 2 x'y
	D.Add(xx, yy)
	D.Add(D, xy)
	return D
}
