// Copyright (c) 2018 Daniel Augusto Rizzi Salvadori. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/danaugrs/go-tsne/examples/data"
	"github.com/danaugrs/go-tsne/tsne"
	"github.com/sjwhitworth/golearn/pca"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {

	// Parameters
	pcaComponents := 50
	perplexity := float64(300)
	learningRate := float64(300)
	fmt.Println("Hi! Check the 'output' directory to see the plots generated while t-SNE runs.")
	fmt.Printf("PCA Components = %v\nPerplexity = %v\nLearning Rate = %v\n\n", pcaComponents, perplexity, learningRate)

	// Initialize the random seed
	rand.Seed(int64(time.Now().Nanosecond()))

	// Load a subset of MNIST with 2500 records
	X, Y := data.LoadMNIST()

	// Pre-process the data with PCA (Principal Component Analysis)
	// reducing the number of dimensions from 784 (28x28) to the top pcaComponents principal components
	Xdense := mat.DenseCopyOf(X)
	pcaTransform := pca.NewPCA(pcaComponents)
	Xt := pcaTransform.FitTransform(Xdense)

	// Create output directory if not exists
	os.Mkdir("output", 0770)

	// Create the t-SNE dimensionality reductor and embed the MNIST data in 2D
	t := tsne.NewTSNE(2, perplexity, learningRate, 300, true)
	t.EmbedData(Xt, func(iter int, divergence float64, embedding mat.Matrix) bool {
		if iter%10 == 0 {
			fmt.Printf("Iteration %d: divergence is %v\n", iter, divergence)
			plotY2D(t.Y, Y, fmt.Sprintf("output/tsne-1-iteration-%d.png", iter))
		}
		return false
	})

}

// plotY2D plots the 2D embedding Y and saves an image of the plot with the specified filename.
func plotY2D(Y, labels mat.Matrix, filename string) {

	// Create a plotter for each of the 10 digit classes
	classes := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	classPlotters := make([]plotter.XYs, len(classes))
	for i := range classes {
		classPlotters[i] = make(plotter.XYs, 0)
	}

	// Populate the class plotters with their respective data
	n, _ := Y.Dims()
	for i := 0; i < n; i++ {
		label := int(labels.At(i, 0))
		classPlotters[label] = append(classPlotters[label], plotter.XY{Y.At(i, 0), Y.At(i, 1)})
	}

	// Create a plot and update title and axes
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "t-SNE MNIST"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Convert plotters to array of empty interfaces
	classPlottersEI := make([]interface{}, len(classes))
	for i := range classes {
		classPlottersEI[i] = classPlotters[i]
	}

	// Add class plotters to the plot
	err = plotutil.AddScatters(p, classPlottersEI...)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file
	err = p.Save(8*vg.Inch, 8*vg.Inch, filename)
	if err != nil {
		panic(err)
	}
}
