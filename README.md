# go-tsne

A Go implementation of [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/), a prize-winning technique for dimensionality reduction particularly well suited for visualizing high-dimensional datasets.

<p float="left">
  <img src="https://github.com/danaugrs/go-tsne/blob/master/examples/mnist2d/mnist2d.gif" alt="mnist2d" width="49%" />
  <img src="https://github.com/danaugrs/go-tsne/blob/master/examples/mnist3d/mnist3d.gif" alt="mnist3d" width="49%" /> 
</p>

### Usage
Import this library:
```Go
    import "github.com/danaugrs/go-tsne/tsne"
```
Create the TSNE object:
```Go
    t := tsne.NewTSNE(2, 300, 300, true)
```
The parameters are
* Number of output dimensions
* Perplexity
* Max number of iterations
* Verbosity

There are two ways to start the t-SNE embedding optimization. The regular way is to provide an `n` by `d` matrix where each row is a datapoint and each column is a dimension:
```Go
    Y := t.EmbedData(X, nil)
```
The alternative is to provide a distance matrix directly:
```Go
    Y := t.EmbedDistances(D, nil)
```
In either case, the returned matrix `Y` will contain the final embedding.

For more fine-grained control, a step function can be provided in either case:
```Go
    Y := t.EmbedData(X, func(iter int, divergence float64, embedding mat.Matrix) bool {
    	fmt.Printf("Iteration %d: divergence is %v\n", iter, divergence)
    	return false
    })
```
The step function has access to the iteration, the current divergence, and the embedding optimized so far. Its return value can be set to true to indicate that the optimization should stop.

### Examples
Two examples are provided - `mnist2d` and `mnist3d`. They both use the same data - a subset of [MNIST](http://yann.lecun.com/exdb/mnist/) with 2500 handwritten digits. `mnist2d` generates plots throughout the optimization process, and `mnist3d` shows the optimization happening in real-time, in 3D. `mnist3d` depends on [G3N](https://github.com/g3n/engine).
To run an example, `cd` to the example's directory and `go run` it, e.g:
```
    cd examples/mnist2d
    go run mnist2d
```

### Support
I hope you enjoy using and learning from go-tsne as much as I enjoyed writing it.

If you come across any issues, please [report them](https://github.com/danaugrs/go-tsne/issues).
