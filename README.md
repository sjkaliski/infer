# infer

Infer is a Go package for running predicitions in TensorFlow models.

[![Build Status](https://travis-ci.org/sjkaliski/infer.png)](https://travis-ci.org/sjkaliski/infer)
[![GoDoc](https://godoc.org/github.com/sjkaliski/infer?status.svg)](https://godoc.org/github.com/sjkaliski/infer)

## Overview

This package provides abstractions for running inferences in TensorFlow models for common types. At the moment it only has methods for images, however in the future it can certainly support more.

## Getting Started

The easiest way to get going is looking at some examples, two have been provided:

1. [Image Recognition API using Inception](examples/inception).
2. [MNIST](examples/mnist)

## Setup

This package requires 

- [Go](https://golang.org/dl/)
- [TensorFlow for Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)

Installation instructions can be found [here](https://www.tensorflow.org/install/install_go). Additionally, a [Dockerfile](examples/Dockerfile) has been included which can be used to run the [examples](examples).

## Usage

### Overview

To use infer, a TensorFlow [Graph](https://www.tensorflow.org/programmers_guide/graphs) is required, as well as a defined Input and Output.

```go
m, _ = infer.New(&infer.Model{
  Graph: graph,
  Input: &infer.Input{
    Key:        "input",
    Dimensions: []int32{100, 100},
  },
  Output: &infer.Output{
    Key: "output",
  },
})
```

Once a new model is defined, inferences can be executed.

```go
results, _ := m.FromImage(file, &infer.ImageOptions{})
```

Results are just an `interface{}` which can be cast as appropriate to inspect results.

### Graph

An `infer.Model` requires a `tf.Graph`. The Graph defines the computations required to determine an output based on a provided input. The Graph can be included in two ways:

1. Create the Graph using Go in your application.
2. Load an existing model.

For the latter, an existing model (containing the graph and weights) can be loaded using Go:

```go
model, _ := ioutil.ReadFile("/path/to/model.pb")
graph := tf.NewGraph()
graph.Import(model, "")
```

For more information on TensorFlow model files, [see here](https://www.tensorflow.org/extend/tool_developers/).

### Input & Output

`infer.Input` and `infer.Output` describe [TensorFlow layers](https://www.tensorflow.org/tutorials/layers). In practice this is the layer the input data should be fed to and the layer from which to fetch results.

Each require a `Key`. This is the unique identifier (name) in the TensorFlow graph for that layer. To see a list of layers and type, the following can be run:

```go
ops := graph.Operations()
for _, o := range ops {
  log.Println(o.Name(), o.Type())
}
```

If you're not using a pre-trained model, the layers can be named, which can ease in identifying the appropriate layers.

### Analyzing Results

At the moment, the resulting value (not the Tensor itself) is returned. This can be cast as necessary. For example, for MNIST

```go
results, err := m.FromImage(img, opts)
if err != nil {
  panic(err)
}

possibilities := results.([][]float32)[0]
```

The results are a slice of a slice (length 10). The wrapping slice contains sets of results (infer does not support batch analysis yet). The interior slice is an array of 10 values ranging from 0.0-1.0, each indicating the probability of a certain number contained in the image.

For convenience, an `infer.Predictions` type is included which implements `sort.Sort` and can be used to encapsulate results. Note: in the future these methods may return `infer.Predictions` directly.
