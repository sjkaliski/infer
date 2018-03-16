package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/sjkaliski/infer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// Load model.
	model, err := ioutil.ReadFile(os.Getenv("MODEL"))
	if err != nil {
		panic(err)
	}

	// Generate new graph.
	graph := tf.NewGraph()
	err = graph.Import(model, "")
	if err != nil {
		panic(err)
	}

	// Create new inference model.
	m, _ := infer.New(&infer.Model{
		Graph: graph,
		Input: &infer.Input{
			Key:        "conv2d_1_input",
			Dimensions: []int32{28, 28},
		},
		Output: &infer.Output{
			Key: "output_node0",
		},
	})

	img, err := os.Open("sample.png")
	if err != nil {
		panic(err)
	}
	defer img.Close()

	opts := &infer.ImageOptions{
		IsGray: true,
	}
	predictions, err := m.FromImage(img, opts)
	if err != nil {
		panic(err)
	}

	fmt.Println("Prediction:", predictions[0].Class)
	fmt.Println("Confidence:", predictions[0].Score)
}
