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
	if err := graph.Import(model, ""); err != nil {
		panic(err)
	}

	// Create new inference model.
	m, _ := infer.New(&infer.Model{
		Graph: graph,
		Input: &infer.Input{
			Key:        "input_input",
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

	// Evaluate input.
	opts := &infer.ImageOptions{
		IsGray: true,
	}
	results, err := m.FromImage(img, opts)
	if err != nil {
		panic(err)
	}

	possibilities := results.([][]float32)[0]

	maxVal := float32(0)
	maxIdx := 0

	for i, p := range possibilities {
		if p > maxVal {
			maxVal = p
			maxIdx = i
		}
	}

	fmt.Println(maxIdx)
}
