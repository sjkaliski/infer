package infer_test

import (
	"bytes"
	"io/ioutil"
	"log"

	"github.com/sjkaliski/infer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	m   *infer.Model
	buf bytes.Buffer
)

func ExampleNew() {
	model, err := ioutil.ReadFile("/path/to/model.pb")
	if err != nil {
		panic(err)
	}

	graph := tf.NewGraph()
	err = graph.Import(model, "")
	if err != nil {
		panic(err)
	}

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
}

func ExampleModel_FromImage() {
	results, err := m.FromImage(&buf, &infer.ImageOptions{
		IsGray: false,
	})
	if err != nil {
		panic(err)
	}
	log.Println(results)
}
