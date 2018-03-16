package main

import (
	"encoding/json"
	_ "image/png"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/sjkaliski/infer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	m      *infer.Model
	labels []string
)

func handler(w http.ResponseWriter, r *http.Request) {
	opts := &infer.ImageOptions{
		IsGray: false,
	}

	predictions, err := m.FromImageWithContext(r.Context(), r.Body, opts)
	if err != nil {
		panic(err)
	}

	data, err := json.Marshal(predictions[:10])
	if err != nil {
		panic(err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}

func main() {
	model, err := ioutil.ReadFile(os.Getenv("MODEL"))
	if err != nil {
		panic(err)
	}

	labelFile, err := ioutil.ReadFile(os.Getenv("LABELS"))
	if err != nil {
		panic(err)
	}
	labels = strings.Split(string(labelFile), "\n")

	graph := tf.NewGraph()
	err = graph.Import(model, "")
	if err != nil {
		panic(err)
	}

	m, _ = infer.New(&infer.Model{
		Graph:   graph,
		Classes: labels,
		Input: &infer.Input{
			Key:        "input",
			Dimensions: []int32{224, 224},
		},
		Output: &infer.Output{
			Key:        "output",
			Dimensions: [][]float32{},
		},
	})

	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
