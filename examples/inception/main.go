package main

import (
	"encoding/json"
	_ "image/png"
	"io/ioutil"
	"net/http"
	"os"
	"sort"
	"strings"

	"github.com/sjkaliski/infer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	m      *infer.Model
	labels []string
)

func handler(w http.ResponseWriter, r *http.Request) {
	// Evaluate image.
	opts := &infer.ImageOptions{
		IsGray: false,
	}
	result, err := m.FromImageWithContext(r.Context(), r.Body, opts)
	if err != nil {
		panic(err)
	}

	// Convert results to prediction.
	probabilities := result.([][]float32)[0][:len(labels)-1]
	predictions := make(infer.Predictions, len(probabilities))
	for i, p := range probabilities {
		predictions[i] = &infer.Prediction{
			Value:      labels[i],
			Confidence: p,
		}
	}

	sort.Sort(sort.Reverse(predictions))
	predictions = predictions[:10]

	data, err := json.Marshal(predictions)
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

	labelsStr := string(labelFile)
	labels = strings.Split(labelsStr, "\n")

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		panic(err)
	}

	m, _ = infer.New(&infer.Model{
		Graph: graph,
		Input: &infer.Input{
			Key:        "input",
			Dimensions: []int32{299, 299},
		},
		Output: &infer.Output{
			Key:        "InceptionV3/Predictions/Softmax",
			Dimensions: [][]float32{},
		},
	})

	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
