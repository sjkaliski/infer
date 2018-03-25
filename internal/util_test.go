package internal

import (
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestTensorToImage(t *testing.T) {
	// invalid input.
	input, _ := tf.NewTensor([]float32{0.0})
	_, err := TensorToImage(input)
	if err != ErrInvalidImageTensor {
		t.Fatalf("Expected ErrInvalidImageTensor, instead got %s", err)
	}
}
