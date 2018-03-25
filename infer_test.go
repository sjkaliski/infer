package infer

import (
	"testing"
)

func TestNew(t *testing.T) {
	_, err := New(&Model{})
	if err != ErrValidInputOutputRequired {
		t.Fatalf("Expected ErrValidInputOutputRequired, instead got %s", err)
	}

	_, err = New(&Model{
		Input: &Input{
			Key: "",
		},
		Output: &Output{
			Key: "",
		},
	})
	if err != ErrValidInputOutputRequired {
		t.Fatalf("Expected ErrValidInputOutputRequired, instead got %s", err)
	}
}
