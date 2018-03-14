package infer

import (
	"bytes"
	"context"
	"errors"
	"image"
	"io"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	// Required to support image decoding.
	_ "image/jpeg"
	_ "image/png"
)

// Errors.
var (
	ErrValidInputOutputRequired = errors.New("a valid input/output is required")
)

// Model is an ML model.
type Model struct {
	Graph  *tf.Graph
	Labels []interface{}
	Input  *Input
	Output *Output
}

// Input is an ML input.
type Input struct {
	Key        string
	Dimensions interface{}
}

// Output is an ML output.
type Output struct {
	Key        string
	Dimensions interface{}
}

// New returns a new infer.Model.
func New(model *Model) (*Model, error) {
	if model.Input == nil || model.Output == nil {
		return nil, ErrValidInputOutputRequired
	}

	if model.Input.Key == "" || model.Output.Key == "" {
		return nil, ErrValidInputOutputRequired
	}

	return model, nil
}

// ImageOptions represent configurable options when evaluating images.
type ImageOptions struct {
	IsGray bool
}

// FromImageWithContext evaluates an image.
func (m *Model) FromImageWithContext(ctx context.Context, r io.Reader, opts *ImageOptions) (interface{}, error) {
	if ctx == nil {
		panic("nil context")
	}

	c := make(chan error)
	var results interface{}
	go func() {
		var err error
		results, err = m.fromImage(r, opts)
		c <- err
	}()

	for {
		select {
		case <-ctx.Done():
			return results, ctx.Err()
		case err := <-c:
			return results, err
		}
	}
}

// FromImage evaluates an image.
func (m *Model) FromImage(r io.Reader, opts *ImageOptions) (interface{}, error) {
	return m.FromImageWithContext(context.Background(), r, opts)
}

func (m *Model) fromImage(r io.Reader, opts *ImageOptions) (interface{}, error) {
	var imgBuf, tensorBuf bytes.Buffer
	w := io.MultiWriter(&imgBuf, &tensorBuf)

	_, err := io.Copy(w, r)
	if err != nil {
		return nil, err
	}

	// Determine image type.
	_, typ, err := image.Decode(&imgBuf)
	if err != nil {
		return nil, err
	}

	// Create tensor from image as string. DecodePng/Jpeg expects this.
	tensor, err := tf.NewTensor(tensorBuf.String())
	if err != nil {
		return nil, err
	}

	scope := op.NewScope()
	input := op.Placeholder(scope, tf.String)

	var channels int64 = 3
	if opts.IsGray {
		channels = 1
	}

	// Create a decoder operation based on image type.
	var decoder tf.Output
	switch typ {
	case "png":
		decoder = op.DecodePng(scope, input, op.DecodePngChannels(channels))
		break
	case "jpeg":
		decoder = op.DecodeJpeg(scope, input, op.DecodeJpegChannels(channels))
		break
	default:
		return nil, errors.New("invalid image")
	}

	// Crop/resize image. This bilinearly resizes the input image.
	// op.CropAndResize provides support for cropping/resizing multiple "boxes"
	// from the input image, however we only require one. Linked below
	// are the python docs, which are more thorough:
	// https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
	cropSize := op.Const(scope.SubScope("crop_size"), m.Input.Dimensions)
	boxes := op.Const(scope.SubScope("boxes"), [][]float32{[]float32{0, 0, 1, 1}})
	boxInd := op.Const(scope.SubScope("box_ind"), []int32{0})
	images := op.ExpandDims(
		scope,
		op.Cast(scope, decoder, tf.Float),
		op.Const(scope.SubScope("batch"), int32(0)),
	)
	cropAndResize := op.CropAndResize(
		scope,
		images,
		boxes,
		boxInd,
		cropSize,
	)

	// Create graph for cropping, initiate session, and execute.
	graph, err := scope.Finalize()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	cropped, err := session.Run(
		map[tf.Output]*tf.Tensor{
			input: tensor,
		},
		[]tf.Output{cropAndResize},
		nil)
	if err != nil {
		return nil, err
	}

	// Pass cropped image to primary model for evaluation.
	result, err := m.eval(cropped[0])
	if err != nil {
		return nil, err
	}

	return result[0].Value(), nil
}

// eval executes the inference using an input against the model graph.
func (m *Model) eval(input *tf.Tensor) ([]*tf.Tensor, error) {
	session, err := tf.NewSession(m.Graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	return session.Run(
		map[tf.Output]*tf.Tensor{
			m.Graph.Operation(m.Input.Key).Output(0): input,
		},
		[]tf.Output{
			m.Graph.Operation(m.Output.Key).Output(0),
		},
		nil,
	)
}
