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

var (
	// ErrValidInputOutputRequired occurs if an invalid Input or Output is provided.
	ErrValidInputOutputRequired = errors.New("a valid Input/Putput is required")
)

// Model is an ML model. It's composed of a computation graph and
// an Input and Output. It provides methods for running inferences
// usnig it's Graph, abiding by it's Input/Output.
type Model struct {
	Graph  *tf.Graph
	Input  *Input
	Output *Output
}

// Input is an ML layer. It is identified by a key and has dimensions
// The dimensions are used to augment or resize the output as appropriate.
type Input struct {
	// Key represents a layer in a TensorFlow model. The selection of a Key
	// determines "where" the Input/Output occurs in the Graph.
	Key string

	// Dimensions represents the size of the input. It can be of any type but
	// must contains the values expected by the layer. It may be used to
	// augment or resize the input so that it conforms to the specified layer.
	Dimensions interface{}
}

// Output is an ML layer. It is identified by a key and has dimensions
// The dimensions are used to augment or resize the output as appropriate.
type Output struct {
	// Key represents a layer in a TensorFlow model. The selection of a Key
	// determines "where" the Input/Output occurs in the Graph.
	Key string

	// Dimensions represents the size of the input. It can be of any type but
	// must contains the values expected by the layer. It may be used to
	// augment or resize the input so that it conforms to the specified layer.
	Dimensions interface{}
}

// New returns a new Model.
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
// Note: for now it is sparse, but included to keep the method signature
// consistent as new options become available.
type ImageOptions struct {
	// IsGray represents whether the Model expects the input image
	// to be grayscale or not. Specifically, whether the image has
	// 3 channels or 1 channel.
	IsGray bool
}

// FromImageWithContext evaluates an image with context. Optional ImageOptions
// can be included to dictate the pre-processing of the input image. The method
// returns an interface of results which can be cast to the appropriate type.
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
	boxes := op.Const(scope.SubScope("boxes"), [][]float32{{0, 0, 1, 1}})
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
