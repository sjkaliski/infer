package internal

import (
	"errors"
	"image"
	"image/color"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	// ErrInvalidImageTensor occurs when the tensor does not exhibit the
	// structure or appearance of an image.
	ErrInvalidImageTensor = errors.New("invalid image tensor")
)

// TensorToImage converts a tf.Tensor into an image.Image. It
// determines the type of image (RGB vs Gray) and returns an
// error if the Tensor is not a valid image.
func TensorToImage(t *tf.Tensor) (image.Image, error) {
	dim := t.Shape()

	if ok := 3 <= len(dim) && len(dim) <= 4; !ok {
		return nil, ErrInvalidImageTensor
	}

	var w, h, c int = 0, 0, 0
	var data [][][]float32

	// Based on the dimension of the input tensor, we can
	// determine if it's a batch of images or a singular image.
	if len(dim) == 3 {
		w, h, c = int(dim[0]), int(dim[1]), int(dim[2])
		data = t.Value().([][][]float32)
	} else if len(dim) == 4 {
		w, h, c = int(dim[1]), int(dim[2]), int(dim[3])
		data = t.Value().([][][][]float32)[0]
	}

	var img image.Image
	rect := image.Rect(0, 0, w, h)
	if c == 1 {
		img = image.NewGray(rect)
	} else if c == 3 {
		img = image.NewRGBA(rect)
	} else {
		return nil, ErrInvalidImageTensor
	}

	for y := 0; y < len(data); y++ {
		for x := 0; x < len(data[y]); x++ {
			p := data[y][x]

			if c == 1 {
				img.(*image.Gray).Set(x, y, color.Gray{uint8(p[0])})
			} else if c == 3 {
				img.(*image.RGBA).Set(x, y, color.RGBA{uint8(p[0]), uint8(p[1]), uint8(p[2]), 255})
			}
		}
	}

	return img, nil
}
