# mnist

Determine which digit (0-9) is contained in an image. The model is constructed in [Keras](https://keras.io)
and can be found in [mnist.py](mnist.py) (taken from their [examples](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)).

## Usage

First build the required container in the [parent directory](../).

```
$ docker build -t sjkaliski/infer/examples/mnist .
$ docker run -i -t --rm sjkaliski/infer/examples/mnist
```
