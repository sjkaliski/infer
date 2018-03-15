# inception

This is an example http API that takes an input image and returns a list of predictions ordered by confidence. It uses the Inception v1 model. It auto-detects the image format and crops / resizes per the model requirements.

## Setup

1. Build the required container in the [parent directory](../).

2. Build and run the example.

```
$ docker build -t sjkaliski/infer/examples/inception .
$ docker run --rm -it -p 8080:8080 sjkaliski/infer/examples/inception
```

## Usage

```
curl -X "POST" "http://127.0.0.1:8080/" -H 'Content-Type: image/png' input.png
```
