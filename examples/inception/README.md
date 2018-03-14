# inception

This is an example http API that takes an input image and returns a list of predictions ordered by confidence. It uses the Inception v3 model. It auto-detects the image format and crops / resizes per the model requirements.

## Setup

First build the required container in the [parent directory](../).

```
$ docker build -t sjkaliski/infer/examples/inception -f examples/inception/Dockerfile .
$ docker run --rm -it -p 8080:8080 sjkaliski/infer/examples/inception
```

## Usage

```
curl -X "POST" "http://127.0.0.1:8080/" -H 'Content-Type: image/png' input.png
```
