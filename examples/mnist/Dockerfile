FROM sjkaliski/infer/examples

WORKDIR $GOPATH/src/github.com/sjkaliski/infer
RUN mkdir -p examples/mnist
COPY ./ ./
WORKDIR examples/mnist

# Train model.
RUN pip install keras h5py pathlib
RUN python mnist.py

# Export model.
RUN apt-get install -y python3-pip
RUN pip3 install tensorflow keras h5py pathlib
RUN curl -O https://raw.githubusercontent.com/amir-abdi/keras_to_tensorflow/master/keras_to_tensorflow.py
RUN python3 keras_to_tensorflow.py -input_model_file=model.h5

# Build application.
ENV MODEL $GOPATH/src/github.com/sjkaliski/infer/examples/mnist/model.h5.pb
RUN go build -o app

ENTRYPOINT [ "./app" ]
