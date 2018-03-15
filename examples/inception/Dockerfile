FROM sjkaliski/infer/examples

RUN curl -O "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
RUN tar -xzvf inception_v3_2016_08_28_frozen.pb.tar.gz

ENV MODEL $GOPATH/src/inception_v3_2016_08_28_frozen.pb
ENV LABELS $GOPATH/src/imagenet_slim_labels.txt

WORKDIR $GOPATH/src/inception
COPY ./ ./

RUN go get && go build -o app
EXPOSE 8080
ENTRYPOINT [ "./app" ]
