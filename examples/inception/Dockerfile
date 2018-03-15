FROM sjkaliski/infer/examples

WORKDIR $GOPATH/src/inception
COPY ./ ./

RUN curl -O "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
RUN unzip inception5h.zip

ENV MODEL tensorflow_inception_graph.pb
ENV LABELS imagenet_comp_graph_label_strings.txt

RUN go get && go build -o app
EXPOSE 8080
ENTRYPOINT [ "./app" ]
