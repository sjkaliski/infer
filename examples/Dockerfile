FROM tensorflow/tensorflow

# Install Git
RUN apt-get update
RUN apt-get install -y git

# Install Go
RUN curl -O https://dl.google.com/go/go1.10.linux-amd64.tar.gz
RUN tar -xzvf go1.10.linux-amd64.tar.gz
RUN mv go /usr/local
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

WORKDIR $GOPATH/src

# Install TensorFlow
RUN curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.6.0.tar.gz" | tar -C /usr/local -xz
RUN ldconfig
RUN go get github.com/tensorflow/tensorflow/tensorflow/go
