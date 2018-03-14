examples-inception:
	@docker build -t sjkaliski/infer/examples/inception -f examples/inception/Dockerfile .
	@docker run --rm -it -p 9090:8080 sjkaliski/infer/examples/inception

examples-mnist:
	@docker build -t sjkaliski/infer/examples/mnist -f examples/mnist/Dockerfile .
	@docker run --rm -it sjkaliski/infer/examples/mnist

.PHONY: examples-inception examples-mnist
