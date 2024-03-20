.PHONY: build run

IMAGE_NAME = flask-sentiment-analysis
CONTAINER_NAME = sentiment-analysis-container

build:
    docker build -t $(IMAGE_NAME) .

run:
    docker run -d -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

stop:
    docker stop $(CONTAINER_NAME)

clean:
    docker rm $(CONTAINER_NAME)
    docker rmi $(IMAGE_NAME)
