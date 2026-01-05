setup:
	uv sync

download:
	uv run ingest_data.py

train:
	uv run train.py

start:
	uv run python predict.py

test:
	uv run pytest

build:
	docker build -t leukemia-service .

run-docker:
	docker run -it --rm -p 9696:9696 leukemia-service
