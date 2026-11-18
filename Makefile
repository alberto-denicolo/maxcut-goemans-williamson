CONFIG ?= configs/thesis.yaml

.PHONY: install test lint quick experiments figures all clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

quick:
	gwmaxcut run all --config configs/quick.yaml --out results/quick
	gwmaxcut plot --config configs/quick.yaml --results results/quick --figures figures/quick

experiments:
	gwmaxcut run all --config $(CONFIG) --out results

figures:
	gwmaxcut plot --config $(CONFIG) --results results --figures figures

all: experiments figures

clean:
	rm -rf results/quick figures/quick .pytest_cache .ruff_cache
