PY ?= python
DATA_DIR ?= data/raw/DataScienceChallenge

.PHONY: install dev run notebook html test lint docker clean

install:            ## install runtime deps and the package (editable)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

dev:                ## install dev deps (tests, lint)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

run:                ## headless pipeline -> reports/
	$(PY) -m latpred.pipeline --data-dir $(DATA_DIR) --output-dir reports

notebook:           ## execute the notebook in place
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_latitude_prediction.ipynb --ExecutePreprocessor.timeout=3600

html: notebook      ## export executed notebook to reports/
	jupyter nbconvert --to html notebooks/01_latitude_prediction.ipynb --output-dir reports

test:               ## unit tests (synthetic data, no challenge data needed)
	$(PY) -m pytest

lint:
	ruff check src tests

docker:             ## build + run in Docker
	./run.sh $(DATA_DIR)

clean:
	rm -rf data/processed/*.csv .pytest_cache **/__pycache__
