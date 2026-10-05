.PHONY: data features train evaluate explain fairness final app api test lint all

PYTHON ?= python

data:
 	$(PYTHON) -m readmit.cli data

features:
 	$(PYTHON) -m readmit.cli features

train:
 	$(PYTHON) -m readmit.cli train

evaluate:
 	$(PYTHON) -m readmit.cli evaluate

explain:
 	$(PYTHON) -m readmit.cli explain

fairness:
 	$(PYTHON) -m readmit.cli fairness

final:
 	$(PYTHON) -m readmit.cli final

app:
 	$(PYTHON) -m readmit.cli app

api:
 	$(PYTHON) -m readmit.cli api

test:
 	pytest -v

lint:
 	ruff check src tests

all: data features train evaluate explain fairness final
