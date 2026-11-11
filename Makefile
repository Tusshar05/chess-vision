PY ?= .venv/bin/python
export PYTHONPATH := .
.PHONY: setup data smoke-train weights eval export bench report test lint up down

setup:  ## create venv, install deps, hooks
	uv venv --python 3.11 --managed-python .venv
	uv pip install --python $(PY) -r requirements.txt
	.venv/bin/pre-commit install || true

data:  ## annotations + sample + test-subset (Mac-sized); full data is fetched inside Colab
	$(PY) -m ml.data.prepare --mode local

smoke-train:  ## tiny CPU/MPS training run to prove the pipeline (NOT for reporting)
	$(PY) -m ml.smoke

weights:  ## place Colab-trained weights from release_assets/ and verify; falls back to smoke
	$(PY) scripts/install_weights.py

eval:  ## detector metrics + end-to-end board accuracy + corner comparison -> reports/eval
	$(PY) -m ml.evaluate

export:  ## ONNX FP32, INT8, CoreML -> weights/
	$(PY) -m ml.export

bench:  ## latency + accuracy by backend -> reports/benchmark.md
	$(PY) -m ml.benchmark

report:  ## regenerate README results tables from reports/
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest --cov=ml --cov=services --cov-report=term-missing --cov-report=xml

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

up:
	docker compose up --build -d

down:
	docker compose down
