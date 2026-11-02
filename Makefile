PY ?= .venv/bin/python
export TORCH_HOME ?= $(CURDIR)/data/torch
RUN = gtimeout 2700 nice -n 10 $(PY)

.PHONY: setup data train eval bench demo test lint plots smoke

setup:
	uv venv -q -p 3.12 && uv pip install -q -e ".[eval,dev]"

data:
	$(PY) scripts/prepare_data.py

# 6 self-supervised runs (SimCLR and BYOL x 3 seeds), ~21 min each on an M5
train:
	for s in 0 1 2; do for m in simclr byol; do $(RUN) scripts/pretrain.py --method $$m --seed $$s; done; done

eval:
	$(RUN) scripts/evaluate_all.py probe
	$(RUN) scripts/evaluate_all.py finetune

bench:
	$(PY) scripts/bench_throughput.py

plots:
	$(PY) scripts/make_tables.py

demo:
	$(PY) -m landprobe.cli classify tests/fixtures/eurosat_tiny.npz --backbone runs/simclr_e40_s0/backbone.pt

smoke:
	$(PY) -m pytest -q tests/test_smoke.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy
