# Mandate Health Service
# Every target is seeded and reproducible. `make data SEED=42` must produce
# byte-identical output on every run.

# Prefer the project venv if one exists, so a clean clone works without the
# reader having to remember to activate anything.
ifneq (,$(wildcard .venv/Scripts/python.exe))
PYTHON  ?= .venv/Scripts/python.exe
else ifneq (,$(wildcard .venv/bin/python))
PYTHON  ?= .venv/bin/python
else
PYTHON  ?= python
endif

SEED    ?= 42
PAYERS  ?= 400
CYCLES  ?= 6
OUT     ?= data

.PHONY: help venv data test clean train eval dash

help:
	@echo "make venv                                                   create .venv and install requirements"
	@echo "make data   SEED=$(SEED) PAYERS=$(PAYERS) CYCLES=$(CYCLES)  generate synthetic world"
	@echo "make test                                                   run test suite"
	@echo "make clean                                                  remove generated data + caches"
	@echo "make train / eval / dash                                    not yet implemented"

venv:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m simulator.generate --seed $(SEED) --payers $(PAYERS) --cycles $(CYCLES) --out $(OUT)

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf data/observable data/ground_truth/*.parquet
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

train:
	$(PYTHON) -m predictor.train

eval:
	@echo "not implemented: evaluation harness lands in Phase 5"
	@exit 1

dash:
	@echo "not implemented: dashboard lands in Phase 6"
	@exit 1
