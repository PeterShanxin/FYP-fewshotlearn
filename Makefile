# Makefile for Few‑Shot EC with ProtoNet over ESM2‑t12‑35M
# Usage examples:
#   make install
#   make fetch
#   make split
#   make embed
#   make train
#   make eval
#   make all
#   make clean-data
#
# Variables:
#   CFG=config.yaml (override with CFG=myconfig.yaml)
#   PY=python        (override with PY=python3, etc.)

PY ?= python
CFG ?= config.yaml

.PHONY: help install fetch split embed train eval all clean clean-data

help:
	@echo "Targets:"
	@echo "  install     - pip install minimal dependencies"
	@echo "  fetch       - download Swiss-Prot (reviewed+EC), snapshot, join TSV+FASTA"
	@echo "  split       - build cluster-free meta-train/val/test splits by EC classes"
	@echo "  embed       - compute ESM2-t12-35M mean-pooled embeddings (.npz)"
	@echo "  train       - episodic training of ProtoNet with early stopping"
	@echo "  eval        - episodic meta-test evaluation (accuracy & macro-F1)"
	@echo "  all         - fetch -> split -> embed -> train -> eval"
	@echo "  clean       - remove Python caches"
	@echo "  clean-data  - remove embeddings, splits, and results"
	@echo ""
	@echo "Variables (override with VAR=value): CFG=$(CFG), PY=$(PY)"

install:
	pip install -r requirements.txt

fetch:
	bash scripts/fetch_uniprot_ec.sh

split:
	$(PY) -m src.prepare_split -c $(CFG)

embed:
	$(PY) -m src.embed_sequences -c $(CFG)

train:
	$(PY) -m src.train_protonet -c $(CFG)

eval:
	$(PY) -m src.eval_protonet -c $(CFG)

all: fetch split embed train eval

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__ *.pyc

clean-data:
	rm -rf data/emb data/splits results
