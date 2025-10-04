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

.PHONY: help install fetch split cluster embed train eval identity-splits identity-benchmark visualize all clean clean-data

	help:
	@echo "Targets:"
	@echo "  install     - pip install minimal dependencies"
	@echo "  fetch       - download Swiss-Prot (reviewed+EC), snapshot, join TSV+FASTA"
	@echo "  split       - build EC class splits (multi-EC expanded)"
	@echo "  cluster     - cluster sequences for identity-aware sampling (MMseqs2 or Python fallback)"
	@echo "  embed       - compute ESM2 mean-pooled embeddings (writes contiguous X.npy + keys.npy)"
	@echo "  train       - episodic training of ProtoNet with early stopping"
	@echo "  eval        - episodic meta-test evaluation (accuracy & macro-F1)"
	@echo "  visualize   - plot multi-threshold results (if summary exists)"
	@echo "  all         - orchestrated run (prepare identity splits + benchmark + visualize)"
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

cluster:
	$(PY) scripts/cluster_sequences.py -c $(CFG)

embed:
	$(PY) -m src.embed_sequences -c $(CFG)

train:
	$(PY) -m src.train_protonet -c $(CFG)

eval:
	$(PY) -m src.eval_protonet -c $(CFG)

# Multi-threshold identity CV helpers
identity-splits:
	$(PY) scripts/prepare_identity_splits.py -c $(CFG)

identity-benchmark:
	$(PY) scripts/run_identity_benchmark.py -c $(CFG)

visualize:
	@if [ -f results/summary_by_id_threshold.json ]; then \
		$(PY) scripts/visualize_identity_benchmark.py --results_dir results --out_dir report/graphs || \
		echo "[viz][note] matplotlib missing or plotting failed; skipping" ; \
	else \
		echo "[viz][note] results/summary_by_id_threshold.json not found; run the benchmark first" ; \
	fi

# Auto-detect: if id_thresholds present, run identity benchmark; otherwise legacy path
all:
	bash scripts/run_all.sh $(CFG)

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__ *.pyc

clean-data:
	rm -rf data/emb data/splits results
