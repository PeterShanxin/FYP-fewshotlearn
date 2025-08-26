#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-config.yaml}

echo "[run_all] Using config: ${CFG}"

# 1) Fetch data (idempotent)
bash scripts/fetch_uniprot_ec.sh

# 2) Prepare splits
python -m src.prepare_split -c "${CFG}"

# 3) Embed sequences
python -m src.embed_sequences -c "${CFG}"

# 4) Train ProtoNet
python -m src.train_protonet -c "${CFG}"

# 5) Evaluate
python -m src.eval_protonet -c "${CFG}"

echo "[run_all] Done. Check results/ for outputs."
