
#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-config.yaml}

# Time logger start
start_time=$(date +%s)

# Auto-activate local venv if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

echo "[run_all] Python: $(command -v python)"
python - <<'PY'
import torch, json
info = dict(cuda_available=torch.cuda.is_available(), device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
print('[run_all] Torch device probe:', json.dumps(info))
PY

echo "[run_all] Using config: ${CFG}"

# 1) Fetch data (idempotent). Prefer Python fetcher (richer retries); fallback to shell.
if python scripts/fetch_uniprot_ec.py; then
	echo "[run_all] Python fetcher succeeded"
else
	echo "[run_all] Python fetcher failed; falling back to shell fetch script" >&2
	bash scripts/fetch_uniprot_ec.sh
fi

# 2) Prepare splits
python -m src.prepare_split -c "${CFG}"

# 3) Embed sequences
python -m src.embed_sequences -c "${CFG}"

# 4) Train ProtoNet
python -m src.train_protonet -c "${CFG}"

# 5) Evaluate
python -m src.eval_protonet -c "${CFG}"

# Time logger end
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))
secs=$((elapsed % 60))
printf "[run_all] Done. Check results/ for outputs.\n"
printf "[run_all] Total runtime: %02dh:%02dm:%02ds\n" $hours $mins $secs
