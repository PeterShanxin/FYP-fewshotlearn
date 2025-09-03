
#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-config.yaml}

# Time logger start
start_time=$(date +%s)
# Track whether the pipeline reached the end successfully
completed=0

# Always print total runtime on exit (success or failure)
on_exit() {
  status=$?
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  hours=$((elapsed / 3600))
  mins=$(((elapsed % 3600) / 60))
  secs=$((elapsed % 60))
  if [ "$completed" -eq 1 ]; then
    printf "[run_all] Done. Check results/ for outputs.\n"
  else
    printf "[run_all] Failed with exit code %d.\n" "$status" >&2
  fi
  printf "[run_all] Total runtime: %02dh:%02dm:%02ds\n" "$hours" "$mins" "$secs"
  exit "$status"
}
trap on_exit EXIT

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

# Respect gpus from config by setting CUDA_VISIBLE_DEVICES if not already set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPUS=$(python - <<PY
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(int(cfg.get('gpus', 1)))
PY
"${CFG}")
  if [ "$GPUS" -gt 0 ]; then
    # Build a device list like 0 or 0,1
    LIST=$(python - <<PY
g = int("${GPUS}")
print(",".join(str(i) for i in range(g)))
PY
)
    export CUDA_VISIBLE_DEVICES="$LIST"
    echo "[run_all] CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
  fi
fi

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
# Improve CUDA allocation behavior for large batches/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.embed_sequences -c "${CFG}"

# 4) Train ProtoNet
python -m src.train_protonet -c "${CFG}"

# 5) Evaluate
python -m src.eval_protonet -c "${CFG}"

# Mark pipeline as completed only if all steps above succeeded
completed=1
