
#!/usr/bin/env bash
set -euo pipefail

# Args:
#   run_all.sh [CFG] [--force-embed]
#   Flags can appear before or after CFG.

CFG="config.yaml"
FORCE_EMBED=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --force-embed) FORCE_EMBED=1 ;;
    -h|--help)
      cat <<EOF
Usage: bash scripts/run_all.sh [config.yaml] [--force-embed]
  --force-embed  Recompute embeddings even if existing files are found
EOF
      exit 0
      ;;
    -*) echo "[run_all] Unknown flag: $1" >&2; exit 1 ;;
    *) CFG="$1" ;;
  esac
  shift
done

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

echo "[run_all] Using config: ${CFG} (force-embed=${FORCE_EMBED})"

# Optionally load HPC modules declared in config (modules: [MMseqs2, CD-HIT])
init_module() {
  # Detect and initialize Environment Modules/Lmod if available
  if type module >/dev/null 2>&1; then
    return 0
  fi
  for f in \
      /etc/profile.d/modules.sh \
      /usr/share/Modules/init/bash \
      /etc/profile.d/lmod.sh \
      /etc/profile.d/z00_lmod.sh \
      /usr/share/lmod/lmod/init/bash; do
    if [ -r "$f" ]; then
      # shellcheck source=/dev/null
      . "$f"
      break
    fi
  done
  type module >/dev/null 2>&1
}

MODULES=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
mods = cfg.get('modules', [])
if isinstance(mods, str):
    mods = [mods]
for m in mods:
    if m and isinstance(m, str):
        print(m)
PY
)

if [ -n "${MODULES}" ]; then
  if init_module; then
    echo "[run_all][modules] Initializing module environment and loading requested modules"
    while IFS= read -r m; do
      [ -z "$m" ] && continue
      if ! module load "$m"; then
        echo "[run_all][modules][warn] failed to load module: $m" >&2
      else
        echo "[run_all][modules] loaded: $m"
      fi
    done <<< "${MODULES}"
  else
    echo "[run_all][modules][note] 'module' command not available; skipping module loads"
  fi
fi

FORCE_FETCH=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(str(bool(cfg.get('force_fetch', False))).lower())
PY
)

# Autodetect identity CV (multi-threshold) from config
IDENTITY_CV=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
ths = cfg.get('id_thresholds', [])
enabled = False
try:
    enabled = isinstance(ths, (list, tuple)) and len(ths) > 0
except Exception:
    enabled = False
print(str(bool(enabled)).lower())
PY
)

# Respect gpus from config by setting CUDA_VISIBLE_DEVICES if not already set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPUS=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(int(cfg.get('gpus', 1)))
PY
)
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
FETCH_ARGS=""
if [ "$FORCE_FETCH" = "true" ]; then
  FETCH_ARGS="--force-download"
fi
if python scripts/fetch_uniprot_ec.py $FETCH_ARGS; then
	echo "[run_all] Python fetcher succeeded"
else
	echo "[run_all] Python fetcher failed; falling back to shell fetch script" >&2
	bash scripts/fetch_uniprot_ec.sh
fi

# If identity CV enabled, run multi-threshold benchmark path and exit
if [ "$IDENTITY_CV" = "true" ]; then
  echo "[run_all] Identity CV detected (id_thresholds present). Running multi-threshold benchmark."
  python scripts/prepare_identity_splits.py -c "${CFG}"
  python scripts/run_identity_benchmark.py -c "${CFG}"
  completed=1
  exit 0
fi

# Otherwise, run the legacy single-threshold path
# 2) Prepare splits
python -m src.prepare_split -c "${CFG}"

# 2.5) Cluster sequences at 50% identity (configurable) to create accession->cluster TSV
echo "[run_all] Clustering sequences for identity-aware sampling"
python scripts/cluster_sequences.py -c "${CFG}"

# 3) Embed sequences (skip if embeddings already exist unless --force-embed)
# Improve CUDA allocation behavior for large batches/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Detect existing embeddings (contiguous X.npy + keys.npy or legacy NPZ)
EMB_BASE=$(python - "${CFG}" <<'PY'
import sys, yaml, os
with open(sys.argv[1],'r') as f:
    cfg = yaml.safe_load(f)
p = cfg['paths']['embeddings']
print(p[:-4] if p.endswith('.npz') else p)
PY
)
EMB_X="${EMB_BASE}.X.npy"
EMB_K="${EMB_BASE}.keys.npy"
EMB_NPZ=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1],'r') as f:
    cfg = yaml.safe_load(f)
print(cfg['paths']['embeddings'])
PY
)

if [ "$FORCE_EMBED" -eq 0 ] && { { [ -f "$EMB_X" ] && [ -f "$EMB_K" ]; } || [ -f "$EMB_NPZ" ]; }; then
  echo "[run_all] Skipping embed: found existing embeddings at $EMB_X / $EMB_K or $EMB_NPZ"
else
  python -m src.embed_sequences -c "${CFG}"
fi

# 4) Train ProtoNet
python -m src.train_protonet -c "${CFG}"

# 5) Evaluate
python -m src.eval_protonet -c "${CFG}"

# Mark pipeline as completed only if all steps above succeeded
completed=1
