
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

# Improve CUDA allocation behavior for large batches/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Identity benchmark (single or multi) based solely on id_thresholds in config
read -r CUTOFFS CUTOFFS_PCT_STR FOLDS MULTI <<EOF
$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
ths = cfg.get('id_thresholds')
cuts = None
if isinstance(ths, (list, tuple)) and len(ths) > 0:
    try:
        vals = [float(x) for x in ths]
        cuts = [v/100.0 for v in vals] if max(vals) > 1 else vals
    except Exception:
        cuts = None
folds = cfg.get('folds')
try:
    folds = int(folds) if folds is not None else 5
except Exception:
    folds = 5
if not cuts:
    print('NA NA', folds, 0)
else:
    pct_str = ",".join(str(int(round(c*100))) for c in cuts)
    dec_str = ",".join(str(c) for c in cuts)
    multi = '1' if len(cuts) > 1 else '0'
    print(dec_str, pct_str, folds, multi)
PY
)
EOF

if [ "$CUTOFFS" = "NA" ]; then
  echo "[run_all][error] 'id_thresholds' is missing or invalid in ${CFG}. Set, e.g.: id_thresholds: [50] or [10,30,50,70,100]" >&2
  exit 2
fi

if [ "$MULTI" = "1" ]; then
  echo "[run_all] Identity benchmark: multi-thresholds detected (${CUTOFFS_PCT_STR}%), folds=${FOLDS}."
else
  echo "[run_all] Identity benchmark: single threshold detected (${CUTOFFS_PCT_STR}%), folds=${FOLDS}."
fi

if [ "$FORCE_EMBED" -eq 1 ]; then FE_FLAG="--force-embed"; else FE_FLAG=""; fi
python scripts/prepare_identity_splits.py -c "${CFG}" --cutoffs "${CUTOFFS}" --folds "${FOLDS}"
python scripts/run_identity_benchmark.py -c "${CFG}" --cutoffs "${CUTOFFS}" --folds "${FOLDS}" ${FE_FLAG}
completed=1
