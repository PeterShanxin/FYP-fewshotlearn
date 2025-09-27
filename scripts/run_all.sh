
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
completed=0

STATUS_LOG=""
LOG_DIR=""
declare -A PHASE_STARTS=()
CURRENT_PHASE=""

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_line() {
  local message=$1
  if [ -n "${STATUS_LOG:-}" ]; then
    printf "%s %s\n" "$(timestamp)" "$message" >> "$STATUS_LOG"
  fi
}

phase_begin() {
  local phase=$1
  CURRENT_PHASE="$phase"
  PHASE_STARTS[$phase]=$(date +%s)
  log_line "phase=${phase} event=start"
}

phase_complete() {
  local phase=$1
  local phase_status=${2:-success}
  local end=$(date +%s)
  local start=${PHASE_STARTS[$phase]:-}
  local elapsed=0
  if [ -n "$start" ]; then
    elapsed=$((end - start))
  fi
  log_line "phase=${phase} event=finish status=${phase_status} duration_seconds=${elapsed}"
  unset PHASE_STARTS[$phase]
  if [ "$CURRENT_PHASE" = "$phase" ]; then
    CURRENT_PHASE=""
  fi
}

# Always print total runtime on exit (success or failure)
on_exit() {
  local status=$?
  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))
  local hours=$((elapsed / 3600))
  local mins=$(((elapsed % 3600) / 60))
  local secs=$((elapsed % 60))

  if [ -n "${STATUS_LOG:-}" ]; then
    if [ "$status" -eq 0 ] && [ "$completed" -eq 1 ]; then
      log_line "pipeline event=completed status=success total_seconds=${elapsed}"
    else
      local phase_elapsed=0
      if [ -n "${CURRENT_PHASE:-}" ]; then
        local start=${PHASE_STARTS[$CURRENT_PHASE]:-}
        if [ -n "$start" ]; then
          phase_elapsed=$((end_time - start))
        fi
        log_line "phase=${CURRENT_PHASE} event=failed exit_code=${status} duration_seconds=${phase_elapsed}"
      fi
      log_line "pipeline event=completed status=failure exit_code=${status} total_seconds=${elapsed}"
    fi
  fi

  if [ "$completed" -eq 1 ] && [ "$status" -eq 0 ]; then
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

RESULTS_DIR=$(python - "${CFG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
paths = cfg.get('paths', {}) or {}
out_dir = Path(paths.get('outputs', 'results')).resolve()
print(out_dir)
PY
)

LOG_DIR="${RESULTS_DIR}/logs"
STATUS_LOG="${LOG_DIR}/run_all_status.log"
mkdir -p "$LOG_DIR"
: > "$STATUS_LOG"
log_line "python path=$(command -v python)"

# Optionally load HPC modules declared in config (modules: [MMseqs2])
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
        log_line "modules event=load status=failed module=${m}"
      else
        echo "[run_all][modules] loaded: $m"
        log_line "modules event=load status=success module=${m}"
      fi
    done <<< "${MODULES}"
  else
    echo "[run_all][modules][note] 'module' command not available; skipping module loads"
    log_line "modules event=skipped reason=command_unavailable"
  fi
fi

FORCE_FETCH=$(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(str(bool(cfg.get('force_fetch', False))).lower())
PY
)

log_line "pipeline event=start config=${CFG} force_embed=${FORCE_EMBED} force_fetch=${FORCE_FETCH} results_dir=${RESULTS_DIR}"

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
    log_line "env event=set_cuda_visible_devices value=${CUDA_VISIBLE_DEVICES}"
  fi
fi

# 1) Fetch data (idempotent). Prefer Python fetcher (richer retries); fallback to shell.
FETCH_ARGS=""
if [ "$FORCE_FETCH" = "true" ]; then
  FETCH_ARGS="--force-download"
fi
phase_begin "fetch"
if python scripts/fetch_uniprot_ec.py $FETCH_ARGS; then
  echo "[run_all] Python fetcher succeeded"
  log_line "phase=fetch detail=python_fetcher status=success"
else
  echo "[run_all] Python fetcher failed; falling back to shell fetch script" >&2
  log_line "phase=fetch detail=python_fetcher status=failed fallback=shell"
  bash scripts/fetch_uniprot_ec.sh
fi
phase_complete "fetch"

# Improve CUDA allocation behavior for large batches/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log_line "env event=set PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

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
log_line "config id_thresholds=${CUTOFFS_PCT_STR} folds=${FOLDS} multi=${MULTI}"

if [ "$FORCE_EMBED" -eq 1 ]; then FE_FLAG="--force-embed"; else FE_FLAG=""; fi
SKIP_PREPARE_FLAG=""
phase_begin "prepare_splits"
read -r HAVE_CACHED_SPLITS <<EOF
$(python - "${CFG}" "${CUTOFFS}" "${FOLDS}" <<'PY'
import json
import sys
from pathlib import Path

import yaml


def parse_cutoffs(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            val = float(chunk)
        except ValueError:
            continue
        if val > 1.0:
            val = val / 100.0
        values.append(val)
    return values


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


cfg_path, cutoffs_raw, folds_raw = sys.argv[1:4]
try:
    folds = int(folds_raw)
except Exception:
    print("0")
    sys.exit(0)

cutoffs = parse_cutoffs(cutoffs_raw)
if not cutoffs:
    print("0")
    sys.exit(0)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

paths = cfg.get("paths", {}) or {}
outputs = Path(paths.get("outputs", "results")).resolve()
identity_def = str(cfg.get("identity_definition", "tool_default"))
stratify_by = str(cfg.get("stratify_by", "EC_top"))
seed = int(cfg.get("random_seed", 42))

ok = True
for cutoff in cutoffs:
    pct = int(round(cutoff * 100))
    split_dir = outputs / f"split-{pct}"
    config_used = split_dir / "config.used.yaml"
    folds_json = split_dir / "folds.json"
    if not (path_exists(config_used) and path_exists(folds_json)):
        ok = False
        break
    try:
        used_cfg = yaml.safe_load(config_used.read_text(encoding="utf-8")) or {}
    except Exception:
        ok = False
        break
    if int(used_cfg.get("id_threshold", -1)) != pct:
        ok = False
        break
    if int(used_cfg.get("folds", folds)) != folds:
        ok = False
        break
    if str(used_cfg.get("identity_definition", identity_def)) != identity_def:
        ok = False
        break
    if str(used_cfg.get("stratify_by", stratify_by)) != stratify_by:
        ok = False
        break
    if int(used_cfg.get("random_seed", seed)) != seed:
        ok = False
        break
    try:
        folds_info = json.loads(folds_json.read_text(encoding="utf-8"))
    except Exception:
        ok = False
        break
    folds_map = folds_info.get("folds", {}) or {}
    if len(folds_map) != folds:
        ok = False
        break
    for idx in range(1, folds + 1):
        fold_dir = split_dir / f"fold-{idx}"
        if not (path_exists(fold_dir / "train.jsonl") and path_exists(fold_dir / "val.jsonl") and path_exists(fold_dir / "test.jsonl")):
            ok = False
            break
    if not ok:
        break

print("1" if ok else "0")
PY
)
EOF

prepare_status="success"
if [ "$HAVE_CACHED_SPLITS" = "1" ]; then
  echo "[run_all] Reusing cached identity splits for thresholds ${CUTOFFS_PCT_STR}% (folds=${FOLDS})."
  SKIP_PREPARE_FLAG="--skip_prepare"
  log_line "phase=prepare_splits detail=reused thresholds=${CUTOFFS_PCT_STR} folds=${FOLDS}"
  prepare_status="cached"
else
  echo "[run_all] Preparing identity splits for thresholds ${CUTOFFS_PCT_STR}% (folds=${FOLDS})."
  python scripts/prepare_identity_splits.py -c "${CFG}" --cutoffs "${CUTOFFS}" --folds "${FOLDS}"
  log_line "phase=prepare_splits detail=generated thresholds=${CUTOFFS_PCT_STR} folds=${FOLDS}"
fi
phase_complete "prepare_splits" "$prepare_status"

phase_begin "benchmark"
python scripts/run_identity_benchmark.py -c "${CFG}" --cutoffs "${CUTOFFS}" --folds "${FOLDS}" ${FE_FLAG} ${SKIP_PREPARE_FLAG}
log_line "phase=benchmark detail=run_identity_benchmark status=success"
phase_complete "benchmark"

phase_begin "visualize"
VIS_STATUS="skipped"
if python - <<'PY'
try:
    import importlib
    importlib.import_module('matplotlib')
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
then
  echo "[run_all][viz] Generating identity benchmark report"
  python scripts/visualize_identity_benchmark.py --results_dir "${RESULTS_DIR}" --out_dir "${RESULTS_DIR}/figures"
  log_line "phase=visualize detail=report status=success"
  VIS_STATUS="success"
else
  echo "[run_all][viz][note] matplotlib unavailable; attempting install"
  log_line "phase=visualize detail=matplotlib_missing action=install_attempt"
  # Try to install matplotlib into the current environment, then retry
  if python -m pip install -q matplotlib >/dev/null 2>&1; then
    if python - <<'PY'
try:
    import importlib
    importlib.import_module('matplotlib')
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
      echo "[run_all][viz] matplotlib installed; generating identity benchmark report"
      python scripts/visualize_identity_benchmark.py --results_dir "${RESULTS_DIR}" --out_dir "${RESULTS_DIR}/figures"
      log_line "phase=visualize detail=report status=success_after_install"
      VIS_STATUS="success_after_install"
    else
      echo "[run_all][viz][note] matplotlib import still failing after install; skipping report"
      log_line "phase=visualize detail=import_failed_after_install status=skipped"
      VIS_STATUS="skipped_import_failed"
    fi
  else
    echo "[run_all][viz][note] pip install failed; skipping identity benchmark report"
    log_line "phase=visualize detail=pip_install_failed status=skipped"
    VIS_STATUS="skipped_install_failed"
  fi
fi
phase_complete "visualize" "$VIS_STATUS"

completed=1
