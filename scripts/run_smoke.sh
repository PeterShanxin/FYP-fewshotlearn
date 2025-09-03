#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run reusing the main pipeline.
# Uses config.smoke.yaml (CPU, tiny limits) and fetches real data as normal.

CFG=${1:-config.smoke.yaml}

# Auto-activate local venv if present
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[smoke] Python: $(command -v python)"

# Ensure dependencies present (idempotent)
python - <<'PY'
import importlib
missing = []
for m in ("torch","numpy","pandas","yaml","tqdm","sklearn"):
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print("[smoke] preflight missing:", ",".join(missing) or "none")
PY

# Install full requirements only if esm is missing (quickest signal for embedding step)
python - <<'PY'
try:
    import esm  # noqa: F401
    print('[smoke] esm present')
except Exception:
    print('[smoke] esm missing -> installing requirements')
    import subprocess, sys
    sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']))
PY

echo "[smoke] Using config: ${CFG}"

# Delegate to the main pipeline with the smoke config.
bash scripts/run_all.sh "${CFG}"

echo "[smoke] Done. Artifacts in results/smoke and data/emb/embeddings_smoke.npz"
