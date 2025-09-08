#!/usr/bin/env bash
# Selectively delete fetched data, clusters, embeddings, splits, and results.
#
# Usage examples:
#   bash scripts/clean_all.sh --yes                  # clean everything (default)
#   bash scripts/clean_all.sh --yes --embeddings     # only embeddings
#   bash scripts/clean_all.sh --yes --fetch --splits # fetch data + splits
#   bash scripts/clean_all.sh --help                 # show help

set -euo pipefail

show_help() {
  cat <<EOF
Usage: bash scripts/clean_all.sh [--yes] [--all] [--fetch] [--clusters] [--embeddings] [--splits] [--results]

Flags:
  --yes          Proceed without interactive confirmation (required for deletion)
  --all          Clean everything (default if no category flags provided)
  --fetch        Remove fetched UniProt data (data/uniprot_ec, data/uniprot_ec_*)
  --clusters     Remove identity clustering artifacts (data/identity/_work*, data/identity/clusters*.tsv)
  --embeddings   Remove embeddings (data/emb)
  --splits       Remove split JSONLs (data/splits, data/splits_*)
  --results      Remove results directory (results)
  -h, --help     Show this help

Examples:
  bash scripts/clean_all.sh --yes                    # clean all
  bash scripts/clean_all.sh --yes --embeddings       # only embeddings
  bash scripts/clean_all.sh --yes --fetch --splits   # fetch data + splits
EOF
}

YES=0
DO_ALL=0
DO_FETCH=0
DO_CLUSTERS=0
DO_EMB=0
DO_SPLITS=0
DO_RESULTS=0

if [ "$#" -eq 0 ]; then
  show_help
  exit 1
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes) YES=1 ;;
    --all) DO_ALL=1 ;;
    --fetch) DO_FETCH=1 ;;
    --clusters) DO_CLUSTERS=1 ;;
    --embeddings) DO_EMB=1 ;;
    --splits) DO_SPLITS=1 ;;
    --results) DO_RESULTS=1 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "[clean] Unknown flag: $1"; show_help; exit 1 ;;
  esac
  shift
done

# If no categories provided, default to ALL
if [ $DO_ALL -eq 0 ] && [ $DO_FETCH -eq 0 ] && [ $DO_CLUSTERS -eq 0 ] && [ $DO_EMB -eq 0 ] && [ $DO_SPLITS -eq 0 ] && [ $DO_RESULTS -eq 0 ]; then
  DO_ALL=1
fi

TARGETS=()
add_target() {
  TARGETS+=("$1")
}

if [ $DO_ALL -eq 1 ] || [ $DO_FETCH -eq 1 ]; then
  add_target "data/uniprot_ec"
  add_target "data/uniprot_ec_*"
fi
if [ $DO_ALL -eq 1 ] || [ $DO_CLUSTERS -eq 1 ]; then
  add_target "data/identity/_work*"
  add_target "data/identity/clusters*.tsv"
fi
if [ $DO_ALL -eq 1 ] || [ $DO_EMB -eq 1 ]; then
  add_target "data/emb"
fi
if [ $DO_ALL -eq 1 ] || [ $DO_SPLITS -eq 1 ]; then
  add_target "data/splits"
  add_target "data/splits_*"
fi
if [ $DO_ALL -eq 1 ] || [ $DO_RESULTS -eq 1 ]; then
  add_target "results"
  add_target "results_*"
fi

echo "[clean] Will remove:" >&2
for t in "${TARGETS[@]}"; do
  echo "  - $t" >&2
done

if [ $YES -ne 1 ]; then
  echo "[clean] Add --yes to proceed." >&2
  exit 1
fi

echo "[clean] Removing…"
for t in "${TARGETS[@]}"; do
  rm -rf $t || true
done
echo "[clean] Done."
