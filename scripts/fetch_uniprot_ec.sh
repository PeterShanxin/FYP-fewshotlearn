#!/usr/bin/env bash
# Fetch Swiss-Prot (reviewed) proteins with EC numbers from UniProt REST,
# log a snapshot (release date, URLs, SHA256), and join TSV+FASTA by accession.
#
# Usage: bash scripts/fetch_uniprot_ec.sh

set -euo pipefail

# --- Paths ---
DATA_ROOT="data/uniprot_ec"
TMP_DIR="${DATA_ROOT}/_tmp"
SNAPSHOT_META="${DATA_ROOT}/snapshot_meta.txt"
TSV_FILE="${DATA_ROOT}/swissprot_ec.tsv"
FASTA_FILE="${DATA_ROOT}/swissprot_ec.fasta"
JOINED_TSV="${DATA_ROOT}/swissprot_ec_joined.tsv"
LONG_TSV="${DATA_ROOT}/swissprot_ec_joined_long.tsv"

mkdir -p "${DATA_ROOT}" "${TMP_DIR}"

# --- Query & URLs ---
QUERY="reviewed:true AND (ec:*)"
# UniProt 2025 field names: 'taxon_id' is not valid; use 'organism_id'
FIELDS="accession,ec,protein_name,organism_id,organism_name,length"
BASE="https://rest.uniprot.org/uniprotkb/stream"
TSV_URL="${BASE}?query=$(python - <<'PY'
import urllib.parse
print(urllib.parse.quote('reviewed:true AND (ec:*)'))
PY
)&format=tsv&fields=${FIELDS}"
FASTA_URL="${BASE}?query=$(python - <<'PY'
import urllib.parse
print(urllib.parse.quote('reviewed:true AND (ec:*)'))
PY
)&format=fasta"

# --- Helpers ---
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    python - <<PY
import hashlib,sys
h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
print(h)
PY
  fi
}

header_release_date() {
  # extract X-UniProt-Release-Date from headers file
  grep -i "^X-UniProt-Release-Date:" "$1" | sed 's/^.*: \(.*\)$/\1/' | tr -d '\r' || true
}

# --- Download TSV ---
TSV_HDR="${TMP_DIR}/tsv_headers.txt"
echo "[fetch] Downloading TSV → ${TSV_FILE}"
curl -sSL -D "${TSV_HDR}" -o "${TSV_FILE}" "${TSV_URL}"
TSV_SHA=$(sha256 "${TSV_FILE}")
TSV_REL=$(header_release_date "${TSV_HDR}")

# --- Download FASTA ---
FASTA_HDR="${TMP_DIR}/fasta_headers.txt"
echo "[fetch] Downloading FASTA → ${FASTA_FILE}"
curl -sSL -D "${FASTA_HDR}" -o "${FASTA_FILE}" "${FASTA_URL}"
FASTA_SHA=$(sha256 "${FASTA_FILE}")
FASTA_REL=$(header_release_date "${FASTA_HDR}")

# --- Snapshot log ---
{
  echo "=== UniProt Enzyme Snapshot ==="
  date -u 
  echo "Query: ${QUERY}"
  echo "TSV_URL: ${TSV_URL}"
  echo "FASTA_URL: ${FASTA_URL}"
  echo "X-UniProt-Release-Date (TSV): ${TSV_REL}"
  echo "X-UniProt-Release-Date (FASTA): ${FASTA_REL}"
  echo "SHA256(TSV):   ${TSV_SHA}  $(basename "${TSV_FILE}")"
  echo "SHA256(FASTA): ${FASTA_SHA}  $(basename "${FASTA_FILE}")"
} | tee "${SNAPSHOT_META}"

# --- Parse FASTA → accession \t sequence ---
FASTA_ACC="${TMP_DIR}/fasta_acc_seq.tsv"
echo "[join] Parsing FASTA headers…"
awk 'BEGIN{OFS="\t"; acc=""; seq=""}
  /^>/{
    if(acc!=""){gsub(/\r/,"",seq); print acc, seq; seq=""}
    # UniProt header: >sp|ACCESSION|ENTRY_NAME …
    split($0, a, "|"); acc=a[2]; next
  }
  {gsub(/[ \t\r]/,""); seq=seq $0}
  END{if(acc!=""){gsub(/\r/,"",seq); print acc, seq}}
' "${FASTA_FILE}" > "${FASTA_ACC}"

# --- Join TSV + FASTA by accession ---
echo "[join] Joining TSV+FASTA by accession → ${JOINED_TSV}"
python - "$TSV_FILE" "$FASTA_ACC" "$JOINED_TSV" "$LONG_TSV" <<'PY'
import sys, csv
from collections import OrderedDict

tsv_path, fasta_acc_path, out_joined, out_long = sys.argv[1:5]
# Read TSV (header present)
rows = []
with open(tsv_path, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        rows.append(r)
# Read FASTA accession→sequence
acc2seq = {}
with open(fasta_acc_path) as f:
    for line in f:
        acc, seq = line.rstrip('\n').split('\t', 1)
        acc2seq[acc] = seq
# Prepare joined rows
joined_header = [
    'accession','ec','protein_name','taxon_id','organism_name','length','sequence'
]
with open(out_joined, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(joined_header)
    for r in rows:
        acc = r.get('Accession') or r.get('accession') or r.get('Entry')
        ec  = r.get('EC number') or r.get('ec') or ''
    rec = [
      acc,
      ec,
      r.get('Protein names') or r.get('protein_name') or '',
      # Map possible taxonomy field names (legacy + current API)
      r.get('Taxon ID') or r.get('taxon_id') or r.get('organism_id') or r.get('Taxonomic identifier') or '',
      r.get('Organism') or r.get('organism_name') or '',
      r.get('Length') or r.get('length') or '',
      acc2seq.get(acc,'')
    ]
        if acc and rec[-1]:
            w.writerow(rec)
# Optional long format by EC levels if easy
def split_ec(ec: str):
    parts = (ec or '').strip().split('.')
    parts += [''] * (4 - len(parts))
    return parts[:4]
with open(out_long, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['accession','ec_full','ec1','ec2','ec3','ec4'])
    for r in rows:
        acc = r.get('Accession') or r.get('accession') or r.get('Entry')
        ec  = (r.get('EC number') or r.get('ec') or '').strip()
        if not acc or not ec:
            continue
        for ec_item in [e.strip() for e in ec.split(';') if e.strip()]:
            ec1,ec2,ec3,ec4 = split_ec(ec_item)
            w.writerow([acc, ec_item, ec1, ec2, ec3, ec4])
print('[done]')
PY

rm -rf "${TMP_DIR}"
echo "[ok] Snapshot written to ${SNAPSHOT_META}"
echo "[ok] Joined table → ${JOINED_TSV}"
