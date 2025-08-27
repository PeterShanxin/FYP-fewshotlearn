import csv
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse
import urllib.request

# Paths
DATA_ROOT = Path('data/uniprot_ec')
TMP_DIR = DATA_ROOT / '_tmp'  # retained for compatibility, though unused
SNAPSHOT_META = DATA_ROOT / 'snapshot_meta.txt'
TSV_FILE = DATA_ROOT / 'swissprot_ec.tsv'
FASTA_FILE = DATA_ROOT / 'swissprot_ec.fasta'
JOINED_TSV = DATA_ROOT / 'swissprot_ec_joined.tsv'
LONG_TSV = DATA_ROOT / 'swissprot_ec_joined_long.tsv'

DATA_ROOT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Query & URLs
QUERY = 'reviewed:true AND (ec:*)'
FIELDS = 'accession,ec,protein_name,taxonomy_id,organism_name,length'
BASE = 'https://rest.uniprot.org/uniprotkb/stream'
encoded_query = urllib.parse.quote(QUERY)
TSV_URL = f"{BASE}?query={encoded_query}&format=tsv&fields={FIELDS}"
FASTA_URL = f"{BASE}?query={encoded_query}&format=fasta"


def _download(url: str, out_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "python-fetcher"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        rel = resp.headers.get('X-UniProt-Release-Date', '')
    out_path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    return rel, sha


print(f"[fetch] Downloading TSV -> {TSV_FILE}")
TSV_REL, TSV_SHA = _download(TSV_URL, TSV_FILE)
print(f"[fetch] Downloading FASTA -> {FASTA_FILE}")
FASTA_REL, FASTA_SHA = _download(FASTA_URL, FASTA_FILE)

# Snapshot log
snapshot_lines = [
    "=== UniProt Enzyme Snapshot ===",
    datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S UTC %Y"),
    f"Query: {QUERY}",
    f"TSV_URL: {TSV_URL}",
    f"FASTA_URL: {FASTA_URL}",
    f"X-UniProt-Release-Date (TSV): {TSV_REL}",
    f"X-UniProt-Release-Date (FASTA): {FASTA_REL}",
    f"SHA256(TSV):   {TSV_SHA}  {TSV_FILE.name}",
    f"SHA256(FASTA): {FASTA_SHA}  {FASTA_FILE.name}",
]
for line in snapshot_lines:
    print(line)
SNAPSHOT_META.write_text("\n".join(snapshot_lines) + "\n", encoding='utf-8')

# Parse FASTA
print("[join] Parsing FASTA headers…")
acc2seq = {}
acc = None
seq_lines = []
for line in FASTA_FILE.read_text().splitlines():
    line = line.strip()
    if line.startswith('>'):
        if acc and seq_lines:
            acc2seq[acc] = ''.join(seq_lines)
        parts = line.split('|')
        acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
        seq_lines = []
    else:
        seq_lines.append(line)
if acc and seq_lines:
    acc2seq[acc] = ''.join(seq_lines)

# Read TSV
rows = []
with TSV_FILE.open(newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows.extend(reader)

# Join TSV + FASTA
print(f"[join] Joining TSV+FASTA by accession -> {JOINED_TSV}")
joined_header = ['accession','ec','protein_name','taxonomy_id','organism_name','length','sequence']
with JOINED_TSV.open('w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(joined_header)
    for r in rows:
        acc = r.get('Accession') or r.get('accession') or r.get('Entry')
        ec = r.get('EC number') or r.get('ec') or ''
        rec = [
            acc,
            ec,
            r.get('Protein names') or r.get('protein_name') or '',
            r.get('Taxonomic identifier') or r.get('taxonomy_id') or '',
            r.get('Organism') or r.get('organism_name') or '',
            r.get('Length') or r.get('length') or '',
            acc2seq.get(acc, '')
        ]
        if acc and rec[-1]:
            w.writerow(rec)

# Optional long format
with LONG_TSV.open('w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['accession','ec_full','ec1','ec2','ec3','ec4'])
    for r in rows:
        acc = r.get('Accession') or r.get('accession') or r.get('Entry')
        ec = (r.get('EC number') or r.get('ec') or '').strip()
        if not acc or not ec:
            continue
        for ec_item in [e.strip() for e in ec.split(';') if e.strip()]:
            parts = ec_item.split('.')
            parts += [''] * (4 - len(parts))
            ec1, ec2, ec3, ec4 = parts[:4]
            w.writerow([acc, ec_item, ec1, ec2, ec3, ec4])

print('[done]')
print(f'[ok] Snapshot written to {SNAPSHOT_META}')
print(f'[ok] Joined table -> {JOINED_TSV}')
