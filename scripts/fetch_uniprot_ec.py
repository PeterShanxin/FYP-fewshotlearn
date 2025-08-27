import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse
import urllib.request
import urllib.error

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
# Original broad query. Some users have reported sporadic HTTP 400 responses from
# UniProt's stream endpoint if certain characters are not encoded the way the
# service expects (notably '*') or if there is transient routing / WAF filtering.
QUERY = 'reviewed:true AND (ec:*)'
# NOTE: 'taxonomy_id' is not a valid UniProt field name (current API uses 'taxon_id').
# Field reference: https://www.uniprot.org/help/return_fields (for latest names).
# Removed taxonomy id field (taxon_id) because the stream endpoint currently
# rejects it with HTTP 400. If taxonomy is required later, it can be parsed
# from the full record or via separate taxonomy queries.
FIELDS = 'accession,ec,protein_name,organism_name,length'
BASE = 'https://rest.uniprot.org/uniprotkb/stream'

def _build_urls(query: str):
    # Encode with '*' escaped as well (safe='') to avoid edge cases.
    encoded_query = urllib.parse.quote(query, safe='')
    tsv_url = f"{BASE}?query={encoded_query}&format=tsv&fields={FIELDS}"
    fasta_url = f"{BASE}?query={encoded_query}&format=fasta"
    return tsv_url, fasta_url

TSV_URL, FASTA_URL = _build_urls(QUERY)


def _download(url: str, out_path: Path, label: str):
    """Download helper with retry & graceful fallback.

    Args:
        url: Full request URL.
        out_path: Destination file path.
        label: Short label for logging.
    Returns:
        (release_date_header, sha256_hex)
    Raises:
        The last exception if all retries fail.
    """
    last_err = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "fsl-ec-fetch/1.0 (+https://github.com/)",
                "Accept": "*/*",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                rel = resp.headers.get('X-UniProt-Release-Date', '')
            out_path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            if attempt > 1:
                print(f"[retry] {label} succeeded on attempt {attempt}")
            return rel, sha
        except urllib.error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', 'ignore')[:500]
            except Exception:
                pass
            print(f"[warn] HTTPError {e.code} for {label} attempt {attempt}: {e.reason}")
            if body:
                print(f"[warn] Body snippet: {body}")
            last_err = e
            # If 400, try a simplified query once (remove parentheses) then re-build URLs.
            if e.code == 400 and attempt == 1 and 'ec:*' in QUERY:
                simple_query = 'reviewed:true AND ec:*'
                print(f"[info] Trying simplified query variant: '{simple_query}'")
                global TSV_URL, FASTA_URL
                TSV_URL, FASTA_URL = _build_urls(simple_query)
                url = TSV_URL if 'tsv' in url else FASTA_URL
            elif e.code in (429, 503):
                # Backoff for rate limit or temporary service error
                import time
                time.sleep(5 * attempt)
        except Exception as e:  # network, timeout, etc.
            print(f"[warn] Error for {label} attempt {attempt}: {e}")
            last_err = e
    # All retries exhausted
    raise last_err  # type: ignore


print(f"[fetch] Query: {QUERY}")
print(f"[fetch] TSV URL   : {TSV_URL}")
print(f"[fetch] FASTA URL : {FASTA_URL}")
print(f"[fetch] Downloading TSV -> {TSV_FILE}")
TSV_REL, TSV_SHA = _download(TSV_URL, TSV_FILE, 'TSV')
print(f"[fetch] Downloading FASTA -> {FASTA_FILE}")
FASTA_REL, FASTA_SHA = _download(FASTA_URL, FASTA_FILE, 'FASTA')

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
