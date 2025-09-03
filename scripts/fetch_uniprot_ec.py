import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse
import urllib.request
import urllib.error
import argparse
import sys

from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI / logging helpers
parser = argparse.ArgumentParser(description="Fetch UniProt EC entries")
parser.add_argument("--verbose", dest="verbose", action="store_true", default=True,
                    help="Enable verbose output")
parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                    help="Disable verbose output")
parser.add_argument("--progress", dest="show_progress", action="store_true",
                    default=True, help="Show progress bar")
parser.add_argument("--no-progress", dest="show_progress", action="store_false",
                    help="Disable progress bar")
parser.add_argument("--data-root", dest="data_root", default="data/uniprot_ec",
                    help="Output directory for downloaded and joined files")
parser.add_argument("--query", dest="query", default='reviewed:true AND (ec:*)',
                    help="UniProt query string (default: reviewed:true AND (ec:*) )")
parser.add_argument("--force-download", dest="force_download", action="store_true", default=False,
                    help="Force re-download even if files exist")
parser.add_argument("--no-join", dest="skip_join", action="store_true", default=False,
                    help="Skip TSV/FASTA parsing and join (only snapshot)")
args = parser.parse_args()

VERBOSE = args.verbose
SHOW_PROGRESS = args.show_progress


def log(msg: str):
    if VERBOSE:
        if SHOW_PROGRESS:
            tqdm.write(msg)
        else:
            print(msg)

# Paths
DATA_ROOT = Path(args.data_root)
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
QUERY = args.query
# NOTE: UniProt uses 'taxon_id' for taxonomy information (formerly 'taxonomy_id').
# Field reference: https://www.uniprot.org/help/return_fields (for latest names).
FIELDS = 'accession,ec,protein_name,organism_id,organism_name,length'  # 'taxon_id' renamed to 'organism_id' in UniProt API
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
    # We modify these globals only if we need to fall back to a simplified query
    # after an HTTP 400; declare up-front to avoid Python treating them as locals
    # after assignment inside the retry loop (which previously caused a
    # SyntaxError due to use-before-declaration).
    global TSV_URL, FASTA_URL, QUERY
    last_err = None
    # If file already exists and force_download is False, reuse it
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force_download:
        data = out_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        log(f"[cache] Using existing {label}: {out_path} ({len(data):,} bytes)")
        return 'cached', sha

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
                log(f"[retry] {label} succeeded on attempt {attempt}")
            return rel, sha
        except urllib.error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', 'ignore')[:500]
            except Exception:
                pass
            log(f"[warn] HTTPError {e.code} for {label} attempt {attempt}: {e.reason}")
            if body:
                log(f"[warn] Body snippet: {body}")
            last_err = e
            # If 400, try a simplified query once (remove parentheses) then re-build URLs.
            if e.code == 400 and attempt == 1 and 'ec:*' in QUERY:
                simple_query = 'reviewed:true AND ec:*'
                log(f"[info] Trying simplified query variant: '{simple_query}'")
                QUERY = simple_query
                TSV_URL, FASTA_URL = _build_urls(QUERY)
                url = TSV_URL if 'tsv' in url else FASTA_URL
            elif e.code in (429, 503):
                # Backoff for rate limit or temporary service error
                import time
                time.sleep(5 * attempt)
        except Exception as e:  # network, timeout, etc.
            log(f"[warn] Error for {label} attempt {attempt}: {e}")
            last_err = e
    # All retries exhausted
    raise last_err  # type: ignore


log(f"[fetch] Query: {QUERY}")
log(f"[fetch] TSV URL   : {TSV_URL}")
log(f"[fetch] FASTA URL : {FASTA_URL}")
log(f"[fetch] Downloading TSV -> {TSV_FILE}")
TSV_REL, TSV_SHA = _download(TSV_URL, TSV_FILE, 'TSV')
log(f"[fetch] Downloading FASTA -> {FASTA_FILE}")
FASTA_REL, FASTA_SHA = _download(FASTA_URL, FASTA_FILE, 'FASTA')

# Snapshot block: write to file and print once BEFORE progress bar to avoid interleaving
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
SNAPSHOT_META.write_text("\n".join(snapshot_lines) + "\n", encoding='utf-8')

if VERBOSE:
    # Print the snapshot once, cleanly, before showing any progress bars
    print("\n".join(snapshot_lines))
    print()  # extra newline for readability

if args.skip_join:
    log('[done]')
    log(f'[ok] Snapshot written to {SNAPSHOT_META}')
    log('[ok] Skipped join (--no-join)')
    sys.exit(0)

# Progress bar for join steps (print only the bar; keep other messages in the postfix)
steps = [
    "Parse FASTA headers",
    "Read TSV",
    "Join TSV+FASTA",
    "Write long format",
    "Write snapshot meta"
]
pbar = tqdm(
    total=len(steps),
    desc="[dataFetch]",
    disable=not SHOW_PROGRESS,
    dynamic_ncols=True,
    leave=False,
    bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
)


# Parse FASTA
if SHOW_PROGRESS:
    pbar.set_postfix_str("Parsing FASTA headers…")
else:
    log("[join] Parsing FASTA headers…")
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
pbar.update(1)
if SHOW_PROGRESS:
    pbar.set_postfix_str("")


# Read TSV
rows = []
with TSV_FILE.open(newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows.extend(reader)
pbar.update(1)
if SHOW_PROGRESS:
    pbar.set_postfix_str("Joining TSV+FASTA…")


# Join TSV + FASTA
if not SHOW_PROGRESS:
    log(f"[join] Joining TSV+FASTA by accession -> {JOINED_TSV}")
joined_header = ['accession','ec','protein_name','taxon_id','organism_name','length','sequence']  # keep legacy column name 'taxon_id' for downstream compatibility
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
            # Map possible taxonomy field names (old/new API variants)
            r.get('Taxon ID') or r.get('taxon_id') or r.get('organism_id') or r.get('Taxonomic identifier') or '',
            r.get('Organism') or r.get('organism_name') or '',
            r.get('Length') or r.get('length') or '',
            acc2seq.get(acc, '')
        ]
        if acc and rec[-1]:
            w.writerow(rec)
pbar.update(1)


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
pbar.update(1)


pbar.update(1)
pbar.close()

# Final, compact messages after the bar is gone
log('[done]')
log(f'[ok] Snapshot written to {SNAPSHOT_META}')
log(f'[ok] Joined table -> {JOINED_TSV}')
