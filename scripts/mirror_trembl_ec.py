#!/usr/bin/env python3
"""Mirror EC-labeled TrEMBL to local cache (TSV + FASTA), with resume/caching.

Downloads (paginated):
- TSV meta: accession, ec, protein_name, organism_id, organism_name, length
- FASTA sequences

Default query filters for quality:
  reviewed:false AND ec:* AND (existence:1 OR existence:2 OR existence:3) AND fragment:false

Skips download if targets exist and --force is not set. For very large mirrors
(millions of rows), prefer running on a data node. Use --max-rows for smoke.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.parse
import urllib.request
import urllib.error

from tqdm.auto import tqdm


SEARCH_BASE = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,ec,protein_name,organism_id,organism_name,length"


def build_search_url(query: str, fmt: str, fields: Optional[str] = None, size: int = 500) -> str:
    enc = urllib.parse.quote(query, safe="")
    url = f"{SEARCH_BASE}?query={enc}&format={fmt}&size={size}"
    if fields and fmt.lower() == "tsv":
        url += f"&fields={fields}"
    return url


def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(','):
        seg = part.strip()
        if 'rel="next"' in seg:
            lt = seg.find('<')
            gt = seg.find('>', lt + 1)
            if lt != -1 and gt != -1:
                return seg[lt + 1:gt]
    return None


def download_paginated(initial_url: str, out_path: Path, label: str, drop_header_after_first: bool, max_rows: int = 0) -> tuple[str, str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    url = initial_url
    page = 0
    first_rel = ''
    hasher = hashlib.sha256()
    total_rows = 0
    progress = tqdm(total=None, unit='B', unit_scale=True, desc=f"[mirror:{label}]", dynamic_ncols=True)
    try:
        while url:
            page += 1
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "fsl-ec-mirror/1.0 (+https://github.com/)",
                    "Accept": "*/*",
                })
                with urllib.request.urlopen(req, timeout=300) as resp:
                    rel = resp.headers.get('X-UniProt-Release-Date', '') or resp.headers.get('X-UniProt-Release', '') or ''
                    if not first_rel:
                        first_rel = rel
                    data = resp.read()
                    progress.update(len(data))
                    if drop_header_after_first and page > 1:
                        nl = data.find(b"\n")
                        if nl != -1:
                            data = data[nl + 1:]
                    # Optionally truncate by rows for smoke (TSV only)
                    if max_rows and drop_header_after_first:
                        lines = data.splitlines(keepends=True)
                        total_rows += max(0, len(lines))
                        if total_rows >= max_rows:
                            over = total_rows - max_rows
                            if over > 0:
                                lines = lines[:-over]
                            data = b"".join(lines)
                            with out_path.open('ab') as fh:
                                fh.write(data)
                            hasher.update(data)
                            break
                    with out_path.open('ab') as fh:
                        fh.write(data)
                    hasher.update(data)
                    url = parse_next_link(resp.headers.get('Link'))
            except urllib.error.HTTPError as e:
                body = ''
                try:
                    body = e.read().decode('utf-8', 'ignore')[:400]
                except Exception:
                    pass
                print(f"[mirror][warn] HTTP {e.code} on page {page}: {e.reason}")
                if body:
                    print(f"[mirror][warn] Body: {body}")
                if e.code in (429, 503):
                    import time
                    time.sleep(5)
                    continue
                raise
    finally:
        progress.close()
    return first_rel, hasher.hexdigest(), total_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/uniprot_ec")
    ap.add_argument("--existence", default="1,2,3", help="Protein existence levels to include (comma list)")
    ap.add_argument("--fragment", action="store_true", help="Include fragments as well (off by default)")
    ap.add_argument("--size", type=int, default=500, help="Page size for UniProt pagination")
    ap.add_argument("--max-rows", type=int, default=0, help="Limit total TSV rows (0 = unlimited; for smoke)")
    ap.add_argument("--force", action="store_true", help="Force re-download even if cached files exist")
    args = ap.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    meta_tsv = root / "trembl_ec.tsv"
    fasta = root / "trembl_ec.fasta"
    snapshot = root / "trembl_snapshot_meta.txt"

    existence_levels = ','.join([s.strip() for s in args.existence.split(',') if s.strip()])
    exists_query = ' OR '.join(f"existence:{lvl}" for lvl in existence_levels.split(',') if lvl)
    frag_filter = "" if args.fragment else " AND fragment:false"
    base_q = f"reviewed:false AND ec:* AND ({exists_query}){frag_filter}"
    print(f"[mirror] Query: {base_q}")

    # Skip if cached
    if meta_tsv.exists() and fasta.exists() and not args.force:
        print("[mirror] Cached files found; skipping download. Use --force to re-download.")
        print(f"[mirror] {meta_tsv} | {meta_tsv.stat().st_size:,} bytes")
        print(f"[mirror] {fasta} | {fasta.stat().st_size:,} bytes")
        return

    # Download TSV (paginated)
    tsv_url = build_search_url(base_q, fmt='tsv', fields=FIELDS, size=int(args.size))
    print(f"[mirror] Downloading TSV → {meta_tsv}")
    tsv_rel, tsv_sha, total_rows = download_paginated(tsv_url, meta_tsv, 'TrEMBL TSV', drop_header_after_first=True, max_rows=int(args.max_rows))
    # Download FASTA (paginated); do not row-limit since FASTA and TSV may diverge; this is best-effort for smoke
    fa_url = build_search_url(base_q, fmt='fasta', size=int(args.size))
    print(f"[mirror] Downloading FASTA → {fasta}")
    fa_rel, fa_sha, _ = download_paginated(fa_url, fasta, 'TrEMBL FASTA', drop_header_after_first=False, max_rows=0)

    # Snapshot
    lines = [
        "=== UniProt TrEMBL EC Mirror ===",
        datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S UTC %Y"),
        f"Query: {base_q}",
        f"TSV_URL: {tsv_url}",
        f"FASTA_URL: {fa_url}",
        f"X-UniProt-Release (TSV): {tsv_rel}",
        f"X-UniProt-Release (FASTA): {fa_rel}",
        f"SHA256(TSV):   {tsv_sha}  {meta_tsv.name}",
        f"SHA256(FASTA): {fa_sha}  {fasta.name}",
        f"TSV_rows: {total_rows}",
    ]
    snapshot.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print("\n".join(lines))


if __name__ == "__main__":
    main()
