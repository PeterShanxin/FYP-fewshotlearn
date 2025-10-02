#!/usr/bin/env python3
"""Build a local SQLite index for the TrEMBL EC mirror.

Creates SQLite DB with tables:
  meta(accession PRIMARY KEY, protein_name, taxon_id, organism_name, length)
  long(accession, ec_full, ec1, ec2, ec3, ec4, PRIMARY KEY(accession, ec_full))
  fasta_index(accession PRIMARY KEY, start_offset, end_offset)

This enables offline top-up by EC with fast accession lookup and sequence
retrieval from the mirrored FASTA.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Tuple


def build_long_from_tsv(conn: sqlite3.Connection, tsv_path: Path) -> Tuple[int, int]:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS meta(accession TEXT PRIMARY KEY, protein_name TEXT, taxon_id TEXT, organism_name TEXT, length INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS long(accession TEXT, ec_full TEXT, ec1 TEXT, ec2 TEXT, ec3 TEXT, ec4 TEXT, PRIMARY KEY(accession, ec_full))")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_long_ec ON long(ec_full)")

    n_meta = 0
    n_long = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            acc = (r.get('Accession') or r.get('accession') or '').strip()
            ec = (r.get('EC number') or r.get('ec') or '').strip()
            if not acc:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO meta(accession, protein_name, taxon_id, organism_name, length) VALUES (?,?,?,?,?)",
                (
                    acc,
                    (r.get('Protein names') or r.get('protein_name') or ''),
                    (r.get('Taxon ID') or r.get('taxon_id') or r.get('organism_id') or ''),
                    (r.get('Organism') or r.get('organism_name') or ''),
                    int((r.get('Length') or r.get('length') or 0) or 0),
                ),
            )
            n_meta += 1
            if not ec:
                continue
            ecs = [e.strip() for e in ec.split(';') if e.strip()]
            for ec_full in ecs:
                parts = ec_full.split('.')
                parts += [''] * (4 - len(parts))
                ec1, ec2, ec3, ec4 = parts[:4]
                cur.execute(
                    "INSERT OR IGNORE INTO long(accession, ec_full, ec1, ec2, ec3, ec4) VALUES (?,?,?,?,?,?)",
                    (acc, ec_full, ec1, ec2, ec3, ec4),
                )
                n_long += 1
    conn.commit()
    return n_meta, n_long


def build_fasta_index(conn: sqlite3.Connection, fasta_path: Path) -> int:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS fasta_index(accession TEXT PRIMARY KEY, start_offset INTEGER, end_offset INTEGER)")
    n = 0
    with open(fasta_path, "rb") as f:
        pos = 0
        acc = None
        start = None
        while True:
            line = f.readline()
            if not line:
                # EOF: finalize last record
                if acc is not None and start is not None:
                    cur.execute("INSERT OR REPLACE INTO fasta_index(accession,start_offset,end_offset) VALUES (?,?,?)", (acc, int(start), int(pos)))
                    n += 1
                break
            if line.startswith(b">"):
                # finalize previous record before starting a new one
                if acc is not None and start is not None:
                    cur.execute("INSERT OR REPLACE INTO fasta_index(accession,start_offset,end_offset) VALUES (?,?,?)", (acc, int(start), int(pos)))
                    n += 1
                # parse new accession
                parts = line.decode('utf-8', 'ignore').split('|')
                acc = parts[1] if len(parts) > 1 else line.decode('utf-8', 'ignore')[1:].split()[0]
                start = f.tell()  # start of sequence content (next byte)
            pos = f.tell()
    conn.commit()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/uniprot_ec")
    ap.add_argument("--db", default="data/uniprot_ec/trembl_index.sqlite")
    ap.add_argument("--tsv", default="data/uniprot_ec/trembl_ec.tsv")
    ap.add_argument("--fasta", default="data/uniprot_ec/trembl_ec.fasta")
    args = ap.parse_args()

    db = Path(args.db)
    tsv = Path(args.tsv)
    fa = Path(args.fasta)
    if not tsv.exists() or not fa.exists():
        raise SystemExit("[index] Missing TSV or FASTA. Run mirror script first.")
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    n_meta, n_long = build_long_from_tsv(conn, tsv)
    n_fa = build_fasta_index(conn, fa)
    print(f"[index] meta={n_meta} long_rows={n_long} fasta_records={n_fa} → {db}")
    if n_meta == 0 or n_long == 0:
        print("[index][warn] TrEMBL index has zero metadata rows; ensure the mirror TSV is complete (use --force).")


if __name__ == "__main__":
    main()
