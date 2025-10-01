#!/usr/bin/env python3
"""Targeted TrEMBL top-up for underfilled EC classes (Option A).

Adds a small, filtered slice of TrEMBL sequences for ECs that don't
have enough Swiss‑Prot sequences to support identity‑disjoint sampling
with the configured K and Q.

What it does
- Reads Swiss‑Prot EC distribution from the existing joined TSVs
  (`swissprot_ec_joined.tsv` and `swissprot_ec_joined_long.tsv`).
- Finds full ECs (x.y.z.w, no '-') with Swiss‑Prot count < target_min.
- For up to `--max-ecs`, fetches TrEMBL entries for each EC with filters:
  reviewed:false AND ec:"x.y.z.w" AND (existence:1 OR 2 OR 3) AND fragment:false
  Limited to `--cap-per-ec` entries per EC to keep the top‑up small.
- Joins TSV+FASTA per EC and writes a merged joined table with a new
  leading column `source` ∈ {SwissProt, TrEMBL} at:
    data/uniprot_ec/merged_ec_joined.tsv
- Optionally overwrites the training split to add fetched TrEMBL accessions
  to underfilled EC classes up to `target_min`.

Usage
  python scripts/topup_trembl_targeted.py \
    --config config.smoke.yaml \
    --target-min auto \
    --cap-per-ec 5 \
    --max-ecs 3 \
    --augment-train

Notes
- Keeps val/test splits unchanged (Swiss‑Prot only). Only the train split
  is optionally augmented. The merged joined TSV is used later by embedding.
- Keeps downloads small for smoke runs; increase caps for real runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import time
import urllib.parse
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class TopUpConfig:
    cfg_path: str
    data_root: Path
    swiss_joined: Path
    swiss_long: Path
    merged_joined: Path
    splits_dir: Path
    target_min: int
    cap_per_ec: int
    max_ecs: int
    augment_train: bool


FIELDS = "accession,ec,protein_name,organism_id,organism_name,length"
SEARCH_BASE = "https://rest.uniprot.org/uniprotkb/search"

# Optional connection pooling via urllib3 if available
_HTTP_BACKEND = "urllib"
_POOL = None
try:  # pragma: no cover - optional dependency at runtime
    import urllib3  # type: ignore

    _HTTP_BACKEND = "urllib3"
except Exception:  # pragma: no cover
    urllib3 = None  # type: ignore


def load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def determine_target_min(cfg: dict, cli_target: str) -> int:
    if cli_target != "auto":
        return max(1, int(cli_target))
    ep = cfg.get("episode", {})
    k = int(ep.get("K_train", ep.get("K", 1)))
    q = int(ep.get("Q_train", ep.get("Q", 5)))
    # Default to at least 6 (1+5) but allow config to drive
    return max(1, k + q)


def build_search_url(query: str, fmt: str, fields: Optional[str] = None, size: int = 500) -> str:
    enc = urllib.parse.quote(query, safe="")
    url = f"{SEARCH_BASE}?query={enc}&format={fmt}&size={size}"
    if fields and fmt.lower() == "tsv":
        url += f"&fields={fields}"
    return url


def _init_pool(concurrency: int) -> None:
    global _POOL
    if _HTTP_BACKEND != "urllib3" or _POOL is not None:
        return
    headers = {
        "User-Agent": "fsl-ec-topup/1.0 (+https://github.com/)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    retries = urllib3.Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 503])  # type: ignore
    _POOL = urllib3.PoolManager(num_pools=max(4, int(concurrency) * 2), headers=headers, retries=retries)  # type: ignore


def http_get(url: str, timeout: int = 120) -> Tuple[bytes, dict]:
    if _HTTP_BACKEND == "urllib3" and _POOL is not None:
        resp = _POOL.request("GET", url, timeout=timeout)  # type: ignore
        # urllib3 Response .data contains full payload
        data = bytes(resp.data or b"")
        headers = {k: v for k, v in resp.headers.items()}  # type: ignore
        return data, headers
    # Fallback: urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "fsl-ec-topup/1.0 (+https://github.com/)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), dict(resp.headers)


def http_get_timed(url: str, timeout: int = 120) -> Tuple[bytes, dict, float]:
    t0 = time.time()
    data, headers = http_get(url, timeout=timeout)
    dt = max(1e-9, time.time() - t0)
    return data, headers, dt


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / float(rps) if rps and rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if self.next_time <= now:
                self.next_time = now + self.interval
                return
            sleep_for = self.next_time - now
            self.next_time += self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def fetch_trembl_for_ec(ec: str, cap: int, timeout: int, limiter: Optional[RateLimiter]) -> Tuple[List[dict], Dict[str, str], Dict[str, float]]:
    """Fetch up to `cap` TrEMBL rows and sequences for an exact full EC.

    Returns: (tsv_rows, acc2seq)
    """
    tiers = [
        f'reviewed:false AND ec:"{ec}" AND (existence:1 OR existence:2 OR existence:3) AND fragment:false',
        f'reviewed:false AND ec:"{ec}" AND (existence:1 OR existence:2 OR existence:3 OR existence:4) AND fragment:false',
        f'reviewed:false AND ec:"{ec}"',
    ]
    last_err: Optional[Exception] = None
    for q in tiers:
        try:
            tsv_url = build_search_url(q, fmt="tsv", fields=FIELDS, size=cap)
            fasta_url = build_search_url(q, fmt="fasta", size=cap)
            if limiter:
                limiter.acquire()
            data, _, dt_tsv = http_get_timed(tsv_url, timeout=timeout)
            # Decode TSV
            rows: List[dict] = []
            text = data.decode("utf-8", "replace")
            reader = csv.DictReader(text.splitlines(), delimiter="\t")
            for r in reader:
                rows.append(r)

            if limiter:
                limiter.acquire()
            fasta_data, _, dt_fa = http_get_timed(fasta_url, timeout=timeout)
            acc2seq: Dict[str, str] = {}
            acc = None
            seq_lines: List[str] = []
            for line in fasta_data.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if acc and seq_lines:
                        acc2seq[acc] = "".join(seq_lines)
                    parts = line.split("|")
                    acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if acc and seq_lines:
                acc2seq[acc] = "".join(seq_lines)

            if rows and acc2seq:
                metrics = {
                    "tsv_bytes": float(len(text.encode('utf-8'))),
                    "fasta_bytes": float(len(fasta_data)),
                    "tsv_sec": float(dt_tsv),
                    "fasta_sec": float(dt_fa),
                }
                return rows, acc2seq, metrics
        except Exception as e:  # store and try next tier
            last_err = e
            continue
    if last_err:
        raise last_err
    return [], {}, {"tsv_bytes": 0.0, "fasta_bytes": 0.0, "tsv_sec": 0.0, "fasta_sec": 0.0}


def is_full_ec(ec: str) -> bool:
    if not ec:
        return False
    parts = ec.split(".")
    if len(parts) != 4:
        return False
    return all(p and p != "-" for p in parts)


def swiss_ec_counts(long_tsv: Path) -> Dict[str, int]:
    counts: Dict[str, Set[str]] = {}
    with open(long_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            ec_full = (r.get("ec_full") or "").strip()
            if not is_full_ec(ec_full):
                continue
            acc = str(r.get("accession") or "").strip()
            if not acc:
                continue
            s = counts.setdefault(ec_full, set())
            s.add(acc)
    return {ec: len(s) for ec, s in counts.items()}


def merged_writer_header(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source", "accession", "ec", "protein_name", "taxon_id", "organism_name", "length", "sequence"])


def append_swiss_to_merged(swiss_joined: Path, out_merged: Path) -> int:
    n = 0
    with open(swiss_joined, "r", encoding="utf-8") as fin, open(out_merged, "a", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        w = csv.writer(fout, delimiter="\t")
        for r in reader:
            acc = r.get("accession") or r.get("Accession")
            seq = r.get("sequence")
            if not acc or not seq:
                continue
            w.writerow([
                "SwissProt",
                r.get("accession") or r.get("Accession") or "",
                r.get("ec") or r.get("EC number") or "",
                r.get("protein_name") or r.get("Protein names") or "",
                r.get("taxon_id") or r.get("Taxon ID") or r.get("organism_id") or r.get("Taxonomic identifier") or "",
                r.get("organism_name") or r.get("Organism") or "",
                r.get("length") or r.get("Length") or "",
                seq,
            ])
            n += 1
    return n


def append_trembl_rows_to_merged(rows: List[dict], acc2seq: Dict[str, str], out_merged: Path) -> int:
    n = 0
    with open(out_merged, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            acc = r.get("Accession") or r.get("accession") or r.get("Entry")
            if not acc:
                continue
            seq = acc2seq.get(acc)
            if not seq:
                continue
            w.writerow([
                "TrEMBL",
                acc,
                r.get("EC number") or r.get("ec") or "",
                r.get("Protein names") or r.get("protein_name") or "",
                r.get("Taxon ID") or r.get("taxon_id") or r.get("organism_id") or r.get("Taxonomic identifier") or "",
                r.get("Organism") or r.get("organism_name") or "",
                r.get("Length") or r.get("length") or "",
                seq,
            ])
            n += 1
    return n


def build_trembl_ec_index(out_merged: Path) -> Dict[str, List[str]]:
    """Return {ec_full: [trembl_accessions]} from the merged table."""
    mapping: Dict[str, List[str]] = {}
    with open(out_merged, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if (r.get("source") or "") != "TrEMBL":
                continue
            acc = (r.get("accession") or "").strip()
            ecs = (r.get("ec") or "").split(";")
            ecs = [e.strip() for e in ecs if e.strip()]
            for ec in ecs:
                if not is_full_ec(ec):
                    continue
                mapping.setdefault(ec, []).append(acc)
    return mapping


def augment_train_split(splits_dir: Path, out_merged: Path, target_min: int) -> Tuple[int, int]:
    """Append TrEMBL accessions into train.jsonl for ECs under target_min.

    Returns: (n_ec_augmented, n_accessions_added)
    """
    train_path = splits_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")
    trembl_index = build_trembl_ec_index(out_merged)

    # Backup
    backup = splits_dir / "train.pre_topup.jsonl"
    if not backup.exists():
        backup.write_bytes(train_path.read_bytes())

    n_ec_aug = 0
    n_added = 0
    out_lines: List[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ec = obj["ec"]
            accs: List[str] = list(obj["accessions"])  # copy
            have = len(accs)
            if have < target_min:
                pool = trembl_index.get(ec, [])
                take = max(0, target_min - have)
                # deterministically take from start; dedup vs existing
                add = []
                seen = set(accs)
                for a in pool:
                    if a in seen:
                        continue
                    add.append(a)
                    seen.add(a)
                    if len(add) >= take:
                        break
                if add:
                    n_ec_aug += 1
                    n_added += len(add)
                    accs.extend(add)
            obj["accessions"] = accs
            out_lines.append(json.dumps(obj))
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return n_ec_aug, n_added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--target-min", default="auto", help="Target min sequences per EC in train (auto=K_train+Q_train)")
    ap.add_argument("--cap-per-ec", type=int, default=20, help="Max TrEMBL entries to fetch per EC")
    ap.add_argument("--max-ecs", type=int, default=50, help="Max ECs to top up (for smoke keep small)")
    ap.add_argument("--augment-train", action="store_true", help="Augment train split with fetched TrEMBL accessions")
    ap.add_argument("--augment-only", action="store_true", help="Only augment train split using existing merged TSV (skip fetching)")
    ap.add_argument("--buffer", type=int, default=0, help="Extra sequences beyond target-min when augmenting train (safety margin)")
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel EC fetch workers (polite: 2-8 recommended)")
    ap.add_argument("--rps", type=float, default=8.0, help="Global requests-per-second cap across all workers")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout per request (seconds)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    data_root = Path(paths.get("data_root", "data/uniprot_ec"))
    # Always read Swiss‑Prot joined as the base; merged file may be configured elsewhere
    swiss_joined = data_root / "swissprot_ec_joined.tsv"
    swiss_long = data_root / "swissprot_ec_joined_long.tsv"
    merged_joined = data_root / "merged_ec_joined.tsv"
    splits_dir = Path(paths.get("splits_dir", "data/splits"))

    target_min = determine_target_min(cfg, args.target_min)
    target_min_aug = target_min + max(0, int(args.buffer))
    cap_per_ec = max(1, int(args.cap_per_ec))
    max_ecs = max(1, int(args.max_ecs))

    if not swiss_joined.exists():
        raise SystemExit(f"Swiss‑Prot joined TSV not found: {swiss_joined}. Run fetch step first.")
    if not swiss_long.exists():
        raise SystemExit(f"Swiss‑Prot long TSV not found: {swiss_long}. Run fetch step first.")

    print(json.dumps({
        "config": args.config,
        "target_min": target_min,
        "target_min_aug": target_min_aug,
        "cap_per_ec": cap_per_ec,
        "max_ecs": max_ecs,
        "augment_train": bool(args.augment_train),
        "swiss_joined": str(swiss_joined),
        "swiss_long": str(swiss_long),
        "merged_joined": str(merged_joined),
        "splits_dir": str(splits_dir),
    }, indent=2))

    # Augment-only path: do not fetch, just apply augmentation using existing merged file
    if args.augment_only:
        if not merged_joined.exists():
            raise SystemExit(f"[topup] --augment-only requested but merged file not found: {merged_joined}")
        if args.augment_train:
            n_ec_aug, n_added = augment_train_split(splits_dir, merged_joined, target_min_aug)
            print(f"[topup] Train split augmented: ECs={n_ec_aug}, accessions_added={n_added} → {splits_dir/'train.jsonl'}")
        else:
            print("[topup] --augment-only set but --augment-train not provided; nothing to do.")
        return

    # Count Swiss‑Prot per full EC
    counts = swiss_ec_counts(swiss_long)
    underfilled = [ec for ec, n in counts.items() if n < target_min]
    underfilled.sort(key=lambda e: counts.get(e, 0))
    take_ecs = underfilled[:max_ecs]

    from tqdm.auto import tqdm
    if not underfilled:
        print("[topup] No underfilled ECs found; nothing to top up.")
    else:
        print(f"[topup] Underfilled ECs: {len(underfilled)}; processing first {len(take_ecs)}")

    merged_writer_header(merged_joined)
    n_sw = append_swiss_to_merged(swiss_joined, merged_joined)

    n_tr_rows = 0
    ec_to_added: Dict[str, int] = {}
    pbar = tqdm(total=len(take_ecs), desc="[topup]", dynamic_ncols=True)
    last_speed = ""

    # Initialize pooling and rate limiting
    _init_pool(args.concurrency)
    limiter = RateLimiter(float(args.rps)) if float(args.rps) > 0 else None

    def _worker(ec: str):
        try:
            rows, acc2seq, m = fetch_trembl_for_ec(ec, cap=cap_per_ec, timeout=int(args.timeout), limiter=limiter)
            # Filter to exact EC rows
            kept: List[dict] = []
            for r in rows:
                ecs = (r.get("EC number") or r.get("ec") or "").split(";")
                ecs = [x.strip() for x in ecs if x.strip()]
                if ec in ecs:
                    kept.append(r)
            return (ec, kept, acc2seq, m, None)
        except Exception as e:
            return (ec, [], {}, {"tsv_bytes": 0.0, "fasta_bytes": 0.0, "tsv_sec": 0.0, "fasta_sec": 0.0}, str(e))

    if take_ecs:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futs = {ex.submit(_worker, ec): ec for ec in take_ecs}
            for fut in as_completed(futs):
                ec, rows_kept, acc2seq, m, err = fut.result()
                added = 0
                if rows_kept:
                    added = append_trembl_rows_to_merged(rows_kept, acc2seq, merged_joined)
                    if added:
                        ec_to_added[ec] = added
                        n_tr_rows += added
                tot_b = float(m.get("tsv_bytes", 0.0) + m.get("fasta_bytes", 0.0))
                tot_s = float(m.get("tsv_sec", 0.0) + m.get("fasta_sec", 0.0))
                mbps = (tot_b / 1e6) / (tot_s or 1e-9)
                if err:
                    last_speed = f"ec={ec} err={str(err)[:32]} net={mbps:.2f}MB/s cum={n_tr_rows}"
                elif not rows_kept:
                    last_speed = f"ec={ec} kept=0 net={mbps:.2f}MB/s cum={n_tr_rows}"
                else:
                    last_speed = f"ec={ec} added={added} net={mbps:.2f}MB/s cum={n_tr_rows}"
                pbar.set_postfix_str(last_speed)
                pbar.update(1)
    pbar.close()

    print(f"[topup] Swiss‑Prot rows: {n_sw}; TrEMBL rows added: {n_tr_rows}")
    if ec_to_added:
        print(f"[topup] EC additions (first 10): {json.dumps(dict(list(ec_to_added.items())[:10]), indent=2)}")
    print(f"[topup] Merged joined written → {merged_joined}")

    if args.augment_train:
        n_ec_aug, n_added = augment_train_split(splits_dir, merged_joined, target_min_aug)
        print(f"[topup] Train split augmented: ECs={n_ec_aug}, accessions_added={n_added} → {splits_dir/'train.jsonl'}")


if __name__ == "__main__":
    main()
