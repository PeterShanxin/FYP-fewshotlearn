#!/usr/bin/env python3
"""Targeted TrEMBL top-up with per-EC caching.

Workflow
--------
1. Inspect Swiss-Prot counts to find underfilled ECs (4-digit, no '-') that
   have fewer than K_train + Q_train sequences (plus optional overshoot).
2. For each underfilled EC fetch ONLY the necessary TrEMBL entries, cache them
   locally (TSV+FASTA merged into ready-to-use rows), and reuse cache on later
   runs. Optional TTL lets you refresh occasionally.
3. Write a merged table `merged_ec_joined.tsv` that combines Swiss-Prot rows
   with the cached TrEMBL rows (unique by accession, labelled with `source`).
4. Optionally augment `train.jsonl` so every EC has at least
   `target_min + overshoot` accessions; val/test remain Swiss-Prot only by
   virtue of the split builder using `split_source: SwissProt`.

This replaces the previous full TrEMBL mirror logic. Use two passes in the
pipeline: (a) fetch+merge before `prepare_split`, (b) offline augment after the
splits are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import urllib.error
import urllib.parse
import urllib.request

try:  # optional dependency for pooled HTTP
    import urllib3  # type: ignore
except Exception:  # pragma: no cover
    urllib3 = None


# ---------------------------------------------------------------------------
# CLI helpers / config
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def determine_target_min(cfg: dict, cli_target: str) -> int:
    if cli_target != "auto":
        return max(1, int(cli_target))
    episode = cfg.get("episode", {})
    k = int(episode.get("K_train", episode.get("K", 1)))
    q = int(episode.get("Q_train", episode.get("Q", 5)))
    return max(1, k + q)


# ---------------------------------------------------------------------------
# Swiss-Prot utilities
# ---------------------------------------------------------------------------

SWISS_COLUMNS = [
    "source",
    "accession",
    "ec",
    "protein_name",
    "taxon_id",
    "organism_name",
    "length",
    "sequence",
]


def load_swiss_counts(long_tsv: Path) -> Dict[str, int]:
    counts: Dict[str, Set[str]] = {}
    with open(long_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            ec_full = (r.get("ec_full") or "").strip()
            if not is_full_ec(ec_full):
                continue
            acc = (r.get("accession") or "").strip()
            if not acc:
                continue
            counts.setdefault(ec_full, set()).add(acc)
    return {ec: len(accs) for ec, accs in counts.items()}


def load_swiss_accessions(joined_tsv: Path) -> Set[str]:
    accs: Set[str] = set()
    with open(joined_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            acc = (r.get("accession") or r.get("Accession") or "").strip()
            if acc:
                accs.add(acc)
    return accs


def write_swiss_to_merged(joined_tsv: Path, out_path: Path) -> Tuple[int, Set[str]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(SWISS_COLUMNS)
    count = 0
    accs: Set[str] = set()
    with open(joined_tsv, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        reader = csv.DictReader(fin, delimiter="\t")
        for r in reader:
            acc = (r.get("accession") or r.get("Accession") or "").strip()
            seq = (r.get("sequence") or "").strip()
            if not acc or not seq:
                continue
            writer.writerow([
                "SwissProt",
                acc,
                (r.get("ec") or r.get("EC number") or "").strip(),
                (r.get("protein_name") or r.get("Protein names") or "").strip(),
                (r.get("taxon_id") or r.get("Taxon ID") or r.get("organism_id") or r.get("Taxonomic identifier") or "").strip(),
                (r.get("organism_name") or r.get("Organism") or "").strip(),
                (r.get("length") or r.get("Length") or "").strip(),
                seq,
            ])
            count += 1
            accs.add(acc)
    return count, accs


# ---------------------------------------------------------------------------
# TrEMBL fetch + caching helpers
# ---------------------------------------------------------------------------

FIELDS = "accession,ec,protein_name,organism_id,organism_name,length"
SEARCH_BASE = "https://rest.uniprot.org/uniprotkb/search"
CACHE_META = "meta.json"
CACHE_DATA = "data.jsonl"

_HTTP_BACKEND = "urllib"
_POOL = None


def init_http_pool(concurrency: int) -> None:
    global _HTTP_BACKEND, _POOL
    if urllib3 is None or _POOL is not None:
        return
    headers = {
        "User-Agent": "fsl-ec-topup/1.0 (+https://github.com/)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    retries = urllib3.Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 503])  # type: ignore
    _POOL = urllib3.PoolManager(num_pools=max(4, concurrency * 2), headers=headers, retries=retries)  # type: ignore
    _HTTP_BACKEND = "urllib3"


def http_get(url: str, timeout: int) -> Tuple[bytes, dict]:
    if _HTTP_BACKEND == "urllib3" and _POOL is not None:
        resp = _POOL.request("GET", url, timeout=timeout)  # type: ignore
        data = bytes(resp.data or b"")
        headers = {k: v for k, v in resp.headers.items()}  # type: ignore
        return data, headers
    req = urllib.request.Request(url, headers={
        "User-Agent": "fsl-ec-topup/1.0 (+https://github.com/)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), dict(resp.headers)


def http_get_timed(url: str, timeout: int) -> Tuple[bytes, dict, float]:
    t0 = datetime.now().timestamp()
    data, headers = http_get(url, timeout)
    dt = max(1e-9, datetime.now().timestamp() - t0)
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
            now = datetime.now().timestamp()
            if self.next_time <= now:
                self.next_time = now + self.interval
                return
            sleep_for = self.next_time - now
            self.next_time += self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def build_search_url(query: str, fmt: str, size: int, fields: Optional[str] = None) -> str:
    enc = urllib.parse.quote(query, safe="")
    url = f"{SEARCH_BASE}?query={enc}&format={fmt}&size={size}"
    if fields and fmt.lower() == "tsv":
        url += f"&fields={fields}"
    return url


def sanitize_ec(ec: str) -> str:
    return ec.replace("/", "_").replace(".", "_").replace("-", "neg")


def cache_paths(cache_dir: Path, ec: str) -> Tuple[Path, Path, Path]:
    safe = sanitize_ec(ec)
    ec_dir = cache_dir / safe
    return ec_dir, ec_dir / CACHE_DATA, ec_dir / CACHE_META


def load_cached_rows(cache_dir: Path, ec: str, refresh_days: int) -> Optional[List[dict]]:
    ec_dir, data_path, meta_path = cache_paths(cache_dir, ec)
    if not data_path.exists() or not meta_path.exists():
        return None
    if refresh_days > 0:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(meta.get("fetched_at"))
        if datetime.utcnow() - fetched_at > timedelta(days=refresh_days):
            return None
    rows: List[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_cached_rows(cache_dir: Path, ec: str, rows: List[dict]) -> None:
    ec_dir, data_path, meta_path = cache_paths(cache_dir, ec)
    ec_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    meta = {
        "fetched_at": datetime.utcnow().isoformat(),
        "row_count": len(rows),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def fetch_trembl_rows(
    ec: str,
    cap: int,
    timeout: int,
    limiter: Optional[RateLimiter],
) -> Tuple[List[dict], Dict[str, float]]:
    tiers = [
        f'reviewed:false AND ec:"{ec}" AND (existence:1 OR existence:2 OR existence:3) AND fragment:false',
        f'reviewed:false AND ec:"{ec}" AND (existence:1 OR existence:2 OR existence:3 OR existence:4) AND fragment:false',
        f'reviewed:false AND ec:"{ec}"',
    ]
    last_err: Optional[Exception] = None
    for q in tiers:
        try:
            tsv_url = build_search_url(q, fmt="tsv", size=cap, fields=FIELDS)
            fasta_url = build_search_url(q, fmt="fasta", size=cap)
            if limiter:
                limiter.acquire()
            data, _, dt_tsv = http_get_timed(tsv_url, timeout)
            rows = list(csv.DictReader(data.decode("utf-8", "replace").splitlines(), delimiter="\t"))
            if limiter:
                limiter.acquire()
            fasta_data, _, dt_fa = http_get_timed(fasta_url, timeout)
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
            prepared = prepare_rows(rows, acc2seq, ec)
            metrics = {
                "tsv_bytes": float(len(data)),
                "fasta_bytes": float(len(fasta_data)),
                "tsv_sec": float(dt_tsv),
                "fasta_sec": float(dt_fa),
            }
            if prepared:
                return prepared, metrics
        except Exception as exc:  # network, HTTP, etc.
            last_err = exc
    if last_err:
        print(f"[topup][warn] fetch failed for EC {ec}: {last_err}")
    return [], {"tsv_bytes": 0.0, "fasta_bytes": 0.0, "tsv_sec": 0.0, "fasta_sec": 0.0}


def prepare_rows(rows: List[dict], acc2seq: Dict[str, str], ec_selected: Optional[str] = None) -> List[dict]:
    prepared: List[dict] = []
    for r in rows:
        acc = (r.get("Accession") or r.get("accession") or r.get("Entry") or "").strip()
        if not acc:
            continue
        seq = acc2seq.get(acc, "").strip()
        if not seq:
            continue
        ecs = (r.get("EC number") or r.get("ec") or "").strip()
        if ec_selected and ec_selected not in [e.strip() for e in ecs.split(';') if e.strip()]:
            continue
        prepared.append(
            {
                "source": "TrEMBL",
                "accession": acc,
                "ec": ecs,
                "protein_name": (r.get("Protein names") or r.get("protein_name") or "").strip(),
                "taxon_id": (r.get("Taxon ID") or r.get("taxon_id") or r.get("organism_id") or r.get("Taxonomic identifier") or "").strip(),
                "organism_name": (r.get("Organism") or r.get("organism_name") or "").strip(),
                "length": (r.get("Length") or r.get("length") or "").strip(),
                "sequence": seq,
            }
        )
    return prepared


# ---------------------------------------------------------------------------
# EC helpers
# ---------------------------------------------------------------------------


def is_full_ec(ec: str) -> bool:
    parts = ec.split('.')
    return len(parts) == 4 and all(part and part != '-' for part in parts)


@dataclass
class ECSelection:
    ec: str
    swiss_count: int
    cached_rows: List[dict]
    selected_rows: List[dict]
    added_count: int
    used_cache: bool
    fetched_bytes: float


# ---------------------------------------------------------------------------
# Selection & augmentation
# ---------------------------------------------------------------------------


def select_rows_for_ec(
    ec: str,
    cached_rows: List[dict],
    existing_acc: Set[str],
    swiss_count: int,
    target: int,
    per_ec_cap: int,
) -> Tuple[List[dict], int]:
    if swiss_count >= target:
        return [], swiss_count
    needed = target - swiss_count
    rows_sorted = sorted(cached_rows, key=lambda r: r["accession"])
    selected: List[dict] = []
    for row in rows_sorted:
        acc = row["accession"]
        if acc in existing_acc:
            continue
        selected.append(row)
        existing_acc.add(acc)
        swiss_count += 1
        if len(selected) >= per_ec_cap or swiss_count >= target:
            break
    return selected, swiss_count


def write_trembl_rows(out_path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in rows:
            writer.writerow(
                [
                    row["source"],
                    row["accession"],
                    row["ec"],
                    row["protein_name"],
                    row["taxon_id"],
                    row["organism_name"],
                    row["length"],
                    row["sequence"],
                ]
            )
            count += 1
    return count


def augment_train_split(
    splits_dir: Path,
    selected_by_ec: Dict[str, List[dict]],
    target_min: int,
) -> Tuple[int, int]:
    train_path = splits_dir / "train.jsonl"
    if not train_path.exists():
        return 0, 0
    added_ec = 0
    added_acc = 0
    cache_iters = {
        ec: iter([row["accession"] for row in rows])
        for ec, rows in selected_by_ec.items()
    }
    out_lines: List[str] = []
    with open(train_path, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            ec = obj["ec"]
            accs: List[str] = list(obj["accessions"])
            need = max(0, target_min - len(accs))
            if need > 0 and ec in cache_iters:
                pool = cache_iters[ec]
                appended = []
                while need > 0:
                    try:
                        candidate = next(pool)
                    except StopIteration:
                        break
                    if candidate in accs:
                        continue
                    accs.append(candidate)
                    appended.append(candidate)
                    need -= 1
                if appended:
                    added_ec += 1
                    added_acc += len(appended)
            obj["accessions"] = accs
            out_lines.append(json.dumps(obj))
    with open(train_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(out_lines) + "\n")
    return added_ec, added_acc


def write_summary(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Targeted TrEMBL top-up with caching")
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--target-min", default="auto")
    ap.add_argument("--max-ecs", type=int, default=None)
    ap.add_argument("--per-ec-cap", type=int, default=None)
    ap.add_argument("--overshoot", type=int, default=None)
    ap.add_argument("--buffer", type=int, default=None)  # alias for overshoot
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--refresh-days", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--rps", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--augment-train", action="store_true")
    ap.add_argument("--augment-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Override config enable flag")
    ap.add_argument("--summary-json", default=None)
    ap.add_argument(
        "--fetch-overshoot-pct",
        type=float,
        default=None,
        help="Extra fraction of needed rows to request per EC (e.g., 0.15 → +15%)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    top_cfg = cfg.get("trembl_topup", {}) or {}
    enabled = bool(top_cfg.get("enable", False))
    if not enabled and not args.force:
        print("[topup] trembl_topup.enable is false; skipping (use --force to override).")
        return

    cache_dir = Path(args.cache_dir or top_cfg.get("cache_dir", "data/uniprot_ec/trembl_cache"))
    per_ec_cap = int(args.per_ec_cap or top_cfg.get("per_ec_cap", 100))
    overshoot = args.overshoot
    if overshoot is None:
        overshoot = args.buffer if args.buffer is not None else top_cfg.get("overshoot", 2)
    overshoot = int(overshoot)
    fetch_overshoot_pct = args.fetch_overshoot_pct
    if fetch_overshoot_pct is None:
        fetch_overshoot_pct = top_cfg.get("fetch_overshoot_pct", 0.15)
    try:
        fetch_overshoot_pct = float(fetch_overshoot_pct)
    except Exception:
        fetch_overshoot_pct = 0.15
    if fetch_overshoot_pct < 0:
        fetch_overshoot_pct = 0.0
    refresh_days = int(args.refresh_days or top_cfg.get("refresh_days", 30))
    concurrency = int(args.concurrency or top_cfg.get("concurrency", 4))
    rps = float(args.rps or top_cfg.get("rps", 8.0))
    timeout = int(args.timeout or top_cfg.get("timeout", 120))
    max_ecs = args.max_ecs or top_cfg.get("max_ecs")
    if max_ecs is not None:
        max_ecs = int(max_ecs)

    paths = cfg.get("paths", {})
    data_root = Path(paths.get("data_root", "data/uniprot_ec"))
    swiss_joined = data_root / "swissprot_ec_joined.tsv"
    swiss_long = data_root / "swissprot_ec_joined_long.tsv"
    merged_joined = data_root / "merged_ec_joined.tsv"
    splits_dir = Path(paths.get("splits_dir", "data/splits"))

    if not swiss_joined.exists() or not swiss_long.exists():
        raise SystemExit("[topup] Swiss-Prot TSVs missing; run fetch step first.")

    target_min_base = determine_target_min(cfg, args.target_min)
    target_min_aug = target_min_base + max(0, overshoot)

    summary: Dict[str, object] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target_min_base": target_min_base,
        "target_min_aug": target_min_aug,
        "fetch_overshoot_pct": fetch_overshoot_pct,
        "needed_before": 0,
        "remaining_after_base": 0,
        "remaining_after_aug": 0,
        "trembl_added": 0,
    }

    counts = load_swiss_counts(swiss_long)
    swiss_accessions = load_swiss_accessions(swiss_joined)
    existing_accessions = set(swiss_accessions)

    underfilled = [ec for ec, c in counts.items() if is_full_ec(ec) and c < target_min_base]
    summary["needed_before"] = len(underfilled)
    underfilled.sort(key=lambda ec: counts.get(ec, 0))
    if max_ecs is not None:
        underfilled = underfilled[:max_ecs]

    if not underfilled:
        print("[topup] No ECs require top-up; writing Swiss-Prot only.")
        write_swiss_to_merged(swiss_joined, merged_joined)
        write_summary(args.summary_json, summary)
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    init_http_pool(concurrency)
    limiter = RateLimiter(rps) if rps > 0 else None
    try:
        backend = _HTTP_BACKEND
    except Exception:
        backend = "urllib"
    print(
        f"[topup] http_backend={backend} concurrency={concurrency} rps={rps} timeout={timeout} "
        f"offline={args.offline} candidates={len(underfilled)} fetch_overshoot_pct={fetch_overshoot_pct}"
    )
    sys.stdout.flush()

    results: List[ECSelection] = []
    total = len(underfilled)
    for idx, ec in enumerate(underfilled, start=1):
        swiss_count = counts.get(ec, 0)
        need_for_target = max(0, target_min_aug - swiss_count)
        base_fetch_cap = max(need_for_target, 1)
        if fetch_overshoot_pct > 0:
            extra_fetch = max(1, math.ceil(base_fetch_cap * fetch_overshoot_pct))
        else:
            extra_fetch = 0
        fetch_cap = min(per_ec_cap, base_fetch_cap + extra_fetch)

        print(
            f"[topup] [{idx}/{total}] EC {ec} processing… need={need_for_target} fetch_cap={fetch_cap} "
            f"extra={extra_fetch}"
        )
        sys.stdout.flush()
        cached_rows = load_cached_rows(cache_dir, ec, refresh_days) if cache_dir else None
        used_cache = cached_rows is not None
        if used_cache:
            print(f"[topup] [{idx}/{total}] EC {ec} using cache (rows={len(cached_rows)})")
            sys.stdout.flush()
        metrics_bytes = 0.0
        if cached_rows is None:
            if args.offline:
                print(f"[topup][warn] EC {ec} missing from cache in offline mode; skipping.")
                cached_rows = []
            else:
                fetched_rows, metrics = fetch_trembl_rows(ec, fetch_cap, timeout, limiter)
                metrics_bytes = metrics.get("tsv_bytes", 0.0) + metrics.get("fasta_bytes", 0.0)
                cached_rows = fetched_rows
                if cache_dir and fetched_rows:
                    save_cached_rows(cache_dir, ec, fetched_rows)
                print(
                    f"[topup] [{idx}/{total}] EC {ec} fetched rows={len(cached_rows)} bytes={int(metrics_bytes)}"
                )
                sys.stdout.flush()
        selected, new_count = select_rows_for_ec(ec, cached_rows or [], existing_accessions, swiss_count, target_min_aug, per_ec_cap)
        counts[ec] = new_count
        results.append(
            ECSelection(
                ec=ec,
                swiss_count=swiss_count,
                cached_rows=cached_rows or [],
                selected_rows=selected,
                added_count=len(selected),
                used_cache=used_cache,
                fetched_bytes=metrics_bytes,
            )
        )

    # Build merged table (Swiss + selected TrEMBL rows)
    swiss_count_written, _ = write_swiss_to_merged(swiss_joined, merged_joined)
    total_trembl_added = write_trembl_rows(
        merged_joined,
        (row for res in results for row in res.selected_rows),
    )
    print(
        f"[topup] Swiss-Prot rows written: {swiss_count_written}; TrEMBL rows added: {total_trembl_added}"
    )
    summary["trembl_added"] = int(total_trembl_added)

    remaining_base = sum(
        1 for ec, cnt in counts.items() if is_full_ec(ec) and cnt < target_min_base
    )
    remaining_aug = sum(
        1 for ec, cnt in counts.items() if is_full_ec(ec) and cnt < target_min_aug
    )
    summary["remaining_after_base"] = int(remaining_base)
    summary["remaining_after_aug"] = int(remaining_aug)
    write_summary(args.summary_json, summary)

    for res in results:
        status = "cache" if res.used_cache else "fetch"
        print(
            f"[topup] EC {res.ec}: added={res.added_count} (status={status}, swiss={res.swiss_count}, total={counts.get(res.ec, res.swiss_count)})"
        )

    if args.augment_train or args.augment_only:
        selected_map = {res.ec: res.selected_rows for res in results if res.selected_rows}
        added_ec, added_acc = augment_train_split(splits_dir, selected_map, target_min_aug)
        print(
            f"[topup] Train split augmented: ECs={added_ec}, accessions_added={added_acc} → {splits_dir / 'train.jsonl'}"
        )


if __name__ == "__main__":
    main()
