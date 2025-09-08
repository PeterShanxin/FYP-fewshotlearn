#!/usr/bin/env python3
"""Cluster sequences at a fixed identity threshold (default: 50%).

Pipeline step to generate an accession->cluster_id TSV for identity-aware
episode sampling. Prefers MMseqs2, then CD-HIT; falls back to a slow
approximate Python clustering if neither binary is available.

Outputs to paths.clusters_tsv from the config.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_splits(splits_dir: Path) -> List[str]:
    accs: set[str] = set()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = splits_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}. Run prepare_split first.")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                for a in obj.get("accessions", []):
                    accs.add(str(a))
    return sorted(accs)


def read_sequences(joined_tsv: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    with open(joined_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        hmap = {h.lower(): i for i, h in enumerate(header)}
        if "accession" not in hmap or "sequence" not in hmap:
            raise KeyError("Joined TSV must contain 'accession' and 'sequence' columns")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            acc = parts[hmap["accession"]]
            seq = parts[hmap["sequence"]].upper()
            if acc and seq:
                seqs[acc] = seq
    return seqs


def write_fasta(pairs: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, seq in pairs:
            f.write(f">{acc}\n")
            # wrap at 80 cols for readability
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def run(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_quiet(cmd: List[str], cwd: Path | None = None, log_file: Path | None = None) -> None:
    """Run a command suppressing console output; optionally write to a log file."""
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab", buffering=0) as f:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=f, stderr=f)
    else:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cluster_with_mmseqs(fasta: Path, workdir: Path, min_id: float, min_cov: float) -> Dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    db = workdir / "seqDB"
    clu = workdir / "seqClu"
    raw_tsv = workdir / "clusters_raw.tsv"

    threads = str(max(1, (os.cpu_count() or 1)))
    logs = workdir / "logs"
    run_quiet(["mmseqs", "createdb", str(fasta), str(db)], log_file=logs / "mmseqs_createdb.log")
    run_quiet([
        "mmseqs", "cluster", str(db), str(clu), str(workdir / "tmp"),
        "--min-seq-id", str(min_id),
        "-c", str(min_cov), "--cov-mode", "1", "--threads", threads,
    ], log_file=logs / "mmseqs_cluster.log")
    run_quiet(["mmseqs", "createtsv", str(db), str(db), str(clu), str(raw_tsv)], log_file=logs / "mmseqs_createtsv.log")

    # Parse TSV mapping representative->member pairs; ensure reps map to self
    mapping: Dict[str, str] = {}
    with open(raw_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, mem = parts[0], parts[1]
            mapping.setdefault(rep, rep)
            mapping[mem] = rep
    return mapping


def cluster_with_cdhit(fasta: Path, workdir: Path, min_id: float) -> Dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    outroot = workdir / "cdhit"
    threads = str(max(1, (os.cpu_count() or 1)))
    # For 50% identity CD-HIT requires word size -n 2
    run_quiet([
        "cd-hit", "-i", str(fasta), "-o", str(outroot), "-c", str(min_id), "-n", "2",
        "-T", threads, "-M", "0",
    ], log_file=workdir / "logs" / "cdhit.log")
    clstr = Path(str(outroot) + ".clstr")
    mapping: Dict[str, str] = {}
    rep = None
    if not clstr.exists():
        return mapping
    with open(clstr, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                rep = None
                continue
            # Example line: "0 123aa, >ACC123... *" or "0 98aa, >ACC456... at 98%"
            try:
                right = line.split(">", 1)[1]
                acc = right.split("...", 1)[0]
            except Exception:
                continue
            if line.endswith("*"):
                rep = acc
                mapping.setdefault(rep, rep)
            if rep is not None:
                mapping[acc] = rep
    return mapping


def sim_ratio(a: str, b: str) -> float:
    # Fallback approximate similarity (SequenceMatcher-based)
    import difflib
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def cluster_greedy(pairs: List[Tuple[str, str]], min_ratio: float) -> Dict[str, str]:
    # Very slow O(N^2) fallback; warn on large N
    N = len(pairs)
    if N > 5000:
        print(f"[cluster][warn] {N} sequences; Python fallback clustering may be very slow."
              " Install MMseqs2 or CD-HIT for speed.")
    mapping: Dict[str, str] = {}
    for i, (acc_i, seq_i) in enumerate(pairs):
        if acc_i in mapping:
            continue
        # new cluster representative = acc_i
        mapping[acc_i] = acc_i
        for j in range(i + 1, N):
            acc_j, seq_j = pairs[j]
            if acc_j in mapping:
                continue
            if sim_ratio(seq_i, seq_j) >= min_ratio:
                mapping[acc_j] = acc_i
    return mapping


def write_clusters_tsv(mapping: Dict[str, str], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        for acc in sorted(mapping.keys()):
            f.write(f"{acc}\t{mapping[acc]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    splits_dir = Path(paths["splits_dir"]) 
    joined_tsv = Path(paths["joined_tsv"]) 
    out_tsv = Path(paths.get("clusters_tsv", "data/identity/clusters.tsv"))

    min_id = float(cfg.get("cluster_identity", 0.5))
    min_cov = float(cfg.get("cluster_coverage", 0.5))

    accs = read_splits(splits_dir)
    seq_map = read_sequences(joined_tsv)
    pairs = [(a, seq_map[a]) for a in accs if a in seq_map]
    if not pairs:
        raise SystemExit("[cluster] No sequences found for splits; aborting.")

    workdir = Path("data/identity/_work")
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "split_sequences.fasta"
    write_fasta(pairs, fasta)

    mapping: Dict[str, str] = {}
    if shutil.which("mmseqs") is not None:
        print(f"[cluster] Using MMseqs2 at min_id={min_id}, min_cov={min_cov} (quiet; logs under {workdir/'logs'})")
        mapping = cluster_with_mmseqs(fasta, workdir, min_id=min_id, min_cov=min_cov)
    elif shutil.which("cd-hit") is not None:
        print(f"[cluster] Using CD-HIT at min_id={min_id} (quiet; logs under {workdir/'logs'})")
        mapping = cluster_with_cdhit(fasta, workdir, min_id=min_id)
    else:
        print("[cluster] MMseqs2/CD-HIT not found; falling back to slow Python clustering (approximate)")
        mapping = cluster_greedy(pairs, min_ratio=min_id)

    if not mapping:
        raise SystemExit("[cluster] Clustering produced no mapping; aborting.")
    write_clusters_tsv(mapping, Path(out_tsv))
    print(f"[cluster] Wrote clusters TSV for {len(mapping)} accessions -> {out_tsv}")


if __name__ == "__main__":
    main()
