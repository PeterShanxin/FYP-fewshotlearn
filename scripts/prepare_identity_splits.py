#!/usr/bin/env python3
"""Prepare identity-cluster-based splits across multiple cutoffs and folds.

Enhancements:
- Build clusters as connected components of the ≥T% identity graph to ensure
  no cross-fold identity leakage at the threshold (when tools available).
- Optional fast path using MMseqs2 pairwise search with coverage filter; fallback
  to a Python O(N^2) approximate identity for small smoke tests.
- Persist artifacts per threshold under results/split-XX/ (clusters, folds, config) and
  split JSONLs under results/split-XX/fold-YY/{train,val,test}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, DefaultDict, Set

import pandas as pd
import numpy as np
import yaml

# Reuse clustering logic (duplicated minimally from scripts/cluster_sequences.py)
import os
import subprocess
import time
from collections import defaultdict


def run(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_quiet(cmd: List[str], cwd: Path | None = None, log_file: Path | None = None) -> None:
    """Run a command while silencing console noise; optionally tee to a log file."""
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "ab", buffering=0)
        try:
            subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, stdout=f, stderr=f)
        finally:
            f.close()
    else:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, stdout=stdout, stderr=stderr)


def mmseqs_pairwise_edges(fasta: Path, workdir: Path, min_id: float, min_cov: float) -> List[Tuple[str, str]]:
    """Use MMseqs2 easy-search to enumerate pairs with identity ≥ min_id and coverage ≥ min_cov.

    Returns undirected edges (a,b) with a < b to de-duplicate.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    aln = workdir / "pairs.tsv"
    tmp = workdir / "tmp"
    threads = str(max(1, (os.cpu_count() or 1)))
    # Use paths relative to workdir because we set cwd=workdir.
    log_path = workdir / "logs" / "mmseqs_easy_search.log"
    # Quiet external tool output; tee to log file for debugging
    run_quiet([
        "mmseqs", "easy-search", str(fasta.name), str(fasta.name), str(aln.name), str(tmp.name),
        "--min-seq-id", str(min_id),
        "-c", str(min_cov), "--cov-mode", "1", "--threads", threads,
    ], cwd=workdir, log_file=log_path)
    edges: Set[Tuple[str, str]] = set()
    with open(aln, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            a, b = parts[0], parts[1]
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            edges.add((lo, hi))
    return sorted(edges)


def cluster_with_cdhit(fasta: Path, workdir: Path, min_id: float) -> Dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    outroot = workdir / "cdhit"
    threads = str(max(1, (os.cpu_count() or 1)))
    log_path = workdir / "logs" / "cdhit.log"
    run_quiet([
        "cd-hit", "-i", str(fasta), "-o", str(outroot), "-c", str(min_id), "-n", "2",
        "-T", threads, "-M", "0",
    ], cwd=None, log_file=log_path)
    clstr = Path(str(outroot) + ".clstr")
    mapping: Dict[str, str] = {}
    rep = None
    with open(clstr, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                rep = None
                continue
            right = line.split(">", 1)[1]
            acc = right.split("...", 1)[0]
            if line.endswith("*"):
                rep = acc
                mapping.setdefault(rep, rep)
            if rep is not None:
                mapping[acc] = rep
    return mapping

def python_pairwise_edges(pairs: List[Tuple[str, str]], min_ratio: float) -> List[Tuple[str, str]]:
    """Very slow O(N^2) approximate edge builder using difflib ratio.

    Returns undirected edges (a,b) with a < b when ratio ≥ min_ratio.
    """
    import difflib
    edges: List[Tuple[str, str]] = []
    N = len(pairs)
    for i in range(N):
        ai, si = pairs[i]
        for j in range(i + 1, N):
            aj, sj = pairs[j]
            r = difflib.SequenceMatcher(a=si, b=sj).ratio()
            if r >= min_ratio:
                lo, hi = (ai, aj) if ai < aj else (aj, ai)
                edges.append((lo, hi))
    return edges

def connected_components(nodes: Iterable[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}
    def find(x: str) -> str:
        px = parent.get(x, x)
        if px != x:
            parent[x] = find(px)
        else:
            parent.setdefault(x, x)
        return parent[x]
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ra_r = rank.get(ra, 0)
        rb_r = rank.get(rb, 0)
        if ra_r < rb_r:
            parent[ra] = rb
        elif ra_r > rb_r:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] = ra_r + 1
    all_nodes = list(nodes)
    for n in all_nodes:
        parent.setdefault(n, n)
        rank.setdefault(n, 0)
    for a, b in edges:
        union(a, b)
    groups: DefaultDict[str, List[str]] = defaultdict(list)
    for n in all_nodes:
        groups[find(n)].append(n)
    # Deterministic order
    return [sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])]


def write_fasta(pairs: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, seq in pairs:
            f.write(f">{acc}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "ec" not in df.columns:
        raise KeyError("'ec' column not found")
    df = df[(df["ec"].notna()) & (df["ec"].astype(str).str.len() > 0)]
    df["ec"] = df["ec"].astype(str).str.strip()
    df["accession"] = df["accession"].astype(str)
    return df


def expand_multi(df: pd.DataFrame, allow_multi_ec: bool) -> pd.DataFrame:
    df = normalize_df(df)
    if not allow_multi_ec:
        mask_single = ~df["ec"].astype(str).str.contains(";")
        return df[mask_single]
    rows = []
    for _, r in df.iterrows():
        ecs = [e.strip() for e in str(r["ec"]).split(";") if e.strip()]
        for ec in ecs:
            rr = r.copy()
            rr["ec"] = ec
            rows.append(rr)
    return pd.DataFrame(rows)[df.columns] if rows else df.iloc[0:0]


def group_by_ec(df: pd.DataFrame) -> Dict[str, List[str]]:
    grp: Dict[str, List[str]] = {}
    for ec, sub in df.groupby("ec"):
        grp[ec] = sub["accession"].astype(str).tolist()
    return grp


def split_classes(classes: List[str], seed: int, frac_val: float = 0.1) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    classes = classes[:]
    rnd.shuffle(classes)
    n = len(classes)
    if n <= 1:
        return classes, []
    n_val = max(1, int(round(frac_val * n)))
    n_train = max(1, n - n_val)
    return classes[:n_train], classes[n_train:n_train + n_val]


def write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--cutoffs", default="0.1,0.3,0.5,0.7,1.0", help="Comma-separated identity cutoffs")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    seed = int(cfg.get("random_seed", 42))
    min_per_class = int(cfg.get("min_sequences_per_class_for_train", 40))
    allow_multi_ec = bool(cfg.get("allow_multi_ec", False))
    stratify_by = str(cfg.get("stratify_by", "EC_top"))
    identity_definition = str(cfg.get("identity_definition", "tool_default"))

    joined = Path(paths["joined_tsv"]).resolve()
    # We will write thresholded splits under results/split-XX
    results_root = Path(cfg["paths"].get("outputs", "results")).resolve()
    out_root = results_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Load full table and expand multi-ECs if needed
    df = pd.read_csv(joined, sep="\t")
    df = expand_multi(df, allow_multi_ec=allow_multi_ec)

    # Optional downsampling for smoke tests (mirror src.prepare_split.py behavior)
    limit_classes = cfg.get("limit_classes")
    limit_per_class = cfg.get("limit_per_class")
    if limit_classes is not None or limit_per_class is not None:
        counts = df.groupby("ec")["accession"].count().sort_values(ascending=False)
        if limit_classes is not None:
            keep_classes = set(counts.head(max(1, int(limit_classes))).index.tolist())
            df = df[df["ec"].isin(keep_classes)]
        if limit_per_class is not None:
            def _take_n(g: pd.DataFrame) -> pd.DataFrame:
                n = int(limit_per_class or 0)
                if len(g) <= n:
                    return g
                return g.sample(n=n, random_state=seed)
            df = df.groupby("ec", group_keys=False).apply(_take_n)

    # Sequence pairs for clustering/components
    acc2seq = {str(r["accession"]): str(r["sequence"]).upper() for _, r in df.iterrows() if pd.notna(r["sequence"]) }
    pairs = sorted(acc2seq.items())
    # Build EC -> list[acc]
    by_ec_all = group_by_ec(df)

    # Identity cutoffs (allow config override: percentages)
    cfg_thresholds = cfg.get("id_thresholds")
    if isinstance(cfg_thresholds, list) and cfg_thresholds:
        try:
            cuts = [float(x) / (100.0 if max(cfg_thresholds) > 1 else 1.0) for x in cfg_thresholds]
        except Exception:
            cuts = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    else:
        cuts = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    K_folds = int(cfg.get("folds", args.folds))

    for cut in cuts:
        pct = int(round(cut * 100))
        print(f"[id-split] cutoff={pct}% | building identity graph → components…")
        # Compute pairwise edges (fast path: MMseqs2; fallback: Python)
        work = Path(f"data/identity/_work_id{pct}")
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "all.fasta"
        write_fasta(pairs, fasta)
        mmseqs_ok = shutil.which("mmseqs") is not None and identity_definition in ("tool_default",)
        edges: List[Tuple[str, str]]
        if mmseqs_ok:
            print(f"[id-split][mmseqs] easy-search min_id={cut} cov>={float(cfg.get('cluster_coverage', 0.5))} (quiet; logs at {work/'logs/mmseqs_easy_search.log'})")
            edges = mmseqs_pairwise_edges(fasta, work, min_id=cut, min_cov=float(cfg.get("cluster_coverage", 0.5)))
            clustering_method = "mmseqs_easy_search"
            identity_def_used = "tool_default"
        else:
            print("[id-split][warn] MMseqs2 not found; using Python approximate pairwise identity (slow, approximate)")
            edges = python_pairwise_edges(pairs, min_ratio=cut)
            clustering_method = "python_difflib_ratio"
            identity_def_used = "global_pairwise"

        # Connected components define clusters at this threshold
        nodes = [a for a, _ in pairs]
        comps = connected_components(nodes, edges)
        print(f"[id-split] components built: {len(comps)} clusters at {pct}%")
        clu2acc: Dict[str, List[str]] = {}
        for comp in comps:
            rep = comp[0]
            clu2acc[rep] = comp

        # Persist cluster artifacts under results/split-XX and clusters TSV under data/identity
        split_dir = out_root / f"split-{pct}"
        split_dir.mkdir(parents=True, exist_ok=True)
        clusters_json = split_dir / "clusters.json"
        with open(clusters_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in clu2acc.items()}, f, indent=2)
        clusters_tsv = split_dir / "clusters.tsv"
        with open(clusters_tsv, "w", encoding="utf-8") as f:
            for cid, accs in sorted(clu2acc.items()):
                for a in accs:
                    f.write(f"{a}\t{cid}\n")
        out_cluster_tsv = Path(f"data/identity/clusters_id{pct}.tsv")
        out_cluster_tsv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clusters_tsv, out_cluster_tsv)

        # Deterministic stratified cluster→fold assignment
        def ec_top(ec: str) -> str:
            return ec.split(".")[0] if "." in ec else ec
        label_fn = (ec_top if stratify_by == "EC_top" else (lambda x: x)) if stratify_by != "none" else (lambda x: "_ALL_")
        # Build acc -> set(labels)
        acc2labels: Dict[str, Set[str]] = defaultdict(set)
        for ec, accs in by_ec_all.items():
            lbl = label_fn(ec)
            for a in accs:
                acc2labels[a].add(lbl)
        cluster_ids = sorted(clu2acc.keys())
        cl_label: Dict[str, Dict[str, int]] = {}
        cl_size: Dict[str, int] = {}
        total_labels: DefaultDict[str, int] = defaultdict(int)
        for cid in cluster_ids:
            cnt: DefaultDict[str, int] = defaultdict(int)
            accs = clu2acc[cid]
            cl_size[cid] = len(accs)
            for a in accs:
                for lbl in acc2labels.get(a, set()):
                    cnt[lbl] += 1
                    total_labels[lbl] += 1
            cl_label[cid] = dict(cnt)
        F = max(1, K_folds)
        folds: List[List[str]] = [[] for _ in range(F)]
        fold_label: List[DefaultDict[str, int]] = [defaultdict(int) for _ in range(F)]
        fold_size: List[int] = [0 for _ in range(F)]
        target_total = sum(cl_size[cid] for cid in cluster_ids)
        target_labels = {k: v / F for k, v in total_labels.items()}
        target_size = target_total / F
        order = sorted(cluster_ids, key=lambda x: (-cl_size[x], x))
        def score_fold(fi: int, cid: str) -> float:
            sse = 0.0
            keys = set(fold_label[fi].keys()) | set(cl_label[cid].keys())
            for k in keys:
                cur = fold_label[fi].get(k, 0)
                add = cl_label[cid].get(k, 0)
                tgt = target_labels.get(k, 0.0)
                sse += (cur + add - tgt) ** 2
            sz = fold_size[fi] + cl_size[cid]
            sse += (sz - target_size) ** 2
            return sse
        for cid in order:
            scores = [(score_fold(fi, cid), fi) for fi in range(F)]
            scores.sort()
            best_fi = scores[0][1]
            folds[best_fi].append(cid)
            fold_size[best_fi] += cl_size[cid]
            for k, v in cl_label[cid].items():
                fold_label[best_fi][k] += v

        folds_json = split_dir / "folds.json"
        folds_payload = {
            "folds": {str(i+1): folds[i] for i in range(F)},
            "fold_sizes": fold_size,
            "fold_label_totals": [{k: int(v) for k, v in d.items()} for d in fold_label],
            "label_target_per_fold": {k: v for k, v in target_labels.items()},
            "identity_threshold_pct": pct,
            "identity_definition": identity_def_used,
            "clustering_method": clustering_method,
            "random_seed": seed,
            "stratify_by": stratify_by,
            "notes": [],
        }
        with open(folds_json, "w", encoding="utf-8") as f:
            json.dump(folds_payload, f, indent=2)

        # Emit per-fold splits
        for fi in range(F):
            test_clusters = set(folds[fi])
            trainval_clusters = set(cluster_ids) - test_clusters
            # Build per-ec accessions restricted to cluster sets
            def filter_by_clusters(by_ec: Dict[str, List[str]], keep: set[str]) -> Dict[str, List[str]]:
                keep_acc = {a for c in keep for a in clu2acc[c]}
                out: Dict[str, List[str]] = {}
                for ec, accs in by_ec.items():
                    sel = [a for a in accs if a in keep_acc]
                    if sel:
                        out[ec] = sel
                return out

            trva = filter_by_clusters(by_ec_all, trainval_clusters)
            te   = filter_by_clusters(by_ec_all, test_clusters)

            # Keep only classes with at least min_per_class in train pool
            trainable_classes = [ec for ec, accs in trva.items() if len(accs) >= min_per_class]
            train_classes, val_classes = split_classes(trainable_classes, seed=seed, frac_val=0.1)

            def to_items(by_ec: Dict[str, List[str]], cls: List[str]) -> List[dict]:
                return [dict(ec=ec, accessions=sorted(by_ec.get(ec, []))) for ec in sorted(cls)]

            # Test includes all classes present in test clusters (no min filter)
            test_items = [dict(ec=ec, accessions=sorted(accs)) for ec, accs in sorted(te.items())]

            out_dir = split_dir / f"fold-{fi+1}"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(out_dir / "train.jsonl", to_items(trva, train_classes))
            write_jsonl(out_dir / "val.jsonl", to_items(trva, val_classes))
            write_jsonl(out_dir / "test.jsonl", test_items)
            print(f"[id-split] cutoff={pct}% fold={fi+1}: train={len(train_classes)} val={len(val_classes)} test_classes={len(te)}")

        # Record cutoff-specific config snapshot
        used_cfg = dict(cfg)
        used_cfg.update({
            "id_threshold": pct,
            "folds": F,
            "identity_definition": identity_def_used,
            "clustering_method": clustering_method,
            "random_seed": seed,
            "stratify_by": stratify_by,
        })
        with open(split_dir / "config.used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(used_cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
