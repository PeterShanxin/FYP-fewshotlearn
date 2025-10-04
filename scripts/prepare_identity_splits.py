#!/usr/bin/env python3
"""Prepare identity-cluster-based splits across multiple cutoffs and folds.

Enhancements:
- Build clusters as connected components of the ≥T% identity graph to ensure
  no cross-fold identity leakage at the threshold (when tools available).
- Fast path uses MMseqs2 pairwise search with coverage filter; fallback
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


def detect_allocated_cpus(config_threads: int | None = None) -> int:
    """Best-effort detection of CPUs allocated to this process.

    Priority:
    1) explicit override (config_threads)
    2) Linux CPU affinity (cgroups/SLURM cpuset)
    3) common scheduler/env hints
    4) os.cpu_count()
    """
    # 1) explicit override
    if config_threads is not None:
        try:
            v = int(config_threads)
            if v > 0:
                return v
        except Exception:
            pass
    # 2) scheduler/env hints (prefer explicit allocations over affinity if present)
    env_keys = [
        "SLURM_CPUS_PER_TASK",  # SLURM per-task allocation
        "NSLOTS",               # SGE/UGE
        "PBS_NP",               # PBS/Torque
        "NCPUS",                # generic
        "OMP_NUM_THREADS",      # OpenMP hint
    ]
    for k in env_keys:
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            parsed = int(str(v).strip())
            if parsed > 0:
                hw = os.cpu_count() or parsed
                return max(1, min(parsed, hw))
        except Exception:
            continue
    # 3) CPU affinity
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        if n > 0:
            return n
    except Exception:
        pass
    # 4) fallback to logical CPUs
    return max(1, int(os.cpu_count() or 1))


def mmseqs_pairwise_edges(
    fasta: Path,
    workdir: Path,
    min_id: float,
    min_cov: float,
    threads: int | None = None,
) -> List[Tuple[str, str]]:
    """Use MMseqs2 easy-search to enumerate pairs with identity ≥ min_id and coverage ≥ min_cov.

    Returns undirected edges (a,b) with a < b to de-duplicate.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    aln = workdir / "pairs.tsv"
    tmp = workdir / "tmp"
    threads = detect_allocated_cpus(threads)
    threads_str = str(threads)
    # Use paths relative to workdir because we set cwd=workdir.
    log_path = workdir / "logs" / "mmseqs_easy_search.log"
    # Quiet external tool output; tee to log file for debugging
    run_quiet([
        "mmseqs", "easy-search", str(fasta.name), str(fasta.name), str(aln.name), str(tmp.name),
        "--min-seq-id", str(min_id),
        "-c", str(min_cov), "--cov-mode", "1", "--threads", threads_str,
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


# CD-HIT support removed for simplicity; MMseqs2 or Python fallback only.


def mmseqs_cluster_components(
    fasta: Path,
    workdir: Path,
    min_id: float,
    min_cov: float,
    threads: int | None = None,
    mode: str = "linclust",
) -> List[List[str]]:
    """Cluster sequences with MMseqs2 and return per-cluster accession lists."""

    workdir.mkdir(parents=True, exist_ok=True)
    threads = detect_allocated_cpus(threads)
    threads_str = str(threads)
    logs = workdir / "logs"

    db_name = "seqdb"
    clu_name = f"seqclu_{mode}"
    tmp_name = f"tmp_{mode}"
    tsv_name = f"clusters_{mode}.tsv"

    run_quiet(
        ["mmseqs", "createdb", str(fasta.name), db_name],
        cwd=workdir,
        log_file=logs / "mmseqs_createdb.log",
    )

    # mmseqs exits non-zero if the target cluster DB already exists from a
    # interrupted run; proactively clear leftovers before clustering.
    for leftover in workdir.glob(f"{clu_name}*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass

    cluster_cmd = [
        "mmseqs",
        mode,
        db_name,
        clu_name,
        tmp_name,
        "--min-seq-id",
        str(min_id),
        "-c",
        str(min_cov),
        "--cov-mode",
        "1",
        "--threads",
        threads_str,
        "--remove-tmp-files",
        "1",
    ]

    run_quiet(cluster_cmd, cwd=workdir, log_file=logs / f"mmseqs_{mode}.log")
    run_quiet(
        ["mmseqs", "createtsv", db_name, db_name, clu_name, tsv_name],
        cwd=workdir,
        log_file=logs / f"mmseqs_createtsv_{mode}.log",
    )

    clusters: DefaultDict[str, Set[str]] = defaultdict(set)
    tsv_path = workdir / tsv_name
    with open(tsv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            clusters[rep].add(rep)
            clusters[rep].add(member)

    return [sorted(list(members)) for rep, members in sorted(clusters.items(), key=lambda kv: kv[0])]


# CD-HIT support removed for simplicity; MMseqs2 or Python fallback only.

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
    # Optional source filter: keep only Swiss‑Prot (or other) when using merged TSV
    split_source = cfg.get("split_source")
    if split_source is not None and "source" in [c.lower() for c in df.columns]:
        # Normalize column names to lowercase for robust access
        df.columns = [c.lower() for c in df.columns]
        df = df[df["source"].astype(str) == str(split_source)]
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
    nodes = [a for a, _ in pairs]
    # Build EC -> list[acc]
    by_ec_all = group_by_ec(df)

    backend_pref_raw = cfg.get("identity_cluster_backend", "easy_search")
    backend_pref = str(backend_pref_raw).strip().lower() if backend_pref_raw is not None else "easy_search"

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
        # Compute components (fast path: MMseqs2; fallback: Python)
        work = Path(f"data/identity/_work_id{pct}")
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "all.fasta"
        write_fasta(pairs, fasta)
        mmseqs_ok = shutil.which("mmseqs") is not None and identity_definition in ("tool_default",)
        comps: List[List[str]]
        if mmseqs_ok:
            cov = float(cfg.get('cluster_coverage', 0.5))
            # Optional override via config: mmseqs_threads
            cfg_thr = cfg.get("mmseqs_threads")
            try:
                cfg_thr_i = int(cfg_thr) if cfg_thr is not None else None
            except Exception:
                cfg_thr_i = None
            thr_used = detect_allocated_cpus(cfg_thr_i)
            backend = backend_pref if backend_pref in {"linclust", "cluster", "easy_search", "edges", "edge"} else "easy_search"
            if backend in {"linclust", "cluster"}:
                mmseqs_mode = "linclust" if backend == "linclust" else "cluster"
                print(
                    f"[id-split][mmseqs] {mmseqs_mode} min_id={cut} cov>={cov} threads={thr_used} "
                    f"(quiet; logs under {work/'logs'})"
                )
                comps = mmseqs_cluster_components(
                    fasta,
                    work,
                    min_id=cut,
                    min_cov=cov,
                    threads=thr_used,
                    mode=mmseqs_mode,
                )
                clustering_method = f"mmseqs_{mmseqs_mode}"
                identity_def_used = "tool_default"
            else:
                print(
                    f"[id-split][mmseqs] easy-search min_id={cut} cov>={cov} threads={thr_used} "
                    f"(quiet; logs at {work/'logs/mmseqs_easy_search.log'})"
                )
                edges = mmseqs_pairwise_edges(fasta, work, min_id=cut, min_cov=cov, threads=thr_used)
                comps = connected_components(nodes, edges)
                clustering_method = "mmseqs_easy_search"
                identity_def_used = "tool_default"
        else:
            print("[id-split][warn] MMseqs2 not found; using Python approximate pairwise identity (slow, approximate)")
            edges = python_pairwise_edges(pairs, min_ratio=cut)
            comps = connected_components(nodes, edges)
            clustering_method = "python_difflib_ratio"
            identity_def_used = "global_pairwise"
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

        # Seed each fold with one of the largest clusters to avoid empty folds.
        seeded = 0
        seed_total = min(F, len(order))
        for fi in range(seed_total):
            cid = order[fi]
            folds[fi].append(cid)
            fold_size[fi] += cl_size[cid]
            for k, v in cl_label[cid].items():
                fold_label[fi][k] += v
            seeded += 1
        # Skip seeded clusters during the main assignment loop.
        order = order[seeded:]
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
            scores = [
                (score_fold(fi, cid), fold_size[fi], fi)
                for fi in range(F)
            ]
            scores.sort()
            best_fi = scores[0][2]
            folds[best_fi].append(cid)
            fold_size[best_fi] += cl_size[cid]
            for k, v in cl_label[cid].items():
                fold_label[best_fi][k] += v

        folds_json = split_dir / "folds.json"
        notes: List[str] = []
        if F == 1:
            notes.append(
                "folds=1 detected: train/val use all clusters; test reuses the same clusters (no hold-out)."
            )
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
            "notes": notes,
        }
        with open(folds_json, "w", encoding="utf-8") as f:
            json.dump(folds_payload, f, indent=2)

        # Emit per-fold splits
        for fi in range(F):
            test_clusters = set(folds[fi])
            if F == 1:
                trainval_clusters = set(cluster_ids)
            else:
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
