"""Build simple, cluster-aware optional meta-train/val/test splits by EC classes.

- Reads the joined TSV produced by scripts/fetch_uniprot_ec.sh
- By default, used to keep only single-EC rows (drop multi-label)
- If allow_multi_ec=true, expands multi-EC rows so each accession appears under all its ECs
- Groups by EC; classes with >= min_sequences_per_class_for_train form the base pool
- Splits base pool classes 80/10/10 into meta-train/val/test (by classes, not by sequences)
- Adds all underfilled classes (< min) into meta-test pool (by class)
- Writes JSONL files with one object per EC class: {"ec": str, "accessions": [..]}
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class Config:
    joined_tsv: Path
    splits_dir: Path
    min_per_class: int
    seed: int
    allow_multi_ec: bool = False
    limit_classes: int | None = None
    limit_per_class: int | None = None
    split_source: str | None = None  # Optional: filter rows by 'source' column when present


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    return Config(
        joined_tsv=Path(paths["joined_tsv"]),
        splits_dir=Path(paths["splits_dir"]),
        min_per_class=int(cfg["min_sequences_per_class_for_train"]),
        seed=int(cfg.get("random_seed", 42)),
        allow_multi_ec=bool(cfg.get("allow_multi_ec", False)),
        limit_classes=(int(cfg.get("limit_classes")) if cfg.get("limit_classes") is not None else None),
        limit_per_class=(int(cfg.get("limit_per_class")) if cfg.get("limit_per_class") is not None else None),
        split_source=(str(cfg.get("split_source")).strip() if cfg.get("split_source") not in (None, "", "null") else None),
    )


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "ec" not in df.columns:
        raise KeyError("'ec' column not found")
    df = df[(df["ec"].notna()) & (df["ec"].astype(str).str.len() > 0)]
    df["ec"] = df["ec"].astype(str).str.strip()
    df["accession"] = df["accession"].astype(str)
    return df


def apply_split_source_filter(df: pd.DataFrame, split_source: str | None) -> pd.DataFrame:
    """Filter to a specific source if requested.

    If 'source' column is present in df and split_source is provided, keep only
    those rows. If the column is missing but split_source equals 'SwissProt', we
    assume all rows are Swiss‑Prot and return df unchanged.
    """
    if split_source is None:
        return df
    cols = [c.lower() for c in df.columns]
    if "source" not in cols:
        # Treat as Swiss‑Prot only file
        return df
    mask = df["source"].astype(str) == str(split_source)
    return df[mask]


def filter_or_expand_ec(df: pd.DataFrame, allow_multi_ec: bool) -> pd.DataFrame:
    """Either keep only single-EC rows or expand multi-EC rows into multiple rows.

    When allow_multi_ec=True, an accession with "ec" like "1.1.1.1; 3.5.4.4" will
    appear twice (once per EC) in the output. This enables multi-label training
    while preserving a simple class→accessions mapping in the JSONL files.
    """
    df = normalize_df(df)
    if not allow_multi_ec:
        mask_single = ~df["ec"].astype(str).str.contains(";")
        return df[mask_single]
    cols = df.columns
    df = df.assign(ec=df["ec"].str.split(";")).explode("ec")
    df["ec"] = df["ec"].str.strip()
    df = df[df["ec"] != ""]
    return df[cols]


def group_by_ec(df: pd.DataFrame) -> Dict[str, List[str]]:
    grp: Dict[str, List[str]] = {}
    for ec, sub in df.groupby("ec"):
        accs = sub["accession"].astype(str).tolist()
        grp[ec] = accs
    return grp


def split_classes(classes: List[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    rnd = random.Random(seed)
    classes = classes[:]
    rnd.shuffle(classes)
    n = len(classes)
    if n <= 1:
        # all to train to keep pipeline running; val/test empty
        return classes, [], []
    if n == 2:
        # ensure we have at least train and val for early stopping logic
        return classes[:1], classes[1:], []

    # For n >= 3, aim for ~80/10/10 but guarantee at least 1 in each split
    n_train = max(1, int(round(0.8 * n)))
    n_val = max(1, int(round(0.1 * n)))
    n_test = max(1, n - n_train - n_val)

    # If we over-allocated due to rounding, trim train first, then val/test
    while n_train + n_val + n_test > n and n_train > 1:
        n_train -= 1
    while n_train + n_val + n_test > n and n_val > 1:
        n_val -= 1
    while n_train + n_val + n_test > n and n_test > 1:
        n_test -= 1

    # Final safeguard
    if n_train + n_val + n_test > n:
        n_train = max(1, n - n_val - n_test)

    train = classes[:n_train]
    val = classes[n_train : n_train + n_val]
    test = classes[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("[prepare_split] config:")
    print(
        json.dumps(
            dict(
                joined_tsv=str(cfg.joined_tsv),
                splits_dir=str(cfg.splits_dir),
                min_per_class=cfg.min_per_class,
                seed=cfg.seed,
                allow_multi_ec=cfg.allow_multi_ec,
                limit_classes=cfg.limit_classes,
                limit_per_class=cfg.limit_per_class,
            ),
            indent=2,
        )
    )

    if not cfg.joined_tsv.exists():
        raise FileNotFoundError(
            f"Joined TSV not found: {cfg.joined_tsv}. Run scripts/fetch_uniprot_ec.sh first."
        )

    df = pd.read_csv(cfg.joined_tsv, sep="\t")
    # Optional source filter (when using a merged file with 'source' column)
    df = apply_split_source_filter(df, cfg.split_source)
    df = filter_or_expand_ec(df, allow_multi_ec=cfg.allow_multi_ec)

    # Optional downsampling for smoke tests: keep only top-N classes and at most
    # K sequences per class. This reduces embedding/training time drastically
    # while still exercising the full pipeline.
    if cfg.limit_classes is not None or cfg.limit_per_class is not None:
        # Count sequences per EC and sort by count desc for determinism
        counts = (
            df.groupby("ec")["accession"].count().sort_values(ascending=False)
        )
        if cfg.limit_classes is not None:
            keep_classes = set(counts.head(max(1, int(cfg.limit_classes))).index.tolist())
            df = df[df["ec"].isin(keep_classes)]
        # Within each class, sample up to limit_per_class sequences (stable if not sampling)
        if cfg.limit_per_class is not None:
            def _take_n(g: pd.DataFrame) -> pd.DataFrame:
                n = int(cfg.limit_per_class or 0)
                if len(g) <= n:
                    return g
                return g.sample(n=n, random_state=cfg.seed)
            df = df.groupby("ec", group_keys=False).apply(_take_n)

    by_ec = group_by_ec(df)
    base_classes = [ec for ec, accs in by_ec.items() if len(accs) >= cfg.min_per_class]
    holdout_classes = [ec for ec, accs in by_ec.items() if len(accs) < cfg.min_per_class]

    tr_c, va_c, te_c = split_classes(base_classes, cfg.seed)

    def classes_to_items(eclist: List[str]) -> List[dict]:
        return [dict(ec=ec, accessions=by_ec[ec]) for ec in sorted(eclist)]

    # meta-test includes the test split from base classes + ALL holdout classes
    test_all = te_c + holdout_classes

    write_jsonl(cfg.splits_dir / "train.jsonl", classes_to_items(tr_c))
    write_jsonl(cfg.splits_dir / "val.jsonl", classes_to_items(va_c))
    write_jsonl(cfg.splits_dir / "test.jsonl", classes_to_items(test_all))

    print(
        f"[prepare_split] classes: base={len(base_classes)} (train={len(tr_c)}, val={len(va_c)}, test={len(te_c)}), "
        f"holdout(<{cfg.min_per_class})={len(holdout_classes)}"
    )
    print(f"[prepare_split] wrote splits to {cfg.splits_dir}")


if __name__ == "__main__":
    main()
