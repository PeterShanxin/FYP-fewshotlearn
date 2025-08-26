"""Build simple, cluster-free meta-train/val/test splits by EC classes.

- Reads the joined TSV produced by scripts/fetch_uniprot_ec.sh
- Keeps only single-EC rows for this minimal trial (drop multi-label)
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


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    return Config(
        joined_tsv=Path(paths["joined_tsv"]),
        splits_dir=Path(paths["splits_dir"]),
        min_per_class=int(cfg["min_sequences_per_class_for_train"]),
        seed=int(cfg.get("random_seed", 42)),
    )


def filter_single_ec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalise column names
    df.columns = [c.lower() for c in df.columns]
    df = df[("ec" in df.columns) & (df["ec"].notna())]
    df = df[(df["ec"].notna()) & (df["ec"].astype(str).str.len() > 0)]
    # keep only single-EC rows (no ';')
    mask_single = ~df["ec"].astype(str).str.contains(";")
    df = df[mask_single]
    # strip spaces
    df["ec"] = df["ec"].astype(str).str.strip()
    return df


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
    n_train = int(0.8 * n)
    n_val = max(1, int(0.1 * n))
    n_test = n - n_train - n_val
    train = classes[:n_train]
    val = classes[n_train : n_train + n_val]
    test = classes[n_train + n_val :]
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
            ),
            indent=2,
        )
    )

    if not cfg.joined_tsv.exists():
        raise FileNotFoundError(
            f"Joined TSV not found: {cfg.joined_tsv}. Run scripts/fetch_uniprot_ec.sh first."
        )

    df = pd.read_csv(cfg.joined_tsv, sep="\t")
    df = filter_single_ec(df)

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
