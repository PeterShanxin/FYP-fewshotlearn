#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_utils import (
    build_model,
    infer_input_dim,
    load_checkpoint,
    load_cfg,
    pick_device,
)
from src.prototype_bank import build_prototypes, save_prototypes

import json as _json
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build global prototype bank from train split.")
    ap.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", required=True, type=Path, help="Output NPZ path for prototypes")
    ap.add_argument("--subprototypes", type=int, default=None, help="Override subprototypes per EC")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)

    embeddings_path = Path(paths["embeddings"])
    train_split = Path(paths["splits_dir"]) / "train.jsonl"
    ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
    subprotos = args.subprototypes
    if subprotos is None:
        subprotos = int(cfg.get("eval", {}).get("subprototypes_per_ec", 1))

    # Optional source filter for prototypes
    proto_src = str(cfg.get("prototypes_source", "mixed") or "mixed").strip().lower()
    filtered_split = train_split
    if proto_src in {"swissprot", "trembl"}:
        # Build accession->source map from joined TSV when available
        joined_tsv = Path(paths.get("joined_tsv", ""))
        acc2src: Dict[str, str] = {}
        if joined_tsv.exists():
            try:
                df = pd.read_csv(joined_tsv, sep="\t", usecols=["accession", "source"])  # type: ignore[arg-type]
                df.columns = [c.lower() for c in df.columns]
                if "source" in df.columns:
                    for acc, src in zip(df["accession"].astype(str), df["source"].astype(str)):
                        acc2src[acc] = src
            except Exception:
                acc2src = {}
        want_src = "SwissProt" if proto_src == "swissprot" else "TrEMBL"
        # Filter the train split to only accessions with the desired source
        tmp_dir = Path("artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_split = tmp_dir / "train.prototypes.filtered.jsonl"
        kept_any = False
        with open(train_split, "r", encoding="utf-8") as fin, open(tmp_split, "w", encoding="utf-8") as fout:
            for line in fin:
                obj = _json.loads(line)
                accs: List[str] = list(obj.get("accessions", []))
                # Default unknown source to SwissProt to avoid dropping in legacy setups
                def _src_of(a: str) -> str:
                    return acc2src.get(a, "SwissProt")
                new_accs = [a for a in accs if _src_of(a) == want_src]
                if not new_accs:
                    continue
                obj["accessions"] = new_accs
                fout.write(_json.dumps(obj) + "\n")
                kept_any = True
        if kept_any:
            filtered_split = tmp_split
        else:
            # If filtering removed everything, fall back to the original split
            filtered_split = train_split

    input_dim = infer_input_dim(embeddings_path)
    model = build_model(cfg, input_dim, device)
    load_checkpoint(model, ckpt_path, device)

    prototypes, train_counts = build_prototypes(
        filtered_split,
        embeddings_path,
        model,
        device=device,
        subprototypes_per_ec=max(1, int(subprotos)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_prototypes(args.out, prototypes, train_counts=train_counts)
    summary = {
        "classes": len(prototypes),
        "total_prototypes": int(sum(arr.shape[0] for arr in prototypes.values())),
        "subprototypes_per_ec": subprotos,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2))
    print(f"[build_prototypes] wrote → {args.out}")


if __name__ == "__main__":
    main()
