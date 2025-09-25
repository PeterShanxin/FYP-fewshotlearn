#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    input_dim = infer_input_dim(embeddings_path)
    model = build_model(cfg, input_dim, device)
    load_checkpoint(model, ckpt_path, device)

    prototypes, train_counts = build_prototypes(
        train_split,
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
