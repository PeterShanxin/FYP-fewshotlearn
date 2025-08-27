"""Episodic evaluation on meta-test for K in K_eval.

Writes results/metrics.json with per-K accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from .episodic_sampler import EpisodeSampler
from .protonet import ProtoConfig, ProtoNet


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(cfg: dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def eval_for_K(model: ProtoNet, sampler: EpisodeSampler, M: int, K: int, Q: int, episodes: int) -> Dict[str, float]:
    ys, yh = [], []
    with torch.no_grad():
        for _ in range(episodes):
            sx, sy, qx, qy = sampler.sample_episode(M, K, Q)
            pred = model.predict(sx, sy, qx)
            ys.extend(qy.cpu().numpy().tolist())
            yh.extend(pred.cpu().numpy().tolist())
    acc = accuracy_score(ys, yh)
    f1 = f1_score(ys, yh, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_f1": float(f1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)

    # Sampler and model init
    test_sampler = EpisodeSampler(paths["embeddings"], Path(paths["splits_dir"]) / "test.jsonl", device, seed=cfg.get("random_seed", 42) + 2)
    pcfg = ProtoConfig(input_dim=test_sampler.dim, projection_dim=int(cfg.get("projection_dim", 256)), temperature=float(cfg.get("temperature", 10.0)))
    model = ProtoNet(pcfg).to(device)

    # Load checkpoint
    ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])  # type: ignore[index]
    model.eval()

    M = int(cfg["episode"]["M"])
    Q = int(cfg["episode"]["Q"])
    n_eval = max(100, int(cfg["episodes"]["val"]))  # run a bit more for test

    results: Dict[str, Dict[str, float]] = {}
    for K in cfg["episode"]["K_eval"]:
        print(f"[eval] M={M} K={K} Q={Q} episodes={n_eval}")
        metrics = eval_for_K(model, test_sampler, M, int(K), Q, n_eval)
        results[f"K={K}"] = metrics
        print(f"[eval] {metrics}")

    out_path = Path(paths["outputs"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
