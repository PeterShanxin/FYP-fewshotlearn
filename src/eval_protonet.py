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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import trange

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


def eval_for_K(
    model: ProtoNet,
    sampler: EpisodeSampler,
    M: int,
    K: int,
    Q: int,
    episodes: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    ys, yh = [], []  # single-label
    multi_label_mode = False
    hits = 0
    total = 0
    # For multi-label thresholded metrics
    y_true_ml: List[np.ndarray] = []
    y_pred_ml: List[np.ndarray] = []
    with torch.no_grad():
        for _ in trange(
            episodes,
            desc="[eval] episodes",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            sx, sy, qx, qy, _classes = sampler.sample_episode(M, K, Q)
            logits, _ = model(sx, sy, qx, None)
            pred = logits.argmax(dim=-1)
            if qy.dim() == 2:
                multi_label_mode = True
                idx = pred.view(-1, 1)
                take = torch.gather(qy, 1, idx).squeeze(1)
                hits += int((take > 0.5).sum().item())
                total += int(qy.shape[0])
                # Thresholded multi-label predictions for PR/F1
                probs = torch.sigmoid(logits)
                y_pred_ml.append((probs >= 0.5).float().cpu().numpy())
                y_true_ml.append(qy.float().cpu().numpy())
            else:
                ys.extend(qy.cpu().numpy().tolist())
                yh.extend(pred.cpu().numpy().tolist())
    if multi_label_mode:
        acc = float(hits / max(total, 1))
        Yt = np.vstack(y_true_ml) if y_true_ml else np.zeros((0, 0), dtype=np.float32)
        Yp = np.vstack(y_pred_ml) if y_pred_ml else np.zeros((0, 0), dtype=np.float32)
        # Micro/macro precision/recall/F1 with zero_division=0 for stability
        micro_p = precision_score(Yt.reshape(-1), Yp.reshape(-1), zero_division=0)
        micro_r = recall_score(Yt.reshape(-1), Yp.reshape(-1), zero_division=0)
        micro_f1 = f1_score(Yt.reshape(-1), Yp.reshape(-1), zero_division=0)
        # Macro over labels (average per class)
        macro_p = precision_score(Yt, Yp, average="macro", zero_division=0)
        macro_r = recall_score(Yt, Yp, average="macro", zero_division=0)
        macro_f1 = f1_score(Yt, Yp, average="macro", zero_division=0)
        return {
            "acc_top1_hit": acc,
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
        }
    # Single-label metrics
    acc = accuracy_score(ys, yh)
    p = precision_score(ys, yh, average="macro", zero_division=0)
    r = recall_score(ys, yh, average="macro", zero_division=0)
    f1 = f1_score(ys, yh, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)
    requested_gpus = int(cfg.get("gpus", 1))

    # Sampler and model init
    multi_label = bool(cfg.get("multi_label", False))
    clusters_tsv = Path(paths.get("clusters_tsv", "")) if paths.get("clusters_tsv") else None
    disjoint = bool(cfg.get("identity_disjoint", False))
    test_sampler = EpisodeSampler(
        Path(paths["embeddings"]), Path(paths["splits_dir"]) / "test.jsonl", device,
        seed=cfg.get("random_seed", 42) + 2, multi_label=multi_label,
        clusters_tsv=clusters_tsv, disjoint_support_query=disjoint,
    )
    pcfg = ProtoConfig(input_dim=test_sampler.dim, projection_dim=int(cfg.get("projection_dim", 256)), temperature=float(cfg.get("temperature", 10.0)))
    model = ProtoNet(pcfg).to(device)
    if requested_gpus > 1 and device.type == "cuda":
        print("[eval] Note: evaluation runs on a single GPU; multi-GPU is used for embeddings only.")

    # Load checkpoint
    ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])  # type: ignore[index]
    model.eval()

    M = int(cfg["episode"]["M"])
    Q = int(cfg["episode"]["Q"])
    # Allow separate eval episodes; fallback to 'val' if unspecified
    episodes_cfg = cfg.get("episodes", {})
    n_eval = int(episodes_cfg.get("eval", max(100, int(episodes_cfg.get("val", 200)))))

    results: Dict[str, Dict[str, float]] = {}
    show_progress = bool(cfg.get("progress", True))
    for K in cfg["episode"]["K_eval"]:
        print(f"[eval] M={M} K={K} Q={Q} episodes={n_eval}")
        metrics = eval_for_K(model, test_sampler, M, int(K), Q, n_eval, show_progress=show_progress)
        results[f"K={K}"] = metrics
        print(f"[eval] {metrics}")

    out_path = Path(paths["outputs"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
