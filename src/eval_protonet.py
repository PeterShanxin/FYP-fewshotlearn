"""Episodic evaluation on meta-test for K in K_eval.

Writes results/metrics.json with per-K accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
    cascade_cfg: Dict[str, Any] | None = None,
    detector_cfg: Dict[str, Any] | None = None,
    tau_multi: float = 0.6,
) -> Dict[str, float]:
    ys, yh = [], []  # single-label
    multi_label_mode = False
    hits = 0
    total = 0
    # For multi-label thresholded metrics
    y_true_ml: List[np.ndarray] = []
    y_pred_ml: List[np.ndarray] = []
    # Track detector routing decisions when cascade is enabled
    route_flags_all: List[int] = []  # 1 if routed to multi-branch, else 0
    cascade_cfg = cascade_cfg or {}
    detector_cfg = detector_cfg or {}
    cascade_enabled = bool(cascade_cfg.get("enabled", False))
    detector_enabled = bool(detector_cfg.get("enabled", False)) and model.detector_enabled
    det_thresh = float(detector_cfg.get("thresh", 0.5))
    tau_multi = float(tau_multi)
    multi_branch = 0
    single_branch = 0
    with torch.no_grad():
        for _ in trange(
            episodes,
            desc="[eval] episodes",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            sx, sy, qx, qy, _classes = sampler.sample_episode(M, K, Q)
            logits, _ = model(sx, sy, qx, None)
            sm = torch.softmax(logits, dim=-1)
            pred = sm.argmax(dim=-1)
            if qy.dim() == 2:
                multi_label_mode = True
                idx = pred.view(-1, 1)
                take = torch.gather(qy, 1, idx).squeeze(1)
                hits += int((take > 0.5).sum().item())
                total += int(qy.shape[0])
                # Thresholded multi-label predictions for PR/F1
                sg = torch.sigmoid(logits)
                if cascade_enabled and detector_enabled:
                    det_logits = model.detect_multi(logits)
                    if det_logits is None:
                        raise RuntimeError("Detector enabled in config but missing on model.")
                    det_scores = torch.sigmoid(det_logits).squeeze(-1)
                    preds = []
                    flags = []
                    for i in range(logits.shape[0]):
                        if det_scores[i] >= det_thresh:
                            row = (sg[i] >= tau_multi).float()
                            if row.sum().item() == 0:
                                top_idx = pred[i].item()
                                row = torch.zeros_like(sg[i])
                                row[top_idx] = 1.0
                            flags.append(1)
                        else:
                            top_idx = pred[i].item()
                            row = torch.zeros_like(sg[i])
                            row[top_idx] = 1.0
                            flags.append(0)
                        preds.append(row)
                    stacked = torch.stack(preds, dim=0)
                    y_pred_ml.append(stacked.cpu().numpy())
                    sum_flags = int(sum(flags))
                    multi_branch += sum_flags
                    single_branch += len(flags) - sum_flags
                    route_flags_all.extend(flags)
                else:
                    y_pred_ml.append((sg >= 0.5).float().cpu().numpy())
                y_true_ml.append(qy.float().cpu().numpy())
            else:
                ys.extend(qy.cpu().numpy().tolist())
                yh.extend(pred.cpu().numpy().tolist())
    if multi_label_mode:
        total_branch = multi_branch + single_branch
        if cascade_enabled and detector_enabled and total_branch > 0:
            pct_multi = 100.0 * multi_branch / total_branch
            print(
                f"[eval][gate] pct_multi={pct_multi:.1f}% ({multi_branch}/{total_branch}) | "
                f"det_thresh={det_thresh:.2f} tau_multi={tau_multi:.2f}"
            )
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
        # Set-based metrics and multi-EC detection quality
        metrics: Dict[str, float] = {
            "acc_top1_hit": acc,
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
        }
        if Yt.size > 0:
            # Cardinalities
            card_true = Yt.sum(axis=1)
            card_pred = Yp.sum(axis=1)
            true_is_multi = card_true > 1.0
            # If routing flags exist (cascade enabled), use them to define predicted-multi; else infer from card_pred
            if cascade_enabled and detector_enabled and len(route_flags_all) == Yt.shape[0]:
                pred_is_multi = np.array(route_flags_all, dtype=bool)
            else:
                pred_is_multi = card_pred > 1.0
            # Multi-EC detection metrics
            tp = float(np.logical_and(true_is_multi, pred_is_multi).sum())
            fp = float(np.logical_and(~true_is_multi, pred_is_multi).sum())
            fn = float(np.logical_and(true_is_multi, ~pred_is_multi).sum())
            det_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            det_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0
            metrics.update(
                {
                    "det_true_multi_rate": float(true_is_multi.mean()),
                    "det_pred_multi_rate": float(np.mean(pred_is_multi.astype(float))),
                    "det_precision": float(det_prec),
                    "det_recall": float(det_rec),
                    "det_f1": float(det_f1),
                }
            )
            # Subset accuracy (exact set) and Jaccard
            inter = np.logical_and(Yt > 0.5, Yp > 0.5).sum(axis=1).astype(float)
            union = np.logical_or(Yt > 0.5, Yp > 0.5).sum(axis=1).astype(float)
            jacc = np.divide(inter, np.maximum(union, 1.0))
            subset_acc = (union == inter) & (union > 0)
            metrics.update(
                {
                    "subset_acc_overall": float(subset_acc.mean() if subset_acc.size > 0 else 0.0),
                    "jaccard_overall": float(jacc.mean() if jacc.size > 0 else 0.0),
                }
            )
            # Multi-only variants
            if true_is_multi.any():
                mask = true_is_multi
                metrics.update(
                    {
                        "subset_acc_multi": float(subset_acc[mask].mean()),
                        "jaccard_multi": float(jacc[mask].mean()),
                        "card_mae_multi": float(np.abs(card_pred[mask] - card_true[mask]).mean()),
                        "mean_true_card_multi": float(card_true[mask].mean()),
                        "mean_pred_card_multi": float(card_pred[mask].mean()),
                        "underpred_ratio_multi": float(((card_pred[mask] < card_true[mask]).astype(float)).mean()),
                        "overpred_ratio_multi": float(((card_pred[mask] > card_true[mask]).astype(float)).mean()),
                    }
                )
            # Overall cardinality error
            metrics.update(
                {
                    "card_mae_overall": float(np.abs(card_pred - card_true).mean()),
                    "mean_true_card_overall": float(card_true.mean()),
                    "mean_pred_card_overall": float(card_pred.mean()),
                }
            )
        return metrics
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
    detector_cfg = cfg.get("detector", {}) or {}
    cascade_cfg = cfg.get("cascade", {}) or {}
    tau_multi = float(cfg.get("tau_multi", 0.6))
    detector_enabled = bool(detector_cfg.get("enabled", False))
    detector_hidden = int(detector_cfg.get("hidden_dim", 32))
    pcfg = ProtoConfig(
        input_dim=test_sampler.dim,
        projection_dim=int(cfg.get("projection_dim", 256)),
        temperature=float(cfg.get("temperature", 10.0)),
        detector_enabled=detector_enabled,
        detector_hidden=detector_hidden,
    )
    model = ProtoNet(pcfg).to(device)
    if requested_gpus > 1 and device.type == "cuda":
        print("[eval] Note: evaluation runs on a single GPU; multi-GPU is used for embeddings only.")

    # Load checkpoint
    ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=False)  # type: ignore[index]
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
        metrics = eval_for_K(
            model,
            test_sampler,
            M,
            int(K),
            Q,
            n_eval,
            show_progress=show_progress,
            cascade_cfg=cascade_cfg,
            detector_cfg=detector_cfg,
            tau_multi=tau_multi,
        )
        results[f"K={K}"] = metrics
        print(f"[eval] {metrics}")

    out_path = Path(paths["outputs"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
