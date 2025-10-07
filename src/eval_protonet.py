"""Episodic evaluation on meta-test for K in K_eval.

Writes results/metrics.json with per-K accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import trange

from .episodic_sampler import EpisodeSampler
from .eval_global import run_global_evaluation
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
    tau_multi: float = 0.6,
    per_ec_thresholds: Dict[str, float] | None = None,
) -> Dict[str, float]:
    ys, yh = [], []  # single-label
    multi_label_mode = False
    hits = 0
    total = 0
    # For multi-label thresholded metrics
    y_true_ml: List[np.ndarray] = []
    y_pred_ml: List[np.ndarray] = []
    tau_multi = float(tau_multi)
    thresholds_map = per_ec_thresholds or {}
    coverage = sampler.class_coverage(K, Q)
    print(
        "[eval][coverage] classes with ≥K+Q: {full}/{total} | tail (≥K,<K+Q): {tail} | excluded: {excluded}".format(
            full=coverage["eligible_full"],
            total=coverage["total_classes"],
            tail=coverage["eligible_support_only"],
            excluded=coverage["excluded"],
        )
    )
    # If nothing can be sampled under current constraints (e.g., identity-disjoint
    # with single-cluster classes in tiny smoke runs), return zeros instead of failing.
    if (coverage["eligible_full"] + coverage["eligible_support_only"]) <= 0:
        print("[eval][coverage] No eligible classes; returning zeros.")
        return {
            "acc_top1_hit": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }
    if coverage["eligible_full"] < M:
        print(
            "[eval][coverage] WARNING: only {full} classes can fill support+query without fallback; requesting M={M}".format(
                full=coverage["eligible_full"],
                M=M,
            )
        )
    with torch.no_grad():
        for _ in trange(
            episodes,
            desc=f"[eval] episodes (M={M}, K={K}, Q={Q})",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            try:
                sx, sy, qx, qy, classes = sampler.sample_episode(M, K, Q)
            except RuntimeError as exc:
                print(f"[eval][warn] skipping episode due to sampling error: {exc}")
                continue
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
                tau_vec = torch.full((sg.shape[-1],), tau_multi, device=sg.device, dtype=sg.dtype)
                if thresholds_map:
                    for j, ec in enumerate(classes):
                        if ec in thresholds_map:
                            tau_vec[j] = float(thresholds_map[ec])
                thresh_matrix = tau_vec.unsqueeze(0).expand_as(sg)
                preds = (sg >= thresh_matrix).float()
                zero_mask = preds.sum(dim=1) == 0
                if zero_mask.any():
                    rows = torch.where(zero_mask)[0]
                    preds[rows] = 0.0
                    top_indices = pred[rows]
                    preds[rows, top_indices] = 1.0
                y_pred_ml.append(preds.cpu().numpy())
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


def print_metrics_table(prefix: str, metrics: Dict[str, float]) -> None:
    if not metrics:
        print(f"{prefix} (no metrics)")
        return
    width = max(len(key) for key in metrics.keys())
    print(prefix)
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, numbers.Real):
            formatted = f"{float(value):.4f}"
        else:
            formatted = str(value)
        print(f"  {key:<{width}} : {formatted}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--mode", choices=["episodic", "global_support"], default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)
    requested_gpus = int(cfg.get("gpus", 1))

    eval_cfg = cfg.get("eval", {}) or {}
    mode = args.mode or str(eval_cfg.get("mode", "episodic"))

    if mode == "global_support":
        prototypes_path = Path(eval_cfg.get("prototypes_path", "artifacts/prototypes.npz"))
        thresholds_path_raw = eval_cfg.get("per_ec_thresholds_path")
        thresholds_path = Path(thresholds_path_raw) if thresholds_path_raw else None
        calibration_path_raw = eval_cfg.get("calibration_path", "artifacts/calibration.json")
        calibration_path = Path(calibration_path_raw) if calibration_path_raw else None
        metrics = run_global_evaluation(
            config_path=Path(args.config),
            prototypes_path=prototypes_path,
            split=eval_cfg.get("split", "test"),
            tau_multi=None,
            temperature=None,
            shortlist_topN=None,
            thresholds_path=thresholds_path,
            calibration_path=calibration_path,
        )
        out_path = Path(paths["outputs"]) / "global_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2))
        print(f"[eval][global] wrote metrics → {out_path}")
        return

    # Episodic evaluation path
    sampler_cfg = cfg.get("sampler", {}) or {}
    multi_label = bool(cfg.get("multi_label", False))
    clusters_tsv = Path(paths.get("clusters_tsv", "")) if paths.get("clusters_tsv") else None
    disjoint = bool(sampler_cfg.get("identity_disjoint", cfg.get("identity_disjoint", False)))
    with_replacement_fallback = bool(sampler_cfg.get("with_replacement_fallback", False))
    fallback_scope = sampler_cfg.get("fallback_scope", "train_only")

    test_sampler = EpisodeSampler(
        Path(paths["embeddings"]),
        Path(paths["splits_dir"]) / "test.jsonl",
        device,
        seed=cfg.get("random_seed", 42) + 2,
        phase="test",
        multi_label=multi_label,
        clusters_tsv=clusters_tsv,
        disjoint_support_query=disjoint,
        with_replacement_fallback=with_replacement_fallback,
        fallback_scope=fallback_scope,
        rare_class_boost="none",
    )
    tau_cfg = eval_cfg.get("tau_multi")
    if tau_cfg is None:
        raise ValueError(
            "Missing 'eval.tau_multi' in configuration. Set a base sigmoid threshold or enable calibration."
        )
    tau_multi = float(tau_cfg)

    # Honor toggles for tuned parameters
    use_calibration = bool(eval_cfg.get("use_calibration", True))
    use_per_ec = bool(eval_cfg.get("use_per_ec_thresholds", True))

    calibration_path_raw = eval_cfg.get("calibration_path")
    if use_calibration and calibration_path_raw:
        calibration_path = Path(calibration_path_raw)
        if calibration_path.exists():
            try:
                with open(calibration_path, "r", encoding="utf-8") as handle:
                    calibration_data = json.load(handle)
            except json.JSONDecodeError as exc:
                print(f"[eval] Warning: failed to parse calibration file {calibration_path}: {exc}")
            else:
                cal_tau = calibration_data.get("tau_multi")
                if cal_tau is not None:
                    tau_multi = float(cal_tau)

    thresholds_path_raw = eval_cfg.get("per_ec_thresholds_path")
    per_ec_thresholds: Dict[str, float] | None = None
    if use_per_ec and thresholds_path_raw:
        thresholds_path = Path(thresholds_path_raw)
        if thresholds_path.exists():
            try:
                with open(thresholds_path, "r", encoding="utf-8") as handle:
                    thresholds_data = json.load(handle)
            except json.JSONDecodeError as exc:
                print(f"[eval] Warning: failed to parse per-EC thresholds {thresholds_path}: {exc}")
            else:
                if isinstance(thresholds_data, dict):
                    per_ec_thresholds = {
                        str(ec): float(value) for ec, value in thresholds_data.items()
                    }
        else:
            print(f"[eval] Warning: per-EC thresholds file not found at {thresholds_path}")

    pcfg = ProtoConfig(
        input_dim=test_sampler.dim,
        projection_dim=int(cfg.get("projection_dim", 256)),
        temperature=float(cfg.get("temperature", 10.0)),
    )
    model = ProtoNet(pcfg).to(device)
    if requested_gpus > 1 and device.type == "cuda":
        print("[eval] Note: evaluation runs on a single GPU; multi-GPU is used for embeddings only.")

    ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=False)  # type: ignore[index]
    model.eval()

    episode_cfg = cfg.get("episode", {}) or {}

    def _resolve_episode_value(primary: str, fallbacks: List[str], default: int) -> int:
        keys = [primary] + fallbacks
        for key in keys:
            if key not in episode_cfg:
                continue
            val = episode_cfg[key]
            if isinstance(val, list):
                if not val:
                    continue
                return int(val[0])
            try:
                return int(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return int(default)

    M = _resolve_episode_value("M_val", ["M"], 10)
    Q = _resolve_episode_value("Q_val", ["Q"], 10)
    K_default = _resolve_episode_value("K_val", ["K"], 1)
    K_eval_cfg = episode_cfg.get("K_eval")
    if isinstance(K_eval_cfg, list) and K_eval_cfg:
        K_values = [int(k) for k in K_eval_cfg]
    else:
        K_values = [K_default]
    # Allow separate eval episodes; fallback to 'val' if unspecified
    episodes_cfg = cfg.get("episodes", {})
    n_eval = int(episodes_cfg.get("eval", max(100, int(episodes_cfg.get("val", 200)))))

    results: Dict[str, Dict[str, float]] = {}
    show_progress = bool(cfg.get("progress", True))
    for K in K_values:
        print(f"[eval] M={M} K={K} Q={Q} episodes={n_eval}")
        metrics = eval_for_K(
            model,
            test_sampler,
            M,
            int(K),
            Q,
            n_eval,
            show_progress=show_progress,
            tau_multi=tau_multi,
            per_ec_thresholds=per_ec_thresholds,
        )
        results[f"K={K}"] = metrics
        print_metrics_table(f"[eval][episodic][K={K}] metrics:", metrics)

    out_path = Path(paths["outputs"]) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
