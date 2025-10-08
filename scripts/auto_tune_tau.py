#!/usr/bin/env python
from __future__ import annotations

"""Calibrate global-support thresholds without relaunching the full pipeline.

This wrapper is invoked by run_identity_benchmark.py for each (cutoff, fold)
to generate calibration artifacts in-place. It performs:
  - Optional temperature calibration (opt_temp: none|bce|brier)
  - Grid search of metric vs tau over the requested range
  - Writes calibration.json under the current fold's outputs directory
  - Optionally emits simple plots under a provided plots dir

It intentionally does not call scripts/run_all.sh to avoid recursive relaunches.
"""

import argparse
import json
import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import sys

# Ensure repository root is on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import evaluator directly to avoid re-running the pipeline
from src.eval_global import GlobalSupportEvaluator
from src.model_utils import load_cfg as load_yaml_cfg, pick_device


TMP_DIR = Path(".tmp_configs")


def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_cfg(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _resolve_calibration_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (eval_cfg, cal_cfg) merged with CLI overrides."""
    eval_cfg = dict(cfg.get("eval") or {})
    cal_cfg = dict(eval_cfg.get("calibration") or {})

    if args.split:
        cal_cfg["split"] = args.split

    existing_range = cal_cfg.get("tau_range") or [0.2, 0.6, 0.02]
    tau_min = args.tau_min if args.tau_min is not None else existing_range[0]
    tau_max = args.tau_max if args.tau_max is not None else existing_range[1]
    tau_step = args.tau_step if args.tau_step is not None else existing_range[2]
    cal_cfg["tau_range"] = [float(tau_min), float(tau_max), float(tau_step)]

    if args.opt_temp is not None:
        cal_cfg["opt_temp"] = args.opt_temp

    if args.shortlist is not None:
        eval_cfg["shortlist_topN"] = int(args.shortlist)

    constraints = dict(cal_cfg.get("constraints") or {})
    if args.min_precision is not None:
        constraints["min_precision"] = float(args.min_precision)
    if args.min_recall is not None:
        constraints["min_recall"] = float(args.min_recall)
    cal_cfg["constraints"] = constraints

    per_ec_cfg = dict(cal_cfg.get("per_ec") or {})
    if args.enable_per_class or args.per_class_mode or args.per_class_target is not None:
        per_ec_cfg["enable"] = True
    if args.per_class_mode:
        per_ec_cfg["mode"] = args.per_class_mode
    if args.per_class_target is not None:
        per_ec_cfg["target"] = float(args.per_class_target)
    if args.per_class_shrink is not None:
        per_ec_cfg["shrink"] = float(args.per_class_shrink)
    if args.per_class_min_positives is not None:
        per_ec_cfg["min_positives"] = int(args.per_class_min_positives)
    cal_cfg["per_ec"] = per_ec_cfg

    if args.plot_all:
        cal_cfg["plot_all"] = True
        cal_cfg.pop("plot_metric", None)
    elif args.plot_metric:
        cal_cfg["plot_all"] = False
        cal_cfg["plot_metric"] = args.plot_metric
    else:
        # Default to plotting only the primary metric when invoked by the benchmark
        cal_cfg.setdefault("plot_all", False)
        cal_cfg.setdefault("plot_metric", "micro_f1")

    if args.plots_dir is not None:
        cal_cfg["plots_dir"] = str(args.plots_dir)

    eval_cfg["calibration"] = cal_cfg
    return eval_cfg, cal_cfg


def _temperature_grid(base: float) -> np.ndarray:
    base = max(1e-3, float(base))
    lo = max(1e-2, base / 10.0)
    hi = min(100.0, base * 10.0)
    # Use a compact log-spaced grid; refine only if needed
    return np.unique(np.clip(np.geomspace(lo, hi, num=25), 1e-3, 1e3))


@torch.no_grad()
def _optimize_temperature(
    logits: torch.Tensor,
    y_true: np.ndarray,
    method: str,
    base_temp: float,
) -> Tuple[float, Dict[str, List[float]]]:
    """Return (best_temperature, curve) for BCE/Brier over a 1-D grid.

    curve contains the evaluated temperatures and losses for optional plotting.
    """
    y = torch.from_numpy(y_true.astype(np.float32))
    temps = _temperature_grid(base_temp)
    losses: List[float] = []
    eps = 1e-7
    for T in temps:
        p = torch.sigmoid(logits / float(T))
        if method == "brier":
            loss = torch.mean((p - y) ** 2)
        else:  # "bce"
            loss = -torch.mean(y * torch.log(p.clamp(min=eps)) + (1 - y) * torch.log((1 - p).clamp(min=eps)))
        losses.append(float(loss.cpu().item()))
    idx = int(np.argmin(losses))
    return float(temps[idx]), {"temperatures": [float(t) for t in temps], "loss": losses}


def _frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = float(start)
    # Inclusive range with tolerance against float accumulation
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def _pick_best_tau(records: List[Dict[str, Any]], constraints: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Pick the record with the best micro_f1 subject to optional constraints."""
    min_p = constraints.get("min_precision")
    min_r = constraints.get("min_recall")
    def ok(rec: Dict[str, Any]) -> bool:
        if min_p is not None and float(rec.get("micro_precision", 0.0)) < float(min_p):
            return False
        if min_r is not None and float(rec.get("micro_recall", 0.0)) < float(min_r):
            return False
        return True
    feasible = [r for r in records if ok(r)]
    pool = feasible if feasible else records
    return max(pool, key=lambda r: float(r.get("micro_f1", 0.0)))


def _maybe_plot_tau_curves(plots_dir: Optional[Path], taus: List[float], series: Dict[str, List[float]]) -> None:
    if plots_dir is None:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Primary: micro P/R/F1
    fig, ax = plt.subplots(figsize=(6, 4))
    if "micro_precision" in series:
        ax.plot(taus, series["micro_precision"], label="micro_precision")
    if "micro_recall" in series:
        ax.plot(taus, series["micro_recall"], label="micro_recall")
    if "micro_f1" in series:
        ax.plot(taus, series["micro_f1"], label="micro_f1")
    ax.set_xlabel("tau")
    ax.set_ylabel("score")
    ax.set_title("Calibration: metrics vs tau")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_vs_tau.png", dpi=150)
    plt.close(fig)


def _maybe_plot_temp_curve(plots_dir: Optional[Path], temp_curve: Dict[str, List[float]], method: str) -> None:
    if plots_dir is None or not temp_curve:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    temps = temp_curve.get("temperatures") or []
    losses = temp_curve.get("loss") or []
    if not temps or not losses:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(temps, losses, marker="o", ms=3)
    ax.set_xlabel("temperature")
    ax.set_ylabel(method.upper())
    ax.set_title(f"Temperature calibration ({method})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"temperature_{method}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tau calibration (and optional temperature calibration) for global-support eval.",
    )
    parser.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    parser.add_argument("--split", type=str, default=None, help="Calibration split (train/train_cal/val/test)")
    parser.add_argument("--tau-min", type=float, default=None)
    parser.add_argument("--tau-max", type=float, default=None)
    parser.add_argument("--tau-step", type=float, default=None)
    parser.add_argument("--opt-temp", choices=("none", "bce", "brier"), default="bce")
    parser.add_argument("--shortlist", type=int, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--enable-per-class", action="store_true")
    parser.add_argument("--per-class-mode", choices=("max_f1", "precision_at", "recall_at"), default=None)
    parser.add_argument("--per-class-target", type=float, default=None)
    parser.add_argument("--per-class-shrink", type=float, default=None)
    parser.add_argument("--per-class-min-positives", type=int, default=None)
    parser.add_argument("--per-class-out", type=Path, default=None, help="Optional JSON path to write per-EC thresholds (ignored unless implemented)")
    parser.add_argument("--plot-all", action="store_true", help="Force plots for all primary metrics")
    parser.add_argument(
        "--plot-metric",
        choices=("micro_f1", "macro_f1", "acc_top1_hit"),
        default=None,
        help="Single metric to plot when --plot-all is not used",
    )
    parser.add_argument("--plots-dir", type=Path, default=None, help="Optional directory for tau plots")
    # Historical no-op (kept for compatibility with callers)
    parser.add_argument("--force-embed", action="store_true", help="Ignored; calibration never embeds.")
    args = parser.parse_args()

    # Load full config and resolve effective calibration knobs
    cfg = load_yaml_cfg(args.config)
    eval_cfg, cal_cfg = _resolve_calibration_cfg(cfg, args)

    # Paths and evaluator
    paths = cfg.get("paths", {}) or {}
    outputs_path = Path(paths.get("outputs", "results")).resolve()
    split = str(cal_cfg.get("split", eval_cfg.get("split", "train")))
    shortlist = int(eval_cfg.get("shortlist_topN", 0))
    base_tau = float(eval_cfg.get("tau_multi", 0.35))
    base_temp = float(eval_cfg.get("temperature", 0.07))
    proto_path = Path(eval_cfg.get("prototypes_path") or (outputs_path / "prototypes.npz"))
    per_ec_path = None  # reserved for future per-EC threshold tuning
    plots_dir = Path(cal_cfg.get("plots_dir")) if cal_cfg.get("plots_dir") else None

    device = pick_device(cfg)
    evaluator = GlobalSupportEvaluator(
        cfg,
        proto_path,
        split,
        device=device,
        shortlist_topN=shortlist,
        per_ec_thresholds=None,
        ensure_top1=bool(eval_cfg.get("ensure_top1", True)),
        show_progress=bool(cfg.get("progress", True)),
    )

    # 1) Optional temperature calibration
    opt_temp = str(cal_cfg.get("opt_temp", "none")).lower()
    best_temp = float(base_temp)
    temp_curve: Dict[str, List[float]] = {}
    if opt_temp in {"bce", "brier"}:
        best_temp, temp_curve = _optimize_temperature(
            evaluator.class_logits, evaluator.y_true, opt_temp, base_temp
        )
        _maybe_plot_temp_curve(plots_dir, temp_curve, opt_temp)

    # 2) Tau grid search against the chosen temperature
    tau_min, tau_max, tau_step = cal_cfg.get("tau_range", [0.2, 0.6, 0.02])
    try:
        tau_min = float(tau_min); tau_max = float(tau_max); tau_step = float(tau_step)
    except Exception:
        tau_min, tau_max, tau_step = 0.2, 0.6, 0.02
    taus = _frange(tau_min, tau_max, tau_step)

    records: List[Dict[str, Any]] = []
    series: Dict[str, List[float]] = {"micro_precision": [], "micro_recall": [], "micro_f1": []}
    for t in taus:
        m = evaluator.evaluate(temperature=best_temp, tau_multi=t, shortlist_topN=shortlist)
        m["tau_multi"] = float(t)
        records.append(m)
        series["micro_precision"].append(float(m.get("micro_precision", 0.0)))
        series["micro_recall"].append(float(m.get("micro_recall", 0.0)))
        series["micro_f1"].append(float(m.get("micro_f1", 0.0)))

    best = _pick_best_tau(records, cal_cfg.get("constraints") or {})
    best_tau = float(best.get("tau_multi", base_tau))
    _maybe_plot_tau_curves(plots_dir, taus, series)

    # 3) Write calibration.json into the current outputs directory
    calib_path = Path(eval_cfg.get("calibration_path") or (outputs_path / "calibration.json"))
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau_multi": float(best_tau),
        "temperature": float(best_temp),
        "shortlist_topN": int(shortlist),
        "opt_temp": opt_temp,
        "grid": {
            "taus": [float(x) for x in taus],
            "micro_precision": series["micro_precision"],
            "micro_recall": series["micro_recall"],
            "micro_f1": series["micro_f1"],
        },
    }
    if temp_curve:
        payload["temperature_curve"] = temp_curve
    with open(calib_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"[auto_tune_tau] Wrote calibration → {calib_path}")
    if plots_dir is not None:
        print(f"[auto_tune_tau] Plots → {plots_dir}")


if __name__ == "__main__":
    main()
