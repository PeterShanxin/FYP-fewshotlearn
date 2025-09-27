#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_global import GlobalSupportEvaluator, _load_thresholds
from src.model_utils import load_cfg, pick_device


MetricKeys = Tuple[str, str, str, str, str]


def _frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def _parse_bool_flag(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected 'true' or 'false'")
    return lowered == "true"


def _resolve_metric_keys(metric: str, bucket: str) -> MetricKeys:
    if bucket != "overall" and metric == "acc_top1_hit":
        raise ValueError("acc_top1_hit objective is only available for the overall bucket")

    def _bucketize(name: str) -> str:
        return name if bucket == "overall" else f"{bucket}_{name}"

    primary = metric if bucket == "overall" else _bucketize(metric)

    if metric == "micro_f1":
        tie1 = _bucketize("macro_f1")
        tie2 = "acc_top1_hit"
    elif metric == "macro_f1":
        tie1 = _bucketize("micro_f1")
        tie2 = "acc_top1_hit"
    else:  # acc_top1_hit
        tie1 = _bucketize("micro_f1")
        tie2 = _bucketize("macro_f1")

    precision_key = _bucketize("micro_precision")
    recall_key = _bucketize("micro_recall")
    return primary, tie1, tie2, precision_key, recall_key


def _optimize_temperature(
    evaluator: GlobalSupportEvaluator,
    loss_type: str,
    *,
    default_temperature: float,
) -> Tuple[float, List[Dict[str, float]]]:
    logits = evaluator.class_logits
    if logits.numel() == 0:
        return default_temperature, []

    y_true = torch.from_numpy(evaluator.y_true).to(dtype=logits.dtype)
    temps = torch.arange(0.03, 0.1501, 0.002)
    history: List[Dict[str, float]] = []

    best_temp = default_temperature
    best_loss = float("inf")
    for temp in temps:
        probs = torch.sigmoid(logits / temp)
        if loss_type == "bce":
            loss = F.binary_cross_entropy(probs, y_true, reduction="mean").item()
        else:
            loss = torch.mean((probs - y_true) ** 2).item()
        history.append({"temperature": float(temp.item()), "loss": float(loss)})
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_temp = float(temp.item())
    return best_temp, history


def _write_report(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        fieldnames = sorted({k for record in records for k in record.keys()})
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(list(records), handle, indent=2)


def _compute_per_class_thresholds(
    evaluator: GlobalSupportEvaluator,
    *,
    temperature: float,
    tau_grid: Sequence[float],
    shortlist: int,
    ensure_top1: bool,
    global_tau: float,
    shrink: float,
    min_positives: int,
    mode: str,
    target: Optional[float],
) -> Dict[str, float]:
    num_classes = evaluator.num_classes
    if num_classes == 0:
        return {}

    if not tau_grid:
        tau_grid = [global_tau]

    probs = torch.sigmoid(evaluator.class_logits / float(temperature))
    if probs.shape[0] == 0:
        adjusted = float((1.0 - shrink) * global_tau + shrink * global_tau)
        return {ec: round(adjusted, 6) for ec in evaluator.class_names}

    y_true_np = evaluator.y_true.astype(np.bool_)
    positives = y_true_np.sum(axis=0)

    if shortlist > 0 and shortlist < num_classes:
        topk = torch.topk(probs, k=shortlist, dim=1)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, topk.indices, True)
    else:
        mask = torch.ones_like(probs, dtype=torch.bool)

    top1_idx = probs.argmax(dim=1)

    stats_per_tau: List[Dict[str, np.ndarray]] = []
    tau_candidates = sorted({round(float(t), 6) for t in tau_grid})

    for tau in tau_candidates:
        decisions = (probs >= tau) & mask
        if ensure_top1:
            no_positive = decisions.sum(dim=1) == 0
            if torch.any(no_positive):
                decisions[no_positive, top1_idx[no_positive]] = True
        preds_np = decisions.cpu().numpy().astype(bool)

        tp = np.sum(preds_np & y_true_np, axis=0)
        fp = np.sum(preds_np & (~y_true_np), axis=0)
        fn = np.sum((~preds_np) & y_true_np, axis=0)

        denom_prec = tp + fp
        denom_rec = tp + fn
        denom_f1 = 2 * tp + fp + fn

        precision = np.divide(tp, denom_prec, out=np.zeros_like(tp, dtype=float), where=denom_prec > 0)
        recall = np.divide(tp, denom_rec, out=np.zeros_like(tp, dtype=float), where=denom_rec > 0)
        f1 = np.divide(2 * tp, denom_f1, out=np.zeros_like(tp, dtype=float), where=denom_f1 > 0)

        stats_per_tau.append(
            {
                "tau": float(tau),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    per_class: Dict[str, float] = {}
    stats_lookup = {stats["tau"]: stats for stats in stats_per_tau}
    shrink = float(np.clip(shrink, 0.0, 1.0))

    for idx, ec in enumerate(evaluator.class_names):
        if positives[idx] < min_positives:
            tau_choice = global_tau
        else:
            tau_choice: Optional[float] = None
            if mode == "max_f1":
                best_score = -1.0
                for stats in stats_per_tau:
                    score = float(stats["f1"][idx])
                    if score > best_score + 1e-12:
                        best_score = score
                        tau_choice = stats["tau"]
            elif mode == "precision_at":
                if target is None:
                    raise ValueError("per-class-target must be provided for precision_at mode")
                for tau in tau_candidates:
                    stats = stats_lookup[tau]
                    if float(stats["precision"][idx]) >= target - 1e-12:
                        tau_choice = tau
                        break
            elif mode == "recall_at":
                if target is None:
                    raise ValueError("per-class-target must be provided for recall_at mode")
                for tau in reversed(tau_candidates):
                    stats = stats_lookup[tau]
                    if float(stats["recall"][idx]) >= target - 1e-12:
                        tau_choice = tau
                        break
            else:
                raise ValueError(f"Unsupported per-class mode: {mode}")

            if tau_choice is None:
                tau_choice = global_tau

        adjusted = (1.0 - shrink) * float(tau_choice) + shrink * float(global_tau)
        adjusted = max(0.0, min(adjusted, 1.0))
        per_class[ec] = round(float(adjusted), 6)

    return per_class


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-search tau/temperature for global eval.")
    ap.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    ap.add_argument("--protos", type=Path, required=True, help="Prototype bank NPZ")
    ap.add_argument("--split", type=str, default="val", help="Split for calibration")
    ap.add_argument("--out", type=Path, default=Path("artifacts/calibration.json"))
    ap.add_argument("--tau-min", type=float, default=0.2)
    ap.add_argument("--tau-max", type=float, default=0.6)
    ap.add_argument("--tau-step", type=float, default=0.025)
    ap.add_argument("--temperature-range", nargs=3, type=float, default=None, metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--metric", choices=("micro_f1", "macro_f1", "acc_top1_hit"), default="micro_f1")
    ap.add_argument("--bucket", choices=("overall", "head", "medium", "tail"), default="overall")
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--ensure-top1", choices=("true", "false"), default=None, help="Override ensure_top1 from config")
    ap.add_argument(
        "--opt-temp",
        choices=("none", "bce", "brier"),
        default="none",
        help="Optional temperature calibration before tau grid search",
    )
    ap.add_argument("--shortlist", type=int, default=None)
    ap.add_argument("--thresholds", type=Path, default=None, help="Optional per-EC thresholds JSON")
    ap.add_argument("--per-class-out", type=Path, default=None, help="Optional per-class thresholds JSON output")
    ap.add_argument(
        "--per-class-shrink",
        type=float,
        default=0.25,
        help="Shrinkage factor towards global tau for per-class thresholds",
    )
    ap.add_argument(
        "--per-class-min-positives",
        type=int,
        default=5,
        help="If class positives are below this count, fall back to global tau",
    )
    ap.add_argument(
        "--per-class-mode",
        choices=("max_f1", "precision_at", "recall_at"),
        default="max_f1",
    )
    ap.add_argument(
        "--per-class-target",
        type=float,
        default=None,
        help="Target precision/recall when using precision_at or recall_at modes",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = pick_device(cfg)
    eval_cfg = cfg.get("eval", {}) or {}

    shortlist = int(args.shortlist if args.shortlist is not None else eval_cfg.get("shortlist_topN", 0))
    thresholds = _load_thresholds(args.thresholds)
    ensure_top1_cfg = bool(eval_cfg.get("ensure_top1", True))
    ensure_top1_flag = _parse_bool_flag(args.ensure_top1)
    ensure_top1 = ensure_top1_cfg if ensure_top1_flag is None else ensure_top1_flag

    evaluator = GlobalSupportEvaluator(
        cfg,
        args.protos,
        args.split,
        device=device,
        shortlist_topN=shortlist,
        per_ec_thresholds=thresholds,
        ensure_top1=ensure_top1,
    )

    tau_values = _frange(args.tau_min, args.tau_max, args.tau_step)
    tau_values = sorted({float(t) for t in tau_values})

    base_temperature = float(eval_cfg.get("temperature", 0.07))
    if args.temperature_range is None:
        temps: List[float] = [base_temperature]
    else:
        t_min, t_max, t_step = args.temperature_range
        temps = _frange(t_min, t_max, t_step)

    temperature_history: List[Dict[str, float]] = []
    if args.opt_temp != "none" and args.temperature_range is None:
        best_temp, temperature_history = _optimize_temperature(
            evaluator,
            args.opt_temp,
            default_temperature=base_temperature,
        )
        temps = [best_temp]

    primary_key, tie1_key, tie2_key, precision_key, recall_key = _resolve_metric_keys(args.metric, args.bucket)

    report_records: List[Dict[str, object]] = []
    best_record: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None

    for temp in temps:
        for tau in tau_values:
            metrics = evaluator.evaluate(
                temperature=temp,
                tau_multi=tau,
                shortlist_topN=shortlist,
            )

            if primary_key not in metrics:
                raise KeyError(f"Metric key '{primary_key}' not found in evaluator output")

            meets_constraints = True
            if args.min_precision is not None:
                precision_val = metrics.get(precision_key)
                if precision_val is None:
                    raise KeyError(f"Precision key '{precision_key}' not found in evaluator output")
                if precision_val < args.min_precision - 1e-12:
                    meets_constraints = False
            if meets_constraints and args.min_recall is not None:
                recall_val = metrics.get(recall_key)
                if recall_val is None:
                    raise KeyError(f"Recall key '{recall_key}' not found in evaluator output")
                if recall_val < args.min_recall - 1e-12:
                    meets_constraints = False

            record = dict(metrics)
            record["meets_constraints"] = bool(meets_constraints)
            record["objective_score"] = float(metrics[primary_key])
            report_records.append(record)

            if not meets_constraints:
                continue

            candidate = (
                float(metrics[primary_key]),
                float(metrics.get(tie1_key, -float("inf"))),
                float(metrics.get(tie2_key, -float("inf"))),
            )

            if best_metrics is None:
                best_metrics = metrics
                best_record = record
                continue

            current = (
                float(best_metrics[primary_key]),
                float(best_metrics.get(tie1_key, -float("inf"))),
                float(best_metrics.get(tie2_key, -float("inf"))),
            )

            if candidate > current:
                best_metrics = metrics
                best_record = record

    if best_metrics is None or best_record is None:
        raise RuntimeError("No candidate satisfied the objective and constraints")

    best_out = {
        "tau_multi": float(best_metrics["tau_multi"]),
        "temperature": float(best_metrics["temperature"]),
        "micro_f1": float(best_metrics.get("micro_f1", 0.0)),
        "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
        "acc_top1_hit": float(best_metrics.get("acc_top1_hit", 0.0)),
        "metric": args.metric,
        "bucket": args.bucket,
        "objective_score": float(best_metrics[primary_key]),
    }

    if temperature_history:
        best_out["temperature_search"] = temperature_history

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(best_out, handle, indent=2)
    print(json.dumps(best_out, indent=2))

    if args.report is not None:
        _write_report(args.report, report_records)

    if args.per_class_out is not None:
        target = args.per_class_target
        if args.per_class_mode in {"precision_at", "recall_at"} and target is None:
            raise ValueError("--per-class-target is required for precision_at and recall_at modes")
        if target is not None and not (0.0 <= target <= 1.0):
            raise ValueError("--per-class-target must lie in [0, 1]")

        per_class_thresholds = _compute_per_class_thresholds(
            evaluator,
            temperature=float(best_metrics["temperature"]),
            tau_grid=sorted({*tau_values, float(best_metrics["tau_multi"])}),
            shortlist=shortlist,
            ensure_top1=ensure_top1,
            global_tau=float(best_metrics["tau_multi"]),
            shrink=float(args.per_class_shrink),
            min_positives=int(args.per_class_min_positives),
            mode=args.per_class_mode,
            target=target,
        )

        args.per_class_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_class_out, "w", encoding="utf-8") as handle:
            json.dump(per_class_thresholds, handle, indent=2)


if __name__ == "__main__":
    main()
