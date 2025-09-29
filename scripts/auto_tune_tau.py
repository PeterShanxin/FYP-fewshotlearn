#!/usr/bin/env python
from __future__ import annotations

"""
Auto-tune global-support thresholds with minimal friction.

What it does:
- Locates or optionally builds a prototype bank.
- Picks a calibration split automatically (train_cal → train → val → test) that has coverage.
- Grid-searches tau (and optionally temperature) and writes a calibration JSON.
- Emits a CSV report of tau/temperature vs metrics and prints a compact summary.

Typical usage:
  python scripts/auto_tune_tau.py --config config.yaml
  python scripts/auto_tune_tau.py --config .tmp_configs/run_id50_fold1.yaml

Outputs (by default):
- Calibration JSON next to the configured eval.calibration_path
- CSV report next to the calibration file (tune_report.csv)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root on sys.path for src.* imports
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_global import GlobalSupportEvaluator, _load_thresholds  # type: ignore
from src.model_utils import build_model, infer_input_dim, load_checkpoint, load_cfg, pick_device  # type: ignore
from src.prototype_bank import build_prototypes, save_prototypes, load_prototypes  # type: ignore


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
    """Compute per-EC thresholds using the evaluator's logits/targets.

    Matches the approach used by scripts/tune_tau.py.
    """
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


def _choose_split_with_coverage(
    cfg: dict,
    prototypes_path: Path,
    candidates: Sequence[str],
) -> Optional[str]:
    """Pick first split that has at least one accession with an EC present in prototypes."""
    try:
        protos, _ = load_prototypes(prototypes_path)
    except Exception:
        return None
    if not protos:
        return None
    proto_ecs = set(protos.keys())
    splits_dir = Path(cfg["paths"]["splits_dir"]) if cfg.get("paths") else None
    if not splits_dir:
        return None

    def _covered_accessions(jsonl: Path) -> int:
        if not jsonl.exists():
            return 0
        count = 0
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ec = str(obj.get("ec", ""))
                if ec in proto_ecs:
                    accs = obj.get("accessions", [])
                    if isinstance(accs, list):
                        count += len(accs)
        return count

    for name in candidates:
        path = Path(splits_dir) / f"{name}.jsonl"
        if _covered_accessions(path) > 0:
            return name
    return None


def _ensure_prototypes(
    cfg: dict,
    desired_out: Path,
) -> Path:
    """Return a prototype NPZ path, building it if necessary when feasible.

    Raises FileNotFoundError if it cannot be located or built.
    """
    out_path = Path(desired_out)
    if out_path.exists():
        return out_path

    # Attempt to build from train split and checkpoint
    paths = cfg.get("paths", {}) or {}
    embeddings_path = Path(paths["embeddings"]) if "embeddings" in paths else None
    splits_dir = Path(paths["splits_dir"]) if "splits_dir" in paths else None
    outputs_dir = Path(paths.get("outputs", "results"))
    ckpt_path = outputs_dir / "checkpoints" / "protonet.pt"
    train_split = (Path(splits_dir) / "train.jsonl") if splits_dir else None

    if not (embeddings_path and train_split and ckpt_path.exists()):
        raise FileNotFoundError(
            "Prototypes not found and cannot auto-build (missing embeddings/train split/checkpoint)."
        )

    device = pick_device(cfg)
    input_dim = infer_input_dim(embeddings_path)
    model = build_model(cfg, input_dim, device)
    load_checkpoint(model, ckpt_path, device)

    prototypes, train_counts = build_prototypes(
        train_split,
        embeddings_path,
        model,
        device=device,
        subprototypes_per_ec=int(cfg.get("eval", {}).get("subprototypes_per_ec", 1)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_prototypes(out_path, prototypes, train_counts=train_counts)
    return out_path


def _write_report(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for record in records for k in record.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-tune tau/temperature for global-support evaluation")
    ap.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    ap.add_argument("--split", type=str, default="auto", choices=("auto", "train_cal", "train", "val", "test"))
    ap.add_argument("--protos", type=Path, default=None, help="Optional path to prototypes NPZ (auto if omitted)")
    ap.add_argument("--tau-min", type=float, default=0.2)
    ap.add_argument("--tau-max", type=float, default=0.6)
    ap.add_argument("--tau-step", type=float, default=0.02)
    ap.add_argument("--temperature-range", nargs=3, type=float, default=None, metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--opt-temp", choices=("none", "bce", "brier"), default="none")
    ap.add_argument("--metric", choices=("micro_f1", "macro_f1", "acc_top1_hit"), default="micro_f1")
    ap.add_argument("--bucket", choices=("overall", "head", "medium", "tail"), default="overall")
    ap.add_argument("--min-precision", type=float, default=0.10,
                    help="Minimum micro-precision constraint (default: 0.10)")
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--shortlist", type=int, default=None)
    ap.add_argument("--ensure-top1", choices=("true", "false"), default=None)
    ap.add_argument("--thresholds", type=Path, default=None, help="Optional per-EC thresholds JSON")
    ap.add_argument("--per-class-out", type=Path, default=None, help="Optional per-class thresholds JSON output")
    ap.add_argument("--per-class-mode", choices=("max_f1", "precision_at", "recall_at"), default="max_f1")
    ap.add_argument("--per-class-target", type=float, default=None,
                    help="Target precision/recall for precision_at/recall_at modes")
    ap.add_argument("--per-class-shrink", type=float, default=0.25,
                    help="Shrink per-EC taus towards global tau (0..1)")
    ap.add_argument("--per-class-min-positives", type=int, default=5,
                    help="If a class has <N positives on the calibration split, fall back to global tau")
    ap.add_argument("--out", type=Path, default=None, help="Calibration JSON output (auto if omitted)")
    ap.add_argument("--report", type=Path, default=None, help="CSV report output (auto if omitted)")
    ap.add_argument("--plot-metric", choices=("micro_f1", "macro_f1", "acc_top1_hit"), default="micro_f1",
                    help="Metric to plot in the tuning curve if matplotlib is available")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = pick_device(cfg)
    eval_cfg = cfg.get("eval", {}) or {}

    shortlist = int(args.shortlist if args.shortlist is not None else eval_cfg.get("shortlist_topN", 0))
    ensure_top1_cfg = bool(eval_cfg.get("ensure_top1", True))
    ensure_top1_flag = _parse_bool_flag(args.ensure_top1)
    ensure_top1 = ensure_top1_cfg if ensure_top1_flag is None else ensure_top1_flag

    # Resolve prototype path (prefer CLI, else config, else build)
    protos_path: Optional[Path] = args.protos
    if protos_path is None:
        p = eval_cfg.get("prototypes_path")
        protos_path = Path(p) if p else Path("artifacts/prototypes.npz")
    try:
        protos_path = _ensure_prototypes(cfg, protos_path)
    except FileNotFoundError:
        if not Path(protos_path).exists():
            raise

    # Resolve out/report paths
    out_path = args.out
    if out_path is None:
        cp = eval_cfg.get("calibration_path")
        out_path = Path(cp) if cp else Path("artifacts/calibration.json")
        # If config points to a folder that doesn't exist yet, place next to prototypes
        out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report
    if report_path is None:
        report_path = out_path.with_name("tune_report.csv")

    # Pick split
    split = args.split
    if split == "auto":
        candidates = ("train_cal", "train", "val", "test")
        chosen = _choose_split_with_coverage(cfg, protos_path, candidates)
        split = chosen or "train"

    thresholds = _load_thresholds(args.thresholds)

    evaluator = GlobalSupportEvaluator(
        cfg,
        protos_path,
        split,
        device=device,
        shortlist_topN=shortlist,
        per_ec_thresholds=thresholds,
        ensure_top1=ensure_top1,
    )

    # Build search grids
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
    best_metrics: Optional[Dict[str, float]] = None
    best_record: Optional[Dict[str, object]] = None

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
        # Fallback: pick the best unconstrained candidate (warn the user)
        if not report_records:
            raise RuntimeError("No candidate satisfied the objective and constraints, and no records to fallback to")
        def _score(rec: Dict[str, object]) -> Tuple[float, float, float]:
            return (
                float(rec.get(primary_key, -float("inf"))),
                float(rec.get(tie1_key, -float("inf"))),
                float(rec.get(tie2_key, -float("inf"))),
            )
        best_record = max(report_records, key=_score)
        # Reuse the record as metrics-like mapping
        best_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_record.items() if isinstance(k, str)}  # type: ignore[assignment]
        print("[warn] No candidate met constraints; falling back to best unconstrained result.")

    best_out = {
        "tau_multi": float(best_metrics["tau_multi"]),
        "temperature": float(best_metrics["temperature"]),
        "micro_f1": float(best_metrics.get("micro_f1", 0.0)),
        "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
        "acc_top1_hit": float(best_metrics.get("acc_top1_hit", 0.0)),
        "metric": args.metric,
        "bucket": args.bucket,
        "objective_score": float(best_metrics[primary_key]),
        "constraints_satisfied": bool(best_record.get("meets_constraints", False) if isinstance(best_record, dict) else False),
        "split": str(split),
        "shortlist_topN": int(shortlist),
    }
    if temperature_history:
        best_out["temperature_search"] = temperature_history

    # Write outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(best_out, handle, indent=2)
    _write_report(report_path, report_records)

    # Print compact summary and a small table
    print(json.dumps(best_out, indent=2))
    # Show top lines (sort by objective)
    try:
        rows = [r for r in report_records if r.get("meets_constraints", False)]
        rows.sort(key=lambda r: (
            float(r.get(primary_key, 0.0)),
            float(r.get(tie1_key, 0.0)),
            float(r.get(tie2_key, 0.0)),
        ), reverse=True)
        head = rows[:10]
        if head:
            print("\nTop candidates (tau, temp, micro_f1, macro_f1, acc_top1):")
            for r in head:
                print(
                    f"  tau={r.get('tau_multi'):0.3f}  T={r.get('temperature'):0.3f}  "
                    f"micro_f1={r.get('micro_f1', 0.0):0.4f}  macro_f1={r.get('macro_f1', 0.0):0.4f}  "
                    f"acc_top1={r.get('acc_top1_hit', 0.0):0.4f}"
                )
        else:
            print("\nNo candidates met constraints; see CSV report for details.")
    except Exception:
        # Don't fail tuning just because of printing
        pass

    # Optional plot: tau vs selected metric, grouped by temperature
    try:
        import matplotlib.pyplot as plt  # type: ignore
        if report_records:
            # collect series
            by_temp: Dict[float, List[Tuple[float, float]]] = {}
            for r in report_records:
                try:
                    t = float(r.get("temperature", 0.0))
                    tau = float(r.get("tau_multi", 0.0))
                    y = float(r.get(args.plot_metric, 0.0))
                except Exception:
                    continue
                by_temp.setdefault(t, []).append((tau, y))

            plt.figure(figsize=(7, 4))
            for t, pts in sorted(by_temp.items(), key=lambda kv: kv[0]):
                pts.sort(key=lambda x: x[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                label = f"T={t:.3f}"
                plt.plot(xs, ys, marker='o', label=label)
            plt.xlabel("tau")
            plt.ylabel(args.plot_metric)
            plt.title("Tuning curve: tau vs " + args.plot_metric)
            plt.grid(True, alpha=0.3)
            if len(by_temp) > 1:
                plt.legend()
            fig_path = report_path.with_name("tune_curve.png")
            plt.tight_layout()
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"\nSaved tuning curve → {fig_path}")
        else:
            print("\nNo report records to plot.")
    except Exception:
        # Matplotlib may be unavailable in some environments; silently skip.
        pass

    # Optional: per-class thresholds
    try:
        if args.per_class_out is not None:
            target = args.per_class_target
            if args.per_class_mode in {"precision_at", "recall_at"} and target is None:
                raise ValueError("--per-class-target is required for precision_at and recall_at modes")
            if target is not None and not (0.0 <= float(target) <= 1.0):
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
            print(f"Saved per-EC thresholds → {args.per_class_out}")
    except Exception as e:
        print(f"[warn] per-class thresholding skipped: {e}")


if __name__ == "__main__":
    main()
