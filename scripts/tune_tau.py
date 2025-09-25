#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_global import GlobalSupportEvaluator, _load_thresholds
from src.model_utils import load_cfg, pick_device


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


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
    ap.add_argument("--shortlist", type=int, default=None)
    ap.add_argument("--thresholds", type=Path, default=None, help="Optional per-EC thresholds JSON")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = pick_device(cfg)
    eval_cfg = cfg.get("eval", {}) or {}
    shortlist = int(args.shortlist if args.shortlist is not None else eval_cfg.get("shortlist_topN", 0))
    thresholds = _load_thresholds(args.thresholds)

    evaluator = GlobalSupportEvaluator(
        cfg,
        args.protos,
        args.split,
        device=device,
        shortlist_topN=shortlist,
        per_ec_thresholds=thresholds,
    )

    tau_values = _frange(args.tau_min, args.tau_max, args.tau_step)
    if args.temperature_range is None:
        temps = [float(eval_cfg.get("temperature", 0.07))]
    else:
        t_min, t_max, t_step = args.temperature_range
        temps = _frange(t_min, t_max, t_step)

    best = None
    records = []
    for temp in temps:
        for tau in tau_values:
            metrics = evaluator.evaluate(temperature=temp, tau_multi=tau, shortlist_topN=shortlist)
            score = metrics["micro_f1"]
            record = {
                "tau_multi": tau,
                "temperature": temp,
                "micro_f1": score,
                "macro_f1": metrics["macro_f1"],
            }
            records.append({**record, "metrics": metrics})
            if best is None or score > best["micro_f1"] or (
                score == best["micro_f1"] and metrics["macro_f1"] > best["macro_f1"]
            ):
                best = record
    if best is None:
        raise RuntimeError("Grid search produced no candidates")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
