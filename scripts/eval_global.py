#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_global import run_global_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate global-support classification.")
    ap.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    ap.add_argument("--protos", type=Path, required=True, help="Path to prototypes NPZ")
    ap.add_argument("--split", type=str, default="test", help="Evaluation split (train/val/test)")
    ap.add_argument("--tau", type=float, default=None, help="Override tau_multi")
    ap.add_argument("--temperature", type=float, default=None, help="Override temperature")
    ap.add_argument("--shortlist", type=int, default=None, help="Override shortlist_topN")
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional JSON mapping EC→threshold",
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/calibration.json"),
        help="Calibration JSON (defaults to artifacts/calibration.json)",
    )
    args = ap.parse_args()

    metrics = run_global_evaluation(
        config_path=args.config,
        prototypes_path=args.protos,
        split=args.split,
        tau_multi=args.tau,
        temperature=args.temperature,
        shortlist_topN=args.shortlist,
        thresholds_path=args.thresholds,
        calibration_path=args.calibration,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
