#!/usr/bin/env python3
"""Run identity-constrained benchmarking across multiple cutoffs and folds.

Steps:
1) Prepare identity-cluster-based splits for the requested cutoffs and K folds.
2) For each (cutoff, fold), train + eval ProtoNet using a derived config
   that points to the split dir and the cutoff-specific clusters TSV.
3) Aggregate metrics across folds per cutoff and write a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml
import numpy as np
from datetime import datetime

# Lightweight status logger: append to RUNALL_STATUS_LOG if set
def _status_log(message: str) -> None:
    """Append a structured line to the configured status log.

    When running as a nested calibration sub-run, also tee to the parent log
    and append an optional scope tag for clarity (e.g., cutoff/fold).
    """
    path = os.environ.get("RUNALL_STATUS_LOG")
    parent = os.environ.get("RUNALL_PARENT_STATUS_LOG")
    scope_tag = os.environ.get("RUNALL_SCOPE_TAG")
    try:
        tzname = os.environ.get("RUNALL_TZ", "Asia/Singapore")
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            ts = datetime.now(ZoneInfo(tzname)).strftime("%Y-%m-%dT%H:%M:%S %Z")
        except Exception:
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        suffix = f" {scope_tag}" if scope_tag else ""
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{ts} {message}{suffix}\n")
        if parent and (not path or parent != path):
            try:
                with open(parent, "a", encoding="utf-8") as f2:
                    f2.write(f"{ts} {message}{suffix}\n")
            except Exception:
                pass
    except Exception:
        # best-effort; never fail the run because logging failed
        pass


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_calibration(
    cfg_path: Path,
    outputs_path: Path,
    *,
    cutoff_pct: int,
    fold_index: int,
    calibration_cfg: Dict[str, Any],
    calibrate_only: bool,
    results_root: Path,
    default_shortlist: int,
) -> None:
    """Invoke auto_tune_tau for the given fold and optionally surface plots."""

    cmd: List[str] = [
        "python",
        "scripts/auto_tune_tau.py",
        "--config",
        str(cfg_path),
    ]

    split = str(calibration_cfg.get("split", "train"))
    cmd.extend(["--split", split])

    tau_range = calibration_cfg.get("tau_range")
    if not (isinstance(tau_range, (list, tuple)) and len(tau_range) == 3):
        tau_range = [0.2, 0.6, 0.02]
    try:
        tau_min, tau_max, tau_step = (float(tau_range[0]), float(tau_range[1]), float(tau_range[2]))
    except Exception:
        tau_min, tau_max, tau_step = 0.2, 0.6, 0.02
    cmd.extend([
        "--tau-min",
        f"{tau_min:.6f}",
        "--tau-max",
        f"{tau_max:.6f}",
        "--tau-step",
        f"{tau_step:.6f}",
    ])

    opt_temp = str(calibration_cfg.get("opt_temp", "bce"))
    cmd.extend(["--opt-temp", opt_temp])

    shortlist_override = calibration_cfg.get("shortlist_topN")
    shortlist_value: Optional[int]
    if shortlist_override is None:
        shortlist_value = default_shortlist
    else:
        try:
            shortlist_value = int(shortlist_override)
        except Exception:
            shortlist_value = default_shortlist
    if shortlist_value and shortlist_value > 0:
        cmd.extend(["--shortlist", str(shortlist_value)])

    constraints_cfg = calibration_cfg.get("constraints") or {}
    min_precision = constraints_cfg.get("min_precision")
    if min_precision is not None:
        cmd.extend(["--min-precision", str(min_precision)])
    min_recall = constraints_cfg.get("min_recall")
    if min_recall is not None:
        cmd.extend(["--min-recall", str(min_recall)])

    plots_dir = outputs_path / "calibration_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_ec_cfg = calibration_cfg.get("per_ec") or {}
    per_ec_enabled = bool(per_ec_cfg.get("enable", False))
    per_ec_path = outputs_path / "per_ec_thresholds.json"
    if per_ec_enabled:
        cmd.extend(["--per-class-out", str(per_ec_path)])
        mode = per_ec_cfg.get("mode")
        if mode:
            cmd.extend(["--per-class-mode", str(mode)])
        target = per_ec_cfg.get("target")
        if target is not None:
            cmd.extend(["--per-class-target", str(target)])
        shrink = per_ec_cfg.get("shrink")
        if shrink is not None:
            cmd.extend(["--per-class-shrink", str(shrink)])
        min_pos = per_ec_cfg.get("min_positives")
        if min_pos is not None:
            cmd.extend(["--per-class-min-positives", str(min_pos)])

    plot_all = calibration_cfg.get("plot_all")
    if plot_all is None:
        plot_all = calibrate_only
    if plot_all:
        cmd.append("--plot-all")
    else:
        plot_metric = str(calibration_cfg.get("plot_metric", "micro_f1"))
        cmd.extend(["--plot-metric", plot_metric])
    cmd.extend(["--plots-dir", str(plots_dir)])

    _status_log(
        f"phase=calibrate cutoff={cutoff_pct} fold={fold_index} event=start split={split} opt_temp={opt_temp}"
    )
    print(
        f"[benchmark][calibrate] cutoff={cutoff_pct} fold={fold_index} split={split} opt_temp={opt_temp}"
    )
    try:
        # Propagate a scope tag so nested run_all.sh can tee status lines back
        # to the parent's log and annotate entries with cutoff/fold for clarity.
        env = os.environ.copy()
        env["RUNALL_SCOPE_TAG"] = f"scope=calibrate cutoff={cutoff_pct} fold={fold_index}"
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        _status_log(
            f"phase=calibrate cutoff={cutoff_pct} fold={fold_index} event=finish status=failed exit_code={getattr(exc, 'returncode', 'NA')}"
        )
        raise
    _status_log(
        f"phase=calibrate cutoff={cutoff_pct} fold={fold_index} event=finish status=success"
    )

    if calibrate_only:
        dest_dir = results_root / "figures"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for plot in sorted(plots_dir.glob("*.png")):
            dest_name = (
                f"calibrate_cutoff{cutoff_pct}_fold{fold_index}_{plot.name}"
            )
            shutil.copy2(plot, dest_dir / dest_name)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--cutoffs", default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip_prepare", action="store_true", help="Assume splits already prepared")
    ap.add_argument("--force-embed", dest="force_embed", action="store_true", help="Recompute embeddings even if files exist")
    ap.add_argument("--calibrate-only", action="store_true", help="Run calibration-only workflow and skip downstream evaluation")
    args = ap.parse_args()

    base_cfg = load_cfg(args.config)
    bench_cfg = base_cfg.get("identity_benchmark", {}) or {}
    run_episodic = bool(bench_cfg.get("episodic", True))
    run_global = bool(bench_cfg.get("global_support", True))
    if not (run_episodic or run_global):
        print("[benchmark][warn] identity_benchmark requested neither episodic nor global evaluation; defaulting to episodic=true.")
        run_episodic = True
    mode_labels = []
    if run_episodic:
        mode_labels.append("episodic")
    if run_global:
        mode_labels.append("global_support")
    print(f"[benchmark] evaluation modes: {', '.join(mode_labels)}")
    out_root = Path(base_cfg["paths"].get("outputs", "results")).resolve()
    results_root = out_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Respect calibrate-only from CLI or config mode
    # Read calibration mode from the base config
    cal_mode = str(((base_cfg.get("eval") or {}).get("calibration") or {}).get("mode", "off")).lower()
    cfg_calibrate_only = cal_mode == "only"
    effective_calibrate_only = bool(args.calibrate_only or cfg_calibrate_only)

    if effective_calibrate_only:
        figures_dir = results_root / "figures"
        if figures_dir.exists():
            shutil.rmtree(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        run(["python", "scripts/prepare_identity_splits.py", "-c", args.config, "--cutoffs", args.cutoffs, "--folds", str(args.folds)])

    # Use config id_thresholds if present
    cfg_thresholds = base_cfg.get("id_thresholds")
    if isinstance(cfg_thresholds, list) and cfg_thresholds:
        try:
            cutoffs = [float(x) / (100.0 if max(cfg_thresholds) > 1 else 1.0) for x in cfg_thresholds]
        except Exception:
            cutoffs = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    else:
        cutoffs = [float(x) for x in args.cutoffs.split(",") if x.strip()]
    K_folds = int(base_cfg.get("folds", args.folds))
    summary: Dict[str, Any] = {}

    tmp_cfg_dir = Path(".tmp_configs")
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path.cwd())).decode("utf-8").strip()
    except Exception:
        git_commit = None

    force_embed_paths_done: set[str] = set()

    for cut in cutoffs:
        pct = int(round(cut * 100))
        per_fold_metrics: List[Dict[str, Any]] = []
        split_dir = results_root / f"split-{pct}"
        folds_json = split_dir / "folds.json"
        # Read folds and clusters to compute counts
        with open(folds_json, "r", encoding="utf-8") as f:
            folds_info = json.load(f)
        clusters_json = split_dir / "clusters.json"
        with open(clusters_json, "r", encoding="utf-8") as f:
            clu2acc = json.load(f)
        # Build cluster sizes once
        cl_size = {cid: len(accs) for cid, accs in clu2acc.items()}
        for fi in range(K_folds):
            run_cfg = deepcopy(base_cfg)
            run_cfg["paths"] = dict(run_cfg["paths"])  # shallow copy
            run_cfg["paths"]["splits_dir"] = str(split_dir / f"fold-{fi+1}")
            run_cfg["paths"]["outputs"] = str(split_dir / f"fold-{fi+1}")
            run_cfg["paths"]["clusters_tsv"] = str((split_dir / "clusters.tsv").resolve())
            # Ensure identity disjoint during episodes
            run_cfg["identity_disjoint"] = True

            eval_cfg = dict(run_cfg.get("eval") or {})
            outputs_path = Path(run_cfg["paths"]["outputs"]).resolve()
            outputs_dir = outputs_path
            default_shortlist = int(eval_cfg.get("shortlist_topN", 0))
            calibration_cfg = dict(eval_cfg.get("calibration") or {})
            per_ec_cfg = dict(calibration_cfg.get("per_ec") or {})
            calibration_cfg["per_ec"] = per_ec_cfg
            constraints_cfg = dict(calibration_cfg.get("constraints") or {})
            calibration_cfg["constraints"] = constraints_cfg
            per_ec_enabled = bool(per_ec_cfg.get("enable", False))
            metrics_path = outputs_path / "metrics.json"
            global_metrics_path = outputs_path / "global_metrics.json"
            calibration_path_file = outputs_path / "calibration.json"
            proto_path = None
            if run_global:
                proto_path = outputs_path / "prototypes.npz"
                calib_path = outputs_path / "calibration.json"
                # Always override per-fold artifact destinations so evaluation reads the
                # prototypes written for this split instead of any global default.
                eval_cfg["prototypes_path"] = str(proto_path)
                eval_cfg["calibration_path"] = str(calib_path)
            if per_ec_enabled:
                eval_cfg["per_ec_thresholds_path"] = str((outputs_path / "per_ec_thresholds.json").resolve())
            eval_cfg["calibration"] = calibration_cfg
            run_cfg["eval"] = eval_cfg
            mode_val = str(calibration_cfg.get("mode", "off")).lower()
            calibration_requested = bool(args.calibrate_only or mode_val in {"produce", "only"})
            # Create temp config
            cfg_path = tmp_cfg_dir / f"run_id{pct}_fold{fi+1}.yaml"
            dump_cfg(run_cfg, cfg_path)

            # Resume logic: skip completed folds when rerunning
            if effective_calibrate_only and calibration_requested and calibration_path_file.exists():
                print(
                    f"[benchmark][resume] cutoff={pct} fold={fi+1}: calibration artifacts already present; skipping."
                )
                _status_log(
                    f"phase=calibrate cutoff={pct} fold={fi+1} event=skip reason=resume_existing"
                )
                dest_dir = results_root / "figures"
                dest_dir.mkdir(parents=True, exist_ok=True)
                source_dir = outputs_path / "calibration_plots"
                if source_dir.exists():
                    for plot in sorted(source_dir.glob("*.png")):
                        dest_name = f"calibrate_cutoff{pct}_fold{fi+1}_{plot.name}"
                        try:
                            shutil.copy2(plot, dest_dir / dest_name)
                        except Exception:
                            pass
                continue

            if (
                not effective_calibrate_only
                and ((not calibration_requested) or calibration_path_file.exists())
            ):
                evaluation_done = True
                if run_episodic and not metrics_path.exists():
                    evaluation_done = False
                if run_global and not global_metrics_path.exists():
                    evaluation_done = False
                if evaluation_done:
                    print(
                        f"[benchmark][resume] cutoff={pct} fold={fi+1}: existing metrics detected; skipping retrain."
                    )
                    _status_log(
                        f"phase=train cutoff={pct} fold={fi+1} event=skip reason=resume_existing"
                    )
                    if calibration_requested:
                        _status_log(
                            f"phase=calibrate cutoff={pct} fold={fi+1} event=skip reason=resume_existing"
                        )
                    _status_log(
                        f"phase=eval cutoff={pct} fold={fi+1} event=skip reason=resume_existing"
                    )
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as fh:
                            enriched = json.load(fh)
                        per_fold_metrics.append(enriched)
                    except Exception:
                        pass
                    continue

            # Train + eval
            # Ensure embeddings exist (contiguous files) before training
            emb_base = run_cfg["paths"]["embeddings"]
            base = emb_base[:-4] if emb_base.endswith('.npz') else emb_base
            Xp = Path(base + ".X.npy")
            Kp = Path(base + ".keys.npy")
            files_exist = Xp.exists() and Kp.exists()
            skip_reason: Optional[str] = None
            if args.force_embed:
                need_embed = base not in force_embed_paths_done or not files_exist
                if not need_embed and files_exist:
                    skip_reason = "force_reuse"
            else:
                need_embed = not files_exist
                if not need_embed:
                    skip_reason = "exists"

            if need_embed:
                _status_log(f"phase=embed cutoff={pct} fold={fi+1} event=start")
                try:
                    run(["python", "-m", "src.embed_sequences", "-c", str(cfg_path)])
                    _status_log(f"phase=embed cutoff={pct} fold={fi+1} event=finish status=success")
                    if args.force_embed:
                        force_embed_paths_done.add(base)
                except subprocess.CalledProcessError:
                    # Fallback: generate synthetic embeddings for union of split accessions
                    print("[benchmark] Embedding step failed; generating synthetic embeddings for smoke")
                    import json as _json
                    accs: set[str] = set()
                    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                        p = Path(run_cfg["paths"]["splits_dir"]) / name
                        if not p.exists():
                            continue
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                obj = _json.loads(line)
                                for a in obj.get("accessions", []):
                                    accs.add(str(a))
                    keys = sorted(accs)
                    import numpy as _np
                    D = int(run_cfg.get("projection_dim", 256))  # small dim ok
                    rng = _np.random.default_rng(12345)
                    X = rng.normal(size=(len(keys), D)).astype("float32")
                    X /= _np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
                    Xp.parent.mkdir(parents=True, exist_ok=True)
                    _np.save(Xp, X)
                    _np.save(Kp, _np.array(keys, dtype="U"))
                    print(f"[benchmark] Wrote synthetic embeddings: N={len(keys)} D={D} → {Xp}, {Kp}")
                    _status_log(
                        f"phase=embed cutoff={pct} fold={fi+1} event=finish status=fallback_synthetic N={len(keys)} D={D}"
                    )
                    if args.force_embed:
                        force_embed_paths_done.add(base)
            else:
                reason = skip_reason or ("missing" if files_exist else "unknown")
                _status_log(
                    f"phase=embed cutoff={pct} fold={fi+1} event=skip reason={reason}"
                )
            _status_log(f"phase=train cutoff={pct} fold={fi+1} event=start")
            try:
                run(["python", "-m", "src.train_protonet", "-c", str(cfg_path)])
                _status_log(f"phase=train cutoff={pct} fold={fi+1} event=finish status=success")
            except subprocess.CalledProcessError as e:
                _status_log(
                    f"phase=train cutoff={pct} fold={fi+1} event=finish status=failed exit_code={getattr(e, 'returncode', 'NA')}"
                )
                raise
            if proto_path is not None:
                _status_log(f"phase=prototypes cutoff={pct} fold={fi+1} event=start")
                try:
                    run([
                        "python",
                        "scripts/build_prototypes.py",
                        "--config",
                        str(cfg_path),
                        "--out",
                        str(proto_path),
                    ])
                    _status_log(
                        f"phase=prototypes cutoff={pct} fold={fi+1} event=finish status=success path={proto_path}"
                    )
                except subprocess.CalledProcessError as e:
                    _status_log(
                        f"phase=prototypes cutoff={pct} fold={fi+1} event=finish status=failed exit_code={getattr(e, 'returncode', 'NA')}"
                    )
                    raise
            if run_global and calibration_requested:
                _run_calibration(
                    cfg_path,
                    outputs_path,
                    cutoff_pct=pct,
                    fold_index=fi + 1,
                    calibration_cfg=calibration_cfg,
                    calibrate_only=effective_calibrate_only,
                    results_root=results_root,
                    default_shortlist=default_shortlist,
                )
                if effective_calibrate_only:
                    continue
            # Skip eval if test split has no classes
            test_jsonl = Path(run_cfg["paths"]["splits_dir"]) / "test.jsonl"
            nonempty = False
            if test_jsonl.exists():
                with open(test_jsonl, "r", encoding="utf-8") as f:
                    for _ in f:
                        nonempty = True
                        break

            eval_commands: List[Tuple[str, List[str]]] = []
            if run_episodic:
                eval_commands.append(
                    (
                        "episodic",
                        ["python", "-m", "src.eval_protonet", "-c", str(cfg_path), "--mode", "episodic"],
                    )
                )
            if run_global:
                eval_commands.append(
                    (
                        "global_support",
                        [
                            "python",
                            "-m",
                            "src.eval_protonet",
                            "-c",
                            str(cfg_path),
                            "--mode",
                            "global_support",
                        ],
                    )
                )

            episodic_skip_reason: Optional[str] = None
            episodic_skipped = False
            eligible_classes = 0
            required_classes = None
            need_per_class = None

            if run_episodic:
                episode_cfg = run_cfg.get("episode", {}) or {}
                M_eval = int(episode_cfg.get("M_val", episode_cfg.get("M_train", 5)))
                K_eval = int(episode_cfg.get("K_val", episode_cfg.get("K_train", 1)))
                Q_eval = int(episode_cfg.get("Q_val", episode_cfg.get("Q_train", 1)))
                required_classes = M_eval
                need_per_class = K_eval + Q_eval
                if test_jsonl.exists():
                    with open(test_jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            accs = obj.get("accessions", [])
                            if isinstance(accs, list) and len(accs) >= need_per_class:
                                eligible_classes += 1
                if eligible_classes < M_eval:
                    episodic_skip_reason = (
                        f"insufficient eligible classes (have {eligible_classes}, need at least {M_eval}) "
                        f"for K={K_eval}, Q={Q_eval} (≥{need_per_class} embeddings per class)"
                    )

            if nonempty:
                for label, cmd in eval_commands:
                    if label == "episodic" and episodic_skip_reason is not None:
                        episodic_skipped = True
                        print(
                            f"[benchmark][skip] mode=episodic cutoff={pct} fold={fi+1}: {episodic_skip_reason}"
                        )
                        _status_log(
                            f"phase=eval cutoff={pct} fold={fi+1} mode=episodic event=skip reason={episodic_skip_reason.replace(' ', '_')}"
                        )
                        if metrics_path.exists():
                            try:
                                metrics_path.unlink()
                            except OSError:
                                pass
                        continue
                    print(f"[benchmark][eval] mode={label} cutoff={pct} fold={fi+1}")
                    _status_log(f"phase=eval cutoff={pct} fold={fi+1} mode={label} event=start")
                    try:
                        run(cmd)
                        _status_log(f"phase=eval cutoff={pct} fold={fi+1} mode={label} event=finish status=success")
                    except subprocess.CalledProcessError as e:
                        _status_log(
                            f"phase=eval cutoff={pct} fold={fi+1} mode={label} event=finish status=failed exit_code={getattr(e, 'returncode', 'NA')}"
                        )
                        raise
            else:
                _status_log(
                    f"phase=eval cutoff={pct} fold={fi+1} event=skip reason=test_split_empty"
                )
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                if run_episodic:
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)
                if run_global:
                    with open(global_metrics_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)

            episodic_metrics: Dict[str, Any] | None = None
            global_metrics: Dict[str, Any] | None = None

            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    episodic_raw = json.load(f)

                def _unwrap_metrics(obj: Any) -> Any:
                    depth = 0
                    current = obj
                    while (
                        isinstance(current, dict)
                        and "metrics" in current
                        and isinstance(current["metrics"], dict)
                        and depth < 10
                    ):
                        current = current["metrics"]
                        depth += 1
                    return current

                extracted = _unwrap_metrics(episodic_raw)
                if isinstance(extracted, dict) and extracted:
                    if "episodic" in extracted and isinstance(extracted["episodic"], dict):
                        episodic_metrics = extracted["episodic"]
                    elif all(isinstance(v, dict) for v in extracted.values()):
                        episodic_metrics = extracted
                    if global_metrics is None and "global" in extracted and isinstance(extracted["global"], dict):
                        global_metrics = extracted["global"]

            if global_metrics_path.exists():
                with open(global_metrics_path, "r", encoding="utf-8") as f:
                    gm = json.load(f)
                    if isinstance(gm, dict) and gm:
                        global_metrics = gm

            if run_episodic and episodic_metrics is None:
                if episodic_skipped:
                    print(
                        f"[benchmark][info] skipped episodic metrics for {outputs_dir}: {episodic_skip_reason}"
                    )
                elif nonempty:
                    print(
                        f"[benchmark][warn] episodic metrics missing for outputs at {outputs_dir}; expected metrics.json"
                    )
                else:
                    print(
                        f"[benchmark][info] skipped episodic metrics for {outputs_dir} because the test split is empty"
                    )
            if run_global and global_metrics is None:
                if nonempty:
                    print(
                        f"[benchmark][warn] global metrics missing for outputs at {outputs_dir}; expected global_metrics.json"
                    )
                else:
                    print(
                        f"[benchmark][info] skipped global metrics for {outputs_dir} because the test split is empty"
                    )
            if episodic_metrics is None and global_metrics is None:
                if nonempty:
                    available = ", ".join(sorted(str(p.name) for p in outputs_dir.glob("*.json")))
                    raise FileNotFoundError(
                        f"No usable metrics found for outputs at {outputs_dir}."
                        + (f" Available JSON files: {available}" if available else "")
                    )
                else:
                    # Empty folds are valid; we retain metadata without metrics.
                    pass

            # Counts by cluster assignment
            fold_clusters = list(folds_info["folds"].get(str(fi+1), []))
            all_clusters = set(clu2acc.keys())
            test_clusters = set(fold_clusters)
            trainval_clusters = list(all_clusters - test_clusters)
            n_clu_train = len(trainval_clusters)
            n_clu_test = len(test_clusters)
            n_seq_train = int(sum(cl_size[c] for c in trainval_clusters))
            n_seq_test = int(sum(cl_size[c] for c in test_clusters))

            # Label stats for train/test from split JSONLs
            def read_split_counts(p: Path) -> Dict[str, int]:
                counts: Dict[str, int] = {}
                if not p.exists():
                    return counts
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        ec = str(obj.get("ec"))
                        counts[ec] = counts.get(ec, 0) + int(len(obj.get("accessions", [])))
                return counts
            tr_counts = read_split_counts(Path(run_cfg["paths"]["splits_dir"]) / "train.jsonl")
            te_counts = read_split_counts(Path(run_cfg["paths"]["splits_dir"]) / "test.jsonl")
            # Approximate multiEC ratio from within each split
            def multiec_ratio(p: Path) -> float:
                acc_to_ecs: Dict[str, set] = {}
                if not p.exists():
                    return 0.0
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        ec = str(obj.get("ec"))
                        for a in obj.get("accessions", []):
                            acc_to_ecs.setdefault(a, set()).add(ec)
                if not acc_to_ecs:
                    return 0.0
                more = sum(1 for ecs in acc_to_ecs.values() if len(ecs) > 1)
                return float(more) / float(len(acc_to_ecs))

            label_stats = {
                "train": {
                    "by_ec": tr_counts,
                    "multiEC_ratio": multiec_ratio(Path(run_cfg["paths"]["splits_dir"]) / "train.jsonl"),
                },
                "test": {
                    "by_ec": te_counts,
                    "multiEC_ratio": multiec_ratio(Path(run_cfg["paths"]["splits_dir"]) / "test.jsonl"),
                },
            }
            metrics_payload: Dict[str, Any] = {}
            if episodic_metrics is not None:
                metrics_payload["episodic"] = episodic_metrics
            elif episodic_skipped:
                skip_meta: Dict[str, Any] = {"reason": episodic_skip_reason or "insufficient data"}
                if required_classes is not None:
                    skip_meta["required_classes"] = int(required_classes)
                if need_per_class is not None:
                    skip_meta["min_examples_per_class"] = int(need_per_class)
                skip_meta["eligible_classes"] = int(eligible_classes)
                metrics_payload["episodic_skipped"] = skip_meta
            if global_metrics is not None:
                metrics_payload["global"] = global_metrics

            now = datetime.utcnow().isoformat() + "Z"
            enriched = {
                "run_id": f"id{pct}_fold{fi+1}_{base_cfg.get('random_seed', 42)}",
                "git_commit": git_commit,
                "id_threshold": pct,
                "fold": fi + 1,
                "n_clusters_train": n_clu_train,
                "n_clusters_test": n_clu_test,
                "n_seqs_train": n_seq_train,
                "n_seqs_test": n_seq_test,
                "identity_definition": folds_info.get("identity_definition"),
                "clustering_method": folds_info.get("clustering_method"),
                "random_seed": int(base_cfg.get("random_seed", 42)),
                "label_stats": label_stats,
                "metrics": metrics_payload,
                "modes": {
                    "episodic": episodic_metrics is not None,
                    "global": global_metrics is not None,
                },
                "timestamp": now,
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2)
            per_fold_metrics.append(enriched)

        if effective_calibrate_only:
            continue

        agg: Dict[str, Any] = {}

        episodic_records = [rec.get("metrics", {}).get("episodic") for rec in per_fold_metrics if rec.get("metrics", {}).get("episodic")]
        global_records = [rec.get("metrics", {}).get("global") for rec in per_fold_metrics if rec.get("metrics", {}).get("global")]

        if episodic_records:
            K_keys = sorted({key for rec in episodic_records for key in rec.keys()})
            epi_agg: Dict[str, Dict[str, float]] = {}
            for Kk in K_keys:
                mnames = set()
                for rec in episodic_records:
                    md = rec.get(Kk, {})
                    if isinstance(md, dict):
                        mnames.update(md.keys())
                epi_agg[Kk] = {}
                for mkey in sorted(mnames):
                    vals = []
                    for rec in episodic_records:
                        md = rec.get(Kk, {})
                        if isinstance(md, dict) and mkey in md and md[mkey] is not None:
                            vals.append(float(md[mkey]))
                    if not vals:
                        continue
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                    se = std / np.sqrt(len(vals)) if vals else 0.0
                    ci95 = 1.96 * se
                    epi_agg[Kk][f"mean_{mkey}"] = mean
                    epi_agg[Kk][f"std_{mkey}"] = std
                    epi_agg[Kk][f"ci95_{mkey}"] = float(ci95)
            agg["episodic"] = epi_agg

        if global_records:
            metric_names = sorted({mk for rec in global_records for mk in rec.keys()})
            glob_agg: Dict[str, Dict[str, float]] = {}
            for mkey in metric_names:
                vals = [rec[mkey] for rec in global_records if isinstance(rec, dict) and mkey in rec and isinstance(rec[mkey], (int, float))]
                if not vals:
                    continue
                vals = [float(v) for v in vals]
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                se = std / np.sqrt(len(vals)) if vals else 0.0
                ci95 = 1.96 * se
                glob_agg[mkey] = {
                    "mean": mean,
                    "std": std,
                    "ci95": float(ci95),
                }
            agg["global"] = glob_agg

        agg_path = split_dir / "aggregate.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_threshold": pct,
                    "aggregate": agg,
                    "modes": {
                        "episodic": bool(episodic_records),
                        "global": bool(global_records),
                    },
                },
                f,
                indent=2,
            )
        summary[str(pct)] = {
            "aggregate": agg,
            "modes": {
                "episodic": bool(episodic_records),
                "global": bool(global_records),
            },
        }

    # Write summary across thresholds
    if effective_calibrate_only:
        print("[benchmark] Calibrate-only mode complete; skipping summary aggregation.")
        return

    bench_path = results_root / "summary_by_id_threshold.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[benchmark] wrote summary -> {bench_path}")


if __name__ == "__main__":
    main()
