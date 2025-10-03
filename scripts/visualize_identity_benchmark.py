#!/usr/bin/env python3
"""Visualize multi-threshold, 5-fold results written under results/.

Generates publication-style plots, a bundled PDF report, and prints a quick textual summary:
- Line plots vs. identity threshold (mean ± 95% CI) for key metrics (K=1)
- Boxplots across folds for Top-1 accuracy (K=1)
- Bar plot of number of clusters vs. threshold

Usage:
  python scripts/visualize_identity_benchmark.py \
      --results_dir results \
      --out_dir results/figures \
      [--no-pdf] [--pdf_path results/figures/identity_benchmark_report.pdf]

Notes:
- Expects results/summary_by_id_threshold.json and split-XX directories
  (XX in {10,30,50,70,100} by default) with per-fold metrics.json files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _try_import_matplotlib():
    try:
        import matplotlib
        # Use non-interactive backend by default for headless environments
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Please install it, e.g.\n"
            "  pip install matplotlib\n"
            f"Import error: {e}"
        )


def load_summary(summary_path: Path) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_thresholds(summary: Dict) -> List[int]:
    ths = [int(k) for k in summary.keys() if k.isdigit()]
    ths.sort()
    return ths


def extract_metric(summary: Dict, metric_key: str, K_key: str = "K=1") -> Tuple[List[int], List[float], List[float]]:
    """Return (thresholds, mean_vals, ci95_vals) for the given nested metric."""
    ths = list_thresholds(summary)
    means, cis = [], []
    for t in ths:
        tkey = str(t)
        agg_all = summary[tkey].get("aggregate", {})
        epi = agg_all.get("episodic", agg_all)
        block = epi.get(K_key)
        if not isinstance(block, dict):
            means.append(float("nan"))
            cis.append(0.0)
            continue
        means.append(float(block.get(f"mean_{metric_key}", float("nan"))))
        cis.append(float(block.get(f"ci95_{metric_key}", 0.0)))
    return ths, means, cis


def extract_global_metric(summary: Dict, metric_key: str) -> Tuple[List[int], List[float], List[float], bool]:
    ths = list_thresholds(summary)
    means, cis = [], []
    any_available = False
    for t in ths:
        tkey = str(t)
        agg_all = summary[tkey].get("aggregate", {})
        glob = agg_all.get("global", {})
        entry = glob.get(metric_key)
        if isinstance(entry, dict) and entry:
            means.append(float(entry.get("mean", float("nan"))))
            cis.append(float(entry.get("ci95", 0.0)))
            any_available = True
        else:
            means.append(float("nan"))
            cis.append(0.0)
    return ths, means, cis, any_available


def count_clusters(tsv_path: Path) -> int:
    """Count unique cluster representatives (second column) in clusters.tsv."""
    reps = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("	")
            if len(parts) >= 2:
                reps.add(parts[1])
    return len(reps)


def collect_cluster_counts(results_dir: Path, thresholds: List[int]) -> List[int]:
    counts: List[int] = []
    for t in thresholds:
        tdir = results_dir / f"split-{t}"
        ctsv = tdir / "clusters.tsv"
        counts.append(count_clusters(ctsv) if ctsv.exists() else 0)
    return counts


def collect_per_fold_metric(
    results_dir: Path,
    thresholds: List[int],
    metric_key: str,
    K_key: str = "K=1",
) -> Dict[int, List[float]]:
    """Gather per-fold values for metric_key from fold-*/metrics.json for the requested K."""
    values: Dict[int, List[float]] = {t: [] for t in thresholds}
    for t in thresholds:
        base = results_dir / f"split-{t}"
        for fold_dir in sorted(base.glob("fold-*")):
            mpath = fold_dir / "metrics.json"
            if not mpath.exists():
                continue
            try:
                obj = json.loads(mpath.read_text())
            except Exception:
                continue
            met = obj.get("metrics", {})
            epi = met.get("episodic", met)
            block = epi.get(K_key)
            if isinstance(block, dict):
                val = block.get(metric_key)
                if val is not None:
                    values[t].append(float(val))
    return values


def plot_line_with_ci(ax, x: List[int], y: List[float], ci: List[float], label: str, color: str):
    import numpy as _np
    x = _np.asarray(x)
    y = _np.asarray(y)
    ci = _np.asarray(ci)
    mask = ~_np.isnan(y)
    if not mask.any():
        return False
    x = x[mask]
    y = y[mask]
    ci = ci[mask]
    ax.plot(x, y, marker="o", label=label, color=color)
    ax.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)
    return True


def single_boxplot(ax, thresholds: List[int], data: Dict[int, List[float]], label: str = "K=1", color: str = "#4C78A8"):
    import numpy as _np

    ths = _np.asarray(thresholds)
    positions = ths.astype(float)
    series = [data[t] for t in thresholds]

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=2.8,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.6),
        medianprops=dict(color="#1F3555"),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(markerfacecolor=color, markeredgecolor=color, alpha=0.4),
    )

    ax.set_xticks(ths)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlim(ths.min() - 3, ths.max() + 3)
    ax.legend([bp["boxes"][0]], [label], loc="best")


def plot_single_bar(ax, x_label: str, mean: float, ci95: float, *, color: str, title: str | None, ylabel: str, ylim: tuple[float, float] | None = None):
    # Draw a single bar with symmetric 95% CI error bar
    xs = [x_label]
    vals = [mean]
    errs = [ci95]
    ax.bar(xs, vals, yerr=errs, color=color, alpha=0.85, capsize=6)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def describe_monotonic_trend(values: List[int]) -> str:
    """Describe the monotonic trend of a sequence as a short word."""
    if not values or len(values) < 2:
        return "stable"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    pos = any(d > 0 for d in diffs)
    neg = any(d < 0 for d in diffs)
    if pos and not neg:
        return "increases"
    if neg and not pos:
        return "decreases"
    if not pos and not neg:
        return "stable"
    return "varies"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="Path to results root")
    ap.add_argument("--out_dir", default="results/figures", help="Where to save figures")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-pdf", action="store_true", help="Disable generating a bundled PDF report")
    ap.add_argument("--pdf_path", default=None, help="Override PDF output path (default: out_dir/identity_benchmark_report.pdf)")
    ap.add_argument("--K", type=int, default=None, help="Episodic K to visualize; auto-detect if omitted")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary_by_id_threshold.json"
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}. Run the benchmark first.")

    summary = load_summary(summary_path)
    thresholds = list_thresholds(summary)

    # Determine which episodic K to use
    def _available_Ks_from_summary(s: Dict) -> List[int]:
        ks: List[int] = []
        for t in list_thresholds(s):
            epi = s.get(str(t), {}).get("aggregate", {}).get("episodic", {})
            for key in epi.keys():
                if isinstance(key, str) and key.startswith("K="):
                    try:
                        kint = int(key.split("=", 1)[1])
                        if kint not in ks:
                            ks.append(kint)
                    except Exception:
                        pass
        ks.sort()
        return ks

    available_Ks = _available_Ks_from_summary(summary)

    def _read_K_from_cfg(cfg_path: Path) -> int | None:
        try:
            import yaml  # type: ignore
        except Exception:
            return None
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return None
        epi = cfg.get("episode", {}) or {}
        # Prefer K_eval if present
        k_eval = epi.get("K_eval")
        if isinstance(k_eval, list) and k_eval:
            try:
                return int(k_eval[0])
            except Exception:
                pass
        if isinstance(k_eval, int):
            return int(k_eval)
        # Fallback to K_val
        k_val = epi.get("K_val")
        if isinstance(k_val, int):
            return int(k_val)
        try:
            return int(k_val) if k_val is not None else None
        except Exception:
            return None

    # 1) CLI override
    chosen_K: int | None = args.K if isinstance(args.K, int) else None
    # 2) Try last run's config
    if chosen_K is None:
        last_cfg = results_dir / "lastrun" / "config.yaml"
        chosen_K = _read_K_from_cfg(last_cfg)
    # 3) If not in available, fallback to available set
    if available_Ks:
        if chosen_K is None or chosen_K not in available_Ks:
            # Use the only or the smallest available K
            chosen_K = available_Ks[0]
    # Final guard
    chosen_K = int(chosen_K or 1)
    K_key = f"K={chosen_K}"
    K_label_str = f"K={chosen_K}"

    plt = _try_import_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    figs: List[object] = []
    is_multi = len(thresholds) >= 2
    single_label = str(thresholds[0]) if thresholds else "NA"

    # 1) Top-1 accuracy vs threshold (episodic, chosen K)
    ths, acc_vals, acc_ci = extract_metric(summary, "acc_top1_hit", K_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        plotted = plot_line_with_ci(ax, ths, acc_vals, acc_ci, label=K_label_str, color="#4C78A8")
        ax.set_title("Top-1 Accuracy vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Accuracy (mean ± 95% CI)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    else:
        val = acc_vals[0] if acc_vals else float("nan")
        ci = acc_ci[0] if acc_ci else 0.0
        plot_single_bar(
            ax,
            x_label=f"{single_label}%",
            mean=val,
            ci95=ci,
            color="#4C78A8",
            title=f"Top-1 Accuracy @ {single_label}%",
            ylabel="Accuracy (mean ± 95% CI)",
            ylim=(0.0, 1.05),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 2) Macro-F1 vs threshold (episodic, chosen K)
    ths, macro_f1, macro_f1_ci = extract_metric(summary, "macro_f1", K_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        plotted = plot_line_with_ci(ax, ths, macro_f1, macro_f1_ci, label=f"Macro-F1 {K_label_str}", color="#4C78A8")
        ax.set_title("Macro-F1 vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Macro-F1 (mean ± 95% CI)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    else:
        val = macro_f1[0] if macro_f1 else float("nan")
        ci = macro_f1_ci[0] if macro_f1_ci else 0.0
        plot_single_bar(
            ax,
            x_label=f"{single_label}%",
            mean=val,
            ci95=ci,
            color="#4C78A8",
            title=f"Macro-F1 @ {single_label}%",
            ylabel="Macro-F1 (mean ± 95% CI)",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 3) Micro-F1 vs threshold (episodic, chosen K)
    ths, micro_f1, micro_f1_ci = extract_metric(summary, "micro_f1", K_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        plotted = plot_line_with_ci(ax, ths, micro_f1, micro_f1_ci, label=f"Micro-F1 {K_label_str}", color="#4C78A8")
        ax.set_title("Micro-F1 vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Micro-F1 (mean ± 95% CI)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    else:
        val = micro_f1[0] if micro_f1 else float("nan")
        ci = micro_f1_ci[0] if micro_f1_ci else 0.0
        plot_single_bar(
            ax,
            x_label=f"{single_label}%",
            mean=val,
            ci95=ci,
            color="#4C78A8",
            title=f"Micro-F1 @ {single_label}%",
            ylabel="Micro-F1 (mean ± 95% CI)",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "micro_f1_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 4) Global metrics vs threshold
    g_ths, g_micro, g_micro_ci, has_micro = extract_global_metric(summary, "micro_f1")
    _, g_macro, g_macro_ci, has_macro = extract_global_metric(summary, "macro_f1")
    _, g_top1, g_top1_ci, has_top1 = extract_global_metric(summary, "acc_top1_hit")
    if has_micro or has_macro:
        fig, ax = plt.subplots(figsize=(7, 4))
        if is_multi:
            plotted_any = False
            if has_micro:
                plotted_any |= bool(plot_line_with_ci(ax, g_ths, g_micro, g_micro_ci, label="Global Micro-F1", color="#54A24B"))
            if has_macro:
                plotted_any |= bool(plot_line_with_ci(ax, g_ths, g_macro, g_macro_ci, label="Global Macro-F1", color="#E45756"))
            ax.set_title("Global Support F1 vs. Identity Threshold")
            ax.set_xlabel("Identity threshold (%)")
            ax.set_ylabel("F1 (mean ± 95% CI)")
            ax.grid(True, alpha=0.3)
            if plotted_any:
                ax.legend()
        else:
            # Single-threshold: avoid overlapping titles when using twinx
            title_primary = None
            if has_micro and has_macro:
                title_primary = f"Global Support F1 @ {single_label}%"
            elif has_micro:
                title_primary = f"Global Micro-F1 @ {single_label}%"
            elif has_macro:
                title_primary = f"Global Macro-F1 @ {single_label}%"

            if has_micro:
                plot_single_bar(
                    ax,
                    x_label=f"{single_label}%",
                    mean=(g_micro[0] if g_micro else float("nan")),
                    ci95=(g_micro_ci[0] if g_micro_ci else 0.0),
                    color="#54A24B",
                    title=title_primary if not has_macro else None,
                    ylabel="F1 (mean ± 95% CI)",
                )
            if has_macro:
                ax2 = ax.twinx() if has_micro else ax
                plot_single_bar(
                    ax2,
                    x_label=f"{single_label}%",
                    mean=(g_macro[0] if g_macro else float("nan")),
                    ci95=(g_macro_ci[0] if g_macro_ci else 0.0),
                    color="#E45756",
                    title=None if has_micro else title_primary,
                    ylabel="F1 (mean ± 95% CI)",
                )
            if has_micro and has_macro:
                ax.set_title(title_primary)
        fig.tight_layout()
        fig.savefig(out_dir / "global_f1_vs_threshold.png", dpi=args.dpi)
        figs.append(fig)

    if has_top1:
        fig, ax = plt.subplots(figsize=(7, 4))
        if is_multi:
            plotted_any = bool(plot_line_with_ci(ax, g_ths, g_top1, g_top1_ci, label="Global Top-1", color="#72B7B2"))
            ax.set_title("Global Top-1 Hit vs. Identity Threshold")
            ax.set_xlabel("Identity threshold (%)")
            ax.set_ylabel("Top-1 hit (mean ± 95% CI)")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if plotted_any:
                ax.legend()
        else:
            plot_single_bar(
                ax,
                x_label=f"{single_label}%",
                mean=(g_top1[0] if g_top1 else float("nan")),
                ci95=(g_top1_ci[0] if g_top1_ci else 0.0),
                color="#72B7B2",
                title=f"Global Top-1 Hit @ {single_label}%",
                ylabel="Top-1 hit (mean ± 95% CI)",
                ylim=(0.0, 1.0),
            )
        fig.tight_layout()
        fig.savefig(out_dir / "global_top1_vs_threshold.png", dpi=args.dpi)
        figs.append(fig)

    # 5) Precision and Recall vs threshold (episodic, chosen K)
    ths, macro_precision, macro_precision_ci = extract_metric(summary, "macro_precision", K_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        plotted = plot_line_with_ci(ax, ths, macro_precision, macro_precision_ci, label=f"Macro-Precision {K_label_str}", color="#4C78A8")
        ax.set_title("Macro-Precision vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Macro-Precision (mean ± 95% CI)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    else:
        val = macro_precision[0] if macro_precision else float("nan")
        ci = macro_precision_ci[0] if macro_precision_ci else 0.0
        plot_single_bar(
            ax,
            x_label=f"{single_label}%",
            mean=val,
            ci95=ci,
            color="#4C78A8",
            title=f"Macro-Precision @ {single_label}%",
            ylabel="Macro-Precision (mean ± 95% CI)",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "macro_precision_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    coverage_ths, coverage_means, coverage_ci, has_coverage = extract_global_metric(summary, "coverage_ratio")
    if has_coverage:
        fig, ax = plt.subplots(figsize=(7, 4))
        if is_multi:
            plotted_any = bool(plot_line_with_ci(ax, coverage_ths, coverage_means, coverage_ci, label="Coverage", color="#B279A2"))
            ax.set_title("Global Support Coverage vs. Identity Threshold")
            ax.set_xlabel("Identity threshold (%)")
            ax.set_ylabel("Coverage ratio (evaluated queries / total)")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            if plotted_any:
                ax.legend()
        else:
            plot_single_bar(
                ax,
                x_label=f"{single_label}%",
                mean=(coverage_means[0] if coverage_means else float("nan")),
                ci95=(coverage_ci[0] if coverage_ci else 0.0),
                color="#B279A2",
                title=f"Global Support Coverage @ {single_label}%",
                ylabel="Coverage ratio (evaluated queries / total)",
                ylim=(0.0, 1.05),
            )
        fig.tight_layout()
        fig.savefig(out_dir / "global_coverage_vs_threshold.png", dpi=args.dpi)
        figs.append(fig)

    ths, macro_recall, macro_recall_ci = extract_metric(summary, "macro_recall", K_key)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        plotted = plot_line_with_ci(ax, ths, macro_recall, macro_recall_ci, label=f"Macro-Recall {K_label_str}", color="#4C78A8")
        ax.set_title("Macro-Recall vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Macro-Recall (mean ± 95% CI)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
    else:
        val = macro_recall[0] if macro_recall else float("nan")
        ci = macro_recall_ci[0] if macro_recall_ci else 0.0
        plot_single_bar(
            ax,
            x_label=f"{single_label}%",
            mean=val,
            ci95=ci,
            color="#4C78A8",
            title=f"Macro-Recall @ {single_label}%",
            ylabel="Macro-Recall (mean ± 95% CI)",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "macro_recall_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 6) Cluster counts vs threshold
    cluster_counts = collect_cluster_counts(results_dir, thresholds)
    cluster_trend = describe_monotonic_trend(cluster_counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_multi:
        ax.bar([str(t) for t in thresholds], cluster_counts, color="#72B7B2")
        ax.set_title("#Clusters vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("#Unique clusters (dataset)")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.bar([f"{single_label}%"], [cluster_counts[0] if cluster_counts else 0], color="#72B7B2")
        ax.set_title(f"#Clusters @ {single_label}%")
        ax.set_ylabel("#Unique clusters (dataset)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clusters_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 7) Per-fold accuracy distribution (boxplots)
    per_fold_acc = collect_per_fold_metric(results_dir, thresholds, "acc_top1_hit", K_key)
    fig, ax = plt.subplots(figsize=(8, 4))
    single_boxplot(ax, thresholds, per_fold_acc, label=K_label_str, color="#4C78A8")
    if is_multi:
        ax.set_title(f"Per-fold Top-1 Accuracy by Threshold ({K_label_str})")
        ax.set_xlabel("Identity threshold (%)")
    else:
        ax.set_title(f"Per-fold Top-1 Accuracy ({K_label_str} @ {single_label}%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_boxplot_by_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 8) Print a short textual summary for quick sense-making
    def _clean(values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[~np.isnan(arr)]

    def describe_single(name: str, values: List[float], fmt: str = "{:.3f}") -> str:
        arr = _clean(values)
        if arr.size == 0:
            return f"- {name}: unavailable"
        return (
            f"- {name}: mean {fmt.format(arr.mean())} "
            f"(min {fmt.format(arr.min())}, max {fmt.format(arr.max())})"
        )

    def describe_global(name: str, values: List[float], fmt: str = "{:.3f}") -> str:
        arr = _clean(values)
        if arr.size == 0:
            return f"- {name}: unavailable"
        return (
            f"- {name}: mean {fmt.format(arr.mean())} "
            f"(min {fmt.format(arr.min())}, max {fmt.format(arr.max())})"
        )

    print("\n[visualize] Summary:" + (f" ({K_label_str}, means across thresholds)" if is_multi else f" ({K_label_str} @ {single_label}%)"))
    if is_multi:
        s_acc = describe_single("Top-1 accuracy", acc_vals)
        s_macro = describe_single("Macro-F1", macro_f1)
        s_micro = describe_single("Micro-F1", micro_f1)
        s_precision = describe_single("Macro-Precision", macro_precision)
        s_recall = describe_single("Macro-Recall", macro_recall)
        s_clu = f"- #Clusters {cluster_trend} with threshold: " + " → ".join(str(c) for c in cluster_counts)
        s_gmicro = describe_global("Global micro-F1", g_micro)
        s_gmacro = describe_global("Global macro-F1", g_macro)
        s_gtop1 = describe_global("Global top-1", g_top1)
        s_cov = describe_global("Global coverage", coverage_means, fmt="{:.1%}")
    else:
        def fmt_value(name: str, vals: List[float], cis: List[float], pct: bool = False) -> str:
            if not vals:
                return f"- {name} @ {single_label}%: unavailable"
            v = vals[0]
            c = (cis[0] if cis else 0.0)
            if np.isnan(v):
                return f"- {name} @ {single_label}%: unavailable"
            if pct:
                return f"- {name} @ {single_label}%: {v:.1%} ± {c:.1%}"
            return f"- {name} @ {single_label}%: {v:.3f} ± {c:.3f}"

        s_acc = fmt_value("Top-1 accuracy", acc_vals, acc_ci)
        s_macro = fmt_value("Macro-F1", macro_f1, macro_f1_ci)
        s_micro = fmt_value("Micro-F1", micro_f1, micro_f1_ci)
        s_precision = fmt_value("Macro-Precision", macro_precision, macro_precision_ci)
        s_recall = fmt_value("Macro-Recall", macro_recall, macro_recall_ci)
        s_clu = f"- #Clusters @ {single_label}%: {cluster_counts[0] if cluster_counts else 'NA'}"
        s_gmicro = fmt_value("Global micro-F1", g_micro, g_micro_ci)
        s_gmacro = fmt_value("Global macro-F1", g_macro, g_macro_ci)
        s_gtop1 = fmt_value("Global top-1", g_top1, g_top1_ci)
        s_cov = fmt_value("Global coverage", coverage_means, coverage_ci, pct=True)
    for line in (s_acc, s_macro, s_micro, s_precision, s_recall, s_gmicro, s_gmacro, s_gtop1, s_cov, s_clu):
        print(line)
    print(f"[visualize] Wrote figures to: {out_dir.resolve()}")

    # 9) Optional PDF report bundling: title + summary page + all plots
    if not args.no_pdf:
        pdf_path = Path(args.pdf_path) if args.pdf_path else (out_dir / "identity_benchmark_report.pdf")
        fig_sum = plt.figure(figsize=(8.5, 11))
        fig_sum.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
        fig_sum.suptitle("Identity Benchmark Report", fontsize=16, fontweight="bold", y=0.97)
        text_lines: List[str] = []
        text_lines.append(f"Thresholds (%): {', '.join(str(t) for t in thresholds)}")
        text_lines.append("")
        text_lines.append(s_acc)
        text_lines.append(s_macro)
        text_lines.append(s_micro)
        text_lines.append(s_precision)
        text_lines.append(s_recall)
        text_lines.append("")
        text_lines.append(s_gmicro)
        text_lines.append(s_gmacro)
        text_lines.append(s_gtop1)
        text_lines.append(s_cov)
        text_lines.append(s_clu)
        body = "\n".join(text_lines)
        fig_sum.text(0.06, 0.90, "Overview", fontsize=12, fontweight="bold", va="top")
        fig_sum.text(0.06, 0.87, body, fontsize=10, va="top")

        with PdfPages(str(pdf_path)) as pdf:
            pdf.savefig(fig_sum)
            for f in figs:
                pdf.savefig(f)
        import matplotlib.pyplot as _plt
        _plt.close(fig_sum)
        for f in figs:
            _plt.close(f)
        print(f"[visualize] Wrote PDF report → {pdf_path}")


if __name__ == "__main__":
    main()
