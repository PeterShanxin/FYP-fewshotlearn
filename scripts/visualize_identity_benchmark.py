#!/usr/bin/env python3
"""Visualize multi-threshold, 5-fold results written under results/.

Generates publication-style plots, a bundled PDF report, and prints a quick textual summary:
- Line plots vs. identity threshold (mean ± 95% CI) for key metrics
- Grouped boxplots across folds for Top-1 accuracy (K=1 vs K=5)
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
    """Return (thresholds, mean_vals, ci95_vals) for the given nested metric.

    metric_key examples: "acc_top1_hit", "macro_f1", "micro_f1", "macro_precision", "macro_recall".
    """
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
            parts = line.split("\t")
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


def collect_per_fold_metric(results_dir: Path, thresholds: List[int], metric_key: str) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """Gather per-fold values for metric_key from fold-*/metrics.json for K=1 and K=5.

    Returns: (values_k1, values_k5) where each is a dict threshold->list of fold values.
    """
    k1: Dict[int, List[float]] = {t: [] for t in thresholds}
    k5: Dict[int, List[float]] = {t: [] for t in thresholds}
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
            if "K=1" in epi:
                v = epi["K=1"].get(metric_key)
                if v is not None:
                    k1[t].append(float(v))
            if "K=5" in epi:
                v = epi["K=5"].get(metric_key)
                if v is not None:
                    k5[t].append(float(v))
    return k1, k5


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


def grouped_boxplot(ax, thresholds: List[int], data_a: Dict[int, List[float]], data_b: Dict[int, List[float]], labels=("K=1", "K=5")):
    import numpy as _np
    ths = _np.asarray(thresholds)
    width = 3  # x-offset for grouping
    positions_a = ths - width / 2
    positions_b = ths + width / 2

    # Prepare data in threshold order
    da = [data_a[t] for t in thresholds]
    db = [data_b[t] for t in thresholds]

    # Boxplots
    bp_a = ax.boxplot(
        da,
        positions=positions_a,
        widths=width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="#4C78A8", alpha=0.6),
        medianprops=dict(color="#1F3555"),
        whiskerprops=dict(color="#4C78A8"),
        capprops=dict(color="#4C78A8"),
        flierprops=dict(markerfacecolor="#4C78A8", markeredgecolor="#4C78A8", alpha=0.5),
    )
    bp_b = ax.boxplot(
        db,
        positions=positions_b,
        widths=width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="#F58518", alpha=0.6),
        medianprops=dict(color="#7A3E0C"),
        whiskerprops=dict(color="#F58518"),
        capprops=dict(color="#F58518"),
        flierprops=dict(markerfacecolor="#F58518", markeredgecolor="#F58518", alpha=0.5),
    )

    # X ticks on group centers
    ax.set_xticks(ths)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.legend([bp_a["boxes"][0], bp_b["boxes"][0]], labels, loc="best")


def describe_monotonic_trend(values: List[int]) -> str:
    """Describe the monotonic trend of a sequence as a short word.

    Returns one of: "increases", "decreases", "stable", "varies".
    """
    if not values or len(values) < 2:
        return "stable"
    diffs = [values[i+1] - values[i] for i in range(len(values) - 1)]
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
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary_by_id_threshold.json"
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}. Run the benchmark first.")

    summary = load_summary(summary_path)
    thresholds = list_thresholds(summary)

    # Load plotting backend lazily
    plt = _try_import_matplotlib()
    # Only import PdfPages after matplotlib is loaded
    from matplotlib.backends.backend_pdf import PdfPages

    figs: List[object] = []  # matplotlib Figure objects, saved to PDF later

    # 1) Accuracy vs threshold (K=1 & K=5)
    ths, acc1, ci1 = extract_metric(summary, "acc_top1_hit", "K=1")
    _, acc5, ci5 = extract_metric(summary, "acc_top1_hit", "K=5")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line_with_ci(ax, ths, acc1, ci1, label="K=1", color="#4C78A8")
    plot_line_with_ci(ax, ths, acc5, ci5, label="K=5", color="#F58518")
    ax.set_title("Top-1 Accuracy vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Accuracy (mean ± 95% CI)")
    ax.set_ylim(0.75, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 2) Macro-F1 and Micro-F1 vs threshold
    ths, mf1_1, mci1 = extract_metric(summary, "macro_f1", "K=1")
    _, mf1_5, mci5 = extract_metric(summary, "macro_f1", "K=5")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line_with_ci(ax, ths, mf1_1, mci1, label="Macro-F1 K=1", color="#4C78A8")
    plot_line_with_ci(ax, ths, mf1_5, mci5, label="Macro-F1 K=5", color="#F58518")
    ax.set_title("Macro-F1 vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Macro-F1 (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    ths, micf1_1, mici1 = extract_metric(summary, "micro_f1", "K=1")
    _, micf1_5, mici5 = extract_metric(summary, "micro_f1", "K=5")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line_with_ci(ax, ths, micf1_1, mici1, label="Micro-F1 K=1", color="#4C78A8")
    plot_line_with_ci(ax, ths, micf1_5, mici5, label="Micro-F1 K=5", color="#F58518")
    ax.set_title("Micro-F1 vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Micro-F1 (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "micro_f1_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    g_ths, g_micro, g_micro_ci, has_micro = extract_global_metric(summary, "micro_f1")
    _, g_macro, g_macro_ci, has_macro = extract_global_metric(summary, "macro_f1")
    if has_micro or has_macro:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        if has_micro:
            plotted |= bool(plot_line_with_ci(ax, g_ths, g_micro, g_micro_ci, label="Global Micro-F1", color="#54A24B"))
        if has_macro:
            plotted |= bool(plot_line_with_ci(ax, g_ths, g_macro, g_macro_ci, label="Global Macro-F1", color="#E45756"))
        ax.set_title("Global Support F1 vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("F1 (mean ± 95% CI)")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "global_f1_vs_threshold.png", dpi=args.dpi)
        figs.append(fig)

    # 3) Precision/Recall vs threshold (macro)
    ths, mp_1, mpci1 = extract_metric(summary, "macro_precision", "K=1")
    _, mp_5, mpci5 = extract_metric(summary, "macro_precision", "K=5")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line_with_ci(ax, ths, mp_1, mpci1, label="Macro-Precision K=1", color="#4C78A8")
    plot_line_with_ci(ax, ths, mp_5, mpci5, label="Macro-Precision K=5", color="#F58518")
    ax.set_title("Macro-Precision vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Macro-Precision (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "macro_precision_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    coverage_ths, coverage_means, coverage_ci, has_coverage = extract_global_metric(summary, "coverage_ratio")
    if has_coverage:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = bool(plot_line_with_ci(ax, coverage_ths, coverage_means, coverage_ci, label="Coverage", color="#B279A2"))
        ax.set_title("Global Support Coverage vs. Identity Threshold")
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel("Coverage ratio (evaluated queries / total)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "global_coverage_vs_threshold.png", dpi=args.dpi)
        figs.append(fig)

    ths, mr_1, mrci1 = extract_metric(summary, "macro_recall", "K=1")
    _, mr_5, mrci5 = extract_metric(summary, "macro_recall", "K=5")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_line_with_ci(ax, ths, mr_1, mrci1, label="Macro-Recall K=1", color="#4C78A8")
    plot_line_with_ci(ax, ths, mr_5, mrci5, label="Macro-Recall K=5", color="#F58518")
    ax.set_title("Macro-Recall vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Macro-Recall (mean ± 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "macro_recall_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 4) Cluster counts vs threshold
    cluster_counts = collect_cluster_counts(results_dir, thresholds)
    cluster_trend = describe_monotonic_trend(cluster_counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(t) for t in thresholds], cluster_counts, color="#72B7B2")
    ax.set_title("#Clusters vs. Identity Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("#Unique clusters (dataset)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clusters_vs_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 5) Per-fold accuracy distribution (boxplots), grouped by K
    k1, k5 = collect_per_fold_metric(results_dir, thresholds, "acc_top1_hit")
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped_boxplot(ax, thresholds, k1, k5, labels=("K=1", "K=5"))
    ax.set_title("Per-fold Top-1 Accuracy by Threshold")
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_boxplot_by_threshold.png", dpi=args.dpi)
    figs.append(fig)

    # 6) Print a short textual summary for quick sense-making
    def _clean(values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[~np.isnan(arr)]

    def describe_change(name: str, y1: List[float], y5: List[float]) -> str:
        arr1 = _clean(y1)
        arr5 = _clean(y5)
        if arr1.size == 0 or arr5.size == 0:
            return f"- {name}: insufficient data"
        return (
            f"- {name}: K=1 mean {arr1.mean():.3f} → K=5 mean {arr5.mean():.3f} "
            f"(Δ={arr5.mean()-arr1.mean():+.3f})"
        )

    def describe_global(name: str, values: List[float], fmt: str = "{:.3f}") -> str:
        arr = _clean(values)
        if arr.size == 0:
            return f"- {name}: unavailable"
        return (
            f"- {name}: mean {fmt.format(arr.mean())} "
            f"(min {fmt.format(arr.min())}, max {fmt.format(arr.max())})"
        )

    print("\n[visualize] Summary (means across thresholds):")
    s_acc = describe_change("Top-1 accuracy", acc1, acc5)
    s_mf1 = describe_change("Macro-F1", mf1_1, mf1_5)
    s_mp = describe_change("Macro-Precision", mp_1, mp_5)
    s_mr = describe_change("Macro-Recall", mr_1, mr_5)
    s_clu = f"- #Clusters {cluster_trend} with threshold: " + " → ".join(str(c) for c in cluster_counts)
    s_gmicro = describe_global("Global micro-F1", g_micro)
    s_gmacro = describe_global("Global macro-F1", g_macro)
    s_cov = describe_global("Global coverage", coverage_means, fmt="{:.1%}")
    for line in (s_acc, s_mf1, s_mp, s_mr, s_gmicro, s_gmacro, s_cov, s_clu):
        print(line)
    print(f"[visualize] Wrote figures to: {out_dir.resolve()}")

    # 7) Optional PDF report bundling: title + summary page + all plots
    if not args.no_pdf:
        pdf_path = Path(args.pdf_path) if args.pdf_path else (out_dir / "identity_benchmark_report.pdf")
        # Build a summary page
        fig_sum = plt.figure(figsize=(8.5, 11))
        fig_sum.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
        fig_sum.suptitle("Identity Benchmark Report", fontsize=16, fontweight="bold", y=0.97)
        text_lines: List[str] = []
        text_lines.append(f"Thresholds (%): {', '.join(str(t) for t in thresholds)}")
        text_lines.append("")
        text_lines.append(s_acc)
        text_lines.append(s_mf1)
        text_lines.append(s_mp)
        text_lines.append(s_mr)
        text_lines.append("")
        text_lines.append(s_gmicro)
        text_lines.append(s_gmacro)
        text_lines.append(s_cov)
        text_lines.append(s_clu)
        body = "\n".join(text_lines)
        fig_sum.text(0.06, 0.90, "Overview", fontsize=12, fontweight="bold", va="top")
        fig_sum.text(0.06, 0.87, body, fontsize=10, va="top")

        with PdfPages(str(pdf_path)) as pdf:
            pdf.savefig(fig_sum)
            for f in figs:
                pdf.savefig(f)
        # Close all figures to free memory
        import matplotlib.pyplot as _plt
        _plt.close(fig_sum)
        for f in figs:
            _plt.close(f)
        print(f"[visualize] Wrote PDF report → {pdf_path}")


if __name__ == "__main__":
    main()
