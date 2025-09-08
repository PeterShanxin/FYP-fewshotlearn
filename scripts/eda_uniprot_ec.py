#!/usr/bin/env python3
"""EDA for UniProt EC dataset fetched by scripts/fetch_uniprot_ec.py.

Outputs summary stats, CSV aggregates, and plots (when libraries available):
- counts at EC1/EC1.2/EC1.2.3/full EC levels (CSVs)
- top-K EC full bar chart (PNG; requires matplotlib)
- EC1 distribution bar chart (PNG; requires matplotlib)
- EC1xEC2 and EC2xEC3 heatmaps (PNG; requires matplotlib)
- hierarchical sunburst and 4-stage Sankey (HTML; requires plotly)

Usage:
  python scripts/eda_uniprot_ec.py \
    --data-root data/uniprot_ec \
    --out-dir results/eda_uniprot_ec \
    --top-k 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import datetime as _dt

import pandas as pd


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _try_import_plotly():
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        return px, go
    except Exception:
        return None, None


def normalize_long_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    required = ["accession", "ec_full", "ec1", "ec2", "ec3", "ec4"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in long TSV")
    # Ensure string dtype and strip
    for c in required:
        df[c] = df[c].astype(str).str.strip()
    # Represent unknowns uniformly as 'NA' (cover '', '-', and 'nan')
    for c in ["ec1", "ec2", "ec3", "ec4"]:
        df[c] = df[c].replace({"-": "NA", "": "NA", "nan": "NA"})
    df["ec_full"] = df["ec_full"].replace({"": pd.NA}).fillna("NA")
    return df


def load_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    long_tsv = data_root / "swissprot_ec_joined_long.tsv"
    joined_tsv = data_root / "swissprot_ec_joined.tsv"
    if not long_tsv.exists():
        raise FileNotFoundError(f"Expected file not found: {long_tsv}. Run scripts/fetch_uniprot_ec.py first.")
    long_df = pd.read_csv(long_tsv, sep="\t", dtype=str)
    long_df = normalize_long_df(long_df)
    joined_df = None
    if joined_tsv.exists():
        try:
            joined_df = pd.read_csv(joined_tsv, sep="\t", dtype=str)
        except Exception:
            joined_df = None
    return long_df, joined_df


def ensure_outdir(out_dir: Path, timestamped: bool) -> Path:
    if timestamped:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def aggregate_counts(long_df: pd.DataFrame, out_dir: Path) -> dict:
    """Compute and save counts at various levels. Returns a dict of basic stats."""
    stats = {}
    # Unique accessions, and multi-EC stats
    acc_counts = long_df.groupby("accession").size().sort_values(ascending=False)
    n_accessions = acc_counts.index.nunique()
    n_ec_rows = len(long_df)
    n_multi_acc = int((acc_counts > 1).sum())
    stats.update(
        total_long_rows=int(n_ec_rows),
        unique_accessions=int(n_accessions),
        multi_ec_accessions=int(n_multi_acc),
        multi_ec_ratio=float(n_multi_acc / n_accessions) if n_accessions else 0.0,
    )

    # Level counts
    def _save_counts(key: str, series: pd.Series):
        df = series.rename("count").reset_index().rename(columns={series.index.name: key})
        df.to_csv(out_dir / f"counts_{key}.csv", index=False)
        return df

    c1 = long_df["ec1"].value_counts()
    df1 = _save_counts("ec1", c1)

    # Pair levels
    ec12 = (long_df["ec1"] + "." + long_df["ec2"]).value_counts()
    df12 = _save_counts("ec12", ec12)

    ec123 = (long_df["ec1"] + "." + long_df["ec2"] + "." + long_df["ec3"]).value_counts()
    df123 = _save_counts("ec123", ec123)

    ecfull = long_df["ec_full"].value_counts()
    dffull = _save_counts("ec_full", ecfull)

    stats.update(
        n_ec1=int(len(df1)),
        n_ec12=int(len(df12)),
        n_ec123=int(len(df123)),
        n_ec_full=int(len(dffull)),
    )
    return stats


def save_heatmap_png(plt, pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(pivot.columns)), max(4, 0.4 * len(pivot.index))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_bar_png(plt, series: pd.Series, title: str, out_path: Path, top_k: int = 30) -> None:
    s = series.sort_values(ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(s)), 5))
    ax.bar(range(len(s)), s.values)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(list(s.index), rotation=90)
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_matplotlib_plots(long_df: pd.DataFrame, out_dir: Path, top_k: int) -> list[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        return ["matplotlib not available; skipped PNG plots"]
    notes = []
    # Bar: EC1 distribution
    save_bar_png(
        plt,
        long_df["ec1"].value_counts(),
        title="EC1 distribution",
        out_path=out_dir / "ec1_distribution.png",
        top_k=50,
    )
    notes.append("Saved ec1_distribution.png")

    # Bar: Top-K full ECs
    save_bar_png(
        plt,
        long_df["ec_full"].value_counts(),
        title=f"Top {top_k} EC full counts",
        out_path=out_dir / "top_ec_full.png",
        top_k=top_k,
    )
    notes.append("Saved top_ec_full.png")

    # Heatmaps
    piv12 = (
        long_df.assign(ec12=long_df["ec1"] + "." + long_df["ec2"])  # not used directly; for clarity
        .pivot_table(index="ec1", columns="ec2", values="accession", aggfunc="count", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    save_heatmap_png(plt, piv12, "EC1 x EC2 counts", out_dir / "heatmap_ec1_ec2.png")
    notes.append("Saved heatmap_ec1_ec2.png")

    piv23 = (
        long_df.pivot_table(index="ec2", columns="ec3", values="accession", aggfunc="count", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    save_heatmap_png(plt, piv23, "EC2 x EC3 counts", out_dir / "heatmap_ec2_ec3.png")
    notes.append("Saved heatmap_ec2_ec3.png")

    return notes


def make_plotly_figures(long_df: pd.DataFrame, out_dir: Path) -> list[str]:
    px, go = _try_import_plotly()
    if px is None or go is None:
        return ["plotly not available; skipped HTML plots"]
    notes = []

    # Sunburst: hierarchical EC1 -> EC2 -> EC3 -> EC4
    agg = long_df.groupby(["ec1", "ec2", "ec3", "ec4"]).size().reset_index(name="count")
    fig_sb = px.sunburst(
        agg,
        path=["ec1", "ec2", "ec3", "ec4"],
        values="count",
        color="ec1",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="EC hierarchy sunburst (counts)"
    )
    out_sb = out_dir / "sunburst_ec_hierarchy.html"
    fig_sb.write_html(str(out_sb))
    notes.append("Saved sunburst_ec_hierarchy.html")

    # Sankey: flows EC1 -> EC2 -> EC3 -> EC4
    def sankey_df(level_a: str, level_b: str) -> pd.DataFrame:
        df = long_df.groupby([level_a, level_b]).size().reset_index(name="count")
        df.columns = ["src", "dst", "count"]
        # Add depth prefixes to disambiguate identical labels across levels
        depth_a = level_a[-1]
        depth_b = level_b[-1]
        df["src"] = df["src"].apply(lambda x: f"EC{depth_a}:{x}")
        df["dst"] = df["dst"].apply(lambda x: f"EC{depth_b}:{x}")
        return df

    e12 = sankey_df("ec1", "ec2")
    e23 = sankey_df("ec2", "ec3")
    e34 = sankey_df("ec3", "ec4")
    edges = pd.concat([e12, e23, e34], ignore_index=True)
    nodes = pd.Index(sorted(pd.unique(edges[["src", "dst"]].values.ravel()))).tolist()
    node_index = {n: i for i, n in enumerate(nodes)}
    link = dict(
        source=[node_index[s] for s in edges["src"].tolist()],
        target=[node_index[t] for t in edges["dst"].tolist()],
        value=edges["count"].astype(int).tolist(),
    )
    fig_sk = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes, pad=10, thickness=12),
                link=link,
                arrangement="snap",
            )
        ]
    )
    fig_sk.update_layout(title_text="EC flow Sankey (EC1→EC4)")
    out_sk = out_dir / "sankey_ec_flow.html"
    fig_sk.write_html(str(out_sk))
    notes.append("Saved sankey_ec_flow.html")

    return notes


def main():
    ap = argparse.ArgumentParser(description="EDA for UniProt EC dataset")
    ap.add_argument("--data-root", default="data/uniprot_ec", help="Directory containing swissprot_ec_joined_long.tsv")
    ap.add_argument("--out-dir", default="results/eda_uniprot_ec", help="Output directory for EDA artifacts")
    ap.add_argument("--top-k", type=int, default=30, help="Top-K for bar charts of full EC")
    ap.add_argument("--no-timestamp", action="store_true", help="Do not create a timestamped subdirectory")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = ensure_outdir(Path(args.out_dir), timestamped=(not args.no_timestamp))

    long_df, joined_df = load_data(data_root)

    # Basic stats
    stats = aggregate_counts(long_df, out_dir)
    if joined_df is not None:
        try:
            stats["joined_rows"] = int(len(joined_df))
            stats["joined_unique_accessions"] = int(joined_df["accession"].astype(str).nunique())
        except Exception:
            pass

    # Plots
    notes = []
    notes += make_matplotlib_plots(long_df, out_dir, top_k=args.top_k)
    notes += make_plotly_figures(long_df, out_dir)
    stats["notes"] = notes

    # Save stats and a short README
    write_json(stats, out_dir / "summary_stats.json")
    readme = []
    readme.append("# EDA: UniProt EC distribution")
    readme.append("")
    readme.append("Artifacts:")
    readme.append("- counts_ec1.csv, counts_ec12.csv, counts_ec123.csv, counts_ec_full.csv")
    if any("ec1_distribution.png" in n for n in notes):
        readme.append("- ec1_distribution.png")
    if any("top_ec_full.png" in n for n in notes):
        readme.append("- top_ec_full.png")
    if any("heatmap_ec1_ec2.png" in n for n in notes):
        readme.append("- heatmap_ec1_ec2.png, heatmap_ec2_ec3.png")
    if any("sunburst_ec_hierarchy.html" in n for n in notes):
        readme.append("- sunburst_ec_hierarchy.html")
    if any("sankey_ec_flow.html" in n for n in notes):
        readme.append("- sankey_ec_flow.html")
    readme.append("")
    readme.append("Notes:")
    for n in notes:
        readme.append(f"- {n}")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # Console summary
    print("[EDA] Output directory:", out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

