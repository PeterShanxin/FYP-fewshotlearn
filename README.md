# Few‑Shot Enzyme Classification (EC) with Identity Clustering, Multi‑EC, and Hierarchical Supervision

Advanced, reproducible pipeline for few‑shot EC classification using **metric‑based Prototypical Networks** over **ESM2** embeddings. The pipeline now adopts three core capabilities by default:
- Identity‑based clustering for disjoint support/query sampling and homology‑aware evaluation
- Multi‑EC handling (multi‑label across episode classes)
- EC hierarchy supervision (auxiliary losses at coarser EC levels)

Runs on CPU or a single small GPU; embedding can leverage multi‑GPU when available.

---

## Quickstart

### 0) Environment (Python 3.10+)
```bash
# with conda (recommended)
conda create -n fsl-ec python=3.10 -y
conda activate fsl-ec
pip install -r requirements.txt
```

### 1) Fetch Swiss‑Prot enzymes and build snapshot
This pulls **reviewed** UniProtKB/Swiss‑Prot entries **with EC numbers**, saves TSV+FASTA, logs a snapshot (release date, URLs, SHA256), and produces a joined TSV.
```bash
bash scripts/fetch_uniprot_ec.sh
# or with the Python fetcher (supports --force-download)
python scripts/fetch_uniprot_ec.py --force-download
```
Outputs go to `data/uniprot_ec/`.

### 2) Prepare EC class splits (multi‑EC expanded)
```bash
python -m src.prepare_split -c config.yaml
```
Writes JSONL files under `data/splits/` – one EC class per line. Multi‑EC rows are expanded by default so each accession appears under all of its ECs.

### 2.5) Cluster sequences for identity‑aware episodes
Create an accession→cluster map using MMseqs2 (preferred). If MMseqs2 is not available on PATH, the script falls back to a slow Python implementation for small tests.
```bash
python scripts/cluster_sequences.py -c config.yaml
```
Outputs `paths.clusters_tsv` (see `config.yaml`) and temporary files under `data/identity/_work/`.

### 3) Compute ESM2 embeddings (mean‑pooled per sequence)
```bash
python -m src.embed_sequences -c config.yaml
```
Creates fast‑loading contiguous files: 
- `data/emb/embeddings.X.npy` (shape `[N, D]`, float32) and 
- `data/emb/embeddings.keys.npy` (accessions, same order),
which load via memory‑mapping for near‑instant startup.

### 4) Train ProtoNet episodically (with hierarchy and multi‑EC)
```bash
python -m src.train_protonet -c config.yaml
```
Saves checkpoint at `results/checkpoints/protonet.pt` and training history at `results/history.json`. Training uses multi‑label targets and hierarchical auxiliary losses by default (see config).

### 5) Evaluate episodically on meta‑test
```bash
python -m src.eval_protonet -c config.yaml
```
Writes `results/metrics.json` keyed by K. In multi‑label mode: top‑1 hit accuracy (prediction counted correct if top‑1 is among true labels) plus micro/macro precision/recall/F1. In single‑label mode: accuracy and macro‑F1.

### One‑liners

- **Unix/macOS** (default config):
  ```bash
  bash scripts/run_all.sh config.yaml
  ```
  Or for a quick smoke test:
  ```bash
  bash scripts/run_all.sh config.smoke.yaml
  ```

- **Windows (PowerShell/CMD)** (default config):
  ```bat
  scripts\run_all.bat config.yaml
  ```
  Or for a smoke test:
  ```bat
  scripts\run_all.bat config.smoke.yaml
  ```

Replace `config.yaml` with any other config file as needed.

## EDA: EC Distribution Overview
Explore the distribution of EC numbers fetched from UniProt (hierarchical EC1→EC4):

Run EDA (CSV summaries only by default):
```bash
python scripts/eda_uniprot_ec.py \
  --data-root data/uniprot_ec \
  --out-dir results/eda_uniprot_ec \
  --no-timestamp
```

Optional plotting dependencies:
- `pip install matplotlib plotly`

Artifacts in `results/eda_uniprot_ec/`:
- `counts_ec1.csv`, `counts_ec12.csv`, `counts_ec123.csv`, `counts_ec_full.csv`
- `summary_stats.json` (totals, unique accessions, multi‑EC ratio)
- If matplotlib is installed: `ec1_distribution.png`, `top_ec_full.png`, `heatmap_ec1_ec2.png`, `heatmap_ec2_ec3.png`
- If plotly is installed: `sunburst_ec_hierarchy.html` (hierarchical “4‑digit” view), `sankey_ec_flow.html` (EC1→EC4 flow)

Notes:
- Unknown digits (`-` or empty) are normalized to `NA` to appear explicitly in plots.
- The EDA expects `data/uniprot_ec/swissprot_ec_joined_long.tsv` created by the fetch step.

## Common Workflows
- Reuse embeddings, retrain only: clean results then train+eval.
  - `bash scripts/clean_all.sh --yes --results`
  - `make train eval CFG=config.yaml`  or  `bash scripts/run_all.sh config.yaml` (skips embed if present)
- Re‑cluster then retrain: clean clusters and run cluster+train+eval.
  - `bash scripts/clean_all.sh --yes --clusters`
  - `make cluster train eval CFG=config.yaml`
- Re‑embed then retrain: clean embeddings and run embed+train+eval.
  - `bash scripts/clean_all.sh --yes --embeddings`
  - `bash scripts/run_all.sh config.yaml`  (auto‑embeds, then trains+evals)
- Force re‑embed even if files exist:
  - `bash scripts/run_all.sh config.yaml --force-embed`
- Identity CV benchmark (multi‑threshold): set `id_thresholds` in the config and run once.
  - `bash scripts/run_all.sh config.yaml` (auto orchestrates prepare+benchmark)
  - Or manual: `python scripts/prepare_identity_splits.py -c config.yaml` then `python scripts/run_identity_benchmark.py -c config.yaml`
- Makefile shortcuts (Unix): targeted stages and full runs.
  - `make all CFG=config.yaml` (fetch→split→cluster→embed→train→eval)
  - `make split cluster train eval CFG=config.yaml`

---

## Defaults (tuned for low resource)
See `config.yaml` for all knobs. Key defaults:

- `embedding.model`: `esm2_t12_35M_UR50D`
- `device`: auto (CUDA if available else CPU)
- Episodes – `M=10`, `K_train=5`, `K_eval=[1,5]`, `Q=10`
- Train/val episodes: `1000` / `200`
- `batch_size_embed=64` (raise on GPU; reduce on CPU)
- ProtoNet: 256‑dim optional projection, cosine scores scaled by `temperature=10.0`
- Split rule: classes with `< 40` sequences are sent to meta‑test pool
- Multi‑EC: `allow_multi_ec=true` (expand multi‑EC rows)
- Identity‑aware episodes: `identity_disjoint=true` with clusters from `paths.clusters_tsv`
- EC hierarchy: `hierarchy_levels=2`, `hierarchy_weight=0.2`
- Random seed: `42`

Paths (relative):
```
 data/uniprot_ec/            # raw data + snapshot
 data/uniprot_ec/swissprot_ec_joined.tsv
 data/emb/embeddings.X.npy
 data/emb/embeddings.keys.npy
 data/splits/{train,val,test}.jsonl
 results/
```

---

## HPC Profile (A40 48GB)
- Use `config.yaml` to leverage a single NVIDIA A40 (48GB) and a multi-core CPU.
- Key changes vs. default:
  - ESM model: `esm2_t33_650M_UR50D` (larger, better quality)
  - Embedding: `batch_size_embed=512`, `fp16: true`, `max_seq_len: 1022`, `dynamic_batch: true`
  - Embeddings are stored as contiguous `X.npy` + `keys.npy` for fast memory‑mapped loading.
  - Episodes: `train=10000`, `val=2000`, `eval=5000` for tighter confidence intervals
  - Device: `auto` (uses CUDA when available)
  - Training knobs: `eval_every=1000`, `episodes_per_val_check=200` (control validation cadence and variance)

Run the full pipeline with the HPC config:
```
bash scripts/run_all.sh config.yaml
```

Notes:
- If you switch to very large ESM models, 48GB may not hold `3B` model at high batch sizes. Start high (e.g., 512) and let `dynamic_batch: true` reduce as needed; you can also lower to 128–256 manually.
- Confidence interval tightening comes from increasing episode count: for episodic accuracy over N total queries, the standard error shrinks ~ O(1/sqrt(N)). Raising `episodes.val` and `episodes.eval` reduces variance.
- Precision:
  - Embedding runs in FP16 on GPU for throughput; full `max_seq_len=1022`.
  - ProtoNet training remains FP32 by default (`fp16_train: false`) as the head is small; AMP brings minimal gains for this model.

## Expected runtime (ballpark)
- **CPU (laptop 4–8 cores)**: fetch ~5–10 min; embed a few thousand seqs at ~5–15 seq/s; training 1k episodes ~10–30 min.
- **Small GPU (e.g., RTX 3060/4070)**: embedding 100–300 seq/s; training 1k episodes ~3–10 min.

Your mileage varies with data volume and CPU/GPU.

---

## Core Features
- Identity‑aware episodes: enabled by default with `identity_disjoint: true`. Generate the clustering map via `scripts/cluster_sequences.py` using MMseqs2 when available (Python fallback for small tests). Defaults: `cluster_identity: 0.5`, `cluster_coverage: 0.5`.
- Multi‑EC support: enabled via `allow_multi_ec: true` (split expansion) and `multi_label: true` (BCE over episode classes). Episodic accuracy counts a prediction as correct if top‑1 is among true labels.
- EC hierarchy: enabled via `hierarchy_levels` (1–3) and `hierarchy_weight` to add auxiliary losses at coarser EC levels using prototype grouping per episode.

### HPC Modules (MMseqs2)
On module-managed clusters, load MMseqs2 before running the pipeline:
```bash
module load MMseqs2
```
If MMseqs2 is not available on PATH, the scripts fall back to a slow Python implementation suitable only for small smoke tests.

You can also declare modules in your config to have `run_all.sh` load them automatically:
```yaml
# config.yaml
modules: [MMseqs2]
```
This is optional; if the `module` command is unavailable, the script continues without loading.

---

## Identity-Constrained Benchmark (10/30/50/70/100)

Goal: enforce that no test sequence shares more than X% identity with any training sequence by splitting at the cluster level, and report how performance changes as homology constraints tighten.

- Config-driven run (recommended): set a list of thresholds in `config.yaml` and run the pipeline.
  ```yaml
  # config.yaml
  id_thresholds: [10, 30, 50, 70, 100]
  folds: 5
  ```
  ```bash
  bash scripts/run_all.sh config.yaml
  ```
  This orchestrates split prep + per-threshold training/eval and writes per-threshold results to `results/split-{pct}/fold-{i}/` along with `results/summary_by_id_threshold.json` (mean, std, and 95% CI per K across folds).

- Manual run (equivalent):
  ```bash
  python scripts/prepare_identity_splits.py -c config.yaml --cutoffs 0.1,0.3,0.5,0.7,1.0 --folds 5
  python scripts/run_identity_benchmark.py -c config.yaml --cutoffs 0.1,0.3,0.5,0.7,1.0 --folds 5
  ```
  Outputs:
  - Per-threshold directories: `results/split-10/`, `results/split-30/`, ... each containing:
    - `fold-1/..fold-5/metrics.json` (per-fold enriched metrics)
    - `aggregate.json` (fold-aggregated metrics for that threshold)
  - Cross-threshold summary: `results/summary_by_id_threshold.json`

Notes:
- Splits are made by clusters, not individual sequences. Entire clusters go to the test fold, guaranteeing the specified identity ceiling between train and test (when MMseqs2 is available).
- Episodic sampling also enforces within-episode identity disjointness when `identity_disjoint: true` (uses the same cluster map).

### Metrics (Eval)
- Single-label: accuracy, macro-precision/recall/F1.
- Multi-label: top-1 hit accuracy (prediction counted correct if top-1 is among true labels) plus thresholded (0.5) micro/macro precision/recall/F1.

### Visualization
Generate figures and a short textual summary from the multi-threshold results.

Requirements:
- `pip install matplotlib`

Usage:
```bash
python scripts/visualize_identity_benchmark.py \
  --results_dir results \
  --out_dir results/figures \
  --dpi 150
```

Also:
- `bash scripts/run_all.sh config.yaml` automatically runs visualization after the benchmark when the summary exists and matplotlib is installed.
- `make visualize` runs the plotting step manually (non-fatal if the summary is missing). The PDF is generated by default; pass `--no-pdf` to the script to disable.

Outputs (`--out_dir`):
- `accuracy_vs_threshold.png` (K=1 and K=5, mean ± 95% CI)
- `macro_f1_vs_threshold.png`, `micro_f1_vs_threshold.png`
- `macro_precision_vs_threshold.png`, `macro_recall_vs_threshold.png`
- `clusters_vs_threshold.png` (#clusters by threshold)
- `accuracy_boxplot_by_threshold.png` (per-fold distributions)
- `identity_benchmark_report.pdf` (bundled multi-page PDF with summary + all plots)

Interpretation tips:
- Accuracy generally increases with higher identity thresholds (task becomes easier as train/test are more similar); K=5 tends to outperform K=1.
- In multi-label mode, recall is typically very high while precision is low at a fixed 0.5 sigmoid threshold; consider threshold tuning if precision is critical.

### Fresh starts, selective clean, and force fetch
- Show clean help and flags:
  - `bash scripts/clean_all.sh --help`
- Clean everything (destructive):
  - `bash scripts/clean_all.sh --yes`
- Selective clean (pick categories):
  - Fetch data: `bash scripts/clean_all.sh --yes --fetch`
  - Clusters: `bash scripts/clean_all.sh --yes --clusters`
  - Embeddings: `bash scripts/clean_all.sh --yes --embeddings`
  - Splits: `bash scripts/clean_all.sh --yes --splits`
  - Results: `bash scripts/clean_all.sh --yes --results`
- Skip/force embedding in the runner:
  - Skip happens automatically if embeddings exist; force rebuild with `--force-embed`.
- Always re‑download UniProt snapshot via config toggle `force_fetch: true` (run_all.sh forwards to the Python fetcher as `--force-download`).

## What this project intentionally **does not** do by default
- Heavy dependencies or distributed training beyond simple DataParallel for embedding.

---

## Reproducibility & snapshotting
- `scripts/fetch_uniprot_ec.sh` logs:
  - UniProt `X-UniProt-Release-Date`
  - the exact REST URLs and query
  - SHA256 checksums of fetched files
- `config.yaml` centralizes all parameters and paths.
- Every script prints key config (M, K, Q, episodes, device) at start.

---

## Citation & licenses
- ESM models by Meta FAIR team (apache‑2.0). UniProt data subject to their terms.
- If you use this template, consider citing Snell et al. (Prototypical Networks) and UniProt.

---

## Troubleshooting
- **CUDA not detected**: the code will fall back to CPU automatically.
- **Slow embedding**: increase `batch_size_embed` if memory allows.
- **Few classes make K+Q sampling hard**: lower `K_train`, `Q`, or `min_sequences_per_class_for_train` in `config.yaml`.
- **Windows shell**: use `scripts/run_all.bat` instead of the Bash script.
