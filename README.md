# Few‑Shot Enzyme Classification (EC) with Prototypical Networks over ESM2‑t12‑35M

Minimal, reproducible project for few‑shot EC classification using **metric‑based Prototypical Networks** trained on **ESM2‑t12‑35M (UR50D)** embeddings. Designed to run on a normal PC (CPU) or a single small GPU.

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
# or on Windows without WSL
python scripts/fetch_uniprot_ec.py
```
Outputs go to `data/uniprot_ec/`.

### 2) Prepare cluster‑free class splits (meta‑train/val/test)
```bash
python -m src.prepare_split -c config.yaml
```
Writes JSONL files under `data/splits/` – one EC class per line.

### 3) Compute ESM2 embeddings (mean‑pooled per sequence)
```bash
python -m src.embed_sequences -c config.yaml
```
Creates `data/emb/embeddings.npz` mapping accession → float32 vector.

### 4) Train ProtoNet episodically
```bash
python -m src.train_protonet -c config.yaml
```
Saves checkpoint at `results/checkpoints/protonet.pt` and training history at `results/history.json`.

### 5) Evaluate episodically on meta‑test
```bash
python -m src.eval_protonet -c config.yaml
```
Writes `results/metrics.json` with accuracy and macro‑F1 for EC‑level labels treated as flat classes.

### One‑liners
- Unix/macOS:
  ```bash
  bash scripts/run_all.sh
  ```
- Windows (PowerShell/CMD):
  ```bat
  scripts\run_all.bat
  ```

---

## Defaults (tuned for low resource)
See `config.yaml` for all knobs. Key defaults:

- `embedding.model`: `esm2_t12_35M_UR50D`
- `device`: auto (CUDA if available else CPU)
- Episodes – `M=10`, `K_train=5`, `K_eval=[1,5]`, `Q=10`
- Train/val episodes: `1000` / `200`
- `batch_size_embed=4` (CPU‑safe)
- ProtoNet: 256‑dim optional projection, cosine scores scaled by `temperature=10.0`
- Split rule: classes with `< 40` sequences are sent to meta‑test pool
- Random seed: `42`

Paths (relative):
```
 data/uniprot_ec/            # raw data + snapshot
 data/uniprot_ec/swissprot_ec_joined.tsv
 data/emb/embeddings.npz
 data/splits/{train,val,test}.jsonl
 results/
```

---

## Expected runtime (ballpark)
- **CPU (laptop 4–8 cores)**: fetch ~5–10 min; embed a few thousand seqs at ~5–15 seq/s; training 1k episodes ~10–30 min.
- **Small GPU (e.g., RTX 3060/4070)**: embedding 100–300 seq/s; training 1k episodes ~3–10 min.

Your mileage varies with data volume and CPU/GPU.

---

## What this project intentionally **does not** do (to stay minimal)
- No sequence identity clustering (this is a quick PC trial). 
- No hierarchical EC modeling – we treat EC IDs as **flat classes** for now.
- No heavy dependencies or distributed training.

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
