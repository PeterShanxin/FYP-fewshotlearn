"""Compute mean-pooled ESM2 embeddings for accessions referenced in splits.

- Reads sequences from the joined TSV (column: sequence)
- Collects the union of accessions appearing in train/val/test splits
- Uses ESM2 (model from config); CPU or GPU
- Writes contiguous arrays: `embeddings.X.npy` (float32, [N, D]) and `embeddings.keys.npy`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
import torch.nn as nn


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def read_splits(splits_dir: Path) -> List[str]:
    accs: set[str] = set()
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = splits_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}. Run prepare_split first.")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                for a in obj["accessions"]:
                    accs.add(a)
    return sorted(accs)


def sequence_map(joined_tsv: Path) -> Dict[str, str]:
    df = pd.read_csv(joined_tsv, sep="\t")
    df.columns = [c.lower() for c in df.columns]
    seq_map = {str(r["accession"]): str(r["sequence"]).upper() for _, r in df.iterrows() if pd.notna(r["sequence"])}
    return seq_map


def batch_iter(items: List[Tuple[str, str]], bs: int):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)
    # Multi-GPU toggle (used for embedding only)
    requested_gpus = int(cfg.get("gpus", 1))
    bs = int(cfg.get("batch_size_embed", 4))
    max_seq_len = int(cfg.get("max_seq_len", 1022))  # ESM2 context limit
    truncate_long = bool(cfg.get("truncate_long_sequences", False))
    use_fp16 = bool(cfg.get("fp16", True)) and torch.cuda.is_available()
    dynamic_batch = bool(cfg.get("dynamic_batch", True))
    show_progress = bool(cfg.get("progress", True))
    verbose = bool(cfg.get("verbose", show_progress))

    print("[embed] config:")
    cfg_display = {
        "model": cfg["embedding"]["model"],
        "device": str(device),
        "batch_size": bs,
        "fp16": bool(use_fp16),
        "max_seq_len": max_seq_len,
        "joined_tsv": paths["joined_tsv"],
        "embeddings_base": paths["embeddings"],
        "truncate_long_sequences": truncate_long,
    }
    print(json.dumps(cfg_display, indent=2))

    # Collect accession set from splits
    accs_needed = read_splits(Path(paths["splits_dir"]))
    seq_map = sequence_map(Path(paths["joined_tsv"]))

    pairs: List[Tuple[str, str]] = []
    skipped_long: List[str] = []
    truncated: List[str] = []
    seq_limit = max_seq_len if max_seq_len > 0 else None
    for acc in accs_needed:
        seq = seq_map.get(acc)
        if seq is None:
            continue
        if seq_limit is not None and len(seq) > seq_limit:
            if truncate_long:
                seq = seq[:seq_limit]
                truncated.append(acc)
            else:
                skipped_long.append(acc)
                continue
        pairs.append((acc, seq))
    # Sort by sequence length (shortest first) to warm up and reduce early OOM risk
    pairs.sort(key=lambda x: len(x[1]))
    missing = [a for a in accs_needed if a not in seq_map]
    if missing:
        print(f"[embed] WARNING: {len(missing)} accessions missing sequences; they will be skipped.")
    if truncated:
        print(
            f"[embed] INFO: {len(truncated)} sequences exceed max_seq_len={seq_limit} and were truncated."
        )
    if skipped_long:
        print(
            f"[embed] WARNING: {len(skipped_long)} sequences exceed max_seq_len={seq_limit} and were skipped."
        )

    # Load ESM model
    try:
        import esm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "[embed] ERROR: Can't import 'esm'. Install with 'pip install fair-esm'."
        ) from e

    model_name = cfg["embedding"]["model"]
    print(f"[embed] loading {model_name} …")
    # Require official pretrained loader; do not attempt manual fallbacks
    if not hasattr(esm, "pretrained"):
        raise SystemExit(
            "[embed] ERROR: esm.pretrained is unavailable. Ensure fair-esm>=2.0.0 is installed."
        )
    if not hasattr(esm.pretrained, model_name):
        raise SystemExit(
            f"[embed] ERROR: model '{model_name}' not found in esm.pretrained. "
            "Check your config and installed fair-esm version."
        )
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    # Prepare device(s)
    multi_gpu = (device.type == "cuda" and torch.cuda.device_count() > 1 and requested_gpus > 1)
    if device.type == "cuda":
        # Anchor on cuda:0 and (optionally) wrap with DataParallel
        primary = torch.device("cuda:0")
        model.to(primary)
        # AMP: use half precision for forward if enabled
        if use_fp16:
            try:
                model.half()
                if verbose:
                    print("[embed] using FP16 inference")
            except Exception:
                if verbose:
                    print("[embed] FP16 conversion failed; continuing in FP32")
        if multi_gpu:
            max_gpus = min(requested_gpus, torch.cuda.device_count())
            device_ids = list(range(max_gpus))
            model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
            if verbose:
                print(f"[embed] DataParallel enabled on GPUs: {device_ids}")
        # Keep tokens on cuda:0; DataParallel scatters to other GPUs
        device = primary
    else:
        model.to(device)
    # Report GPU usage to terminal immediately after loading
    if device.type == "cuda":
        if multi_gpu:
            used_ids = device_ids  # type: ignore[name-defined]
            gcount = len(used_ids)
            glist = ",".join(f"cuda:{i}" for i in used_ids)
        else:
            gcount = 1
            glist = "cuda:0"
        print(f"[embed] using {gcount} GPU(s): {glist}", flush=True)
    else:
        print("[embed] using 0 GPU(s) (CPU)", flush=True)
    batch_converter = alphabet.get_batch_converter()

    out: Dict[str, np.ndarray] = {}
    def _embed_batch(batch_list, current_bs):
        data = [(name, seq) for name, seq in batch_list]
        _, _, tokens = batch_converter(data)
        if device.type == "cuda":
            tokens = tokens.pin_memory().to(device, non_blocking=True)
        else:
            tokens = tokens.to(device)
        autocast_ctx = (
            torch.amp.autocast(device.type, dtype=torch.float16)
            if use_fp16 and device.type == "cuda"
            else torch.amp.autocast(device.type, enabled=False)
        )
        # Determine num_layers for both bare model and DataParallel wrapper
        num_layers = getattr(getattr(model, 'module', model), 'num_layers')
        with autocast_ctx:
            reps = model(tokens, repr_layers=[num_layers], return_contacts=False)["representations"][num_layers]
        for i, (name, _seq) in enumerate(batch_list):
            tok = tokens[i]
            rep = reps[i]
            mask = (
                (tok != alphabet.padding_idx)
                & (tok != alphabet.cls_idx)
                & (tok != alphabet.eos_idx)
            )
            vec = rep[mask].mean(dim=0).float().cpu().numpy().astype("float32")
            out[name] = vec

    with torch.no_grad():
        i = 0
        total = len(pairs)
        tqdm_kwargs = {}
        if show_progress:
            tqdm_kwargs["file"] = sys.stdout
        pbar = tqdm(
            total=total,
            disable=not show_progress,
            desc="[embed]",
            dynamic_ncols=True,
            **tqdm_kwargs,
        )
        while i < total:
            batch_list = pairs[i:i+bs]
            try:
                _embed_batch(batch_list, bs)
                i += len(batch_list)
                if show_progress:
                    pbar.update(len(batch_list))
                    pbar.set_postfix(bs=bs, refresh=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if not dynamic_batch:
                    if show_progress:
                        pbar.close()
                    raise
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"\n[embed][OOM] reducing batch size to {bs} and retrying…", flush=True)
                else:
                    print("\n[embed][OOM] batch_size=1 still OOM on GPU; falling back to CPU for remaining sequences.", flush=True)
                    device = torch.device("cpu")
                    model.to(device)
                    use_fp16 = False
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and dynamic_batch:
                    torch.cuda.empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        print(f"\n[embed][OOM] reducing batch size to {bs} and retrying…", flush=True)
                    else:
                        print("\n[embed][OOM] unrecoverable at batch_size=1; aborting.", flush=True)
                        if show_progress:
                            pbar.close()
                        raise
                else:
                    if show_progress:
                        pbar.close()
                    raise
        if show_progress:
            pbar.close()

    out_path = Path(paths["embeddings"]) 
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive contiguous array paths from the configured embeddings base (accepts legacy .npz suffix)
    base_str = str(out_path)
    if base_str.endswith(".npz"):
        base_str = base_str[:-4]
    X_path = Path(base_str + ".X.npy")
    keys_path = Path(base_str + ".keys.npy")

    # Write contiguous array + keys (fast mmap loading)
    if out:
        # Deterministic key order (sorted) for reproducibility
        keys_list = sorted(out.keys())
        X = np.stack([out[k] for k in keys_list]).astype("float32", copy=False)
        np.save(X_path, X)
        np.save(keys_path, np.array(keys_list, dtype="U"))
        print(
            f"[embed] wrote contiguous embeddings: N={len(keys_list)} dim={X.shape[1]} → {X_path}, {keys_path}"
        )
    else:
        print("[embed] WARNING: no embeddings to write (empty output)")

    # Legacy NPZ output removed: embeddings are stored only as contiguous arrays.


if __name__ == "__main__":
    main()
