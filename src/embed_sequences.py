"""Compute mean-pooled ESM2-t12-35M embeddings for accessions referenced in splits.

- Reads sequences from the joined TSV (column: sequence)
- Collects the union of accessions appearing in train/val/test splits
- Uses esm2_t12_35M_UR50D; CPU-safe by default (batch size from config)
- Saves a compressed .npz mapping accession -> float32 vector
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
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
        "out_npz": paths["embeddings"],
    }
    print(json.dumps(cfg_display, indent=2))

    # Collect accession set from splits
    accs_needed = read_splits(Path(paths["splits_dir"]))
    seq_map = sequence_map(Path(paths["joined_tsv"]))

    pairs = [(a, seq_map[a][:max_seq_len]) for a in accs_needed if a in seq_map]
    # Sort by sequence length (shortest first) to warm up and reduce early OOM risk
    pairs.sort(key=lambda x: len(x[1]))
    missing = [a for a in accs_needed if a not in seq_map]
    if missing:
        print(f"[embed] WARNING: {len(missing)} accessions missing sequences; they will be skipped.")

    # Load ESM model
    try:
        import esm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "[embed] ERROR: Can't import 'esm'. Install the correct package with 'pip install fair-esm'."
        ) from e

    model_name = cfg["embedding"]["model"]
    print(f"[embed] loading {model_name} …")
    if hasattr(esm, "pretrained"):
        model, alphabet = getattr(esm.pretrained, model_name)()
    else:
        # Manual construction path (since namespace package lacks pretrained helper)
        if model_name != "esm2_t12_35M_UR50D":
            raise SystemExit(
                "[embed] ERROR: 'esm.pretrained' missing and manual fallback only implemented for esm2_t12_35M_UR50D."
            )
        try:
            from esm.model import esm2
        except Exception as e:  # pragma: no cover
            raise SystemExit("[embed] ERROR: Could not import esm.model.esm2 for manual fallback") from e
        # Architecture index for t12_35M is 12 (layer count). Hidden size inferred internally.
        model = esm2.ESM2(n_layers=12, embed_dim=480, attention_heads=20)  # values per published spec
        # Build minimal Alphabet equivalent (FAIR ESM normally supplies). We'll approximate using tokens from model.esm2.
        # Simpler: require fair-esm instead of reconstructing; but provide emergency path.
        class _Alphabet:
            def __init__(self):
                standard_toks = list("ACDEFGHIKLMNPQRSTVWY")
                self.padding_idx = 0
                self.cls_idx = 1
                self.eos_idx = 2
                self.toks = ["<pad>", "<cls>", "<eos>"] + standard_toks
                self._tok_to_int = {t: i for i, t in enumerate(self.toks)}

            def get_batch_converter(self):  # minimal batch converter
                def convert(pairs):
                    batch_size = len(pairs)
                    max_len = max(len(seq) for _, seq in pairs)
                    import torch
                    tokens = torch.full((batch_size, max_len + 2), self.padding_idx, dtype=torch.long)
                    for i, (name, seq) in enumerate(pairs):
                        tokens[i,0] = self.cls_idx
                        for j, ch in enumerate(seq):
                            tokens[i, j+1] = self._tok_to_int.get(ch, self.padding_idx)
                        tokens[i, len(seq)+1] = self.eos_idx
                    return None, [p[0] for p in pairs], tokens
                return convert

        alphabet = _Alphabet()
        print("[embed] WARNING: Using fallback ESM2 construction without pretrained weights (random init). Install fair-esm for real embeddings.")
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
        pbar = tqdm(total=total, disable=not show_progress, desc="[embed]", dynamic_ncols=True)
        while i < total:
            batch_list = pairs[i:i+bs]
            try:
                _embed_batch(batch_list, bs)
                i += len(batch_list)
                if show_progress:
                    pbar.update(len(batch_list))
                    pbar.set_postfix(bs=bs, refresh=False)
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

    # Derive contiguous array paths from the configured NPZ path
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

    # Optionally write legacy NPZ for portability (uncompressed for faster load)
    if bool(cfg.get("write_legacy_npz", False)):
        np.savez(out_path, **out)
        dim = next(iter(out.values())).shape[0] if out else 0
        print(f"[embed] wrote legacy NPZ with {len(out)} arrays (dim={dim}) → {out_path}")


if __name__ == "__main__":
    main()
