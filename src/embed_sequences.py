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
    bs = int(cfg.get("batch_size_embed", 4))

    print("[embed] config:")
    print(
        json.dumps(
            dict(
                model=cfg["embedding"]["model"],
                device=str(device),
                batch_size=bs,
                joined_tsv=paths["joined_tsv"],
                out_npz=paths["embeddings"],
            ),
            indent=2,
        )
    )

    # Collect accession set from splits
    accs_needed = read_splits(Path(paths["splits_dir"]))
    seq_map = sequence_map(Path(paths["joined_tsv"]))

    pairs = [(a, seq_map[a]) for a in accs_needed if a in seq_map]
    missing = [a for a in accs_needed if a not in seq_map]
    if missing:
        print(f"[embed] WARNING: {len(missing)} accessions missing sequences; they will be skipped.")

    # Load ESM model
    import esm  # noqa: WPS433

    model_name = cfg["embedding"]["model"]
    print(f"[embed] loading {model_name} …")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()

    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in batch_iter(pairs, bs):
            data = [(name, seq) for name, seq in batch]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            reps = model(tokens, repr_layers=[model.num_layers], return_contacts=False)[
                "representations"
            ][model.num_layers]
            for i, (name, _seq) in enumerate(batch):
                tok = tokens[i]
                rep = reps[i]
                # mask: exclude padding, CLS, EOS
                mask = (
                    (tok != alphabet.padding_idx)
                    & (tok != alphabet.cls_idx)
                    & (tok != alphabet.eos_idx)
                )
                vec = rep[mask].mean(dim=0).cpu().numpy().astype("float32")
                out[name] = vec

    out_path = Path(paths["embeddings"]) 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    dim = next(iter(out.values())).shape[0] if out else 0
    print(f"[embed] wrote {len(out)} embeddings (dim={dim}) → {out_path}")


if __name__ == "__main__":
    main()
