"""Prototype bank construction utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from .episodic_sampler import SplitIndex


def _load_embeddings(embeddings_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    base = str(embeddings_path)
    if base.endswith(".npz"):
        base = base[:-4]
    X_path = Path(base + ".X.npy")
    keys_path = Path(base + ".keys.npy")
    if not (X_path.exists() and keys_path.exists()):
        raise FileNotFoundError(
            f"Embeddings not found. Expected contiguous arrays at {X_path} and {keys_path}."
        )
    X = np.load(X_path, mmap_mode="r")  # type: ignore[attr-defined]
    keys = np.load(keys_path, allow_pickle=False)  # type: ignore[attr-defined]
    if X.shape[0] != keys.shape[0]:
        raise ValueError(f"Embeddings/key count mismatch: {X.shape[0]} vs {keys.shape[0]}")
    key2row = {str(k): int(i) for i, k in enumerate(keys.tolist())}  # type: ignore[attr-defined]
    return X, keys, key2row


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def _batch(items: Iterable[int], size: int) -> Iterable[Tuple[int, ...]]:
    block: list[int] = []
    for item in items:
        block.append(item)
        if len(block) == size:
            yield tuple(block)
            block.clear()
    if block:
        yield tuple(block)


def build_prototypes(
    train_jsonl: Path,
    embeddings_path: Path,
    model: torch.nn.Module,
    *,
    device: torch.device,
    subprototypes_per_ec: int = 1,
    batch_size: int = 1024,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Construct class prototypes from train split embeddings.

    Returns a tuple (prototypes, train_counts) where prototypes maps EC →
    array of centroids and train_counts records the number of training
    sequences contributing to each EC.
    """

    model.eval()
    split = SplitIndex.from_jsonl(Path(train_jsonl))
    X, _keys, key2row = _load_embeddings(Path(embeddings_path))
    class2rows: Dict[str, list[int]] = {}
    for ec, accs in split.by_class.items():
        rows = [key2row[a] for a in accs if a in key2row]
        if rows:
            class2rows[ec] = rows

    cache: Dict[int, np.ndarray] = {}
    prototypes: Dict[str, np.ndarray] = {}
    train_counts: Dict[str, int] = {}

    with torch.no_grad():
        for ec in sorted(class2rows.keys()):
            rows = class2rows[ec]
            train_counts[ec] = len(rows)
            missing = [idx for idx in rows if idx not in cache]
            for chunk in _batch(missing, batch_size):
                if not chunk:
                    continue
                batch_np = np.stack([X[idx] for idx in chunk]).astype(np.float32)
                tensor = torch.from_numpy(batch_np).to(device)
                embedded = model.embed(tensor).detach().cpu().numpy()
                for idx, vec in zip(chunk, embedded):
                    cache[idx] = vec.astype(np.float32)
            embeds = np.stack([cache[idx] for idx in rows]).astype(np.float32)
            if subprototypes_per_ec > 1 and embeds.shape[0] >= subprototypes_per_ec:
                km = KMeans(n_clusters=subprototypes_per_ec, n_init="auto", random_state=0)
                km.fit(embeds)
                centroids = km.cluster_centers_.astype(np.float32)
            else:
                centroids = embeds.mean(axis=0, keepdims=True).astype(np.float32)
            prototypes[ec] = _normalize_rows(centroids)

    return prototypes, train_counts


def save_prototypes(
    npz_path: Path,
    prototypes: Dict[str, np.ndarray],
    *,
    train_counts: Dict[str, int] | None = None,
) -> None:
    keys = np.array(list(prototypes.keys()))
    proto_counts = np.array([prototypes[k].shape[0] for k in keys], dtype=np.int32)
    train_counts_arr = (
        np.array([train_counts.get(k, 0) for k in keys], dtype=np.int32)
        if train_counts is not None
        else np.zeros_like(proto_counts)
    )
    if keys.size > 0:
        dim = prototypes[keys[0]].shape[1]
        data = np.vstack([prototypes[k] for k in keys]).astype(np.float32)
    else:
        dim = 0
        data = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(
        npz_path,
        keys=keys,
        proto_counts=proto_counts,
        train_counts=train_counts_arr,
        dim=np.array([dim], dtype=np.int32),
        data=data,
    )


def load_prototypes(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    with np.load(npz_path, allow_pickle=False) as arrays:
        keys = arrays["keys"]
        proto_counts = arrays["proto_counts"]
        train_counts_arr = arrays["train_counts"]
        data = arrays["data"]
        dim = int(arrays["dim"][0]) if "dim" in arrays else data.shape[1]
    prototypes: Dict[str, np.ndarray] = {}
    train_counts: Dict[str, int] = {}
    offset = 0
    for key, count, train_count in zip(keys.tolist(), proto_counts.tolist(), train_counts_arr.tolist()):
        key_str = str(key)
        if count <= 0:
            continue
        slice_arr = data[offset : offset + count]
        offset += count
        prototypes[key_str] = slice_arr.astype(np.float32).reshape(count, dim)
        train_counts[key_str] = int(train_count)
    return prototypes, train_counts

