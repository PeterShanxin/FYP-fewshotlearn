from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.episodic_sampler import EpisodeSampler


def _write_embeddings(base: Path, keys: list[str], vectors: list[list[float]]) -> None:
    x_path = base.with_suffix(".X.npy")
    keys_path = base.with_suffix(".keys.npy")
    np.save(x_path, np.array(vectors, dtype=np.float32))
    np.save(keys_path, np.array(keys, dtype="U8"))


def _write_split(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_fallback_applies_only_during_training(tmp_path: Path) -> None:
    base = tmp_path / "embeddings"
    keys = ["a1", "a2"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    _write_embeddings(base, keys, vectors)

    split_path = tmp_path / "train.jsonl"
    _write_split(split_path, [{"ec": "A", "accessions": keys}])

    seq_lookup = {"a1": "AAAA", "a2": "BBBB"}

    train_sampler = EpisodeSampler(
        base,
        split_path,
        torch.device("cpu"),
        seed=42,
        phase="train",
        multi_label=False,
        with_replacement_fallback=True,
        fallback_scope="train_only",
        sequence_lookup=seq_lookup,
    )

    sx, sy, qx, qy, classes = train_sampler.sample_episode(M=1, K=2, Q=2)

    assert classes == ["A"]
    assert sx.shape[0] == 2
    assert qx.shape[0] == 2
    assert train_sampler.fallback_events == 1
    assert train_sampler.fallback_per_ec["A"] == 1
    assert train_sampler.usage_support_counts["A"] == 2
    assert train_sampler.usage_query_counts["A"] == 2

    val_sampler = EpisodeSampler(
        base,
        split_path,
        torch.device("cpu"),
        seed=7,
        phase="val",
        multi_label=False,
        with_replacement_fallback=True,
        fallback_scope="train_only",
        sequence_lookup=seq_lookup,
    )

    with pytest.raises(RuntimeError):
        val_sampler.sample_episode(M=1, K=2, Q=2)

