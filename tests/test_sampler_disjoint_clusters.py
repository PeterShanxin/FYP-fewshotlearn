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


def _write_clusters(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for acc, cid in mapping.items():
            handle.write(f"{acc}\t{cid}\n")


def test_disjoint_sampling_resamples_on_cluster_shortage(tmp_path: Path) -> None:
    base = tmp_path / "embeddings"
    keys = ["a1", "a2", "a3", "b1", "b2"]
    vectors = [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [10.0, 0.0],
        [11.0, 0.0],
    ]
    _write_embeddings(base, keys, vectors)

    split_path = tmp_path / "split.jsonl"
    _write_split(
        split_path,
        [
            {"ec": "A", "accessions": ["a1", "a2", "a3"]},
            {"ec": "B", "accessions": ["b1", "b2"]},
        ],
    )

    clusters_path = tmp_path / "clusters.tsv"
    _write_clusters(
        clusters_path,
        {
            "a1": "ca1",
            "a2": "ca2",
            "a3": "ca3",
            "b1": "cb",
            "b2": "cb",
        },
    )

    sampler = EpisodeSampler(
        base,
        split_path,
        torch.device("cpu"),
        seed=123,
        phase="train",
        disjoint_support_query=True,
        clusters_tsv=clusters_path,
    )

    sx, sy, qx, qy, classes = sampler.sample_episode(M=1, K=1, Q=1)

    assert classes == ["A"]
    assert sampler.cluster_shortage_counts["B"] >= 1
    assert sampler.cluster_shortage_dropped_episodes == 0

    support_vec = sx.numpy()[0]
    query_vec = qx.numpy()[0]
    assert support_vec[0] != query_vec[0]


def test_disjoint_sampling_raises_after_repeated_cluster_shortage(tmp_path: Path) -> None:
    base = tmp_path / "embeddings"
    keys = ["c1", "c2"]
    vectors = [[5.0, 0.0], [6.0, 0.0]]
    _write_embeddings(base, keys, vectors)

    split_path = tmp_path / "split.jsonl"
    _write_split(split_path, [{"ec": "C", "accessions": keys}])

    clusters_path = tmp_path / "clusters.tsv"
    _write_clusters(clusters_path, {"c1": "cc", "c2": "cc"})

    sampler = EpisodeSampler(
        base,
        split_path,
        torch.device("cpu"),
        seed=7,
        phase="train",
        disjoint_support_query=True,
        clusters_tsv=clusters_path,
    )

    with pytest.raises(RuntimeError) as excinfo:
        sampler.sample_episode(M=1, K=1, Q=1)

    assert "Unable to sample episode" in str(excinfo.value)
    assert sampler.cluster_shortage_dropped_episodes == 1
    assert sampler.cluster_shortage_counts["C"] >= sampler._max_cluster_resample_attempts
    assert sampler.cluster_shortage_last_drop == {"C": sampler._max_cluster_resample_attempts}
