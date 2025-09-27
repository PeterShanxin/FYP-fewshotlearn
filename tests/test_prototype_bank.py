from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.prototype_bank import build_prototypes
from src.protonet import ProtoConfig, ProtoNet


def _write_embeddings(base: Path, keys: list[str], vectors: list[list[float]]) -> None:
    np.save(base.with_suffix(".X.npy"), np.array(vectors, dtype=np.float32))
    np.save(base.with_suffix(".keys.npy"), np.array(keys, dtype="U8"))


def _write_split(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _identity_model(dim: int) -> ProtoNet:
    cfg = ProtoConfig(input_dim=dim, projection_dim=dim, temperature=1.0)
    model = ProtoNet(cfg)
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(dim))
        model.proj.bias.zero_()
    model.eval()
    return model


def test_build_prototypes_handles_multi_ec(tmp_path: Path) -> None:
    base = tmp_path / "emb"
    keys = ["x1", "x2", "x3"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    _write_embeddings(base, keys, vectors)

    split_entries = [
        {"ec": "A", "accessions": ["x1", "x2"]},
        {"ec": "B", "accessions": ["x1", "x3"]},
    ]
    split_path = tmp_path / "train.jsonl"
    _write_split(split_path, split_entries)

    model = _identity_model(2).to(torch.device("cpu"))

    prototypes, counts = build_prototypes(
        split_path,
        base,
        model,
        device=torch.device("cpu"),
        subprototypes_per_ec=1,
        batch_size=2,
    )

    assert set(prototypes.keys()) == {"A", "B"}
    assert counts["A"] == 2
    assert counts["B"] == 2
    assert prototypes["A"].shape == (1, 2)
    assert prototypes["B"].shape == (1, 2)
    assert np.allclose(np.linalg.norm(prototypes["A"], axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(prototypes["B"], axis=1), 1.0, atol=1e-5)

    prototypes_multi, _ = build_prototypes(
        split_path,
        base,
        model,
        device=torch.device("cpu"),
        subprototypes_per_ec=2,
        batch_size=2,
    )
    assert prototypes_multi["A"].shape == (2, 2)
    assert prototypes_multi["B"].shape == (2, 2)
