from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.eval_global import run_global_evaluation
from src.model_utils import build_model
from src.prototype_bank import save_prototypes


def _write_embeddings(base: Path, keys: list[str], vectors: list[list[float]]) -> None:
    np.save(base.with_suffix(".X.npy"), np.array(vectors, dtype=np.float32))
    np.save(base.with_suffix(".keys.npy"), np.array(keys, dtype="U8"))


def _write_split(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_global_eval_thresholds(tmp_path: Path) -> None:
    embeddings_base = tmp_path / "embeddings"
    keys = ["x1", "x2"]
    vectors = [[1.0, 0.0], [0.6, 0.8]]
    _write_embeddings(embeddings_base, keys, vectors)

    split_dir = tmp_path / "splits"
    _write_split(
        split_dir / "test.jsonl",
        [
            {"ec": "A", "accessions": ["x1", "x2"]},
            {"ec": "B", "accessions": ["x2"]},
        ],
    )

    outputs_dir = tmp_path / "outputs"
    ckpt_dir = outputs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.yaml"
    prototypes_path = tmp_path / "protos.npz"

    cfg = {
        "paths": {
            "embeddings": str(embeddings_base),
            "splits_dir": str(split_dir),
            "outputs": str(outputs_dir),
            "runs": str(tmp_path / "runs"),
        },
        "projection_dim": 2,
        "temperature": 1.0,
        "eval": {
            "mode": "global_support",
            "shortlist_topN": 0,
            "temperature": 1.0,
            "tau_multi": 0.5,
            "per_ec_thresholds_path": None,
            "subprototypes_per_ec": 1,
            "prototypes_path": str(prototypes_path),
            "calibration_path": None,
            "split": "test",
        },
    }

    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)

    model = build_model(cfg, input_dim=2, device=torch.device("cpu"))
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(2))
        model.proj.bias.zero_()
    torch.save({"model": model.state_dict()}, ckpt_dir / "protonet.pt")

    prototypes = {
        "A": np.array([[1.0, 0.0]], dtype=np.float32),
        "B": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    counts = {"A": 2, "B": 1}
    save_prototypes(prototypes_path, prototypes, train_counts=counts)

    metrics_loose = run_global_evaluation(
        config_path=config_path,
        prototypes_path=prototypes_path,
        split="test",
        tau_multi=0.5,
        temperature=1.0,
        shortlist_topN=0,
        thresholds_path=None,
        calibration_path=None,
    )

    metrics_strict = run_global_evaluation(
        config_path=config_path,
        prototypes_path=prototypes_path,
        split="test",
        tau_multi=0.7,
        temperature=1.0,
        shortlist_topN=0,
        thresholds_path=None,
        calibration_path=None,
    )

    assert pytest.approx(metrics_loose["macro_recall"], rel=1e-5) == 1.0
    assert pytest.approx(metrics_strict["macro_recall"], rel=1e-5) == 0.75
    assert metrics_loose["macro_recall"] > metrics_strict["macro_recall"]
    assert pytest.approx(metrics_loose["coverage_ratio"], rel=1e-5) == 1.0


def test_global_eval_handles_zero_queries(tmp_path: Path) -> None:
    embeddings_base = tmp_path / "embeddings"
    keys = ["x1", "x2"]
    vectors = [[1.0, 0.0], [0.6, 0.8]]
    _write_embeddings(embeddings_base, keys, vectors)

    split_dir = tmp_path / "splits"
    _write_split(
        split_dir / "test.jsonl",
        [
            {"ec": "C", "accessions": ["x1"]},
            {"ec": "D", "accessions": ["x2"]},
        ],
    )

    outputs_dir = tmp_path / "outputs"
    ckpt_dir = outputs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.yaml"
    prototypes_path = tmp_path / "protos.npz"

    cfg = {
        "paths": {
            "embeddings": str(embeddings_base),
            "splits_dir": str(split_dir),
            "outputs": str(outputs_dir),
            "runs": str(tmp_path / "runs"),
        },
        "projection_dim": 2,
        "temperature": 1.0,
        "eval": {
            "mode": "global_support",
            "shortlist_topN": 0,
            "temperature": 1.0,
            "tau_multi": 0.5,
            "per_ec_thresholds_path": None,
            "subprototypes_per_ec": 1,
            "prototypes_path": str(prototypes_path),
            "calibration_path": None,
            "split": "test",
        },
    }

    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)

    model = build_model(cfg, input_dim=2, device=torch.device("cpu"))
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(2))
        model.proj.bias.zero_()
    torch.save({"model": model.state_dict()}, ckpt_dir / "protonet.pt")

    prototypes = {
        "A": np.array([[1.0, 0.0]], dtype=np.float32),
        "B": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    counts = {"A": 2, "B": 1}
    save_prototypes(prototypes_path, prototypes, train_counts=counts)

    metrics = run_global_evaluation(
        config_path=config_path,
        prototypes_path=prototypes_path,
        split="test",
        tau_multi=0.5,
        temperature=1.0,
        shortlist_topN=0,
        thresholds_path=None,
        calibration_path=None,
    )

    assert metrics["queries_evaluated"] == 0
    assert metrics["coverage_ratio"] == 0.0
    assert metrics["micro_f1"] == 0.0
