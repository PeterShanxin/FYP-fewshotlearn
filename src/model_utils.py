"""Shared helpers for loading configuration and ProtoNet models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .protonet import ProtoConfig, ProtoNet


def load_cfg(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pick_device(cfg: Dict[str, Any]) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def infer_input_dim(embeddings_path: Path) -> int:
    base = str(embeddings_path)
    if base.endswith(".npz"):
        base = base[:-4]
    x_path = Path(base + ".X.npy")
    if not x_path.exists():
        raise FileNotFoundError(f"Embeddings array not found: {x_path}")
    arr = np.load(x_path, mmap_mode="r")  # type: ignore[arg-type]
    return int(arr.shape[1])


def build_model(cfg: Dict[str, Any], input_dim: int, device: torch.device) -> ProtoNet:
    detector_cfg = cfg.get("detector", {}) or {}
    model_cfg = ProtoConfig(
        input_dim=input_dim,
        projection_dim=int(cfg.get("projection_dim", 256)),
        temperature=float(cfg.get("temperature", 10.0)),
        detector_enabled=bool(detector_cfg.get("enabled", False)),
        detector_hidden=int(detector_cfg.get("hidden_dim", 32)),
    )
    return ProtoNet(model_cfg).to(device)


def load_checkpoint(model: ProtoNet, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run training first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=False)  # type: ignore[index]
    model.eval()

