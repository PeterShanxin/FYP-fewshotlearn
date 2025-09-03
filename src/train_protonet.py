"""Episodic training loop for Prototypical Networks with early stopping.

- Prints key config at start
- Tracks episodic val accuracy; saves best checkpoint
- Writes results/history.json with simple metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
from contextlib import contextmanager

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm, trange

from .episodic_sampler import EpisodeSampler
from .protonet import ProtoConfig, ProtoNet


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(cfg: dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def ensure_dirs(paths: Dict[str, str]) -> None:
    Path(paths["outputs"]).mkdir(parents=True, exist_ok=True)
    Path(paths["embeddings"]).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def evaluating(model: ProtoNet):
    """Temporarily set the model to eval mode."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def episodic_accuracy(
    model: ProtoNet,
    val_sampler: EpisodeSampler,
    M: int,
    K: int,
    Q: int,
    episodes: int,
    show_progress: bool = True,
) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        with evaluating(model):
            for _ in trange(
                episodes,
                desc="[val] episodes",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
                position=1,
            ):
                sx, sy, qx, qy = val_sampler.sample_episode(M, K, Q)
                pred = model.predict(sx, sy, qx)
                correct += int((pred == qy).sum().item())
                total += int(qy.numel())
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)
    ensure_dirs(paths)

    torch.manual_seed(cfg.get("random_seed", 42))
    np.random.seed(cfg.get("random_seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("random_seed", 42))

    # Samplers
    train_sampler = EpisodeSampler(paths["embeddings"], Path(paths["splits_dir"]) / "train.jsonl", device, seed=cfg.get("random_seed", 42))
    val_sampler   = EpisodeSampler(paths["embeddings"], Path(paths["splits_dir"]) / "val.jsonl", device, seed=cfg.get("random_seed", 42) + 1)

    # Model
    pcfg = ProtoConfig(input_dim=train_sampler.dim, projection_dim=int(cfg.get("projection_dim", 256)), temperature=float(cfg.get("temperature", 10.0)))
    model = ProtoNet(pcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    M = int(cfg["episode"]["M"])
    K = int(cfg["episode"]["K_train"])
    Q = int(cfg["episode"]["Q"])
    n_train = int(cfg["episodes"]["train"])
    n_val_eval = max(1, int(cfg["episodes"]["val"]))

    show_progress = bool(cfg.get("progress", True))
    verbose = bool(cfg.get("verbose", show_progress))

    print("[train] config:")
    print(json.dumps({
        "device": str(device), "M": M, "K_train": K, "Q": Q, "episodes_train": n_train,
        "projection_dim": int(cfg.get("projection_dim", 256)),
        "temperature": float(cfg.get("temperature", 10.0))
    }, indent=2))

    best_acc = -1.0
    best_state = None
    history = {"val_acc": [], "checkpoints_after_episode": []}
    patience_checks = 10
    checks_without_improve = 0
    eval_every = max(50, n_train // 10)

    model.train()
    pbar = tqdm(
        range(n_train), desc="episodes", dynamic_ncols=True, leave=True, disable=not show_progress
    )
    for ep in pbar:
        sx, sy, qx, qy = train_sampler.sample_episode(M, K, Q)
        logits, loss = model(sx, sy, qx, qy)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % eval_every == 0 or ep == n_train - 1:
            acc = episodic_accuracy(
                model, val_sampler, M, K, Q, episodes=min(50, n_val_eval), show_progress=show_progress
            )
            history["val_acc"].append(acc)
            history["checkpoints_after_episode"].append(ep + 1)
            if show_progress:
                pbar.set_postfix({"val_acc": f"{acc:.4f}", "best": f"{max(best_acc, 0):.4f}"}, refresh=True)
            elif verbose:
                print(f"[train] episode {ep+1}: val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_state = {"model": model.state_dict()}
                checks_without_improve = 0
                if show_progress:
                    pbar.set_postfix({"val_acc": f"{acc:.4f}", "best": f"{best_acc:.4f}", "note": "new best"}, refresh=True)
                elif verbose:
                    print(f"[train] best_val_acc improved to {best_acc:.4f} | checkpoint will be saved")
            else:
                checks_without_improve += 1
            if checks_without_improve >= patience_checks:
                if verbose:
                    print("[train] Early stopping triggered.")
                break

    if show_progress:
        pbar.close()

    # Save best checkpoint
    out_dir = Path(paths["outputs"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "protonet.pt"
    if best_state is None:
        best_state = {"model": model.state_dict()}
    torch.save(best_state, ckpt_path)

    # Save history
    hist_path = Path(paths["outputs"]) / "history.json"
    with open(hist_path, "w") as f:
        json.dump(dict(history=history, best_val_acc=float(best_acc)), f, indent=2)

    print(f"[train] best_val_acc={best_acc:.4f} | checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
