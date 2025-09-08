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
    multi_label: bool = False,
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
                sx, sy, qx, qy, _classes = val_sampler.sample_episode(M, K, Q)
                pred = model.predict(sx, sy, qx)
                if multi_label and qy.dim() == 2:
                    take = torch.gather(qy, 1, pred.view(-1, 1)).squeeze(1)
                    correct += int((take > 0.5).sum().item())
                    total += int(qy.shape[0])
                else:
                    correct += int((pred == qy).sum().item())
                    total += int(qy.numel())
    return correct / max(total, 1)


def _apply_hierarchy_loss(
    model: ProtoNet,
    s: torch.Tensor,
    q: torch.Tensor,
    sy: torch.Tensor,
    qy: torch.Tensor,
    classes: list,
    cfg: dict,
    multi_label: bool,
) -> torch.Tensor:
    """Optional hierarchical supervision over EC levels."""
    hier_levels = int(cfg.get("hierarchy_levels", 0))
    hier_w = float(cfg.get("hierarchy_weight", 0.0))
    if hier_levels <= 0 or hier_w <= 0.0:
        return s.new_tensor(0.0)

    device = s.device
    use_lvls = [lvl for lvl in range(1, min(hier_levels, 3) + 1)]
    per_w = hier_w / max(len(use_lvls), 1)
    extra = s.new_tensor(0.0)

    for lvl in use_lvls:
        parts = [ec.split('.') for ec in classes]
        coarse = ['.'.join(p[:lvl]) for p in parts]
        uniq = {t: i for i, t in enumerate(sorted(set(coarse)))}
        if len(uniq) < 2:
            continue

        y_coarse = torch.tensor(
            [uniq[coarse[int(y.item())]] for y in sy],
            dtype=torch.long,
            device=device,
        )
        P_c = model.prototypes(s, y_coarse)
        logits_c = (q @ P_c.T) / model.temp

        if multi_label and qy.dim() == 2:
            q_c = torch.zeros((qy.shape[0], len(uniq)), dtype=torch.float32, device=device)
            for j, t in enumerate(coarse):
                gid = uniq[t]
                q_c[:, gid] = torch.maximum(q_c[:, gid], qy[:, j])
            extra = extra + per_w * torch.nn.functional.binary_cross_entropy_with_logits(logits_c, q_c)
        else:
            yq_c = torch.tensor(
                [uniq[coarse[int(y.item())]] for y in qy],
                dtype=torch.long,
                device=device,
            )
            extra = extra + per_w * torch.nn.functional.cross_entropy(logits_c, yq_c)

    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    device = pick_device(cfg)
    requested_gpus = int(cfg.get("gpus", 1))
    ensure_dirs(paths)

    torch.manual_seed(cfg.get("random_seed", 42))
    np.random.seed(cfg.get("random_seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("random_seed", 42))

    # Options
    multi_label = bool(cfg.get("multi_label", False))
    clusters_tsv = Path(paths.get("clusters_tsv", "")) if paths.get("clusters_tsv") else None
    disjoint = bool(cfg.get("identity_disjoint", False))

    # Samplers
    train_sampler = EpisodeSampler(
        Path(paths["embeddings"]), Path(paths["splits_dir"]) / "train.jsonl", device,
        seed=cfg.get("random_seed", 42), multi_label=multi_label,
        clusters_tsv=clusters_tsv, disjoint_support_query=disjoint,
    )
    val_sampler   = EpisodeSampler(
        Path(paths["embeddings"]), Path(paths["splits_dir"]) / "val.jsonl", device,
        seed=cfg.get("random_seed", 42) + 1, multi_label=multi_label,
        clusters_tsv=clusters_tsv, disjoint_support_query=disjoint,
    )
    # Detect empty val split to optionally skip validation
    has_val = False
    try:
        val_path = Path(paths["splits_dir"]) / "val.jsonl"
        if val_path.exists():
            with open(val_path, "r", encoding="utf-8") as f:
                for _ in f:
                    has_val = True
                    break
    except Exception:
        has_val = False

    # Model
    pcfg = ProtoConfig(input_dim=train_sampler.dim, projection_dim=int(cfg.get("projection_dim", 256)), temperature=float(cfg.get("temperature", 10.0)))
    model = ProtoNet(pcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = bool(cfg.get("fp16_train", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    M = int(cfg["episode"]["M"])
    K = int(cfg["episode"]["K_train"])
    Q = int(cfg["episode"]["Q"])
    n_train = int(cfg["episodes"]["train"])
    n_val_eval = max(1, int(cfg["episodes"]["val"]))
    # Validation cadence and size knobs
    eval_every = int(cfg.get("eval_every", max(50, n_train // 10)))
    episodes_per_val_check = int(cfg.get("episodes_per_val_check", 50))

    show_progress = bool(cfg.get("progress", True))
    verbose = bool(cfg.get("verbose", show_progress))

    print("[train] config:")
    print(json.dumps({
        "device": str(device), "gpus_requested": requested_gpus, "M": M, "K_train": K, "Q": Q, "episodes_train": n_train,
        "projection_dim": int(cfg.get("projection_dim", 256)),
        "temperature": float(cfg.get("temperature", 10.0)),
        "fp16_train": use_amp,
        "eval_every": eval_every,
        "episodes_per_val_check": episodes_per_val_check,
        "multi_label": multi_label,
        "identity_disjoint": disjoint,
        "has_val": has_val,
    }, indent=2))
    if requested_gpus > 1 and device.type == "cuda":
        print("[train] Note: training runs on a single GPU; multi-GPU is used for embeddings only.")

    best_acc = -1.0
    best_state = None
    history = {"val_acc": [], "checkpoints_after_episode": []}
    patience_checks = 10
    checks_without_improve = 0

    model.train()
    pbar = tqdm(
        range(n_train), desc="episodes", dynamic_ncols=True, leave=True, disable=not show_progress
    )
    for ep in pbar:
        sx, sy, qx, qy, classes = train_sampler.sample_episode(M, K, Q)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, loss = model(sx, sy, qx, qy)
                s = model.embed(sx)
                q = model.embed(qx)
                loss = loss + _apply_hierarchy_loss(model, s, q, sy, qy, classes, cfg, multi_label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits, loss = model(sx, sy, qx, qy)
            s = model.embed(sx)
            q = model.embed(qx)
            loss = loss + _apply_hierarchy_loss(model, s, q, sy, qy, classes, cfg, multi_label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if has_val and ((ep + 1) % eval_every == 0 or ep == n_train - 1):
            acc = episodic_accuracy(
                model, val_sampler, M, K, Q, episodes=min(episodes_per_val_check, n_val_eval), show_progress=show_progress, multi_label=multi_label
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
