"""Episodic training loop for Prototypical Networks with early stopping.

- Prints key config at start
- Tracks episodic val accuracy; saves best checkpoint
- Optionally saves last checkpoint and re-checks best vs last on a larger
  validation slice at the end to select the final checkpoint
- Writes results/history.json with simple metrics
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional
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


def load_sequence_lookup(joined_tsv: Path) -> Dict[str, str]:
    """Streaming TSV reader that builds accession → sequence map."""
    lookup: Dict[str, str] = {}
    if not joined_tsv.exists():
        raise FileNotFoundError(f"Joined TSV not found for sequence lookup: {joined_tsv}")
    with open(joined_tsv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Joined TSV lacks header: {joined_tsv}")
        lower_map = {name.lower(): name for name in reader.fieldnames}
        acc_key = lower_map.get("accession")
        seq_key = lower_map.get("sequence")
        if acc_key is None or seq_key is None:
            raise KeyError(
                f"Joined TSV must contain 'accession' and 'sequence' columns (found: {reader.fieldnames})"
            )
        for row in reader:
            acc = row.get(acc_key)
            seq = row.get(seq_key)
            if not acc or not seq:
                continue
            lookup[str(acc)] = str(seq).upper()
    return lookup


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
    coverage = val_sampler.class_coverage(K, Q)
    print(
        "[val][coverage] classes with ≥K+Q: {full}/{total} | tail (≥K,<K+Q): {tail} | excluded: {excluded}".format(
            full=coverage["eligible_full"],
            total=coverage["total_classes"],
            tail=coverage["eligible_support_only"],
            excluded=coverage["excluded"],
        )
    )
    if coverage["eligible_full"] < M:
        print(
            "[val][coverage] WARNING: only {full} classes can fill support+query without fallback; requesting M={M}".format(
                full=coverage["eligible_full"],
                M=M,
            )
        )
    with torch.no_grad():
        with evaluating(model):
            for _ in trange(
                episodes,
                desc=f"[val] episodes (M={M}, K={K}, Q={Q})",
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


def _state_dict_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Copy a state_dict to CPU (to reduce GPU memory pressure while caching)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        try:
            out[k] = v.detach().to("cpu")
        except Exception:
            out[k] = v.detach()
    return out


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
    sampler_cfg = cfg.get("sampler", {}) or {}
    disjoint = bool(sampler_cfg.get("identity_disjoint", cfg.get("identity_disjoint", False)))
    with_replacement_fallback = bool(sampler_cfg.get("with_replacement_fallback", False))
    fallback_scope = sampler_cfg.get("fallback_scope", "train_only")
    rare_class_boost = sampler_cfg.get("rare_class_boost", "none")

    seq_lookup: Optional[Dict[str, str]] = None
    if with_replacement_fallback:
        seq_lookup = load_sequence_lookup(Path(paths["joined_tsv"]))

    runs_root = Path(paths.get("runs", Path(paths["outputs"]).parent / "runs"))
    exp_id = str(cfg.get("exp_id") or cfg.get("run_id") or Path(paths["outputs"]).name)
    usage_dir = runs_root / exp_id

    # Samplers
    train_sampler = EpisodeSampler(
        Path(paths["embeddings"]),
        Path(paths["splits_dir"]) / "train.jsonl",
        device,
        seed=cfg.get("random_seed", 42),
        phase="train",
        multi_label=multi_label,
        clusters_tsv=clusters_tsv,
        disjoint_support_query=disjoint,
        with_replacement_fallback=with_replacement_fallback,
        fallback_scope=fallback_scope,
        rare_class_boost=rare_class_boost,
        sequence_lookup=seq_lookup,
        usage_log_dir=usage_dir,
    )
    val_sampler = EpisodeSampler(
        Path(paths["embeddings"]),
        Path(paths["splits_dir"]) / "val.jsonl",
        device,
        seed=cfg.get("random_seed", 42) + 1,
        phase="val",
        multi_label=multi_label,
        clusters_tsv=clusters_tsv,
        disjoint_support_query=disjoint,
        with_replacement_fallback=with_replacement_fallback,
        fallback_scope=fallback_scope,
        rare_class_boost="none",
        sequence_lookup=seq_lookup if fallback_scope == "all" else None,
    )
    episode_cfg = cfg.get("episode", {}) or {}

    def _resolve_episode_value(primary: str, fallbacks: list[str], default: int) -> int:
        keys = [primary] + fallbacks
        for key in keys:
            if key not in episode_cfg:
                continue
            val = episode_cfg[key]
            if isinstance(val, list):
                if not val:
                    continue
                return int(val[0])
            try:
                return int(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return int(default)

    M_train = _resolve_episode_value("M_train", ["M"], 10)
    K_train = _resolve_episode_value("K_train", ["K"], 1)
    Q_train = _resolve_episode_value("Q_train", ["Q"], 5)
    M_val = _resolve_episode_value("M_val", ["M"], M_train)
    K_val = _resolve_episode_value("K_val", ["K_eval", "K"], K_train)
    Q_val = _resolve_episode_value("Q_val", ["Q_eval", "Q"], Q_train)

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
    pcfg = ProtoConfig(
        input_dim=train_sampler.dim,
        projection_dim=int(cfg.get("projection_dim", 256)),
        temperature=float(cfg.get("temperature", 10.0)),
    )
    model = ProtoNet(pcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = bool(cfg.get("fp16_train", False)) and device.type == "cuda"
    # Torch 2.x introduces torch.amp.GradScaler with a device_type argument; older
    # releases only provide torch.cuda.amp.GradScaler without that parameter. Pick
    # whichever variant is available to stay compatible across versions.
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        grad_scaler_cls = amp_mod.GradScaler
    else:
        grad_scaler_cls = torch.cuda.amp.GradScaler

    try:
        scaler = grad_scaler_cls(device_type=device.type, enabled=use_amp)
    except TypeError:
        scaler = grad_scaler_cls(enabled=use_amp)
    n_train = int(cfg["episodes"]["train"])
    n_val_eval = max(1, int(cfg["episodes"]["val"]))
    # Validation cadence and size knobs
    eval_every = int(cfg.get("eval_every", max(50, n_train // 10)))
    episodes_per_val_check = int(cfg.get("episodes_per_val_check", 50))
    # Early stopping and improvement sensitivity
    patience_checks = int(cfg.get("patience_checks", 10))
    min_delta = float(cfg.get("min_delta", 0.0))
    # Save options and final selection
    save_last = bool(cfg.get("save_last", True))
    final_val_episodes = int(cfg.get("final_val_episodes", 0))

    show_progress = bool(cfg.get("progress", True))
    verbose = bool(cfg.get("verbose", show_progress))

    print("[train] config:")
    print(json.dumps({
        "device": str(device),
        "gpus_requested": requested_gpus,
        "M_train": M_train,
        "K_train": K_train,
        "Q_train": Q_train,
        "M_val": M_val,
        "K_val": K_val,
        "Q_val": Q_val,
        "episodes_train": n_train,
        "projection_dim": int(cfg.get("projection_dim", 256)),
        "temperature": float(cfg.get("temperature", 10.0)),
        "fp16_train": use_amp,
        "eval_every": eval_every,
        "episodes_per_val_check": episodes_per_val_check,
        "patience_checks": patience_checks,
        "min_delta": min_delta,
        "save_last": save_last,
        "final_val_episodes": final_val_episodes,
        "multi_label": multi_label,
        "identity_disjoint": disjoint,
        "rare_class_boost": rare_class_boost,
        "fallback_scope": fallback_scope,
        "has_val": has_val,
    }, indent=2))
    if requested_gpus > 1 and device.type == "cuda":
        print("[train] Note: training runs on a single GPU; multi-GPU is used for embeddings only.")

    best_acc = -1.0
    best_state = None
    last_state = None
    history = {"val_acc": [], "checkpoints_after_episode": []}
    checks_without_improve = 0

    model.train()
    pbar = tqdm(
        range(n_train), desc="episodes", dynamic_ncols=True, leave=True, disable=not show_progress
    )
    for ep in pbar:
        sx, sy, qx, qy, classes = train_sampler.sample_episode(M_train, K_train, Q_train)
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
                model,
                val_sampler,
                M_val,
                K_val,
                Q_val,
                episodes=min(episodes_per_val_check, n_val_eval),
                show_progress=show_progress,
                multi_label=multi_label,
            )
            history["val_acc"].append(acc)
            history["checkpoints_after_episode"].append(ep + 1)
            if show_progress:
                pbar.set_postfix({"val_acc": f"{acc:.4f}", "best": f"{max(best_acc, 0):.4f}"}, refresh=True)
            elif verbose:
                print(f"[train] episode {ep+1}: val_acc={acc:.4f}")
            if acc > (best_acc + min_delta):
                best_acc = acc
                best_state = {"model": _state_dict_cpu(model.state_dict())}
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

    # Prepare checkpoints directory
    out_dir = Path(paths["outputs"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always capture the last state dict
    try:
        last_state = {"model": _state_dict_cpu(model.state_dict())}
    except Exception:
        last_state = {"model": model.state_dict()}

    # Ensure we have at least something to save as best
    if best_state is None:
        best_state = last_state

    # Optional final re-check on a larger validation slice: best vs last
    final_choice = "best"
    final_best_acc = best_acc
    if has_val and final_val_episodes > 0:
        if verbose:
            print(f"[train] Final re-check: evaluating best vs last on {final_val_episodes} val episodes …")
        sampler_state = val_sampler.snapshot_state()

        # Evaluate best
        try:
            model.load_state_dict(best_state["model"], strict=False)  # type: ignore[index]
        except Exception:
            pass
        acc_best = episodic_accuracy(
            model,
            val_sampler,
            M_val,
            K_val,
            Q_val,
            episodes=min(final_val_episodes, n_val_eval),
            show_progress=show_progress,
            multi_label=multi_label,
        )
        val_sampler.restore_state(sampler_state)

        # Evaluate last
        try:
            model.load_state_dict(last_state["model"], strict=False)  # type: ignore[index]
        except Exception:
            pass
        acc_last = episodic_accuracy(
            model,
            val_sampler,
            M_val,
            K_val,
            Q_val,
            episodes=min(final_val_episodes, n_val_eval),
            show_progress=show_progress,
            multi_label=multi_label,
        )
        # Restore sampler state so subsequent consumers see a consistent progression
        val_sampler.restore_state(sampler_state)
        if acc_last > acc_best + 1e-12:
            final_choice = "last"
            final_best_acc = acc_last
            if verbose:
                print(f"[train] Final selection: last (acc={acc_last:.4f}) > best (acc={acc_best:.4f})")
        else:
            final_choice = "best"
            final_best_acc = acc_best
            if verbose:
                print(f"[train] Final selection: best (acc={acc_best:.4f}) >= last (acc={acc_last:.4f})")
        # Load the chosen weights back into the model for downstream save
        chosen = best_state if final_choice == "best" else last_state
        try:
            model.load_state_dict(chosen["model"], strict=False)  # type: ignore[index]
        except Exception:
            pass
    else:
        # Keep model as-is (last) if no final re-check requested
        final_choice = "best"
        final_best_acc = best_acc

    # Save checkpoints
    ckpt_path = out_dir / "protonet.pt"
    chosen_state = {"model": model.state_dict()}
    torch.save(chosen_state, ckpt_path)
    # Optionally persist best and last for inspection
    if save_last:
        try:
            torch.save(best_state, out_dir / "protonet.best.pt")
        except Exception:
            pass
        try:
            torch.save(last_state, out_dir / "protonet.last.pt")
        except Exception:
            pass

    # Save history
    hist_path = Path(paths["outputs"]) / "history.json"
    with open(hist_path, "w") as f:
        json.dump(
            dict(
                history=history,
                best_val_acc=float(best_acc),
                final_choice=str(final_choice),
                final_val_acc=float(final_best_acc if final_best_acc is not None else best_acc),
            ),
            f,
            indent=2,
        )

    try:
        train_sampler.write_usage_csv(usage_dir)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[train] WARNING: failed to write sampler stats ({exc})")

    print(f"[train] best_val_acc={best_acc:.4f} | final_choice={final_choice} | checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
