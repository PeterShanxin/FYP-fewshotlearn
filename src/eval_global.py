"""Global-support evaluation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from .episodic_sampler import SplitIndex
from .model_utils import build_model, infer_input_dim, load_checkpoint, load_cfg, pick_device
from .prototype_bank import load_prototypes


def _load_thresholds(path: Optional[Path]) -> Dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): float(v) for k, v in data.items()}


def _normalize(arr: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(arr, p=2, dim=-1)


def _compute_bucket_indices(train_counts: np.ndarray) -> Dict[str, np.ndarray]:
    buckets = {
        "tail": train_counts <= 5,
        "medium": (train_counts > 5) & (train_counts <= 25),
        "head": train_counts > 25,
    }
    return {k: np.where(mask)[0] for k, mask in buckets.items() if mask.any()}


def _metrics_for_bucket(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    micro_p = precision_score(yt, yp, zero_division=0)
    micro_r = recall_score(yt, yp, zero_division=0)
    micro_f1 = f1_score(yt, yp, zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }


@dataclass
class CoverageStats:
    total_queries: int
    covered_queries: int
    missing_embedding: int
    missing_labels: int


class GlobalSupportEvaluator:
    def __init__(
        self,
        cfg: dict,
        prototypes_path: Path,
        split: str,
        *,
        device: torch.device,
        shortlist_topN: int,
        per_ec_thresholds: Optional[Dict[str, float]] = None,
        ensure_top1: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.split = split
        self.per_ec_thresholds = per_ec_thresholds or {}
        self.ensure_top1 = ensure_top1
        self.shortlist_topN = int(shortlist_topN)
        self.show_progress = bool(show_progress)

        self.prototypes, self.train_counts = load_prototypes(prototypes_path)
        if not self.prototypes:
            raise ValueError("Loaded prototype bank is empty")

        self.class_names = sorted(self.prototypes.keys())
        self.num_classes = len(self.class_names)
        self.train_counts_array = np.array([self.train_counts.get(ec, 0) for ec in self.class_names], dtype=np.int32)

        paths = cfg["paths"]
        embeddings_path = Path(paths["embeddings"])
        ckpt_path = Path(paths["outputs"]) / "checkpoints" / "protonet.pt"
        input_dim = infer_input_dim(embeddings_path)
        self.model = build_model(cfg, input_dim, device)
        load_checkpoint(self.model, ckpt_path, device)

        proto_matrix, proto_offsets = self._stack_prototypes()
        self.class_logits = self._prepare_queries(
            Path(paths["splits_dir"]) / f"{split}.jsonl",
            embeddings_path,
            proto_matrix,
            proto_offsets,
        )

    def _stack_prototypes(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        protos: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for ec in self.class_names:
            arr = self.prototypes[ec].astype(np.float32)
            protos.append(arr)
            count = arr.shape[0]
            offsets.append((cursor, count))
            cursor += count
        matrix = torch.from_numpy(np.vstack(protos)).to(self.device)
        return matrix, offsets

    def _prepare_queries(
        self,
        split_path: Path,
        embeddings_path: Path,
        proto_matrix: torch.Tensor,
        proto_offsets: List[Tuple[int, int]],
    ) -> torch.Tensor:
        split = SplitIndex.from_jsonl(split_path)
        acc2ecs: Dict[str, List[str]] = {}
        for ec, accs in split.by_class.items():
            for acc in accs:
                acc2ecs.setdefault(acc, []).append(ec)
        total_queries = len(acc2ecs)

        base = str(embeddings_path)
        if base.endswith(".npz"):
            base = base[:-4]
        X_path = Path(base + ".X.npy")
        keys_path = Path(base + ".keys.npy")
        X = np.load(X_path, mmap_mode="r")  # type: ignore[arg-type]
        keys = np.load(keys_path, allow_pickle=False)  # type: ignore[arg-type]
        key2row = {str(k): int(i) for i, k in enumerate(keys.tolist())}  # type: ignore[attr-defined]

        rows: List[int] = []
        y_true_rows: List[List[int]] = []
        missing_embedding = 0
        missing_labels = 0

        acc_iter = acc2ecs.items()
        progress_iter = None
        if self.show_progress:
            progress_iter = tqdm(
                acc_iter,
                total=total_queries,
                desc="[global] collecting queries",
                dynamic_ncols=True,
                leave=False,
            )
            acc_iter = progress_iter
        for acc, ecs in acc_iter:
            if acc not in key2row:
                missing_embedding += 1
                continue
            labels = [self.class_names.index(ec) for ec in ecs if ec in self.prototypes]
            if not labels:
                missing_labels += 1
                continue
            rows.append(key2row[acc])
            y_true_rows.append(labels)

        if progress_iter is not None:
            progress_iter.close()

        covered_queries = len(rows)
        self.coverage = CoverageStats(
            total_queries=total_queries,
            covered_queries=covered_queries,
            missing_embedding=missing_embedding,
            missing_labels=missing_labels,
        )

        if covered_queries == 0:
            self.y_true = np.zeros((0, self.num_classes), dtype=np.float32)
            return torch.zeros((0, self.num_classes), dtype=torch.float32)

        # Build multi-hot targets
        y_true = np.zeros((covered_queries, self.num_classes), dtype=np.float32)
        for i, labels in enumerate(y_true_rows):
            for lab in labels:
                y_true[i, lab] = 1.0
        self.y_true = y_true

        # Embed queries and compute logits (pre-temperature)
        query_embeddings: List[torch.Tensor] = []
        batch_size = 1024
        with torch.no_grad():
            batch_iter = range(0, covered_queries, batch_size)
            progress_embed = None
            if self.show_progress:
                progress_embed = tqdm(
                    batch_iter,
                    desc="[global] embedding queries",
                    dynamic_ncols=True,
                    leave=False,
                )
                batch_iter = progress_embed
            for start in batch_iter:
                chunk = rows[start : start + batch_size]
                batch_np = np.stack([X[idx] for idx in chunk]).astype(np.float32)
                tensor = torch.from_numpy(batch_np).to(self.device)
                embedded = self.model.embed(tensor)
                query_embeddings.append(embedded)
            if progress_embed is not None:
                progress_embed.close()
        Z = torch.cat(query_embeddings, dim=0)

        scores = Z @ proto_matrix.T  # [Q, total_proto]
        class_logits = torch.empty((Z.shape[0], self.num_classes), device=self.device)
        for idx, (start, count) in enumerate(proto_offsets):
            if count == 1:
                class_logits[:, idx] = scores[:, start]
            else:
                class_logits[:, idx] = torch.logsumexp(scores[:, start : start + count], dim=1)
        return class_logits.detach().cpu()

    def evaluate(
        self,
        *,
        temperature: float,
        tau_multi: float,
        shortlist_topN: Optional[int] = None,
        per_ec_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        shortlist = self.shortlist_topN if shortlist_topN is None else int(shortlist_topN)
        thresholds = torch.full((self.num_classes,), float(tau_multi), dtype=torch.float32)
        thresh_map = self.per_ec_thresholds.copy()
        if per_ec_thresholds:
            thresh_map.update(per_ec_thresholds)
        for idx, ec in enumerate(self.class_names):
            if ec in thresh_map:
                thresholds[idx] = float(thresh_map[ec])

        logits = self.class_logits / float(temperature)
        y_true = self.y_true

        if logits.shape[0] == 0:
            overall = {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            }
            bucket_indices = _compute_bucket_indices(self.train_counts_array)
            for name in bucket_indices.keys():
                overall.update(
                    {
                        f"{name}_micro_f1": 0.0,
                        f"{name}_macro_f1": 0.0,
                        f"{name}_micro_recall": 0.0,
                        f"{name}_micro_precision": 0.0,
                    }
                )
            overall.update(
                {
                    "temperature": float(temperature),
                    "tau_multi": float(tau_multi),
                    "shortlist_topN": int(shortlist),
                    "queries_total": int(self.coverage.total_queries),
                    "queries_evaluated": int(self.coverage.covered_queries),
                    "coverage_ratio": 0.0,
                    "missing_embedding": int(self.coverage.missing_embedding),
                    "missing_labels": int(self.coverage.missing_labels),
                }
            )
            return overall

        probs = torch.sigmoid(logits)

        # Top-1 hit accuracy (argmax against multi-hot ground truth),
        # analogous to episodic "acc_top1_hit" for comparability.
        if y_true.size > 0:
            top1_idx = probs.argmax(dim=1).cpu().numpy()
            top1_hits = (y_true[np.arange(y_true.shape[0]), top1_idx] > 0.5).astype(np.float32)
            acc_top1 = float(top1_hits.mean())
        else:
            acc_top1 = 0.0

        if shortlist > 0 and shortlist < self.num_classes:
            topk = torch.topk(probs, k=shortlist, dim=1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk.indices, True)
        else:
            mask = torch.ones_like(probs, dtype=torch.bool)

        thresholds = thresholds.to(probs.device).unsqueeze(0)
        decisions = torch.where(mask, probs >= thresholds, torch.zeros_like(probs, dtype=torch.bool))
        if self.ensure_top1:
            no_positive = decisions.sum(dim=1) == 0
            if no_positive.any():
                top1 = probs.argmax(dim=1)
                decisions[torch.arange(decisions.shape[0], device=probs.device), top1] = True

        y_pred = decisions.cpu().numpy().astype(np.int8)

        metrics = _metrics_for_bucket(y_true, y_pred)
        overall = {
            "acc_top1_hit": float(acc_top1),
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }

        bucket_indices = _compute_bucket_indices(self.train_counts_array)
        for name, idxs in bucket_indices.items():
            y_true_b = y_true[:, idxs]
            y_pred_b = y_pred[:, idxs]
            bucket_metrics = _metrics_for_bucket(y_true_b, y_pred_b)
            overall.update({
                f"{name}_micro_f1": bucket_metrics["micro_f1"],
                f"{name}_macro_f1": bucket_metrics["macro_f1"],
                f"{name}_micro_recall": bucket_metrics["micro_recall"],
                f"{name}_micro_precision": bucket_metrics["micro_precision"],
            })

        overall.update(
            {
                "temperature": float(temperature),
                "tau_multi": float(tau_multi),
                "shortlist_topN": int(shortlist),
                "queries_total": int(self.coverage.total_queries),
                "queries_evaluated": int(self.coverage.covered_queries),
                "coverage_ratio": float(
                    self.coverage.covered_queries / max(1, self.coverage.total_queries)
                ),
                "missing_embedding": int(self.coverage.missing_embedding),
                "missing_labels": int(self.coverage.missing_labels),
            }
        )
        return overall


def run_global_evaluation(
    config_path: Path,
    prototypes_path: Path,
    split: str,
    *,
    tau_multi: Optional[float] = None,
    temperature: Optional[float] = None,
    shortlist_topN: Optional[int] = None,
    thresholds_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
) -> Dict[str, float]:
    cfg = load_cfg(config_path)
    device = pick_device(cfg)
    eval_cfg = cfg.get("eval", {}) or {}
    show_progress = bool(cfg.get("progress", True))

    base_tau = float(eval_cfg.get("tau_multi", 0.35))
    base_temp = float(eval_cfg.get("temperature", 0.07))
    base_shortlist = int(eval_cfg.get("shortlist_topN", 0))

    tau = float(tau_multi if tau_multi is not None else base_tau)
    temp = float(temperature if temperature is not None else base_temp)
    shortlist = int(shortlist_topN if shortlist_topN is not None else base_shortlist)
    ensure_top1 = bool(eval_cfg.get("ensure_top1", True))

    thresholds = _load_thresholds(thresholds_path)
    if calibration_path is not None and calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calib = json.load(handle)
        if tau_multi is None and "tau_multi" in calib:
            tau = float(calib["tau_multi"])
        if temperature is None and "temperature" in calib:
            temp = float(calib["temperature"])

    evaluator = GlobalSupportEvaluator(
        cfg,
        Path(prototypes_path),
        split,
        device=device,
        shortlist_topN=shortlist,
        per_ec_thresholds=thresholds,
        ensure_top1=ensure_top1,
        show_progress=show_progress,
    )
    return evaluator.evaluate(
        temperature=temp,
        tau_multi=tau,
        shortlist_topN=shortlist,
    )
