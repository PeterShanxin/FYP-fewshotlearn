"""On-the-fly M-way K-shot episodic sampler with rare-class fallback.

Loads:
- embeddings: contiguous memory-mapped arrays `embeddings.X.npy` (+ `embeddings.keys.npy`)
- split JSONL files (train/val/test) mapping ec -> accessions

Supports:
- identity-disjoint sampling via optional accession→cluster mapping
- frequency-aware class sampling
- targeted with-replacement fallback for under-filled classes (train only)
- lightweight embedding-space augmentation to decorrelate duplicated samples
- usage accounting with CSV export for diagnostics
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm

    def _progress_write(msg: str) -> None:
        """Write sampler logs without breaking active tqdm bars."""

        _tqdm.write(msg)

except ImportError:  # pragma: no cover - tqdm is an optional runtime dep

    def _progress_write(msg: str) -> None:
        print(msg)

from . import augment


class ClusterShortageError(RuntimeError):
    """Raised when a class lacks enough clusters for disjoint sampling."""

    def __init__(self, ec: str, have: int, need: int) -> None:
        super().__init__(
            f"Class {ec} has only {have} clusters available but requires {need} "
            "for disjoint support/query sampling."
        )
        self.ec = ec
        self.have = have
        self.need = need


@dataclass
class SplitIndex:
    """Index of EC classes to accession lists."""

    by_class: Dict[str, List[str]]
    classes: List[str]

    @staticmethod
    def from_jsonl(path: Path) -> "SplitIndex":
        by: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                by[obj["ec"]] = list(obj["accessions"])
        return SplitIndex(by_class=by, classes=sorted(by))


class EpisodeSampler:
    """Samples episodes (M-way, K-shot, Q queries per class)."""

    def __init__(
        self,
        embeddings_npz: Path,
        split_jsonl: Path,
        device: torch.device,
        seed: int = 42,
        *,
        phase: str = "train",
        multi_label: bool = False,
        clusters_tsv: Optional[Path] = None,
        disjoint_support_query: bool = False,
        with_replacement_fallback: bool = False,
        fallback_scope: str = "train_only",
        rare_class_boost: str = "none",
        sequence_lookup: Optional[Dict[str, str]] = None,
        usage_log_dir: Optional[Path] = None,
        view_dropout: float = 0.08,
        view_noise_sigma: float = 0.01,
    ) -> None:
        self.device = device
        self.phase = phase.lower()
        self._base_seed = int(seed)
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.multi_label = bool(multi_label)
        self.disjoint_support_query = bool(disjoint_support_query)
        self._eval_tail_fallback = self.phase in {"val", "test"}

        fallback_scope_norm = (fallback_scope or "train_only").lower().strip()
        if fallback_scope_norm not in {"train_only", "all", "none"}:
            fallback_scope_norm = "train_only"
        self.fallback_scope = fallback_scope_norm
        self.with_replacement_fallback = bool(with_replacement_fallback)
        self.allow_fallback = self.with_replacement_fallback and (
            (self.phase == "train" and self.fallback_scope in {"train_only", "all"})
            or (self.phase != "train" and self.fallback_scope == "all")
        )
        self.rare_class_boost = (rare_class_boost or "none").lower().strip()
        if self.rare_class_boost not in {"none", "inverse_log_freq"}:
            self.rare_class_boost = "none"

        # Embeddings (contiguous arrays)
        emb_path = Path(embeddings_npz)
        base_str = str(emb_path)
        if base_str.endswith(".npz"):
            base_str = base_str[:-4]
        X_path = Path(base_str + ".X.npy")
        keys_path = Path(base_str + ".keys.npy")
        if not (X_path.exists() and keys_path.exists()):
            raise FileNotFoundError(
                f"[load-emb] contiguous embeddings not found. Expected: {X_path} and {keys_path}. "
                f"Run the embedding step to generate them."
            )

        self.X = np.load(X_path, mmap_mode="r")  # type: ignore[attr-defined]
        self.keys = np.load(keys_path, allow_pickle=False)  # type: ignore[attr-defined]
        if self.X.shape[0] != self.keys.shape[0]:
            raise ValueError(
                f"[load-emb] X rows ({self.X.shape[0]}) != keys ({self.keys.shape[0]})"
            )
        print(f"[load-emb] using contiguous X.npy (mmap): shape={self.X.shape}")
        self.key2row: Dict[str, int] = {k: i for i, k in enumerate(self.keys.tolist())}  # type: ignore[attr-defined]
        self.dim = int(self.X.shape[1])

        # Split/index structures
        self.split = SplitIndex.from_jsonl(Path(split_jsonl))
        self.acc2ecs: Dict[str, Set[str]] = defaultdict(set)
        for ec, accs in self.split.by_class.items():
            for a in accs:
                self.acc2ecs[a].add(ec)
        self.class2idx: Dict[str, List[int]] = {}
        for ec, accs in self.split.by_class.items():
            rows = [self.key2row[a] for a in accs if a in self.key2row]
            if rows:
                self.class2idx[ec] = rows
        self.class_counts: Dict[str, int] = {ec: len(rows) for ec, rows in self.class2idx.items()}

        # Optional: accession → cluster map for identity-aware sampling
        self.acc2cluster: Dict[str, str] = {}
        if clusters_tsv is not None:
            p = Path(clusters_tsv)
            if not p.exists():
                raise FileNotFoundError(f"clusters_tsv not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    a, cid = parts[0], parts[1]
                    self.acc2cluster[a] = cid

        self.class_cluster_counts: Dict[str, int] = {}
        if self.disjoint_support_query and self.acc2cluster:
            for ec, rows in self.class2idx.items():
                clusters = {self.acc2cluster.get(self._idx2acc(idx), f"_na_{idx}") for idx in rows}
                self.class_cluster_counts[ec] = len(clusters)

        # Sequence lookup (needed for augmentation in fallback mode)
        self.seq_lookup: Dict[str, str] = dict(sequence_lookup or {})
        if self.allow_fallback and not self.seq_lookup:
            print(
                "[sampler] WARNING: fallback enabled but no sequence lookup provided; "
                "fallback views will rely on stochastic noise only."
            )
        self._warned_seq_missing = False

        # Usage accounting
        self.usage_support_counts: DefaultDict[str, int] = defaultdict(int)
        self.usage_query_counts: DefaultDict[str, int] = defaultdict(int)
        self.fallback_per_ec: DefaultDict[str, int] = defaultdict(int)
        self.total_support = 0
        self.total_query = 0
        self.fallback_events = 0
        self._stats_dirty = False
        self.usage_log_path = (usage_log_dir / "sampler_stats.csv") if usage_log_dir else None
        self._current_underfilled: Set[str] = set()
        self._last_pick_stats: Dict[str, int] = {}
        self._skipped_single_cluster: Set[str] = set()

        # Cluster shortage diagnostics (for disjoint support/query sampling)
        self.cluster_shortage_counts: DefaultDict[str, int] = defaultdict(int)
        self.cluster_shortage_events = 0
        self.cluster_shortage_dropped_episodes = 0
        self.cluster_shortage_last_drop: Dict[str, int] = {}
        self._max_cluster_resample_attempts = 10

        # View augmentation hyper-parameters
        self._view_dropout = float(view_dropout)
        self._view_noise_sigma = float(view_noise_sigma)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Return a copy of the internal RNG state for reproducible replays."""

        return {
            "random": self.rng.getstate(),
            "np": copy.deepcopy(self._np_rng.bit_generator.state),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore RNG state captured via :meth:`snapshot_state`."""

        if not state:
            return
        rand_state = state.get("random")
        if rand_state is not None:
            self.rng.setstate(rand_state)
        np_state = state.get("np")
        if np_state is not None:
            self._np_rng.bit_generator.state = np_state

    def _idx2acc(self, idx: int) -> str:
        return str(self.keys[idx])  # type: ignore[index]

    def _class_weight(self, ec: str) -> float:
        freq = max(1, self.class_counts.get(ec, 0))
        if self.rare_class_boost == "inverse_log_freq":
            denom = math.log(1.0 + float(freq))
            return 1.0 / denom if denom > 0 else 0.0
        return 1.0

    def _weighted_sample_classes(self, candidates: List[str], count: int) -> List[str]:
        if count <= 0:
            return []
        if self.rare_class_boost == "inverse_log_freq" and len(candidates) > 0:
            weights = np.array([self._class_weight(ec) for ec in candidates], dtype=np.float64)
            total = float(weights.sum())
            if total > 0:
                if count >= len(candidates):
                    chosen = candidates[:]
                    self.rng.shuffle(chosen)
                    return chosen
                probs = weights / total
                chosen = self._np_rng.choice(candidates, size=count, replace=False, p=probs)
                return chosen.tolist()
        pool = candidates[:]
        self.rng.shuffle(pool)
        return pool[:count]

    def _pick_classes(self, M: int, need: int, support: int, allow_underfilled: bool) -> List[str]:
        eligible_full: List[str] = []
        eligible_support_only: List[str] = []
        total_classes = 0
        require_clusters = self.disjoint_support_query and bool(self.acc2cluster)
        for ec in self.split.classes:
            count = self.class_counts.get(ec, 0)
            if count == 0:
                continue
            total_classes += 1
            if require_clusters:
                cluster_count = self.class_cluster_counts.get(ec, 0)
                if cluster_count < 2:
                    self._note_single_cluster_skip(ec, cluster_count)
                    continue
            if count >= need:
                eligible_full.append(ec)
            elif allow_underfilled and count >= support:
                eligible_support_only.append(ec)

        allow_tail = allow_underfilled
        candidate_pool: List[str]
        selected_tail: List[str] = []
        if len(eligible_full) >= M or not allow_tail:
            candidate_pool = eligible_full
            if len(candidate_pool) < M:
                raise RuntimeError(
                    f"Not enough classes with embeddings: have {len(candidate_pool)}, need {M} (need={need}, allow_underfilled={allow_underfilled})"
                )
            picked = self._weighted_sample_classes(candidate_pool, M)
        else:
            needed_tail = M - len(eligible_full)
            if len(eligible_full) + len(eligible_support_only) < M:
                raise RuntimeError(
                    f"Not enough classes with embeddings: have {len(eligible_full) + len(eligible_support_only)}, need {M} (need={need}, allow_underfilled={allow_underfilled})"
                )
            picked = self._weighted_sample_classes(eligible_full, len(eligible_full))
            tail_choices = self._weighted_sample_classes(eligible_support_only, needed_tail)
            selected_tail = tail_choices[:]
            picked.extend(tail_choices)
            self.rng.shuffle(picked)

        self._current_underfilled = set(selected_tail)
        self._last_pick_stats = {
            "total_classes": total_classes,
            "eligible_full": len(eligible_full),
            "eligible_support_only": len(eligible_support_only),
            "selected_total": len(picked),
            "selected_underfilled": len(selected_tail),
        }
        return picked

    def _clusters_for(self, pool_idx: List[int]) -> Dict[str, List[int]]:
        clusters: DefaultDict[str, List[int]] = defaultdict(list)
        for i_row in pool_idx:
            acc = self._idx2acc(i_row)
            cid = self.acc2cluster.get(acc, f"_na_{i_row}")
            clusters[cid].append(i_row)
        return clusters

    def _sample_standard(self, ec: str, pool_idx: List[int], K: int, Q: int) -> Tuple[List[int], List[int]]:
        need = K + Q
        if len(pool_idx) < need:
            raise RuntimeError(
                f"Class {ec} has only {len(pool_idx)} samples, requires {need} (no fallback allowed)."
            )
        if self.acc2cluster and self.disjoint_support_query:
            clusters = self._clusters_for(pool_idx)
            cids = list(clusters.keys())
            self.rng.shuffle(cids)
            if len(cids) < need:
                raise ClusterShortageError(ec, len(cids), need)
            support_cids = cids[:K]
            query_cids = cids[K : K + Q]
            support_idx = [self.rng.choice(clusters[cid]) for cid in support_cids]
            query_idx = [self.rng.choice(clusters[cid]) for cid in query_cids]
        else:
            chosen = self.rng.sample(pool_idx, need)
            support_idx = chosen[:K]
            query_idx = chosen[K:]
        return support_idx, query_idx

    def _sample_with_replacement(self, ec: str, pool_idx: List[int], K: int, Q: int) -> Tuple[List[int], List[int]]:
        if not pool_idx:
            raise RuntimeError(f"Class {ec} has no embeddings available for fallback sampling")

        if self.acc2cluster and self.disjoint_support_query:
            clusters = self._clusters_for(pool_idx)
            cids = list(clusters.keys())
            if not cids:
                # fall back to plain behaviour
                return self._sample_with_replacement_plain(pool_idx, K, Q)
            self.rng.shuffle(cids)
            need = K + Q
            if len(cids) < need:
                # Maintain support/query cluster disjointness even under shortage
                # by partitioning available clusters between sides and sampling
                # with replacement within each side's cluster set. If fewer than
                # 2 clusters exist, escalate to caller to resample classes.
                if len(cids) < 2:
                    raise ClusterShortageError(ec, len(cids), need)
                self._record_cluster_shortage(ec, len(cids), need)
                # Proportional split of clusters into support/query bins
                s_clusters = max(1, min(len(cids) - 1, int(round(len(cids) * (K / float(max(K + Q, 1)))))))
                q_clusters = max(1, len(cids) - s_clusters)
                s_cids = cids[:s_clusters]
                q_cids = cids[s_clusters:]
                # Round-robin draw within each bin with replacement
                support_idx = [self.rng.choice(clusters[s_cids[i % len(s_cids)]]) for i in range(K)]
                query_idx = [self.rng.choice(clusters[q_cids[i % len(q_cids)]]) for i in range(Q)]
                return support_idx, query_idx
            support_cids = cids[:K]
            query_cids = cids[K : K + Q]
            support_idx = [self.rng.choice(clusters[cid]) for cid in support_cids]
            query_idx = [self.rng.choice(clusters[cid]) for cid in query_cids]
            return support_idx, query_idx

        return self._sample_with_replacement_plain(pool_idx, K, Q)

    def _sample_with_replacement_plain(self, pool_idx: List[int], K: int, Q: int) -> Tuple[List[int], List[int]]:
        pool = pool_idx[:]
        self.rng.shuffle(pool)
        support_idx: List[int] = []
        query_idx: List[int] = []

        available = pool[:]
        while len(support_idx) < K:
            if available:
                support_idx.append(available.pop())
            else:
                support_idx.append(self.rng.choice(pool_idx))
        available_for_query = [idx for idx in pool_idx if idx not in support_idx]
        self.rng.shuffle(available_for_query)
        while len(query_idx) < Q:
            if available_for_query:
                query_idx.append(available_for_query.pop())
            else:
                query_idx.append(self.rng.choice(pool_idx))
        return support_idx, query_idx

    def _rng_for_variant(self, acc: str, occurrence: int, seq_variant: Optional[str]) -> np.random.Generator:
        base = f"{self._base_seed}|{acc}|{occurrence}|{seq_variant or ''}"
        digest = hashlib.sha1(base.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        return np.random.default_rng(seed)

    def _augment_embedding(self, base: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        view = np.array(base, dtype=np.float32, copy=True)
        if self._view_dropout > 0:
            drop_mask = rng_local.random(view.shape[0]) < self._view_dropout
            if drop_mask.all():
                drop_mask[rng_local.integers(0, view.shape[0])] = False
            view = np.where(drop_mask, 0.0, view)
        if self._view_noise_sigma > 0:
            noise = rng_local.normal(loc=0.0, scale=self._view_noise_sigma, size=view.shape[0])
            view = view + noise.astype(np.float32)
        base_norm = float(np.linalg.norm(base))
        new_norm = float(np.linalg.norm(view))
        if base_norm > 0 and new_norm > 0:
            view = view * (base_norm / new_norm)
        return view.astype(np.float32, copy=False)

    def _views_for_index(self, idx: int, count: int) -> List[np.ndarray]:
        base = np.array(self.X[idx], dtype=np.float32, copy=True)  # type: ignore[index]
        if count <= 1:
            return [base]
        acc = self._idx2acc(idx)
        seq = self.seq_lookup.get(acc)
        variants: List[np.ndarray] = [base]
        variant_occurrence = 1
        while len(variants) < count:
            if seq:
                view_a, view_b = augment.make_two_views(seq, rng=self.rng)
                for seq_variant in (view_a, view_b):
                    rng_local = self._rng_for_variant(acc, variant_occurrence, seq_variant)
                    variants.append(self._augment_embedding(base, rng_local))
                    variant_occurrence += 1
                    if len(variants) >= count:
                        break
                seq = view_a  # feed augmented sequence forward for diversity
            else:
                rng_local = self._rng_for_variant(acc, variant_occurrence, None)
                variants.append(self._augment_embedding(base, rng_local))
                variant_occurrence += 1
        return variants[:count]

    def _take_view(
        self,
        cache: Dict[int, List[np.ndarray]],
        occurrences: DefaultDict[int, int],
        idx: int,
    ) -> np.ndarray:
        views = cache.get(idx)
        if views is None:
            views = self._views_for_index(idx, 1)
            cache[idx] = views
        pos = occurrences[idx]
        occurrences[idx] += 1
        if pos >= len(views):
            views = self._views_for_index(idx, pos + 1)
            cache[idx] = views
        return views[pos]

    def _record_cluster_shortage(self, ec: str, have: int, need: int) -> None:
        self.cluster_shortage_counts[ec] += 1
        self.cluster_shortage_events += 1
        if self.cluster_shortage_events <= 5 or self.cluster_shortage_events % 50 == 0:
            _progress_write(
                f"[sampler][{self.phase}] cluster shortage for EC {ec}: "
                f"clusters={have}, need={need}"
            )

    def _note_single_cluster_skip(self, ec: str, clusters: int) -> None:
        if ec in self._skipped_single_cluster:
            return
        self._skipped_single_cluster.add(ec)
        _progress_write(
            f"[sampler][{self.phase}] skipping EC {ec}: clusters={clusters}, need≥2 for disjoint support/query"
        )

    def _log_fallback(self, ec: str, have: int, need: int) -> None:
        self.fallback_events += 1
        if self.fallback_events <= 5 or self.fallback_events % 50 == 0:
            _progress_write(
                f"[sampler][{self.phase}] fallback triggered for EC {ec}: "
                f"available={have}, need={need}"
            )

    def _build_episode(
        self,
        classes: List[str],
        K: int,
        Q: int,
        need: int,
        allow_underfilled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        support_x: List[np.ndarray] = []
        support_y: List[int] = []
        query_x: List[np.ndarray] = []
        query_y: List[int] = []
        query_multi: List[List[int]] = []
        view_cache: Dict[int, List[np.ndarray]] = {}
        occurrences: DefaultDict[int, int] = defaultdict(int)

        for label, ec in enumerate(classes):
            pool_idx = self.class2idx.get(ec, [])
            if not pool_idx:
                raise RuntimeError(f"No embeddings for class {ec}")
            count = len(pool_idx)
            use_fallback = count < need and allow_underfilled
            if use_fallback:
                s_idx, q_idx = self._sample_with_replacement(ec, pool_idx, K, Q)
                self._log_fallback(ec, count, need)
                self.fallback_per_ec[ec] += 1
            else:
                try:
                    s_idx, q_idx = self._sample_standard(ec, pool_idx, K, Q)
                except ClusterShortageError as err:
                    if not self.allow_fallback:
                        raise
                    s_idx, q_idx = self._sample_with_replacement(ec, pool_idx, K, Q)
                    self._log_fallback(ec, err.have, err.need)
                    self.fallback_per_ec[ec] += 1
                else:
                    if self.disjoint_support_query:
                        overlap = set(s_idx).intersection(q_idx)
                        if overlap:
                            raise AssertionError(
                                f"support/query overlap without fallback for EC {ec}: {overlap}"
                            )
            combined = s_idx + q_idx
            if use_fallback:
                counts = Counter(combined)
                for idx, cnt in counts.items():
                    if cnt > 1:
                        view_cache[idx] = self._views_for_index(idx, cnt)
            self.usage_support_counts[ec] += len(s_idx)
            self.usage_query_counts[ec] += len(q_idx)
            self.total_support += len(s_idx)
            self.total_query += len(q_idx)

            for i_row in s_idx:
                vec = self._take_view(view_cache, occurrences, i_row)
                support_x.append(vec)
                support_y.append(label)
            for i_row in q_idx:
                vec = self._take_view(view_cache, occurrences, i_row)
                query_x.append(vec)
                if not self.multi_label:
                    query_y.append(label)
                else:
                    acc = self._idx2acc(i_row)
                    row = [1 if ec2 in self.acc2ecs.get(acc, set()) else 0 for ec2 in classes]
                    query_multi.append(row)

        self._stats_dirty = True

        sx = torch.from_numpy(np.stack(support_x).astype(np.float32)).to(self.device)
        qx = torch.from_numpy(np.stack(query_x).astype(np.float32)).to(self.device)
        sy = torch.tensor(support_y, dtype=torch.long, device=self.device)
        if self.multi_label:
            qy_np = np.array(query_multi, dtype=np.float32)
            qy = torch.tensor(qy_np, dtype=torch.float32, device=self.device)
        else:
            qy = torch.tensor(query_y, dtype=torch.long, device=self.device)
        return sx, sy, qx, qy, classes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_episode(
        self,
        M: int,
        K: int,
        Q: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        if K <= 0 or Q <= 0:
            raise ValueError("K and Q must be positive integers")
        need = K + Q
        # Allow selecting underfilled tail classes during eval as well
        # (see: Handle tail classes during evaluation #22)
        allow_underfilled = self.allow_fallback or self._eval_tail_fallback
        attempts = 0
        shortage_attempts: DefaultDict[str, int] = defaultdict(int)

        while attempts < self._max_cluster_resample_attempts:
            attempts += 1
            classes = self._pick_classes(M, need, K, allow_underfilled)
            try:
                return self._build_episode(classes, K, Q, need, allow_underfilled)
            except ClusterShortageError as err:
                shortage_attempts[err.ec] += 1
                self._record_cluster_shortage(err.ec, err.have, err.need)
                continue

        self.cluster_shortage_dropped_episodes += 1
        self.cluster_shortage_last_drop = dict(shortage_attempts)
        shortage_list = ", ".join(
            f"{ec} (attempts={cnt})" for ec, cnt in sorted(shortage_attempts.items())
        )
        _progress_write(
            f"[sampler][{self.phase}] dropping episode after {attempts} attempts due to "
            f"cluster shortages: {shortage_list or 'unknown'}"
        )
        raise RuntimeError(
            "Unable to sample episode with disjoint support/query clusters. "
            "Adjust K/Q or exclude problematic ECs."
        )

    def write_usage_csv(self, target_dir: Optional[Path] = None) -> Optional[Path]:
        if not self._stats_dirty and target_dir is None and self.usage_log_path is None:
            return None
        out_path = None
        if target_dir is not None:
            out_path = Path(target_dir) / "sampler_stats.csv"
        elif self.usage_log_path is not None:
            out_path = self.usage_log_path
        if out_path is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ec", "support_count", "query_count", "fallback_episodes"])
            for ec in sorted(self.split.classes):
                writer.writerow(
                    [
                        ec,
                        int(self.usage_support_counts.get(ec, 0)),
                        int(self.usage_query_counts.get(ec, 0)),
                        int(self.fallback_per_ec.get(ec, 0)),
                    ]
                )
        self._stats_dirty = False
        print(f"[sampler] wrote usage stats → {out_path}")
        return out_path

    def class_coverage(self, K: int, Q: int) -> Dict[str, int]:
        """Return how many classes have sufficient samples for (K, Q)."""

        need = K + Q
        total = 0
        eligible_full = 0
        eligible_support_only = 0
        require_clusters = self.disjoint_support_query and bool(self.acc2cluster)
        for ec in self.split.classes:
            count = self.class_counts.get(ec, 0)
            if count <= 0:
                continue
            total += 1
            if require_clusters:
                cluster_count = self.class_cluster_counts.get(ec, 0)
                if cluster_count < 2:
                    continue
            if count >= need:
                eligible_full += 1
            elif count >= K:
                eligible_support_only += 1
        return {
            "total_classes": total,
            "eligible_full": eligible_full,
            "eligible_support_only": eligible_support_only,
            "excluded": max(total - eligible_full - eligible_support_only, 0),
        }

    @property
    def last_pick_stats(self) -> Dict[str, int]:
        """Return stats from the most recent :meth:`sample_episode` call."""

        return dict(self._last_pick_stats)
