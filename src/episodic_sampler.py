"""On-the-fly M-way K-shot episodic sampler from cached embeddings.

Loads:
- embeddings: contiguous memory-mapped arrays `embeddings.X.npy` (+ `embeddings.keys.npy`)
- split JSONL files (train/val/test) mapping ec -> accessions

Provides EpisodeSampler that returns support/query tensors and labels per episode.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm


@dataclass
class SplitIndex:
    # ec class -> list of accessions
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
    """Samples episodes (M-way, K-shot, Q-queries per class).

    Options:
    - multi_label: if True, returns query_y as multi-hot [Nq, M] for multi-EC
      evaluation/training within the episode (support_y remains single-class
      indices for prototype construction per episode class).
    - clusters_tsv: optional path to a two-column TSV "accession\tcluster_id"
      used for identity-aware disjoint sampling of support/query pools.
    - disjoint_support_query: if True and clusters are provided, queries are
      preferentially drawn from clusters not used for supports within each class.
    """

    def __init__(
        self,
        embeddings_npz: Path,
        split_jsonl: Path,
        device: torch.device,
        seed: int = 42,
        *,
        multi_label: bool = False,
        clusters_tsv: Optional[Path] = None,
        disjoint_support_query: bool = False,
    ) -> None:
        self.device = device
        self.rng = random.Random(seed)
        self.multi_label = bool(multi_label)
        self.disjoint_support_query = bool(disjoint_support_query)

        # Normalize paths and derive contiguous array paths from the configured base
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
        # Memory-map contiguous arrays and build index
        self.X = np.load(X_path, mmap_mode="r")  # type: ignore[attr-defined]
        self.keys = np.load(keys_path, allow_pickle=False)  # type: ignore[attr-defined]
        if self.X.shape[0] != self.keys.shape[0]:
            raise ValueError(
                f"[load-emb] X rows ({self.X.shape[0]}) != keys ({self.keys.shape[0]})"
            )
        print(f"[load-emb] using contiguous X.npy (mmap): shape={self.X.shape}")
        # Build key → row index
        self.key2row: Dict[str, int] = {k: i for i, k in enumerate(self.keys.tolist())}  # type: ignore[attr-defined]
        self.dim = int(self.X.shape[1])

        # Build split index and per-class pools using either indices or accessions
        self.split = SplitIndex.from_jsonl(split_jsonl)
        # Build acc -> set(ec) map within this split (for multi-label targets)
        self.acc2ecs: Dict[str, Set[str]] = defaultdict(set)
        for ec, accs in self.split.by_class.items():
            for a in accs:
                self.acc2ecs[a].add(ec)
        self.class2idx: Dict[str, List[int]] = {}
        for ec, accs in self.split.by_class.items():
            self.class2idx[ec] = [self.key2row[a] for a in accs if a in self.key2row]

        # Optional: load clusters for identity-aware sampling
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

    def _pick_classes(self, M: int) -> List[str]:
        classes = [c for c in self.split.classes if len(self.class2idx.get(c, [])) > 0]
        if len(classes) < M:
            raise RuntimeError(f"Not enough classes with embeddings: have {len(classes)}, need {M}")
        self.rng.shuffle(classes)
        return classes[:M]

    def sample_episode(self, M: int, K: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Sample a single episode.

        Each class must provide at least ``K + Q`` distinct examples. If a class
        has fewer available, a ``RuntimeError`` is raised.

        Args:
            M: Number of classes.
            K: Number of support samples per class. Must be positive.
            Q: Number of query samples per class. Must be positive.

        Raises:
            ValueError: If ``K`` or ``Q`` is not a positive integer.
            RuntimeError: If any class has fewer than ``K + Q`` samples.
        """
        if K <= 0 or Q <= 0:
            raise ValueError("K and Q must be positive integers")
        classes = self._pick_classes(M)
        support_x, support_y, query_x, query_y = [], [], [], []
        query_multi: List[List[int]] = []  # used only if multi_label
        for label, ec in enumerate(classes):
            pool_idx = self.class2idx.get(ec, [])
            if len(pool_idx) == 0:
                raise RuntimeError(f"No embeddings for class {ec}")
            need = K + Q
            if len(pool_idx) < need:
                raise RuntimeError(
                    f"Class {ec} has only {len(pool_idx)} samples, but requires at least {need} (K + Q)"
                )
            # Helper to get accession for a row
            def idx2acc(i_row: int) -> str:
                return str(self.keys[i_row])  # type: ignore[attr-defined]
            if self.acc2cluster and self.disjoint_support_query:
                # Group by cluster
                clusters: DefaultDict[str, List[int]] = defaultdict(list)
                for i_row in pool_idx:
                    clusters[self.acc2cluster.get(idx2acc(i_row), f"_na_{i_row}")].append(i_row)
                cids = list(clusters.keys())
                self.rng.shuffle(cids)
                # Pick support from distinct clusters (with replacement if few)
                s_idx: List[int] = []
                if len(cids) > 0:
                    for ii in range(K):
                        cid = cids[ii % len(cids)]
                        s_idx.append(self.rng.choice(clusters[cid]))
                    used = set(cids[: min(K, len(cids))])
                else:
                    s_idx = [self.rng.choice(pool_idx) for _ in range(K)]
                    used = set()
                # Queries prefer clusters not used for support
                rem_cids = [c for c in cids if c not in used]
                q_idx: List[int] = []
                if len(rem_cids) > 0:
                    self.rng.shuffle(rem_cids)
                    for ii in range(Q):
                        cid = rem_cids[ii % len(rem_cids)]
                        q_idx.append(self.rng.choice(clusters[cid]))
                else:
                    # fallback: sample from any cluster
                    q_idx = [self.rng.choice(pool_idx) for _ in range(Q)]
            else:
                chosen = self.rng.sample(pool_idx, need)
                s_idx = chosen[:K]
                q_idx = chosen[K:]
            for i_row in s_idx:
                support_x.append(self.X[i_row])  # type: ignore[attr-defined]
                support_y.append(label)
            for i_row in q_idx:
                query_x.append(self.X[i_row])  # type: ignore[attr-defined]
                if not self.multi_label:
                    query_y.append(label)
                else:
                    acc = idx2acc(i_row)
                    row = [1 if ec2 in self.acc2ecs.get(acc, set()) else 0 for ec2 in classes]
                    query_multi.append(row)
        sx = torch.from_numpy(np.stack(support_x).astype(np.float32)).to(self.device)
        qx = torch.from_numpy(np.stack(query_x).astype(np.float32)).to(self.device)
        sy = torch.tensor(support_y, dtype=torch.long, device=self.device)
        if self.multi_label:
            qy = torch.tensor(np.array(query_multi, dtype=np.float32), dtype=torch.float32, device=self.device)
        else:
            qy = torch.tensor(query_y, dtype=torch.long, device=self.device)
        return sx, sy, qx, qy, classes
