"""On-the-fly M-way K-shot episodic sampler from cached embeddings.

Loads:
- embeddings: NPZ mapping accession -> vector
- split JSONL files (train/val/test) mapping ec -> accessions

Provides EpisodeSampler that returns support/query tensors and integer labels per episode.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Samples episodes (M-way, K-shot, Q-queries per class)."""

    def __init__(
        self,
        embeddings_npz: Path,
        split_jsonl: Path,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.device = device
        self.rng = random.Random(seed)

        # Normalize paths and detect contiguous format alongside the configured NPZ path
        emb_path = Path(embeddings_npz)
        base_str = str(emb_path)
        if base_str.endswith(".npz"):
            base_str = base_str[:-4]
        X_path = Path(base_str + ".X.npy")
        keys_path = Path(base_str + ".keys.npy")

        self._use_mmap = False
        self.emb_map: Dict[str, np.ndarray] = {}

        if X_path.exists() and keys_path.exists():
            # Fast path: memory-map contiguous array and build index
            self._use_mmap = True
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
        else:
            # Legacy NPZ path: load arrays into a dict with a visible progress bar
            print(f"[load-emb] using legacy NPZ: {emb_path}")
            npz = np.load(emb_path, allow_pickle=False)
            self.emb_map = {}
            for k in tqdm(npz.files, desc="[load-emb] npz", dynamic_ncols=True):
                self.emb_map[k] = npz[k]
            any_vec = next(iter(self.emb_map.values())) if self.emb_map else np.zeros(1, dtype=np.float32)
            self.dim = int(any_vec.shape[0])

        # Build split index and per-class pools using either indices or accessions
        self.split = SplitIndex.from_jsonl(split_jsonl)
        if self._use_mmap:
            self.class2idx: Dict[str, List[int]] = {}
            for ec, accs in self.split.by_class.items():
                self.class2idx[ec] = [self.key2row[a] for a in accs if a in self.key2row]
        else:
            self.class2acc: Dict[str, List[str]] = {}
            for ec, accs in self.split.by_class.items():
                self.class2acc[ec] = [a for a in accs if a in self.emb_map]

    def _pick_classes(self, M: int) -> List[str]:
        if self._use_mmap:
            classes = [c for c in self.split.classes if len(getattr(self, 'class2idx', {}).get(c, [])) > 0]
        else:
            classes = [c for c in self.split.classes if len(getattr(self, 'class2acc', {}).get(c, [])) > 0]
        if len(classes) < M:
            raise RuntimeError(f"Not enough classes with embeddings: have {len(classes)}, need {M}")
        self.rng.shuffle(classes)
        return classes[:M]

    def sample_episode(self, M: int, K: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a single episode.

        Args:
            M: Number of classes.
            K: Number of support samples per class. Must be positive.
            Q: Number of query samples per class. Must be positive.

        Raises:
            ValueError: If ``K`` or ``Q`` is not a positive integer.
        """
        if K <= 0 or Q <= 0:
            raise ValueError("K and Q must be positive integers")
        classes = self._pick_classes(M)
        support_x, support_y, query_x, query_y = [], [], [], []
        for label, ec in enumerate(classes):
            if self._use_mmap:
                pool_idx = self.class2idx.get(ec, [])
                if len(pool_idx) == 0:
                    raise RuntimeError(f"No embeddings for class {ec}")
                need = K + Q
                if len(pool_idx) >= need:
                    chosen = self.rng.sample(pool_idx, need)
                else:
                    chosen = [self.rng.choice(pool_idx) for _ in range(need)]
                s_idx = chosen[:K]
                q_idx = chosen[K:]
                for i_row in s_idx:
                    support_x.append(self.X[i_row])  # type: ignore[attr-defined]
                    support_y.append(label)
                for i_row in q_idx:
                    query_x.append(self.X[i_row])  # type: ignore[attr-defined]
                    query_y.append(label)
            else:
                pool = self.class2acc.get(ec, [])
                if len(pool) == 0:
                    raise RuntimeError(f"No embeddings for class {ec}")
                need = K + Q
                # sample with replacement if not enough items
                if len(pool) >= need:
                    chosen = self.rng.sample(pool, need)
                else:
                    chosen = [self.rng.choice(pool) for _ in range(need)]
                s_acc = chosen[:K]
                q_acc = chosen[K:]
                for a in s_acc:
                    support_x.append(self.emb_map[a])
                    support_y.append(label)
                for a in q_acc:
                    query_x.append(self.emb_map[a])
                    query_y.append(label)
        sx = torch.from_numpy(np.stack(support_x).astype(np.float32)).to(self.device)
        qx = torch.from_numpy(np.stack(query_x).astype(np.float32)).to(self.device)
        sy = torch.tensor(support_y, dtype=torch.long, device=self.device)
        qy = torch.tensor(query_y, dtype=torch.long, device=self.device)
        return sx, sy, qx, qy
