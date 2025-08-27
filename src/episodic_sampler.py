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
        npz = np.load(embeddings_npz, allow_pickle=False)
        self.emb_map: Dict[str, np.ndarray] = {k: npz[k] for k in npz.files}
        self.split = SplitIndex.from_jsonl(split_jsonl)
        # build class -> available vectors
        self.class2acc: Dict[str, List[str]] = {}
        for ec, accs in self.split.by_class.items():
            self.class2acc[ec] = [a for a in accs if a in self.emb_map]
        # embedding dim
        any_vec = next(iter(self.emb_map.values())) if self.emb_map else np.zeros(1, dtype=np.float32)
        self.dim = int(any_vec.shape[0])

    def _pick_classes(self, M: int) -> List[str]:
        classes = [c for c in self.split.classes if len(self.class2acc.get(c, [])) > 0]
        if len(classes) < M:
            raise RuntimeError(f"Not enough classes with embeddings: have {len(classes)}, need {M}")
        self.rng.shuffle(classes)
        return classes[:M]

    def sample_episode(self, M: int, K: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = self._pick_classes(M)
        support_x, support_y, query_x, query_y = [], [], [], []
        for label, ec in enumerate(classes):
            pool = self.class2acc[ec]
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

