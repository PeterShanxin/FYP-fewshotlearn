"""Prototypical Network with cosine similarity and optional projection.

- Input embeddings are L2-normalized
- Optional Linear projection to 256 dims (configurable)
- Scores = cosine(query, prototypes) / temperature
- Loss = cross-entropy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProtoConfig:
    input_dim: int
    projection_dim: int = 256
    temperature: float = 10.0


class ProtoNet(nn.Module):
    def __init__(self, cfg: ProtoConfig) -> None:
        super().__init__()
        self.temp = float(cfg.temperature)
        if cfg.projection_dim and cfg.projection_dim > 0:
            self.proj = nn.Linear(cfg.input_dim, cfg.projection_dim)
            out_dim = cfg.projection_dim
        else:
            self.proj = nn.Identity()
            out_dim = cfg.input_dim
        self.out_dim = out_dim

    @staticmethod
    def l2n(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2n(self.proj(x))

    def prototypes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes as mean of support vectors per label.
        Args:
          x: [N, D] support embeddings
          y: [N] integer labels in [0..M-1]
        Returns:
          P: [M, D] prototypes
        """
        M = int(y.max().item()) + 1
        D = x.shape[1]
        P = torch.zeros((M, D), device=x.device, dtype=x.dtype)
        for c in range(M):
            mask = (y == c)
            P[c] = x[mask].mean(dim=0)
        return self.l2n(P)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Embed
        s = self.embed(support_x)
        q = self.embed(query_x)
        # Prototypes and cosine scores
        P = self.prototypes(s, support_y)  # [M, D]
        logits = (q @ P.T) / self.temp     # [|Q|, M]
        loss = None
        if query_y is not None:
            # Multi-task: if query_y is 2D (multi-hot) → BCEWithLogits; else CE
            if query_y.dim() == 2:
                # BCE expects float targets in {0,1}
                target = query_y.float()
                loss = F.binary_cross_entropy_with_logits(logits, target)
            else:
                loss = F.cross_entropy(logits, query_y)
        return logits, loss

    @torch.inference_mode()
    def predict(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(support_x, support_y, query_x, None)
        return logits.argmax(dim=-1)
