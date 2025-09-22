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
    detector_enabled: bool = False
    detector_hidden: int = 32


class MultiECDetector(nn.Module):
    """Tiny MLP that predicts whether a query should emit multiple ECs."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        self.detector_enabled = bool(cfg.detector_enabled)
        feature_dim = 7
        if self.detector_enabled:
            self.detector = MultiECDetector(feature_dim, int(cfg.detector_hidden))
        else:
            self.detector = None

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
        # Sum embeddings for each class index
        P = torch.zeros((M, D), device=x.device, dtype=x.dtype)
        P.index_add_(0, y, x)
        # Count support examples per class
        counts = torch.zeros(M, device=x.device, dtype=x.dtype)
        counts.index_add_(0, y, torch.ones_like(y, dtype=x.dtype))
        P = P / counts.unsqueeze(1)
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

    def features_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Construct detector features from logits without altering grad flow."""
        probs = torch.softmax(logits, dim=-1)
        topk = probs.topk(k=min(3, probs.shape[-1]), dim=-1)
        vals = topk.values
        pad = torch.zeros((logits.shape[0], max(0, 3 - vals.shape[-1])), device=logits.device, dtype=logits.dtype)
        padded = torch.cat([vals, pad], dim=-1)
        p1, p2, p3 = padded.unbind(dim=-1)
        margin12 = p1 - p2
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        lmax = logits.max(dim=-1).values
        lstd = logits.std(dim=-1, unbiased=False)
        return torch.stack((p1, p2, p3, margin12, entropy, lmax, lstd), dim=-1)

    def detect_multi(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.detector_enabled or self.detector is None:
            return None
        feats = self.features_from_logits(logits)
        return self.detector(feats)

    @torch.inference_mode()
    def predict(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(support_x, support_y, query_x, None)
        return logits.argmax(dim=-1)
