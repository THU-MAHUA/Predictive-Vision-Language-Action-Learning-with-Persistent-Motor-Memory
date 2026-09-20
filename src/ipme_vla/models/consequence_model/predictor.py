"""Predictive visual and state consequence heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PredictiveConsequenceModel(nn.Module):
    """Decode consequence tokens into visual and proprioceptive changes."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        visual_dimension: int,
        state_dimension: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.queries = nn.Parameter(torch.empty(1, num_tokens, hidden_size))
        nn.init.normal_(self.queries, std=init_std)
        self.visual_head = nn.Linear(hidden_size, visual_dimension)
        self.state_head = nn.Linear(hidden_size, state_dimension)

    def expanded_queries(self, batch_size: int, device, dtype) -> Tensor:
        return self.queries.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(self, consequence_hidden: Tensor) -> dict[str, Tensor]:
        pooled = consequence_hidden.mean(dim=1).float()
        return {
            "consequence_hidden": consequence_hidden,
            "pred_visual_delta": self.visual_head(pooled),
            "pred_state_delta": self.state_head(pooled),
        }

    @staticmethod
    def visual_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)

    @staticmethod
    def state_loss(prediction: Tensor, target: Tensor, dimensions: int) -> Tensor:
        return F.smooth_l1_loss(
            prediction[..., :dimensions],
            target[..., :dimensions],
            reduction="none",
        ).mean(dim=-1)
