"""Dataset-statistics normalization modules."""

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Modified for the standalone IPME-VLA project.

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _as_tensor(value: Any, fallback: Tensor) -> Tensor:
    if value is None:
        return fallback
    if isinstance(value, Tensor):
        return value.detach().clone().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


class TensorNormalizer(nn.Module):
    """Mean/std normalization with checkpointed statistics."""

    def __init__(self, stats: dict[str, Any] | None, dimension: int):
        super().__init__()
        stats = stats or {}
        self.register_buffer("mean", _as_tensor(stats.get("mean"), torch.zeros(dimension)))
        self.register_buffer("std", _as_tensor(stats.get("std"), torch.ones(dimension)))

    def forward(self, tensor: Tensor) -> Tensor:
        return (tensor - self.mean) / self.std.clamp_min(1e-8)


class TensorUnnormalizer(nn.Module):
    """Inverse of :class:`TensorNormalizer`."""

    def __init__(self, stats: dict[str, Any] | None, dimension: int):
        super().__init__()
        stats = stats or {}
        self.register_buffer("mean", _as_tensor(stats.get("mean"), torch.zeros(dimension)))
        self.register_buffer("std", _as_tensor(stats.get("std"), torch.ones(dimension)))

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor * self.std + self.mean

