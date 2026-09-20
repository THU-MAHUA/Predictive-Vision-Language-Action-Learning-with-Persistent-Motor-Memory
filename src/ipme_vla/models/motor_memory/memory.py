"""Persistent motor-memory state for IPME-VLA."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PersistentMotorMemory(nn.Module):
    """Learned episode initial state and normalized recurrent updates."""

    def __init__(self, num_tokens: int, hidden_size: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.initial_memory = nn.Parameter(torch.empty(1, num_tokens, hidden_size))
        nn.init.normal_(self.initial_memory, std=init_std)
        self.norm = nn.LayerNorm(hidden_size)

    def initial(self, batch_size: int, device=None, dtype=None) -> Tensor:
        memory = self.initial_memory
        if device is not None or dtype is not None:
            memory = memory.to(device=device or memory.device, dtype=dtype or memory.dtype)
        return memory.expand(batch_size, -1, -1)

    def update(self, consequence_hidden: Tensor) -> Tensor:
        if consequence_hidden.shape[-2:] != (self.num_tokens, self.hidden_size):
            raise ValueError(
                "consequence hidden state must have shape "
                f"(batch, {self.num_tokens}, {self.hidden_size})"
            )
        return self.norm(consequence_hidden)

    @staticmethod
    def select(valid: Tensor, next_memory: Tensor, previous_memory: Tensor) -> Tensor:
        return torch.where(valid[:, None, None], next_memory, previous_memory)
