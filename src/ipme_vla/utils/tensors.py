"""Tensor and attention helpers used by IPME-VLA."""

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Modified for the standalone IPME-VLA project.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape (batch_size,)")
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = (2 * math.pi / period)[None] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    if pad_masks.ndim != 2 or att_masks.ndim != 2:
        raise ValueError("pad_masks and att_masks must both be rank two")
    cumsum = torch.cumsum(att_masks, dim=1)
    causal = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = pad_masks[:, None, :] & pad_masks[:, :, None]
    return causal & valid


def resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = 0.0) -> Tensor:
    if img.ndim != 4:
        raise ValueError(f"expected image tensor (B,C,H,W), received {tuple(img.shape)}")
    current_height, current_width = img.shape[2:]
    ratio = max(current_width / width, current_height / height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    resized = F.interpolate(
        img,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    return F.pad(
        resized,
        (width - resized_width, 0, height - resized_height, 0),
        value=pad_value,
    )


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] > new_dim:
        raise ValueError(f"feature dimension {vector.shape[-1]} exceeds configured maximum {new_dim}")
    if vector.shape[-1] == new_dim:
        return vector
    output = vector.new_zeros(*vector.shape[:-1], new_dim)
    output[..., : vector.shape[-1]] = vector
    return output


def pad_tensor(tensor: Tensor, max_len: int, pad_value: float = 0.0) -> Tensor:
    if tensor.shape[1] > max_len:
        return tensor[:, :max_len]
    if tensor.shape[1] == max_len:
        return tensor
    output = torch.full(
        (tensor.shape[0], max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    output[:, : tensor.shape[1]] = tensor
    return output


def sample_beta(alpha: float, beta: float, batch_size: int, device: torch.device) -> Tensor:
    gamma1 = torch.empty(batch_size, device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty(batch_size, device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

