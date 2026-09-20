"""Versioned IPME-VLA checkpoint IO and legacy key compatibility."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ipme_vla.config import IPMEVLAConfig

CHECKPOINT_FORMAT = "ipme-vla"
CHECKPOINT_VERSION = 1
_VARIANT_BUFFER = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def remap_legacy_key(key: str) -> str:
    """Map checkpoints trained under the original VLAb package to this package."""
    key = key.replace("model._orig_mod.", "model.")
    key = _VARIANT_BUFFER.sub(".buffer_", key)
    normalization_keys = (
        ("normalize_inputs.buffer_observation_state.", "normalize_state."),
        ("normalize_targets.buffer_action.", "normalize_action."),
        ("unnormalize_outputs.buffer_action.", "unnormalize_action."),
    )
    for old, new in normalization_keys:
        if key.startswith(old):
            return new + key[len(old) :]
    mappings = (
        ("model.vlm_with_expert.", "model.action_expert.backbone."),
        ("model.state_proj.", "model.action_expert.state_proj."),
        ("model.action_in_proj.", "model.action_expert.action_in_proj."),
        ("model.action_out_proj.", "model.action_expert.action_out_proj."),
        ("model.action_time_mlp_in.", "model.action_expert.action_time_mlp_in."),
        ("model.action_time_mlp_out.", "model.action_expert.action_time_mlp_out."),
        ("model.initial_motor_memory", "model.motor_memory.initial_memory"),
        ("model.consequence_queries", "model.consequence_model.queries"),
        ("model.motor_memory_norm.", "model.motor_memory.norm."),
        ("model.visual_consequence_head.", "model.consequence_model.visual_head."),
        ("model.state_consequence_head.", "model.consequence_model.state_head."),
    )
    for old, new in mappings:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def remap_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {remap_legacy_key(key): value for key, value in state_dict.items()}


def _load_raw(path: Path, map_location: str | torch.device) -> Any:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=False)


def _adapt_normalization_shapes(
    state_dict: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = dict(state_dict)
    for key in (
        "normalize_state.mean",
        "normalize_state.std",
        "normalize_action.mean",
        "normalize_action.std",
        "unnormalize_action.mean",
        "unnormalize_action.std",
    ):
        if key not in output or key not in reference:
            continue
        value, target = output[key], reference[key]
        if value.shape == target.shape:
            continue
        if value.ndim != 1 or target.ndim != 1 or value.numel() > target.numel():
            continue
        padded = target.detach().clone()
        padded[: value.numel()] = value.to(dtype=padded.dtype)
        output[key] = padded
    return output


def save_checkpoint(
    path: str | Path,
    policy: nn.Module,
    config: IPMEVLAConfig,
    *,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int | None = None,
    normalization_stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "epoch": epoch,
        "config": config.to_dict(),
        "normalization_stats": normalization_stats or {},
        "motor_memory": {
            "num_tokens": config.num_memory_tokens,
            "hidden_state_is_episode_persistent": True,
            "reset_at_episode_boundary": True,
        },
        "extra": extra or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    policy: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = _load_raw(Path(path), map_location)
    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
        if int(payload.get("version", -1)) > CHECKPOINT_VERSION:
            raise RuntimeError(
                f"checkpoint version {payload['version']} is newer than supported "
                f"version {CHECKPOINT_VERSION}"
            )
        state_dict = payload["model_state"]
    else:
        state_dict = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state_dict, dict):
            raise RuntimeError("legacy checkpoint does not contain a model state dictionary")
        state_dict = remap_legacy_state_dict(state_dict)
        state_dict = _adapt_normalization_shapes(state_dict, policy.state_dict())
        payload = {
            "format": "legacy-ipme-vla",
            "version": 0,
            "model_state": state_dict,
            "step": int(payload.get("step", 0)) if isinstance(payload, dict) else 0,
            "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
            "normalization_stats": {},
        }
    incompatible = policy.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    payload["missing_keys"] = list(incompatible.missing_keys)
    payload["unexpected_keys"] = list(incompatible.unexpected_keys)
    return payload


def inspect_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = _load_raw(Path(path), map_location)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("checkpoint is not in the versioned IPME-VLA format")
    return payload
