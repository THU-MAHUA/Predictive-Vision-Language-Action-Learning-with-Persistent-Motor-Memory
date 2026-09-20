"""Experiment YAML loading and command-line overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ipme_vla.config import IPMEVLAConfig


def load_experiment(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    return values


def model_config(values: dict[str, Any]) -> IPMEVLAConfig:
    return IPMEVLAConfig.from_dict(deepcopy(values.get("model", {})))


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    return requested
