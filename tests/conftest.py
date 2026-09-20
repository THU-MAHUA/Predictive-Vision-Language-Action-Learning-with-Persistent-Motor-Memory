"""Test environment checks."""

import pytest


def require_torch() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed", allow_module_level=True)
    if not hasattr(torch, "Tensor"):
        pytest.skip("The available torch namespace is incomplete", allow_module_level=True)
