from conftest import require_torch

require_torch()

import torch
from torch import nn

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.utils.checkpoint import (
    load_checkpoint,
    remap_legacy_state_dict,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    config = IPMEVLAConfig()
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        config,
        optimizer=optimizer,
        step=12,
        normalization_stats={"action": {"mean": [0.0], "std": [1.0]}},
    )
    restored = nn.Linear(3, 2)
    payload = load_checkpoint(path, restored)
    assert payload["step"] == 12
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_legacy_ipme_key_remapping() -> None:
    state = {
        "model._orig_mod.vlm_with_expert.foo": torch.tensor(1),
        "model.initial_motor_memory": torch.tensor(2),
        "model.visual_consequence_head.weight": torch.tensor(3),
    }
    mapped = remap_legacy_state_dict(state)
    assert "model.action_expert.backbone.foo" in mapped
    assert "model.motor_memory.initial_memory" in mapped
    assert "model.consequence_model.visual_head.weight" in mapped
