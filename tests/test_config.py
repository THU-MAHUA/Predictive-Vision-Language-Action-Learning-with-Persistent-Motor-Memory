from pathlib import Path

import pytest

from ipme_vla.config import IPMEVLAConfig


def test_base_configuration_loads() -> None:
    config = IPMEVLAConfig.from_yaml(
        Path(__file__).parents[1] / "configs" / "ipme_vla_base.yaml"
    )
    assert config.chunk_size == 50
    assert config.predictive_horizon == 10
    assert config.observation_delta_indices == [0, 10, 20, 30, 40]
    assert config.action_delta_indices == list(range(80))


def test_configuration_rejects_incompatible_memory_tokens() -> None:
    with pytest.raises(ValueError, match="must equal"):
        IPMEVLAConfig(num_memory_tokens=3, num_consequence_tokens=4)


def test_configuration_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        IPMEVLAConfig.from_dict({"private_path": "/tmp"})
