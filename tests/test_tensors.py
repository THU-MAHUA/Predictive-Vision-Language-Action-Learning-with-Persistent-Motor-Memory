import pytest

from conftest import require_torch

require_torch()

import torch

from ipme_vla.utils.tensors import make_att_2d_masks


def test_block_attention_mask() -> None:
    padding = torch.ones(1, 5, dtype=torch.bool)
    blocks = torch.tensor([[0, 0, 1, 0, 1]], dtype=torch.bool)
    mask = make_att_2d_masks(padding, blocks)
    assert mask.shape == (1, 5, 5)
    assert mask[0, 1, 0]
    assert not mask[0, 0, 2]
    assert mask[0, 3, 2]


def test_attention_requires_rank_two() -> None:
    with pytest.raises(ValueError):
        make_att_2d_masks(torch.ones(2), torch.ones(1, 2))
