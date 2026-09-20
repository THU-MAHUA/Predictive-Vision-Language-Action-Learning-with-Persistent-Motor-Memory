from conftest import require_torch

require_torch()

import torch
from torch import nn

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.constants import ACTION, OBS_STATE, TASK
from ipme_vla.models.vla.modeling import IPMEVLAModel
from ipme_vla.models.vla.policy import IPMEVLAPolicy


class TinyTokenizer:
    def __call__(self, prompts, **kwargs):
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }


class TinyActionExpert(nn.Module):
    expert_hidden_size = 8
    visual_dimension = 6

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.tensor(0.1))

    def embed_prefix(self, images, masks, language_tokens, language_masks, state):
        batch = state.shape[0]
        visual = state[:, :1].expand(batch, self.visual_dimension) * self.weight
        placeholder = state.new_zeros(batch, 1, 1)
        boolean = torch.ones(batch, 1, dtype=torch.bool, device=state.device)
        return placeholder, boolean, boolean, visual

    def flow_loss(self, prefix, state, actions, memory, noise=None, time=None):
        prediction = actions * self.weight + memory.mean() * self.weight
        return prediction.square(), prediction

    def consequence_hidden(self, prefix, clean_actions, memory, queries, state):
        action_value = clean_actions.mean(dim=(1, 2), keepdim=True)
        return memory + queries * self.weight + action_value * self.weight

    def denoise(self, prefix, state, memory, noise=None):
        shape = (state.shape[0], self.config.chunk_size, self.config.max_action_dim)
        return torch.ones(shape, device=state.device) * self.weight


def make_policy():
    config = IPMEVLAConfig(
        chunk_size=6,
        n_action_steps=2,
        predictive_horizon=2,
        recurrent_unroll_steps=2,
        num_memory_tokens=2,
        num_consequence_tokens=2,
        max_state_dim=4,
        max_action_dim=3,
        state_dim=4,
        action_dim=3,
        image_features=["observation.images.main"],
        resize_imgs_with_padding=(8, 8),
        num_steps=2,
    )
    model = IPMEVLAModel(config, TinyActionExpert(config))
    return IPMEVLAPolicy(
        config,
        {"state": {"mean": [0] * 4, "std": [1] * 4}, "action": {"mean": [0] * 3, "std": [1] * 3}},
        model=model,
        language_tokenizer=TinyTokenizer(),
    )


def make_batch():
    return {
        OBS_STATE: torch.randn(2, 3, 4),
        ACTION: torch.randn(2, 8, 3),
        "observation.images.main": torch.rand(2, 3, 3, 8, 8),
        "observation.images.main_padding_mask": torch.ones(2, 3, dtype=torch.bool),
        f"{OBS_STATE}_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        TASK: ["move object", "move object"],
    }


def test_all_losses_backward_optimizer_and_generation() -> None:
    policy = make_policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = policy.model.action_expert.weight.detach().clone()
    loss, metrics = policy(make_batch())
    assert loss.requires_grad
    assert set(metrics) >= {
        "loss_fm",
        "loss_visual_consequence",
        "loss_state_consequence",
    }
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, policy.model.action_expert.weight.detach())

    observation = make_batch()
    observation = {
        key: value[:, :1] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value
        for key, value in observation.items()
        if key not in {ACTION, "action_is_pad"}
    }
    actions = policy.predict_action_chunk(observation)
    assert actions.shape == (2, 6, 3)


def test_memory_persists_until_reset() -> None:
    policy = make_policy()
    observation = make_batch()
    observation = {
        key: value[:, :1] if isinstance(value, torch.Tensor) and value.ndim >= 2 else value
        for key, value in observation.items()
        if key not in {ACTION, "action_is_pad"}
    }
    policy.select_action(observation)
    first_memory = policy._motor_memory
    assert first_memory is not None
    policy.select_action(observation)
    assert policy._motor_memory is first_memory
    policy.reset()
    assert policy._motor_memory is None
    assert not policy._action_queue
