"""Core IPME-VLA model composition."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.models.action_expert import FlowMatchingActionExpert
from ipme_vla.models.consequence_model import PredictiveConsequenceModel
from ipme_vla.models.motor_memory import PersistentMotorMemory


class IPMEVLAModel(nn.Module):
    """Flow matching augmented by consequence prediction and persistent memory."""

    def __init__(
        self,
        config: IPMEVLAConfig,
        action_expert: FlowMatchingActionExpert | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_expert = action_expert or FlowMatchingActionExpert(config)
        hidden_size = self.action_expert.expert_hidden_size
        self.consequence_model = PredictiveConsequenceModel(
            config.num_consequence_tokens,
            hidden_size,
            self.action_expert.visual_dimension,
            config.max_state_dim,
            config.motor_memory_init_std,
        )
        self.motor_memory = PersistentMotorMemory(
            config.num_memory_tokens,
            hidden_size,
            config.motor_memory_init_std,
        )

    def initial_memory(self, batch_size: int, device=None, dtype=None) -> Tensor:
        return self.motor_memory.initial(batch_size, device, dtype)

    def encode_prefix(self, images, image_masks, language_tokens, language_masks, state):
        return self.action_expert.embed_prefix(
            images, image_masks, language_tokens, language_masks, state
        )

    def flow_loss(self, prefix, state, actions, memory, noise=None, time=None):
        return self.action_expert.flow_loss(
            prefix, state, actions, memory, noise=noise, time=time
        )

    def predict_consequence(
        self,
        prefix,
        clean_initial_actions: Tensor,
        memory: Tensor,
        state: Tensor,
    ) -> dict[str, Tensor]:
        queries = self.consequence_model.expanded_queries(
            clean_initial_actions.shape[0],
            clean_initial_actions.device,
            memory.dtype,
        )
        hidden = self.action_expert.consequence_hidden(
            prefix, clean_initial_actions, memory, queries, state
        )
        prediction = self.consequence_model(hidden)
        prediction["next_memory"] = self.motor_memory.update(hidden)
        return prediction

    @torch.no_grad()
    def generate(
        self,
        images,
        image_masks,
        language_tokens,
        language_masks,
        state,
        memory=None,
        noise=None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        memory = (
            memory
            if memory is not None
            else self.initial_memory(state.shape[0], state.device, torch.float32)
        )
        prefix = self.encode_prefix(images, image_masks, language_tokens, language_masks, state)
        actions = self.action_expert.denoise(prefix, state, memory, noise=noise)
        prediction = self.predict_consequence(
            prefix,
            actions[:, : self.config.predictive_horizon],
            memory,
            state,
        )
        return actions, prediction["next_memory"], prediction
