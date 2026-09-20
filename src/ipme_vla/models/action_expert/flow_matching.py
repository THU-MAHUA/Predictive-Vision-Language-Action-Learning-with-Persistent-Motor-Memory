"""Flow-matching action expert used by IPME-VLA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.models.backbone.vlm_expert import VisionLanguageActionBackbone
from ipme_vla.utils.tensors import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
    sample_beta,
)


class FlowMatchingActionExpert(nn.Module):
    """Joint VLM/action-expert network with IPME conditioning tokens."""

    def __init__(
        self,
        config: IPMEVLAConfig,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or VisionLanguageActionBackbone(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
        )
        if hasattr(self.backbone, "configure_peft"):
            self.backbone.configure_peft(config)

        self.expert_hidden_size = int(self.backbone.expert_hidden_size)
        self.visual_dimension = int(self.backbone.config.text_config.hidden_size)
        self.state_to_prefix = config.state_to_prefix
        state_hidden = self.visual_dimension if self.state_to_prefix else self.expert_hidden_size
        self.state_proj = nn.Linear(config.max_state_dim, state_hidden)
        self.action_in_proj = nn.Linear(config.max_action_dim, self.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.expert_hidden_size * 2, self.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.expert_hidden_size, self.expert_hidden_size)
        for parameter in self.state_proj.parameters():
            parameter.requires_grad = config.train_state_proj

    def sample_noise(self, shape, device) -> Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device) -> Tensor:
        return (sample_beta(1.5, 1.0, batch_size, device) * 0.999 + 0.001).float()

    def embed_action_tokens(self, actions: Tensor, timestep: Tensor) -> Tensor:
        action_embedding = self.action_in_proj(actions)
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            actions.device,
        ).to(dtype=action_embedding.dtype)
        time_embedding = time_embedding[:, None].expand_as(action_embedding)
        embedding = self.action_time_mlp_in(torch.cat([action_embedding, time_embedding], dim=-1))
        return self.action_time_mlp_out(F.silu(embedding))

    def embed_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embeddings: list[Tensor] = []
        pad_masks: list[Tensor] = []
        attention_blocks: list[int] = []
        image_summaries: list[Tensor] = []
        summary_masks: list[Tensor] = []

        for image, image_mask in zip(images, image_masks, strict=False):
            image_embedding = self.backbone.embed_image(image)
            embeddings.append(image_embedding * math.sqrt(image_embedding.shape[-1]))
            pad_masks.append(image_mask[:, None].expand(image.shape[0], image_embedding.shape[1]))
            attention_blocks.extend([0] * image_embedding.shape[1])
            image_summaries.append(image_embedding.float().mean(dim=1))
            summary_masks.append(image_mask.bool())

        language_embedding = self.backbone.embed_language_tokens(language_tokens)
        embeddings.append(language_embedding * math.sqrt(language_embedding.shape[-1]))
        pad_masks.append(language_masks)
        attention_blocks.extend([0] * language_embedding.shape[1])

        if state is not None and self.state_to_prefix:
            state_embedding = self.state_proj(state)
            state_embedding = state_embedding[:, None] if state_embedding.ndim == 2 else state_embedding
            embeddings.append(state_embedding)
            pad_masks.append(torch.ones(state_embedding.shape[:2], dtype=torch.bool, device=state.device))
            attention_blocks.extend([1] * state_embedding.shape[1])

        prefix = torch.cat(embeddings, dim=1)
        prefix_pad = torch.cat(pad_masks, dim=1)
        prefix_attention = torch.tensor(
            attention_blocks, dtype=torch.bool, device=prefix.device
        )
        if self.config.prefix_length > 0 and prefix.shape[1] < self.config.prefix_length:
            prefix = pad_tensor(prefix, self.config.prefix_length)
            prefix_pad = pad_tensor(prefix_pad, self.config.prefix_length)
            prefix_attention = pad_tensor(
                prefix_attention[None], self.config.prefix_length
            )[0]
        prefix_attention = prefix_attention[None].expand(prefix.shape[0], -1)

        if image_summaries:
            summaries = torch.stack(image_summaries, dim=1)
            masks = torch.stack(summary_masks, dim=1).to(dtype=summaries.dtype)
            visual_summary = (summaries * masks[..., None]).sum(dim=1)
            visual_summary = visual_summary / masks.sum(dim=1, keepdim=True).clamp_min(1)
            visual_summary = F.layer_norm(visual_summary, (visual_summary.shape[-1],))
        else:
            visual_summary = prefix.new_zeros((prefix.shape[0], self.visual_dimension)).float()
        return prefix, prefix_pad, prefix_attention, visual_summary.detach()

    def _flow_suffix(
        self, noisy_actions: Tensor, timestep: Tensor, memory: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if noisy_actions.shape[1] != self.config.chunk_size:
            raise ValueError("flow suffix must contain exactly chunk_size actions")
        action_tokens = self.embed_action_tokens(noisy_actions, timestep)
        suffix = torch.cat([memory, action_tokens], dim=1)
        padding = torch.ones(suffix.shape[:2], dtype=torch.bool, device=suffix.device)
        blocks = [1] + [0] * (self.config.num_memory_tokens - 1)
        blocks += [1] * self.config.chunk_size
        attention = torch.tensor(blocks, dtype=torch.bool, device=suffix.device)
        return suffix, padding, attention[None].expand(suffix.shape[0], -1)

    def _predictive_suffix(
        self,
        clean_actions: Tensor,
        memory: Tensor,
        consequence_queries: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if clean_actions.shape[1] != self.config.predictive_horizon:
            raise ValueError("predictive suffix must contain predictive_horizon actions")
        timestep = torch.zeros(clean_actions.shape[0], device=clean_actions.device)
        action_tokens = self.embed_action_tokens(clean_actions, timestep)
        suffix = torch.cat([memory, action_tokens, consequence_queries], dim=1)
        padding = torch.ones(suffix.shape[:2], dtype=torch.bool, device=suffix.device)
        blocks = [1] + [0] * (self.config.num_memory_tokens - 1)
        blocks += [1] * self.config.predictive_horizon
        blocks += [1] + [0] * (self.config.num_consequence_tokens - 1)
        attention = torch.tensor(blocks, dtype=torch.bool, device=suffix.device)
        return suffix, padding, attention[None].expand(suffix.shape[0], -1)

    def _run(
        self,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor],
        suffix: Tensor,
        suffix_padding: Tensor,
        suffix_attention: Tensor,
    ) -> Tensor:
        padding = torch.cat([prefix[1], suffix_padding], dim=1)
        attention_blocks = torch.cat([prefix[2], suffix_attention], dim=1)
        attention = make_att_2d_masks(padding, attention_blocks)
        position_ids = torch.cumsum(padding, dim=1) - 1
        (_, suffix_output), _ = self.backbone(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix[0], suffix],
            use_cache=False,
            fill_kv_cache=False,
        )
        return suffix_output

    def _with_optional_state(
        self,
        suffix: Tensor,
        padding: Tensor,
        attention: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        if self.state_to_prefix:
            return suffix, padding, attention, 0
        state_embedding = self.state_proj(state)
        state_embedding = state_embedding[:, None] if state_embedding.ndim == 2 else state_embedding
        state_padding = torch.ones(state_embedding.shape[:2], dtype=torch.bool, device=state.device)
        return (
            torch.cat([state_embedding, suffix], dim=1),
            torch.cat([state_padding, padding], dim=1),
            torch.cat([state_padding, attention], dim=1),
            state_embedding.shape[1],
        )

    def velocity(
        self,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor],
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        memory: Tensor,
    ) -> Tensor:
        suffix, padding, attention = self._flow_suffix(noisy_actions, timestep, memory)
        suffix, padding, attention, offset = self._with_optional_state(
            suffix, padding, attention, state
        )
        output = self._run(prefix, suffix, padding, attention)
        start = offset + self.config.num_memory_tokens
        action_hidden = output[:, start : start + self.config.chunk_size]
        return self.action_out_proj(action_hidden.float())

    def flow_loss(
        self,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor],
        state: Tensor,
        actions: Tensor,
        memory: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        noise = noise if noise is not None else self.sample_noise(actions.shape, actions.device)
        time = time if time is not None else self.sample_time(actions.shape[0], actions.device)
        noisy_actions = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions
        prediction = self.velocity(prefix, state, noisy_actions, time, memory)
        return F.mse_loss(prediction, target, reduction="none"), prediction

    def consequence_hidden(
        self,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor],
        clean_actions: Tensor,
        memory: Tensor,
        consequence_queries: Tensor,
        state: Tensor,
    ) -> Tensor:
        suffix, padding, attention = self._predictive_suffix(
            clean_actions, memory, consequence_queries
        )
        suffix, padding, attention, offset = self._with_optional_state(
            suffix, padding, attention, state
        )
        output = self._run(prefix, suffix, padding, attention)
        start = offset + self.config.num_memory_tokens + self.config.predictive_horizon
        return output[:, start : start + self.config.num_consequence_tokens]

    @torch.no_grad()
    def denoise(
        self,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor],
        state: Tensor,
        memory: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        batch_size = state.shape[0]
        actions = noise
        if actions is None:
            actions = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )
        dt = torch.tensor(-1.0 / self.config.num_steps, device=state.device)
        time = torch.tensor(1.0, device=state.device)
        while time >= -dt / 2:
            velocity = self.velocity(prefix, state, actions, time.expand(batch_size), memory)
            actions = actions + dt * velocity
            time = time + dt
        return actions
