"""Public training and inference policy for IPME-VLA."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoProcessor

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.constants import ACTION, OBS_STATE, TASK
from ipme_vla.models.vla.modeling import IPMEVLAModel
from ipme_vla.utils.normalization import TensorNormalizer, TensorUnnormalizer
from ipme_vla.utils.tensors import pad_tensor, pad_vector, resize_with_pad


def _padded_stats(
    stats: dict[str, Any] | None,
    dimension: int,
) -> dict[str, Tensor]:
    stats = stats or {}
    output: dict[str, Tensor] = {}
    defaults = {"mean": 0.0, "std": 1.0}
    for name, fill in defaults.items():
        value = torch.as_tensor(stats.get(name, []), dtype=torch.float32).flatten()
        padded = torch.full((dimension,), fill, dtype=torch.float32)
        padded[: min(value.numel(), dimension)] = value[:dimension]
        output[name] = padded
    return output


class IPMEVLAPolicy(nn.Module):
    """Standalone IPME-VLA interface for training and episodic inference."""

    config_class = IPMEVLAConfig
    name = "ipme_vla"

    def __init__(
        self,
        config: IPMEVLAConfig,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        model: IPMEVLAModel | None = None,
        language_tokenizer=None,
    ) -> None:
        super().__init__()
        self.config = config
        dataset_stats = dataset_stats or {}
        self.normalize_state = TensorNormalizer(
            _padded_stats(dataset_stats.get("state"), config.max_state_dim),
            config.max_state_dim,
        )
        self.normalize_action = TensorNormalizer(
            _padded_stats(dataset_stats.get("action"), config.max_action_dim),
            config.max_action_dim,
        )
        self.unnormalize_action = TensorUnnormalizer(
            _padded_stats(dataset_stats.get("action"), config.max_action_dim),
            config.max_action_dim,
        )
        self.model = model or IPMEVLAModel(config)
        if language_tokenizer is None:
            language_tokenizer = AutoProcessor.from_pretrained(config.vlm_model_name).tokenizer
        self.language_tokenizer = language_tokenizer
        self.reset()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset(self) -> None:
        """Reset action buffering and persistent memory at an episode boundary."""
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self._motor_memory: Tensor | None = None
        self._last_prediction: dict[str, Tensor] | None = None

    def get_optim_params(self):
        return self.parameters()

    def _prepare_language(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        tasks = batch[TASK]
        if isinstance(tasks, str):
            tasks = [tasks]
        batch_size = batch[OBS_STATE].shape[0]
        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            raise ValueError(f"received {len(tasks)} tasks for a batch of {batch_size}")
        prompts = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        tokenized = self.language_tokenizer(
            prompts,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        device = batch[OBS_STATE].device
        return (
            tokenized["input_ids"].to(device),
            tokenized["attention_mask"].to(device=device, dtype=torch.bool),
        )

    def _image_keys(self, batch: dict[str, Any]) -> list[str]:
        keys = self.config.image_features or sorted(
            key for key in batch if key.startswith("observation.images.")
        )
        present = [key for key in keys if key in batch]
        if not present:
            raise ValueError(
                "No configured image feature is present. Set model.image_features or "
                "use LeRobot visual feature names."
            )
        if self.config.shuffle_camera_positions and self.training:
            present = random.sample(present, len(present))
        if self.config.reverse_images_order:
            present.reverse()
        return present

    @staticmethod
    def _observation_slot(tensor: Tensor, step: int, horizon: int) -> int:
        if tensor.ndim < 3:
            return 0
        dense_index = step * horizon
        return dense_index if tensor.shape[1] > dense_index else step

    def _prepare_images(
        self,
        batch: dict[str, Any],
        step: int = 0,
    ) -> tuple[list[Tensor], list[Tensor]]:
        images: list[Tensor] = []
        masks: list[Tensor] = []
        keys = self._image_keys(batch)
        reference_image: Tensor | None = None
        reference_mask: Tensor | None = None
        for key in keys:
            image = batch[key]
            if image.ndim == 5:
                slot = self._observation_slot(
                    batch[OBS_STATE], step, self.config.predictive_horizon
                )
                image = image[:, min(slot, image.shape[1] - 1)]
            if self.config.resize_imgs_with_padding:
                image = resize_with_pad(image, *self.config.resize_imgs_with_padding, pad_value=0)
            image = image * 2.0 - 1.0
            mask = batch.get(f"{key}_padding_mask")
            if mask is None:
                mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            elif mask.ndim > 1:
                slot = self._observation_slot(
                    batch[OBS_STATE], step, self.config.predictive_horizon
                )
                mask = mask[:, min(slot, mask.shape[1] - 1)]
            images.append(image)
            masks.append(mask.bool())
            reference_image, reference_mask = image, mask.bool()
        for _ in range(self.config.empty_cameras):
            if reference_image is None or reference_mask is None:
                break
            images.append(torch.full_like(reference_image, -1))
            masks.append(torch.zeros_like(reference_mask))
        return images, masks

    def _prepare_state(self, batch: dict[str, Any], step: int = 0) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim > 2:
            slot = self._observation_slot(state, step, self.config.predictive_horizon)
            state = state[:, min(slot, state.shape[1] - 1)]
        return self.normalize_state(pad_vector(state, self.config.max_state_dim))

    def _prepare_actions(self, batch: dict[str, Any], start: int, length: int) -> Tensor:
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions[:, None]
        actions = actions[:, start : start + length]
        actions = pad_tensor(actions, length)
        return self.normalize_action(pad_vector(actions, self.config.max_action_dim))

    def _valid_transition(
        self,
        batch: dict[str, Any],
        step: int,
        action_start: int,
        current_masks: list[Tensor],
        future_masks: list[Tensor],
    ) -> Tensor:
        valid = torch.ones(
            batch[OBS_STATE].shape[0], dtype=torch.bool, device=batch[OBS_STATE].device
        )
        state_padding = batch.get(f"{OBS_STATE}_is_pad")
        if state_padding is not None and state_padding.ndim > 1:
            current = min(step, state_padding.shape[1] - 1)
            future = min(step + 1, state_padding.shape[1] - 1)
            valid &= ~state_padding[:, current].bool()
            valid &= ~state_padding[:, future].bool()
        action_padding = batch.get("action_is_pad", batch.get("actions_id_pad"))
        if action_padding is not None:
            window = action_padding[
                :, action_start : action_start + self.config.predictive_horizon
            ]
            valid &= ~window.bool().any(dim=1)
        valid &= torch.stack(current_masks, dim=1).any(dim=1)
        valid &= torch.stack(future_masks, dim=1).any(dim=1)
        return valid

    def _temporal_forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        batch_size = batch[OBS_STATE].shape[0]
        memory = self.model.initial_memory(
            batch_size, batch[OBS_STATE].device, torch.float32
        )
        language_tokens, language_masks = self._prepare_language(batch)
        flow_losses: list[Tensor] = []
        visual_losses: list[Tensor] = []
        state_losses: list[Tensor] = []
        visual_mae_values: list[Tensor] = []
        state_mae_values: list[Tensor] = []

        for step in range(self.config.recurrent_unroll_steps):
            action_start = step * self.config.predictive_horizon
            images, image_masks = self._prepare_images(batch, step)
            state = self._prepare_state(batch, step)
            prefix = self.model.encode_prefix(
                images, image_masks, language_tokens, language_masks, state
            )
            actions = self._prepare_actions(batch, action_start, self.config.chunk_size)
            flow, _ = self.model.flow_loss(prefix, state, actions, memory)
            action_padding = batch.get("action_is_pad", batch.get("actions_id_pad"))
            if action_padding is not None:
                mask = action_padding[
                    :, action_start : action_start + self.config.chunk_size
                ]
                mask = pad_tensor(mask, self.config.chunk_size, pad_value=True)
                flow = flow * (~mask.bool()).unsqueeze(-1)
            flow_losses.append(flow.mean(dim=(1, 2)))

            future_images, future_masks = self._prepare_images(batch, step + 1)
            future_state = self._prepare_state(batch, step + 1)
            future_prefix = self.model.encode_prefix(
                future_images,
                future_masks,
                language_tokens,
                language_masks,
                future_state,
            )
            valid = self._valid_transition(
                batch, step, action_start, image_masks, future_masks
            )
            prediction = self.model.predict_consequence(
                prefix,
                actions[:, : self.config.predictive_horizon],
                memory,
                state,
            )
            if self.config.enable_visual_consequence:
                target_visual = future_prefix[3].detach() - prefix[3].detach()
                loss = self.model.consequence_model.visual_loss(
                    prediction["pred_visual_delta"], target_visual
                )
                visual_losses.append(torch.where(valid, loss, torch.zeros_like(loss)))
                visual_mae = (
                    prediction["pred_visual_delta"] - target_visual
                ).abs().mean(dim=-1)
                visual_mae_values.append(
                    torch.where(valid, visual_mae, torch.zeros_like(visual_mae))
                )
            if self.config.enable_state_consequence:
                target_state = (future_state - state).detach()
                dimensions = self.config.state_dim or min(
                    batch[OBS_STATE].shape[-1], self.config.max_state_dim
                )
                loss = self.model.consequence_model.state_loss(
                    prediction["pred_state_delta"], target_state, dimensions
                )
                state_losses.append(torch.where(valid, loss, torch.zeros_like(loss)))
                state_mae = (
                    prediction["pred_state_delta"][:, :dimensions]
                    - target_state[:, :dimensions]
                ).abs().mean(dim=-1)
                state_mae_values.append(
                    torch.where(valid, state_mae, torch.zeros_like(state_mae))
                )
            memory = self.model.motor_memory.select(
                valid, prediction["next_memory"], memory
            )

        zero = batch[OBS_STATE].new_zeros(())
        flow_loss = torch.stack(flow_losses, dim=1).mean() if flow_losses else zero
        visual_loss = (
            torch.stack(visual_losses, dim=1).mean() if visual_losses else zero
        )
        state_loss = torch.stack(state_losses, dim=1).mean() if state_losses else zero
        visual_mae = (
            torch.stack(visual_mae_values, dim=1).mean()
            if visual_mae_values
            else zero
        )
        state_mae = (
            torch.stack(state_mae_values, dim=1).mean() if state_mae_values else zero
        )
        total = flow_loss
        total = total + self.config.visual_consequence_weight * visual_loss
        total = total + self.config.state_consequence_weight * state_loss
        return total, {
            "loss": float(total.detach()),
            "loss_fm": float(flow_loss.detach()),
            "loss_visual_consequence": float(visual_loss.detach()),
            "loss_state_consequence": float(state_loss.detach()),
            "visual_delta_mae": float(visual_mae.detach()),
            "state_delta_mae": float(state_mae.detach()),
            "motor_memory_norm": float(memory.detach().norm()),
        }

    def forward(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        temporal = batch[OBS_STATE].ndim > 2 and batch[ACTION].ndim > 2
        if temporal and self.config.enable_recurrent_training:
            return self._temporal_forward(batch)
        images, image_masks = self._prepare_images(batch)
        state = self._prepare_state(batch)
        language_tokens, language_masks = self._prepare_language(batch)
        prefix = self.model.encode_prefix(
            images, image_masks, language_tokens, language_masks, state
        )
        actions = self._prepare_actions(batch, 0, self.config.chunk_size)
        memory = self.model.initial_memory(actions.shape[0], actions.device, torch.float32)
        losses, _ = self.model.flow_loss(
            prefix, state, actions, memory, noise=noise, time=time
        )
        loss = losses.mean()
        return loss, {"loss": float(loss.detach()), "loss_fm": float(loss.detach())}

    @torch.no_grad()
    def _generate(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        images, image_masks = self._prepare_images(batch)
        state = self._prepare_state(batch)
        language_tokens, language_masks = self._prepare_language(batch)
        actions, next_memory, diagnostics = self.model.generate(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            memory=self._motor_memory,
            noise=noise,
        )
        action_dimension = self.config.action_dim or self.config.max_action_dim
        actions = self.unnormalize_action(actions)[..., :action_dimension]
        return actions, next_memory.detach(), diagnostics

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
    ) -> Tensor:
        self.eval()
        actions, _, _ = self._generate(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
    ) -> Tensor:
        self.eval()
        if not self._action_queue:
            actions, next_memory, diagnostics = self._generate(batch, noise=noise)
            self._motor_memory = next_memory
            self._last_prediction = {
                key: value.detach()
                for key, value in diagnostics.items()
                if isinstance(value, Tensor)
            }
            self._action_queue.extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )
        return self._action_queue.popleft()
