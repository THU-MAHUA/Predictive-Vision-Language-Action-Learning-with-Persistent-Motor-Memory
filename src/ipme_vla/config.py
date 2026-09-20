"""Configuration for the standalone IPME-VLA policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PEFTConfig:
    r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: str = "q_proj,v_proj"


@dataclass
class IPMEVLAConfig:
    """Serializable model configuration with the original IPME-VLA defaults."""

    chunk_size: int = 50
    n_action_steps: int = 10
    predictive_horizon: int = 10
    recurrent_unroll_steps: int = 4
    num_memory_tokens: int = 4
    num_consequence_tokens: int = 4
    visual_consequence_weight: float = 0.1
    state_consequence_weight: float = 0.1
    motor_memory_init_std: float = 0.02
    enable_visual_consequence: bool = True
    enable_state_consequence: bool = True
    enable_recurrent_training: bool = True

    max_state_dim: int = 32
    max_action_dim: int = 32
    state_dim: int | None = None
    action_dim: int | None = None
    image_features: list[str] = field(default_factory=list)
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0
    reverse_images_order: bool = False
    shuffle_camera_positions: bool = False

    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    num_steps: int = 10
    use_cache: bool = False
    min_period: float = 4e-3
    max_period: float = 4.0

    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    attention_mode: str = "cross_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = -1
    expert_width_multiplier: float = 0.75
    state_to_prefix: bool = False
    prefix_length: int = -1
    add_image_special_tokens: bool = False
    peft_method: str = ""
    peft_target_model: str = ""
    peft_config: PEFTConfig = field(default_factory=PEFTConfig)

    def __post_init__(self) -> None:
        self.resize_imgs_with_padding = tuple(self.resize_imgs_with_padding)
        if isinstance(self.peft_config, dict):
            self.peft_config = PEFTConfig(**self.peft_config)
        if self.num_memory_tokens <= 0 or self.num_consequence_tokens <= 0:
            raise ValueError("memory and consequence token counts must be positive")
        if self.num_memory_tokens != self.num_consequence_tokens:
            raise ValueError("num_memory_tokens must equal num_consequence_tokens")
        if not 1 <= self.predictive_horizon <= self.chunk_size:
            raise ValueError("predictive_horizon must be in [1, chunk_size]")
        if self.n_action_steps != self.predictive_horizon:
            raise ValueError("n_action_steps must equal predictive_horizon")
        if self.recurrent_unroll_steps < 1:
            raise ValueError("recurrent_unroll_steps must be at least 1")
        if self.state_dim is not None and self.state_dim > self.max_state_dim:
            raise ValueError("state_dim exceeds max_state_dim")
        if self.action_dim is not None and self.action_dim > self.max_action_dim:
            raise ValueError("action_dim exceeds max_action_dim")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [i * self.predictive_horizon for i in range(self.recurrent_unroll_steps + 1)]

    @property
    def action_delta_indices(self) -> list[int]:
        total = (self.recurrent_unroll_steps - 1) * self.predictive_horizon + self.chunk_size
        return list(range(total))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "IPMEVLAConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown IPME-VLA configuration keys: {', '.join(unknown)}")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IPMEVLAConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        return cls.from_dict(values.get("model", values))

