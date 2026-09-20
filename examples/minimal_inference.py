"""Minimal Python inference example for a recorded LeRobot sample."""

import torch

from ipme_vla import IPMEVLAConfig, IPMEVLAPolicy
from ipme_vla.data import LeRobotV21Dataset
from ipme_vla.utils.checkpoint import inspect_checkpoint, load_checkpoint

CHECKPOINT = "outputs/ipme_vla/checkpoint_final.pt"
DATASET = "/path/to/local/lerobot-v21-dataset"

saved = inspect_checkpoint(CHECKPOINT)
config = IPMEVLAConfig.from_dict(saved["config"])
dataset = LeRobotV21Dataset(DATASET, config)
policy = IPMEVLAPolicy(config, saved["normalization_stats"]).eval()
load_checkpoint(CHECKPOINT, policy)

sample = dataset[0]
batch = {
    key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
    for key, value in sample.items()
    if key not in {"episode_index", "frame_index"}
}
policy.reset()
action_chunk = policy.predict_action_chunk(batch)
print(action_chunk.shape)
