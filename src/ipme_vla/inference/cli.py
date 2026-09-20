"""Checkpoint inference on a recorded LeRobot observation."""

from __future__ import annotations

import argparse
import json

import torch

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.data import LeRobotV21Dataset
from ipme_vla.models.vla.policy import IPMEVLAPolicy
from ipme_vla.training.engine import move_batch
from ipme_vla.utils.checkpoint import inspect_checkpoint, load_checkpoint
from ipme_vla.utils.configuration import load_experiment, model_config, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an IPME-VLA action chunk.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/ipme_vla_base.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def _batch_sample(sample: dict) -> dict:
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in sample.items()
        if key not in {"episode_index", "frame_index"}
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        saved = inspect_checkpoint(args.checkpoint)
    except RuntimeError:
        saved = None
    config = (
        IPMEVLAConfig.from_dict(saved["config"])
        if saved is not None
        else model_config(load_experiment(args.config))
    )
    dataset = LeRobotV21Dataset(args.dataset, config)
    if not 0 <= args.sample_index < len(dataset):
        raise SystemExit(f"--sample-index must be in [0, {len(dataset) - 1}]")
    device = resolve_device(args.device)
    normalization_stats = (
        saved.get("normalization_stats", {}) if saved is not None else {}
    )
    if not normalization_stats:
        normalization_stats = dataset.normalization_stats()
    policy = IPMEVLAPolicy(config, normalization_stats).to(device)
    load_checkpoint(args.checkpoint, policy, map_location=device)
    batch = move_batch(_batch_sample(dataset[args.sample_index]), device)
    policy.reset()
    actions = policy.predict_action_chunk(batch)
    print(json.dumps({"actions": actions.cpu().tolist()}, indent=2))


if __name__ == "__main__":
    main()
