"""Offline validation-loss evaluation for IPME-VLA checkpoints."""

from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.data import LeRobotV21Dataset
from ipme_vla.models.vla.policy import IPMEVLAPolicy
from ipme_vla.training.engine import evaluate_loader
from ipme_vla.utils.checkpoint import inspect_checkpoint, load_checkpoint
from ipme_vla.utils.configuration import load_experiment, model_config, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate IPME-VLA losses on recorded data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/ipme_vla_base.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--episodes", type=int, nargs="*")
    return parser


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
    dataset = LeRobotV21Dataset(args.dataset, config, args.episodes)
    normalization_stats = (
        saved.get("normalization_stats", {}) if saved is not None else {}
    )
    if not normalization_stats:
        normalization_stats = dataset.normalization_stats()
    device = resolve_device(args.device)
    policy = IPMEVLAPolicy(config, normalization_stats).to(device)
    load_checkpoint(args.checkpoint, policy, map_location=device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate_loader(policy, loader, device, args.max_batches)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
