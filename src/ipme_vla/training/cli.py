"""Command-line training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ipme_vla.data import LeRobotV21Dataset, LeRobotV21Metadata, split_episodes
from ipme_vla.models.vla.policy import IPMEVLAPolicy
from ipme_vla.training.engine import (
    create_optimizer,
    create_scheduler,
    seed_everything,
    train_steps,
)
from ipme_vla.utils.checkpoint import inspect_checkpoint, load_checkpoint
from ipme_vla.utils.configuration import load_experiment, model_config, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IPME-VLA on a local LeRobot v2.1 dataset.")
    parser.add_argument("--config", default="configs/ipme_vla_base.yaml")
    parser.add_argument("--dataset", help="Local LeRobot v2.1 dataset root.")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", help="Versioned IPME-VLA checkpoint to resume.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    experiment = load_experiment(args.config)
    data = experiment.setdefault("data", {})
    training = experiment.setdefault("training", {})
    dataset_root = args.dataset or data.get("dataset_root")
    if not dataset_root:
        raise SystemExit("--dataset is required (or set data.dataset_root in YAML)")
    if args.output_dir:
        training["output_dir"] = args.output_dir
    if args.steps is not None:
        training["steps"] = args.steps
    if args.device:
        training["device"] = args.device
    seed = int(training.get("seed", 1000))
    seed_everything(seed)
    metadata = LeRobotV21Metadata(dataset_root)

    config = model_config(experiment)
    metadata.configure_model(config)
    normalization_stats = {}
    if args.resume:
        try:
            saved = inspect_checkpoint(args.resume)
        except RuntimeError:
            saved = None
        if saved is not None:
            config = model_config({"model": saved["config"]})
            metadata.configure_model(config)
            normalization_stats = saved.get("normalization_stats", {})

    explicit_episodes = data.get("train_episodes")
    if explicit_episodes is None:
        train_episodes, validation_episodes = split_episodes(
            metadata, float(data.get("validation_fraction", 0.05)), seed
        )
    else:
        train_episodes = [int(value) for value in explicit_episodes]
        validation_episodes = [
            index for index in sorted(metadata.episodes) if index not in train_episodes
        ]
    train_dataset = LeRobotV21Dataset(dataset_root, config, train_episodes)
    if not normalization_stats:
        normalization_stats = train_dataset.normalization_stats()
    validation_dataset = (
        LeRobotV21Dataset(dataset_root, config, validation_episodes)
        if validation_episodes
        else None
    )
    device = resolve_device(str(training.get("device", "cuda")))
    policy = IPMEVLAPolicy(config, normalization_stats).to(device)
    optimizer = create_optimizer(policy, training)
    scheduler = create_scheduler(optimizer, training)
    start_step = 0
    if args.resume:
        resumed = load_checkpoint(
            args.resume,
            policy,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_step = int(resumed.get("step", 0))
        print(f"loaded {Path(args.resume)} at step {start_step}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=int(training.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(training.get("num_workers", 4)),
        )
        if validation_dataset is not None
        else None
    )
    train_steps(
        policy,
        train_loader,
        validation_loader,
        config,
        training,
        normalization_stats,
        device=device,
        start_step=start_step,
        optimizer=optimizer,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
