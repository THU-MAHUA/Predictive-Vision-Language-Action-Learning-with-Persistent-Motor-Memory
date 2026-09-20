"""Training and validation loops for IPME-VLA."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ipme_vla.config import IPMEVLAConfig
from ipme_vla.models.vla.policy import IPMEVLAPolicy
from ipme_vla.utils.checkpoint import save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def create_optimizer(
    policy: IPMEVLAPolicy,
    training: dict[str, Any],
) -> torch.optim.Optimizer:
    base_lr = float(training.get("learning_rate", 2.5e-5))
    vlm_lr = float(training.get("vlm_learning_rate", 0.0))
    vlm_parameters = []
    other_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".action_expert.backbone.vlm." in name:
            vlm_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = [{"params": other_parameters, "lr": base_lr}]
    if vlm_parameters:
        groups.append({"params": vlm_parameters, "lr": vlm_lr})
    return torch.optim.AdamW(
        groups,
        betas=tuple(training.get("betas", (0.9, 0.95))),
        eps=float(training.get("epsilon", 1e-8)),
        weight_decay=float(training.get("weight_decay", 1e-10)),
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    peak = float(training.get("learning_rate", 2.5e-5))
    floor = float(training.get("decay_learning_rate", 2.5e-6))
    warmup = max(0, int(training.get("warmup_steps", 1000)))
    decay = max(warmup + 1, int(training.get("decay_steps", 30000)))

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / (decay - warmup)))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return floor / peak + (1 - floor / peak) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def evaluate_loader(
    policy: IPMEVLAPolicy,
    loader: Iterable[dict[str, Any]],
    device: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    policy.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = policy(move_batch(batch, device))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    if not count:
        return {}
    return {key: value / count for key, value in totals.items()}


def train_steps(
    policy: IPMEVLAPolicy,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: IPMEVLAConfig,
    training: dict[str, Any],
    normalization_stats: dict[str, Any],
    *,
    device: str,
    start_step: int = 0,
    optimizer=None,
    scheduler=None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = optimizer or create_optimizer(policy, training)
    scheduler = scheduler or create_scheduler(optimizer, training)
    mixed_precision = bool(training.get("mixed_precision", False)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    total_steps = int(training.get("steps", 100000))
    grad_clip = float(training.get("grad_clip_norm", 10.0))
    output_dir = Path(training.get("output_dir", "outputs/ipme_vla"))
    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)

    for step in range(start_step, total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=mixed_precision,
        ):
            loss, metrics = policy(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        completed = step + 1

        if completed % int(training.get("log_every", 100)) == 0:
            values = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            print(f"step={completed} lr={scheduler.get_last_lr()[0]:.3e} {values}")

        validate_every = int(training.get("validate_every", 2000))
        if validation_loader is not None and validate_every and completed % validate_every == 0:
            validation = evaluate_loader(
                policy,
                validation_loader,
                device,
                int(training.get("max_validation_batches", 100)),
            )
            values = " ".join(f"{key}={value:.6f}" for key, value in validation.items())
            print(f"validation step={completed} {values}")

        save_every = int(training.get("save_every", 2000))
        if save_every and completed % save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_{completed:08d}.pt",
                policy,
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed,
                normalization_stats=normalization_stats,
            )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        policy,
        config,
        optimizer=optimizer,
        scheduler=scheduler,
        step=total_steps,
        normalization_stats=normalization_stats,
    )
    return optimizer, scheduler
