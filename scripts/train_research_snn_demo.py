#!/usr/bin/env python3
"""Minimal training demo for the 2025-2026 modular SNN components."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn import (  # noqa: E402
    ResearchSpikingTransformerSNN,
    ThresholdGuardingLoss,
    research_snn_training_step,
)


def make_high_frequency_batch(
    batch_size: int,
    time_steps: int,
    image_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a binary spike-image task where class 1 contains a checker path."""
    x = (torch.rand(batch_size, time_steps, 1, image_size, image_size, device=device) < 0.02).float()
    y = torch.randint(0, 2, (batch_size,), device=device)
    checker = (torch.arange(image_size, device=device).view(1, -1)
               + torch.arange(image_size, device=device).view(-1, 1)) % 2
    smooth = torch.zeros(image_size, image_size, device=device)
    smooth[image_size // 4:image_size // 2, image_size // 4:image_size // 2] = 1.0
    for batch in range(batch_size):
        event_time = torch.randint(1, time_steps, (1,), device=device).item()
        if y[batch].item() == 1:
            x[batch, event_time, 0] = checker
        else:
            x[batch, event_time, 0] = smooth
    confidence = torch.full((batch_size,), 0.85, device=device)
    confidence[x.flatten(1).mean(dim=1) > 0.45] = 0.4
    return {"x": x, "y": y, "confidence": confidence}


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = ResearchSpikingTransformerSNN(
        in_channels=1,
        num_classes=2,
        image_size=args.image_size,
        patch_size=4,
        embed_dim=args.embed_dim,
        depth=args.depth,
        tgo_noise_std=args.noise_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tgo = ThresholdGuardingLoss(margin=args.tgo_margin)

    for step in range(1, args.steps + 1):
        batch = make_high_frequency_batch(args.batch_size, args.time_steps, args.image_size, device)
        metrics = research_snn_training_step(
            model,
            batch,
            optimizer,
            tgo_loss=tgo,
            tgo_weight=args.tgo_weight,
            spike_l1=args.spike_l1,
        )
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:04d} loss={metrics['loss']:.4f} "
                f"task={metrics['task_loss']:.4f} tgo={metrics['tgo_loss']:.4f} "
                f"spike_rate={metrics['spike_rate']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--tgo-margin", type=float, default=0.2)
    parser.add_argument("--tgo-weight", type=float, default=0.05)
    parser.add_argument("--spike-l1", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
