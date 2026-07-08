#!/usr/bin/env python3
"""Train a tiny DST-SNN on a delayed two-spike temporal pattern.

This script is intentionally small: it generates synthetic spike trains where
class 1 contains feature 0 followed by feature 1 after a fixed delay. The model
must learn to emit an output spike near the second event.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn import DendriticSNN  # noqa: E402


def make_batch(
    batch_size: int,
    time_steps: int,
    in_features: int,
    pattern_delay: int,
    noise_rate: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.rand(batch_size, time_steps, in_features, device=device) < noise_rate).float()
    y = torch.zeros(batch_size, 1, device=device)
    positives = torch.rand(batch_size, device=device) < 0.5
    starts = torch.randint(1, time_steps - pattern_delay - 1, (batch_size,), device=device)

    for batch in range(batch_size):
        if positives[batch]:
            t0 = starts[batch].item()
            x[batch, t0, 0] = 1.0
            x[batch, t0 + pattern_delay, 1] = 1.0
            y[batch, 0] = 1.0
        else:
            t0 = starts[batch].item()
            x[batch, t0, 0] = 1.0
            wrong_delay = max(1, pattern_delay // 2)
            x[batch, min(time_steps - 1, t0 + wrong_delay), 1] = 1.0

    return x, y


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = DendriticSNN(
        in_features=args.in_features,
        out_features=1,
        num_branches=args.branches,
        max_delay=args.max_delay,
        learnable_delay=args.learnable_delay,
        threshold=args.threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        x, y = make_batch(
            args.batch_size,
            args.time_steps,
            args.in_features,
            args.pattern_delay,
            args.noise_rate,
            device,
        )
        out = model(x)
        logits = out["membrane"].amax(dim=1)
        spike_rate = out["spikes"].mean()
        loss = F.binary_cross_entropy_with_logits(logits, y) + args.spike_l1 * spike_rate

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y).float().mean().item()
            print(f"step={step:04d} loss={loss.item():.4f} acc={acc:.3f} spike_rate={spike_rate.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=40)
    parser.add_argument("--in-features", type=int, default=12)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--max-delay", type=int, default=12)
    parser.add_argument("--pattern-delay", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--noise-rate", type=float, default=0.015)
    parser.add_argument("--spike-l1", type=float, default=0.001)
    parser.add_argument("--learnable-delay", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
