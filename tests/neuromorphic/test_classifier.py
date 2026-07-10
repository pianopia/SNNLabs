from __future__ import annotations

import torch

from benchmarks.neuromorphic.classifier import SnnClassifier


def test_forward_shapes_plain():
    model = SnnClassifier(in_features=16, num_classes=4, num_branches=4, max_delay=4)
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)
    assert out["spikes"].shape == (2, 8, 4)
    assert out["membrane"].shape == (2, 8, 4)


def test_forward_shapes_with_chrono_frontend():
    model = SnnClassifier(
        in_features=16,
        num_classes=4,
        num_branches=4,
        max_delay=4,
        use_chrono=True,
        chrono_hidden=12,
    )
    x = torch.rand(2, 8, 16)
    out = model(x)
    assert out["logits"].shape == (2, 4)


def test_logits_are_differentiable():
    model = SnnClassifier(in_features=8, num_classes=3, num_branches=2, max_delay=2)
    x = torch.rand(2, 5, 8)
    out = model(x)
    loss = out["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
