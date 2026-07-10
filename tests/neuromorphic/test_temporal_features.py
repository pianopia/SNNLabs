from __future__ import annotations

import torch

from benchmarks.neuromorphic.classifier import SnnClassifier
from benchmarks.neuromorphic.temporal_features import (
    TemporalFeatureFrontEnd,
    causal_ema,
    temporal_difference,
)


def test_causal_ema_and_diff_shapes():
    x = torch.zeros(2, 5, 3)
    x[:, 0, 0] = 1.0
    x[:, 2, 1] = 1.0
    rate = causal_ema(x, alpha=0.5)
    diff = temporal_difference(x)
    assert rate.shape == x.shape
    assert diff.shape == x.shape
    assert rate[0, 0, 0] == 0.5
    assert diff[0, 1, 0] == -1.0


def test_temporal_frontend_stack_and_project():
    front = TemporalFeatureFrontEnd(4, project_to=0)
    x = torch.rand(2, 6, 4)
    out = front(x)
    assert out.shape == (2, 6, 12)

    front_p = TemporalFeatureFrontEnd(4, project_to=8)
    out_p = front_p(x)
    assert out_p.shape == (2, 6, 8)


def test_classifier_with_temporal_features_forward():
    model = SnnClassifier(
        in_features=8,
        num_classes=3,
        num_branches=2,
        max_delay=2,
        use_temporal_features=True,
        temporal_project_to=6,
        hidden_features=4,
        threshold=0.3,
    )
    x = torch.rand(2, 5, 8)
    out = model(x)
    assert out["logits"].shape == (2, 3)
    loss = out["logits"].sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
