from __future__ import annotations

import torch
from torch import nn

from src.dst_snn.eval.metrics import (
    accuracy,
    latency_percentiles,
    majority_class_accuracy,
    model_size,
    spike_stats,
)


def test_accuracy_counts_matching_argmax():
    preds = torch.tensor([0, 1, 2, 2])
    targets = torch.tensor([0, 1, 2, 0])
    assert accuracy(preds, targets) == 0.75


def test_majority_class_accuracy():
    targets = torch.tensor([0, 1, 1, 2, 1])
    assert majority_class_accuracy(targets, num_classes=3) == 0.6


def test_majority_class_accuracy_empty_is_zero():
    assert majority_class_accuracy(torch.tensor([]), num_classes=3) == 0.0


def test_majority_class_accuracy_rejects_invalid_class_count():
    try:
        majority_class_accuracy(torch.tensor([0]), num_classes=0)
    except ValueError as exc:
        assert "num_classes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_latency_percentiles():
    out = latency_percentiles([10.0, 20.0, 30.0, 40.0])
    assert out["p50"] == 25.0
    assert out["mean"] == 25.0
    assert out["p95"] >= 38.0


def test_spike_stats():
    spikes = torch.zeros(2, 3, 2)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 0] = 1.0
    spikes[1, 2, 1] = 1.0
    stats = spike_stats(spikes)
    assert stats["spikes_per_inference"] == 1.5
    assert stats["active_neuron_fraction"] == 0.5


def test_model_size():
    module = nn.Linear(4, 2)
    size = model_size(module)
    assert size["param_count"] == 10
    assert size["model_bytes"] == 40
