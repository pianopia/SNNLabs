from __future__ import annotations

import torch

from benchmarks.neuromorphic.decision_latency import (
    decision_latency_fraction,
    running_predictions,
)


def test_running_predictions_tracks_cumulative_argmax():
    spikes = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    preds = running_predictions(spikes)
    assert preds.shape == (1, 3)
    assert preds[0, 0].item() == 0
    assert preds[0, 2].item() == 1


def test_decision_latency_confirmed_early():
    spikes = torch.zeros(1, 4, 2)
    spikes[0, :, 1] = 1.0
    targets = torch.tensor([1])
    frac = decision_latency_fraction(spikes, targets, confirm_window=2)
    assert frac == 0.25


def test_decision_latency_never_correct_is_one():
    spikes = torch.zeros(1, 4, 2)
    spikes[0, :, 0] = 1.0
    targets = torch.tensor([1])
    frac = decision_latency_fraction(spikes, targets, confirm_window=2)
    assert frac == 1.0
