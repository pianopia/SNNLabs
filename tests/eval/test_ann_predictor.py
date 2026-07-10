from __future__ import annotations

import torch

from src.dst_snn.eval.baselines.ann_predictor import (
    DenseAnnPredictor,
    train_ann_predictor_step,
)


def test_ann_predictor_shapes_and_macs():
    model = DenseAnnPredictor(16, 4, hidden=8)
    sensory = torch.rand(2, 5, 16)
    motor = torch.rand(2, 5, 4)
    out = model(sensory, motor)
    assert out.shape == (2, 5, 16)
    loss = model.prediction_loss(sensory, motor)
    assert loss.ndim == 0
    macs = model.mac_ops_per_inference(5)
    expected = (16 + 4) * 8 * 5 + 8 * 16 * 5
    assert macs == float(expected)


def test_ann_predictor_train_step_decreases_on_constant():
    torch.manual_seed(0)
    model = DenseAnnPredictor(8, 2, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sensory = torch.zeros(1, 4, 8)
    sensory[:, 1:, :] = 1.0
    motor = torch.zeros(1, 4, 2)
    losses = [train_ann_predictor_step(model, opt, sensory, motor) for _ in range(20)]
    assert losses[-1] < losses[0]
