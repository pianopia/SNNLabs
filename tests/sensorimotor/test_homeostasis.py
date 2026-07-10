from __future__ import annotations

import torch

from src.dst_snn.sensorimotor.homeostasis import (
    ExperienceBuffer,
    HomeostasisController,
    representation_stability,
    sleep_replay,
)
from src.dst_snn.sensorimotor.world_model import PredictiveWorldModel


def test_homeostasis_raises_threshold_for_hot_neurons():
    ctrl = HomeostasisController(target_rate=0.05, ema_alpha=1.0)
    spikes = torch.zeros(1, 4, 3)
    spikes[..., 0] = 1.0  # neuron 0 always fires
    stats = ctrl.update(spikes)
    offsets = ctrl.threshold_offsets()
    assert offsets[0] > offsets[1]
    assert stats["mean_rate"] > 0.0


def test_experience_buffer_prefers_high_salience():
    buf = ExperienceBuffer(capacity=8)
    low = torch.zeros(1, 2, 4)
    high = torch.ones(1, 2, 4)
    motor = torch.zeros(1, 2, 2)
    buf.add(low, motor, salience=0.1)
    buf.add(high, motor, salience=0.9)
    batch = buf.high_salience_batch(k=1)
    assert batch is not None
    sensory, _ = batch
    assert float(sensory.mean()) == 1.0


def test_sleep_replay_runs_optimizer_steps():
    model = PredictiveWorldModel(sensory_size=6, motor_size=2, latent_size=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    buf = ExperienceBuffer(capacity=8)
    sensory = torch.rand(1, 4, 6)
    motor = torch.rand(1, 4, 2)
    for _ in range(4):
        buf.add(sensory, motor, salience=1.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = sleep_replay(model, opt, buf, steps=2, batch_k=2)
    assert stats["replay_steps"] == 2.0
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed


def test_representation_stability_identical_is_high():
    z = torch.ones(2, 3, 4)
    stats = representation_stability([z, z, z])
    assert stats["mean_cosine"] > 0.99
    assert stats["mean_abs_delta"] < 1e-6
    assert stats["stability"] > 0.9
