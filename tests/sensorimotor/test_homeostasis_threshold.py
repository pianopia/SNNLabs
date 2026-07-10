from __future__ import annotations

import torch

from src.dst_snn import ChronoPlasticLIFLayer
from src.dst_snn.sensorimotor.homeostasis import HomeostasisController
from src.dst_snn.sensorimotor.world_model import PredictiveWorldModel, train_world_model_step


def test_positive_threshold_offset_reduces_chrono_spikes():
    torch.manual_seed(0)
    layer = ChronoPlasticLIFLayer(8, 16, threshold=0.5, noise_std=0.0)
    layer.eval()
    x = torch.ones(2, 10, 8) * 0.8
    base = layer(x)["spikes"].mean().item()
    high_thr = layer(x, threshold_offset=torch.full((16,), 1.5))["spikes"].mean().item()
    # Raising threshold by 1.5 should suppress most spikes relative to base.
    assert high_thr <= base
    assert high_thr < 0.5


def test_negative_threshold_offset_increases_chrono_spikes():
    torch.manual_seed(0)
    layer = ChronoPlasticLIFLayer(8, 16, threshold=1.5, noise_std=0.0)
    layer.eval()
    x = torch.ones(2, 12, 8) * 0.6
    base = layer(x)["spikes"].mean().item()
    low_thr = layer(x, threshold_offset=torch.full((16,), -1.0))["spikes"].mean().item()
    assert low_thr >= base


def test_world_model_applies_homeostasis_offsets():
    torch.manual_seed(1)
    model = PredictiveWorldModel(sensory_size=12, motor_size=4, latent_size=8, threshold=0.4)
    model.eval()
    sensory = torch.ones(1, 6, 12) * 0.9
    motor = torch.ones(1, 6, 4) * 0.5

    # Prime controller as if all latent neurons were saturated.
    homeo = HomeostasisController(target_rate=0.05, ema_alpha=1.0, gain=4.0, max_offset=1.0)
    hot = torch.ones(1, 4, 8)
    homeo.update(hot)
    assert homeo.threshold_offsets().mean().item() > 0.2

    without = model(sensory, motor)["spikes"].mean().item()
    with_homeo = model(sensory, motor, homeostasis=homeo)["spikes"].mean().item()
    assert with_homeo <= without


def test_train_step_updates_homeostasis_and_records_rate():
    torch.manual_seed(2)
    model = PredictiveWorldModel(sensory_size=10, motor_size=3, latent_size=6, threshold=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    homeo = HomeostasisController(target_rate=0.1, ema_alpha=0.5, gain=2.0)
    sensory = torch.rand(2, 5, 10)
    motor = torch.rand(2, 5, 3)
    metrics = train_world_model_step(model, opt, sensory, motor, homeostasis=homeo)
    assert "latent_spike_rate" in metrics
    assert "homeo_mean_rate" in metrics
    assert homeo.ema_rates is not None
    assert homeo.ema_rates.numel() == 6


def test_homeostasis_loop_raises_offset_and_curbs_rate():
    """With fixed weights and strong input, offsets engage and later rates fall."""
    torch.manual_seed(3)
    model = PredictiveWorldModel(sensory_size=16, motor_size=4, latent_size=12, threshold=0.3)
    model.eval()
    homeo = HomeostasisController(target_rate=0.05, ema_alpha=0.5, gain=4.0, max_offset=1.0)
    sensory = torch.ones(2, 8, 16) * 0.95
    motor = torch.ones(2, 8, 4) * 0.8
    rates = []
    offsets = []
    for _ in range(12):
        with torch.no_grad():
            out = model(sensory, motor, homeostasis=homeo, update_homeostasis=True)
        rates.append(float(out["spikes"].mean().item()))
        offsets.append(float(homeo.threshold_offsets().mean().item()))
    # Feedback loop: high initial activity → positive offset → lower later activity.
    assert max(offsets) > 0.0
    assert rates[0] > 0.0
    assert rates[-1] <= rates[0] + 1e-6
    # Free-running (no offset) should fire at least as much as regulated late rate.
    free = model(sensory, motor)["spikes"].mean().item()
    assert free >= rates[-1] - 1e-6
