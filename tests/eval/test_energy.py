from __future__ import annotations

import math

from src.dst_snn.eval.energy import (
    EnergyModel,
    dense_linear_mac_ops,
    dense_mac_energy_pj,
    energy_ratio,
    estimate_snn_classifier_ops,
    snn_energy_pj,
)


def test_defaults():
    m = EnergyModel()
    assert m.mac_pj == 0.9
    assert m.ac_pj == 0.1
    assert m.source


def test_snn_energy_is_spikes_times_fanout_times_ac():
    m = EnergyModel()
    assert snn_energy_pj(total_spikes=100, fanout=10, model=m) == 100.0


def test_dense_energy_is_macs_times_mac_cost():
    m = EnergyModel()
    assert dense_mac_energy_pj(mac_ops=1000, model=m) == 900.0


def test_energy_ratio_reports_efficiency_factor():
    assert energy_ratio(snn_pj=100.0, dense_pj=900.0) == 9.0


def test_energy_ratio_infinite_when_snn_zero():
    assert math.isinf(energy_ratio(snn_pj=0.0, dense_pj=900.0))


def test_dense_linear_mac_ops_scales_with_time():
    assert dense_linear_mac_ops(10, 5, time_steps=4) == 200.0


def test_estimate_snn_classifier_ops_includes_hidden_layers():
    plain = estimate_snn_classifier_ops(
        in_features=100,
        num_classes=10,
        time_bins=8,
        spikes_per_inference=5.0,
    )
    deep = estimate_snn_classifier_ops(
        in_features=100,
        num_classes=10,
        time_bins=8,
        hidden_features=64,
        chrono_hidden=32,
        spikes_per_inference=5.0,
    )
    assert deep["dense_mac_ops"] > plain["dense_mac_ops"]
    assert deep["layer_count"] == 3.0
    assert plain["layer_count"] == 1.0
    assert deep["effective_fanout"] > 0.0
