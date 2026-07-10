from __future__ import annotations

from pathlib import Path

from benchmarks.sensorimotor.run_synthetic_loop import (
    SyntheticSensorimotorRunner,
    _loss_reduction,
)
from src.dst_snn.eval.runner import run_benchmarks


def test_loss_reduction():
    assert _loss_reduction([1.0, 0.9, 0.5, 0.4]) > 0.0
    assert _loss_reduction([0.4, 0.5, 0.9, 1.0]) == 0.0


def test_synthetic_sensorimotor_runner_outputs_result(tmp_path: Path):
    runner = SyntheticSensorimotorRunner(
        steps=4,
        feature_size=24,
        motor_size=8,
        time_steps=4,
        latent_size=8,
        seed=0,
    )
    results = run_benchmarks([runner], tmp_path)
    result = results[0]
    assert result.benchmark == "synthetic-sensorimotor"
    assert result.metrics.quality_metric == "prediction_loss_reduction"
    assert 0.0 <= result.metrics.quality <= 1.0
    assert result.metrics.extra["steps"] == 4
    assert len(result.metrics.extra["losses"]) == 4
    assert result.metrics.extra["closed_loop"] is True
    assert "linear_probe_accuracy" in result.metrics.extra
    assert "cluster_purity" in result.metrics.extra
    assert "energy_accounting" in result.metrics.extra
    assert result.metrics.extra["actuator_commands_received"] >= 0
    assert (tmp_path / "synthetic-sensorimotor.json").exists()


def test_synthetic_runner_with_ann_baseline(tmp_path: Path):
    runner = SyntheticSensorimotorRunner(
        steps=6,
        feature_size=24,
        motor_size=8,
        time_steps=4,
        latent_size=8,
        seed=1,
        with_ann_baseline=True,
    )
    results = run_benchmarks([runner], tmp_path)
    result = results[0]
    assert result.baseline is not None
    assert result.baseline.quality_metric == "prediction_loss_reduction"
    assert result.baseline.extra["energy_accounting"] == "sensorimotor_ann_mac_proxy_v1"
    assert result.baseline.extra["dense_mac_ops"] > 0
