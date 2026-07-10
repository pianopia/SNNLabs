from __future__ import annotations

import numpy as np
import torch

from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime
from src.dst_snn.sensorimotor.world_model import (
    LearningProgress,
    PredictiveWorldModel,
    train_world_model_step,
)


def test_world_model_forward_and_train_step():
    model = PredictiveWorldModel(sensory_size=12, motor_size=4, latent_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sensory = torch.rand(2, 5, 12)
    motor = torch.rand(2, 5, 4)
    out = model(sensory, motor)
    assert out["prediction"].shape == (2, 5, 12)
    assert out["motor_logits"].shape == (2, 5, 4)
    metrics = train_world_model_step(model, optimizer, sensory, motor)
    assert metrics["prediction_loss"] > 0


def test_learning_progress_intrinsic_reward():
    progress = LearningProgress(alpha=0.5)
    first = progress.update(1.0)
    second = progress.update(0.5)
    third = progress.update(0.8)
    assert first["intrinsic_reward"] == 0.0
    assert second["learning_progress"] > 0.0
    assert second["intrinsic_reward"] > 0.0
    assert third["intrinsic_reward"] == 0.0


def test_runtime_tick_decodes_mock_action():
    registry = ModuleRegistry(feature_size=32, motor_size=16)
    sensor = SyntheticSensor()
    actuator = MockActuator()
    runtime = SensorimotorRuntime(registry, time_steps=4)
    runtime.ingest(sensor.register())
    runtime.ingest(actuator.register())
    runtime.ingest(sensor.observe(0))

    activity = np.zeros(16, dtype=np.float32)
    activity[registry.motor_index(actuator.id, "left") % 16] = 1.0
    result = runtime.tick(activity)
    assert result["spikes"].shape == (4, 32)
    assert result["actions"][0].payload["command"] == "left"
