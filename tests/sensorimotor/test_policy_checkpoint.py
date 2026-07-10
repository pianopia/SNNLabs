from __future__ import annotations

from pathlib import Path

import torch

from src.dst_snn.sensorimotor.checkpoint import (
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)
from src.dst_snn.sensorimotor.policy import IntrinsicMotorPolicy
from src.dst_snn.sensorimotor.protocol import register_message
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.world_model import LearningProgress, PredictiveWorldModel


def _registry() -> ModuleRegistry:
    registry = ModuleRegistry(feature_size=16, motor_size=8)
    registry.apply(
        register_message(
            module_id="motor",
            role="actuator",
            modality="motor",
            shape=[1],
            action_space={"commands": ["left", "right"]},
        )
    )
    return registry


def test_intrinsic_policy_activity_and_update():
    registry = _registry()
    policy = IntrinsicMotorPolicy(epsilon=0.0, seed=0)
    motor, selected = policy.activity(registry)
    assert motor.shape == (8,)
    assert len(selected) == 1
    policy.update(selected, intrinsic_reward=1.0)
    key = f"{selected[0]['module_id']}:{selected[0]['command']}"
    assert policy.command_scores[key] > 0.0
    restored = IntrinsicMotorPolicy.from_state_dict(policy.state_dict())
    assert restored.command_scores == policy.command_scores


def test_world_model_checkpoint_roundtrip(tmp_path: Path):
    model = PredictiveWorldModel(sensory_size=8, motor_size=4, latent_size=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    progress = LearningProgress(ema_loss=0.5, alpha=0.2)
    path = tmp_path / "world-model.pt"
    save_world_model_checkpoint(path, model, optimizer, progress, extra={"step": 3})

    restored, restored_optimizer, restored_progress, extra = load_world_model_checkpoint(
        path,
        with_optimizer=True,
    )
    assert restored.sensory_size == 8
    assert restored.motor_size == 4
    assert restored.latent_size == 6
    assert restored_optimizer is not None
    assert restored_progress is not None
    assert restored_progress.ema_loss == 0.5
    assert extra["step"] == 3
