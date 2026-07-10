from __future__ import annotations

import numpy as np

from src.dst_snn.sensorimotor.codec import decode_action, encode_observations
from src.dst_snn.sensorimotor.protocol import SensorimotorMessage, register_message
from src.dst_snn.sensorimotor.registry import ModuleRegistry


def test_registry_apply_register_observation_deregister():
    registry = ModuleRegistry(feature_size=32, motor_size=8)
    registry.apply(register_message(module_id="sensor", role="sensor", modality="synthetic", shape=[1]))
    registry.apply(SensorimotorMessage(type="observation", id="sensor", payload={"value": 0.75}))
    assert "sensor" in registry.modules
    assert registry.latest_observations["sensor"]["value"] == 0.75
    registry.apply(SensorimotorMessage(type="deregister", id="sensor"))
    assert "sensor" not in registry.modules
    assert "sensor" not in registry.latest_observations


def test_encode_observations_fixed_shape_and_sparse():
    registry = ModuleRegistry(feature_size=32, motor_size=8)
    registry.apply(register_message(module_id="sensor", role="sensor", modality="synthetic", shape=[1]))
    registry.apply(SensorimotorMessage(type="observation", id="sensor", payload={"value": 0.75, "label": "hot"}))
    spikes = encode_observations(registry, time_steps=5)
    assert spikes.shape == (5, 32)
    assert spikes.sum() > 0
    assert spikes.max() <= 1.0


def test_decode_action_command():
    registry = ModuleRegistry(feature_size=32, motor_size=16)
    registry.apply(
        register_message(
            module_id="motor",
            role="actuator",
            modality="motor",
            shape=[1],
            action_space={"commands": ["left", "right"]},
        )
    )
    activity = np.zeros(16, dtype=np.float32)
    activity[registry.motor_index("motor", "right") % 16] = 0.9
    actions = decode_action(registry, activity)
    assert len(actions) == 1
    assert actions[0].payload["command"] == "right"
