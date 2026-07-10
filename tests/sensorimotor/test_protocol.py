from __future__ import annotations

import pytest

from src.dst_snn.sensorimotor.protocol import (
    SensorimotorMessage,
    message_from_json,
    message_to_json,
    register_message,
)


def test_message_roundtrip():
    msg = SensorimotorMessage(type="observation", id="cam", payload={"x": 1})
    restored = message_from_json(message_to_json(msg))
    assert restored.type == "observation"
    assert restored.id == "cam"
    assert restored.payload["x"] == 1


def test_register_message_shape():
    msg = register_message(module_id="arm", role="actuator", modality="motor", shape=[2], action_space={"axes": ["x"]})
    assert msg.type == "register"
    assert msg.payload["action_space"]["axes"] == ["x"]


def test_invalid_message_type_rejected():
    with pytest.raises(ValueError):
        message_from_json('{"type":"bad","id":"x","payload":{}}')
