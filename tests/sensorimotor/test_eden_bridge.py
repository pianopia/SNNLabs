"""Tests for EDEN ↔ Python sensorimotor bridge (offline)."""

from __future__ import annotations

from src.dst_snn.sensorimotor.eden_bridge import (
    EdenBridgeSession,
    eden_events_to_messages,
    message_to_eden_event,
)
from src.dst_snn.sensorimotor.protocol import SensorimotorMessage


def test_body_event_to_observation():
    msgs = eden_events_to_messages(
        [{"kind": "body", "value": 0.2, "meta": {"gait_drive": 0.5, "overload": 0.1}}]
    )
    assert len(msgs) == 1
    assert msgs[0].type == "observation"
    assert msgs[0].payload["values"][4] == 0.5  # gait_drive
    assert msgs[0].payload["values"][0] == 0.1  # overload


def test_global_signal_and_spike():
    msgs = eden_events_to_messages(
        [
            {"kind": "global_signal", "value": 0.8, "meta": {"novelty": 0.3}},
            {"kind": "spike", "value": 12.0},
        ]
    )
    assert msgs[0].type == "global_signal"
    assert msgs[0].payload["reward"] == 0.8
    assert msgs[1].type == "trace"


def test_message_to_eden_event_action():
    msg = SensorimotorMessage(type="action", id="core", payload={"values": [0.2, 0.4]})
    event = message_to_eden_event(msg)
    assert event["kind"] == "action"
    assert event["meta"]["values"] == [0.2, 0.4]


def test_session_decode_batch():
    session = EdenBridgeSession()
    reg = session.register()
    assert reg.type == "register"
    msgs = session.decode_inbound_json(
        '[{"kind":"body","meta":{"nearest_stimulus":0.7}}]'
    )
    assert msgs[0].type == "observation"
    assert msgs[0].payload["values"][2] == 0.7


def test_session_decode_protocol_message():
    session = EdenBridgeSession()
    text = (
        '{"type":"observation","id":"eden-body","ts":1.0,'
        '"payload":{"values":[1,2,3,4,5,6]}}'
    )
    msgs = session.decode_inbound_json(text)
    assert len(msgs) == 1
    assert msgs[0].payload["values"][0] == 1
