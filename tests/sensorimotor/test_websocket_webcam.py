from __future__ import annotations

import asyncio

import numpy as np

from src.dst_snn.sensorimotor.modules.webcam_sensor import WebcamSensor, opencv_available
from src.dst_snn.sensorimotor.protocol import message_from_json, message_to_json
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime
from src.dst_snn.sensorimotor.websocket_transport import (
    LocalMessageHub,
    handle_module_message,
    websockets_available,
)


def test_local_message_hub_fanout():
    async def _run() -> None:
        hub = LocalMessageHub()
        q1 = hub.subscribe()
        q2 = hub.subscribe()
        from src.dst_snn.sensorimotor.protocol import SensorimotorMessage

        msg = SensorimotorMessage(type="trace", id="core", payload={"step": 1})
        await hub.publish(msg)
        assert (await q1.get()).payload["step"] == 1
        assert (await q2.get()).payload["step"] == 1

    asyncio.run(_run())


def test_handle_module_message_ticks_on_observation():
    async def _run() -> None:
        runtime = SensorimotorRuntime(ModuleRegistry(feature_size=16, motor_size=8), time_steps=4)
        sensor = WebcamSensor(use_camera=False, width=16, height=12)
        await handle_module_message(runtime, sensor.register())
        outbound = await handle_module_message(runtime, sensor.observe(0))
        types = [m.type for m in outbound]
        assert "action" in types or "global_signal" in types or "trace" in types
        assert runtime.step == 1
        assert runtime.global_signal["fatigue"] >= 0.0

    asyncio.run(_run())


def test_webcam_synthetic_emits_motion_after_second_frame():
    sensor = WebcamSensor(use_camera=False, width=32, height=24)
    first = sensor.observe(0)
    second = sensor.observe(1)
    assert first.payload["source"] == "synthetic"
    assert first.payload["motion"] == 0.0
    assert second.payload["motion"] >= 0.0
    assert "event_density" in second.payload


def test_fatigue_increases_with_activity():
    runtime = SensorimotorRuntime(ModuleRegistry(feature_size=16, motor_size=8), time_steps=4)
    # Inject a dense observation via synthetic-like payload by direct registry use.
    from src.dst_snn.sensorimotor.modules import SyntheticSensor

    sensor = SyntheticSensor()
    runtime.ingest(sensor.register())
    fatigues = []
    for step in range(12):
        runtime.ingest(sensor.observe(step))
        # Force high motor activity to keep spikes high through encoding.
        result = runtime.tick(motor_activity=np.ones(8, dtype=np.float64))
        fatigues.append(result["global_signal"]["fatigue"])
    assert fatigues[-1] >= fatigues[0]


def test_websockets_available_is_bool():
    assert isinstance(websockets_available(), bool)
    assert isinstance(opencv_available(), bool)


def test_message_json_roundtrip_for_ws_payload():
    sensor = WebcamSensor(use_camera=False)
    msg = sensor.register()
    restored = message_from_json(message_to_json(msg))
    assert restored.type == "register"
    assert restored.id == "webcam-sensor"
