"""Offline tests for serial motor / tactile bridges (no real hardware)."""

from __future__ import annotations

from src.dst_snn.sensorimotor.modules.serial_bridge import (
    MockSerialPort,
    SerialMotorBridge,
    SerialTactileSensor,
    encode_line,
)
from src.dst_snn.sensorimotor.protocol import SensorimotorMessage


def test_motor_bridge_writes_json_command():
    port = MockSerialPort()
    motor = SerialMotorBridge(n_channels=3)
    motor.attach(port)
    msg = SensorimotorMessage(type="action", id="core", payload={"values": [0.5, -0.25, 1.0]})
    cmd = motor.on_action(msg)
    assert cmd == [0.5, -0.25, 1.0]
    assert len(port.written) == 1
    assert b'"type":"motor"' in port.written[0]
    assert motor.last_command == cmd


def test_motor_bridge_binary():
    port = MockSerialPort()
    motor = SerialMotorBridge(n_channels=2, binary=True)
    motor.attach(port)
    motor.on_action(SensorimotorMessage(type="action", id="c", payload={"values": [1.0, 2.0]}))
    assert len(port.written[0]) == 8


def test_tactile_poll_json():
    port = MockSerialPort()
    sensor = SerialTactileSensor(n_taxels=4)
    sensor.attach(port)
    sensor.inject_raw_json({"type": "tactile", "values": [0.1, 0.2, 0.3, 0.4]})
    obs = sensor.poll()
    assert obs is not None
    assert obs.type == "observation"
    assert obs.payload["values"] == [0.1, 0.2, 0.3, 0.4]


def test_tactile_poll_binary():
    port = MockSerialPort()
    sensor = SerialTactileSensor(n_taxels=2, binary=True)
    sensor.attach(port)
    sensor.inject_floats([1.5, -0.5])
    obs = sensor.poll()
    assert obs is not None
    assert abs(obs.payload["values"][0] - 1.5) < 1e-5
    assert abs(obs.payload["values"][1] + 0.5) < 1e-5


def test_encode_line_roundtrip_register():
    reg = SerialMotorBridge().register()
    line = encode_line(reg)
    assert line.endswith(b"\n")
    assert b"serial-motor" in line or b"motor" in line
