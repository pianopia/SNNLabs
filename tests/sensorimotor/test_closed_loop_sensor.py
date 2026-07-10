from __future__ import annotations

from src.dst_snn.sensorimotor.modules import MockActuator, SyntheticSensor
from src.dst_snn.sensorimotor.protocol import SensorimotorMessage


def test_synthetic_sensor_phase_bin_in_payload():
    sensor = SyntheticSensor(n_phase_bins=8)
    obs = sensor.observe(0)
    assert "phase_bin" in obs.payload
    assert 0 <= int(obs.payload["phase_bin"]) < 8


def test_actuator_shifts_sensor_phase():
    sensor = SyntheticSensor()
    actuator = MockActuator(on_command=sensor.apply_motor)
    before = sensor.phase_shift
    actuator.handle(
        SensorimotorMessage(
            type="action",
            id=actuator.id,
            payload={"command": "right"},
        )
    )
    assert sensor.phase_shift == before + sensor.right_delta
    actuator.handle(
        SensorimotorMessage(
            type="action",
            id=actuator.id,
            payload={"command": "left"},
        )
    )
    assert abs(sensor.phase_shift - before) < 1e-9


def test_open_loop_actuator_records_without_shift():
    sensor = SyntheticSensor()
    actuator = MockActuator(on_command=None)
    shift_before = sensor.phase_shift
    actuator.handle(
        SensorimotorMessage(
            type="action",
            id=actuator.id,
            payload={"command": "right"},
        )
    )
    assert len(actuator.received) == 1
    assert sensor.phase_shift == shift_before
