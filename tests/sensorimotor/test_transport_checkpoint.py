from __future__ import annotations

from pathlib import Path

from src.dst_snn.sensorimotor.modules import SyntheticSensor
from src.dst_snn.sensorimotor.registry import ModuleRegistry
from src.dst_snn.sensorimotor.runtime import SensorimotorRuntime
from src.dst_snn.sensorimotor.transport import read_jsonl, replay_jsonl, write_jsonl


def test_jsonl_roundtrip_and_replay(tmp_path: Path):
    sensor = SyntheticSensor()
    messages = [sensor.register(), sensor.observe(0), sensor.observe(1)]
    path = tmp_path / "stream.jsonl"
    write_jsonl(messages, path)

    restored = list(read_jsonl(path))
    assert [message.type for message in restored] == ["register", "observation", "observation"]

    runtime = SensorimotorRuntime(ModuleRegistry(feature_size=16, motor_size=8), time_steps=4)
    results = replay_jsonl(runtime, path)
    assert len(results) == 2
    assert results[-1]["messages"][-1].type == "trace"
    assert runtime.step == 2


def test_runtime_save_load(tmp_path: Path):
    sensor = SyntheticSensor()
    runtime = SensorimotorRuntime(ModuleRegistry(feature_size=16, motor_size=8), time_steps=4)
    runtime.ingest(sensor.register())
    runtime.ingest(sensor.observe(0))
    runtime.tick()

    path = tmp_path / "runtime.json"
    runtime.save(path)
    restored = SensorimotorRuntime.load(path)
    assert restored.step == 1
    assert restored.registry.feature_size == 16
    assert "synthetic-sensor" in restored.registry.modules
    assert restored.registry.latest_observations["synthetic-sensor"]["step"] == 0
