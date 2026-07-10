"""Deterministic synthetic sensor for headless tests."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..protocol import SensorimotorMessage, register_message


@dataclass
class SyntheticSensor:
    id: str = "synthetic-sensor"
    phase: float = 0.0

    def register(self) -> SensorimotorMessage:
        return register_message(module_id=self.id, role="sensor", modality="synthetic", shape=[2])

    def observe(self, step: int) -> SensorimotorMessage:
        value = 0.5 + 0.5 * math.sin(self.phase + step * 0.25)
        return SensorimotorMessage(
            type="observation",
            id=self.id,
            payload={"signal": value, "step": step % 16},
        )
