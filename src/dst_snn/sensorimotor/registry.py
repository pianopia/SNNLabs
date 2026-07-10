"""Dynamic module registry with fixed-dimensional sparse mappings."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from .protocol import SensorimotorMessage


@dataclass(frozen=True)
class ModuleSpec:
    id: str
    role: str
    modality: str
    shape: tuple[int, ...] = ()
    action_space: dict[str, Any] = field(default_factory=dict)


class ModuleRegistry:
    def __init__(self, feature_size: int = 512, motor_size: int = 64) -> None:
        if feature_size <= 0 or motor_size <= 0:
            raise ValueError("feature_size and motor_size must be positive")
        self.feature_size = feature_size
        self.motor_size = motor_size
        self.modules: dict[str, ModuleSpec] = {}
        self.latest_observations: dict[str, dict[str, Any]] = {}

    def register(self, spec: ModuleSpec) -> None:
        self.modules[spec.id] = spec

    def deregister(self, module_id: str) -> None:
        self.modules.pop(module_id, None)
        self.latest_observations.pop(module_id, None)

    def apply(self, message: SensorimotorMessage) -> None:
        if message.type == "register":
            payload = message.payload
            self.register(
                ModuleSpec(
                    id=message.id,
                    role=str(payload["role"]),
                    modality=str(payload["modality"]),
                    shape=tuple(int(v) for v in payload.get("shape", ())),
                    action_space=dict(payload.get("action_space", {})),
                )
            )
        elif message.type == "deregister":
            self.deregister(message.id)
        elif message.type == "observation":
            if message.id in self.modules:
                self.latest_observations[message.id] = message.payload

    def feature_index(self, module_id: str, key: str) -> int:
        return stable_hash(f"{module_id}:{key}") % self.feature_size

    def motor_index(self, module_id: str, key: str) -> int:
        return stable_hash(f"{module_id}:{key}") % self.motor_size

    def actuator_specs(self) -> list[ModuleSpec]:
        return [
            spec
            for spec in self.modules.values()
            if spec.role in {"actuator", "both"} and spec.action_space
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_size": self.feature_size,
            "motor_size": self.motor_size,
            "modules": [
                {
                    "id": spec.id,
                    "role": spec.role,
                    "modality": spec.modality,
                    "shape": list(spec.shape),
                    "action_space": spec.action_space,
                }
                for spec in self.modules.values()
            ],
            "latest_observations": self.latest_observations,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ModuleRegistry":
        registry = cls(
            feature_size=int(state.get("feature_size", 512)),
            motor_size=int(state.get("motor_size", 64)),
        )
        for spec in state.get("modules", []):
            registry.register(
                ModuleSpec(
                    id=str(spec["id"]),
                    role=str(spec["role"]),
                    modality=str(spec["modality"]),
                    shape=tuple(int(v) for v in spec.get("shape", ())),
                    action_space=dict(spec.get("action_space", {})),
                )
            )
        registry.latest_observations = {
            str(module_id): dict(payload)
            for module_id, payload in state.get("latest_observations", {}).items()
            if module_id in registry.modules
        }
        return registry


def stable_hash(value: str) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16)
