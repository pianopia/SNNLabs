"""Intrinsic-reward motor policy for actuator command exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np

from .registry import ModuleRegistry, ModuleSpec


@dataclass
class IntrinsicMotorPolicy:
    """Choose actuator commands with scores updated by intrinsic reward."""

    epsilon: float = 0.1
    temperature: float = 0.5
    learning_rate: float = 0.2
    seed: int = 0
    command_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def activity(self, registry: ModuleRegistry) -> tuple[np.ndarray, list[dict[str, str]]]:
        motor = np.zeros(registry.motor_size, dtype=np.float32)
        selected: list[dict[str, str]] = []
        for spec in registry.actuator_specs():
            command = self.choose_command(spec)
            if command is None:
                continue
            motor[registry.motor_index(spec.id, command) % registry.motor_size] = 1.0
            selected.append({"module_id": spec.id, "command": command})
        return motor, selected

    def choose_command(self, spec: ModuleSpec) -> str | None:
        commands = spec.action_space.get("commands")
        if not isinstance(commands, list) or not commands:
            return None
        commands = [str(command) for command in commands]
        if self._rng.random() < self.epsilon:
            return self._rng.choice(commands)
        weights = []
        for command in commands:
            score = self.command_scores.get(self._key(spec.id, command), 0.0)
            weights.append(math.exp(score / max(1e-6, self.temperature)))
        total = sum(weights)
        threshold = self._rng.random() * total
        cumulative = 0.0
        for command, weight in zip(commands, weights):
            cumulative += weight
            if cumulative >= threshold:
                return command
        return commands[-1]

    def update(self, selected: list[dict[str, str]], intrinsic_reward: float) -> None:
        reward = max(0.0, min(1.0, float(intrinsic_reward)))
        for item in selected:
            key = self._key(item["module_id"], item["command"])
            previous = self.command_scores.get(key, 0.0)
            self.command_scores[key] = previous + self.learning_rate * (reward - previous)

    def state_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "command_scores": self.command_scores,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "IntrinsicMotorPolicy":
        return cls(
            epsilon=float(state.get("epsilon", 0.1)),
            temperature=float(state.get("temperature", 0.5)),
            learning_rate=float(state.get("learning_rate", 0.2)),
            seed=int(state.get("seed", 0)),
            command_scores={str(k): float(v) for k, v in state.get("command_scores", {}).items()},
        )

    @staticmethod
    def _key(module_id: str, command: str) -> str:
        return f"{module_id}:{command}"
