"""Small in-process sensorimotor runtime loop."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .codec import decode_action, encode_observations
from .protocol import SensorimotorMessage
from .registry import ModuleRegistry


class SensorimotorRuntime:
    def __init__(self, registry: ModuleRegistry, *, time_steps: int = 8) -> None:
        self.registry = registry
        self.time_steps = time_steps
        self.last_action_messages: list[SensorimotorMessage] = []
        self.global_signal = {"arousal": 0.0, "reward": 0.0, "novelty": 0.0, "fatigue": 0.0}

    def ingest(self, message: SensorimotorMessage) -> None:
        self.registry.apply(message)

    def tick(self, motor_activity: np.ndarray | None = None) -> dict[str, Any]:
        spikes = encode_observations(self.registry, time_steps=self.time_steps)
        if motor_activity is None:
            motor_activity = spikes.mean(axis=0)[: self.registry.motor_size]
            if motor_activity.shape[0] < self.registry.motor_size:
                motor_activity = np.pad(motor_activity, (0, self.registry.motor_size - motor_activity.shape[0]))
        actions = decode_action(self.registry, motor_activity)
        self.last_action_messages = actions
        novelty = float((spikes > 0).mean())
        self.global_signal = {
            "arousal": min(1.0, float(spikes.mean() * 4.0)),
            "reward": novelty,
            "novelty": novelty,
            "fatigue": 0.0,
        }
        return {
            "at": time.time(),
            "spikes": spikes,
            "actions": actions,
            "global_signal": self.global_signal,
        }
