"""Small in-process sensorimotor runtime loop."""

from __future__ import annotations

import json
from pathlib import Path
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
        self.step = 0

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
        spike_rate = float(spikes.mean())
        # Fatigue rises with sustained firing and slowly recovers otherwise.
        prev_fatigue = float(self.global_signal.get("fatigue", 0.0))
        fatigue = min(1.0, max(0.0, prev_fatigue * 0.97 + spike_rate * 0.15 - 0.01))
        arousal = min(1.0, spike_rate * 4.0 * (1.0 - 0.4 * fatigue))
        self.global_signal = {
            "arousal": arousal,
            "reward": novelty * (1.0 - 0.5 * fatigue),
            "novelty": novelty,
            "fatigue": fatigue,
        }
        self.step += 1
        global_signal = SensorimotorMessage(
            type="global_signal",
            id="core",
            payload=self.global_signal,
        )
        trace = SensorimotorMessage(
            type="trace",
            id="core",
            payload={
                "step": self.step,
                "spike_count": float(spikes.sum()),
                "active_fraction": float((spikes > 0).mean()),
                "actions": [message.payload for message in actions],
            },
        )
        return {
            "at": time.time(),
            "spikes": spikes,
            "actions": actions,
            "global_signal": self.global_signal,
            "messages": [*actions, global_signal, trace],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "time_steps": self.time_steps,
            "step": self.step,
            "registry": self.registry.state_dict(),
            "global_signal": self.global_signal,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SensorimotorRuntime":
        runtime = cls(
            ModuleRegistry.from_state_dict(state["registry"]),
            time_steps=int(state.get("time_steps", 8)),
        )
        runtime.step = int(state.get("step", 0))
        runtime.global_signal = dict(state.get("global_signal", runtime.global_signal))
        return runtime

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SensorimotorRuntime":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
