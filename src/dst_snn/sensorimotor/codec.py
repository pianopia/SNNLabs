"""Encode observations to spikes and decode motor activity to actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .protocol import SensorimotorMessage
from .registry import ModuleRegistry, ModuleSpec


def _flatten_values(prefix: str, value: Any) -> list[tuple[str, float]]:
    if isinstance(value, bool):
        return [(prefix, 1.0 if value else 0.0)]
    if isinstance(value, (int, float)):
        return [(prefix, float(value))]
    if isinstance(value, str):
        return [(f"{prefix}:{value}", 1.0)]
    if isinstance(value, Mapping):
        out: list[tuple[str, float]] = []
        for key, nested in sorted(value.items()):
            out.extend(_flatten_values(f"{prefix}.{key}" if prefix else str(key), nested))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out = []
        for index, nested in enumerate(value):
            out.extend(_flatten_values(f"{prefix}[{index}]", nested))
        return out
    return [(f"{prefix}:{type(value).__name__}", 1.0)]


def encode_observations(
    registry: ModuleRegistry,
    observations: Mapping[str, dict[str, Any]] | None = None,
    *,
    time_steps: int = 8,
) -> np.ndarray:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    source = observations if observations is not None else registry.latest_observations
    spikes = np.zeros((time_steps, registry.feature_size), dtype=np.float32)
    for module_offset, (module_id, payload) in enumerate(sorted(source.items())):
        if module_id not in registry.modules:
            continue
        for key, raw_value in _flatten_values("", payload):
            value = max(0.0, min(1.0, abs(float(raw_value))))
            index = registry.feature_index(module_id, key)
            step = (module_offset + registry.feature_index(module_id, f"time:{key}")) % time_steps
            spikes[step, index] = max(spikes[step, index], value)
    return spikes


def decode_action(
    registry: ModuleRegistry,
    motor_activity: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[SensorimotorMessage]:
    activity = np.asarray(motor_activity, dtype=np.float32).reshape(-1)
    messages: list[SensorimotorMessage] = []
    for spec in registry.actuator_specs():
        action = _decode_for_spec(registry, spec, activity, threshold)
        if action:
            messages.append(SensorimotorMessage(type="action", id=spec.id, payload=action))
    return messages


def _decode_for_spec(
    registry: ModuleRegistry,
    spec: ModuleSpec,
    activity: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    commands = spec.action_space.get("commands")
    if isinstance(commands, list) and commands:
        scored = []
        for command in commands:
            index = registry.motor_index(spec.id, str(command))
            scored.append((float(activity[index % len(activity)]), str(command)))
        score, command = max(scored, key=lambda item: item[0])
        if score >= threshold:
            return {"command": command, "confidence": score}
    axes = spec.action_space.get("axes")
    if isinstance(axes, list) and axes:
        values = {}
        for axis in axes:
            pos = activity[registry.motor_index(spec.id, f"{axis}:pos") % len(activity)]
            neg = activity[registry.motor_index(spec.id, f"{axis}:neg") % len(activity)]
            value = float(pos - neg)
            if abs(value) >= threshold:
                values[str(axis)] = value
        if values:
            return {"axes": values}
    return {}
