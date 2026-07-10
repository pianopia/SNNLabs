"""Mock actuator that records action messages and can drive a synthetic sensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..protocol import SensorimotorMessage, register_message

# Optional effect sink: command string → side effect (e.g. sensor phase shift).
MotorEffect = Callable[[str], None]


@dataclass
class MockActuator:
    id: str = "mock-actuator"
    received: list[dict] = field(default_factory=list)
    on_command: Optional[MotorEffect] = None

    def register(self) -> SensorimotorMessage:
        return register_message(
            module_id=self.id,
            role="actuator",
            modality="motor",
            shape=[1],
            action_space={"commands": ["left", "right", "stop"], "axes": ["x"]},
        )

    def handle(self, message: SensorimotorMessage) -> None:
        if message.type != "action" or message.id != self.id:
            return
        payload = dict(message.payload) if isinstance(message.payload, dict) else {}
        self.received.append(payload)
        command = payload.get("command")
        if command is not None and self.on_command is not None:
            self.on_command(str(command))

    def apply_selected(self, selected: list[dict[str, Any]]) -> None:
        """Apply policy-selected commands without going through decode messages."""
        for item in selected:
            if item.get("module_id") != self.id:
                continue
            command = item.get("command")
            if command is None:
                continue
            payload = {"command": str(command)}
            self.received.append(payload)
            if self.on_command is not None:
                self.on_command(str(command))
