"""Mock actuator that records action messages."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..protocol import SensorimotorMessage, register_message


@dataclass
class MockActuator:
    id: str = "mock-actuator"
    received: list[dict] = field(default_factory=list)

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
        self.received.append(message.payload)
