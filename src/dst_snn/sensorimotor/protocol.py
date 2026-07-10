"""Language-neutral JSON message protocol for sensorimotor modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Literal

MessageType = Literal[
    "register",
    "deregister",
    "observation",
    "action",
    "global_signal",
    "trace",
]


@dataclass(frozen=True)
class SensorimotorMessage:
    type: MessageType
    id: str
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


def message_to_json(message: SensorimotorMessage) -> str:
    return json.dumps(asdict(message), ensure_ascii=False, separators=(",", ":"))


def message_from_json(text: str) -> SensorimotorMessage:
    data = json.loads(text)
    msg_type = data.get("type")
    if msg_type not in {
        "register",
        "deregister",
        "observation",
        "action",
        "global_signal",
        "trace",
    }:
        raise ValueError(f"unknown message type: {msg_type!r}")
    if not data.get("id"):
        raise ValueError("message id is required")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return SensorimotorMessage(
        type=msg_type,
        id=str(data["id"]),
        ts=float(data.get("ts", time.time())),
        payload=payload,
    )


def register_message(
    *,
    module_id: str,
    role: str,
    modality: str,
    shape: list[int],
    action_space: dict[str, Any] | None = None,
) -> SensorimotorMessage:
    return SensorimotorMessage(
        type="register",
        id=module_id,
        payload={
            "role": role,
            "modality": modality,
            "shape": shape,
            "action_space": action_space or {},
        },
    )
