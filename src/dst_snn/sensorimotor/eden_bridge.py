"""EDEN (TypeScript body SNN) ↔ Python sensorimotor protocol bridge.

Maps EDEN ``SnnTraceEvent``-shaped payloads onto the language-neutral
sensorimotor protocol so the PyTorch core can consume browser/EDEN modules
over WebSocket (or in-process for tests).

EDEN event kinds observed in ``EDEN/src/snn/modules.ts`` / ``lif``:
  - ``body`` — body state / overload / gait (→ observation)
  - ``global_signal`` — reward / arousal-like scalar (→ global_signal)
  - ``spike`` — optional spike mass (→ trace)

Outbound Python ``action`` / ``global_signal`` / ``trace`` messages can be
translated back to EDEN-friendly dicts for the dashboard client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable
import time

from .protocol import (
    SensorimotorMessage,
    message_from_json,
    message_to_json,
    register_message,
)


def _num(meta: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not meta:
        return default
    value = meta.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class EdenTraceEvent:
    """Minimal EDEN trace event (mirrors TS ``SnnTraceEvent`` loosely)."""

    kind: str
    value: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = "eden"


def eden_event_from_dict(data: dict[str, Any]) -> EdenTraceEvent:
    return EdenTraceEvent(
        kind=str(data.get("kind") or data.get("type") or "body"),
        value=float(data.get("value") or 0.0),
        meta=dict(data.get("meta") or {}),
        ts=float(data.get("ts") or time.time()),
        id=str(data.get("id") or "eden"),
    )


def eden_events_to_messages(
    events: Iterable[EdenTraceEvent | dict[str, Any]],
    *,
    module_id: str = "eden-body",
) -> list[SensorimotorMessage]:
    """Convert a batch of EDEN trace events into protocol messages."""
    out: list[SensorimotorMessage] = []
    for raw in events:
        event = raw if isinstance(raw, EdenTraceEvent) else eden_event_from_dict(raw)
        if event.kind == "body":
            values = [
                _num(event.meta, "overload"),
                _num(event.meta, "ambient_stimulus"),
                _num(event.meta, "nearest_stimulus"),
                _num(event.meta, "deformation"),
                _num(event.meta, "gait_drive"),
                float(event.value),
            ]
            out.append(
                SensorimotorMessage(
                    type="observation",
                    id=module_id,
                    ts=event.ts,
                    payload={
                        "values": values,
                        "modality": "eden_body",
                        "source": "eden",
                        "meta": event.meta,
                    },
                )
            )
        elif event.kind == "global_signal":
            out.append(
                SensorimotorMessage(
                    type="global_signal",
                    id=module_id,
                    ts=event.ts,
                    payload={
                        "reward": float(event.value),
                        "arousal": _num(event.meta, "arousal", float(event.value)),
                        "novelty": _num(event.meta, "novelty"),
                        "fatigue": _num(event.meta, "fatigue"),
                        "source": "eden",
                    },
                )
            )
        elif event.kind in {"spike", "trace"}:
            out.append(
                SensorimotorMessage(
                    type="trace",
                    id=module_id,
                    ts=event.ts,
                    payload={
                        "kind": event.kind,
                        "value": float(event.value),
                        "meta": event.meta,
                        "source": "eden",
                    },
                )
            )
        else:
            out.append(
                SensorimotorMessage(
                    type="trace",
                    id=module_id,
                    ts=event.ts,
                    payload={"kind": event.kind, "value": float(event.value), "meta": event.meta},
                )
            )
    return out


def message_to_eden_event(message: SensorimotorMessage) -> dict[str, Any]:
    """Map a protocol message back to an EDEN-friendly event dict."""
    if message.type == "action":
        values = message.payload.get("values") or message.payload.get("command") or []
        return {
            "kind": "action",
            "id": message.id,
            "ts": message.ts,
            "value": float(values[0]) if values else 0.0,
            "meta": {"values": list(values), "source": "python-core"},
        }
    if message.type == "global_signal":
        reward = float(message.payload.get("reward") or message.payload.get("intrinsic_reward") or 0.0)
        return {
            "kind": "global_signal",
            "id": message.id,
            "ts": message.ts,
            "value": reward,
            "meta": {k: v for k, v in message.payload.items() if k != "reward"},
        }
    if message.type == "trace":
        return {
            "kind": str(message.payload.get("kind") or "trace"),
            "id": message.id,
            "ts": message.ts,
            "value": float(message.payload.get("value") or 0.0),
            "meta": dict(message.payload.get("meta") or message.payload),
        }
    if message.type == "observation":
        values = message.payload.get("values") or []
        return {
            "kind": "body",
            "id": message.id,
            "ts": message.ts,
            "value": float(values[0]) if values else 0.0,
            "meta": {
                "values": list(values),
                "modality": message.payload.get("modality"),
            },
        }
    return {
        "kind": message.type,
        "id": message.id,
        "ts": message.ts,
        "value": 0.0,
        "meta": dict(message.payload),
    }


def eden_register_message(*, module_id: str = "eden-body") -> SensorimotorMessage:
    return register_message(
        module_id=module_id,
        role="both",
        modality="eden_body",
        shape=[6],
        action_space={"type": "continuous", "dims": 4, "range": [-1.0, 1.0]},
    )


@dataclass
class EdenBridgeSession:
    """Stateful helper: register EDEN, ingest events, emit protocol JSON."""

    module_id: str = "eden-body"
    registered: bool = False

    def register(self) -> SensorimotorMessage:
        self.registered = True
        return eden_register_message(module_id=self.module_id)

    def ingest_events(self, events: Iterable[EdenTraceEvent | dict[str, Any]]) -> list[SensorimotorMessage]:
        return eden_events_to_messages(events, module_id=self.module_id)

    def encode_outbound(self, messages: Iterable[SensorimotorMessage]) -> list[str]:
        return [message_to_json(m) for m in messages]

    def decode_inbound_json(self, text: str) -> list[SensorimotorMessage]:
        """Accept either a protocol message or an EDEN event / event batch."""
        data = json.loads(text)
        if isinstance(data, list):
            return eden_events_to_messages(data, module_id=self.module_id)
        if isinstance(data, dict) and data.get("type") in {
            "register",
            "deregister",
            "observation",
            "action",
            "global_signal",
            "trace",
        }:
            return [message_from_json(text)]
        if isinstance(data, dict) and ("kind" in data or "meta" in data):
            return eden_events_to_messages([data], module_id=self.module_id)
        raise ValueError(f"unrecognized EDEN bridge payload: {type(data)}")
