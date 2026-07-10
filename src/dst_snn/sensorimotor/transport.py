"""JSONL transport helpers for sensorimotor messages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from .protocol import SensorimotorMessage, message_from_json, message_to_json
from .runtime import SensorimotorRuntime


def write_jsonl(messages: Iterable[SensorimotorMessage], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(message_to_json(message) + "\n")


def read_jsonl(path: Path) -> Iterator[SensorimotorMessage]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield message_from_json(line)


def replay_jsonl(runtime: SensorimotorRuntime, path: Path) -> list[dict]:
    """Replay register/deregister/observation messages into a runtime.

    A runtime tick is emitted after each observation so JSONL logs can be used as
    deterministic smoke tests for module streams.
    """
    results: list[dict] = []
    for message in read_jsonl(path):
        runtime.ingest(message)
        if message.type == "observation":
            results.append(runtime.tick())
    return results
