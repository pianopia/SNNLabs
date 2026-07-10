"""Async WebSocket transport for sensorimotor messages.

Uses the optional ``websockets`` package when installed. Without it, the
in-process ``LocalMessageHub`` still supports multi-consumer fan-out for tests
and single-process demos. The wire format is one JSON message per WS frame
(same schema as JSONL transport).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, DefaultDict, Optional

from .protocol import SensorimotorMessage, message_from_json, message_to_json
from .runtime import SensorimotorRuntime

MessageHandler = Callable[[SensorimotorMessage], Optional[Awaitable[None]]]


@dataclass
class LocalMessageHub:
    """In-process pub/sub hub that mirrors WebSocket fan-out semantics."""

    _subscribers: DefaultDict[str, list[asyncio.Queue[SensorimotorMessage]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def subscribe(self, topic: str = "all") -> asyncio.Queue[SensorimotorMessage]:
        queue: asyncio.Queue[SensorimotorMessage] = asyncio.Queue()
        self._subscribers[topic].append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SensorimotorMessage], topic: str = "all") -> None:
        if queue in self._subscribers.get(topic, []):
            self._subscribers[topic].remove(queue)

    async def publish(self, message: SensorimotorMessage, topic: str = "all") -> None:
        for queue in list(self._subscribers.get(topic, [])):
            await queue.put(message)
        if topic != "all":
            for queue in list(self._subscribers.get("all", [])):
                await queue.put(message)


async def handle_module_message(runtime: SensorimotorRuntime, message: SensorimotorMessage) -> list[SensorimotorMessage]:
    """Ingest one inbound module message and optionally advance a tick."""
    runtime.ingest(message)
    outbound: list[SensorimotorMessage] = []
    if message.type == "observation":
        result = runtime.tick()
        outbound.extend(result.get("messages", []))
    return outbound


def websockets_available() -> bool:
    try:
        import websockets  # noqa: F401
    except Exception:
        return False
    return True


async def serve_runtime(
    runtime: SensorimotorRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    hub: Optional[LocalMessageHub] = None,
):
    """Serve a WebSocket endpoint that bridges modules to an SNN runtime.

    Requires ``websockets``. Each client sends JSON sensorimotor messages;
    observation frames trigger a runtime tick and broadcast action /
    global_signal / trace messages to all connected clients.
    """
    if not websockets_available():
        raise ImportError("Install websockets (`pip install websockets`) for WebSocket transport.")

    import websockets
    from websockets.server import WebSocketServerProtocol

    hub = hub or LocalMessageHub()
    clients: set[WebSocketServerProtocol] = set()

    async def broadcast(message: SensorimotorMessage) -> None:
        await hub.publish(message)
        dead: list[WebSocketServerProtocol] = []
        payload = message_to_json(message)
        for client in list(clients):
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            clients.discard(client)

    async def handler(websocket: WebSocketServerProtocol) -> None:
        clients.add(websocket)
        try:
            async for raw in websocket:
                try:
                    message = message_from_json(str(raw))
                except Exception:
                    continue
                outbound = await handle_module_message(runtime, message)
                for out in outbound:
                    await broadcast(out)
        finally:
            clients.discard(websocket)

    return await websockets.serve(handler, host, port)
