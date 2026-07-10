"""Serial / USB motor-actuator and tactile-sensor protocol bridges.

These modules speak the shared sensorimotor JSON protocol. They do **not**
require physical hardware: a ``MockSerialPort`` records writes and can inject
bytes for tests. Real ports use optional ``pyserial`` when installed.

Design note (embodied runtime B-6 / open issues): real motor arms and tactile
arrays reuse this same bridge; transport details stay here so the core runtime
stays hardware-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable
import json
import struct
import time

from ..protocol import (
    SensorimotorMessage,
    message_from_json,
    message_to_json,
    register_message,
)


@runtime_checkable
class SerialPortLike(Protocol):
    def write(self, data: bytes) -> int: ...

    def read(self, size: int = 1) -> bytes: ...

    def close(self) -> None: ...

    @property
    def in_waiting(self) -> int: ...


@dataclass
class MockSerialPort:
    """In-memory serial port for offline tests and CI."""

    written: list[bytes] = field(default_factory=list)
    _rx: bytearray = field(default_factory=bytearray)
    closed: bool = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise OSError("port closed")
        self.written.append(bytes(data))
        return len(data)

    def read(self, size: int = 1) -> bytes:
        if self.closed:
            raise OSError("port closed")
        chunk = bytes(self._rx[:size])
        del self._rx[:size]
        return chunk

    def inject(self, data: bytes) -> None:
        self._rx.extend(data)

    def close(self) -> None:
        self.closed = True

    @property
    def in_waiting(self) -> int:
        return len(self._rx)


def open_serial_port(
    port: str,
    *,
    baudrate: int = 115200,
    timeout: float = 0.05,
) -> SerialPortLike:
    """Open a real serial port via pyserial, or raise with install hint."""
    try:
        import serial  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install pyserial (`pip install pyserial`) for real serial ports, "
            "or pass MockSerialPort for offline use."
        ) from exc
    return serial.Serial(port, baudrate=baudrate, timeout=timeout)


def encode_line(message: SensorimotorMessage) -> bytes:
    """JSONL framing: one message per newline-terminated line."""
    return (message_to_json(message) + "\n").encode("utf-8")


def decode_lines(buffer: bytearray) -> list[SensorimotorMessage]:
    """Pop complete newline-delimited JSON messages from a buffer."""
    messages: list[SensorimotorMessage] = []
    while True:
        try:
            idx = buffer.index(b"\n")
        except ValueError:
            break
        line = bytes(buffer[:idx]).decode("utf-8", errors="replace").strip()
        del buffer[: idx + 1]
        if not line:
            continue
        messages.append(message_from_json(line))
    return messages


@dataclass
class SerialMotorBridge:
    """Actuator bridge: core ``action`` messages → serial motor commands.

    Wire format (ASCII JSONL) for each command:
      {"type":"motor","channels":[...],"ts":...}
    Binary fallback (8 floats little-endian) when ``binary=True``.
    """

    id: str = "serial-motor"
    port: SerialPortLike | None = None
    n_channels: int = 4
    binary: bool = False
    _last_command: list[float] = field(default_factory=list, init=False)

    def register(self) -> SensorimotorMessage:
        return register_message(
            module_id=self.id,
            role="actuator",
            modality="motor",
            shape=[self.n_channels],
            action_space={"type": "continuous", "dims": self.n_channels, "range": [-1.0, 1.0]},
        )

    def attach(self, port: SerialPortLike) -> None:
        self.port = port

    def on_action(self, message: SensorimotorMessage) -> list[float]:
        if message.type != "action":
            raise ValueError("SerialMotorBridge expects action messages")
        values = message.payload.get("values") or message.payload.get("command") or []
        if not isinstance(values, list):
            values = list(values)
        cmd = [float(v) for v in values[: self.n_channels]]
        while len(cmd) < self.n_channels:
            cmd.append(0.0)
        self._last_command = cmd
        if self.port is not None:
            if self.binary:
                payload = struct.pack(f"<{self.n_channels}f", *cmd)
                self.port.write(payload)
            else:
                body = json.dumps(
                    {"type": "motor", "channels": cmd, "ts": time.time(), "id": self.id},
                    separators=(",", ":"),
                )
                self.port.write((body + "\n").encode("utf-8"))
        return cmd

    @property
    def last_command(self) -> list[float]:
        return list(self._last_command)


@dataclass
class SerialTactileSensor:
    """Tactile / force sensor bridge: serial samples → ``observation`` messages.

    Expects either JSONL lines:
      {"type":"tactile","values":[...]}
    or raw little-endian float32 arrays of length ``n_taxels``.
    """

    id: str = "serial-tactile"
    port: SerialPortLike | None = None
    n_taxels: int = 8
    binary: bool = False
    _rx_buf: bytearray = field(default_factory=bytearray, init=False)
    _last_values: list[float] = field(default_factory=list, init=False)

    def register(self) -> SensorimotorMessage:
        return register_message(
            module_id=self.id,
            role="sensor",
            modality="tactile",
            shape=[self.n_taxels],
        )

    def attach(self, port: SerialPortLike) -> None:
        self.port = port

    def poll(self) -> Optional[SensorimotorMessage]:
        """Read available serial data and return an observation if complete."""
        if self.port is None:
            return None
        waiting = getattr(self.port, "in_waiting", 0) or 0
        if waiting > 0:
            self._rx_buf.extend(self.port.read(waiting))
        if self.binary:
            need = 4 * self.n_taxels
            if len(self._rx_buf) < need:
                return None
            chunk = bytes(self._rx_buf[:need])
            del self._rx_buf[:need]
            values = list(struct.unpack(f"<{self.n_taxels}f", chunk))
        else:
            # Device JSONL may be raw tactile dicts or full protocol messages.
            try:
                idx = self._rx_buf.index(b"\n")
            except ValueError:
                return None
            line = bytes(self._rx_buf[:idx]).decode("utf-8", errors="replace").strip()
            del self._rx_buf[: idx + 1]
            if not line:
                return None
            try:
                msg = message_from_json(line)
                raw = msg.payload.get("values") or msg.payload.get("data") or []
            except (ValueError, json.JSONDecodeError, KeyError):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return None
                raw = obj.get("values") or obj.get("data") or []
            values = [float(v) for v in raw[: self.n_taxels]]
            while len(values) < self.n_taxels:
                values.append(0.0)
        self._last_values = values
        return SensorimotorMessage(
            type="observation",
            id=self.id,
            payload={"values": values, "modality": "tactile"},
        )

    def inject_raw_json(self, obj: dict[str, Any]) -> None:
        """Test helper: inject a tactile JSON line as if from the device."""
        if self.port is None or not isinstance(self.port, MockSerialPort):
            raise RuntimeError("inject_raw_json requires MockSerialPort")
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self.port.inject(line.encode("utf-8"))

    def inject_floats(self, values: list[float]) -> None:
        if self.port is None or not isinstance(self.port, MockSerialPort):
            raise RuntimeError("inject_floats requires MockSerialPort")
        if self.binary:
            self.port.inject(struct.pack(f"<{self.n_taxels}f", *values[: self.n_taxels]))
        else:
            self.inject_raw_json({"type": "tactile", "values": values[: self.n_taxels]})

    @property
    def last_values(self) -> list[float]:
        return list(self._last_values)


def parse_device_jsonl(data: bytes) -> list[dict[str, Any]]:
    """Parse raw device JSONL that may not be full SensorimotorMessage."""
    out: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out
