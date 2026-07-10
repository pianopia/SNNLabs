"""Embodied sensorimotor runtime primitives."""

from .codec import decode_action, encode_observations
from .protocol import SensorimotorMessage, message_from_json, message_to_json
from .registry import ModuleRegistry, ModuleSpec

__all__ = [
    "ModuleRegistry",
    "ModuleSpec",
    "SensorimotorMessage",
    "decode_action",
    "encode_observations",
    "message_from_json",
    "message_to_json",
]
