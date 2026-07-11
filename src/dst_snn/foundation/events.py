"""Modality-aware signed event coding and temporal compression.

The representation deliberately keeps polarity and small integer magnitude.
Binary rate coding loses semantic sign and often needs many timesteps; signed
multi-level events provide a GPU-friendly reference representation that can
also be lowered to repeated binary events on neuromorphic hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    time_steps: int
    threshold: float = 0.125
    max_event_level: int = 3

    def __post_init__(self) -> None:
        if self.time_steps < 1:
            raise ValueError("time_steps must be positive")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 1 <= self.max_event_level <= 127:
            raise ValueError("max_event_level must be in [1, 127]")


# Text retains more temporal precision; high-dimensional visual/audio streams
# are compressed more aggressively. These are runtime defaults, not claims of
# universally optimal scales.
DEFAULT_MODALITY_SPECS: dict[str, ModalitySpec] = {
    "text": ModalitySpec("text", time_steps=4, threshold=0.125, max_event_level=3),
    "image": ModalitySpec("image", time_steps=3, threshold=0.16, max_event_level=3),
    "audio": ModalitySpec("audio", time_steps=3, threshold=0.12, max_event_level=3),
    "video": ModalitySpec("video", time_steps=2, threshold=0.18, max_event_level=3),
    "sensor": ModalitySpec("sensor", time_steps=2, threshold=0.10, max_event_level=3),
    "action": ModalitySpec("action", time_steps=2, threshold=0.10, max_event_level=3),
}


def compressed_time_steps(levels: int) -> int:
    """Return logarithmic temporal depth for ``levels`` quantization levels."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    return max(1, int(math.ceil(math.log2(levels))))


class SignedEventEncoder:
    """Streaming sigma-delta encoder with residual state.

    Consecutive calls preserve residual error, so unchanged inputs naturally
    become silent while small changes accumulate until they are informative.
    """

    def __init__(self, feature_size: int, *, threshold: float, max_event_level: int = 3):
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 1 <= max_event_level <= 127:
            raise ValueError("max_event_level must be in [1, 127]")
        self.feature_size = int(feature_size)
        self.threshold = float(threshold)
        self.max_event_level = int(max_event_level)
        self.reconstruction = np.zeros(self.feature_size, dtype=np.float32)

    def reset(self) -> None:
        self.reconstruction.fill(0.0)

    def encode(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        if source.ndim == 1:
            source = source[None, :]
        if source.ndim != 2 or source.shape[1] != self.feature_size:
            raise ValueError(
                f"expected [steps, {self.feature_size}], got {source.shape}"
            )
        events = np.zeros(source.shape, dtype=np.int8)
        for index, row in enumerate(source):
            delta = row - self.reconstruction
            level = np.trunc(delta / self.threshold)
            level = np.clip(level, -self.max_event_level, self.max_event_level).astype(np.int8)
            events[index] = level
            self.reconstruction += level.astype(np.float32) * self.threshold
        return events


def temporal_compress(events: np.ndarray, target_steps: int, *, max_event_level: int = 7) -> np.ndarray:
    """Sum contiguous event windows into fewer signed integer timesteps."""
    source = np.asarray(events)
    if source.ndim != 2:
        raise ValueError("events must have shape [steps, features]")
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    if source.shape[0] == 0:
        return np.zeros((target_steps, source.shape[1]), dtype=np.int8)
    boundaries = np.linspace(0, source.shape[0], target_steps + 1).astype(int)
    out = np.zeros((target_steps, source.shape[1]), dtype=np.int16)
    for step in range(target_steps):
        start, end = boundaries[step], boundaries[step + 1]
        if end > start:
            out[step] = source[start:end].astype(np.int16).sum(axis=0)
    return np.clip(out, -max_event_level, max_event_level).astype(np.int8)


def align_and_fuse_modalities(
    streams: Mapping[str, np.ndarray],
    *,
    specs: Mapping[str, ModalitySpec] | None = None,
) -> tuple[np.ndarray, dict[str, slice]]:
    """Compress each modality at its own scale, then align and concatenate.

    Shorter streams are left-aligned and zero-padded. This preserves immediate
    first-response events without repeating them merely to match another
    modality's clock.
    """
    if not streams:
        raise ValueError("at least one modality stream is required")
    selected = specs or DEFAULT_MODALITY_SPECS
    compressed: list[tuple[str, np.ndarray]] = []
    for name, stream in streams.items():
        source = np.asarray(stream)
        if source.ndim != 2:
            raise ValueError(f"{name} stream must have shape [steps, features]")
        spec = selected.get(name, ModalitySpec(name, time_steps=2))
        compressed.append(
            (name, temporal_compress(source, min(spec.time_steps, max(1, source.shape[0]))))
        )
    total_steps = max(events.shape[0] for _, events in compressed)
    total_features = sum(events.shape[1] for _, events in compressed)
    fused = np.zeros((total_steps, total_features), dtype=np.int8)
    slices: dict[str, slice] = {}
    offset = 0
    for name, events in compressed:
        width = events.shape[1]
        fused[: events.shape[0], offset : offset + width] = events
        slices[name] = slice(offset, offset + width)
        offset += width
    return fused, slices
