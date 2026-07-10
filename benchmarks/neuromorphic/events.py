"""Convert event-camera event streams into DST-SNN spike tensors."""

from __future__ import annotations

import numpy as np


def bin_events_to_frames(
    x,
    y,
    t,
    p,
    *,
    width: int,
    height: int,
    time_bins: int,
    t_start=None,
    t_end=None,
) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    t = np.asarray(t).astype(np.float64)
    p = np.asarray(p)
    if time_bins <= 0 or width <= 0 or height <= 0:
        raise ValueError("time_bins, width, and height must be positive")

    frames = np.zeros((time_bins, 2, height, width), dtype=np.float32)
    if t.size == 0:
        return frames

    start = float(t.min()) if t_start is None else float(t_start)
    end = float(t.max()) if t_end is None else float(t_end)
    span = max(1e-9, end - start)
    bin_idx = ((t - start) / span * time_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, time_bins - 1)
    pol = (p > 0).astype(np.int64)

    xi = np.clip(x, 0, width - 1)
    yi = np.clip(y, 0, height - 1)
    np.add.at(frames, (bin_idx, pol, yi, xi), 1.0)
    return frames


def frames_to_spike_tensor(frames: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError("frames must have shape [time, 2, height, width]")
    time_bins = frames.shape[0]
    flat = frames.reshape(time_bins, -1)
    return (flat >= threshold).astype(np.float32)
