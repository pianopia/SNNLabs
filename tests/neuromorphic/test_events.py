from __future__ import annotations

import numpy as np

from benchmarks.neuromorphic.events import (
    bin_events_to_frames,
    frames_to_spike_tensor,
)


def test_bins_events_by_time_and_polarity():
    x = np.array([0, 1])
    y = np.array([0, 1])
    t = np.array([0, 9])
    p = np.array([1, 0])
    frames = bin_events_to_frames(x, y, t, p, width=2, height=2, time_bins=10)
    assert frames.shape == (10, 2, 2, 2)
    assert frames[0, 1, 0, 0] == 1.0
    assert frames[9, 0, 1, 1] == 1.0
    assert frames.sum() == 2.0


def test_frames_to_spike_tensor_is_binary_and_flat():
    frames = np.zeros((3, 2, 2, 2), dtype=np.float32)
    frames[0, 0, 0, 0] = 5.0
    spikes = frames_to_spike_tensor(frames, threshold=1.0)
    assert spikes.shape == (3, 8)
    assert spikes[0, 0] == 1.0
    assert spikes.sum() == 1.0
