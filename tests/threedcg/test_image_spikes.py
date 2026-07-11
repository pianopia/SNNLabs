from __future__ import annotations

import numpy as np

from src.dst_snn.threedcg.image_spikes import (
    image_to_spikes,
    load_image_array,
    spike_feature_size,
)


def test_load_array_and_feature_size():
    img = np.zeros((4, 4, 3), dtype=np.float64)
    img[1:3, 1:3] = 1.0
    loaded = load_image_array(img)
    assert loaded.shape == (4, 4, 3)
    assert spike_feature_size(4, 4, include_edges=True) == 32
    assert spike_feature_size(4, 4, include_edges=False) == 16


def test_image_to_spikes_shape_and_deterministic():
    img = np.linspace(0, 1, 8 * 8 * 3).reshape(8, 8, 3)
    a = image_to_spikes(img, time_bins=4, seed=0, max_side=8)
    b = image_to_spikes(img, time_bins=4, seed=0, max_side=8)
    c = image_to_spikes(img, time_bins=4, seed=1, max_side=8)
    assert a.shape == (4, spike_feature_size(8, 8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(a)).issubset({0.0, 1.0})
