from __future__ import annotations

import numpy as np

from src.dst_snn.threedcg.track2_occupancy import (
    occupancy_to_mesh,
    spikes_to_occupancy,
    track2_from_spikes,
)


def test_spikes_to_occupancy_shape():
    spikes = np.random.default_rng(0).random((4, 64))
    grid = spikes_to_occupancy(spikes, resolution=4, threshold=0.2)
    assert grid.shape == (4, 4, 4)
    assert grid.sum() > 0


def test_occupancy_to_mesh_and_asset():
    grid = np.zeros((3, 3, 3))
    grid[1, 1, 1] = 1.0
    mesh = occupancy_to_mesh(grid, origin=[-0.5, -0.5, -0.5], extents=[1, 1, 1])
    assert len(mesh.vertices) > 0
    spikes = np.ones((2, 27), dtype=np.float32)
    asset = track2_from_spikes(spikes, resolution=3)
    assert np.asarray(asset.vertices).shape[0] > 0
