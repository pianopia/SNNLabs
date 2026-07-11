"""Track 2: spikes → coarse occupancy field → mesh (scaffold).

Full SDF regression is future work; this first increment maps temporal mean
spike rates onto a low-res occupancy grid and emits a box-soup mesh compatible
with the existing scorer.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh


def spikes_to_occupancy(
    spikes: np.ndarray,
    *,
    resolution: int = 8,
    threshold: float = 0.15,
) -> np.ndarray:
    """Map spike tensor to a ``[R,R,R]`` occupancy grid in [0, 1].

    Mean activity over time is reshaped / interpolated into a cubic volume.
    """
    res = max(2, int(resolution))
    arr = np.asarray(spikes, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("spikes must be [time, features]")
    mean = arr.mean(axis=0)
    n = mean.size
    # Fold 1D features into a cube via sorted histogram-like fill.
    target = res * res * res
    if n >= target:
        vol = mean[:target].reshape(res, res, res)
    else:
        # Tile / pad
        reps = int(np.ceil(target / max(n, 1)))
        tiled = np.tile(mean, reps)[:target]
        vol = tiled.reshape(res, res, res)
    # Normalize
    vmax = float(vol.max()) if vol.size else 1.0
    if vmax > 0:
        vol = vol / vmax
    # Soft threshold to binary-ish occupancy
    occ = (vol >= threshold).astype(np.float64)
    if occ.sum() == 0:
        # Ensure at least center cell occupied
        c = res // 2
        occ[c, c, c] = 1.0
    return occ


def occupancy_to_mesh(
    grid: np.ndarray,
    *,
    origin: Optional[Sequence[float]] = None,
    extents: Optional[Sequence[float]] = None,
) -> trimesh.Trimesh:
    """Convert occupancy grid to a concatenated box soup."""
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError("grid must be 3D")
    res = g.shape[0]
    origin_arr = np.asarray(origin if origin is not None else (-0.5, -0.5, -0.5), dtype=np.float64)
    extents_arr = np.asarray(extents if extents is not None else (1.0, 1.0, 1.0), dtype=np.float64)
    extents_arr = np.maximum(extents_arr, 1e-6)
    cell = extents_arr / res
    meshes = []
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if g[i, j, k] <= 0:
                    continue
                center = origin_arr + (np.array([i, j, k], dtype=np.float64) + 0.5) * cell
                box = trimesh.creation.box(extents=cell * 0.95)
                box.apply_translation(center - box.centroid)
                meshes.append(box)
    if not meshes:
        mesh = trimesh.creation.box(extents=extents_arr * 0.5)
        mesh.apply_translation(origin_arr + 0.5 * extents_arr - mesh.centroid)
        return mesh
    return trimesh.util.concatenate(meshes)


def track2_from_spikes(
    spikes: np.ndarray,
    *,
    resolution: int = 8,
    threshold: float = 0.15,
    origin: Optional[Sequence[float]] = None,
    extents: Optional[Sequence[float]] = None,
    head: Optional["Track2OccupancyHead"] = None,
) -> Asset:
    if head is not None:
        grid = head.predict_occupancy(spikes)
    else:
        grid = spikes_to_occupancy(spikes, resolution=resolution, threshold=threshold)
    mesh = occupancy_to_mesh(grid, origin=origin, extents=extents)
    return asset_from_trimesh(mesh)


class Track2OccupancyHead:
    """MLP: mean-pooled spikes → occupancy logits of shape ``R³``."""

    def __init__(self, in_features: int, *, resolution: int = 8, seed: int = 0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install PyTorch for Track2OccupancyHead.") from exc

        self._torch = torch
        self.in_features = int(in_features)
        self.resolution = max(2, int(resolution))
        out = self.resolution ** 3
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, out),
        )
        with torch.no_grad():
            for p in self.net.parameters():
                p.normal_(0.0, 0.02, generator=g)

    def _features(self, spikes: np.ndarray):
        torch = self._torch
        x = torch.as_tensor(np.asarray(spikes, dtype=np.float32).mean(axis=0), dtype=torch.float32)
        if x.numel() != self.in_features:
            buf = torch.zeros(self.in_features, dtype=torch.float32)
            n = min(self.in_features, int(x.numel()))
            buf[:n] = x.reshape(-1)[:n]
            return buf
        return x

    def forward_logits(self, features) -> "object":
        return self.net(features)

    def predict_occupancy(self, spikes: np.ndarray, *, threshold: float = 0.35) -> np.ndarray:
        torch = self._torch
        x = self._features(spikes)
        with torch.no_grad():
            logits = self.forward_logits(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        grid = probs.reshape(self.resolution, self.resolution, self.resolution)
        occ = (grid >= threshold).astype(np.float64)
        if occ.sum() == 0:
            occ[self.resolution // 2, self.resolution // 2, self.resolution // 2] = 1.0
        return occ
