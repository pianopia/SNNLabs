"""Continuous SDF fields for Track2 (offline, no GPU).

Stores signed distance on a regular grid (negative inside). Meshing uses
occupancy (sdf <= iso) + box soup; optional skimage marching cubes if present.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from src.dst_snn.threedcg.track2_occupancy import occupancy_to_mesh


def mesh_to_sdf(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 12,
    padding: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate SDF on a grid via nearest surface distance + winding occupancy.

    Returns ``(sdf, origin, extents)`` with shape ``[R,R,R]``.
    """
    res = max(4, int(resolution))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return np.ones((res, res, res)), np.zeros(3), np.ones(3)
    lo = verts.min(axis=0) - padding
    hi = verts.max(axis=0) + padding
    extents = np.maximum(hi - lo, 1e-6)
    origin = lo.copy()
    # sample grid centers
    xs = origin[0] + (np.arange(res) + 0.5) * extents[0] / res
    ys = origin[1] + (np.arange(res) + 0.5) * extents[1] / res
    zs = origin[2] + (np.arange(res) + 0.5) * extents[2] / res
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    # proximity query
    try:
        closest, distance, _ = mesh.nearest.on_surface(grid)
        # signed: negative if inside
        try:
            inside = mesh.contains(grid)
        except Exception:
            # fallback: use winding via proximity only (unsigned)
            inside = np.zeros(len(grid), dtype=bool)
        sdf = distance.astype(np.float64)
        sdf[inside] *= -1.0
    except Exception:
        # very rough fallback: distance to centroid - radius
        c = mesh.centroid
        r = float(np.linalg.norm(verts - c, axis=1).mean())
        sdf = np.linalg.norm(grid - c, axis=1) - r
    return sdf.reshape(res, res, res), origin, extents


def sdf_to_occupancy(sdf: np.ndarray, *, iso: float = 0.0) -> np.ndarray:
    return (np.asarray(sdf, dtype=np.float64) <= float(iso)).astype(np.float64)


def sdf_to_mesh(
    sdf: np.ndarray,
    *,
    origin: Optional[Sequence[float]] = None,
    extents: Optional[Sequence[float]] = None,
    iso: float = 0.0,
) -> trimesh.Trimesh:
    """Extract mesh from SDF. Prefer marching cubes; else box soup."""
    grid = np.asarray(sdf, dtype=np.float64)
    origin_arr = np.asarray(origin if origin is not None else (-0.5, -0.5, -0.5), dtype=np.float64)
    extents_arr = np.asarray(extents if extents is not None else (1.0, 1.0, 1.0), dtype=np.float64)
    try:
        from skimage import measure  # type: ignore

        # skimage expects spacing
        spacing = extents_arr / np.array(grid.shape, dtype=np.float64)
        verts, faces, normals, _ = measure.marching_cubes(grid, level=iso, spacing=spacing)
        verts = verts + origin_arr
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
        if len(mesh.vertices) > 0:
            return mesh
    except Exception:
        pass
    occ = sdf_to_occupancy(grid, iso=iso)
    return occupancy_to_mesh(occ, origin=origin_arr.tolist(), extents=extents_arr.tolist())


class Track2SdfHead:
    """MLP: mean spikes → continuous SDF grid (tanh-scaled)."""

    def __init__(self, in_features: int, *, resolution: int = 10, seed: int = 0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install PyTorch for Track2SdfHead.") from exc

        self._torch = torch
        self.in_features = int(in_features)
        self.resolution = max(4, int(resolution))
        out = self.resolution ** 3
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out),
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

    def forward_sdf(self, features):
        # unbounded linear output; training uses raw SDF targets scaled
        return self.net(features)

    def predict_sdf(self, spikes: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = self._features(spikes)
        with torch.no_grad():
            out = self.forward_sdf(x).cpu().numpy()
        return out.reshape(self.resolution, self.resolution, self.resolution)

    def predict_mesh(
        self,
        spikes: np.ndarray,
        *,
        origin: Optional[Sequence[float]] = None,
        extents: Optional[Sequence[float]] = None,
    ) -> Asset:
        sdf = self.predict_sdf(spikes)
        mesh = sdf_to_mesh(sdf, origin=origin, extents=extents, iso=0.0)
        return asset_from_trimesh(mesh)
