"""Minimal 3DCG generators for Phase-0+ scaffolding (not full Track 1/2).

Provides deterministic candidates the scorer can evaluate offline:
  - ``primitive_fit``: axis-aligned box / sphere / cylinder fit to reference
    point cloud extents (simple geometry regression stand-in).
  - ``voxel_occupancy``: coarse occupancy grid from reference AABB, exported
    as a marching-cubes-free box soup (stable without scipy MC).

A real SNN image→3D model replaces these later; the harness and scorer stay fixed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

from .asset import Asset, asset_from_trimesh
from .scorer import score_to_result
from src.dst_snn.eval.result import RunResult

GeneratorKind = Literal["convex_hull", "primitive_fit", "voxel_occupancy"]


def _bounds(reference: Asset) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(reference.vertices, dtype=np.float64)
    if verts.size == 0:
        return np.zeros(3), np.ones(3)
    return verts.min(axis=0), verts.max(axis=0)


def primitive_fit_candidate(
    reference: Asset,
    *,
    kind: Literal["box", "sphere", "cylinder"] = "box",
) -> Asset:
    """Fit a simple primitive to the reference AABB / centroid."""
    lo, hi = _bounds(reference)
    extents = np.maximum(hi - lo, 1e-6)
    center = 0.5 * (lo + hi)
    if kind == "box":
        mesh = trimesh.creation.box(extents=extents)
    elif kind == "sphere":
        radius = float(0.5 * np.linalg.norm(extents))
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=max(radius, 1e-3))
    elif kind == "cylinder":
        height = float(extents[1])
        radius = float(0.5 * max(extents[0], extents[2]))
        mesh = trimesh.creation.cylinder(radius=max(radius, 1e-3), height=max(height, 1e-3))
    else:  # pragma: no cover
        raise ValueError(kind)
    mesh.apply_translation(center - mesh.centroid)
    return asset_from_trimesh(mesh)


def voxel_occupancy_candidate(
    reference: Asset,
    *,
    resolution: int = 8,
) -> Asset:
    """Fill the reference AABB with a coarse grid of unit cubes (occupancy proxy).

    Samples reference vertices onto a grid; cells that contain ≥1 vertex become
    boxes. Empty references fall back to a single center cube.
    """
    lo, hi = _bounds(reference)
    extents = np.maximum(hi - lo, 1e-6)
    verts = np.asarray(reference.vertices, dtype=np.float64)
    res = max(2, int(resolution))
    if verts.size == 0:
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(0.5 * (lo + hi) - mesh.centroid)
        return asset_from_trimesh(mesh)

    norm = (verts - lo) / extents
    idx = np.clip((norm * res).astype(int), 0, res - 1)
    occupied = set(map(tuple, idx.tolist()))
    cell = extents / res
    meshes = []
    for i, j, k in sorted(occupied):
        center = lo + (np.array([i, j, k], dtype=np.float64) + 0.5) * cell
        box = trimesh.creation.box(extents=cell * 0.95)
        box.apply_translation(center - box.centroid)
        meshes.append(box)
    if not meshes:
        mesh = trimesh.creation.box(extents=extents * 0.5)
        mesh.apply_translation(0.5 * (lo + hi) - mesh.centroid)
        return asset_from_trimesh(mesh)
    merged = trimesh.util.concatenate(meshes)
    return asset_from_trimesh(merged)


def generate_candidate(reference: Asset, kind: GeneratorKind = "primitive_fit", **kwargs) -> Asset:
    if kind == "convex_hull":
        from .baseline import convex_hull_candidate

        return convex_hull_candidate(reference)
    if kind == "primitive_fit":
        return primitive_fit_candidate(reference, kind=kwargs.get("primitive", "box"))
    if kind == "voxel_occupancy":
        return voxel_occupancy_candidate(reference, resolution=int(kwargs.get("resolution", 8)))
    raise ValueError(f"unknown generator kind: {kind!r}")


def run_generator(
    reference: Asset,
    *,
    asset_id: str,
    kind: GeneratorKind = "primitive_fit",
    **kwargs,
) -> RunResult:
    candidate = generate_candidate(reference, kind, **kwargs)
    result = score_to_result(candidate, reference, asset_id=asset_id)
    result.meta["generator"] = kind
    result.meta["generator_kwargs"] = {k: v for k, v in kwargs.items() if k != "reference"}
    result.model = f"generator:{kind}"
    return result
