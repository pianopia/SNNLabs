"""UV-unwrap quality metrics for a candidate asset."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset

_GRID = 64


def _tri_area_2d(p0, p1, p2) -> float:
    return 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _rasterize_counts(asset: Asset) -> np.ndarray:
    counts = np.zeros((_GRID, _GRID), dtype=np.int64)
    assert asset.uv is not None
    uv = asset.uv
    xs = np.linspace(0, 1, _GRID, endpoint=False) + 0.5 / _GRID
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for face in asset.faces:
        a, b, c = uv[face[0]], uv[face[1]], uv[face[2]]
        if _tri_area_2d(a, b, c) < 1e-12:
            continue
        v0 = b - a
        v1 = c - a
        v2 = grid_pts - a
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        d20 = v2 @ v0
        d21 = v2 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        counts.ravel()[(u > 0) & (v > 0) & (w > 0)] += 1
    return counts


def _chart_count(asset: Asset) -> int:
    parent = list(range(len(asset.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    used = set()
    for face in asset.faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
        used.update(int(i) for i in face)
    return len({find(i) for i in used})


def uv_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    if candidate.uv is None or len(candidate.faces) == 0:
        return {
            "has_uv": 0.0,
            "uv_coverage": None,
            "uv_overlap_ratio": None,
            "uv_stretch": None,
            "chart_count": None,
        }
    counts = _rasterize_counts(candidate)
    occupied = (counts > 0).sum()
    total = counts.size
    overlap = (counts > 1).sum()

    stretches = []
    for face in candidate.faces:
        uv_area = _tri_area_2d(candidate.uv[face[0]], candidate.uv[face[1]], candidate.uv[face[2]])
        e1 = candidate.vertices[face[1]] - candidate.vertices[face[0]]
        e2 = candidate.vertices[face[2]] - candidate.vertices[face[0]]
        geo_area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        if uv_area > 1e-9 and geo_area > 1e-9:
            stretches.append(abs(np.log(uv_area / geo_area)))

    return {
        "has_uv": 1.0,
        "uv_coverage": float(occupied / total),
        "uv_overlap_ratio": float(overlap / max(1, occupied)),
        "uv_stretch": float(np.mean(stretches)) if stretches else None,
        "chart_count": float(_chart_count(candidate)),
    }
