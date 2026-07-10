from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.uv import uv_metrics


def _plane_with_uv() -> Asset:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    asset = asset_from_trimesh(mesh)
    asset.uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return asset


def test_no_uv_returns_none_fields():
    out = uv_metrics(asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1))))
    assert out["has_uv"] == 0.0
    assert out["uv_coverage"] is None
    assert out["chart_count"] is None


def test_full_square_uv_has_high_coverage_single_chart():
    out = uv_metrics(_plane_with_uv())
    assert out["has_uv"] == 1.0
    assert out["uv_coverage"] > 0.9
    assert out["chart_count"] == 1
    assert out["uv_overlap_ratio"] == 0.0
