from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.topology import topology_metrics


def test_box_topology():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    out = topology_metrics(box, box)
    assert out["poly_count_ratio"] == 1.0
    assert out["vertex_count_ratio"] == 1.0
    assert out["is_watertight"] == 1.0
    assert out["is_manifold"] == 1.0
    assert out["ngon_ratio"] == 0.0


def test_poly_ratio_differs_for_denser_candidate():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=3))
    assert topology_metrics(sphere, box)["poly_count_ratio"] > 1.0
