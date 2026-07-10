from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.rig import hierarchy_depths, rig_metrics


def _rigged(names, parents) -> Asset:
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.bones = names
    box.bone_parents = parents
    return box


def test_hierarchy_depths_chain():
    assert hierarchy_depths([-1, 0, 1]) == [0, 1, 2]


def test_rig_metrics_matching_skeletons():
    ref = _rigged(["root", "spine", "head"], [-1, 0, 1])
    cand = _rigged(["root", "spine", "head"], [-1, 0, 1])
    out = rig_metrics(cand, ref)
    assert out["has_rig"] == 1.0
    assert out["bone_count_ratio"] == 1.0
    assert out["hierarchy_depth_diff"] == 0.0


def test_rig_metrics_unrigged_reference_returns_none():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    cand = _rigged(["root"], [-1])
    out = rig_metrics(cand, ref)
    assert out["bone_count_ratio"] is None
    assert out["has_rig"] == 1.0
