from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.rig import hierarchy_depths, hierarchy_edit_distance, rig_metrics


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
    assert out["hierarchy_edit_distance"] == 0.0


def test_hierarchy_edit_distance_detects_reparent():
    ref = _rigged(["root", "spine", "head"], [-1, 0, 1])
    # head attached to root instead of spine
    cand = _rigged(["root", "spine", "head"], [-1, 0, 0])
    dist = hierarchy_edit_distance(cand, ref)
    assert dist is not None and dist > 0.0


def test_rig_metrics_unrigged_reference_returns_none():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    cand = _rigged(["root"], [-1])
    out = rig_metrics(cand, ref)
    assert out["bone_count_ratio"] is None
    assert out["hierarchy_edit_distance"] is None
    assert out["has_rig"] == 1.0
