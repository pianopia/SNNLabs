from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.baseline import convex_hull_candidate, run_baseline


def test_convex_hull_candidate_is_asset_with_faces():
    ref = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    cand = convex_hull_candidate(ref)
    assert len(cand.faces) > 0
    assert cand.vertices.shape[1] == 3


def test_run_baseline_produces_result():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    result = run_baseline(ref, asset_id="unit-box")
    assert result.benchmark == "eden14-image-to-3d"
    assert 0.0 <= result.metrics.quality <= 1.0
    assert result.metrics.extra["scores"]["geometry"]["volume_iou"] > 0.8
