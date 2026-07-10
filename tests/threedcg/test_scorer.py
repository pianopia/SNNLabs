from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.scorer import aggregate_quality, score_assets, score_to_result


def _box():
    return asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))


def test_score_assets_has_all_families():
    assert set(score_assets(_box(), _box())) == {"geometry", "topology", "uv", "rig", "skin", "texture"}


def test_identical_assets_score_higher_than_dissimilar():
    box = _box()
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=3))
    same = aggregate_quality(score_assets(box, box))
    diff = aggregate_quality(score_assets(sphere, box))
    assert 0.0 <= diff <= same <= 1.0
    assert same > diff


def test_score_to_result_shape():
    result = score_to_result(_box(), _box(), asset_id="unit-box", build_latency_ms=12.0)
    assert result.benchmark == "eden14-image-to-3d"
    assert result.metrics.quality_metric == "3dcg_composite"
    assert result.metrics.latency_ms_p50 == 12.0
    assert "scores" in result.metrics.extra
    assert result.meta["asset_id"] == "unit-box"
