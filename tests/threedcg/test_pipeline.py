from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from src.dst_snn.threedcg.pipeline import (
    generate_from_image,
    run_pipeline_score,
    synthetic_box_image,
)


def test_generate_track1_with_reference_extents():
    img = synthetic_box_image(size=16)
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    cand = generate_from_image(img, track="track1", reference=ref, seed=0)
    assert cand.vertices is not None
    assert len(cand.vertices) > 0


def test_pipeline_score_quality_positive():
    img = synthetic_box_image(size=16)
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    result = run_pipeline_score(img, ref, track="track1", asset_id="unit-box", seed=0)
    assert result.benchmark == "eden14-image-to-3d"
    assert result.metrics.quality > 0.0
    assert "track" in result.meta
    assert result.metrics.spikes_per_inference >= 0.0


def test_track2_pipeline():
    img = synthetic_box_image(size=16)
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    result = run_pipeline_score(img, ref, track="track2", asset_id="unit-box", seed=1, resolution=4)
    assert result.metrics.quality >= 0.0
    assert result.model.startswith("snn-3dcg:")
