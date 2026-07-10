"""Tests for minimal 3DCG generators."""

from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.generator import (
    generate_candidate,
    primitive_fit_candidate,
    run_generator,
    voxel_occupancy_candidate,
)


def test_primitive_fit_box_scores():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1.0, 2.0, 0.5)))
    cand = primitive_fit_candidate(ref, kind="box")
    assert cand.vertices is not None
    result = run_generator(ref, asset_id="box", kind="primitive_fit")
    assert 0.0 <= result.metrics.quality <= 1.0
    assert result.meta["generator"] == "primitive_fit"


def test_voxel_and_convex_kinds():
    ref = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=1, radius=0.5))
    vox = voxel_occupancy_candidate(ref, resolution=4)
    assert len(vox.vertices) > 0
    hull = generate_candidate(ref, "convex_hull")
    assert len(hull.vertices) > 0
