from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.render_similarity import render_available, ssim
from benchmarks.threedcg.texture import texture_metrics


def test_no_material_returns_none():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = []
    out = texture_metrics(box)
    assert out["has_material"] == 0.0
    assert out["pbr_channel_completeness"] is None


def test_full_pbr_material():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = [{
        "has_albedo": True,
        "has_normal": True,
        "has_roughness": True,
        "has_metallic": True,
        "texture_sizes": [(1024, 1024), (512, 512)],
    }]
    out = texture_metrics(box)
    assert out["has_material"] == 1.0
    assert out["pbr_channel_completeness"] == 1.0
    assert out["max_texture_resolution"] == 1024.0


def test_partial_pbr_material():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = [{
        "has_albedo": True,
        "has_normal": False,
        "has_roughness": False,
        "has_metallic": False,
        "texture_sizes": [],
    }]
    out = texture_metrics(box)
    assert out["pbr_channel_completeness"] == 0.25
    assert out["max_texture_resolution"] is None


def test_ssim_identical_is_one():
    img = np.ones((16, 16), dtype=np.float64) * 0.5
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_ssim_differs_for_different_images():
    assert ssim(np.zeros((16, 16)), np.ones((16, 16))) < 0.5


def test_render_available_is_bool():
    assert isinstance(render_available(), bool)
