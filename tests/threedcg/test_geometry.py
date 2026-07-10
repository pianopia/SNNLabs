from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.geometry import chamfer_distance, geometry_metrics, volumetric_iou


def _box(scale=1.0):
    return asset_from_trimesh(trimesh.creation.box(extents=(scale, scale, scale)))


def test_identical_boxes_have_low_chamfer():
    assert chamfer_distance(_box(), _box()) < 0.05


def test_box_vs_sphere_has_higher_chamfer_than_box_vs_box():
    box = _box()
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=2, radius=0.5))
    assert chamfer_distance(box, sphere) > chamfer_distance(box, _box())


def test_identical_boxes_have_high_iou():
    assert volumetric_iou(_box(), _box()) > 0.9


def test_geometry_metrics_keys():
    assert set(geometry_metrics(_box(), _box())) == {"chamfer", "volume_iou", "normal_consistency"}
