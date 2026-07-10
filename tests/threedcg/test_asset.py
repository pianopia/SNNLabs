from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh


def test_asset_from_trimesh_box():
    asset = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    assert isinstance(asset, Asset)
    assert asset.vertices.shape[1] == 3
    assert asset.faces.shape[1] == 3
    assert asset.vertex_normals.shape == asset.vertices.shape
    assert asset.bones == []
    assert asset.skin_weights is None


def test_asset_uv_optional():
    asset = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    assert asset.uv is None
