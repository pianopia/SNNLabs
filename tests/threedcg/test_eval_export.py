"""Smoke tests for eval/export helpers (no long training)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.threedcg.asset import asset_from_trimesh
from src.dst_snn.threedcg.dataset import make_sample
from src.dst_snn.threedcg.pipeline import generate_from_image
import trimesh


def test_generate_rich_program_exports_asset_fields():
    from src.dst_snn.threedcg.ops import (
        OP_ADD_ARMATURE,
        OP_ADD_BOX,
        OP_ASSIGN_MATERIAL,
        OP_AUTO_WEIGHTS,
        OP_FINISH,
        OP_SMART_UV,
        MeshOp,
        ops_to_asset,
    )

    asset = ops_to_asset(
        [
            MeshOp(OP_ADD_BOX, {"extents": [1, 1.2, 0.8]}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "M"}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 2}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    )
    assert asset.uv is not None
    assert len(asset.bones) == 2
    assert asset.skin_weights is not None


def test_export_glb_roundtrip(tmp_path: Path):
    import trimesh as tm

    sample = make_sample(shape="box", extents=(1, 1, 1), seed=0)
    mesh = tm.Trimesh(vertices=sample.asset.vertices, faces=sample.asset.faces, process=False)
    path = tmp_path / "t.glb"
    mesh.export(path)
    assert path.is_file() and path.stat().st_size > 100


def test_scripted_vs_mock_backend_quality_runs():
    # Just ensure both backends produce scorable assets
    from benchmarks.threedcg.scorer import score_to_result
    from src.dst_snn.threedcg.pipeline import synthetic_box_image

    img = synthetic_box_image(size=16)
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    a = generate_from_image(img, track="track1", reference=ref, mesh_backend="trimesh", seed=0)
    b = generate_from_image(img, track="track1", reference=ref, mesh_backend="mock", seed=0)
    qa = score_to_result(a, ref, asset_id="a").metrics.quality
    qb = score_to_result(b, ref, asset_id="b").metrics.quality
    assert 0.0 <= qa <= 1.0
    assert 0.0 <= qb <= 1.0
