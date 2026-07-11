"""Offline tests for MeshOp → Blender adapter (mock scene; no bpy required)."""

from __future__ import annotations

from pathlib import Path

from src.dst_snn.threedcg.bpy_adapter import (
    MockBlenderScene,
    apply_ops_to_scene,
    bpy_available,
    execute_ops_on_backend,
    ops_to_asset_bpy,
)
from src.dst_snn.threedcg.mesh_backend import backend_info, execute_ops_backend, resolve_backend
from src.dst_snn.threedcg.ops import (
    OP_ADD_BOX,
    OP_ADD_SPHERE,
    OP_FINISH,
    OP_SCALE,
    OP_TRANSLATE,
    OP_UNION,
    MeshOp,
)
from src.dst_snn.threedcg.pipeline import generate_from_image, synthetic_box_image


def test_bpy_available_is_bool():
    assert isinstance(bpy_available(), bool)


def test_mock_scene_applies_ops(tmp_path: Path):
    from src.dst_snn.threedcg.ops import OP_BEVEL, OP_EXTRUDE, OP_SUBDIVIDE

    scene = MockBlenderScene()
    ops = [
        MeshOp(OP_ADD_BOX, {"extents": [1.0, 2.0, 0.5]}),
        MeshOp(OP_TRANSLATE, {"delta": [1.0, 0.0, 0.0]}),
        MeshOp(OP_SCALE, {"factors": [1.5, 1.0, 1.0]}),
        MeshOp(OP_EXTRUDE, {"distance": 0.3, "axis": 2}),
        MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
        MeshOp(OP_BEVEL, {"offset": 0.04, "segments": 1}),
        MeshOp(OP_ADD_SPHERE, {"radius": 0.4}),
        MeshOp(OP_UNION),
        MeshOp(OP_FINISH),
    ]
    apply_ops_to_scene(ops, scene)
    assert any(x.startswith("add_box") for x in scene.ops_log)
    assert any(x.startswith("extrude:") for x in scene.ops_log)
    assert any(x.startswith("subdivide:") for x in scene.ops_log)
    assert any(x.startswith("bevel:") for x in scene.ops_log)
    assert any(x.startswith("union") for x in scene.ops_log)
    path = execute_ops_on_backend(ops, scene=MockBlenderScene(), export_path=tmp_path / "out.glb")
    assert path.is_file()
    assert path.stat().st_size > 100


def test_ops_to_asset_mock():
    asset = ops_to_asset_bpy(
        [MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]}), MeshOp(OP_FINISH)],
        scene=MockBlenderScene(),
    )
    assert asset.vertices is not None
    assert len(asset.vertices) > 0


def test_mesh_backend_resolve_and_execute():
    info = backend_info()
    assert "trimesh" in info["supported"]
    assert resolve_backend("trimesh") == "trimesh"
    assert resolve_backend("mock") == "mock"
    auto = resolve_backend("auto")
    assert auto in {"trimesh", "bpy"}

    asset_t = execute_ops_backend(
        [MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]})],
        backend="trimesh",
    )
    asset_m = execute_ops_backend(
        [MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]})],
        backend="mock",
    )
    assert len(asset_t.vertices) > 0
    assert len(asset_m.vertices) > 0


def test_pipeline_mock_backend():
    from benchmarks.threedcg.asset import asset_from_trimesh
    import trimesh

    img = synthetic_box_image(size=16)
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    cand = generate_from_image(img, track="track1", reference=ref, mesh_backend="mock", seed=0)
    assert len(cand.vertices) > 0
