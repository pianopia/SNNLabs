"""Tests for armature/UV/weights, continuous SDF, and op sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dst_snn.threedcg.ops import (
    OP_ADD_ARMATURE,
    OP_ADD_BOX,
    OP_ADD_CYLINDER,
    OP_ADD_SPHERE,
    OP_ASSIGN_MATERIAL,
    OP_AUTO_WEIGHTS,
    OP_EXTRUDE,
    OP_FINISH,
    OP_SMART_UV,
    OP_SUBDIVIDE,
    MeshOp,
    VOCABULARY,
    execute_ops_with_state,
    ops_to_asset,
)
from src.dst_snn.threedcg.sdf import mesh_to_sdf, sdf_to_mesh, sdf_to_occupancy
from src.dst_snn.threedcg.sequence import (
    ids_to_program,
    program_to_ids,
    template_program,
)
from src.dst_snn.threedcg.train import train_track1_sequence, train_track2_sdf


def test_vocabulary_includes_finishing_ops():
    for name in (OP_ADD_ARMATURE, OP_SMART_UV, OP_ASSIGN_MATERIAL, OP_AUTO_WEIGHTS):
        assert name in VOCABULARY


def test_smart_uv_armature_weights_material():
    ops = [
        MeshOp(OP_ADD_BOX, {"extents": [1.0, 1.5, 0.8]}),
        MeshOp(OP_EXTRUDE, {"distance": 0.2, "axis": 1}),
        MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
        MeshOp(OP_SMART_UV, {}),
        MeshOp(OP_ASSIGN_MATERIAL, {"name": "Skin", "albedo": True, "roughness": True}),
        MeshOp(OP_ADD_ARMATURE, {"bones": 3}),
        MeshOp(OP_AUTO_WEIGHTS, {}),
        MeshOp(OP_FINISH, {}),
    ]
    mesh, state = execute_ops_with_state(ops)
    assert len(mesh.vertices) > 0
    assert state.uv is not None
    assert state.uv.shape[1] == 2
    assert len(state.bones) == 3
    assert state.skin_weights is not None
    assert state.skin_weights.shape[0] == len(mesh.vertices)
    # rows sum ~1
    assert np.allclose(state.skin_weights.sum(axis=1), 1.0, atol=1e-5)
    asset = ops_to_asset(ops)
    assert asset.uv is not None
    assert len(asset.bones) == 3
    assert asset.skin_weights is not None
    assert asset.materials and asset.materials[0].get("has_albedo") is True


def test_sdf_roundtrip_occupancy():
    import trimesh

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf, origin, extents = mesh_to_sdf(mesh, resolution=8)
    assert sdf.shape == (8, 8, 8)
    # center should be inside (negative or low)
    c = sdf[4, 4, 4]
    assert c < sdf[0, 0, 0]
    occ = sdf_to_occupancy(sdf)
    assert occ.sum() > 0
    out = sdf_to_mesh(sdf, origin=origin, extents=extents)
    assert len(out.vertices) > 0


def test_sequence_template_and_ids():
    prog = template_program("box", (1.0, 1.2, 0.9))
    assert prog[-1].name == OP_FINISH
    ids = program_to_ids(prog)
    assert ids.shape == (8,)
    back = ids_to_program(ids, extents=(1.0, 1.2, 0.9))
    assert back[0].name in {OP_ADD_BOX, OP_ADD_SPHERE, OP_ADD_CYLINDER}


def test_train_sequence_and_sdf(tmp_path: Path):
    r_seq = train_track1_sequence(
        epochs=12,
        n_samples=24,
        seed=0,
        lr=2e-2,
        out_path=tmp_path / "seq.pt",
        image_size=16,
        time_bins=4,
    )
    assert r_seq.final_loss < r_seq.extra["first_loss"]

    r_sdf = train_track2_sdf(
        epochs=8,
        n_samples=12,
        seed=1,
        lr=2e-2,
        resolution=6,
        out_path=tmp_path / "sdf.pt",
        image_size=16,
        time_bins=4,
    )
    assert r_sdf.final_loss <= r_sdf.extra["first_loss"] * 1.1
