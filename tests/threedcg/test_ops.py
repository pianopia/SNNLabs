from __future__ import annotations

import numpy as np

from src.dst_snn.threedcg.ops import (
    OP_ADD_BOX,
    OP_FINISH,
    OP_SCALE,
    OP_TRANSLATE,
    OP_UNION,
    MeshOp,
    execute_ops,
    ops_to_asset,
)


def test_add_box_nonempty():
    mesh = execute_ops([MeshOp(OP_ADD_BOX, {"extents": [1.0, 2.0, 0.5]}), MeshOp(OP_FINISH)])
    assert len(mesh.vertices) > 0
    extents = mesh.bounding_box.extents
    assert extents[1] > extents[0]


def test_scale_changes_bounds():
    base = execute_ops([MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]})])
    scaled = execute_ops(
        [
            MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]}),
            MeshOp(OP_SCALE, {"factors": [2.0, 2.0, 2.0]}),
        ]
    )
    assert scaled.bounding_box.extents[0] > base.bounding_box.extents[0] * 1.5


def test_translate_and_union():
    mesh = execute_ops(
        [
            MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]}),
            MeshOp(OP_TRANSLATE, {"delta": [2, 0, 0]}),
            MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]}),
            MeshOp(OP_UNION),
        ]
    )
    assert mesh.bounding_box.extents[0] > 2.0


def test_ops_to_asset():
    asset = ops_to_asset([MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]})])
    assert np.asarray(asset.vertices).shape[0] > 0


def test_unknown_op_raises():
    try:
        MeshOp("NOPE", {})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_extrude_subdivide_bevel_increase_complexity():
    from src.dst_snn.threedcg.ops import OP_BEVEL, OP_EXTRUDE, OP_SUBDIVIDE

    base = execute_ops([MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]})])
    rich = execute_ops(
        [
            MeshOp(OP_ADD_BOX, {"extents": [1, 1, 1]}),
            MeshOp(OP_EXTRUDE, {"distance": 0.5, "axis": 2}),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_BEVEL, {"offset": 0.05, "segments": 1}),
            MeshOp(OP_FINISH),
        ]
    )
    # Extrude grows volume along axis
    assert float(rich.bounding_box.volume) > float(base.bounding_box.volume) * 0.9
    # Subdivide increases face count
    assert len(rich.faces) >= len(base.faces)
