from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import trimesh

from benchmarks.threedcg.asset import load_asset


def _write_skinned_glb(path: Path) -> None:
    """Write a minimal two-bone skinned triangle mesh as GLB."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.uint16)
    joints = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    weights = np.array(
        [
            [0.7, 0.3, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    inverse_bind = np.tile(np.eye(4, dtype=np.float32), (2, 1)).reshape(2, 4, 4)

    parts = [
        vertices.tobytes(),
        indices.tobytes(),
        joints.tobytes(),
        weights.tobytes(),
        inverse_bind.astype(np.float32).tobytes(),
    ]
    # Align each chunk to 4 bytes.
    aligned: list[bytes] = []
    offsets: list[int] = []
    cursor = 0
    for part in parts:
        pad = (4 - (len(part) % 4)) % 4
        offsets.append(cursor)
        aligned.append(part + b"\x00" * pad)
        cursor += len(part) + pad
    blob = b"".join(aligned)

    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0"),
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[
            pygltflib.Node(name="Armature", children=[1, 2], skin=0, mesh=0),
            pygltflib.Node(name="root"),
            pygltflib.Node(name="child"),
        ],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=0,
                            JOINTS_0=2,
                            WEIGHTS_0=3,
                        ),
                        indices=1,
                    )
                ]
            )
        ],
        skins=[
            pygltflib.Skin(
                name="skin",
                joints=[1, 2],
                inverseBindMatrices=4,
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=3,
                type=pygltflib.VEC3,
                max=[1.0, 1.0, 0.0],
                min=[0.0, 0.0, 0.0],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=3,
                type=pygltflib.SCALAR,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=3,
                type=pygltflib.VEC4,
            ),
            pygltflib.Accessor(
                bufferView=3,
                componentType=pygltflib.FLOAT,
                count=3,
                type=pygltflib.VEC4,
            ),
            pygltflib.Accessor(
                bufferView=4,
                componentType=pygltflib.FLOAT,
                count=2,
                type=pygltflib.MAT4,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(buffer=0, byteOffset=offsets[0], byteLength=len(parts[0]), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=offsets[1], byteLength=len(parts[1]), target=pygltflib.ELEMENT_ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=offsets[2], byteLength=len(parts[2]), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=offsets[3], byteLength=len(parts[3]), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=offsets[4], byteLength=len(parts[4])),
        ],
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    # Parent bone hierarchy: root(-1) -> child(0)
    gltf.nodes[1].children = [2]
    gltf.save(str(path))


def test_load_asset_reads_skin_weights(tmp_path: Path):
    path = tmp_path / "skinned.glb"
    _write_skinned_glb(path)
    asset = load_asset(str(path))
    assert asset.bones == ["root", "child"]
    assert asset.bone_parents == [-1, 0]
    assert asset.skin_weights is not None
    assert asset.skin_weights.shape == (3, 2)
    # Vertex 0: 0.7 on root, 0.3 on child
    np.testing.assert_allclose(asset.skin_weights[0], [0.7, 0.3], atol=1e-5)
    np.testing.assert_allclose(asset.skin_weights[2], [0.0, 1.0], atol=1e-5)


def test_unskinned_box_has_no_skin_weights(tmp_path: Path):
    path = tmp_path / "box.glb"
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.export(path)
    asset = load_asset(str(path))
    assert asset.bones == []
    assert asset.skin_weights is None
