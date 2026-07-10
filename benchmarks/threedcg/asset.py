"""Load 3D assets into a normalized Asset for scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import trimesh


@dataclass
class Asset:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    uv: Optional[np.ndarray] = None
    bones: list[str] = field(default_factory=list)
    bone_parents: list[int] = field(default_factory=list)
    skin_weights: Optional[np.ndarray] = None
    materials: list[dict[str, Any]] = field(default_factory=list)


def _extract_uv(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != len(mesh.vertices):
        return None
    return uv


def _extract_materials(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return []
    sizes: list[tuple[int, int]] = []

    def _size(image) -> None:
        if image is not None and hasattr(image, "size"):
            sizes.append((int(image.size[0]), int(image.size[1])))

    base = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    normal = getattr(material, "normalTexture", None)
    _size(base)
    _size(normal)
    return [{
        "has_albedo": base is not None,
        "has_normal": normal is not None,
        "has_roughness": getattr(material, "roughnessFactor", None) is not None
        or getattr(material, "metallicRoughnessTexture", None) is not None,
        "has_metallic": getattr(material, "metallicFactor", None) is not None
        or getattr(material, "metallicRoughnessTexture", None) is not None,
        "texture_sizes": sizes,
    }]


def asset_from_trimesh(mesh: trimesh.Trimesh) -> Asset:
    return Asset(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float64),
        uv=_extract_uv(mesh),
        materials=_extract_materials(mesh),
    )


def _concat_mesh(scene_or_mesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = list(scene_or_mesh.geometry.values())
        if not geometries:
            raise ValueError("scene has no geometry")
        return trimesh.util.concatenate(geometries)
    raise TypeError(f"unsupported load result: {type(scene_or_mesh)!r}")


def load_asset(path: str) -> Asset:
    loaded = trimesh.load(path, process=False)
    mesh = _concat_mesh(loaded)
    asset = asset_from_trimesh(mesh)
    _augment_with_gltf_skin(path, asset)
    return asset


def _gltf_component_dtype(component_type: int):
    # glTF componentType codes → NumPy dtypes.
    mapping = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    if component_type not in mapping:
        raise ValueError(f"unsupported glTF componentType: {component_type}")
    return mapping[component_type]


def _gltf_type_count(type_name: str) -> int:
    counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    if type_name not in counts:
        raise ValueError(f"unsupported glTF accessor type: {type_name}")
    return counts[type_name]


def _read_gltf_accessor(gltf, accessor_index: int) -> np.ndarray:
    """Read one glTF accessor into a dense NumPy array of shape [count, components]."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    if blob is None:
        raise ValueError("glTF has no binary buffer")
    dtype = _gltf_component_dtype(accessor.componentType)
    n_comp = _gltf_type_count(accessor.type)
    count = int(accessor.count)
    byte_offset = int(buffer_view.byteOffset or 0) + int(accessor.byteOffset or 0)
    byte_stride = int(buffer_view.byteStride or 0)
    item_size = int(np.dtype(dtype).itemsize * n_comp)
    if byte_stride and byte_stride != item_size:
        rows = []
        for i in range(count):
            start = byte_offset + i * byte_stride
            rows.append(np.frombuffer(blob, dtype=dtype, count=n_comp, offset=start))
        data = np.stack(rows, axis=0)
    else:
        data = np.frombuffer(blob, dtype=dtype, count=count * n_comp, offset=byte_offset)
        data = data.reshape(count, n_comp)
    return np.asarray(data, dtype=np.float64)


def _extract_skin_weights(gltf, asset: Asset) -> Optional[np.ndarray]:
    """Build dense [V, B] skin weights from the first skinned mesh primitive."""
    if not gltf.meshes or not asset.bones:
        return None
    n_bones = len(asset.bones)
    n_verts = len(asset.vertices)
    weights = np.zeros((n_verts, n_bones), dtype=np.float64)
    found = False
    for mesh in gltf.meshes:
        for prim in mesh.primitives or []:
            attrs = prim.attributes
            joints_idx = getattr(attrs, "JOINTS_0", None)
            weights_idx = getattr(attrs, "WEIGHTS_0", None)
            if joints_idx is None or weights_idx is None:
                continue
            try:
                joint_ids = _read_gltf_accessor(gltf, int(joints_idx))
                weight_vals = _read_gltf_accessor(gltf, int(weights_idx))
            except Exception:  # pragma: no cover - malformed accessors
                continue
            if joint_ids.shape[0] != n_verts or weight_vals.shape[0] != n_verts:
                # Mesh vertex count may differ from concatenated trimesh vertices;
                # still populate what we can when sizes match the first mesh only.
                if joint_ids.shape[0] > n_verts:
                    joint_ids = joint_ids[:n_verts]
                    weight_vals = weight_vals[:n_verts]
                elif joint_ids.shape[0] < n_verts:
                    pad = n_verts - joint_ids.shape[0]
                    joint_ids = np.pad(joint_ids, ((0, pad), (0, 0)))
                    weight_vals = np.pad(weight_vals, ((0, pad), (0, 0)))
            for v in range(min(n_verts, joint_ids.shape[0])):
                for k in range(min(4, joint_ids.shape[1], weight_vals.shape[1])):
                    bone = int(joint_ids[v, k])
                    w = float(weight_vals[v, k])
                    if 0 <= bone < n_bones and w > 0.0:
                        weights[v, bone] += w
            found = True
            break
        if found:
            break
    if not found:
        return None
    return weights


def _augment_with_gltf_skin(path: str, asset: Asset) -> None:
    """Populate bones/parents/skin_weights from a glTF skin, if any."""
    if not str(path).lower().endswith((".glb", ".gltf")):
        return
    try:
        import pygltflib
    except ImportError:  # pragma: no cover
        return
    try:
        gltf = pygltflib.GLTF2().load(path)
    except Exception:  # pragma: no cover
        return
    if not gltf.skins:
        return
    skin = gltf.skins[0]
    joints = skin.joints or []
    asset.bones = [gltf.nodes[j].name or f"bone_{j}" for j in joints]
    joint_set = {j: i for i, j in enumerate(joints)}
    parents = [-1] * len(joints)
    for node_index, node in enumerate(gltf.nodes or []):
        for child in node.children or []:
            if child in joint_set and node_index in joint_set:
                parents[joint_set[child]] = joint_set[node_index]
    asset.bone_parents = parents
    asset.skin_weights = _extract_skin_weights(gltf, asset)
