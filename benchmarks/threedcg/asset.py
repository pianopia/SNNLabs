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


def _augment_with_gltf_skin(path: str, asset: Asset) -> None:
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
    for node_index, node in enumerate(gltf.nodes):
        for child in node.children or []:
            if child in joint_set and node_index in joint_set:
                parents[joint_set[child]] = joint_set[node_index]
    asset.bone_parents = parents
