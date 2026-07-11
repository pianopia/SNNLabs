"""Track 1 mesh construction op vocabulary + trimesh executor.

Includes rich construction ops (EXTRUDE / SUBDIVIDE / BEVEL) and finishing ops
(ADD_ARMATURE / SMART_UV / ASSIGN_MATERIAL / AUTO_WEIGHTS). Blender live ops live
in ``bpy_adapter``; this module stays offline-safe for CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh

OP_ADD_BOX = "ADD_BOX"
OP_ADD_SPHERE = "ADD_SPHERE"
OP_ADD_CYLINDER = "ADD_CYLINDER"
OP_TRANSLATE = "TRANSLATE"
OP_SCALE = "SCALE"
OP_UNION = "UNION"
OP_EXTRUDE = "EXTRUDE"
OP_SUBDIVIDE = "SUBDIVIDE"
OP_BEVEL = "BEVEL"
OP_ADD_ARMATURE = "ADD_ARMATURE"
OP_SMART_UV = "SMART_UV"
OP_ASSIGN_MATERIAL = "ASSIGN_MATERIAL"
OP_AUTO_WEIGHTS = "AUTO_WEIGHTS"
OP_FINISH = "FINISH"

VOCABULARY: tuple[str, ...] = (
    OP_ADD_BOX,
    OP_ADD_SPHERE,
    OP_ADD_CYLINDER,
    OP_TRANSLATE,
    OP_SCALE,
    OP_UNION,
    OP_EXTRUDE,
    OP_SUBDIVIDE,
    OP_BEVEL,
    OP_ADD_ARMATURE,
    OP_SMART_UV,
    OP_ASSIGN_MATERIAL,
    OP_AUTO_WEIGHTS,
    OP_FINISH,
)

# Compact subset used by sequence learning heads
SEQUENCE_VOCAB: tuple[str, ...] = (
    OP_ADD_BOX,
    OP_ADD_SPHERE,
    OP_ADD_CYLINDER,
    OP_EXTRUDE,
    OP_SUBDIVIDE,
    OP_BEVEL,
    OP_SMART_UV,
    OP_ASSIGN_MATERIAL,
    OP_ADD_ARMATURE,
    OP_AUTO_WEIGHTS,
    OP_FINISH,
)


@dataclass(frozen=True)
class MeshOp:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in VOCABULARY:
            raise ValueError(f"unknown mesh op {self.name!r}; known={VOCABULARY}")


@dataclass
class BuildState:
    """Side-channel metadata produced alongside the mesh stack."""

    bones: list[str] = field(default_factory=list)
    bone_parents: list[int] = field(default_factory=list)
    bone_positions: list[list[float]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    skin_weights: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None


def _primitive(name: str, params: dict[str, Any]) -> trimesh.Trimesh:
    if name == OP_ADD_BOX:
        extents = params.get("extents", [1.0, 1.0, 1.0])
        mesh = trimesh.creation.box(extents=list(extents))
    elif name == OP_ADD_SPHERE:
        radius = float(params.get("radius", 0.5))
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=max(radius, 1e-3))
    elif name == OP_ADD_CYLINDER:
        radius = float(params.get("radius", 0.35))
        height = float(params.get("height", 1.0))
        mesh = trimesh.creation.cylinder(radius=max(radius, 1e-3), height=max(height, 1e-3))
    else:  # pragma: no cover
        raise ValueError(name)
    offset = params.get("center") or params.get("translate")
    if offset is not None:
        mesh.apply_translation(np.asarray(offset, dtype=np.float64))
    return mesh


def _axis_index(axis: Any) -> int:
    if isinstance(axis, int):
        return int(np.clip(axis, 0, 2))
    if isinstance(axis, str):
        key = axis.lower()
        if key in {"x", "0"}:
            return 0
        if key in {"y", "1"}:
            return 1
        if key in {"z", "2"}:
            return 2
    return 2


def _extrude_mesh(mesh: trimesh.Trimesh, *, distance: float, axis: int = 2) -> trimesh.Trimesh:
    distance = float(distance)
    if abs(distance) < 1e-9:
        return mesh.copy()
    m = mesh.copy()
    bounds = m.bounds
    span = float(bounds[1, axis] - bounds[0, axis])
    span = max(span, 1e-3)
    factors = np.ones(3, dtype=np.float64)
    factors[axis] = (span + abs(distance)) / span
    center = m.centroid.copy()
    m.apply_translation(-center)
    m.apply_scale(factors)
    m.apply_translation(center)
    shift = np.zeros(3)
    shift[axis] = 0.5 * distance
    m.apply_translation(shift)
    return m


def _subdivide_mesh(mesh: trimesh.Trimesh, *, cuts: int = 1) -> trimesh.Trimesh:
    m = mesh.copy()
    n = max(1, min(int(cuts), 3))
    for _ in range(n):
        try:
            m = m.subdivide()
        except Exception:
            break
    return m


def _bevel_mesh(mesh: trimesh.Trimesh, *, offset: float = 0.05, segments: int = 1) -> trimesh.Trimesh:
    m = _subdivide_mesh(mesh, cuts=max(1, min(int(segments), 2)))
    try:
        if hasattr(m, "vertex_normals") and m.vertex_normals is not None:
            offs = float(np.clip(offset, 0.0, 0.25))
            m.vertices = np.asarray(m.vertices) + np.asarray(m.vertex_normals) * offs * 0.15
    except Exception:
        pass
    return m


def _smart_uv(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Planar + cylindrical hybrid UV (offline stand-in for smart project)."""
    m = mesh.copy()
    v = np.asarray(m.vertices, dtype=np.float64)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    # XY planar component
    uv_xy = (v[:, :2] - lo[:2]) / span[:2]
    # cylindrical around Y
    ang = np.arctan2(v[:, 2] - (lo[2] + hi[2]) * 0.5, v[:, 0] - (lo[0] + hi[0]) * 0.5)
    u_cyl = (ang + np.pi) / (2 * np.pi)
    v_cyl = (v[:, 1] - lo[1]) / span[1]
    # blend by height dominance
    tall = float(span[1] / span.max())
    uv = np.stack(
        [
            (1 - tall) * uv_xy[:, 0] + tall * u_cyl,
            (1 - tall) * uv_xy[:, 1] + tall * v_cyl,
        ],
        axis=1,
    )
    uv = np.clip(uv, 0.0, 1.0)
    try:
        m.visual = trimesh.visual.TextureVisuals(uv=uv)
    except Exception:
        pass
    return m, uv


def _assign_material(state: BuildState, params: dict[str, Any]) -> None:
    mat = {
        "name": str(params.get("name", "Material")),
        "has_albedo": bool(params.get("albedo", True)),
        "has_normal": bool(params.get("normal", False)),
        "has_roughness": bool(params.get("roughness", True)),
        "has_metallic": bool(params.get("metallic", False)),
        "color": list(params.get("color", [0.7, 0.7, 0.75])),
        "texture_sizes": [(int(params.get("tex_res", 256)), int(params.get("tex_res", 256)))]
        if params.get("albedo", True)
        else [],
    }
    state.materials = [mat]


def _add_armature(state: BuildState, mesh: trimesh.Trimesh, params: dict[str, Any]) -> None:
    n_bones = max(1, min(int(params.get("bones", 3)), 8))
    bounds = mesh.bounds
    lo, hi = bounds[0], bounds[1]
    bones = []
    parents = []
    positions = []
    for i in range(n_bones):
        t = (i + 0.5) / n_bones
        # chain along Y
        pos = [
            float(0.5 * (lo[0] + hi[0])),
            float(lo[1] + t * (hi[1] - lo[1])),
            float(0.5 * (lo[2] + hi[2])),
        ]
        bones.append(f"bone_{i}")
        parents.append(-1 if i == 0 else i - 1)
        positions.append(pos)
    state.bones = bones
    state.bone_parents = parents
    state.bone_positions = positions


def _auto_weights(state: BuildState, mesh: trimesh.Trimesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n = len(verts)
    if not state.bone_positions:
        # single root bone
        state.bones = ["bone_0"]
        state.bone_parents = [-1]
        state.bone_positions = [mesh.centroid.tolist()]
    bones = np.asarray(state.bone_positions, dtype=np.float64)
    # inverse-distance weights, top-4 influences
    d = np.linalg.norm(verts[:, None, :] - bones[None, :, :], axis=2) + 1e-4
    inv = 1.0 / d
    # keep strongest 4
    k = min(4, bones.shape[0])
    idx = np.argpartition(-inv, kth=k - 1, axis=1)[:, :k]
    weights = np.zeros((n, bones.shape[0]), dtype=np.float64)
    for i in range(n):
        sel = idx[i]
        w = inv[i, sel]
        w = w / (w.sum() + 1e-9)
        weights[i, sel] = w
    state.skin_weights = weights
    return weights


def execute_ops(ops: Iterable[MeshOp]) -> trimesh.Trimesh:
    """Execute ops; metadata-only side effects are discarded (use ``execute_ops_with_state``)."""
    mesh, _ = execute_ops_with_state(ops)
    return mesh


def execute_ops_with_state(ops: Iterable[MeshOp]) -> tuple[trimesh.Trimesh, BuildState]:
    """Execute a linear op program into a mesh + build metadata."""
    stack: list[trimesh.Trimesh] = []
    state = BuildState()
    for op in ops:
        if op.name in {OP_ADD_BOX, OP_ADD_SPHERE, OP_ADD_CYLINDER}:
            stack.append(_primitive(op.name, op.params))
        elif op.name == OP_TRANSLATE:
            if not stack:
                continue
            delta = np.asarray(op.params.get("delta", [0.0, 0.0, 0.0]), dtype=np.float64)
            stack[-1] = stack[-1].copy()
            stack[-1].apply_translation(delta)
        elif op.name == OP_SCALE:
            if not stack:
                continue
            factors = op.params.get("factors", op.params.get("scale", [1.0, 1.0, 1.0]))
            if isinstance(factors, (int, float)):
                factors = [float(factors)] * 3
            stack[-1] = stack[-1].copy()
            stack[-1].apply_scale(np.asarray(factors, dtype=np.float64))
        elif op.name == OP_EXTRUDE:
            if not stack:
                continue
            distance = float(op.params.get("distance", op.params.get("amount", 0.25)))
            axis = _axis_index(op.params.get("axis", 2))
            stack[-1] = _extrude_mesh(stack[-1], distance=distance, axis=axis)
        elif op.name == OP_SUBDIVIDE:
            if not stack:
                continue
            cuts = int(op.params.get("cuts", op.params.get("levels", 1)))
            stack[-1] = _subdivide_mesh(stack[-1], cuts=cuts)
        elif op.name == OP_BEVEL:
            if not stack:
                continue
            offset = float(op.params.get("offset", op.params.get("width", 0.05)))
            segments = int(op.params.get("segments", 1))
            stack[-1] = _bevel_mesh(stack[-1], offset=offset, segments=segments)
        elif op.name == OP_UNION:
            if len(stack) >= 2:
                stack = [trimesh.util.concatenate(stack)]
        elif op.name == OP_ADD_ARMATURE:
            mesh = stack[-1] if stack else trimesh.creation.box(extents=(1, 1, 1))
            _add_armature(state, mesh, op.params)
        elif op.name == OP_SMART_UV:
            if not stack:
                continue
            stack[-1], uv = _smart_uv(stack[-1])
            state.uv = uv
        elif op.name == OP_ASSIGN_MATERIAL:
            _assign_material(state, op.params)
        elif op.name == OP_AUTO_WEIGHTS:
            mesh = stack[-1] if stack else trimesh.creation.box(extents=(1, 1, 1))
            if not state.bones:
                _add_armature(state, mesh, {"bones": int(op.params.get("bones", 3))})
            _auto_weights(state, mesh)
        elif op.name == OP_FINISH:
            break
        else:  # pragma: no cover
            raise ValueError(op.name)
    if not stack:
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    elif len(stack) == 1:
        mesh = stack[0]
    else:
        mesh = trimesh.util.concatenate(stack)
    # Multi-part programs often UV a single stack item then concatenate —
    # re-unwrap so scorer never sees len(uv) != len(vertices).
    if state.uv is None or len(np.asarray(state.uv)) != len(mesh.vertices):
        try:
            mesh, uv = _smart_uv(mesh)
            state.uv = uv
        except Exception:
            state.uv = None
    # Skin weights must also match final vertex count
    if state.skin_weights is not None and state.skin_weights.shape[0] != len(mesh.vertices):
        try:
            _auto_weights(state, mesh)
        except Exception:
            state.skin_weights = None
    return mesh, state


def ops_to_asset(ops: Iterable[MeshOp]) -> Asset:
    mesh, state = execute_ops_with_state(ops)
    asset = asset_from_trimesh(mesh)
    if state.uv is not None and len(np.asarray(state.uv)) == len(mesh.vertices):
        asset.uv = state.uv
    if state.bones:
        asset.bones = list(state.bones)
        asset.bone_parents = list(state.bone_parents)
    if state.skin_weights is not None and state.skin_weights.shape[0] == len(mesh.vertices):
        asset.skin_weights = state.skin_weights
    if state.materials:
        asset.materials = list(state.materials)
    return asset
