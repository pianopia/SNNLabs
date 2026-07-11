"""MeshOp → Blender ``bpy`` adapter (optional dependency).

When Blender's Python module ``bpy`` is available (Blender embedded or
``bpy`` wheel in a Blender build), Track1 op sequences run as real mesh
operators and export glTF/GLB. Without ``bpy``, call sites should use
``mesh_backend`` auto-fallback to trimesh.

This module also exposes a ``SceneBackend`` protocol and ``MockBlenderScene``
so unit tests can verify op mapping without installing Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, runtime_checkable
import tempfile

import numpy as np

from src.dst_snn.threedcg.ops import (
    OP_ADD_ARMATURE,
    OP_ADD_BOX,
    OP_ADD_CYLINDER,
    OP_ADD_SPHERE,
    OP_ASSIGN_MATERIAL,
    OP_AUTO_WEIGHTS,
    OP_BEVEL,
    OP_EXTRUDE,
    OP_FINISH,
    OP_SCALE,
    OP_SMART_UV,
    OP_SUBDIVIDE,
    OP_TRANSLATE,
    OP_UNION,
    MeshOp,
    _axis_index,
    _bevel_mesh,
    _extrude_mesh,
    _subdivide_mesh,
)


def bpy_available() -> bool:
    try:
        import bpy  # noqa: F401
    except Exception:
        return False
    return True


@runtime_checkable
class SceneBackend(Protocol):
    """Minimal scene API shared by real bpy and the mock."""

    def clear(self) -> None: ...

    def add_box(self, extents: list[float], location: list[float] | None = None) -> str: ...

    def add_sphere(self, radius: float, location: list[float] | None = None) -> str: ...

    def add_cylinder(
        self, radius: float, height: float, location: list[float] | None = None
    ) -> str: ...

    def translate(self, name: str, delta: list[float]) -> None: ...

    def scale(self, name: str, factors: list[float]) -> None: ...

    def extrude(self, name: str, distance: float, axis: int = 2) -> None: ...

    def subdivide(self, name: str, cuts: int = 1) -> None: ...

    def bevel(self, name: str, offset: float = 0.05, segments: int = 1) -> None: ...

    def add_armature(self, name: str, bones: int = 3) -> None: ...

    def smart_uv(self, name: str) -> None: ...

    def assign_material(self, name: str, params: dict[str, Any]) -> None: ...

    def auto_weights(self, name: str) -> None: ...

    def union_all(self) -> str: ...

    def export_glb(self, path: Path) -> Path: ...

    def object_names(self) -> list[str]: ...


@dataclass
class MockObject:
    name: str
    kind: str
    location: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockBlenderScene:
    """Records MeshOp-equivalent calls for offline tests (no bpy)."""

    objects: dict[str, MockObject] = field(default_factory=dict)
    _counter: int = 0
    last_export: Optional[Path] = None
    ops_log: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.objects.clear()
        self._counter = 0
        self.ops_log.append("clear")

    def _new_name(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}_{self._counter}"

    def add_box(self, extents: list[float], location: list[float] | None = None) -> str:
        name = self._new_name("Cube")
        self.objects[name] = MockObject(
            name=name,
            kind="box",
            location=list(location or [0, 0, 0]),
            scale=[float(e) / 2.0 for e in extents],  # Blender cube is 2 units default
            params={"extents": list(extents)},
        )
        self.ops_log.append(f"add_box:{name}")
        return name

    def add_sphere(self, radius: float, location: list[float] | None = None) -> str:
        name = self._new_name("Sphere")
        self.objects[name] = MockObject(
            name=name,
            kind="sphere",
            location=list(location or [0, 0, 0]),
            scale=[float(radius)] * 3,
            params={"radius": float(radius)},
        )
        self.ops_log.append(f"add_sphere:{name}")
        return name

    def add_cylinder(
        self, radius: float, height: float, location: list[float] | None = None
    ) -> str:
        name = self._new_name("Cylinder")
        self.objects[name] = MockObject(
            name=name,
            kind="cylinder",
            location=list(location or [0, 0, 0]),
            scale=[float(radius), float(height) / 2.0, float(radius)],
            params={"radius": float(radius), "height": float(height)},
        )
        self.ops_log.append(f"add_cylinder:{name}")
        return name

    def translate(self, name: str, delta: list[float]) -> None:
        obj = self.objects[name]
        obj.location = [a + b for a, b in zip(obj.location, delta)]
        self.ops_log.append(f"translate:{name}")

    def scale(self, name: str, factors: list[float]) -> None:
        obj = self.objects[name]
        obj.scale = [a * b for a, b in zip(obj.scale, factors)]
        self.ops_log.append(f"scale:{name}")

    def extrude(self, name: str, distance: float, axis: int = 2) -> None:
        obj = self.objects[name]
        # Grow extents / height along axis in stored params
        if obj.kind == "box":
            ex = list(obj.params.get("extents", [1.0, 1.0, 1.0]))
            while len(ex) < 3:
                ex.append(1.0)
            ex[axis] = float(ex[axis]) + abs(float(distance))
            obj.params["extents"] = ex
        elif obj.kind == "cylinder":
            if axis == 1:
                obj.params["height"] = float(obj.params.get("height", 1.0)) + abs(float(distance))
            else:
                obj.params["radius"] = float(obj.params.get("radius", 0.35)) + abs(float(distance)) * 0.25
        else:
            obj.params["radius"] = float(obj.params.get("radius", 0.5)) + abs(float(distance)) * 0.25
        obj.params["extrude_distance"] = float(obj.params.get("extrude_distance", 0.0)) + float(distance)
        obj.params["extrude_axis"] = int(axis)
        self.ops_log.append(f"extrude:{name}:{distance}:{axis}")

    def subdivide(self, name: str, cuts: int = 1) -> None:
        obj = self.objects[name]
        obj.params["subdiv"] = int(obj.params.get("subdiv", 0)) + max(1, int(cuts))
        self.ops_log.append(f"subdivide:{name}:{cuts}")

    def bevel(self, name: str, offset: float = 0.05, segments: int = 1) -> None:
        obj = self.objects[name]
        obj.params["bevel_offset"] = float(obj.params.get("bevel_offset", 0.0)) + float(offset)
        obj.params["bevel_segments"] = int(obj.params.get("bevel_segments", 0)) + max(1, int(segments))
        self.ops_log.append(f"bevel:{name}:{offset}:{segments}")

    def add_armature(self, name: str, bones: int = 3) -> None:
        if name in self.objects:
            self.objects[name].params["armature_bones"] = max(1, int(bones))
        self.ops_log.append(f"armature:{name}:{bones}")

    def smart_uv(self, name: str) -> None:
        if name in self.objects:
            self.objects[name].params["smart_uv"] = True
        self.ops_log.append(f"smart_uv:{name}")

    def assign_material(self, name: str, params: dict[str, Any]) -> None:
        if name in self.objects:
            self.objects[name].params["material"] = dict(params)
        self.ops_log.append(f"assign_material:{name}")

    def auto_weights(self, name: str) -> None:
        if name in self.objects:
            self.objects[name].params["auto_weights"] = True
        self.ops_log.append(f"auto_weights:{name}")

    def union_all(self) -> str:
        names = list(self.objects.keys())
        if not names:
            return self.add_box([1, 1, 1])
        if len(names) == 1:
            return names[0]
        # Collapse to first object; record join
        keep = names[0]
        for n in names[1:]:
            self.objects.pop(n, None)
        self.ops_log.append(f"union:{keep}")
        return keep

    def export_glb(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal placeholder GLB bytes (not a valid full mesh) — tests only check path write.
        # For mock, write a tiny valid-enough file via trimesh instead.
        import trimesh

        meshes = []
        for obj in self.objects.values():
            if obj.kind == "box":
                ex = obj.params.get("extents", [1, 1, 1])
                m = trimesh.creation.box(extents=ex)
            elif obj.kind == "sphere":
                m = trimesh.creation.icosphere(subdivisions=1, radius=float(obj.params.get("radius", 0.5)))
            else:
                m = trimesh.creation.cylinder(
                    radius=float(obj.params.get("radius", 0.35)),
                    height=float(obj.params.get("height", 1.0)),
                )
            # Apply recorded rich ops as trimesh approximations
            if float(obj.params.get("extrude_distance", 0.0)) != 0.0:
                m = _extrude_mesh(
                    m,
                    distance=float(obj.params["extrude_distance"]),
                    axis=int(obj.params.get("extrude_axis", 2)),
                )
            subdiv = int(obj.params.get("subdiv", 0))
            if subdiv > 0:
                m = _subdivide_mesh(m, cuts=subdiv)
            if float(obj.params.get("bevel_offset", 0.0)) > 0:
                m = _bevel_mesh(
                    m,
                    offset=float(obj.params["bevel_offset"]),
                    segments=int(obj.params.get("bevel_segments", 1)),
                )
            if obj.params.get("smart_uv"):
                from src.dst_snn.threedcg.ops import _smart_uv

                m, _ = _smart_uv(m)
            m.apply_scale(obj.scale)
            m.apply_translation(obj.location)
            meshes.append(m)
        if not meshes:
            meshes = [trimesh.creation.box(extents=(1, 1, 1))]
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        mesh.export(path)
        self.last_export = path
        self.ops_log.append(f"export:{path}")
        return path

    def object_names(self) -> list[str]:
        return list(self.objects.keys())


class BpyScene:
    """Live Blender scene wrapper (requires ``bpy``)."""

    def __init__(self) -> None:
        try:
            import bpy
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Blender Python module `bpy` is not available. "
                "Install Blender and run inside its Python, or use backend=trimesh."
            ) from exc
        self.bpy = bpy
        self._names: list[str] = []

    def clear(self) -> None:  # pragma: no cover - needs Blender
        bpy = self.bpy
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        # purge orphans lightly
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        self._names = []

    def _active_name(self) -> str:  # pragma: no cover
        obj = self.bpy.context.view_layer.objects.active
        if obj is None:
            raise RuntimeError("no active Blender object")
        self._names.append(obj.name)
        return obj.name

    def add_box(self, extents: list[float], location: list[float] | None = None) -> str:  # pragma: no cover
        bpy = self.bpy
        loc = location or [0.0, 0.0, 0.0]
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        obj = bpy.context.view_layer.objects.active
        obj.scale = (float(extents[0]) / 2.0, float(extents[1]) / 2.0, float(extents[2]) / 2.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return self._active_name()

    def add_sphere(self, radius: float, location: list[float] | None = None) -> str:  # pragma: no cover
        bpy = self.bpy
        loc = location or [0.0, 0.0, 0.0]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(radius), location=loc)
        return self._active_name()

    def add_cylinder(
        self, radius: float, height: float, location: list[float] | None = None
    ) -> str:  # pragma: no cover
        bpy = self.bpy
        loc = location or [0.0, 0.0, 0.0]
        bpy.ops.mesh.primitive_cylinder_add(
            radius=float(radius), depth=float(height), location=loc
        )
        return self._active_name()

    def translate(self, name: str, delta: list[float]) -> None:  # pragma: no cover
        bpy = self.bpy
        obj = bpy.data.objects[name]
        obj.location.x += float(delta[0])
        obj.location.y += float(delta[1])
        obj.location.z += float(delta[2])

    def scale(self, name: str, factors: list[float]) -> None:  # pragma: no cover
        bpy = self.bpy
        obj = bpy.data.objects[name]
        obj.scale = (
            obj.scale.x * float(factors[0]),
            obj.scale.y * float(factors[1]),
            obj.scale.z * float(factors[2]),
        )
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    def _enter_edit(self, name: str) -> None:  # pragma: no cover
        bpy = self.bpy
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects[name]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")

    def _exit_edit(self) -> None:  # pragma: no cover
        self.bpy.ops.object.mode_set(mode="OBJECT")

    def extrude(self, name: str, distance: float, axis: int = 2) -> None:  # pragma: no cover
        bpy = self.bpy
        self._enter_edit(name)
        bpy.ops.mesh.select_all(action="SELECT")
        vec = [0.0, 0.0, 0.0]
        vec[int(axis) % 3] = float(distance)
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={"value": tuple(vec)},
        )
        self._exit_edit()

    def subdivide(self, name: str, cuts: int = 1) -> None:  # pragma: no cover
        bpy = self.bpy
        self._enter_edit(name)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=max(1, min(int(cuts), 5)))
        self._exit_edit()

    def bevel(self, name: str, offset: float = 0.05, segments: int = 1) -> None:  # pragma: no cover
        bpy = self.bpy
        self._enter_edit(name)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.bevel(
            offset=float(max(0.001, offset)),
            segments=max(1, min(int(segments), 4)),
            affect="EDGES",
        )
        self._exit_edit()

    def add_armature(self, name: str, bones: int = 3) -> None:  # pragma: no cover
        bpy = self.bpy
        bpy.ops.object.armature_add(enter_editmode=True)
        arm = bpy.context.view_layer.objects.active
        # simple chain
        for i in range(1, max(1, int(bones))):
            bpy.ops.armature.extrude_move(TRANSFORM_OT_translate={"value": (0, 0, 0.25)})
        bpy.ops.object.mode_set(mode="OBJECT")
        # parent mesh to armature (optional bind later)
        mesh_obj = bpy.data.objects.get(name)
        if mesh_obj is not None and arm is not None:
            mesh_obj.parent = arm

    def smart_uv(self, name: str) -> None:  # pragma: no cover
        bpy = self.bpy
        self._enter_edit(name)
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            bpy.ops.uv.smart_project()
        except Exception:
            bpy.ops.uv.unwrap(method="ANGLE_BASED")
        self._exit_edit()

    def assign_material(self, name: str, params: dict[str, Any]) -> None:  # pragma: no cover
        bpy = self.bpy
        obj = bpy.data.objects[name]
        mat = bpy.data.materials.new(name=str(params.get("name", "SNNMat")))
        mat.use_nodes = True
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    def auto_weights(self, name: str) -> None:  # pragma: no cover
        bpy = self.bpy
        obj = bpy.data.objects.get(name)
        if obj is None:
            return
        arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        if arm is None:
            self.add_armature(name, bones=3)
            arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        if arm is None:
            return
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except Exception:
            pass

    def union_all(self) -> str:  # pragma: no cover
        bpy = self.bpy
        objs = [o for o in bpy.data.objects if o.type == "MESH"]
        if not objs:
            return self.add_box([1, 1, 1])
        if len(objs) == 1:
            return objs[0].name
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        return bpy.context.view_layer.objects.active.name

    def export_glb(self, path: Path) -> Path:  # pragma: no cover
        bpy = self.bpy
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
        return path

    def object_names(self) -> list[str]:  # pragma: no cover
        return [o.name for o in self.bpy.data.objects if o.type == "MESH"]


def apply_ops_to_scene(ops: Iterable[MeshOp], scene: SceneBackend) -> SceneBackend:
    """Apply MeshOp stack semantics on a SceneBackend (bpy or mock)."""
    scene.clear()
    stack: list[str] = []
    for op in ops:
        if op.name == OP_ADD_BOX:
            extents = [float(x) for x in op.params.get("extents", [1.0, 1.0, 1.0])]
            while len(extents) < 3:
                extents.append(extents[-1] if extents else 1.0)
            loc = op.params.get("center") or op.params.get("translate")
            name = scene.add_box(extents, list(loc) if loc is not None else None)
            stack.append(name)
        elif op.name == OP_ADD_SPHERE:
            radius = float(op.params.get("radius", 0.5))
            loc = op.params.get("center") or op.params.get("translate")
            name = scene.add_sphere(radius, list(loc) if loc is not None else None)
            stack.append(name)
        elif op.name == OP_ADD_CYLINDER:
            radius = float(op.params.get("radius", 0.35))
            height = float(op.params.get("height", 1.0))
            loc = op.params.get("center") or op.params.get("translate")
            name = scene.add_cylinder(radius, height, list(loc) if loc is not None else None)
            stack.append(name)
        elif op.name == OP_TRANSLATE:
            if not stack:
                continue
            delta = [float(x) for x in op.params.get("delta", [0.0, 0.0, 0.0])]
            while len(delta) < 3:
                delta.append(0.0)
            scene.translate(stack[-1], delta)
        elif op.name == OP_SCALE:
            if not stack:
                continue
            factors = op.params.get("factors", op.params.get("scale", [1.0, 1.0, 1.0]))
            if isinstance(factors, (int, float)):
                factors = [float(factors)] * 3
            factors = [float(x) for x in factors]
            while len(factors) < 3:
                factors.append(1.0)
            scene.scale(stack[-1], factors)
        elif op.name == OP_EXTRUDE:
            if not stack:
                continue
            distance = float(op.params.get("distance", op.params.get("amount", 0.25)))
            axis = _axis_index(op.params.get("axis", 2))
            scene.extrude(stack[-1], distance, axis)
        elif op.name == OP_SUBDIVIDE:
            if not stack:
                continue
            cuts = int(op.params.get("cuts", op.params.get("levels", 1)))
            scene.subdivide(stack[-1], cuts)
        elif op.name == OP_BEVEL:
            if not stack:
                continue
            offset = float(op.params.get("offset", op.params.get("width", 0.05)))
            segments = int(op.params.get("segments", 1))
            scene.bevel(stack[-1], offset, segments)
        elif op.name == OP_ADD_ARMATURE:
            target = stack[-1] if stack else scene.add_box([1, 1, 1])
            if not stack:
                stack.append(target)
            scene.add_armature(stack[-1], bones=int(op.params.get("bones", 3)))
        elif op.name == OP_SMART_UV:
            if not stack:
                continue
            scene.smart_uv(stack[-1])
        elif op.name == OP_ASSIGN_MATERIAL:
            target = stack[-1] if stack else scene.add_box([1, 1, 1])
            if not stack:
                stack.append(target)
            scene.assign_material(stack[-1], dict(op.params))
        elif op.name == OP_AUTO_WEIGHTS:
            if not stack:
                continue
            scene.auto_weights(stack[-1])
        elif op.name == OP_UNION:
            if len(stack) >= 1:
                name = scene.union_all()
                stack = [name]
        elif op.name == OP_FINISH:
            break
        else:
            raise ValueError(f"unsupported op for bpy adapter: {op.name}")
    if not stack:
        scene.add_box([1.0, 1.0, 1.0])
    return scene


def execute_ops_on_backend(
    ops: Iterable[MeshOp],
    *,
    scene: SceneBackend | None = None,
    export_path: Path | None = None,
) -> Path:
    """Run ops on the given scene (or live BpyScene) and export GLB path."""
    if scene is None:
        if not bpy_available():
            raise ImportError("bpy not available; pass MockBlenderScene or use trimesh backend")
        scene = BpyScene()
    apply_ops_to_scene(ops, scene)
    if export_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        export_path = Path(tmp.name)
        tmp.close()
    return scene.export_glb(Path(export_path))


def ops_to_asset_bpy(
    ops: Iterable[MeshOp],
    *,
    scene: SceneBackend | None = None,
    export_path: Path | None = None,
):
    """Execute ops via bpy/mock and load result as Asset."""
    from benchmarks.threedcg.asset import load_asset

    path = execute_ops_on_backend(ops, scene=scene, export_path=export_path)
    return load_asset(str(path))
