"""Select mesh construction backend: trimesh (default) or Blender bpy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

from benchmarks.threedcg.asset import Asset
from src.dst_snn.threedcg.ops import MeshOp, execute_ops, ops_to_asset

BackendName = Literal["trimesh", "bpy", "auto", "mock"]


def resolve_backend(name: BackendName | str = "auto") -> str:
    key = (name or "auto").lower()
    if key == "auto":
        from src.dst_snn.threedcg.bpy_adapter import bpy_available

        return "bpy" if bpy_available() else "trimesh"
    if key in {"trimesh", "bpy", "mock"}:
        return key
    raise ValueError(f"unknown mesh backend: {name!r}")


def execute_ops_backend(
    ops: Iterable[MeshOp],
    *,
    backend: BackendName | str = "auto",
    export_path: Optional[Path] = None,
) -> Asset:
    """Build an Asset from MeshOps using the selected backend."""
    chosen = resolve_backend(backend)
    if chosen == "trimesh":
        return ops_to_asset(ops)
    if chosen == "mock":
        from src.dst_snn.threedcg.bpy_adapter import MockBlenderScene, ops_to_asset_bpy

        return ops_to_asset_bpy(ops, scene=MockBlenderScene(), export_path=export_path)
    if chosen == "bpy":
        from src.dst_snn.threedcg.bpy_adapter import bpy_available, ops_to_asset_bpy

        if not bpy_available():
            raise ImportError(
                "backend=bpy requested but Blender `bpy` is not importable. "
                "Use backend=trimesh or install/run inside Blender."
            )
        return ops_to_asset_bpy(ops, export_path=export_path)
    raise ValueError(chosen)


def backend_info() -> dict:
    from src.dst_snn.threedcg.bpy_adapter import bpy_available

    return {
        "default": resolve_backend("auto"),
        "bpy_available": bpy_available(),
        "supported": ["trimesh", "bpy", "auto", "mock"],
    }
