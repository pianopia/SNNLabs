# MeshOp → Blender bpy Adapter Implementation Plan

**Status (2026-07-11):** Implemented.

**Goal:** Thin adapter from Track1 `MeshOp` sequences to Blender `bpy` (when installed), with trimesh fallback for CI and machines without Blender.

**Architecture:**
- Shared op vocabulary stays in `ops.py`
- `bpy_adapter.py` maps ADD_*/TRANSLATE/SCALE/UNION/FINISH → bpy primitives + export GLB
- `mesh_backend.py` selects `trimesh` | `bpy` | `auto`
- Pipeline / CLI accept `--backend`

**Constraints:** Tests never require Blender; mock scene covers mapping logic offline.
