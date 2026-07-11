# Rich Blender MeshOps Implementation Plan

**Status (2026-07-11):** Implemented.

**Goal:** Extend Track1 `MeshOp` vocabulary with construction ops (`EXTRUDE`, `SUBDIVIDE`, `BEVEL`) executed on trimesh (CI), MockBlenderScene, and live `bpy` when available.

**Architecture:** Same `MeshOp` names everywhere; `SceneBackend` protocol gains extrude/subdivide/bevel; trimesh approximates where exact bmesh ops are unavailable.

**Out of scope:** Armature, smart UV, auto weights (next depth).
