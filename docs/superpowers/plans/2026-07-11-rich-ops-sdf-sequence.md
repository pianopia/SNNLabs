# Rich Ops + Continuous SDF + Op Sequences Implementation Plan

**Status (2026-07-11):** Implemented.

**Goal:** Continue Track1/2 depth: armature/UV/material/weights MeshOps; continuous SDF Track2; multi-op sequence head training.

**Architecture:**
- MeshOps `ADD_ARMATURE`, `SMART_UV`, `ASSIGN_MATERIAL`, `AUTO_WEIGHTS` on trimesh + mock + bpy stubs
- `sdf.py` continuous field + Track2SdfHead + training
- `sequence.py` short op-program labels + Track1SequenceHead

**CI:** No Blender required; offline tests only.
