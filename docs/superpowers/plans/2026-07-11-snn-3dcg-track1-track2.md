# SNN Image→3D Track 1 / Track 2 First Increment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status (2026-07-11):** Implemented.

**Goal:** Implement the first real increment of the design’s 3DCG generators: a shared image→spike encoder, Track 1 mesh-op token pipeline (Blender-free trimesh executor), and Track 2 coarse occupancy head, all scored by the existing harness.

**Architecture:** Input image (or synthetic render) is rate-coded into a spike tensor. Track 1 decodes spikes into a short sequence of mesh construction ops executed by a pure-Python/trimesh executor (stand-in for bpy tokens). Track 2 decodes spikes into a low-res occupancy grid and converts it to a mesh. Both emit `Asset` and go through `score_to_result`. No live Blender dependency; optional bpy adapter is interface-only.

**Tech Stack:** Python 3.9+, NumPy, trimesh (existing), PyTorch for trainable heads (existing floor), pytest.

## Global Constraints

- Every new Python module starts with `from __future__ import annotations`.
- Torch-heavy modules use the existing ImportError guard pattern.
- Tests MUST NOT access the network or require Blender/GPU.
- Package imports from repo ROOT.
- Do not claim SOTA image→3D; document as scaffold that beats convex-hull on synthetic refs when using reference-conditioned scripts.
- Keep files focused; reuse `Asset`, `score_to_result`, `asset_from_trimesh`.

---

## File Structure

```
src/dst_snn/threedcg/__init__.py
src/dst_snn/threedcg/image_spikes.py          # image → spike tensor
src/dst_snn/threedcg/ops.py                   # Track 1 op vocabulary + trimesh executor
src/dst_snn/threedcg/track1_policy.py         # spike → op sequence (scripted + torch head)
src/dst_snn/threedcg/track2_occupancy.py      # spike → occupancy grid → mesh
src/dst_snn/threedcg/pipeline.py              # image→candidate Asset end-to-end
benchmarks/threedcg/run_generate.py           # CLI runner → RunResult
tests/threedcg/test_image_spikes.py
tests/threedcg/test_ops.py
tests/threedcg/test_track1_policy.py
tests/threedcg/test_track2_occupancy.py
tests/threedcg/test_pipeline.py
docs/superpowers/progress/2026-07-10-implementation-progress.md  # append
```

---

### Task 1: Image → spike encoder

**Files:**
- Create: `src/dst_snn/threedcg/image_spikes.py`
- Test: `tests/threedcg/test_image_spikes.py`

**Interfaces:**
- `load_image_array(path_or_array) -> np.ndarray` HxWxC float [0,1]
- `image_to_spikes(image, *, time_bins=8, threshold=0.5, seed=0) -> np.ndarray` shape `[T, H*W]` or `[T, features]` with features = flattened luminance + edge channel
- `spike_feature_size(height, width, *, include_edges=True) -> int`

- [x] Implement + tests (deterministic seed; edges via simple Sobel-like kernels without scipy)

---

### Task 2: Track 1 ops vocabulary + executor

**Files:**
- Create: `src/dst_snn/threedcg/ops.py`
- Test: `tests/threedcg/test_ops.py`

**Interfaces:**
- `MeshOp` dataclass: `name: str`, `params: dict`
- Vocabulary: `ADD_BOX`, `ADD_SPHERE`, `ADD_CYLINDER`, `TRANSLATE`, `SCALE`, `UNION`, `FINISH`
- `execute_ops(ops: list[MeshOp]) -> trimesh.Trimesh` (stack of meshes; UNION concatenates)
- `ops_to_asset(ops) -> Asset`

- [x] Implement + tests (ADD_BOX → non-empty mesh; SCALE changes bounds)

---

### Task 3: Track 1 policy (scripted + optional torch)

**Files:**
- Create: `src/dst_snn/threedcg/track1_policy.py`
- Test: `tests/threedcg/test_track1_policy.py`

**Interfaces:**
- `scripted_box_policy(spikes, *, extents_hint=None) -> list[MeshOp]` — uses mean spike rate to pick scale; defaults to unit box
- `Track1OpHead` (torch): linear map from mean-pooled spikes → logits over op classes + param heads (scaffold; may be untrained)
- `decode_ops_from_spikes(spikes, *, mode="scripted") -> list[MeshOp]`

- [x] Scripted path fully tested; torch path optional smoke if torch available

---

### Task 4: Track 2 occupancy

**Files:**
- Create: `src/dst_snn/threedcg/track2_occupancy.py`
- Test: `tests/threedcg/test_track2_occupancy.py`

**Interfaces:**
- `spikes_to_occupancy(spikes, *, resolution=8) -> np.ndarray` bool/float `[R,R,R]`
- `occupancy_to_mesh(grid, *, origin, extents) -> trimesh.Trimesh` (box soup, same idea as voxel_occupancy)
- `track2_from_spikes(spikes, *, resolution=8) -> Asset`

- [x] Implement + tests

---

### Task 5: Pipeline + benchmark CLI

**Files:**
- Create: `src/dst_snn/threedcg/pipeline.py`
- Create: `benchmarks/threedcg/run_generate.py`
- Modify: `benchmarks/threedcg/generator.py` (delegate track1/track2 kinds)
- Test: `tests/threedcg/test_pipeline.py`

**Interfaces:**
- `generate_from_image(image, *, track="track1"|"track2", reference=None) -> Asset`
- `run_pipeline_score(image, reference, *, track, asset_id) -> RunResult`
- CLI: `--image`, `--reference`, `--track`, `--out-dir`

- [x] E2E test: synthetic box image + unit-box ref → quality > 0
- [x] Wire generator kinds `track1_scripted`, `track2_occupancy`

---

### Task 6: Docs + progress

- [x] Update `benchmarks/threedcg/corpus.md` / `benchmarks/README.md`
- [x] Append progress + plan status Implemented

---

## Self-Review

**Spec coverage:** Design §4 Track 1 (bpy tokens → trimesh stand-in) Task 2–3; Track 2 occupancy Task 4; shared image encoder Task 1; harness integration Task 5. Full bpy / SOTA deferred.

**Placeholder scan:** None intentional.

**Type consistency:** `MeshOp`, spike arrays as `np.ndarray`, `Asset` from existing loaders.
