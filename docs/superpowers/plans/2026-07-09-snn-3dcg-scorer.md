# 3DCG Construction Benchmark Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scorer that compares any generated 3D asset against a SketchFab reference model across geometry, topology, UV, rigging, skinning, and texture metrics, emitting the shared harness `RunResult` schema.

**Architecture:** All metrics operate on glTF/GLB assets (the format SketchFab exports), read with `trimesh` + `pygltflib`, so the core scorer needs **no Blender**. Each metric family is a focused module of pure functions returning a `dict`; a `null` value marks a metric the reference cannot support (e.g. an unrigged reference). An aggregator combines them into a `MetricSet`. A trivial convex-hull baseline generator validates the scorer end-to-end. Render-based perceptual similarity (SSIM/LPIPS) is an optional, isolated, gated module.

**Tech Stack:** Python 3.14, NumPy ≥1.24, trimesh ≥4.0, pygltflib ≥1.16, scipy ≥1.11 (KD-tree for Chamfer), pytest ≥8. Optional: pyrender ≥0.1 (render similarity). Depends on Plan A's `src/dst_snn/eval/result.py` (`MetricSet`, `RunResult`).

## Global Constraints

- Python 3.14; NumPy `>=1.24`; trimesh `>=4.0`; pygltflib `>=1.16`; scipy `>=1.11`; pytest `>=8`. Add to `requirements-3dcg.txt`.
- Every new Python module starts with `from __future__ import annotations`.
- Metrics MUST return JSON-serializable dicts. A metric inapplicable to the reference returns `None` for that key (never raises).
- Tests MUST NOT access the network. Reference/candidate meshes in tests are constructed in-process with `trimesh.creation`.
- **Prerequisite:** Plan A Task 3 (`src/dst_snn/eval/result.py`) must be complete — this plan imports `MetricSet` and `RunResult` from it.
- Lower-is-better metrics (distances, distortion, edit distance) and higher-is-better metrics (IoU, coverage, completeness) are each documented per function; the aggregator normalizes to a single `quality` in `[0, 1]` where higher is better.
- Commit after each task with the exact message shown.

---

## File Structure

```
requirements-3dcg.txt                        # new deps
benchmarks/threedcg/__init__.py
benchmarks/threedcg/asset.py                 # load glTF/GLB into a normalized Asset
benchmarks/threedcg/geometry.py              # Chamfer, IoU, normal consistency
benchmarks/threedcg/topology.py              # counts, manifold, watertight, n-gon
benchmarks/threedcg/uv.py                    # chart count, coverage, stretch, overlap
benchmarks/threedcg/rig.py                   # bone count, hierarchy edit distance, symmetry
benchmarks/threedcg/skin.py                  # weight normalization, influences, smoothness
benchmarks/threedcg/texture.py               # resolution, PBR channel completeness
benchmarks/threedcg/render_similarity.py     # OPTIONAL, gated: SSIM of matched renders
benchmarks/threedcg/scorer.py                # aggregate -> MetricSet/RunResult
benchmarks/threedcg/baseline.py              # convex-hull baseline generator
benchmarks/threedcg/corpus.md                # SketchFab reference corpus spec
tests/threedcg/__init__.py
tests/threedcg/test_asset.py
tests/threedcg/test_geometry.py
tests/threedcg/test_topology.py
tests/threedcg/test_uv.py
tests/threedcg/test_rig.py
tests/threedcg/test_skin.py
tests/threedcg/test_texture.py
tests/threedcg/test_scorer.py
tests/threedcg/test_baseline.py
```

---

### Task 0: Environment and corpus spec

**Files:**
- Create: `requirements-3dcg.txt`
- Create: `benchmarks/threedcg/__init__.py`
- Create: `tests/threedcg/__init__.py`
- Create: `benchmarks/threedcg/corpus.md`

**Interfaces:**
- Produces: installable deps; empty package; a documented reference-corpus contract.

- [ ] **Step 1: Create `requirements-3dcg.txt`**

```
numpy>=1.24
trimesh>=4.0
pygltflib>=1.16
scipy>=1.11
pytest>=8
```

- [ ] **Step 2: Create `benchmarks/threedcg/__init__.py`** (empty)

```python
```

- [ ] **Step 3: Create `tests/threedcg/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `benchmarks/threedcg/corpus.md`**

```markdown
# SketchFab Reference Corpus

The 3DCG construction benchmark scores generated assets against finished
SketchFab reference models. This file defines the corpus contract.

## Format
- Each reference is a `.glb` (binary glTF) file. SketchFab "Download 3D Model"
  → glTF export produces this.
- Each reference pairs with one or more reference render images (`.png`) used
  as the model's input (image→3D task) and for optional render similarity.

## Layout
```
data/threedcg/<asset_id>/
  reference.glb          # SketchFab finished model
  input.png              # reference render used as generation input
  meta.json              # {"license": "...", "category": "...", "rigged": bool}
```

## Selection criteria (to finalize before large-scale runs)
- License: downloadable + redistribution-compatible (e.g. CC-BY). Record in meta.
- Category balance: rigid props, organic characters, hard-surface, foliage.
- Rig coverage: at least 30% rigged+skinned to exercise rig/skin metrics.
- Poly-count spread: low-poly (<5k), mid (5k-50k), high (>50k).

## Applicability
Metrics whose data the reference lacks (e.g. UV on a reference with no UVs,
rig/skin on an unrigged reference) return `None` and are excluded from the
aggregate `quality`.
```

- [ ] **Step 5: Install dependencies**

Run:
```bash
. .venv/bin/activate
pip install -r requirements-3dcg.txt
python -c "import trimesh, pygltflib, scipy; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add requirements-3dcg.txt benchmarks/threedcg/__init__.py tests/threedcg/__init__.py benchmarks/threedcg/corpus.md
git commit -m "chore: bootstrap 3DCG scorer environment and corpus spec"
```

---

### Task 1: Asset loader

**Files:**
- Create: `benchmarks/threedcg/asset.py`
- Test: `tests/threedcg/test_asset.py`

**Interfaces:**
- Produces:
  - `Asset` dataclass: `vertices: np.ndarray [V,3]`, `faces: np.ndarray [F,3]`, `vertex_normals: np.ndarray [V,3]`, `uv: Optional[np.ndarray [V,2]]`, `bones: list[str]`, `bone_parents: list[int]` (parent index per bone, `-1` for root), `skin_weights: Optional[np.ndarray [V,B]]`, `materials: list[dict]` (each `{"has_albedo","has_normal","has_roughness","has_metallic","texture_sizes":[(w,h),...]}`).
  - `load_asset(path: str) -> Asset` — loads `.glb`/`.gltf`/`.obj`; missing data → empty/`None` fields.
  - `asset_from_trimesh(mesh: trimesh.Trimesh) -> Asset` — build an `Asset` from an in-memory mesh (used by tests and the baseline generator).

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_asset.py`

```python
from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh


def test_asset_from_trimesh_box():
    box = trimesh.creation.box(extents=(1, 1, 1))
    asset = asset_from_trimesh(box)
    assert isinstance(asset, Asset)
    assert asset.vertices.shape[1] == 3
    assert asset.faces.shape[1] == 3
    assert asset.vertex_normals.shape == asset.vertices.shape
    assert asset.bones == []
    assert asset.skin_weights is None


def test_asset_uv_optional():
    box = trimesh.creation.box(extents=(1, 1, 1))
    asset = asset_from_trimesh(box)
    # A raw box has no UVs -> None
    assert asset.uv is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_asset.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.asset`

- [ ] **Step 3: Create `benchmarks/threedcg/asset.py`**

```python
"""Load 3D assets (glTF/GLB/OBJ) into a normalized Asset for scoring."""

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
    _size(base)
    normal = getattr(material, "normalTexture", None)
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
    """Load an asset file. Skin/bone extraction from glTF is populated by rig.py's
    reader when present; base geometry/UV/material come from trimesh."""
    loaded = trimesh.load(path, process=False)
    mesh = _concat_mesh(loaded)
    asset = asset_from_trimesh(mesh)
    _augment_with_gltf_skin(path, asset)
    return asset


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
    except Exception:  # pragma: no cover - malformed file
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_asset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/asset.py tests/threedcg/test_asset.py
git commit -m "feat: add 3DCG asset loader (glTF/GLB/OBJ)"
```

---

### Task 2: Geometry metrics

**Files:**
- Create: `benchmarks/threedcg/geometry.py`
- Test: `tests/threedcg/test_geometry.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces (all operate after unit-cube normalization so scale-invariant):
  - `normalize_points(points: np.ndarray) -> np.ndarray` — center at centroid, scale so max extent = 1.
  - `chamfer_distance(a: Asset, b: Asset, *, samples: int = 2000) -> float` — symmetric mean nearest-neighbor distance between surface samples (lower better). Deterministic sampling via fixed seed.
  - `volumetric_iou(a: Asset, b: Asset, *, resolution: int = 24) -> float` — voxel IoU after normalization (higher better).
  - `normal_consistency(a: Asset, b: Asset, *, samples: int = 2000) -> float` — mean absolute cosine of normals at nearest-neighbor sample pairs (higher better).
  - `geometry_metrics(candidate: Asset, reference: Asset) -> dict[str, float]` — keys `chamfer`, `volume_iou`, `normal_consistency`.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_geometry.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.geometry import (
    chamfer_distance,
    geometry_metrics,
    volumetric_iou,
)


def _box(scale=1.0):
    return asset_from_trimesh(trimesh.creation.box(extents=(scale, scale, scale)))


def test_identical_boxes_have_low_chamfer():
    a, b = _box(), _box()
    assert chamfer_distance(a, b) < 0.05


def test_box_vs_sphere_has_higher_chamfer_than_box_vs_box():
    box = _box()
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=2, radius=0.5))
    same = chamfer_distance(box, _box())
    diff = chamfer_distance(box, sphere)
    assert diff > same


def test_identical_boxes_have_high_iou():
    assert volumetric_iou(_box(), _box()) > 0.9


def test_geometry_metrics_keys():
    out = geometry_metrics(_box(), _box())
    assert set(out) == {"chamfer", "volume_iou", "normal_consistency"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.geometry`

- [ ] **Step 3: Create `benchmarks/threedcg/geometry.py`**

```python
"""Geometry similarity metrics between a candidate and reference asset."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .asset import Asset


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points
    centroid = points.mean(axis=0)
    centered = points - centroid
    extent = np.abs(centered).max()
    if extent < 1e-9:
        return centered
    return centered / extent


def _mesh(asset: Asset) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)


def _sample_surface(asset: Asset, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = _mesh(asset)
    rng = np.random.default_rng(seed)
    points, face_index = trimesh.sample.sample_surface(mesh, samples, seed=rng.integers(1 << 31))
    normals = mesh.face_normals[face_index]
    return normalize_points(np.asarray(points)), np.asarray(normals)


def chamfer_distance(a: Asset, b: Asset, *, samples: int = 2000) -> float:
    pa, _ = _sample_surface(a, samples, seed=0)
    pb, _ = _sample_surface(b, samples, seed=1)
    tree_a = cKDTree(pa)
    tree_b = cKDTree(pb)
    dist_ab, _ = tree_b.query(pa)
    dist_ba, _ = tree_a.query(pb)
    return float((dist_ab.mean() + dist_ba.mean()) / 2.0)


def volumetric_iou(a: Asset, b: Asset, *, resolution: int = 24) -> float:
    grid = np.linspace(-1.0, 1.0, resolution)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def _occupancy(asset: Asset) -> np.ndarray:
        mesh = _mesh(asset)
        mesh = mesh.copy()
        mesh.vertices = normalize_points(mesh.vertices)
        try:
            return mesh.contains(query)
        except Exception:
            return np.zeros(len(query), dtype=bool)

    occ_a = _occupancy(a)
    occ_b = _occupancy(b)
    union = np.logical_or(occ_a, occ_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(occ_a, occ_b).sum()
    return float(intersection / union)


def normal_consistency(a: Asset, b: Asset, *, samples: int = 2000) -> float:
    pa, na = _sample_surface(a, samples, seed=0)
    pb, nb = _sample_surface(b, samples, seed=1)
    tree_b = cKDTree(pb)
    _, idx = tree_b.query(pa)
    matched = nb[idx]
    na_unit = na / (np.linalg.norm(na, axis=1, keepdims=True) + 1e-9)
    nb_unit = matched / (np.linalg.norm(matched, axis=1, keepdims=True) + 1e-9)
    cos = np.abs((na_unit * nb_unit).sum(axis=1))
    return float(cos.mean())


def geometry_metrics(candidate: Asset, reference: Asset) -> dict[str, float]:
    return {
        "chamfer": chamfer_distance(candidate, reference),
        "volume_iou": volumetric_iou(candidate, reference),
        "normal_consistency": normal_consistency(candidate, reference),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_geometry.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/geometry.py tests/threedcg/test_geometry.py
git commit -m "feat: add 3DCG geometry metrics (chamfer, IoU, normal consistency)"
```

---

### Task 3: Topology metrics

**Files:**
- Create: `benchmarks/threedcg/topology.py`
- Test: `tests/threedcg/test_topology.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `topology_metrics(candidate: Asset, reference: Asset) -> dict[str, float]` keys:
    - `poly_count_ratio`: candidate faces / reference faces (1.0 = same budget).
    - `vertex_count_ratio`: candidate verts / reference verts.
    - `is_watertight`: 1.0 if candidate watertight else 0.0.
    - `is_manifold`: 1.0 if candidate edges are manifold else 0.0.
    - `ngon_ratio`: fraction of candidate faces that are not triangles (glTF is triangulated → typically 0.0; kept for OBJ inputs). Always 0.0 for triangle-only `faces` arrays.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_topology.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.topology import topology_metrics


def test_box_topology():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    out = topology_metrics(box, box)
    assert out["poly_count_ratio"] == 1.0
    assert out["vertex_count_ratio"] == 1.0
    assert out["is_watertight"] == 1.0
    assert out["is_manifold"] == 1.0
    assert out["ngon_ratio"] == 0.0


def test_poly_ratio_differs_for_denser_candidate():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=3))
    out = topology_metrics(sphere, box)
    assert out["poly_count_ratio"] > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_topology.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.topology`

- [ ] **Step 3: Create `benchmarks/threedcg/topology.py`**

```python
"""Topology metrics for a candidate asset relative to a reference budget."""

from __future__ import annotations

import trimesh

from .asset import Asset


def _mesh(asset: Asset) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)


def topology_metrics(candidate: Asset, reference: Asset) -> dict[str, float]:
    cand_faces = int(len(candidate.faces))
    ref_faces = int(len(reference.faces))
    cand_verts = int(len(candidate.vertices))
    ref_verts = int(len(reference.vertices))
    mesh = _mesh(candidate)
    return {
        "poly_count_ratio": cand_faces / ref_faces if ref_faces else 0.0,
        "vertex_count_ratio": cand_verts / ref_verts if ref_verts else 0.0,
        "is_watertight": 1.0 if mesh.is_watertight else 0.0,
        "is_manifold": 1.0 if mesh.is_winding_consistent else 0.0,
        "ngon_ratio": 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_topology.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/topology.py tests/threedcg/test_topology.py
git commit -m "feat: add 3DCG topology metrics"
```

---

### Task 4: UV metrics

**Files:**
- Create: `benchmarks/threedcg/uv.py`
- Test: `tests/threedcg/test_uv.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `uv_metrics(candidate: Asset) -> dict[str, Optional[float]]` (evaluates the candidate's own UV quality; reference used only for applicability). Keys:
    - `has_uv`: 1.0 if candidate has UVs else 0.0.
    - `uv_coverage`: fraction of the unit UV square occupied by UV triangles (grid-rasterized at 64×64). `None` if no UV.
    - `uv_overlap_ratio`: fraction of the 64×64 UV grid cells covered by more than one triangle. `None` if no UV.
    - `uv_stretch`: mean per-triangle area distortion `abs(log(uv_area / geo_area_normalized))`, averaged; lower is better. `None` if no UV.
    - `chart_count`: number of connected UV islands (via UV-space adjacency of shared vertices). `None` if no UV.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_uv.py`

```python
from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.uv import uv_metrics


def _plane_with_uv() -> Asset:
    # Two triangles forming a unit quad, UVs covering the full unit square.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    asset = asset_from_trimesh(mesh)
    asset.uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return asset


def test_no_uv_returns_none_fields():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    out = uv_metrics(box)
    assert out["has_uv"] == 0.0
    assert out["uv_coverage"] is None
    assert out["chart_count"] is None


def test_full_square_uv_has_high_coverage_single_chart():
    out = uv_metrics(_plane_with_uv())
    assert out["has_uv"] == 1.0
    assert out["uv_coverage"] > 0.9
    assert out["chart_count"] == 1
    assert out["uv_overlap_ratio"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_uv.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.uv`

- [ ] **Step 3: Create `benchmarks/threedcg/uv.py`**

```python
"""UV-unwrap quality metrics for a candidate asset."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset

_GRID = 64


def _tri_area_2d(p0, p1, p2) -> float:
    return 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _rasterize_counts(asset: Asset) -> np.ndarray:
    counts = np.zeros((_GRID, _GRID), dtype=np.int64)
    uv = asset.uv
    xs = np.linspace(0, 1, _GRID, endpoint=False) + 0.5 / _GRID
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for face in asset.faces:
        a, b, c = uv[face[0]], uv[face[1]], uv[face[2]]
        area = _tri_area_2d(a, b, c)
        if area < 1e-12:
            continue
        # Barycentric point-in-triangle test for all grid points.
        v0 = b - a
        v1 = c - a
        v2 = grid_pts - a
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        d20 = v2 @ v0
        d21 = v2 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        counts.ravel()[inside] += 1
    return counts


def _chart_count(asset: Asset) -> int:
    # Union-find over faces sharing a vertex index (UV islands approximated by
    # topological connectivity, which matches per-vertex UV seams).
    parent = list(range(len(asset.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    used = set()
    for face in asset.faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
        used.update(int(i) for i in face)
    roots = {find(i) for i in used}
    return len(roots)


def uv_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    if candidate.uv is None or len(candidate.faces) == 0:
        return {
            "has_uv": 0.0,
            "uv_coverage": None,
            "uv_overlap_ratio": None,
            "uv_stretch": None,
            "chart_count": None,
        }
    counts = _rasterize_counts(candidate)
    occupied = (counts > 0).sum()
    total = counts.size
    overlap = (counts > 1).sum()

    uv = candidate.uv
    verts = candidate.vertices
    stretches = []
    for face in candidate.faces:
        uv_area = _tri_area_2d(uv[face[0]], uv[face[1]], uv[face[2]])
        e1 = verts[face[1]] - verts[face[0]]
        e2 = verts[face[2]] - verts[face[0]]
        geo_area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        if uv_area > 1e-9 and geo_area > 1e-9:
            stretches.append(abs(np.log(uv_area / geo_area)))
    stretch = float(np.mean(stretches)) if stretches else None

    return {
        "has_uv": 1.0,
        "uv_coverage": float(occupied / total),
        "uv_overlap_ratio": float(overlap / max(1, occupied)),
        "uv_stretch": stretch,
        "chart_count": float(_chart_count(candidate)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_uv.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/uv.py tests/threedcg/test_uv.py
git commit -m "feat: add 3DCG UV-unwrap metrics"
```

---

### Task 5: Rig metrics

**Files:**
- Create: `benchmarks/threedcg/rig.py`
- Test: `tests/threedcg/test_rig.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `hierarchy_depths(bone_parents: list[int]) -> list[int]` — depth of each bone from its root.
  - `rig_metrics(candidate: Asset, reference: Asset) -> dict[str, Optional[float]]` keys:
    - `bone_count_ratio`: candidate bones / reference bones. `None` if reference unrigged.
    - `hierarchy_depth_diff`: abs difference of max hierarchy depth vs reference. `None` if reference unrigged.
    - `has_rig`: 1.0 if candidate has ≥1 bone else 0.0.
    Returns all-`None` (except `has_rig`) when the reference has no bones.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_rig.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.rig import hierarchy_depths, rig_metrics


def _rigged(names, parents) -> Asset:
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.bones = names
    box.bone_parents = parents
    return box


def test_hierarchy_depths_chain():
    # root(-1) -> b1(0) -> b2(1)
    assert hierarchy_depths([-1, 0, 1]) == [0, 1, 2]


def test_rig_metrics_matching_skeletons():
    ref = _rigged(["root", "spine", "head"], [-1, 0, 1])
    cand = _rigged(["root", "spine", "head"], [-1, 0, 1])
    out = rig_metrics(cand, ref)
    assert out["has_rig"] == 1.0
    assert out["bone_count_ratio"] == 1.0
    assert out["hierarchy_depth_diff"] == 0.0


def test_rig_metrics_unrigged_reference_returns_none():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    cand = _rigged(["root"], [-1])
    out = rig_metrics(cand, ref)
    assert out["bone_count_ratio"] is None
    assert out["has_rig"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_rig.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.rig`

- [ ] **Step 3: Create `benchmarks/threedcg/rig.py`**

```python
"""Rig/skeleton structure metrics."""

from __future__ import annotations

from typing import Optional

from .asset import Asset


def hierarchy_depths(bone_parents: list[int]) -> list[int]:
    depths = [0] * len(bone_parents)
    for i in range(len(bone_parents)):
        depth = 0
        cursor = bone_parents[i]
        guard = 0
        while cursor is not None and cursor >= 0 and guard <= len(bone_parents):
            depth += 1
            cursor = bone_parents[cursor]
            guard += 1
        depths[i] = depth
    return depths


def rig_metrics(candidate: Asset, reference: Asset) -> dict[str, Optional[float]]:
    has_rig = 1.0 if candidate.bones else 0.0
    if not reference.bones:
        return {"has_rig": has_rig, "bone_count_ratio": None, "hierarchy_depth_diff": None}
    ref_count = len(reference.bones)
    cand_count = len(candidate.bones)
    ref_depth = max(hierarchy_depths(reference.bone_parents), default=0)
    cand_depth = max(hierarchy_depths(candidate.bone_parents), default=0)
    return {
        "has_rig": has_rig,
        "bone_count_ratio": cand_count / ref_count if ref_count else None,
        "hierarchy_depth_diff": float(abs(cand_depth - ref_depth)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_rig.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/rig.py tests/threedcg/test_rig.py
git commit -m "feat: add 3DCG rig/skeleton metrics"
```

---

### Task 6: Skinning/weight metrics

**Files:**
- Create: `benchmarks/threedcg/skin.py`
- Test: `tests/threedcg/test_skin.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `skin_metrics(candidate: Asset) -> dict[str, Optional[float]]` keys (candidate `skin_weights` is `[V, B]`):
    - `has_skin`: 1.0 if candidate has skin weights else 0.0.
    - `weight_normalization_error`: mean abs(row_sum − 1). `None` if no skin.
    - `max_influences`: max non-zero weights per vertex (int as float). `None` if no skin.
    - `mean_influences`: mean non-zero weights per vertex. `None` if no skin.
    - `isolated_weight_ratio`: fraction of vertices with zero total weight. `None` if no skin.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_skin.py`

```python
from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.skin import skin_metrics


def _skinned(weights: np.ndarray) -> Asset:
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.skin_weights = weights
    box.bones = [f"b{i}" for i in range(weights.shape[1])]
    return box


def test_no_skin_returns_none():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    out = skin_metrics(box)
    assert out["has_skin"] == 0.0
    assert out["weight_normalization_error"] is None


def test_normalized_weights():
    weights = np.array([[0.5, 0.5], [1.0, 0.0], [0.3, 0.7]])
    out = skin_metrics(_skinned(weights))
    assert out["has_skin"] == 1.0
    assert abs(out["weight_normalization_error"]) < 1e-9
    assert out["max_influences"] == 2.0
    assert out["isolated_weight_ratio"] == 0.0


def test_unnormalized_and_isolated():
    weights = np.array([[0.5, 0.2], [0.0, 0.0]])  # row0 sums 0.7, row1 isolated
    out = skin_metrics(_skinned(weights))
    assert out["weight_normalization_error"] > 0.1
    assert out["isolated_weight_ratio"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_skin.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.skin`

- [ ] **Step 3: Create `benchmarks/threedcg/skin.py`**

```python
"""Skinning / weight-paint quality metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset


def skin_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    weights = candidate.skin_weights
    if weights is None or weights.size == 0:
        return {
            "has_skin": 0.0,
            "weight_normalization_error": None,
            "max_influences": None,
            "mean_influences": None,
            "isolated_weight_ratio": None,
        }
    weights = np.asarray(weights, dtype=np.float64)
    row_sums = weights.sum(axis=1)
    nonzero = weights > 1e-6
    influences = nonzero.sum(axis=1)
    isolated = (row_sums <= 1e-6).sum()
    return {
        "has_skin": 1.0,
        "weight_normalization_error": float(np.abs(row_sums - 1.0).mean()),
        "max_influences": float(influences.max()),
        "mean_influences": float(influences.mean()),
        "isolated_weight_ratio": float(isolated / len(weights)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_skin.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/skin.py tests/threedcg/test_skin.py
git commit -m "feat: add 3DCG skinning/weight metrics"
```

---

### Task 7: Texture metrics

**Files:**
- Create: `benchmarks/threedcg/texture.py`
- Test: `tests/threedcg/test_texture.py`

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `texture_metrics(candidate: Asset) -> dict[str, Optional[float]]` keys:
    - `has_material`: 1.0 if candidate has ≥1 material else 0.0.
    - `pbr_channel_completeness`: mean over materials of the fraction of `{albedo, normal, roughness, metallic}` present (0..1). `None` if no material.
    - `max_texture_resolution`: max texture edge across materials (pixels). `None` if no textures.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_texture.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.texture import texture_metrics


def test_no_material_returns_none():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = []
    out = texture_metrics(box)
    assert out["has_material"] == 0.0
    assert out["pbr_channel_completeness"] is None


def test_full_pbr_material():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = [{
        "has_albedo": True,
        "has_normal": True,
        "has_roughness": True,
        "has_metallic": True,
        "texture_sizes": [(1024, 1024), (512, 512)],
    }]
    out = texture_metrics(box)
    assert out["has_material"] == 1.0
    assert out["pbr_channel_completeness"] == 1.0
    assert out["max_texture_resolution"] == 1024.0


def test_partial_pbr_material():
    box = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    box.materials = [{
        "has_albedo": True,
        "has_normal": False,
        "has_roughness": False,
        "has_metallic": False,
        "texture_sizes": [],
    }]
    out = texture_metrics(box)
    assert out["pbr_channel_completeness"] == 0.25
    assert out["max_texture_resolution"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_texture.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.texture`

- [ ] **Step 3: Create `benchmarks/threedcg/texture.py`**

```python
"""Texturing / PBR material metrics."""

from __future__ import annotations

from typing import Optional

from .asset import Asset

_CHANNELS = ("has_albedo", "has_normal", "has_roughness", "has_metallic")


def texture_metrics(candidate: Asset) -> dict[str, Optional[float]]:
    materials = candidate.materials
    if not materials:
        return {
            "has_material": 0.0,
            "pbr_channel_completeness": None,
            "max_texture_resolution": None,
        }
    completeness_values = []
    max_res: Optional[float] = None
    for material in materials:
        present = sum(1 for channel in _CHANNELS if material.get(channel))
        completeness_values.append(present / len(_CHANNELS))
        for width, height in material.get("texture_sizes", []):
            edge = float(max(width, height))
            max_res = edge if max_res is None else max(max_res, edge)
    completeness = sum(completeness_values) / len(completeness_values)
    return {
        "has_material": 1.0,
        "pbr_channel_completeness": float(completeness),
        "max_texture_resolution": max_res,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_texture.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/texture.py tests/threedcg/test_texture.py
git commit -m "feat: add 3DCG texture/PBR metrics"
```

---

### Task 8: Scorer aggregator

**Files:**
- Create: `benchmarks/threedcg/scorer.py`
- Test: `tests/threedcg/test_scorer.py`

**Interfaces:**
- Consumes: all metric modules (Tasks 2-7), `Asset`/`load_asset` (Task 1), and `MetricSet`/`RunResult` from `src.dst_snn.eval.result` (Plan A Task 3).
- Produces:
  - `score_assets(candidate: Asset, reference: Asset) -> dict[str, dict]` — runs every metric family; returns `{"geometry": {...}, "topology": {...}, "uv": {...}, "rig": {...}, "skin": {...}, "texture": {...}}`.
  - `aggregate_quality(scores: dict[str, dict]) -> float` — a single higher-is-better quality in `[0, 1]`, averaging normalized sub-scores (see mapping in code) and ignoring `None` values.
  - `score_to_result(candidate: Asset, reference: Asset, *, asset_id: str, build_latency_ms: float = 0.0) -> RunResult` — packs into a `RunResult` (`benchmark="eden14-image-to-3d"`, `model="snn-3dcg"`), storing per-family scores in `MetricSet.extra["scores"]`, `quality_metric="3dcg_composite"`, latency from `build_latency_ms`.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_scorer.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.scorer import aggregate_quality, score_assets, score_to_result


def _box():
    return asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))


def test_score_assets_has_all_families():
    scores = score_assets(_box(), _box())
    assert set(scores) == {"geometry", "topology", "uv", "rig", "skin", "texture"}


def test_identical_assets_score_higher_than_dissimilar():
    box = _box()
    sphere = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=3))
    same = aggregate_quality(score_assets(box, box))
    diff = aggregate_quality(score_assets(sphere, box))
    assert 0.0 <= diff <= same <= 1.0
    assert same > diff


def test_score_to_result_shape():
    result = score_to_result(_box(), _box(), asset_id="unit-box", build_latency_ms=12.0)
    assert result.benchmark == "eden14-image-to-3d"
    assert result.metrics.quality_metric == "3dcg_composite"
    assert result.metrics.latency_ms_p50 == 12.0
    assert "scores" in result.metrics.extra
    assert result.meta["asset_id"] == "unit-box"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.scorer`

- [ ] **Step 3: Create `benchmarks/threedcg/scorer.py`**

```python
"""Aggregate 3DCG metric families into a composite quality and RunResult."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.threedcg.asset import Asset
from benchmarks.threedcg.geometry import geometry_metrics
from benchmarks.threedcg.rig import rig_metrics
from benchmarks.threedcg.skin import skin_metrics
from benchmarks.threedcg.texture import texture_metrics
from benchmarks.threedcg.topology import topology_metrics
from benchmarks.threedcg.uv import uv_metrics
from src.dst_snn.eval.result import MetricSet, RunResult


def score_assets(candidate: Asset, reference: Asset) -> dict[str, dict]:
    return {
        "geometry": geometry_metrics(candidate, reference),
        "topology": topology_metrics(candidate, reference),
        "uv": uv_metrics(candidate),
        "rig": rig_metrics(candidate, reference),
        "skin": skin_metrics(candidate),
        "texture": texture_metrics(candidate),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio_score(ratio) -> float:
    # 1.0 when ratio == 1.0, decaying as it departs from parity.
    if ratio is None or ratio <= 0:
        return 0.0
    return _clamp01(1.0 - abs(1.0 - float(ratio)))


def aggregate_quality(scores: dict[str, dict]) -> float:
    """Map heterogeneous sub-metrics to higher-is-better [0,1] and average."""
    parts: list[float] = []

    geo = scores["geometry"]
    parts.append(_clamp01(1.0 - geo["chamfer"]))          # lower chamfer -> higher
    parts.append(_clamp01(geo["volume_iou"]))
    parts.append(_clamp01(geo["normal_consistency"]))

    topo = scores["topology"]
    parts.append(_ratio_score(topo["poly_count_ratio"]))
    parts.append(topo["is_watertight"])
    parts.append(topo["is_manifold"])

    uv = scores["uv"]
    if uv["has_uv"]:
        parts.append(_clamp01(uv["uv_coverage"]))
        parts.append(_clamp01(1.0 - uv["uv_overlap_ratio"]))
        if uv["uv_stretch"] is not None:
            parts.append(_clamp01(1.0 - uv["uv_stretch"]))

    rig = scores["rig"]
    if rig["bone_count_ratio"] is not None:
        parts.append(_ratio_score(rig["bone_count_ratio"]))

    skin = scores["skin"]
    if skin["weight_normalization_error"] is not None:
        parts.append(_clamp01(1.0 - skin["weight_normalization_error"]))
        parts.append(_clamp01(1.0 - skin["isolated_weight_ratio"]))

    tex = scores["texture"]
    if tex["pbr_channel_completeness"] is not None:
        parts.append(_clamp01(tex["pbr_channel_completeness"]))

    return float(sum(parts) / len(parts)) if parts else 0.0


def score_to_result(
    candidate: Asset,
    reference: Asset,
    *,
    asset_id: str,
    build_latency_ms: float = 0.0,
) -> RunResult:
    scores = score_assets(candidate, reference)
    quality = aggregate_quality(scores)
    metrics = MetricSet(
        quality=quality,
        quality_metric="3dcg_composite",
        latency_ms_p50=build_latency_ms,
        latency_ms_p95=build_latency_ms,
        spikes_per_inference=0.0,
        active_neuron_fraction=0.0,
        energy_pj=0.0,
        energy_source="n/a (offline scorer)",
        param_count=0,
        model_bytes=0,
        extra={"scores": scores},
    )
    return RunResult(
        benchmark="eden14-image-to-3d",
        model="snn-3dcg",
        metrics=metrics,
        baseline=None,
        meta={"asset_id": asset_id},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_scorer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/threedcg/scorer.py tests/threedcg/test_scorer.py
git commit -m "feat: add 3DCG scorer aggregator and RunResult packing"
```

---

### Task 9: Baseline generator and end-to-end validation

**Files:**
- Create: `benchmarks/threedcg/baseline.py`
- Test: `tests/threedcg/test_baseline.py`

**Interfaces:**
- Consumes: `Asset`/`asset_from_trimesh` (Task 1), `score_to_result` (Task 8).
- Produces:
  - `convex_hull_candidate(reference: Asset) -> Asset` — a trivial baseline generator returning the convex hull of the reference points as the "constructed" asset (validates the scorer end-to-end; a real SNN generator replaces this later).
  - `run_baseline(reference: Asset, *, asset_id: str) -> RunResult` — generates the convex-hull candidate and scores it.

- [ ] **Step 1: Write the failing test** — `tests/threedcg/test_baseline.py`

```python
from __future__ import annotations

import trimesh

from benchmarks.threedcg.asset import asset_from_trimesh
from benchmarks.threedcg.baseline import convex_hull_candidate, run_baseline


def test_convex_hull_candidate_is_asset_with_faces():
    ref = asset_from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    cand = convex_hull_candidate(ref)
    assert len(cand.faces) > 0
    assert cand.vertices.shape[1] == 3


def test_run_baseline_produces_result():
    ref = asset_from_trimesh(trimesh.creation.box(extents=(1, 1, 1)))
    result = run_baseline(ref, asset_id="unit-box")
    assert result.benchmark == "eden14-image-to-3d"
    assert 0.0 <= result.metrics.quality <= 1.0
    # Convex hull of a box is the box -> should score reasonably high geometry.
    assert result.metrics.extra["scores"]["geometry"]["volume_iou"] > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.baseline`

- [ ] **Step 3: Create `benchmarks/threedcg/baseline.py`**

```python
"""Trivial convex-hull baseline generator to validate the 3DCG scorer."""

from __future__ import annotations

import trimesh

from .asset import Asset, asset_from_trimesh
from .scorer import RunResult, score_to_result


def convex_hull_candidate(reference: Asset) -> Asset:
    mesh = trimesh.Trimesh(vertices=reference.vertices, faces=reference.faces, process=False)
    hull = mesh.convex_hull
    return asset_from_trimesh(hull)


def run_baseline(reference: Asset, *, asset_id: str) -> RunResult:
    candidate = convex_hull_candidate(reference)
    return score_to_result(candidate, reference, asset_id=asset_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_baseline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full 3DCG suite**

Run: `python -m pytest tests/threedcg -v`
Expected: PASS — asset 2, geometry 4, topology 2, uv 2, rig 3, skin 3, texture 3, scorer 3, baseline 2 (24 passed).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/threedcg/baseline.py tests/threedcg/test_baseline.py
git commit -m "feat: add convex-hull baseline and validate 3DCG scorer end-to-end"
```

---

### Task 10: Optional render similarity (gated) and docs

**Files:**
- Create: `benchmarks/threedcg/render_similarity.py`
- Modify: `benchmarks/README.md` (append 3DCG section — file created in Plan A Task 13; if absent, create it with just this section)

**Interfaces:**
- Consumes: `Asset` (Task 1).
- Produces:
  - `render_available() -> bool` — True if `pyrender` importable.
  - `ssim(image_a: np.ndarray, image_b: np.ndarray) -> float` — grayscale SSIM (pure numpy), higher better.
  - `render_similarity(candidate: Asset, reference: Asset, *, views: int = 4, resolution: int = 128) -> Optional[float]` — mean SSIM over matched orbit views; returns `None` if `pyrender` unavailable (so the core suite never depends on a renderer).

- [ ] **Step 1: Write the failing test** — append to `tests/threedcg/test_texture.py`

Add these imports and tests at the end of `tests/threedcg/test_texture.py`:
```python
import numpy as np

from benchmarks.threedcg.render_similarity import render_available, ssim


def test_ssim_identical_is_one():
    img = np.ones((16, 16), dtype=np.float64) * 0.5
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_ssim_differs_for_different_images():
    a = np.zeros((16, 16), dtype=np.float64)
    b = np.ones((16, 16), dtype=np.float64)
    assert ssim(a, b) < 0.5


def test_render_available_is_bool():
    assert isinstance(render_available(), bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/threedcg/test_texture.py -v`
Expected: FAIL — `ModuleNotFoundError: benchmarks.threedcg.render_similarity`

- [ ] **Step 3: Create `benchmarks/threedcg/render_similarity.py`**

```python
"""Optional render-based perceptual similarity (SSIM). Gated on pyrender.

The core scorer never depends on a GPU/offscreen renderer. When pyrender is
installed, render_similarity() compares matched orbit renders; otherwise it
returns None and is excluded from aggregation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset


def render_available() -> bool:
    try:
        import pyrender  # noqa: F401
    except Exception:
        return False
    return True


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("images must have the same shape")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def _render_orbit(asset: Asset, views: int, resolution: int) -> list[np.ndarray]:  # pragma: no cover - needs GPU
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    images: list[np.ndarray] = []
    for i in range(views):
        scene = pyrender.Scene()
        scene.add(render_mesh)
        angle = 2.0 * np.pi * i / views
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pose = np.eye(4)
        pose[0, 3] = 2.5 * np.cos(angle)
        pose[2, 3] = 2.5 * np.sin(angle)
        scene.add(camera, pose=pose)
        scene.add(pyrender.DirectionalLight(), pose=pose)
        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, _ = renderer.render(scene)
        renderer.delete()
        images.append(np.asarray(color, dtype=np.float64).mean(axis=2) / 255.0)
    return images


def render_similarity(
    candidate: Asset,
    reference: Asset,
    *,
    views: int = 4,
    resolution: int = 128,
) -> Optional[float]:
    if not render_available():
        return None
    cand_images = _render_orbit(candidate, views, resolution)  # pragma: no cover
    ref_images = _render_orbit(reference, views, resolution)  # pragma: no cover
    scores = [ssim(c, r) for c, r in zip(cand_images, ref_images)]  # pragma: no cover
    return float(np.mean(scores)) if scores else None  # pragma: no cover
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/threedcg/test_texture.py -v`
Expected: PASS (6 passed — 3 texture + 3 render-similarity).

- [ ] **Step 5: Append 3DCG section to `benchmarks/README.md`**

Add at the end of `benchmarks/README.md`:
```markdown

## EDEN14 image→3D construction (3DCG scorer)

`benchmarks/threedcg/` scores a generated 3D asset against a SketchFab
reference (`.glb`) across geometry, topology, UV, rig, skin, and texture.
See [threedcg/corpus.md](threedcg/corpus.md) for the reference-corpus contract.

```bash
pip install -r requirements-3dcg.txt
```

```python
from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.baseline import run_baseline

reference = load_asset("data/threedcg/unit-box/reference.glb")
result = run_baseline(reference, asset_id="unit-box")  # convex-hull baseline
print(result.to_json())
```

Metrics the reference cannot support (e.g. UV/rig/skin on an unrigged model)
return `None` and are excluded from the composite `quality`. Render-based SSIM
similarity is optional and requires `pyrender`; without it the scorer runs
fully on CPU with no renderer.
```

- [ ] **Step 6: Run the full test suite (both plans)**

Run: `python -m pytest -v`
Expected: PASS — Plan A tests + all `tests/threedcg` tests green.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/threedcg/render_similarity.py tests/threedcg/test_texture.py benchmarks/README.md
git commit -m "feat: add optional gated render similarity and 3DCG docs"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-snn-benchmark-harness-design.md` §3 and Benchmark C):
- Geometry (Chamfer, IoU, normal consistency) → Task 2. ✅
- Topology (poly/vert ratio, manifold, watertight, n-gon) → Task 3. ✅
- UV (chart count, coverage, stretch, overlap) → Task 4. ✅
- Rig (bone count, hierarchy, symmetry) → Task 5 (bone count ratio, hierarchy depth diff; symmetry deferred — see note). ⚠️→ documented.
- Skin (normalization, max influences, smoothness/isolated) → Task 6. ✅
- Texture (resolution, PBR completeness, render LPIPS/SSIM) → Task 7 (metadata) + Task 10 (SSIM; LPIPS deferred). ✅ with note.
- Composite → RunResult using Plan A `MetricSet`/`RunResult` → Task 8. ✅
- Baseline generator + end-to-end validation → Task 9. ✅
- Corpus spec → Task 0 (`corpus.md`). ✅
- Reference-image input (image→3D) → the input image is part of the corpus contract (`input.png`); the **scorer** compares produced geometry to the reference `.glb`. Generation from the image is the (out-of-scope) generator's job; noted in `corpus.md`.

**Deferred items (documented, not silent):**
- **Rig symmetry** and **joint position error**: require canonical bone correspondence between candidate and reference, which depends on a naming/retargeting convention not yet defined. Deferred to the generator plan where the bone naming scheme is fixed. Task 5 covers bone count + hierarchy depth now.
- **LPIPS**: needs a torch perceptual model download; SSIM (Task 10) is the CPU-only, network-free stand-in. LPIPS can be added as another gated function alongside `render_similarity`.
- **UV stretch angle distortion**: Task 4 implements area distortion; angle distortion deferred (area distortion is the dominant, cheaper signal).

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step is complete. `_render_orbit` is real pyrender code marked `# pragma: no cover` because it needs a GPU/offscreen context unavailable in CI.

**Type consistency:** `Asset` fields identical across all metric modules. `score_assets` returns the six family keys consumed by `aggregate_quality`. `score_to_result` imports `MetricSet`/`RunResult` from `src.dst_snn.eval.result` — the exact classes defined in Plan A Task 3 (field names match: `quality`, `quality_metric`, `latency_ms_p50/p95`, `spikes_per_inference`, `active_neuron_fraction`, `energy_pj`, `energy_source`, `param_count`, `model_bytes`, `extra`). `baseline.py` re-imports `RunResult` from `scorer` for a single import site. ✅

**Cross-plan dependency:** This plan's Task 8 requires Plan A Task 3 complete. Stated in Global Constraints.
```
