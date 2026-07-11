# SketchFab Reference Corpus

The 3DCG construction benchmark scores generated assets against finished
SketchFab reference models. This file defines the corpus contract.

## Format
- Each reference is a `.glb` (binary glTF) file. SketchFab "Download 3D Model"
  to glTF export produces this.
- Each reference pairs with one or more reference render images (`.png`) used
  as the model's input (image to 3D task) and for optional render similarity.

## Layout
```text
data/threedcg/<asset_id>/
  reference.glb
  input.png
  meta.json
```

`meta.json` contains fields such as `{"license": "...", "category": "...", "rigged": true}`.

## Offline smoke entry

Generate a synthetic, network-free multi-asset catalog (not SketchFab content):

```bash
# Single unit box (legacy):
python scripts/build_threedcg_unit_corpus.py

# Full synthetic catalog (rigid / organic / hard-surface / character / foliage + families):
python scripts/build_threedcg_corpus.py
```

This writes `data/threedcg/<asset_id>/{reference.glb,input.png,meta.json}` plus
`data/threedcg/catalog.json` for local scorer + generator validation. Note:
`data/` may be gitignored; regenerate after clone.

## External mesh corpus (A)

Import **local** licensed packs (SketchFab glTF, ShapeNet mirrors, internal
assets). Nothing is downloaded automatically.

```bash
# Flat / nested folder of .glb/.obj/.ply/.stl
python scripts/import_threedcg_external.py --src /path/to/meshes --max 200

# ShapeNetCore-like tree (*/models/model_normalized.obj)
python scripts/import_threedcg_external.py --src /data/ShapeNetCore.v2 --shapenet --max 500

# Rescan + rewrite catalog.json only
python scripts/import_threedcg_external.py --rebuild-catalog
```

Each import writes `reference.glb` (normalized), silhouette `input.png`, and
`meta.json` with `license`, `category`, `family`, `source`.

### Train on corpus

```bash
# 75% external corpus + 25% synthetic families (default when data/threedcg exists)
python scripts/train_threedcg_quality.py --track all --epochs 25 --n-samples 48 \
  --corpus-root data/threedcg --mix-synthetic 0.25

# Synthetic only
python scripts/train_threedcg_quality.py --no-corpus
```

### Export corpus references into EDEN

```bash
python scripts/export_threedcg_for_eden.py --from-corpus --max-corpus 24 --clear-generated
# or multi-family teacher/seq pack:
python scripts/export_threedcg_for_eden.py --diverse-pack --clear-generated
```

Drop **licensed** SketchFab (or other) references into the same layout when
available; set `meta.json` `license` accordingly.

### Built-in generators (not full SNN image→3D)

```bash
python benchmarks/threedcg/run_score.py \
  --reference data/threedcg/unit-box/reference.glb \
  --generator primitive_fit --asset-id unit-box
```

Kinds: `convex_hull`, `primitive_fit`, `voxel_occupancy`,
`track1_scripted`, `track2_occupancy` (see `benchmarks/threedcg/generator.py`
and `src/dst_snn/threedcg/`).

### SNN Track 1 / Track 2 (first increment)

```bash
# Track 1: image → spikes → mesh-op tokens → trimesh (bpy stand-in)
python benchmarks/threedcg/run_generate.py \
  --reference data/threedcg/unit-box/reference.glb \
  --image data/threedcg/unit-box/input.png \
  --track track1 --asset-id unit-box

# Track 2: image → spikes → occupancy grid → mesh
python benchmarks/threedcg/run_generate.py \
  --reference data/threedcg/unit-box/reference.glb \
  --synthetic --track track2 --asset-id unit-box
```

Modules: `src/dst_snn/threedcg/{image_spikes,ops,track1_policy,track2_occupancy,pipeline,bpy_adapter,mesh_backend,train}.py`.
Track1 MeshOps can execute via **trimesh** (default), **mock** (CI bpy mapping), or live **bpy** when Blender is installed.
Not SOTA image→3D; supervised heads + thin Blender adapter are in place; richer bpy ops later.

## Selection Criteria
- License: downloadable and redistribution-compatible, recorded in `meta.json`.
- Category balance: rigid props, organic characters, hard-surface, foliage.
- Rig coverage: at least 30% rigged and skinned to exercise rig/skin metrics.
- Poly-count spread: low-poly (<5k), mid (5k-50k), high (>50k).

## Applicability
Metrics whose data the reference lacks, such as UVs on a reference with no UVs
or rig/skin on an unrigged reference, return `None` and are excluded from the
aggregate `quality`.
