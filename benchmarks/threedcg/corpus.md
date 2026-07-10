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

Generate a synthetic, network-free unit box (not a SketchFab asset):

```bash
python scripts/build_threedcg_unit_corpus.py
```

This writes `data/threedcg/unit-box/{reference.glb,input.png,meta.json}` for
local scorer validation. Replace with licensed references under the same layout
for real evaluation. Note: `data/` is gitignored; regenerate after clone.

## Selection Criteria
- License: downloadable and redistribution-compatible, recorded in `meta.json`.
- Category balance: rigid props, organic characters, hard-surface, foliage.
- Rig coverage: at least 30% rigged and skinned to exercise rig/skin metrics.
- Poly-count spread: low-poly (<5k), mid (5k-50k), high (>50k).

## Applicability
Metrics whose data the reference lacks, such as UVs on a reference with no UVs
or rig/skin on an unrigged reference, return `None` and are excluded from the
aggregate `quality`.
