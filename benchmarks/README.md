# Benchmarks

## Neuromorphic Classification

`benchmarks/neuromorphic/` converts event-camera datasets into
`[time, features]` spike tensors and wraps the PyTorch DST-SNN as a spike-count
classifier.

```bash
pip install -r requirements-dst-snn.txt -r requirements-bench.txt
python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 1 --limit-train 128 --limit-test 128
python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs-gesture --epochs 1 --limit-train 128 --limit-test 128
```

The runner scripts may download datasets through `tonic`; unit tests use
synthetic events only and do not access the network.

## EDEN14 Image To 3D Construction

`benchmarks/threedcg/` scores a generated 3D asset against a SketchFab reference
(`.glb`) across geometry, topology, UV, rig, skin, and texture. See
[threedcg/corpus.md](threedcg/corpus.md) for the reference-corpus contract.

```bash
pip install -r requirements-3dcg.txt
```

```python
from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.baseline import run_baseline

reference = load_asset("data/threedcg/unit-box/reference.glb")
result = run_baseline(reference, asset_id="unit-box")
print(result.to_json())
```

Metrics the reference cannot support, such as UV/rig/skin on an unrigged model,
return `None` and are excluded from the composite `quality`. Render-based SSIM
similarity is optional and requires `pyrender`; without it the scorer runs on
CPU with no renderer.
