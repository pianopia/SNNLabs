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

For faster first-pass real-data smoke tests, use the official test split only
and split it locally into tiny train/eval subsets:

```bash
python benchmarks/neuromorphic/run_nmnist.py \
  --root data/nmnist --epochs 1 --limit-train 64 --limit-test 64 \
  --time-bins 8 --batch-size 16 --smoke-from-test \
  --out-dir artifacts/benchmarks/nmnist-smoke

python benchmarks/neuromorphic/run_dvs_gesture.py \
  --root data/dvs-gesture --epochs 1 --limit-train 32 --limit-test 32 \
  --time-bins 8 --downsample 8 --batch-size 8 --smoke-from-test \
  --out-dir artifacts/benchmarks/dvs-smoke
```

As of the July 2026 smoke run, the tonic figshare URL for DVS Gesture returned
an AWS WAF challenge to non-browser clients. The same preprocessed archive is
available from Zenodo record `8060604`; place `ibmGestureTest.tar.gz` under
`data/dvs-gesture/DVSGesture/` and tonic will verify its md5 before extracting.

## Synthetic Sensorimotor Loop

`benchmarks/sensorimotor/run_synthetic_loop.py` runs the in-process
sensorimotor runtime with a synthetic sensor, mock actuator, and predictive
world model. It emits the shared `RunResult` schema and requires no hardware or
dataset download.

```bash
pip install -r requirements-dst-snn.txt -r requirements-bench.txt
python benchmarks/sensorimotor/run_synthetic_loop.py --steps 32
```

The quality metric is `prediction_loss_reduction`, computed from the first and
last loss windows. Extra metrics include the loss history and mean intrinsic
reward from the EMA learning-progress tracker.

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
