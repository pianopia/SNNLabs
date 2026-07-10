# Benchmarks

# SNN Benchmarks

Shared evaluation harness (`src/dst_snn/eval/`) plus neuromorphic, 3DCG, and
sensorimotor runners. Every runner emits the same `RunResult` schema: quality,
latency (p50/p95), spikes-per-inference, energy (pJ, AC/MAC model), and model
size.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dst-snn.txt -r requirements-bench.txt
# Optional:
pip install -r requirements-3dcg.txt
pip install websockets  # sensorimotor WebSocket transport
```

## Energy model

`EnergyModel` defaults to 45nm (`0.9 pJ/MAC`, `0.1 pJ/AC`) and records its
`source` in every result. SNN energy ≈ spikes × effective fan-out × AC cost
(multi-layer models use a width-weighted fan-out). Dense baseline energy is the
MAC cost of same-width linear layers evaluated at every time bin.

Compare with `energy_ratio(snn_pj, dense_pj)` (also written as
`extra.energy_ratio_dense_over_snn` and `baseline.energy_pj`). Optional
`--with-ann-baseline` trains a small mean-pool MLP and replaces the quality
baseline with `ann_mlp_accuracy` while still reporting dense-MAC energy.

Wall-clock energy via macOS `powermetrics` is not wired yet (optional design
item).

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
  --root data/nmnist --epochs 5 --limit-train 1024 --limit-test 512 \
  --time-bins 12 --batch-size 64 --smoke-from-test --seed 7 \
  --readout max_membrane --threshold 0.7 \
  --out-dir artifacts/benchmarks/nmnist-smoke-stratified

python benchmarks/neuromorphic/run_dvs_gesture.py \
  --root data/dvs-gesture --epochs 5 --limit-train 168 --limit-test 96 \
  --time-bins 16 --downsample 8 --batch-size 8 --smoke-from-test --seed 7 \
  --threshold 0.5 --readout max_membrane --chrono-hidden 64 \
  --out-dir artifacts/benchmarks/dvs-smoke-stratified-chrono
```

Smoke mode uses a deterministic class-stratified split when tonic exposes
cached `targets`; this avoids misleading prefix-only subsets such as the
N-MNIST test split's initial run of digit `0` samples. Result JSON includes a
majority-class baseline and uniform chance accuracy for quick interpretation.

Current real-data smoke observations:

- N-MNIST, stratified 1024/512, 5 epochs: accuracy `0.502`, majority baseline
  `0.113`, p50 latency `0.436 ms`, spikes/inference `11.47`.
- N-MNIST after deterministic model seeding: accuracy `0.508`, majority
  baseline `0.113`, p50 latency `0.599 ms`, spikes/inference `8.11`.
- DVS Gesture direct classifier, stratified 168/96, 5 epochs: at or below the
  majority baseline (`0.094`).
- DVS Gesture with a hidden spiking layer shows a weak but real smoke signal in
  some deterministic seeds. Example:

```bash
python benchmarks/neuromorphic/run_dvs_gesture.py \
  --root data/dvs-gesture --epochs 5 --limit-train 168 --limit-test 96 \
  --time-bins 16 --downsample 8 --batch-size 8 --smoke-from-test --seed 13 \
  --no-chrono --threshold 0.1 --hidden-features 64 --hidden-threshold 0.1 \
  --hidden-output spikes --readout spike_count \
  --out-dir artifacts/benchmarks/dvs-smoke-hidden64-t01-spikes-seed13
```

This run reached accuracy `0.135` vs majority baseline `0.094`, with p50
latency `0.211 ms`, decision-latency fraction `0.873`, and
spikes/inference `173.64`. The result is still seed-sensitive and not yet a
stable benchmark claim.

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
(`.glb`) across geometry, topology, UV, rig, skin, texture, and optional render
SSIM. See [threedcg/corpus.md](threedcg/corpus.md) for the reference-corpus
contract. Real glTF skins populate `skin_weights` from `JOINTS_0`/`WEIGHTS_0`.

```bash
pip install -r requirements-3dcg.txt
# Offline synthetic unit corpus (no network):
python scripts/build_threedcg_unit_corpus.py
python benchmarks/threedcg/run_score.py \
  --reference data/threedcg/unit-box/reference.glb --convex-hull --asset-id unit-box
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
is gated on `pyrender` and, when available, is included under
`extra.scores.render` and the aggregate quality.

## Multi-seed summary

```bash
# Single configuration, multiple seeds
python benchmarks/neuromorphic/run_multi_seed.py \
  --benchmark dvs-gesture --seeds 0,1,2,13,42 --smoke-from-test \
  --limit-train 96 --limit-test 64 --epochs 5 \
  --hidden-features 64 --threshold 0.1 --no-chrono --readout spike_count

# Fixed 4-config DVS sweep + comparison table
python scripts/run_dvs_multi_seed_sweep.py
# → artifacts/benchmarks/dvs-multi-seed/comparison.md
```

Writes per-seed JSON plus `summary.json` / `summary.md` (`quality_mean` /
`quality_std`, seeds above majority) and `report.md`.

### Latest DVS smoke multi-seed (2026-07-10)

Protocol: stratified smoke-from-test, 5 seeds, downsample 8. Majority ≈ 0.094.

| config | mean±std | above maj | note |
|---|---:|---:|---|
| A direct dendritic | 0.088±0.013 | 1/5 | flat features |
| B hidden64 low thr | 0.102±0.017 | 1/5 | best dense; seed-fragile |
| C temporal+hidden | 0.094±0.000 | 0/5 | majority collapse |
| D chrono+temporal | 0.094±0.000 | 0/5 | zero spikes |
| **E Conv-PLIF (spatial)** | **0.333±0.020** | **5/5** | research-aligned win |

**Conclusion:** keep event **spatial structure** + **Conv-BN-PLIF** (Fang /
SEW-ResNet style). Flat dendritic classifiers were the main bottleneck.

```bash
python benchmarks/neuromorphic/run_dvs_gesture.py \
  --backbone conv-plif --smoke-from-test \
  --limit-train 168 --limit-test 96 --epochs 8 \
  --time-bins 16 --downsample 8 --batch-size 8 --threshold 1.0
```

See `artifacts/benchmarks/dvs-multi-seed/comparison.md`.

### Conv-PLIF vs matched Frame-CNN (smoke-large pilot)

```bash
python scripts/run_dvs_fulltrain_pilot.py
# needs ibmGestureTrain.tar.gz for full-train; else uses large smoke-from-test
```

| model | mean acc (3 seeds) | vs majority |
|---|---:|---|
| Conv-PLIF SNN | **0.380±0.053** | 3/3 above |
| Frame-CNN (same width) | 0.354±0.015 | 3/3 above |

Details: `artifacts/benchmarks/dvs-fulltrain-pilot/pilot_report.md`.

## Temporal feature front-end

`--use-temporal-features` stacks raw spikes with a causal EMA rate and
temporal difference, optionally projected with `--temporal-project-to`. Useful
for DVS-style temporal structure before Chrono/hidden layers.

## Optional wall-clock energy (macOS)

`src/dst_snn/eval/powermetrics.py` wraps macOS `powermetrics` when available
(usually needs root). Unit tests only parse sample text; runners do not require
it.

## Results

Runners write `<name>.json` and a combined `report.md` under
`artifacts/benchmarks/` (or a path you pass via `--out-dir`).
