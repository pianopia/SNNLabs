# Implementation Progress - 2026-07-10

Scope read:
- `docs/superpowers/specs/2026-07-09-snn-benchmark-harness-design.md`
- `docs/superpowers/specs/2026-07-09-embodied-sensorimotor-runtime-design.md`
- `docs/superpowers/plans/2026-07-09-snn-eval-harness-neuromorphic.md`
- `docs/superpowers/plans/2026-07-09-snn-3dcg-scorer.md`

Progress:
- Completed Plan A Task 0 bootstrap files: `requirements-bench.txt`, `pytest.ini`, and test import setup.
- Completed Plan A Tasks 1-4 shared eval harness: energy model, metrics, result schema/report, and benchmark runner loop.
- Completed Plan A Tasks 5-6 web learner fixes: novelty is computed against the pre-observation vocabulary, and reward now modulates the SNN loss plus the optimizer step scale and is returned in training output.
- Completed Plan A Tasks 7-9 unit-tested neuromorphic pieces: event binning, spike tensor conversion, DST-SNN classifier wrapper, and decision-latency metric.
- Completed Plan A Tasks 10-12 implementation files: tonic dataset wrappers plus N-MNIST and DVS Gesture runner CLIs. Real dataset smoke runs are not executed yet.
- Completed Plan A Task 13 documentation equivalent in `benchmarks/README.md`.
- Completed Plan B Task 0 environment/corpus files.
- Completed Plan B Tasks 1-10 implementation and tests: asset loader, geometry/topology/UV/rig/skin/texture metrics, scorer aggregation, convex-hull baseline, optional gated SSIM render similarity, and benchmark docs.
- Adjusted `src/dst_snn.__init__` and `src/dst_snn.eval.__init__` to lazily import torch-heavy symbols so scorer-only code can import `MetricSet`/`RunResult` without requiring PyTorch.
- Completed first increment of the embodied sensorimotor runtime spec:
  - JSON message protocol.
  - Dynamic module registry with fixed feature/motor hashing.
  - Observation-to-spike encoder and motor-action decoder.
  - Predictive world model using `ChronoPlasticLIFLayer`.
  - Minimal in-process runtime loop.
  - Synthetic sensor and mock actuator modules.
- Extended sensorimotor runtime:
  - `global_signal` and `trace` messages emitted from runtime ticks.
  - JSONL `read_jsonl`/`write_jsonl`/`replay_jsonl` helpers for offline module streams.
  - Runtime `save`/`load` checkpointing for registry and loop state.
  - EMA learning-progress tracker and intrinsic reward metric for world-model training.
  - Usage notes in `docs/sensorimotor-runtime.md`.
- Added an offline synthetic sensorimotor benchmark runner:
  - `benchmarks/sensorimotor/run_synthetic_loop.py`.
  - Uses `SyntheticSensor`, `MockActuator`, `SensorimotorRuntime`, and `PredictiveWorldModel`.
  - Emits shared `RunResult` JSON plus benchmark report through `run_benchmarks`.
  - Quality metric is `prediction_loss_reduction`; extras include loss history and mean intrinsic reward.
- Added intrinsic-reward action policy and predictive-model checkpoints:
  - `src/dst_snn/sensorimotor/policy.py` (`IntrinsicMotorPolicy`).
  - `src/dst_snn/sensorimotor/checkpoint.py` for world-model + optimizer + learning-progress checkpoints.
  - Synthetic sensorimotor runner now records `policy_scores` and `selected_commands`.

Verification:
- `python3 -m compileall -q src/dst_snn/eval benchmarks tests src/dst_snn/web_autonomous_learner.py` passed.
- `python3 - <<'PY' ... from src.dst_snn.eval.result import MetricSet, RunResult ... PY` passed without PyTorch installed.
- Created local `.venv` and installed `requirements-dst-snn.txt`, `requirements-bench.txt`, and `requirements-3dcg.txt`.
- `. .venv/bin/activate && python -m pytest -v` passed: 67 tests.
- Runner import/help checks passed:
  - `python benchmarks/neuromorphic/run_nmnist.py --help`
  - `python benchmarks/neuromorphic/run_dvs_gesture.py --help`
  - `python benchmarks/sensorimotor/run_synthetic_loop.py --help`
- Synthetic sensorimotor runner smoke test passed:
  - `python benchmarks/sensorimotor/run_synthetic_loop.py --steps 4 --feature-size 24 --motor-size 8 --time-steps 4 --latent-size 8 --out-dir /tmp/snnlabs-sensorimotor-bench`
- Real-data neuromorphic smoke tests passed with `--smoke-from-test`:
  - N-MNIST command:
    `python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 1 --limit-train 64 --limit-test 64 --time-bins 8 --batch-size 16 --smoke-from-test --out-dir artifacts/benchmarks/nmnist-smoke`
  - N-MNIST result: accuracy `0.484375`, p50 latency `0.4874 ms`, p95 latency `0.8208 ms`, spikes/inference `5.921875`, params `23402`.
  - DVS Gesture command:
    `python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs-gesture --epochs 1 --limit-train 32 --limit-test 32 --time-bins 8 --downsample 8 --batch-size 8 --smoke-from-test --out-dir artifacts/benchmarks/dvs-smoke`
  - DVS Gesture result: accuracy `0.0625`, decision latency fraction `0.9140625`, p50 latency `0.1215 ms`, p95 latency `0.1307 ms`, spikes/inference `4.40625`, params `5915`.
  - DVS note: tonic's figshare URL returned an AWS WAF challenge; downloaded the md5-identical `ibmGestureTest.tar.gz` from Zenodo record `8060604`, then tonic verified and extracted it.

Not completed yet:
- Full train-split N-MNIST/DVS runs. Current real-data smoke tests use official test split subsets to avoid multi-GB first-run downloads.
- WebSocket transport and real hardware bridges for sensorimotor modules. Current increment is in-process plus JSONL replay and remains protocol-compatible.
