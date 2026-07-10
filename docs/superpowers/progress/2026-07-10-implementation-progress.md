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

Verification:
- `python3 -m compileall -q src/dst_snn/eval benchmarks tests src/dst_snn/web_autonomous_learner.py` passed.
- `python3 - <<'PY' ... from src.dst_snn.eval.result import MetricSet, RunResult ... PY` passed without PyTorch installed.
- Created local `.venv` and installed `requirements-dst-snn.txt`, `requirements-bench.txt`, and `requirements-3dcg.txt`.
- `. .venv/bin/activate && python -m pytest -v` passed: 60 tests.
- Runner import/help checks passed:
  - `python benchmarks/neuromorphic/run_nmnist.py --help`
  - `python benchmarks/neuromorphic/run_dvs_gesture.py --help`

Not completed yet:
- Real N-MNIST/DVS runner smoke tests, which may download datasets through `tonic`.
- Real hardware/WebSocket transport for sensorimotor modules. Current increment is in-process and protocol-compatible.
