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
- Continued neuromorphic validation and accuracy work:
  - Added deterministic random/class-stratified smoke splitting for tonic datasets with cached `targets`; this fixed the misleading N-MNIST prefix split where the first test samples were all digit `0`.
  - Added classifier readout modes: `spike_count`, `max_membrane`, and `mean_membrane`.
  - Exposed benchmark tuning flags for threshold, branch count, delay, Chrono frontend, readout mode, and seed.
  - Added majority-class baseline metrics plus uniform chance accuracy to N-MNIST and DVS result JSON.
  - Stratified N-MNIST smoke:
    `python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 5 --limit-train 1024 --limit-test 512 --time-bins 12 --batch-size 64 --smoke-from-test --seed 7 --readout max_membrane --threshold 0.7 --out-dir artifacts/benchmarks/nmnist-smoke-stratified`
    Result: accuracy `0.501953125`, majority baseline `0.11328125`, uniform chance `0.1`, p50 latency `0.4359 ms`, p95 latency `0.4897 ms`, spikes/inference `11.47265625`, params `23402`.
  - DVS Gesture ablations remain weak:
    - No-Chrono spike-count setting, stratified 168/96, 5 epochs: accuracy `0.0625`, majority baseline `0.09375`.
    - Chrono max-membrane setting, stratified 168/96, 5 epochs: accuracy `0.09375`, majority baseline `0.09375`, output spikes `0.0`.
    Interpretation: current direct DST-SNN classifier/input representation does not yet learn DVS Gesture above baseline; next work should add a hidden spiking representation or a stronger temporal/event-feature frontend before claiming DVS accuracy.
- Latest full test suite after these changes:
  - `. .venv/bin/activate && python -m pytest -v` passed: 79 tests.
- Continued DVS accuracy work:
  - Added optional hidden DST-SNN layer to `SnnClassifier` via `hidden_features`.
  - Added `hidden_output` modes (`spikes` or `membrane`) so the output head can consume either hidden spikes or hidden membrane traces.
  - Added `hidden_threshold`, `hidden_features`, and `hidden_output` CLI flags to N-MNIST and DVS runners.
  - Fixed runner reproducibility by calling `torch.manual_seed(self.seed)` before model creation; split, DataLoader shuffle, and model initialization now share the benchmark seed.
  - Fixed `active_neuron_fraction` reporting in N-MNIST and DVS runners to use `spike_stats` instead of a placeholder `0.0`.
  - Seeded N-MNIST smoke after the reproducibility fix:
    `python benchmarks/neuromorphic/run_nmnist.py --root data/nmnist --epochs 5 --limit-train 1024 --limit-test 512 --time-bins 12 --batch-size 64 --smoke-from-test --seed 7 --readout max_membrane --threshold 0.7 --out-dir artifacts/benchmarks/nmnist-smoke-stratified-seeded`
    Result: accuracy `0.5078125`, majority baseline `0.11328125`, p50 latency `0.5987 ms`, p95 latency `0.8043 ms`, spikes/inference `8.109375`, active fraction `0.2285`.
  - DVS hidden-layer ablations:
    - `hidden_features=64`, thresholds `0.3`, hidden spikes to max-membrane readout, seed `13`: accuracy `0.1145833358168602` vs majority baseline `0.09375`.
    - Same, seed `42`: accuracy `0.125` vs majority baseline `0.09375`.
    - Lower-threshold fully spiking readout, seed `13`:
      `python benchmarks/neuromorphic/run_dvs_gesture.py --root data/dvs-gesture --epochs 5 --limit-train 168 --limit-test 96 --time-bins 16 --downsample 8 --batch-size 8 --smoke-from-test --seed 13 --no-chrono --threshold 0.1 --hidden-features 64 --hidden-threshold 0.1 --hidden-output spikes --readout spike_count --out-dir artifacts/benchmarks/dvs-smoke-hidden64-t01-spikes-seed13`
      Result: accuracy `0.1354166716337204`, majority baseline `0.09375`, p50 latency `0.2108 ms`, p95 latency `0.3191 ms`, decision-latency fraction `0.8730`, spikes/inference `173.6354`.
    Interpretation: hidden spiking layer provides the first above-baseline DVS smoke signal, but it is seed-sensitive and not strong enough for a stable benchmark claim.

Gap-fill implementation (2026-07-10 continued):
- glTF `skin_weights` extraction from `JOINTS_0`/`WEIGHTS_0` accessors in `benchmarks/threedcg/asset.py` (+ skinned GLB unit test).
- Dense MAC energy packing via `estimate_snn_classifier_ops` / `pack_snn_energy`; runners write `dense_energy_pj` and `energy_ratio_dense_over_snn`.
- Optional `--with-ann-baseline` (mean-pool MLP) under `src/dst_snn/eval/baselines/`.
- `render` family optionally included in `score_assets` / `aggregate_quality` when pyrender SSIM is available.
- Offline unit corpus builder `scripts/build_threedcg_unit_corpus.py` and CLI `benchmarks/threedcg/run_score.py`.
- Sensorimotor: `LocalMessageHub` + optional `serve_runtime` WebSocket, `WebcamSensor` (OpenCV or synthetic), fatigue dynamics on ticks.
- Root `README.md` Benchmarks section and expanded `benchmarks/README.md` energy docs.

Further gap-fill (continued):
- Temporal feature front-end (`causal EMA` + `Δt`) in `TemporalFeatureFrontEnd`, wired into `SnnClassifier` and both neuromorphic runners (`--use-temporal-features`).
- Multi-seed summarizer `benchmarks/neuromorphic/run_multi_seed.py`.
- Sensorimotor homeostasis controller, experience buffer, sleep-replay, representation stability; synthetic runner records them in `extra`.
- 3DCG: `weight_smoothness` (Laplacian) and `hierarchy_edit_distance` (parent-edge Jaccard).
- Optional `src/dst_snn/eval/powermetrics.py` (macOS, best-effort).

Homeostasis threshold wiring (completed):
- `ChronoPlasticLIFCell` / `ChronoPlasticLIFLayer` accept `threshold_offset` and pass it into `NoisySpikingActivation`.
- `HomeostasisController.tensor_offsets` / richer rate stats (`instant_rate`, deficit, min/max offset).
- `PredictiveWorldModel` + `train_world_model_step` apply previous-step offsets then update EMA rates.
- Synthetic sensorimotor runner uses wired homeostasis; extras include latent rates and `homeostasis_wired_to_threshold`.
- Tests in `tests/sensorimotor/test_homeostasis_threshold.py`.

DVS multi-seed + research-aligned Conv-PLIF:
- Dense dendritic sweep: best was B hidden64 **0.102±0.017** (1/5 seeds above majority) — seed-fragile.
- Literature (SEW-ResNet / SpikingJelly DVS): keep 2D event frames + Conv-BN-PLIF.
- Implemented `plif.py` + `conv_snn.py` (`ConvPLIFClassifier`), dataset `mode=frames`, runner `--backbone conv-plif`.
- Conv-PLIF multi-seed 168/96 × 5 seeds × 8 epochs: **0.333±0.020**, **5/5 above majority**, margin **+0.24**.
- First *stable* learner in this harness (not SOTA; full-train SEW-ResNet is ~97%).

Full-train pilot attempt + Frame-CNN baseline:
- `FrameCnnClassifier` matched to Conv-PLIF topology; `--with-ann-baseline` on `conv-plif` trains it.
- Official train download: figshare fails; Zenodo `ibmGestureTrain.tar.gz` (~2.3GB) started; not complete at pilot time.
- Pilot `scripts/run_dvs_fulltrain_pilot.py` fell back to **smoke-large** (200/64, 12 ep, seeds 0–2):
  - Conv-PLIF **0.380±0.053**, CNN **0.354±0.015**, both 3/3 above majority (0.094).
  - SNN slightly higher mean accuracy; energy proxy far lower for SNN.
- Report: `artifacts/benchmarks/dvs-fulltrain-pilot/pilot_report.md`.

Milestone freeze + Phase 0 closeout plan:
- Snapshot: `docs/superpowers/progress/2026-07-10-milestone-snapshot.md`
- Next plan: `docs/superpowers/plans/2026-07-10-phase0-closeout-dvs-training.md`
- Full-train DVS results frozen: Conv-PLIF **0.447±0.020**, SEW **0.490±0.020**, CNN-parity (see `artifacts/benchmarks/dvs-fulltrain-sew/`).
- Closeout implementation: shared `spatial_ops` MAC estimator, fair energy accounting string, cosine LR option on DVS runner.

Sensorimotor closed-loop claim metrics (post-closeout, 2026-07-10):
- True closed loop: `SyntheticSensor.apply_motor` + `MockActuator.on_command` so actions shift sensor phase.
- Representation probes: `src/dst_snn/sensorimotor/probe.py` (linear probe, nearest centroid, k-means purity).
- ANN predictive baseline: `DenseAnnPredictor` + `--with-ann-baseline` on synthetic runner.
- Fair energy packing on sensorimotor: SNN AC vs dense MAC proxy; ANN MAC energy in `baseline`.
- Runner extras: `closed_loop`, `phase_shift_range`, `linear_probe_accuracy`, `cluster_purity`, `energy_accounting`.
- Tests: probe, closed-loop sensor, ann predictor, runner with ANN baseline.

DVS controlled recipes (closeout plan Goal remainder, 2026-07-10):
- `benchmarks/neuromorphic/recipes.py`: named presets `parity-ds8`, `hires-ds4`, `hires-smoke`, `smoke-spatial`.
- Wired as `--recipe` on `run_dvs_gesture.py` and `run_multi_seed.py` (explicit CLI flags override).
- `hires-ds4` = higher spatial resolution preset (ds=4, cosine LR) from closeout plan goal.
- Energy model config-file override: `EnergyModel.from_json_file` / `from_mapping` (design §1).
- Tests: `tests/neuromorphic/test_recipes.py`, `tests/eval/test_energy_config.py`.

LLM baseline interface (Phase 0 design remainder, 2026-07-10):
- Optional eval interface (not product path): `src/dst_snn/eval/baselines/llm_backend.py`,
  `llm_classifier.py`, `benchmarks/neuromorphic/llm_baseline_util.py`.
- Scripted/majority offline backends for CI; opt-in `http` OpenAI-compatible chat.
- N-MNIST / DVS: `--with-llm-baseline`, `--llm-backend`, `--llm-max-samples`.
- Energy accounting `llm_api_external_v1` / `llm_token_proxy_v1` — not comparable to AC/MAC.
- Plan: `docs/superpowers/plans/2026-07-10-llm-baseline-interface.md`.
- Tests: `tests/eval/test_llm_baseline.py`.

Remainder closeout (2026-07-10 continued) — **completed**:
- Hires full-train freeze: `artifacts/benchmarks/dvs-hires-fulltrain/`
  - conv-plif hires-ds4 seeds 0–2: **SNN 0.537±0.039**, CNN mean 0.342, 3/3 > majority (wall ~39 min).
- HW bridges: `SerialMotorBridge` / `SerialTactileSensor` + `MockSerialPort` (tests offline; pyserial optional).
- 3DCG corpus: `scripts/build_threedcg_corpus.py` → 5 synthetic assets + `catalog.json`.
- 3DCG generators: `benchmarks/threedcg/generator.py` + `--generator` on `run_score.py`.
- EDEN bridge: `src/dst_snn/sensorimotor/eden_bridge.py` + `EDEN/src/snn/sensorimotorBridge.ts`.
- LLM multi-seed: `artifacts/benchmarks/llm-baseline-multiseed/`
  - scripted majority: 0.100±0.000 (seeds 0–2, n=32)
  - HTTP sample seed0 n=6: quality 0.167, p50 ~903 ms
- Plan: `docs/superpowers/plans/2026-07-10-remainder-closeout.md`.

External drop-ins only (layout/code ready, not repo content):
- Licensed SketchFab GLBs under `data/threedcg/<id>/`.
- Live serial devices.

Phase 1 image→3D Track 1/2 first increment (2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-snn-3dcg-track1-track2.md`
- Package `src/dst_snn/threedcg/`:
  - `image_spikes` — luminance + edge rate coding
  - `ops` — ADD_BOX/SPHERE/CYLINDER/TRANSLATE/SCALE/UNION/FINISH via trimesh
  - `track1_policy` — scripted + optional `Track1OpHead` (torch scaffold)
  - `track2_occupancy` — spikes → occupancy grid → box-soup mesh
  - `pipeline` — image → Asset → `RunResult`
- CLI: `benchmarks/threedcg/run_generate.py`; generator kinds on `run_score.py`
- Tests: `tests/threedcg/test_{image_spikes,ops,track1_policy,track2_occupancy,pipeline}.py`
- Remaining depth: live Blender verification for full armature bind; production SDF quality; EDEN bridge for generators.

Rich bpy MeshOps (2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-rich-bpy-meshops.md`
- Vocabulary: `EXTRUDE`, `SUBDIVIDE`, `BEVEL` (+ existing ADD/TRANSFORM/UNION)
- trimesh approximations offline; MockBlenderScene logs + mesh export; live `BpyScene` uses edit-mode ops
- Tests: ops volume/faces growth; mock op log

Rich ops + continuous SDF + sequences (2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-rich-ops-sdf-sequence.md`
- MeshOps: `ADD_ARMATURE`, `SMART_UV`, `ASSIGN_MATERIAL`, `AUTO_WEIGHTS` (+ prior EXTRUDE/SUBDIV/BEVEL)
- `BuildState` attaches bones/uv/materials/skin_weights onto `Asset`
- `sdf.py` continuous SDF grid + `Track2SdfHead` + train `--track track2_sdf`
- `sequence.py` teacher programs + `Track1SequenceHead` + train `--track track1_seq`
- Pipeline tracks: `track1_sequence`, `track2_sdf`
- Tests: `tests/threedcg/test_rich_ops_sdf_sequence.py`

Verification / tooling (continued):
- `scripts/eval_threedcg_generators.py` — train + held-out scorer quality report
  - Eval freeze: track2 0.62→0.68 trained; track2_sdf 0.71 mean; track1 scripted still tops unit-box (uses ref extents)
  - Report: `artifacts/threedcg/eval/report.md`
- `scripts/export_threedcg_for_eden.py` — GLB into `EDEN/public/generated/`
- `run_generate.py` supports track1_sequence / track2_sdf + default checkpoints

Quality closed-loop training (item 1, 2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-quality-closed-loop-training.md`
- `quality_loop.py`: soft Chamfer (diff) + scorer quality REINFORCE (discrete) + quality-gated occupancy BCE
- CLI: `scripts/train_threedcg_quality.py --track all`
- Checkpoints: `track1_quality.pt`, `track1_seq_quality.pt`, `track2_quality.pt`
- Smoke: seq quality **0.625→0.672**; track2 loss **0.098→0.038**; tests green
- Pipeline prefers `*_quality.pt` when present for trained/sequence tracks

EDEN auto-spawn generated GLBs (item 2, 2026-07-11):
- `export_threedcg_for_eden.py` writes/refreshes `EDEN/public/generated/manifest.json`
- `EDEN/src/snn/generatedAssets.ts` fetches manifest + ring layout
- `Game.tsx`: biotope ON → auto-spawn `/generated/*.glb` as world entities (glbUrl)
- Chat: `auto-spawned N generated GLB(s) from /generated`

Blender bpy adapter (recommended next, 2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-blender-bpy-adapter.md`
- `bpy_adapter.py` — MeshOp → SceneBackend; live `BpyScene` when `bpy` installed
- `MockBlenderScene` for CI; export real GLB via trimesh inside mock
- `mesh_backend.py` — `trimesh` | `bpy` | `auto` | `mock`
- Pipeline/CLI: `--backend`; default remains trimesh for CI
- Tests: `tests/threedcg/test_bpy_adapter.py`

3DCG supervised training (original plan return, 2026-07-11):
- Plan: `docs/superpowers/plans/2026-07-11-threedcg-supervised-training.md`
- `dataset.py` — synthetic silhouette images + occupancy labels offline
- `train.py` — Track1 CE+MSE, Track2 BCE occupancy; checkpoints `.pt` + `.json`
- CLI: `scripts/train_threedcg_generators.py --track both`
- Pipeline modes: `track1_trained` / `track2_trained` load checkpoints
- Tests: `tests/threedcg/test_train.py` (loss decreases; load + generate)

EDEN autonomous generated body + biotope (2026-07-11):
- Spec: `docs/superpowers/specs/2026-07-11-eden-autonomous-generated-body-design.md`
- SNN biotope **enabled by default**; procedural GLB body per creature (no setup UI)
- Modules: `proceduralBody.ts`, `bodyGlb.ts`, `generatedBodyRegistry.ts`
- `Game.tsx`: auto body on spawn, glbUrl on local render + WS entity, seed restore
- Next: Python ↔ EDEN mutual learning over the same glbUrl / sensor loop

Vision + morph + external construct (2026-07-11):
- Spec/plan: `docs/superpowers/specs/2026-07-11-vision-morph-construct-design.md`,
  `docs/superpowers/plans/2026-07-11-vision-morph-construct.md`
- Coarse vision from nearby entity size/shape (`visionShape.ts`)
- New neurons: vision width/height/depth/novelty, imitate_shape, construct_object
- Morph reward (match improvement) + construct reward; goal `imitateAndConstruct` default
- Auto world props (crate/pillar/boulder) for inspiration; SNN can spawn external builds
- Smoke: morph toward tall vision; ≥1 construct over 400 steps; tsc/build green
