# Phase 0 Closeout + Fair Energy + DVS Training Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the Phase 0 milestone record, make SNN vs Frame-CNN energy comparison use the **same MAC accounting**, and add a small DVS training-recipe improvement (LR schedule + optional higher spatial resolution preset) so the next accuracy push is controlled rather than ad-hoc.

**Status (2026-07-10):** Implemented. Fair MAC via `spatial_ops`; cosine `--lr-schedule`; named recipes in `benchmarks/neuromorphic/recipes.py` including `hires-ds4` (downsample=4, cosine) and `parity-ds8` (freeze match); milestone docs written.

**Architecture:** Shared `compute_conv_mac_ops` helper feeds both SNN dense-proxy and Frame-CNN energy. DVS runner optionally uses cosine LR decay. Docs point at the milestone snapshot. No LLM baseline in this plan (separate later plan). No 3DCG generator (out of Phase 0).

**Tech Stack:** Python 3.9+, PyTorch ≥2.2, NumPy, pytest ≥8 (existing floors).

## Global Constraints

- Every new Python module starts with `from __future__ import annotations`.
- Torch imports use existing guard pattern where modules are torch-heavy.
- Tests MUST NOT access the network.
- Package imports from repo ROOT: `from src.dst_snn...` / `from benchmarks...`.
- Do not claim SOTA DVS accuracy; document vs Frame-CNN and majority only.
- Keep files focused; reuse existing `EnergyModel` / `dense_mac_energy_pj` / `snn_energy_pj`.

---

## File Structure

```
docs/superpowers/progress/2026-07-10-milestone-snapshot.md   # already written; link from progress
src/dst_snn/eval/baselines/spatial_ops.py                    # NEW: shared spatial MAC estimator
src/dst_snn/eval/baselines/frame_cnn.py                      # use spatial_ops
benchmarks/neuromorphic/run_dvs_gesture.py                   # fair energy + cosine LR
tests/eval/test_spatial_ops.py                               # NEW
docs/superpowers/progress/2026-07-10-implementation-progress.md
benchmarks/README.md
artifacts/benchmarks/dvs-fulltrain-sew/INTERPRETATION.md     # NEW short freeze note
```

---

### Task 1: Shared spatial MAC estimator

**Files:**
- Create: `src/dst_snn/eval/baselines/spatial_ops.py`
- Create: `tests/eval/test_spatial_ops.py`
- Modify: `src/dst_snn/eval/baselines/frame_cnn.py`
- Modify: `src/dst_snn/eval/baselines/__init__.py`

**Interfaces:**
- Produces:
  - `conv2d_mac_ops(in_c, out_c, k, h_out, w_out) -> float`
  - `estimate_three_stage_conv_macs(*, in_channels, channels: tuple[int,int,int], height, width, time_bins) -> float`
    — counts MACs for three 3×3 conv stages with strides (1,2,2) plus final linear `channels[-1] * num_classes` if `num_classes` given.
  - `FrameCnnClassifier.mac_ops_per_inference` delegates to this helper.

- [ ] **Step 1: Write failing tests** — `tests/eval/test_spatial_ops.py`

```python
from __future__ import annotations

from src.dst_snn.eval.baselines.spatial_ops import (
    conv2d_mac_ops,
    estimate_three_stage_conv_macs,
)
from src.dst_snn.eval.baselines.frame_cnn import FrameCnnClassifier


def test_conv2d_mac_ops():
    # 3x3, 2->4, 8x8 out
    assert conv2d_mac_ops(2, 4, 3, 8, 8) == 2 * 4 * 9 * 8 * 8


def test_three_stage_scales_with_time():
    one = estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=1, num_classes=11
    )
    four = estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=4, num_classes=11
    )
    assert four == 4 * one


def test_frame_cnn_matches_helper():
    model = FrameCnnClassifier(2, 11, channels=(8, 16, 16))
    assert model.mac_ops_per_inference(4, 16, 16) == estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=4, num_classes=11
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/eval/test_spatial_ops.py -v`  
Expected: FAIL — import error

- [ ] **Step 3: Implement `spatial_ops.py` and wire `FrameCnnClassifier`**

```python
# spatial_ops.py — pure functions, no torch required
from __future__ import annotations

def conv2d_mac_ops(in_c: int, out_c: int, kernel: int, h_out: int, w_out: int) -> float:
    return float(in_c * out_c * kernel * kernel * h_out * w_out)


def estimate_three_stage_conv_macs(
    *,
    in_channels: int,
    channels: tuple[int, int, int],
    height: int,
    width: int,
    time_bins: int,
    num_classes: int = 0,
) -> float:
    c1, c2, c3 = channels
    h1, w1 = height, width
    h2, w2 = max(1, height // 2), max(1, width // 2)
    h3, w3 = max(1, height // 4), max(1, width // 4)
    per_step = (
        conv2d_mac_ops(in_channels, c1, 3, h1, w1)
        + conv2d_mac_ops(c1, c2, 3, h2, w2)
        + conv2d_mac_ops(c2, c3, 3, h3, w3)
        + float(c3 * num_classes)
    )
    return per_step * float(time_bins)
```

Update `FrameCnnClassifier.mac_ops_per_inference` to call `estimate_three_stage_conv_macs`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/eval/test_spatial_ops.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (if user requested commits)

---

### Task 2: Fair energy in DVS runner for conv-plif / sew-plif

**Files:**
- Modify: `benchmarks/neuromorphic/run_dvs_gesture.py` (energy block for spatial backbones)

**Interfaces:**
- Consumes: `estimate_three_stage_conv_macs`, model channel widths, `spatial_hw`, `time_bins`
- Produces: `MetricSet.extra` keys:
  - `dense_mac_ops` — same formula as Frame-CNN for conv-plif topology (or sew approx documented)
  - `dense_energy_pj` — MAC energy for that dense path
  - `energy_accounting` — string `"shared_spatial_mac_proxy_v1"`
  - When CNN baseline present, `baseline.extra["energy_accounting"]` same string and same mac ops for matched topology

- [ ] **Step 1: Replace ad-hoc dense_macs for `conv-plif` with helper**

For `backbone == "conv-plif"`:
```python
h, w = self.spatial_hw
dense_macs = estimate_three_stage_conv_macs(
    in_channels=2,
    channels=self.plif_channels,
    height=h,
    width=w,
    time_bins=self.time_bins,
    num_classes=NUM_CLASSES,
)
```

For `sew-plif`, keep an explicit approximate function in the same module or add `estimate_sew_macs` in `spatial_ops.py` that counts stem + residual blocks consistently with `SewConvPLIFClassifier` topology (document in docstring).

- [ ] **Step 2: When training Frame-CNN baseline, require CNN MAC ops == SNN dense_mac_ops for conv-plif**

In baseline packing:
```python
assert abs(cnn_macs - dense_macs) < 1.0  # only for conv-plif matched channels
```
For sew-plif, store both `cnn_mac_ops` and `snn_dense_proxy_mac_ops` without forcing equality if topologies differ; set `energy_accounting` to `"sew_vs_shallow_cnn_proxy_v1"`.

- [ ] **Step 3: Smoke import**

Run: `python benchmarks/neuromorphic/run_dvs_gesture.py --help`  
Expected: exit 0

---

### Task 3: Cosine LR schedule option on DVS runner

**Files:**
- Modify: `benchmarks/neuromorphic/run_dvs_gesture.py`
- Test: lightweight unit test of schedule helper if extracted

**Interfaces:**
- Produces: `--lr-schedule {constant,cosine}` (default `constant` for back-compat)
- Training loop uses `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)` when cosine selected
- Record `extra["lr_schedule"]` and final LR in result

- [ ] **Step 1: Add argparse + runner field `lr_schedule: str = "constant"`**
- [ ] **Step 2: Wire scheduler in `run()` after optimizer creation**
- [ ] **Step 3: Step scheduler once per epoch end**
- [ ] **Step 4: pytest still green; help shows flag**

---

### Task 4: Milestone docs wiring

**Files:**
- Modify: `docs/superpowers/progress/2026-07-10-implementation-progress.md`
- Modify: `benchmarks/README.md`
- Create: `artifacts/benchmarks/dvs-fulltrain-sew/INTERPRETATION.md`

**Content for INTERPRETATION.md:**
- Point to full-train table
- Explicit claim language: "CNN-parity shallow SNN; not SOTA; energy proxy asymmetric until shared_spatial_mac_proxy_v1"
- Link to milestone snapshot

- [ ] **Step 1: Write INTERPRETATION.md**
- [ ] **Step 2: Update progress Not completed section**
- [ ] **Step 3: Link snapshot from benchmarks/README**

---

### Task 5: Full suite verification

- [ ] **Step 1: `python -m pytest -q`**
Expected: all pass

- [ ] **Step 2: Optional short smoke (if data present)**  
`python benchmarks/neuromorphic/run_dvs_gesture.py --backbone conv-plif --with-ann-baseline --smoke-from-test --limit-train 64 --limit-test 32 --epochs 1 --out-dir /tmp/dvs-fair-energy`  
Expected: result JSON has `energy_accounting` and baseline CNN quality key

---

## Self-Review

**Spec coverage:** Phase 0 energy “SNN AC vs dense MAC + source recorded” → Task 1–2. DVS runners → Task 2–3. Milestone freeze → Task 4. LLM baseline deferred (explicit). 3DCG generator out of scope.

**Placeholder scan:** None intentional.

**Type consistency:** `plif_channels` / `spatial_hw` / `NUM_CLASSES` match runner fields.
