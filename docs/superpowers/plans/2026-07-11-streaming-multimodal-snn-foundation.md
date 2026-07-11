# Streaming Multimodal SNN Foundation Implementation Plan

**Status (2026-07-11):** Phases 1–2 implemented; later phases planned.

**Design:** `docs/superpowers/specs/2026-07-11-streaming-multimodal-snn-foundation-design.md`

## Phase 1 — representation and runtime contract

- [x] Signed multi-level streaming event encoder with residual state
- [x] Modality-specific temporal scales and temporal compression
- [x] Aligned multimodal event fusion with stable modality slices
- [x] Constant-memory streaming spiking SSM reference runtime
- [x] Stable confidence-based early exit
- [x] Comparable sparsity/AC/MAC/state-memory report and offline smoke CLI
- [x] Unit tests

## Phase 2 — trainable backbone and teacher bridge

- [x] Torch signed-integer spiking SSM block with surrogate gradient
- [x] Teacher adapter protocol and cached intermediate-feature dataset
- [x] Block-wise ANN→SNN replacement trainer
- [x] Losses: task, feature alignment, spike budget, early-exit calibration
- [x] Text next-token and image-text retrieval minimum benchmarks

Phase 2 synthetic smoke freeze (`seed=0`):

- text next-token: student accuracy `0.000 → 1.000`, teacher `1.000`, event rate `0.628`
- image-text retrieval R@1: student `0.000 → 1.000`, teacher `1.000`, event rate `0.475`
- artifact: `artifacts/benchmarks/foundation-phase2-smoke.json`

These deliberately small synthetic tasks validate learning and distillation
mechanics only. They are not evidence of general language or vision capability.

## Phase 3 — unified multimodal capability

- [ ] Tokenizers/adapters for text, image, audio, video, 3D and sensor/action
- [ ] Sparse modality router and specialist experts
- [ ] Text/audio/image/3D generation heads with per-head parity gates
- [ ] External episodic/semantic memory and retrieval
- [ ] EDEN online sensor/action integration using the shared event contract

## Phase 4 — deployment co-design

- [ ] Dense CPU/GPU sparse-kernel baseline with real RSS/VRAM/TTFT metrics
- [ ] Neuromorphic/custom accelerator backend
- [ ] Wall-power measurement and joules-per-successful-task harness
- [ ] Quantization, pruning, speculative/early-exit decoding
- [ ] Reproducible capability/efficiency report against matched local MLLMs

## Non-negotiable completion rule

The project may claim that it exceeds an ANN/CNN/MLLM only when quality,
latency, memory and measured energy all pass on the same task set and hardware.
Estimated AC/MAC savings alone are insufficient.
