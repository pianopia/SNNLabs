# LLM Baseline Interface Implementation Plan

> **For agentic workers:** Steps use checkbox syntax. Offline tests must not use the network.

**Goal:** Complete the Phase 0 design item “対 LLM ベースライン” as an **optional eval interface** (not product path): rasterized/summarized event frames → LLM class id, recorded in the shared harness schema with explicit non-AC/MAC energy accounting.

**Status (2026-07-10):** Implemented.

**Architecture:**
- `LlmBackend` protocol + `ScriptedLlmBackend` (CI) + `HttpChatLlmBackend` (opt-in OpenAI-compatible).
- Compact numeric sample summaries (flat spikes or event frames) → text prompt → parse class id.
- N-MNIST / DVS runners: `--with-llm-baseline`, `--llm-backend`, `--llm-max-samples`.
- LLM metrics always in `metrics.extra.llm_baseline`; fill `baseline` when no ANN/CNN.

**Tech Stack:** Python 3.9+, stdlib `urllib` for HTTP (no new deps), pytest.

## Tasks

- [x] Task 1: `llm_backend.py` + `llm_classifier.py` + package exports
- [x] Task 2: Offline unit tests (`tests/eval/test_llm_baseline.py`)
- [x] Task 3: Wire DVS + N-MNIST runners + `llm_baseline_util.py`
- [x] Task 4: Docs (`benchmarks/README.md`, progress, this plan)

## Out of scope

- 3DCG LLM→Blender agent (design Track C; separate plan)
- Claiming LLM quality numbers without a real API multi-seed run
- Product routing / LLM-as-core
