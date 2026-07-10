# Remainder Closeout Implementation Plan

**Goal:** Complete the progress-log items left after Phase 0 + LLM baseline.

**Status (2026-07-10):** Implemented (code + offline freezes; physical SketchFab/HW optional).

## Tasks

- [x] Hires-ds4 multi-seed full-train script + freeze artifacts
- [x] Serial motor + tactile bridges with MockSerialPort tests
- [x] Multi-asset synthetic 3DCG corpus (licensed-ready layout)
- [x] Minimal 3DCG generators (`primitive_fit`, `voxel_occupancy`)
- [x] EDEN ↔ Python protocol bridge (Python + TS client)
- [x] LLM multi-seed freeze (scripted + HTTP sample when key present)

## Commands

```bash
python scripts/run_dvs_hires_fulltrain.py --backbone conv-plif --seeds 0,1,2
python scripts/build_threedcg_corpus.py
python scripts/run_llm_baseline_multiseed.py --seeds 0,1,2 --http --http-samples 6
python -m pytest -q
```
