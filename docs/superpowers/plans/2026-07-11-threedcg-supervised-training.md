# 3DCG Track1/2 Supervised Training Implementation Plan

> Return to original design path: **learnable** image→3D generators (not EDEN body SNN).

**Status (2026-07-11):** Implemented.

**Goal:** Add offline supervised training so Track1 (primitive class + extents) and Track2 (occupancy) heads improve from synthetic image→reference pairs, with checkpoints loadable by the pipeline.

**Architecture:** Synthetic dataset from trimesh primitives → image_to_spikes → torch heads. Loss: CE on shape class + MSE extents (Track1); BCE occupancy (Track2). Checkpoints under `artifacts/threedcg/checkpoints/`.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, trimesh, pytest.
