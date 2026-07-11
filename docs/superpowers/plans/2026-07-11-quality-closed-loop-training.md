# Quality Closed-Loop Training Implementation Plan

**Status (2026-07-11):** Implemented.

**Goal:** Use scorer **quality** (and differentiable Chamfer proxy) as training signal so generators improve on what we actually measure—not only CE/MSE on synthetic labels.

**Architecture:**
1. **Differentiable proxy:** soft Chamfer between predicted primitive surface samples and reference samples (Track1 extents).
2. **Non-differentiable quality:** scorer composite; used via **REINFORCE** on discrete class / op sequences.
3. Hybrid loss = supervised + λ_proxy * chamfer + λ_rl * (−log π · advantage).

**Offline only; no network; no Blender required.**
