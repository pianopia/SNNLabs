# Vision Shape Inspiration + Morph + External Construct Implementation Plan

**Status (2026-07-11):** Implemented (first increment)

**Goal:** Give EDEN embodied SNN (1) coarse visual shape sensing of nearby world objects, (2) reward to morph toward observed shapes, (3) motor + reward to place external objects inspired by vision — without heavy UI or expensive vision models.

**Architecture:**
- `visionShape.ts` — pure geometry features from entity size/shape (stand-in for camera vision)
- Extend `SnnEnvironmentStimulus` with shape channels
- New sensors/motors in `lif.ts` + morph pull + construct intent on creature
- `Game.tsx` fills vision from `players` map; executes construct as local/WS entity spawn

**Out of scope:** Real RGB CNN vision, Blender Track1, multi-agent construction markets.

## Tasks
- [x] visionShape helpers + tests (tsx smoke)
- [x] Stimulus + neurons + rewards in lif.ts
- [x] Game collect + construct spawn
- [x] Docs / tsc
