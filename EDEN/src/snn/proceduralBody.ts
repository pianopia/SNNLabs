/**
 * Deterministic procedural creature body from a seed (no UI params).
 * Produces a simple multi-part mesh (torso + head + limbs) as positions/indices.
 */

import { buildGlb, meshToObjectUrl, type MeshGeometry } from './bodyGlb';

export type GeneratedBodyMeta = {
  seed: number;
  bounds: [number, number, number];
  collisionRadius: number;
  createdAt: number;
};

type BoxSpec = {
  cx: number;
  cy: number;
  cz: number;
  sx: number;
  sy: number;
  sz: number;
};

const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

const pushBox = (positions: number[], indices: number[], box: BoxSpec) => {
  const { cx, cy, cz, sx, sy, sz } = box;
  const hx = sx / 2;
  const hy = sy / 2;
  const hz = sz / 2;
  const base = positions.length / 3;
  const corners: Array<[number, number, number]> = [
    [cx - hx, cy - hy, cz - hz],
    [cx + hx, cy - hy, cz - hz],
    [cx + hx, cy + hy, cz - hz],
    [cx - hx, cy + hy, cz - hz],
    [cx - hx, cy - hy, cz + hz],
    [cx + hx, cy - hy, cz + hz],
    [cx + hx, cy + hy, cz + hz],
    [cx - hx, cy + hy, cz + hz],
  ];
  for (const [x, y, z] of corners) {
    positions.push(x, y, z);
  }
  // 12 triangles
  const faces = [
    0, 1, 2, 0, 2, 3, // -z
    4, 6, 5, 4, 7, 6, // +z
    0, 4, 5, 0, 5, 1, // -y
    2, 6, 7, 2, 7, 3, // +y
    0, 3, 7, 0, 7, 4, // -x
    1, 5, 6, 1, 6, 2, // +x
  ];
  for (const i of faces) indices.push(base + i);
};

/** Build mesh geometry for a creature body. */
export const buildProceduralBodyMesh = (seed: number): { mesh: MeshGeometry; meta: GeneratedBodyMeta } => {
  const rnd = mulberry32(seed || 1);
  const torsoW = 0.55 + rnd() * 0.35;
  const torsoH = 0.7 + rnd() * 0.45;
  const torsoD = 0.4 + rnd() * 0.25;
  const headR = 0.22 + rnd() * 0.12;
  const limbLen = 0.35 + rnd() * 0.25;
  const limbT = 0.1 + rnd() * 0.06;
  const stance = 0.15 + rnd() * 0.12;

  const positions: number[] = [];
  const indices: number[] = [];

  // Torso centered slightly above ground
  pushBox(positions, indices, {
    cx: 0,
    cy: torsoH / 2 + 0.15,
    cz: 0,
    sx: torsoW,
    sy: torsoH,
    sz: torsoD,
  });
  // Head
  pushBox(positions, indices, {
    cx: 0,
    cy: torsoH + 0.15 + headR,
    cz: 0,
    sx: headR * 2,
    sy: headR * 2,
    sz: headR * 2,
  });
  // Legs
  pushBox(positions, indices, {
    cx: -stance,
    cy: limbLen / 2,
    cz: 0,
    sx: limbT,
    sy: limbLen,
    sz: limbT,
  });
  pushBox(positions, indices, {
    cx: stance,
    cy: limbLen / 2,
    cz: 0,
    sx: limbT,
    sy: limbLen,
    sz: limbT,
  });
  // Arms
  const armY = torsoH * 0.65 + 0.15;
  pushBox(positions, indices, {
    cx: -(torsoW / 2 + limbLen / 2),
    cy: armY,
    cz: 0,
    sx: limbLen,
    sy: limbT,
    sz: limbT,
  });
  pushBox(positions, indices, {
    cx: torsoW / 2 + limbLen / 2,
    cy: armY,
    cz: 0,
    sx: limbLen,
    sy: limbT,
    sz: limbT,
  });

  const mesh: MeshGeometry = {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };

  let maxX = 0;
  let maxY = 0;
  let maxZ = 0;
  for (let i = 0; i < positions.length; i += 3) {
    maxX = Math.max(maxX, Math.abs(positions[i]));
    maxY = Math.max(maxY, Math.abs(positions[i + 1]));
    maxZ = Math.max(maxZ, Math.abs(positions[i + 2]));
  }
  const bounds: [number, number, number] = [maxX * 2, maxY, maxZ * 2];
  const collisionRadius = Math.max(0.35, Math.hypot(maxX, maxZ) * 1.05);

  return {
    mesh,
    meta: {
      seed,
      bounds,
      collisionRadius,
      createdAt: Date.now(),
    },
  };
};

export const generateBodyObjectUrl = (seed: number, key: string): { url: string; meta: GeneratedBodyMeta } => {
  const { mesh, meta } = buildProceduralBodyMesh(seed);
  const url = meshToObjectUrl(mesh, key);
  return { url, meta };
};

/** For tests: raw GLB bytes. */
export const generateBodyGlbBytes = (seed: number): ArrayBuffer => {
  const { mesh } = buildProceduralBodyMesh(seed);
  return buildGlb(mesh);
};

export const seedFromCreatureId = (creatureId: string): number => {
  let h = 2166136261;
  for (let i = 0; i < creatureId.length; i += 1) {
    h ^= creatureId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // Mix with time-stable salt so different ids differ; keep deterministic per id
  return (h >>> 0) || 1;
};
