/**
 * Persist procedural body seeds per creature (blob URLs are recreated on load).
 */

import {
  generateBodyObjectUrl,
  seedFromCreatureId,
  type GeneratedBodyMeta,
} from './proceduralBody';
import { revokeAllBodyUrls, revokeBodyUrl } from './bodyGlb';

const STORAGE_KEY = 'eden14:generated-body:v1';

export type BodyRecord = GeneratedBodyMeta & { glbUrl: string };

type StoredMap = Record<string, { seed: number; bounds: [number, number, number]; collisionRadius: number }>;

const memory = new Map<string, BodyRecord>();

const readStore = (): StoredMap => {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as StoredMap;
  } catch {
    return {};
  }
};

const writeStore = (map: StoredMap): void => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // quota / private mode — ignore
  }
};

/** Ensure a creature has a generated body URL (create or restore from seed). */
export const ensureGeneratedBody = (creatureId: string): BodyRecord => {
  const existing = memory.get(creatureId);
  if (existing) return existing;

  const store = readStore();
  const saved = store[creatureId];
  const seed = saved?.seed ?? seedFromCreatureId(creatureId);
  const { url, meta } = generateBodyObjectUrl(seed, creatureId);
  const record: BodyRecord = {
    ...meta,
    seed,
    bounds: saved?.bounds ?? meta.bounds,
    collisionRadius: saved?.collisionRadius ?? meta.collisionRadius,
    glbUrl: url,
  };
  memory.set(creatureId, record);
  store[creatureId] = {
    seed: record.seed,
    bounds: record.bounds,
    collisionRadius: record.collisionRadius,
  };
  writeStore(store);
  return record;
};

export const getGeneratedBody = (creatureId: string): BodyRecord | undefined => memory.get(creatureId);

export const clearGeneratedBodies = (): void => {
  revokeAllBodyUrls();
  memory.clear();
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
};

export const dropGeneratedBody = (creatureId: string): void => {
  revokeBodyUrl(creatureId);
  memory.delete(creatureId);
  const store = readStore();
  delete store[creatureId];
  writeStore(store);
};
