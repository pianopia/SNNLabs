/**
 * Coarse "vision" of nearby object shapes — not RGB CNN.
 * Uses entity size/shape as a stand-in for silhouette / 3D extent cues
 * an embodied agent could extract from depth or bounding volumes.
 */

export type ShapeFeatures = {
  /** Normalized 0–1.4 channels suitable as membrane currents */
  width: number;
  height: number;
  depth: number;
  volume: number;
  /** Aspect: tall vs wide (−1 wide … +1 tall), mapped later */
  aspect: number;
  /** World position of the observed object */
  x: number;
  z: number;
  /** 0–1.4 proximity-weighted salience */
  salience: number;
  label: string;
};

export type BodyExtent = {
  widthX: number;
  heightY: number;
  depthZ: number;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Map raw meters-ish size to sensor current range. */
export const normalizeExtent = (meters: number, typical = 1.2): number =>
  clamp(meters / Math.max(0.2, typical), 0, 1.4);

export const sizeToFeatures = (
  size: number[] | undefined,
  position: { x: number; z: number },
  observer: { x: number; z: number },
  label = 'object',
  shapeHint?: string,
): ShapeFeatures => {
  const sx = Math.max(0.05, size?.[0] ?? 1);
  const sy = Math.max(0.05, size?.[1] ?? size?.[0] ?? 1);
  const sz = Math.max(0.05, size?.[2] ?? size?.[0] ?? 1);
  // Sphere/cylinder: equalize axes slightly for cue stability
  const adj =
    shapeHint === 'sphere'
      ? { sx: (sx + sy + sz) / 3, sy: (sx + sy + sz) / 3, sz: (sx + sy + sz) / 3 }
      : shapeHint === 'cylinder'
        ? { sx, sy: sy * 1.1, sz }
        : { sx, sy, sz };
  const dist = Math.hypot(position.x - observer.x, position.z - observer.z);
  const proximity = clamp(1 - dist / 10, 0, 1);
  const volume = adj.sx * adj.sy * adj.sz;
  const aspect = clamp((adj.sy - Math.max(adj.sx, adj.sz)) / Math.max(adj.sy, adj.sx, adj.sz, 0.2), -1, 1);
  return {
    width: normalizeExtent(adj.sx),
    height: normalizeExtent(adj.sy),
    depth: normalizeExtent(adj.sz),
    volume: normalizeExtent(Math.cbrt(volume), 1.0),
    aspect,
    x: position.x,
    z: position.z,
    salience: clamp(proximity * (0.35 + normalizeExtent(Math.cbrt(volume)) * 0.4), 0, 1.4),
    label,
  };
};

/** Similarity of body extents to observed shape features (1 = match). */
export const shapeMatchScore = (body: BodyExtent, vision: Pick<ShapeFeatures, 'width' | 'height' | 'depth'>): number => {
  const bw = normalizeExtent(body.widthX);
  const bh = normalizeExtent(body.heightY);
  const bd = normalizeExtent(body.depthZ);
  const err =
    Math.abs(bw - vision.width) + Math.abs(bh - vision.height) + Math.abs(bd - vision.depth);
  // max err ~ 4.2; map to 0–1 match
  return clamp(1 - err / 2.4, 0, 1);
};

/** Soft target body extents from vision (denormalize roughly). */
export const visionToBodyTarget = (vision: Pick<ShapeFeatures, 'width' | 'height' | 'depth'>): BodyExtent => ({
  widthX: clamp(0.55 + vision.width * 0.85, 0.55, 1.8),
  heightY: clamp(0.55 + vision.height * 0.95, 0.55, 1.9),
  depthZ: clamp(0.55 + vision.depth * 0.85, 0.55, 1.8),
});

export type WorldShapeSource = {
  id: string;
  name?: string;
  x: number;
  z: number;
  size?: number[];
  shape?: string;
  isNpc?: boolean;
};

/**
 * Pick the most salient nearby non-self object as visual shape inspiration.
 */
export const pickVisualShapeInspiration = (
  observer: { id: string; x: number; z: number },
  sources: Iterable<WorldShapeSource>,
  isSelf?: (id: string, name?: string) => boolean,
): ShapeFeatures | null => {
  let best: ShapeFeatures | null = null;
  for (const src of sources) {
    if (src.id === observer.id) continue;
    if (isSelf?.(src.id, src.name)) continue;
    const features = sizeToFeatures(
      src.size,
      { x: src.x, z: src.z },
      { x: observer.x, z: observer.z },
      src.name ?? src.id,
      src.shape,
    );
    if (!best || features.salience > best.salience) best = features;
  }
  if (!best || best.salience < 0.04) return null;
  return best;
};
