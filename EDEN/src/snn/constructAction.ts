/**
 * External object construction intents from embodied SNN motors.
 * Game layer turns these into local entities / WS createEntity.
 */

export type ConstructShape = 'box' | 'sphere' | 'cylinder';

export type ConstructRequest = {
  shape: ConstructShape;
  size: [number, number, number];
  x: number;
  y: number;
  z: number;
  inspiredBy?: string;
  seed: number;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Map vision-normalized channels + body to a placeable primitive. */
export const buildConstructRequest = (args: {
  creatureX: number;
  creatureZ: number;
  headingSin: number;
  headingCos: number;
  visionWidth: number;
  visionHeight: number;
  visionDepth: number;
  visionLabel?: string;
  seed: number;
}): ConstructRequest => {
  const w = clamp(0.35 + args.visionWidth * 0.7, 0.3, 1.6);
  const h = clamp(0.35 + args.visionHeight * 0.8, 0.3, 1.8);
  const d = clamp(0.35 + args.visionDepth * 0.7, 0.3, 1.6);
  // Prefer box; tall → cylinder; more isotropic → sphere
  const spread = Math.max(w, h, d) - Math.min(w, h, d);
  let shape: ConstructShape = 'box';
  if (h > w * 1.25 && h > d * 1.25) shape = 'cylinder';
  else if (spread < 0.25) shape = 'sphere';

  const placeDist = 1.4 + Math.max(w, d) * 0.35;
  return {
    shape,
    size: [w, h, d],
    x: args.creatureX + args.headingSin * placeDist,
    y: h * 0.5,
    z: args.creatureZ + args.headingCos * placeDist,
    inspiredBy: args.visionLabel,
    seed: args.seed,
  };
};
