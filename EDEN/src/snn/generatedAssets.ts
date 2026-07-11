/**
 * Load Python-exported 3DCG assets from /generated/manifest.json
 * and turn them into world entities (no UI).
 */

export type GeneratedAssetEntry = {
  name: string;
  url: string;
  track?: string;
  family?: string;
  quality?: number;
  n_vertices?: number;
  n_faces?: number;
  has_uv?: boolean;
  has_skin?: boolean;
  bones?: string[];
};

export type GeneratedManifest = {
  version: number;
  updatedAt?: string;
  assets: GeneratedAssetEntry[];
};

export const GENERATED_MANIFEST_URL = '/generated/manifest.json';

export async function fetchGeneratedManifest(
  url = GENERATED_MANIFEST_URL,
): Promise<GeneratedManifest | null> {
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return null;
    const data = (await res.json()) as GeneratedManifest;
    if (!data || !Array.isArray(data.assets)) return null;
    return data;
  } catch {
    return null;
  }
}

/** Place assets in a ring around origin so SNN vision can see them. */
export function layoutGeneratedSpawns(
  assets: GeneratedAssetEntry[],
  origin: { x: number; z: number },
  radius = 4.5,
): Array<GeneratedAssetEntry & { x: number; y: number; z: number; id: string }> {
  const n = Math.max(1, assets.length);
  return assets.map((asset, index) => {
    const angle = (index / n) * Math.PI * 2 + 0.4;
    return {
      ...asset,
      id: `generated-${asset.name}`,
      x: origin.x + Math.cos(angle) * radius,
      y: 0.6,
      z: origin.z + Math.sin(angle) * radius,
    };
  });
}
