/**
 * Minimal glTF 2.0 binary (GLB) builder — no external deps.
 * One mesh, one buffer, TRIANGLES, POSITION only (float32).
 */

export type MeshGeometry = {
  positions: Float32Array; // xyz * n
  indices: Uint32Array;
};

const textEncoder = new TextEncoder();

const pad4 = (n: number) => (4 - (n % 4)) % 4;

const concat = (parts: Uint8Array[]): Uint8Array => {
  const total = parts.reduce((sum, p) => sum + p.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out;
};

/** Build a single-mesh GLB ArrayBuffer from positions + triangle indices. */
export const buildGlb = (mesh: MeshGeometry): ArrayBuffer => {
  const positions = mesh.positions;
  const indices = mesh.indices;
  if (positions.length < 9 || indices.length < 3) {
    throw new Error('mesh too small for GLB');
  }

  // Bounds for accessor min/max
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  const posBytes = new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength);
  // Align indices to 4 bytes after positions
  const posPad = pad4(posBytes.byteLength);
  const indexBytes = new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength);
  const indexPad = pad4(indexBytes.byteLength);

  const binParts: Uint8Array[] = [posBytes];
  if (posPad) binParts.push(new Uint8Array(posPad));
  const indexByteOffset = posBytes.byteLength + posPad;
  binParts.push(indexBytes);
  if (indexPad) binParts.push(new Uint8Array(indexPad));
  const binary = concat(binParts);

  const json = {
    asset: { version: '2.0', generator: 'eden-procedural-body' },
    buffers: [{ byteLength: binary.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBytes.byteLength, target: 34962 },
      {
        buffer: 0,
        byteOffset: indexByteOffset,
        byteLength: indexBytes.byteLength,
        target: 34963,
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: positions.length / 3,
        type: 'VEC3',
        max: [maxX, maxY, maxZ],
        min: [minX, minY, minZ],
      },
      {
        bufferView: 1,
        componentType: 5125,
        count: indices.length,
        type: 'SCALAR',
      },
    ],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0 },
            indices: 1,
            mode: 4,
          },
        ],
      },
    ],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };

  const jsonText = JSON.stringify(json);
  const jsonBytes = textEncoder.encode(jsonText);
  const jsonPad = pad4(jsonBytes.byteLength);
  const jsonChunkLength = jsonBytes.byteLength + jsonPad;
  const binChunkLength = binary.byteLength;

  // GLB: 12-byte header + JSON chunk + BIN chunk
  const totalLength = 12 + 8 + jsonChunkLength + 8 + binChunkLength;
  const out = new ArrayBuffer(totalLength);
  const view = new DataView(out);
  const u8 = new Uint8Array(out);

  // magic glTF
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);

  let o = 12;
  view.setUint32(o, jsonChunkLength, true);
  o += 4;
  view.setUint32(o, 0x4e4f534a, true); // JSON
  o += 4;
  u8.set(jsonBytes, o);
  o += jsonBytes.byteLength;
  for (let i = 0; i < jsonPad; i += 1) u8[o + i] = 0x20; // space pad
  o += jsonPad;

  view.setUint32(o, binChunkLength, true);
  o += 4;
  view.setUint32(o, 0x004e4942, true); // BIN\0
  o += 4;
  u8.set(binary, o);

  return out;
};

const activeUrls = new Map<string, string>();

/** Encode GLB bytes as a stable data URL (blob: URLs break useGLTF / HMR / StrictMode). */
export const glbToDataUrl = (glb: ArrayBuffer): string => {
  const bytes = new Uint8Array(glb);
  // Avoid spread on large arrays (call stack limits).
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return `data:model/gltf-binary;base64,${btoa(binary)}`;
};

export const meshToObjectUrl = (mesh: MeshGeometry, key: string): string => {
  const prev = activeUrls.get(key);
  if (prev?.startsWith('blob:')) {
    URL.revokeObjectURL(prev);
  }
  const glb = buildGlb(mesh);
  // Prefer data: — survives React remounts; no revoke races with drei useGLTF.
  const url = glbToDataUrl(glb);
  activeUrls.set(key, url);
  return url;
};

export const revokeBodyUrl = (key: string): void => {
  const prev = activeUrls.get(key);
  if (prev?.startsWith('blob:')) {
    URL.revokeObjectURL(prev);
  }
  activeUrls.delete(key);
};

export const revokeAllBodyUrls = (): void => {
  for (const url of activeUrls.values()) {
    if (url.startsWith('blob:')) URL.revokeObjectURL(url);
  }
  activeUrls.clear();
};
