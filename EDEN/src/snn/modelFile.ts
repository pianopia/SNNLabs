import type { EmbodiedSnnSnapshot, SnnLearningGoal } from './lif';

const EDEN_SNN_MAGIC = 'EDENSNN1';
const EDEN_SNN_MIME = 'application/x-edensnn';

export type EdenSnnModelEntry = {
  storageKey?: string;
  creatureId: string;
  modelKind?: 'embodied-life' | 'browser-language';
  snapshot: EmbodiedSnnSnapshot | Record<string, unknown>;
};

export type EdenSnnModelFile = {
  kind: 'eden14-snn-life-model';
  version: 1 | 2;
  container: typeof EDEN_SNN_MAGIC;
  exportedAt: string;
  modelCount: number;
  learningGoal?: SnnLearningGoal;
  models: EdenSnnModelEntry[];
};

export const encodeEdenSnnModelFile = (payload: Omit<EdenSnnModelFile, 'kind' | 'version' | 'container' | 'modelCount'>) => {
  const modelFile: EdenSnnModelFile = {
    kind: 'eden14-snn-life-model',
    version: 2,
    container: EDEN_SNN_MAGIC,
    modelCount: payload.models.length,
    ...payload,
  };
  const encodedPayload = new TextEncoder().encode(JSON.stringify(modelFile));
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  const header = new ArrayBuffer(magic.length + 4);
  const headerBytes = new Uint8Array(header);
  headerBytes.set(magic, 0);
  new DataView(header).setUint32(magic.length, encodedPayload.byteLength, true);

  return new Blob([header, encodedPayload], { type: EDEN_SNN_MIME });
};

export const decodeEdenSnnModelFile = async (file: File): Promise<EdenSnnModelFile> => {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const magicBytes = new TextEncoder().encode(EDEN_SNN_MAGIC);
  if (bytes.length < magicBytes.length + 4) {
    throw new Error('File is too small to be an EDEN SNN model.');
  }

  for (let index = 0; index < magicBytes.length; index += 1) {
    if (bytes[index] !== magicBytes[index]) {
      throw new Error('Invalid EDEN SNN model header.');
    }
  }

  const payloadLength = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(magicBytes.length, true);
  const payloadStart = magicBytes.length + 4;
  const payloadEnd = payloadStart + payloadLength;
  if (payloadEnd > bytes.length) {
    throw new Error('EDEN SNN model payload is truncated.');
  }

  const parsed = JSON.parse(new TextDecoder().decode(bytes.slice(payloadStart, payloadEnd))) as EdenSnnModelFile;
  if (parsed.kind !== 'eden14-snn-life-model' || parsed.container !== EDEN_SNN_MAGIC || !Array.isArray(parsed.models)) {
    throw new Error('Unsupported EDEN SNN model payload.');
  }
  return parsed;
};
