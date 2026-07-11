import type { SnnTraceEvent } from './lif';

export type SnnRenderState = {
  auraColor: string;
  emissiveIntensity: number;
  scale: number;
  label: string;
};

export type SnnChatSignal = {
  text: string;
  priority: number;
};

export const initialSnnRenderState: SnnRenderState = {
  auraColor: '#b6ff4d',
  emissiveIntensity: 0.55,
  scale: 1,
  label: 'SNN Life',
};

const numberMeta = (event: SnnTraceEvent, key: string) => {
  const value = event.meta?.[key];
  return typeof value === 'number' ? value : 0;
};

export const deriveSnnRenderState = (
  events: SnnTraceEvent[],
  previous: SnnRenderState,
): SnnRenderState => {
  const body = events.find((event) => event.kind === 'body');
  const construct = events.find(
    (event) => event.kind === 'global_signal' && event.label === 'construct',
  );
  const overload = body ? numberMeta(body, 'overload') : 0;
  const ambient = body ? numberMeta(body, 'ambient_stimulus') : 0;
  const nearest = body ? numberMeta(body, 'nearest_stimulus') : 0;
  const deformation = body ? numberMeta(body, 'deformation') : 0;
  const gaitDrive = body ? numberMeta(body, 'gait_drive') : 0;
  const morphReward = body ? numberMeta(body, 'morph_reward') : 0;
  const visionMatch = body ? numberMeta(body, 'vision_match') : 0;
  const visionSal = body ? numberMeta(body, 'vision_salience') : 0;
  const reward = events.find((event) => event.kind === 'global_signal')?.value ?? 0;

  if (!body && !construct) {
    return previous;
  }

  if (construct) {
    return {
      auraColor: '#7dffb3',
      emissiveIntensity: 1.05,
      scale: 1.18,
      label: 'BUILDING',
    };
  }

  if (morphReward > 0.08 || visionMatch > 0.55) {
    return {
      auraColor: '#ffe566',
      emissiveIntensity: 0.95,
      scale: 1.08 + Math.min(0.14, visionMatch * 0.16),
      label: `MORPH ${(visionMatch * 100).toFixed(0)}%`,
    };
  }

  if (visionSal > 0.15) {
    return {
      auraColor: '#72f7ff',
      emissiveIntensity: 0.82,
      scale: 1.06,
      label: 'IMITATING',
    };
  }

  if (overload > 0.35) {
    return {
      auraColor: '#ff765c',
      emissiveIntensity: 0.85,
      scale: 1.08,
      label: 'overload',
    };
  }

  if (deformation > 0.38) {
    return {
      auraColor: '#ffbf69',
      emissiveIntensity: 0.78,
      scale: 1.02 + Math.min(0.08, deformation * 0.12),
      label: 'adapting',
    };
  }

  if (nearest > 0.2) {
    return {
      auraColor: '#72f7ff',
      emissiveIntensity: 0.75,
      scale: 1.04,
      label: 'sensing',
    };
  }

  if (gaitDrive > 0.42) {
    return {
      auraColor: '#9dff8f',
      emissiveIntensity: 0.72,
      scale: 1.02,
      label: 'moving',
    };
  }

  if (ambient > 0.2) {
    return {
      auraColor: '#d8a8ff',
      emissiveIntensity: 0.68,
      scale: 1.03,
      label: 'attuned',
    };
  }

  return {
    auraColor: '#b6ff4d',
    emissiveIntensity: Math.max(0.45, Math.min(0.75, reward * 0.55)),
    scale: 1,
    label: 'alive',
  };
};

export const deriveSnnChatSignal = (
  events: SnnTraceEvent[],
  lastMessageAt: number,
  now: number,
): SnnChatSignal | null => {
  // Morph/construct feedback should feel continuous, not once per 6s.
  if (now - lastMessageAt < 2800) return null;

  const construct = events.find(
    (event) => event.kind === 'global_signal' && event.label === 'construct',
  );
  if (construct) {
    const shape = construct.meta?.shape;
    const inspired = construct.meta?.inspired_by;
    return {
      text: `★ construct ${typeof shape === 'string' ? shape : 'object'}${inspired ? ` ← ${inspired}` : ''}`,
      priority: 4,
    };
  }

  const body = events.find((event) => event.kind === 'body');
  if (!body) return null;

  const overload = numberMeta(body, 'overload');
  const ambient = numberMeta(body, 'ambient_stimulus');
  const nearest = numberMeta(body, 'nearest_stimulus');
  const deformation = numberMeta(body, 'deformation');
  const gaitDrive = numberMeta(body, 'gait_drive');
  const morphReward = numberMeta(body, 'morph_reward');
  const visionMatch = numberMeta(body, 'vision_match');
  const visionLabel = typeof body.meta?.vision_label === 'string' ? body.meta.vision_label : '';
  const bodyW = numberMeta(body, 'body_width');
  const bodyH = numberMeta(body, 'body_height');
  const bodyD = numberMeta(body, 'body_depth');
  const curiositySpike = events.some((event) => event.kind === 'spike' && event.label === 'interneuron:curiosity');
  const defenseSpike = events.some((event) => event.kind === 'spike' && event.label === 'interneuron:defense');

  if (morphReward > 0.05 || visionMatch > 0.4) {
    return {
      text: `morph→${visionLabel || 'shape'} match=${(visionMatch * 100).toFixed(0)}% body=${bodyW.toFixed(2)}×${bodyH.toFixed(2)}×${bodyD.toFixed(2)}`,
      priority: 3,
    };
  }

  if (overload > 0.45 || defenseSpike) {
    return { text: `刺激過多を検知: overload=${overload.toFixed(2)}`, priority: 3 };
  }

  if (nearest > 0.3 || curiositySpike) {
    return { text: `近接刺激を探索中: stimulus=${nearest.toFixed(2)}`, priority: 2 };
  }

  if (deformation > 0.4) {
    return { text: `身体形状を調整中: deformation=${deformation.toFixed(2)}`, priority: 2 };
  }

  if (gaitDrive > 0.5) {
    return { text: `関節駆動で移動中: gait=${gaitDrive.toFixed(2)}`, priority: 1 };
  }

  if (ambient > 0.28) {
    return { text: `環境刺激に同調: ambient=${ambient.toFixed(2)}`, priority: 1 };
  }

  return null;
};
