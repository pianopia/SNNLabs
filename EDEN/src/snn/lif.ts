export type SnnTraceEvent = {
  tMs: number;
  kind: 'input' | 'membrane' | 'spike' | 'weight' | 'global_signal' | 'body' | 'neuromodulator' | 'delayed_reward' | 'consolidation' | 'homeostasis';
  neuron?: number;
  synapse?: number;
  value: number;
  label?: string;
  meta?: Record<string, number | string | boolean>;
};

export type SnnLearningGoal = 'wander' | 'seekStimulus' | 'avoidOverload';
export type SnnNeuronType = 'lif' | 'si-lif';

export type SnnEnvironmentStimulus = {
  nearestX?: number;
  nearestZ?: number;
  nearestIntensity: number;
  ambientIntensity: number;
  overload: number;
  collisionIntensity?: number;
  collisionNormalX?: number;
  collisionNormalZ?: number;
};

export type EmbodiedSnnSnapshot = {
  version: 1 | 2 | 3;
  savedAt: number;
  creature: {
    energy: number;
    wanderX: number;
    wanderZ: number;
    retargetAtMs: number;
    body?: SnnBodyState;
  };
  network: {
    tMs: number;
    learningRate: number;
    rewardSignal: number;
    neuronType?: SnnNeuronType;
    spikeRangeD?: number;
    spikeStats?: SpikeStats;
    neuromodulators?: Neuromodulators;
    consolidationStats?: ConsolidationStats;
    neurons: Array<{
      id: number;
      label: string;
      v: number;
      refractoryLeftMs: number;
    }>;
    synapses: Array<{
      id: number;
      pre: number;
      post: number;
      w: number;
      aPre: number;
      aPost: number;
      stability?: number;
      delayedCredit?: number;
    }>;
  };
};

export type SnnBodyState = {
  widthX: number;
  heightY: number;
  depthZ: number;
  asymmetry: number;
  jointPhase: number;
  jointSwing: number;
  limbReach: number;
  rigWeight: number;
  deformation: number;
  gaitDrive: number;
  mass: number;
  drag: number;
};

export type SnnEmbodiedBodyDescription = {
  shape: 'custom';
  size: [number, number, number];
  rotation: [number, number, number];
  geometry: { vertices: number[]; indices: number[] };
  physics: {
    mass: number;
    drag: number;
    collisionRadius: number;
    gaitDrive: number;
  };
  rig: {
    jointPhase: number;
    jointSwing: number;
    limbReach: number;
    rigWeight: number;
  };
};

export type Neuromodulators = {
  dopamine: number;
  acetylcholine: number;
  norepinephrine: number;
  serotonin: number;
  fatigue: number;
};

export type ConsolidationStats = {
  cycles: number;
  stabilizedSynapses: number;
  delayedRewards: number;
  delayedCreditAssignments: number;
};

type Neuron = {
  id: number;
  label: string;
  v: number;
  rest: number;
  reset: number;
  threshold: number;
  tauMs: number;
  refractoryMs: number;
  refractoryLeftMs: number;
};

type Synapse = {
  id: number;
  pre: number;
  post: number;
  w: number;
  minW: number;
  maxW: number;
  delayMs: number;
  aPre: number;
  aPost: number;
  tauPreMs: number;
  tauPostMs: number;
  dPre: number;
  dPost: number;
  stability: number;
  delayedCredit: number;
  lastUpdatedMs: number;
};

type SpikeDelivery = {
  deliverAtMs: number;
  post: number;
  synapse: number;
  amount: number;
};

type EligibilityTrace = {
  pre: number;
  post: number;
  e: number;
  lastAtMs: number;
};

export type SpikeStats = {
  ticks: number;
  positiveSpikes: number;
  negativeSpikes: number;
  absoluteSpikeMass: number;
  sparseAcOps: number;
  denseMacOps: number;
};

type Network = {
  tMs: number;
  dtMs: number;
  learningEnabled: boolean;
  learningRate: number;
  rewardSignal: number;
  neuronType: SnnNeuronType;
  spikeRangeD: number;
  spikeStats: SpikeStats;
  neuromodulators: Neuromodulators;
  consolidationStats: ConsolidationStats;
  neurons: Neuron[];
  synapses: Synapse[];
  pending: SpikeDelivery[];
  eligibility: EligibilityTrace[];
};

export type EmbodiedSnnCreature = {
  id: string;
  name: string;
  x: number;
  y: number;
  z: number;
  energy: number;
  wanderX: number;
  wanderZ: number;
  retargetAtMs: number;
  body: SnnBodyState;
  network: Network;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const neuron = (id: number, label: string, threshold: number, tauMs: number): Neuron => ({
  id,
  label,
  v: 0,
  rest: 0,
  reset: 0,
  threshold,
  tauMs,
  refractoryMs: 2,
  refractoryLeftMs: 0,
});

const synapse = (id: number, pre: number, post: number, w: number, dPre = 0.014, dPost = -0.015): Synapse => ({
  id,
  pre,
  post,
  w,
  minW: 0,
  maxW: 1,
  delayMs: 1,
  aPre: 0,
  aPost: 0,
  tauPreMs: 20,
  tauPostMs: 20,
  dPre,
  dPost,
  stability: 0,
  delayedCredit: 0,
  lastUpdatedMs: 0,
});

const emptySpikeStats = (): SpikeStats => ({
  ticks: 0,
  positiveSpikes: 0,
  negativeSpikes: 0,
  absoluteSpikeMass: 0,
  sparseAcOps: 0,
  denseMacOps: 0,
});

const emptyNeuromodulators = (): Neuromodulators => ({
  dopamine: 0,
  acetylcholine: 0.25,
  norepinephrine: 0,
  serotonin: 0.35,
  fatigue: 0,
});

const emptyConsolidationStats = (): ConsolidationStats => ({
  cycles: 0,
  stabilizedSynapses: 0,
  delayedRewards: 0,
  delayedCreditAssignments: 0,
});

const defaultBodyState = (): SnnBodyState => ({
  widthX: 1,
  heightY: 1,
  depthZ: 1,
  asymmetry: 0,
  jointPhase: 0,
  jointSwing: 0.22,
  limbReach: 0.45,
  rigWeight: 0.5,
  deformation: 0,
  gaitDrive: 0,
  mass: 1,
  drag: 0.18,
});

const makeNetwork = (): Network => ({
  tMs: 0,
  dtMs: 16,
  learningEnabled: true,
  learningRate: 0.55,
  rewardSignal: 1,
  neuronType: 'si-lif',
  spikeRangeD: 4,
  spikeStats: emptySpikeStats(),
  neuromodulators: emptyNeuromodulators(),
  consolidationStats: emptyConsolidationStats(),
  neurons: [
    neuron(0, 'sensor:target_left', 0.8, 32),
    neuron(1, 'sensor:target_right', 0.8, 32),
    neuron(2, 'sensor:target_far', 0.9, 36),
    neuron(3, 'interneuron:approach_left', 0.95, 44),
    neuron(4, 'interneuron:approach_right', 0.95, 44),
    neuron(5, 'motor:left', 0.95, 40),
    neuron(6, 'motor:right', 0.95, 40),
    neuron(7, 'motor:forward', 0.9, 40),
    neuron(8, 'sensor:stimulus_left', 0.8, 32),
    neuron(9, 'sensor:stimulus_right', 0.8, 32),
    neuron(10, 'sensor:ambient_stimulus', 0.85, 36),
    neuron(11, 'sensor:overload', 0.85, 36),
    neuron(12, 'interneuron:curiosity', 0.9, 44),
    neuron(13, 'interneuron:defense', 0.9, 44),
    neuron(14, 'sensor:collision', 0.75, 28),
    neuron(15, 'sensor:body_compact', 0.82, 34),
    neuron(16, 'sensor:body_extended', 0.82, 34),
    neuron(17, 'sensor:joint_motion', 0.78, 30),
    neuron(18, 'sensor:rig_load', 0.84, 36),
    neuron(19, 'interneuron:morphology', 0.9, 42),
    neuron(20, 'motor:body_expand', 0.92, 38),
    neuron(21, 'motor:body_contract', 0.92, 38),
    neuron(22, 'motor:joint_swing', 0.9, 36),
    neuron(23, 'motor:stabilize_pose', 0.92, 40),
  ],
  synapses: [
    synapse(0, 0, 3, 0.48),
    synapse(1, 1, 4, 0.48),
    synapse(2, 2, 7, 0.52),
    synapse(3, 3, 5, 0.54),
    synapse(4, 4, 6, 0.54),
    synapse(5, 5, 3, 0.14, 0.006, -0.007),
    synapse(6, 6, 4, 0.14, 0.006, -0.007),
    synapse(7, 8, 12, 0.5),
    synapse(8, 9, 12, 0.5),
    synapse(9, 10, 12, 0.34),
    synapse(10, 11, 13, 0.58),
    synapse(11, 12, 7, 0.45),
    synapse(12, 13, 5, 0.36),
    synapse(13, 13, 6, 0.36),
    synapse(14, 14, 13, 0.7),
    synapse(15, 15, 19, 0.42),
    synapse(16, 16, 19, 0.46),
    synapse(17, 17, 22, 0.48),
    synapse(18, 18, 23, 0.52),
    synapse(19, 19, 20, 0.38),
    synapse(20, 19, 21, 0.32),
    synapse(21, 12, 22, 0.34),
    synapse(22, 13, 21, 0.42),
    synapse(23, 14, 23, 0.64),
    synapse(24, 20, 7, 0.18, 0.006, -0.007),
    synapse(25, 22, 7, 0.2, 0.006, -0.007),
    synapse(26, 23, 13, 0.22, 0.006, -0.007),
  ],
  pending: [],
  eligibility: [],
});

export const createEmbodiedSnnCreature = (
  x: number,
  z: number,
  id = 'snn-life-001',
  name = 'SNN Life 1',
): EmbodiedSnnCreature => ({
  id,
  name,
  x,
  y: 0.5,
  z,
  energy: 1,
  wanderX: x + 2,
  wanderZ: z,
  retargetAtMs: 0,
  body: defaultBodyState(),
  network: makeNetwork(),
});

export const snapshotEmbodiedSnnCreature = (creature: EmbodiedSnnCreature): EmbodiedSnnSnapshot => ({
  version: 3,
  savedAt: Date.now(),
  creature: {
    energy: creature.energy,
    wanderX: creature.wanderX,
    wanderZ: creature.wanderZ,
    retargetAtMs: creature.retargetAtMs,
    body: { ...creature.body },
  },
  network: {
    tMs: creature.network.tMs,
    learningRate: creature.network.learningRate,
    rewardSignal: creature.network.rewardSignal,
    neuronType: creature.network.neuronType,
    spikeRangeD: creature.network.spikeRangeD,
    spikeStats: { ...creature.network.spikeStats },
    neuromodulators: { ...creature.network.neuromodulators },
    consolidationStats: { ...creature.network.consolidationStats },
    neurons: creature.network.neurons.map((cell) => ({
      id: cell.id,
      label: cell.label,
      v: cell.v,
      refractoryLeftMs: cell.refractoryLeftMs,
    })),
    synapses: creature.network.synapses.map((syn) => ({
      id: syn.id,
      pre: syn.pre,
      post: syn.post,
      w: syn.w,
      aPre: syn.aPre,
      aPost: syn.aPost,
      stability: syn.stability,
      delayedCredit: syn.delayedCredit,
    })),
  },
});

export const restoreEmbodiedSnnCreature = (
  creature: EmbodiedSnnCreature,
  snapshot: EmbodiedSnnSnapshot,
) => {
  if (snapshot.version !== 1 && snapshot.version !== 2 && snapshot.version !== 3) return;

  creature.energy = clamp(snapshot.creature.energy, 0.05, 1);
  creature.wanderX = snapshot.creature.wanderX;
  creature.wanderZ = snapshot.creature.wanderZ;
  creature.retargetAtMs = snapshot.creature.retargetAtMs;
  creature.body = {
    ...defaultBodyState(),
    ...(snapshot.creature.body ?? {}),
  };
  creature.body.widthX = clamp(creature.body.widthX, 0.55, 1.8);
  creature.body.heightY = clamp(creature.body.heightY, 0.55, 1.9);
  creature.body.depthZ = clamp(creature.body.depthZ, 0.55, 1.8);
  creature.body.rigWeight = clamp(creature.body.rigWeight, 0, 1);
  creature.body.limbReach = clamp(creature.body.limbReach, 0.1, 1.25);
  creature.network.tMs = Math.max(0, snapshot.network.tMs);
  creature.network.learningRate = clamp(snapshot.network.learningRate, 0, 4);
  creature.network.rewardSignal = clamp(snapshot.network.rewardSignal, 0, 4);
  creature.network.neuronType = snapshot.network.neuronType ?? creature.network.neuronType;
  creature.network.spikeRangeD = clamp(snapshot.network.spikeRangeD ?? creature.network.spikeRangeD, 1, 16);
  creature.network.spikeStats = snapshot.network.spikeStats
    ? { ...emptySpikeStats(), ...snapshot.network.spikeStats }
    : emptySpikeStats();
  creature.network.neuromodulators = snapshot.network.neuromodulators
    ? { ...emptyNeuromodulators(), ...snapshot.network.neuromodulators }
    : emptyNeuromodulators();
  creature.network.consolidationStats = snapshot.network.consolidationStats
    ? { ...emptyConsolidationStats(), ...snapshot.network.consolidationStats }
    : emptyConsolidationStats();

  for (const saved of snapshot.network.neurons) {
    const cell = creature.network.neurons.find((item) => item.id === saved.id && item.label === saved.label);
    if (!cell) continue;
    cell.v = saved.v;
    cell.refractoryLeftMs = Math.max(0, saved.refractoryLeftMs);
  }

  for (const saved of snapshot.network.synapses) {
    const syn = creature.network.synapses.find((item) => item.id === saved.id && item.pre === saved.pre && item.post === saved.post);
    if (!syn) continue;
    syn.w = clamp(saved.w, syn.minW, syn.maxW);
    syn.aPre = saved.aPre;
    syn.aPost = saved.aPost;
    syn.stability = saved.stability ?? syn.stability;
    syn.delayedCredit = saved.delayedCredit ?? syn.delayedCredit;
  }
};

const record = (
  events: SnnTraceEvent[],
  network: Network,
  kind: SnnTraceEvent['kind'],
  value: number,
  options: Omit<SnnTraceEvent, 'tMs' | 'kind' | 'value'> = {},
) => {
  events.push({ tMs: network.tMs, kind, value, ...options });
};

const plasticityScale = (network: Network) => (
  network.learningEnabled
    ? network.learningRate
      * network.rewardSignal
      * clamp(0.5 + network.neuromodulators.dopamine * 0.35 + network.neuromodulators.acetylcholine * 0.25 - network.neuromodulators.fatigue * 0.22, 0.08, 1.6)
    : 0
);

const rememberEligibility = (network: Network, syn: Synapse, spikeValue: number) => {
  const existing = network.eligibility.find((item) => item.pre === syn.pre && item.post === syn.post);
  const e = clamp((existing?.e ?? 0) * 0.72 + Math.abs(spikeValue) * 0.42, 0, 4);
  if (existing) {
    existing.e = e;
    existing.lastAtMs = network.tMs;
  } else {
    network.eligibility.push({ pre: syn.pre, post: syn.post, e, lastAtMs: network.tMs });
  }
  network.eligibility = network.eligibility
    .filter((item) => network.tMs - item.lastAtMs < 3200)
    .slice(-160);
};

const applyDelayedReward = (network: Network, events: SnnTraceEvent[], outcome: number) => {
  if (Math.abs(outcome) < 0.04 || network.eligibility.length === 0) return;
  let assignments = 0;
  network.eligibility = network.eligibility.flatMap((trace) => {
    const age = network.tMs - trace.lastAtMs;
    if (age > 3200) return [];
    const credit = trace.e * Math.exp(-age / 1800);
    if (credit < 0.002) return [];
    const syn = network.synapses.find((item) => item.pre === trace.pre && item.post === trace.post);
    if (syn) {
      const stabilityBrake = 1 - syn.stability * 0.4;
      syn.w = clamp(syn.w + 0.035 * outcome * credit * stabilityBrake, syn.minW, syn.maxW);
      syn.delayedCredit += outcome * credit;
      assignments += 1;
    }
    return [{ ...trace, e: credit * 0.82 }];
  });
  if (assignments > 0) {
    network.consolidationStats.delayedRewards += 1;
    network.consolidationStats.delayedCreditAssignments += assignments;
    record(events, network, 'delayed_reward', outcome, { label: 'eligibility_credit', meta: { assignments } });
  }
};

const updateNeuromodulators = (network: Network, reward: number, salience: number) => {
  const n = network.neuromodulators;
  const error = reward - n.dopamine;
  n.dopamine = clamp(n.dopamine * 0.86 + error * 0.3, -1, 1);
  n.acetylcholine = clamp(n.acetylcholine * 0.84 + salience * 0.3, 0, 1);
  n.norepinephrine = clamp(n.norepinephrine * 0.88 + salience * 0.38, 0, 1);
  n.serotonin = clamp(n.serotonin * 0.96 + Math.max(0, reward) * 0.05 - Math.max(0, -reward) * 0.06, 0, 1);
  n.fatigue = clamp(n.fatigue * 0.99 + 0.0008, 0, 1);
};

const consolidateEmbodiedMemory = (network: Network, events: SnnTraceEvent[]) => {
  if (Math.floor(network.tMs) % 4096 > network.dtMs) return;
  let stabilized = 0;
  for (const syn of network.synapses) {
    const strong = syn.w > 0.52 || Math.abs(syn.delayedCredit) > 0.6;
    if (strong) {
      syn.stability = clamp(syn.stability + 0.035, 0, 1);
      syn.w = clamp(syn.w + syn.stability * 0.006, syn.minW, syn.maxW);
      stabilized += 1;
    } else if (syn.stability < 0.12) {
      syn.w = clamp(syn.w * 0.998, syn.minW, syn.maxW);
    }
  }
  if (stabilized > 0) {
    network.consolidationStats.cycles += 1;
    network.consolidationStats.stabilizedSynapses += stabilized;
    record(events, network, 'consolidation', stabilized, { label: 'sleep_replay' });
  }
};

const signedSpikeValue = (network: Network, cell: Neuron) => {
  if (network.neuronType === 'lif') return cell.v >= cell.threshold ? 1 : 0;
  const normalized = cell.v / Math.max(1e-6, cell.threshold);
  return clamp(Math.round(normalized), -network.spikeRangeD, network.spikeRangeD);
};

const applyPreSpike = (network: Network, events: SnnTraceEvent[], pre: number, spikeValue: number) => {
  const scale = plasticityScale(network);
  for (const syn of network.synapses) {
    if (syn.pre !== pre) continue;
    syn.aPre += syn.dPre * spikeValue;
    syn.w = clamp(syn.w + syn.aPost * scale * (1 - syn.stability * 0.35), syn.minW, syn.maxW);
    syn.lastUpdatedMs = network.tMs;
    rememberEligibility(network, syn, spikeValue);
    network.pending.push({
      deliverAtMs: network.tMs + syn.delayMs,
      post: syn.post,
      synapse: syn.id,
      amount: syn.w * spikeValue,
    });
    network.spikeStats.sparseAcOps += Math.abs(spikeValue);
    record(events, network, 'weight', syn.w, {
      synapse: syn.id,
      meta: { pre: syn.pre, post: syn.post, rule: 'on_pre', plasticity_scale: scale, spike_value: spikeValue },
    });
  }
};

const applyPostSpike = (network: Network, events: SnnTraceEvent[], post: number, spikeValue: number) => {
  const scale = plasticityScale(network);
  for (const syn of network.synapses) {
    if (syn.post !== post) continue;
    syn.aPost += syn.dPost * spikeValue;
    syn.w = clamp(syn.w + syn.aPre * scale * (1 - syn.stability * 0.35), syn.minW, syn.maxW);
    syn.lastUpdatedMs = network.tMs;
    rememberEligibility(network, syn, spikeValue);
    record(events, network, 'weight', syn.w, {
      synapse: syn.id,
      meta: { pre: syn.pre, post: syn.post, rule: 'on_post', plasticity_scale: scale, spike_value: spikeValue },
    });
  }
};

const stepNetwork = (network: Network, externalCurrents: number[]) => {
  const events: SnnTraceEvent[] = [];
  const currents = [...externalCurrents];

  for (const syn of network.synapses) {
    syn.aPre *= Math.exp(-network.dtMs / syn.tauPreMs);
    syn.aPost *= Math.exp(-network.dtMs / syn.tauPostMs);
  }

  for (let i = 0; i < currents.length; i += 1) {
    if (Math.abs(currents[i]) > 1e-12) {
      record(events, network, 'input', currents[i], { neuron: i, label: network.neurons[i].label });
    }
  }

  const remaining: SpikeDelivery[] = [];
  for (const delivery of network.pending) {
    if (delivery.deliverAtMs <= network.tMs) {
      currents[delivery.post] += delivery.amount;
      record(events, network, 'input', delivery.amount, {
        neuron: delivery.post,
        synapse: delivery.synapse,
        label: network.neurons[delivery.post].label,
      });
    } else {
      remaining.push(delivery);
    }
  }
  network.pending = remaining;

  const spiked: Array<{ id: number; value: number }> = [];
  network.spikeStats.ticks += 1;
  network.spikeStats.denseMacOps += network.synapses.length;
  network.neurons.forEach((cell, index) => {
    if (cell.refractoryLeftMs > 0) {
      cell.refractoryLeftMs = Math.max(0, cell.refractoryLeftMs - network.dtMs);
      record(events, network, 'membrane', cell.v, { neuron: index, label: cell.label });
      return;
    }

    cell.v += (cell.rest - cell.v) * (network.dtMs / cell.tauMs) + currents[index];
    record(events, network, 'membrane', cell.v, { neuron: index, label: cell.label });

    const spikeValue = signedSpikeValue(network, cell);
    if (spikeValue !== 0) {
      cell.v = cell.reset;
      cell.refractoryLeftMs = cell.refractoryMs;
      spiked.push({ id: index, value: spikeValue });
      if (spikeValue > 0) network.spikeStats.positiveSpikes += spikeValue;
      if (spikeValue < 0) network.spikeStats.negativeSpikes += Math.abs(spikeValue);
      network.spikeStats.absoluteSpikeMass += Math.abs(spikeValue);
      record(events, network, 'spike', spikeValue, {
        neuron: index,
        label: cell.label,
        meta: { neuron_type: network.neuronType, spike_range_d: network.spikeRangeD },
      });
    }
  });

  for (const spike of spiked) applyPreSpike(network, events, spike.id, spike.value);
  for (const spike of spiked) applyPostSpike(network, events, spike.id, spike.value);

  network.tMs += network.dtMs;
  return events;
};

const nextWanderTarget = (creature: EmbodiedSnnCreature) => {
  const seed = Math.sin(creature.network.tMs * 12.9898 + creature.x * 78.233 + creature.z * 37.719);
  const angle = (seed - Math.floor(seed)) * Math.PI * 2;
  const radius = 2.5 + Math.abs(Math.sin(creature.network.tMs * 0.017)) * 3.5;
  creature.wanderX = creature.x + Math.sin(angle) * radius;
  creature.wanderZ = creature.z + Math.cos(angle) * radius;
  creature.retargetAtMs = creature.network.tMs + 2800;
};

const ensureBodyState = (creature: EmbodiedSnnCreature) => {
  creature.body ??= defaultBodyState();
  return creature.body;
};

const activationFromEvents = (network: Network, events: SnnTraceEvent[], label: string) => {
  const cell = network.neurons.find((item) => item.label === label);
  const membrane = cell ? clamp(cell.v / cell.threshold, 0, 1) : 0;
  const spikeValue = events
    .filter((event) => event.kind === 'spike' && event.label === label)
    .reduce((sum, event) => sum + event.value, 0);
  return spikeValue > 0 ? clamp(spikeValue / network.spikeRangeD, 0, 1) : membrane;
};

const updateBodyState = (
  creature: EmbodiedSnnCreature,
  events: SnnTraceEvent[],
  dt: number,
  forward: number,
  turn: number,
  stimulus: SnnEnvironmentStimulus,
) => {
  const body = ensureBodyState(creature);
  const expand = activationFromEvents(creature.network, events, 'motor:body_expand');
  const contract = activationFromEvents(creature.network, events, 'motor:body_contract');
  const joint = activationFromEvents(creature.network, events, 'motor:joint_swing');
  const stabilize = activationFromEvents(creature.network, events, 'motor:stabilize_pose');
  const collision = clamp(stimulus.collisionIntensity ?? 0, 0, 1.4);
  const bodySignal = expand - contract;
  const learningRate = clamp(dt * 3.5, 0.02, 0.18);

  body.jointPhase = (body.jointPhase + dt * (1.5 + forward * 4.4 + joint * 2.2)) % (Math.PI * 2);
  body.jointSwing = clamp(body.jointSwing * 0.94 + (0.16 + joint * 0.56 + forward * 0.18 - stabilize * 0.22 - collision * 0.2) * 0.06, 0.04, 1.05);
  body.limbReach = clamp(body.limbReach * 0.96 + (0.28 + expand * 0.42 + joint * 0.28 - contract * 0.18) * 0.04, 0.1, 1.25);
  body.rigWeight = clamp(body.rigWeight * 0.95 + (0.28 + joint * 0.34 + stabilize * 0.26 + collision * 0.22) * 0.05, 0, 1);
  body.asymmetry = clamp(body.asymmetry * 0.9 + turn * 0.16 + Math.sin(body.jointPhase) * body.jointSwing * 0.045, -1, 1);
  body.widthX = clamp(body.widthX + (bodySignal * 0.32 - collision * 0.08 + Math.abs(turn) * 0.06) * learningRate, 0.58, 1.75);
  body.heightY = clamp(body.heightY + (expand * 0.18 - contract * 0.2 + stabilize * 0.08 - collision * 0.05) * learningRate, 0.62, 1.85);
  body.depthZ = clamp(body.depthZ + (bodySignal * 0.18 + forward * 0.08 - collision * 0.07) * learningRate, 0.58, 1.75);
  body.deformation = clamp(body.deformation * 0.88 + (collision * 0.32 + Math.abs(bodySignal) * 0.1 + Math.abs(turn) * 0.08 - stabilize * 0.16), 0, 1);
  body.gaitDrive = clamp(body.gaitDrive * 0.82 + (forward * (0.35 + joint * 0.55) + expand * 0.08 - body.drag * 0.06), 0, 1.4);
  body.mass = clamp((body.widthX * body.heightY * body.depthZ) ** 0.72, 0.45, 2.4);
  body.drag = clamp(0.12 + Math.max(body.widthX, body.depthZ) * 0.08 + body.deformation * 0.26 + body.rigWeight * 0.08, 0.08, 0.68);
  creature.y = clamp(0.34 + body.heightY * 0.22 + Math.abs(Math.sin(body.jointPhase)) * body.jointSwing * 0.06, 0.28, 0.9);
};

export const describeEmbodiedSnnBody = (
  creature: EmbodiedSnnCreature,
  renderScale = 1,
): SnnEmbodiedBodyDescription => {
  const body = ensureBodyState(creature);
  const scale = clamp(renderScale, 0.35, 2.6);
  const sx = 0.38 * body.widthX * scale;
  const sy = 0.48 * body.heightY * scale;
  const sz = 0.38 * body.depthZ * scale;
  const phase = body.jointPhase;
  const swing = Math.sin(phase) * body.jointSwing * body.rigWeight;
  const counterSwing = Math.cos(phase) * body.jointSwing * (1 - body.rigWeight * 0.35);
  const reach = body.limbReach * scale;
  const asym = body.asymmetry;
  const vertices = [
    0, sy, 0,
    sx * (1 + asym * 0.12), 0, 0,
    0, 0, sz * (1 + Math.max(0, asym) * 0.08),
    -sx * (1 - asym * 0.12), 0, 0,
    0, 0, -sz * (1 + Math.max(0, -asym) * 0.08),
    0, -sy, 0,
    sx + swing * 0.16, -sy * 0.28, reach * 0.34,
    -sx + counterSwing * 0.16, -sy * 0.28, -reach * 0.34,
    swing * 0.12, -sy * 0.72, sz + reach * 0.38,
    counterSwing * 0.12, -sy * 0.72, -sz - reach * 0.38,
  ];
  const indices = [
    0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1,
    5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
    1, 6, 2, 3, 7, 4, 2, 8, 5, 4, 9, 5,
  ];
  const width = Math.max(sx * 2, sx * 2 + Math.abs(swing) * 0.22);
  const depth = Math.max(sz * 2, sz * 2 + reach * 0.76);
  const height = sy * 2;
  return {
    shape: 'custom',
    size: [width, height, depth],
    rotation: [
      Math.sin(phase) * body.jointSwing * 0.08,
      asym * 0.32,
      Math.cos(phase) * body.jointSwing * 0.05,
    ],
    geometry: { vertices, indices },
    physics: {
      mass: body.mass,
      drag: body.drag,
      collisionRadius: Math.max(width, depth) * 0.5,
      gaitDrive: body.gaitDrive,
    },
    rig: {
      jointPhase: body.jointPhase,
      jointSwing: body.jointSwing,
      limbReach: body.limbReach,
      rigWeight: body.rigWeight,
    },
  };
};

export const stepEmbodiedCreature = (
  creature: EmbodiedSnnCreature,
  deltaSeconds: number,
  goal: SnnLearningGoal = 'wander',
  stimulus: SnnEnvironmentStimulus = { nearestIntensity: 0, ambientIntensity: 0, overload: 0 },
) => {
  const dt = Math.min(deltaSeconds, 0.05);
  if (goal === 'wander' && (creature.network.tMs >= creature.retargetAtMs || Math.hypot(creature.wanderX - creature.x, creature.wanderZ - creature.z) < 0.7)) {
    nextWanderTarget(creature);
  }

  let desiredX = creature.wanderX;
  let desiredZ = creature.wanderZ;
  if (goal === 'seekStimulus' && stimulus.nearestX !== undefined && stimulus.nearestZ !== undefined) {
    desiredX = stimulus.nearestX;
    desiredZ = stimulus.nearestZ;
  } else if (goal === 'avoidOverload') {
    const awayX = stimulus.nearestX === undefined ? creature.x - creature.wanderX : creature.x - stimulus.nearestX;
    const awayZ = stimulus.nearestZ === undefined ? creature.z - creature.wanderZ : creature.z - stimulus.nearestZ;
    const length = Math.hypot(awayX, awayZ) || 1;
    desiredX = creature.x + (awayX / length) * 4;
    desiredZ = creature.z + (awayZ / length) * 4;
  }

  const beforeDistance = Math.hypot(desiredX - creature.x, desiredZ - creature.z);
  const stimulusDistanceBefore = stimulus.nearestX === undefined || stimulus.nearestZ === undefined
    ? Infinity
    : Math.hypot(stimulus.nearestX - creature.x, stimulus.nearestZ - creature.z);
  const dx = desiredX - creature.x;
  const dz = desiredZ - creature.z;
  const headingError = Math.atan2(dx, dz);
  const currents = new Array(creature.network.neurons.length).fill(0);
  const body = ensureBodyState(creature);
  const bodyExtent = (body.widthX + body.heightY + body.depthZ) / 3;
  const bodyCompact = clamp(1.15 - bodyExtent + (stimulus.collisionIntensity ?? 0) * 0.22, 0, 1.4);
  const bodyExtended = clamp(bodyExtent - 0.86 + body.limbReach * 0.18 + body.gaitDrive * 0.12, 0, 1.4);
  const jointMotion = clamp(Math.abs(body.jointSwing) * 0.72 + body.gaitDrive * 0.26 + Math.abs(body.asymmetry) * 0.16, 0, 1.4);
  const rigLoad = clamp(body.mass * 0.22 + body.drag * 0.9 + body.deformation * 0.45 + (stimulus.collisionIntensity ?? 0) * 0.24, 0, 1.4);

  if (headingError < -0.08) {
    currents[0] = clamp(Math.abs(headingError) / Math.PI + 0.35, 0, 1.35);
    currents[1] = -clamp(Math.abs(headingError) / Math.PI, 0, 0.85);
  }
  if (headingError > 0.08) {
    currents[1] = clamp(Math.abs(headingError) / Math.PI + 0.35, 0, 1.35);
    currents[0] = -clamp(Math.abs(headingError) / Math.PI, 0, 0.85);
  }
  if (beforeDistance > 0.3) currents[2] = clamp(beforeDistance / 6, 0, 1.2);

  if (stimulus.nearestX !== undefined && stimulus.nearestZ !== undefined && stimulus.nearestIntensity > 0.02) {
    const stimulusHeading = Math.atan2(stimulus.nearestX - creature.x, stimulus.nearestZ - creature.z);
    if (stimulusHeading < -0.08) {
      currents[8] = clamp(Math.abs(stimulusHeading) / Math.PI + stimulus.nearestIntensity, 0, 1.4);
      currents[9] = -clamp(stimulus.nearestIntensity, 0, 0.9);
    }
    if (stimulusHeading > 0.08) {
      currents[9] = clamp(Math.abs(stimulusHeading) / Math.PI + stimulus.nearestIntensity, 0, 1.4);
      currents[8] = -clamp(stimulus.nearestIntensity, 0, 0.9);
    }
  }
  currents[10] = clamp(stimulus.ambientIntensity, 0, 1.4);
  currents[11] = clamp(stimulus.overload, 0, 1.4);
  currents[14] = clamp(stimulus.collisionIntensity ?? 0, 0, 1.4);
  currents[15] = bodyCompact;
  currents[16] = bodyExtended;
  currents[17] = jointMotion;
  currents[18] = rigLoad;

  const events = stepNetwork(creature.network, currents);
  const activation = (label: string) => activationFromEvents(creature.network, events, label);
  const turn = activation('motor:right') - activation('motor:left');
  const forward = beforeDistance > 0.2
    ? clamp(Math.max(activation('motor:forward'), currents[2] * 0.7), 0, 1)
    : 0;
  updateBodyState(creature, events, dt, forward, turn, stimulus);
  const bodyAfter = ensureBodyState(creature);
  const bodyDrive = clamp(0.82 + bodyAfter.gaitDrive * 0.24 - bodyAfter.drag * 0.18 - Math.max(0, bodyAfter.mass - 1) * 0.08, 0.42, 1.28);
  const moveSpeed = 1.45 * dt * forward * bodyDrive;
  const turnStep = 0.9 * dt * turn * clamp(1 + bodyAfter.asymmetry * 0.12, 0.72, 1.28);
  const direction = headingError + turnStep;

  creature.x += Math.sin(direction) * moveSpeed;
  creature.z += Math.cos(direction) * moveSpeed;
  creature.energy = clamp(creature.energy - 0.002 * dt - 0.003 * dt * forward, 0.05, 1);

  const afterDistance = Math.hypot(desiredX - creature.x, desiredZ - creature.z);
  const stimulusDistanceAfter = stimulus.nearestX === undefined || stimulus.nearestZ === undefined
    ? Infinity
    : Math.hypot(stimulus.nearestX - creature.x, stimulus.nearestZ - creature.z);
  const bodyStability = clamp(1 - bodyAfter.deformation * 0.58 - Math.abs(bodyAfter.asymmetry) * 0.18 - (stimulus.collisionIntensity ?? 0) * 0.34, 0, 1);
  const gaitReward = clamp(bodyAfter.gaitDrive * 0.1 + bodyStability * 0.08 - bodyAfter.drag * 0.05, -0.16, 0.18);
  const reward = (() => {
    const collisionPenalty = clamp((stimulus.collisionIntensity ?? 0) * 0.6, 0, 0.6);
    if (goal === 'wander') return (afterDistance < beforeDistance ? 1.15 : 0.55) + gaitReward;
    if (goal === 'seekStimulus') {
      const noveltyReward = stimulusDistanceAfter < stimulusDistanceBefore ? 1.35 : 0.45;
      return noveltyReward + clamp(stimulus.nearestIntensity * 0.25, 0, 0.25) - collisionPenalty + gaitReward;
    }
    if (goal === 'avoidOverload') {
      if (stimulus.overload < 0.25) return 1.05 + bodyStability * 0.08;
      return (stimulusDistanceAfter > stimulusDistanceBefore ? 1.35 : 0.3) - collisionPenalty + gaitReward;
    }
    return 0.7 - collisionPenalty + gaitReward;
  })();
  const salience = clamp(
    stimulus.nearestIntensity * 0.35
      + stimulus.ambientIntensity * 0.2
      + stimulus.overload * 0.28
      + (stimulus.collisionIntensity ?? 0) * 0.45
      + forward * 0.12
      + bodyAfter.deformation * 0.18
      + bodyAfter.gaitDrive * 0.08,
    0,
    1,
  );
  updateNeuromodulators(creature.network, reward, salience);
  creature.network.rewardSignal = Math.max(0.05, reward) * (0.35 + creature.energy * 0.65);
  applyDelayedReward(creature.network, events, reward - 0.7);
  consolidateEmbodiedMemory(creature.network, events);

  record(events, creature.network, 'global_signal', creature.network.rewardSignal, { label: 'reward' });
  record(events, creature.network, 'neuromodulator', creature.network.neuromodulators.dopamine, { label: 'dopamine' });
  record(events, creature.network, 'neuromodulator', creature.network.neuromodulators.acetylcholine, { label: 'acetylcholine' });
  record(events, creature.network, 'body', afterDistance, {
    label: creature.name,
    meta: {
      x: creature.x,
      z: creature.z,
      desired_x: desiredX,
      desired_z: desiredZ,
      energy: creature.energy,
      goal,
      nearest_stimulus: stimulus.nearestIntensity,
      ambient_stimulus: stimulus.ambientIntensity,
      overload: stimulus.overload,
      collision: stimulus.collisionIntensity ?? 0,
      body_width: bodyAfter.widthX,
      body_height: bodyAfter.heightY,
      body_depth: bodyAfter.depthZ,
      body_asymmetry: bodyAfter.asymmetry,
      joint_phase: bodyAfter.jointPhase,
      joint_swing: bodyAfter.jointSwing,
      limb_reach: bodyAfter.limbReach,
      rig_weight: bodyAfter.rigWeight,
      deformation: bodyAfter.deformation,
      gait_drive: bodyAfter.gaitDrive,
      mass: bodyAfter.mass,
      drag: bodyAfter.drag,
      body_stability: bodyStability,
    },
  });

  return events;
};
