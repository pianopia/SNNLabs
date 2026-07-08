export type BrowserOperation =
  | 'navigate'
  | 'search'
  | 'click'
  | 'read'
  | 'submit'
  | 'observe';

export type LanguageTokenNeuron = {
  id: number;
  token: string;
  count: number;
  v: number;
  threshold: number;
  lastSeenStep: number;
  positiveSpikeMass: number;
  negativeSpikeMass: number;
};

export type LanguageAssociation = {
  id: number;
  pre: number;
  post: number;
  w: number;
  aPre: number;
  aPost: number;
};

export type BrowserLanguageObservation = {
  step: number;
  operation: BrowserOperation;
  source: string;
  text: string;
  reward: number;
  tokenCount: number;
};

export type LanguageSnnTraceEvent = {
  step: number;
  kind: 'operation' | 'token_spike' | 'association' | 'reward';
  value: number;
  label: string;
  meta?: Record<string, number | string>;
};

export type BrowserLanguageSnnSnapshot = {
  version: 1;
  domain: 'browser-language';
  savedAt: number;
  config: {
    neuronType: 'si-lif';
    spikeRangeD: number;
    maxVocabulary: number;
    learningRate: number;
  };
  stats: {
    steps: number;
    observations: number;
    totalTokens: number;
    positiveSpikes: number;
    negativeSpikes: number;
    sparseAcOps: number;
    denseMacOps: number;
  };
  vocabulary: LanguageTokenNeuron[];
  associations: LanguageAssociation[];
  observations: BrowserLanguageObservation[];
};

export type BrowserLanguageSnnState = BrowserLanguageSnnSnapshot;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const normalizeToken = (token: string) => token.trim().toLowerCase();

const languageSegmenter = typeof Intl !== 'undefined' && 'Segmenter' in Intl
  ? new Intl.Segmenter(['ja', 'en'], { granularity: 'word' })
  : null;

const tokenizeLanguage = (text: string) => {
  const source = text.replace(/\s+/g, ' ').trim();
  if (!source) return [];

  if (languageSegmenter) {
    return Array.from(languageSegmenter.segment(source)).flatMap((segment) => {
      if (!segment.isWordLike) return [];
      const token = normalizeToken(segment.segment);
      return token ? [token] : [];
    });
  }

  return source.split(/[^\p{L}\p{N}_-]+/u).flatMap((token) => {
    const normalized = normalizeToken(token);
    return normalized ? [normalized] : [];
  });
};

export const createBrowserLanguageSnn = (maxVocabulary = 128): BrowserLanguageSnnState => ({
  version: 1,
  domain: 'browser-language',
  savedAt: Date.now(),
  config: {
    neuronType: 'si-lif',
    spikeRangeD: 4,
    maxVocabulary,
    learningRate: 0.18,
  },
  stats: {
    steps: 0,
    observations: 0,
    totalTokens: 0,
    positiveSpikes: 0,
    negativeSpikes: 0,
    sparseAcOps: 0,
    denseMacOps: 0,
  },
  vocabulary: [],
  associations: [],
  observations: [],
});

const findOrCreateNeuron = (state: BrowserLanguageSnnState, token: string) => {
  let neuron = state.vocabulary.find((item) => item.token === token);
  if (neuron) return neuron;

  if (state.vocabulary.length >= state.config.maxVocabulary) {
    neuron = state.vocabulary[0];
    for (const candidate of state.vocabulary) {
      if (candidate.count < neuron.count || (candidate.count === neuron.count && candidate.lastSeenStep < neuron.lastSeenStep)) {
        neuron = candidate;
      }
    }
    neuron.token = token;
    neuron.count = 0;
    neuron.v = 0;
    neuron.positiveSpikeMass = 0;
    neuron.negativeSpikeMass = 0;
    return neuron;
  }

  neuron = {
    id: state.vocabulary.length,
    token,
    count: 0,
    v: 0,
    threshold: 1,
    lastSeenStep: 0,
    positiveSpikeMass: 0,
    negativeSpikeMass: 0,
  };
  state.vocabulary.push(neuron);
  return neuron;
};

const findOrCreateAssociation = (state: BrowserLanguageSnnState, pre: number, post: number) => {
  let association = state.associations.find((item) => item.pre === pre && item.post === post);
  if (association) return association;
  association = {
    id: state.associations.length,
    pre,
    post,
    w: 0.08,
    aPre: 0,
    aPost: 0,
  };
  state.associations.push(association);
  return association;
};

export const observeBrowserLanguage = (
  state: BrowserLanguageSnnState,
  params: {
    operation: BrowserOperation;
    source: string;
    text: string;
    reward: number;
  },
) => {
  const events: LanguageSnnTraceEvent[] = [];
  const tokens = tokenizeLanguage(params.text).slice(0, 256);
  const reward = clamp(params.reward, -1, 1);
  state.stats.steps += 1;
  state.stats.observations += 1;
  state.stats.totalTokens += tokens.length;
  state.stats.denseMacOps += Math.max(1, state.associations.length);
  state.savedAt = Date.now();

  state.observations.unshift({
    step: state.stats.steps,
    operation: params.operation,
    source: params.source,
    text: params.text.slice(0, 800),
    reward,
    tokenCount: tokens.length,
  });
  state.observations = state.observations.slice(0, 40);

  events.push({
    step: state.stats.steps,
    kind: 'operation',
    value: tokens.length,
    label: params.operation,
    meta: { source: params.source },
  });

  let previousNeuron: LanguageTokenNeuron | null = null;
  for (const token of tokens) {
    const neuron = findOrCreateNeuron(state, token);
    neuron.count += 1;
    neuron.lastSeenStep = state.stats.steps;
    neuron.v = neuron.v * 0.72 + 0.55 + Math.abs(reward) * 0.25;
    const signedSpike = clamp(Math.round(neuron.v * (reward < 0 ? -1 : 1)), -state.config.spikeRangeD, state.config.spikeRangeD);
    if (signedSpike !== 0) {
      neuron.v = 0;
      if (signedSpike > 0) {
        neuron.positiveSpikeMass += signedSpike;
        state.stats.positiveSpikes += signedSpike;
      } else {
        neuron.negativeSpikeMass += Math.abs(signedSpike);
        state.stats.negativeSpikes += Math.abs(signedSpike);
      }
      events.push({
        step: state.stats.steps,
        kind: 'token_spike',
        value: signedSpike,
        label: token,
        meta: { neuron: neuron.id },
      });
    }

    if (previousNeuron) {
      const association = findOrCreateAssociation(state, previousNeuron.id, neuron.id);
      association.aPre = association.aPre * 0.82 + 0.08;
      association.aPost = association.aPost * 0.82 + 0.08;
      association.w = clamp(association.w + state.config.learningRate * (0.3 + association.aPre + association.aPost) * reward, -1, 1);
      state.stats.sparseAcOps += Math.max(1, Math.abs(signedSpike));
      events.push({
        step: state.stats.steps,
        kind: 'association',
        value: association.w,
        label: `${previousNeuron.token} -> ${neuron.token}`,
        meta: { pre: previousNeuron.id, post: neuron.id },
      });
    }
    previousNeuron = neuron;
  }

  events.push({
    step: state.stats.steps,
    kind: 'reward',
    value: reward,
    label: 'language_reward',
  });

  return events;
};

export const snapshotBrowserLanguageSnn = (state: BrowserLanguageSnnState): BrowserLanguageSnnSnapshot => ({
  ...state,
  savedAt: Date.now(),
  config: { ...state.config },
  stats: { ...state.stats },
  vocabulary: state.vocabulary.map((neuron) => ({ ...neuron })),
  associations: state.associations.map((association) => ({ ...association })),
  observations: state.observations.map((observation) => ({ ...observation })),
});
