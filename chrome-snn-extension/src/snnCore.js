export const EDEN_SNN_MAGIC = 'EDENSNN1';
export const MODEL_STORAGE_KEY = 'elfentierBrowserSnnModel';
export const EVENT_STORAGE_KEY = 'elfentierBrowserSnnEvents';
export const SETTINGS_STORAGE_KEY = 'elfentierBrowserSnnLearningSettings';
export const MODEL_INDEX_STORAGE_KEY = 'elfentierBrowserSnnModelIndex';
export const TAB_STATE_STORAGE_KEY = 'elfentierBrowserSnnTabStates';
export const DEFAULT_MODEL_ID = 'default';

export const modelStorageKey = (modelId = DEFAULT_MODEL_ID) =>
  modelId === DEFAULT_MODEL_ID ? MODEL_STORAGE_KEY : `${MODEL_STORAGE_KEY}:${modelId}`;

export const eventStorageKey = (modelId = DEFAULT_MODEL_ID) =>
  modelId === DEFAULT_MODEL_ID ? EVENT_STORAGE_KEY : `${EVENT_STORAGE_KEY}:${modelId}`;

export const DEFAULT_LEARNING_SETTINGS = {
  learningEnabled: false,
  autonomousExploreEnabled: false,
  autonomousIntervalSeconds: 18,
  autonomousMaxNavigations: 24,
  inputCapture: 'metadata',
  sensitiveTextMode: 'redact',
  moralLearningMode: 'shape',
  maxVocabulary: 0,
  computeBackend: 'auto',
  performanceBudgetPercent: 65,
  privacySensitivity: 0.9,
  harmSensitivity: 0.55,
  deceptionSensitivity: 0.58,
  consentSensitivity: 0.62,
  prosocialWeight: 0.18,
  moralPenaltyWeight: 0.42,
  sensitiveValueReward: 0.24,
  instinctLearningMode: 'shape',
  crisisSensitivity: 0.65,
  instinctPenaltyWeight: 0.48,
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const withLearningSettings = (settings = {}) => ({ ...DEFAULT_LEARNING_SETTINGS, ...(settings || {}) });
const normalizeToken = (token) => token.trim().toLowerCase();
const STRUCTURAL_TOKENS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'so', 'because', 'if', 'then', 'than', 'to', 'of', 'in', 'on', 'at',
  'for', 'from', 'by', 'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that',
  'these', 'those', 'it', 'its', 'i', 'you', 'we', 'they', 'he', 'she',
  'の', 'に', 'は', 'を', 'が', 'と', 'で', 'も', 'へ', 'や', 'から', 'まで', 'より', 'です', 'ます',
  'そして', 'しかし', 'また', 'つまり', 'だから', 'ので', 'ため', 'こと', 'これ', 'それ', 'あれ'
]);

const tokenWeight = (token) => {
  const value = String(token || '');
  if (value.startsWith('image:') || value.startsWith('audio:') || value.startsWith('video:')) return 0.86;
  if (value.startsWith('body:')) return 0.72;
  if (value.startsWith('media:') || value.startsWith('moral:') || value.startsWith('source:')) return 0.72;
  if (value.startsWith('lex:')) return 0.7;
  if (value.startsWith('instinct:')) return 0.82;
  if (value.startsWith('visual:')) return 0.78;
  if (value.startsWith('context:')) return 0.58;
  if (STRUCTURAL_TOKENS.has(value)) return 0.16;
  if (value.length === 1 && /[\p{L}]/u.test(value)) return 0.28;
  return 1;
};

const isStructuralToken = (token) => tokenWeight(token) < 0.3;
const isDiscoveryToken = (token) => {
  const value = String(token || '');
  if (value.startsWith('context:') || value.startsWith('visual:') || value.startsWith('media:')) return false;
  if (value.startsWith('image:') || value.startsWith('audio:') || value.startsWith('video:') || value.startsWith('body:')) return false;
  if (value.startsWith('moral:') || value.startsWith('instinct:') || value.startsWith('module:')) return false;
  if (value.startsWith('lex:') || value.startsWith('source:')) return false;
  return tokenWeight(value) >= 0.3;
};
const pageOrigin = (url) => {
  try {
    return url ? new URL(url).origin : 'unknown';
  } catch {
    return 'unknown';
  }
};

const isMediaEvent = (eventType) => String(eventType || '').startsWith('media_');
const isVisualEvent = (eventType) => String(eventType || '').startsWith('visual_')
  || String(eventType || '').startsWith('text_visual')
  || String(eventType || '') === 'autonomous_visual'
  || Boolean(eventType && String(eventType).startsWith('camera_'));

const eventVisualModality = (event) => {
  const kind = String(event.visualKind || '').toLowerCase();
  if (kind.includes('image')) return 'image';
  if (kind.includes('video') || kind.includes('camera') || kind.includes('stream')) return 'video';
  if (isVisualEvent(event.eventType)) return 'image';
  return '';
};

const tokenModalities = (token, event) => {
  const value = String(token || '');
  const modalities = new Set();
  if (value.startsWith('visual:')) {
    const visual = eventVisualModality(event);
    if (visual) modalities.add(visual);
    else modalities.add('image');
  } else if (value.startsWith('image:')) {
    modalities.add('image');
  } else if (value.startsWith('audio:')) {
    modalities.add('audio');
  } else if (value.startsWith('video:')) {
    modalities.add('video');
  } else if (value.startsWith('body:')) {
    modalities.add('body');
  } else if (value.startsWith('media:')) {
    const kind = String(event.mediaKind || '').toLowerCase();
    if (kind === 'audio') modalities.add('audio');
    else if (kind === 'video') modalities.add('video');
  } else if (value.startsWith('source:caption') || value.startsWith('source:transcript') || value.startsWith('source:visual-text') || value.startsWith('source:text-region')) {
    modalities.add('text');
  } else if (
    !value.includes(':')
    || value.startsWith('lex:')
  ) {
    modalities.add('text');
  }
  return [...modalities];
};

const mediaStateTokens = (event) => {
  if (!isMediaEvent(event.eventType)) return [];
  const kind = event.mediaKind || 'media';
  const progress = clamp(Number(event.mediaProgress || 0), 0, 1);
  const progressBucket = Math.floor(progress * 10);
  const rate = Number(event.mediaPlaybackRate || 1);
  const volume = clamp(Number(event.mediaVolume || 0), 0, 1);
  const duration = Number(event.mediaDuration || 0);
  const sizeLabel = Number(event.mediaWidth || 0) >= 1280 ? 'wide'
    : Number(event.mediaWidth || 0) > 0 ? 'small'
      : 'unknown-size';
  return [
    `media:${kind}`,
    `media:event:${event.eventType}`,
    `media:progress:${progressBucket}`,
    `media:rate:${rate.toFixed(1)}`,
    `media:volume:${Math.round(volume * 10)}`,
    `media:${event.mediaPaused ? 'paused' : 'playing'}`,
    `media:${event.mediaMuted ? 'muted' : 'audible'}`,
    `media:size:${sizeLabel}`,
    event.mediaCueText ? 'media:has-caption' : 'media:no-caption',
    event.mediaTranscriptText ? 'media:has-transcript' : 'media:no-transcript',
    duration > 0 ? `media:duration:${duration > 1800 ? 'long' : duration > 300 ? 'medium' : 'short'}` : 'media:duration:stream',
  ];
};

const visualStateTokens = (event) => {
  const supplied = Array.isArray(event.visualTokens) ? event.visualTokens.filter(Boolean).map(String) : [];
  if (supplied.length === 0 && !isVisualEvent(event.eventType) && !event.visualKind) return [];
  const width = Number(event.visualWidth || 0);
  const height = Number(event.visualHeight || 0);
  const visible = clamp(Number(event.visualVisibleRatio || 0), 0, 1);
  const aspect = width > 0 && height > 0 ? width / height : 1;
  return [
    'visual:cortex:v1',
    `visual:event:${event.eventType || 'unknown'}`,
    `visual:kind:${event.visualKind || 'unknown'}`,
    `visual:readable:${event.visualReadablePixels ? 'yes' : 'no'}`,
    `visual:visible:${Math.round(visible * 10)}`,
    aspect > 1.7 ? 'visual:aspect:wide' : aspect < 0.75 ? 'visual:aspect:tall' : 'visual:aspect:balanced',
    width * height > 1200000 ? 'visual:resolution:high' : width * height > 180000 ? 'visual:resolution:medium' : 'visual:resolution:low',
    ...supplied,
  ];
};

const contextTokens = (event) => {
  const origin = pageOrigin(event.url);
  const urlHash = Array.from(origin).reduce((hash, char) => ((hash * 31) + char.charCodeAt(0)) % 997, 7);
  const viewport = Number(event.viewportHeight || 0) >= 900 ? 'tall'
    : Number(event.viewportHeight || 0) > 0 ? 'compact'
      : 'unknown';
  const scrollBucket = Math.floor(clamp(Number(event.scrollY || 0) / 6000, 0, 1) * 10);
  const tag = String(event.tagName || '').toLowerCase().replace(/[^a-z0-9_-]/g, '').slice(0, 24);
  return [
    `context:event:${event.eventType || 'unknown'}`,
    `context:origin:${urlHash}`,
    `context:viewport:${viewport}`,
    `context:scroll:${scrollBucket}`,
    ...(tag ? [`context:tag:${tag}`] : []),
  ];
};

const segmenter = typeof Intl !== 'undefined' && 'Segmenter' in Intl
  ? new Intl.Segmenter(['ja', 'en'], { granularity: 'word' })
  : null;

export const createModel = () => ({
  version: 1,
  domain: 'browser-language',
  savedAt: Date.now(),
  source: 'chrome-extension',
  config: {
    neuronType: 'si-lif',
    spikeRangeD: 4,
    maxVocabulary: 0,
    learningRate: 0.14,
    homeostasisInterval: 16,
    consolidationInterval: 48,
    delayedRewardWindow: 32,
    eligibilityDecay: 0.88,
    delayedLearningRate: 0.045,
    instinctDelayedWindow: 48,
    sleepReplayCycles: 3,
    targetFiringRate: 0.04,
    maxOutgoingWeight: 4.5,
    moralPenaltyWeight: 0.42,
    moralProsocialWeight: 0.18,
    maxObservationMemory: 180,
  },
  moralState: {
    harmAversion: 0.55,
    privacyRespect: 0.72,
    consentSensitivity: 0.62,
    honestyBias: 0.58,
    prosocialDrive: 0.35,
  },
  instinctState: {
    threatAversion: 0.62,
    noveltyCaution: 0.48,
    escapeDrive: 0.55,
  },
  instinct: {
    actionTrace: [],
  },
  neuromodulators: {
    dopamine: 0,
    acetylcholine: 0.25,
    norepinephrine: 0,
    serotonin: 0.35,
    fatigue: 0,
  },
  stats: {
    steps: 0,
    observations: 0,
    totalTokens: 0,
    positiveSpikes: 0,
    negativeSpikes: 0,
    sparseAcOps: 0,
    denseMacOps: 0,
    userEvents: 0,
    autonomousEvents: 0,
    mediaEvents: 0,
    visualEvents: 0,
    crossModalRelations: 0,
    crossModalUpdates: 0,
    pages: 0,
    consolidationCycles: 0,
    stabilizedSynapses: 0,
    delayedRewards: 0,
    delayedCreditAssignments: 0,
    rewardPredictionError: 0,
    sleepCycles: 0,
    sleepReplayedSynapses: 0,
    sleepScaledSynapses: 0,
    sleepInhibitedSynapses: 0,
    moralEvents: 0,
    moralPenalties: 0,
    instinctEvents: 0,
    instinctAvoidances: 0,
    instinctDelayedAvoidance: 0,
    discoveryRewards: 0,
    novelWordsRewarded: 0,
    newPageDiscoveries: 0,
    lexicalLinks: 0,
    privacyRedactions: 0,
    analogDecisions: 0,
    analogDrive: 0,
    synapticScaled: 0,
    modelBytes: 0,
    lastStepMs: 0,
    avgStepMs: 0,
    cpuLoadEstimate: 0,
    computeBackend: 'cpu',
    performanceBudgetPercent: 65,
    learningThrottleMs: 0,
    gpuTokenLimit: 0,
    gpuThrottleMs: 0,
  },
  vocabulary: [],
  associations: [],
  crossModalRelations: [],
  eligibility: {
    synapses: [],
  },
  observations: [],
  pageStats: {},
});

const normalizeCollection = (value) => {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== 'object') return [];
  return Object.values(value);
};

const ensureBrainFields = (model) => {
  model.config ??= {};
  model.config.homeostasisInterval ??= 16;
  model.config.maxVocabulary ??= 0;
  model.config.learningRate ??= 0.14;
  model.config.spikeRangeD ??= 4;
  model.config.consolidationInterval ??= 48;
  model.config.delayedRewardWindow ??= 32;
  model.config.eligibilityDecay ??= 0.88;
  model.config.delayedLearningRate ??= 0.045;
  model.config.instinctDelayedWindow ??= 48;
  model.config.sleepReplayCycles ??= 3;
  model.config.targetFiringRate ??= 0.04;
  model.config.maxOutgoingWeight ??= 4.5;
  model.config.moralPenaltyWeight ??= 0.42;
  model.config.moralProsocialWeight ??= 0.18;
  model.config.maxObservationMemory ??= 180;
  model.moralState ??= {};
  model.moralState.harmAversion ??= 0.55;
  model.moralState.privacyRespect ??= 0.72;
  model.moralState.consentSensitivity ??= 0.62;
  model.moralState.honestyBias ??= 0.58;
  model.moralState.prosocialDrive ??= 0.35;
  model.instinctState ??= {};
  model.instinctState.threatAversion ??= 0.62;
  model.instinctState.noveltyCaution ??= 0.48;
  model.instinctState.escapeDrive ??= 0.55;
  model.instinct ??= {};
  model.instinct.actionTrace ??= [];
  model.neuromodulators ??= {};
  model.neuromodulators.dopamine ??= 0;
  model.neuromodulators.acetylcholine ??= 0.25;
  model.neuromodulators.norepinephrine ??= 0;
  model.neuromodulators.serotonin ??= 0.35;
  model.neuromodulators.fatigue ??= 0;
  model.stats ??= {};
  model.stats.mediaEvents ??= 0;
  model.stats.visualEvents ??= 0;
  model.stats.crossModalRelations ??= 0;
  model.stats.crossModalUpdates ??= 0;
  model.stats.autonomousEvents ??= 0;
  model.stats.consolidationCycles ??= 0;
  model.stats.stabilizedSynapses ??= 0;
  model.stats.delayedRewards ??= 0;
  model.stats.delayedCreditAssignments ??= 0;
  model.stats.rewardPredictionError ??= 0;
  model.stats.sleepCycles ??= 0;
  model.stats.sleepReplayedSynapses ??= 0;
  model.stats.sleepScaledSynapses ??= 0;
  model.stats.sleepInhibitedSynapses ??= 0;
  model.stats.moralEvents ??= 0;
  model.stats.moralPenalties ??= 0;
  model.stats.instinctEvents ??= 0;
  model.stats.instinctAvoidances ??= 0;
  model.stats.instinctDelayedAvoidance ??= 0;
  model.stats.discoveryRewards ??= 0;
  model.stats.novelWordsRewarded ??= 0;
  model.stats.newPageDiscoveries ??= 0;
  model.stats.lexicalLinks ??= 0;
  model.stats.privacyRedactions ??= 0;
  model.stats.analogDecisions ??= 0;
  model.stats.analogDrive ??= 0;
  model.stats.synapticScaled ??= 0;
  model.stats.modelBytes ??= 0;
  model.stats.lastStepMs ??= 0;
  model.stats.avgStepMs ??= 0;
  model.stats.cpuLoadEstimate ??= 0;
  model.stats.computeBackend ??= 'cpu';
  model.stats.performanceBudgetPercent ??= 65;
  model.stats.learningThrottleMs ??= 0;
  model.stats.gpuTokenLimit ??= 0;
  model.stats.gpuThrottleMs ??= 0;
  model.stats.gpuThrottledSteps ??= 0;
  model.eligibility ??= {};
  model.eligibility.synapses ??= [];
  model.vocabulary = normalizeCollection(model.vocabulary).map((neuron, index) => ({
    ...neuron,
    id: Number.isFinite(neuron.id) ? neuron.id : index,
  }));
  const usedNeuronIds = new Set();
  for (const neuron of model.vocabulary) {
    if (!Number.isFinite(neuron.id) || usedNeuronIds.has(neuron.id)) neuron.id = usedNeuronIds.size;
    usedNeuronIds.add(neuron.id);
    neuron.threshold ??= 1;
    neuron.stability ??= 0;
    neuron.lastConsolidatedStep ??= 0;
    neuron.importance ??= tokenWeight(neuron.token);
    neuron.role ??= isStructuralToken(neuron.token) ? 'structural' : 'semantic';
  }
  model.associations = normalizeCollection(model.associations).filter((edge) => edge && edge.pre !== undefined && edge.post !== undefined);
  for (const edge of model.associations) {
    edge.stability ??= 0;
    edge.replayCount ??= 0;
    edge.lastUpdatedStep ??= 0;
    edge.d1Go ??= Math.max(0, edge.w || 0);
    edge.d2NoGo ??= Math.max(0, -(edge.w || 0));
    edge.rewardPrediction ??= 0;
  }
  model.crossModalRelations = normalizeCollection(model.crossModalRelations)
    .filter((relation) => relation && relation.a !== undefined && relation.b !== undefined);
  for (const relation of model.crossModalRelations) {
    relation.w ??= 0.04;
    relation.coactivity ??= 0;
    relation.stability ??= 0;
    relation.lastUpdatedStep ??= 0;
    relation.modalities ??= [relation.fromModality || 'unknown', relation.toModality || 'unknown'];
  }
};

const createRuntimeIndex = (model) => ({
  tokenToNeuron: new Map((model.vocabulary || []).map((neuron) => [neuron.token, neuron])),
  associationByKey: new Map((model.associations || []).map((edge) => [`${edge.pre}:${edge.post}`, edge])),
  crossModalByKey: new Map((model.crossModalRelations || []).map((relation) => [`${Math.min(relation.a, relation.b)}:${Math.max(relation.a, relation.b)}`, relation])),
});

const moralLexicon = {
  harm: [
    'harm', 'hurt', 'abuse', 'violence', 'weapon', 'attack', 'kill', 'suicide', 'self-harm',
    '危害', '暴力', '攻撃', '殺害', '自傷', '虐待', '脅迫'
  ],
  privacy: [
    'password', 'passcode', 'secret', 'token', 'api key', 'private key', 'credit card', 'ssn',
    'address', 'phone number', 'email address', 'credential',
    'パスワード', '秘密', '秘密鍵', '認証情報', '住所', '電話番号', 'メールアドレス', 'クレジットカード', 'マイナンバー'
  ],
  deception: [
    'scam', 'phishing', 'fake', 'fraud', 'impersonate', 'lie', 'deceive', 'steal account',
    '詐欺', 'フィッシング', 'なりすまし', '偽装', '嘘', '騙す'
  ],
  consent: [
    'consent', 'permission', 'opt-in', 'unauthorized', 'without permission',
    '同意', '許可', '無断', '承諾', '本人確認'
  ],
  prosocial: [
    'help', 'support', 'safety', 'protect', 'explain', 'consent', 'privacy', 'respect', 'care',
    '支援', '安全', '保護', '説明', '同意', 'プライバシー', '尊重', '配慮'
  ],
};

const sensitivePatterns = [
  /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
  /\b(?:\d[ -]*?){13,19}\b/g,
  /\b(?:sk|pk|rk|ghp|github_pat)_[A-Za-z0-9_\-]{16,}\b/g,
  /\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S+/gi,
  /\b\d{3}-\d{4}-\d{4}\b/g,
];

const countLexiconHits = (text, words) => {
  const source = text.toLowerCase();
  let hits = 0;
  for (const word of words) {
    if (source.includes(word.toLowerCase())) hits += 1;
  }
  return hits;
};

const redactSensitiveText = (text) => {
  let redacted = String(text ?? '');
  let count = 0;
  for (const pattern of sensitivePatterns) {
    redacted = redacted.replace(pattern, () => {
      count += 1;
      return '[redacted-sensitive]';
    });
  }
  return { text: redacted, count };
};

const moralAppraisal = (event, rawText, settings = DEFAULT_LEARNING_SETTINGS) => {
  const learningSettings = withLearningSettings(settings);
  const harm = countLexiconHits(rawText, moralLexicon.harm);
  const privacy = countLexiconHits(rawText, moralLexicon.privacy);
  const deception = countLexiconHits(rawText, moralLexicon.deception);
  const consent = countLexiconHits(rawText, moralLexicon.consent);
  const prosocial = countLexiconHits(rawText, moralLexicon.prosocial);
  const redacted = redactSensitiveText(rawText);
  const privacyRisk = clamp(((privacy + redacted.count * 2) / 5) * learningSettings.privacySensitivity, 0, 1);
  const harmRisk = clamp((harm / 4) * learningSettings.harmSensitivity, 0, 1);
  const deceptionRisk = clamp((deception / 4) * learningSettings.deceptionSensitivity, 0, 1);
  const consentRisk = clamp(((consent > 0 && rawText.toLowerCase().includes('without' ) ? consent + 1 : consent) / 4) * learningSettings.consentSensitivity, 0, 1);
  const prosocialSignal = clamp(prosocial / 5, 0, 1);
  const moralRisk = clamp(
    harmRisk * 0.45 + privacyRisk * 0.42 + deceptionRisk * 0.35 + consentRisk * 0.25,
    0,
    1
  );
  const sensitiveValue = clamp(privacyRisk * learningSettings.sensitiveValueReward, 0, 0.5);
  const moralValence = clamp(prosocialSignal * 0.42 + sensitiveValue - moralRisk, -1, 1);
  return {
    harmRisk,
    privacyRisk,
    deceptionRisk,
    consentRisk,
    prosocialSignal,
    moralRisk,
    moralValence,
    redactedText: redacted.text,
    redactionCount: redacted.count,
    eventType: event.eventType,
  };
};

const moralTokens = (appraisal) => {
  const tokens = [];
  if (appraisal.harmRisk > 0) tokens.push('moral:harm-risk');
  if (appraisal.privacyRisk > 0) tokens.push('moral:privacy-risk');
  if (appraisal.deceptionRisk > 0) tokens.push('moral:deception-risk');
  if (appraisal.consentRisk > 0) tokens.push('moral:consent-risk');
  if (appraisal.prosocialSignal > 0) tokens.push('moral:prosocial');
  if (appraisal.redactionCount > 0) tokens.push('moral:redacted-sensitive');
  return tokens;
};

const instinctLexicon = {
  danger: [
    'danger', 'warning', 'caution', 'hazard', 'emergency', 'alert', 'critical', 'unsafe',
    '危険', '警告', '注意', '緊急', '危害', '危ない'
  ],
  malware: [
    'malware', 'virus', 'infected', 'compromised', 'security alert', 'trojan', 'ransomware',
    'ウイルス', '感染', '不正', 'セキュリティ警告', 'マルウェア', 'ランサム'
  ],
  phishing: [
    'verify your account', 'account suspended', 'unusual activity', 'confirm identity', 'act now',
    '口座停止', 'アカウント確認', '今すぐ', '緊急確認', 'アカウントがロック', '本人確認', 'お支払い情報'
  ],
  trap: [
    'do not close', 'do not leave', 'your computer', 'call support', 'support line',
    'ページを閉じない', '今すぐお電話', '閉じないで', 'サポート窓口', '画面を閉じない'
  ],
  failure: [
    '404', '500', '502', '503', 'not found', 'access denied', 'connection failed', 'site blocked',
    '見つかりません', '接続できません', 'エラー', 'アクセス拒否', 'ブロックされ'
  ],
  shock: [
    'blood', 'gore', 'graphic', 'disturbing', 'viewer discretion',
    '残酷', '流血', '閲覧注意', 'グロ'
  ],
};

const isSuspiciousUrl = (url) => {
  try {
    const parsed = new URL(String(url || ''));
    const host = parsed.hostname.toLowerCase();
    if (['localhost', '127.0.0.1', '0.0.0.0'].includes(host)) return false;
    if (parsed.protocol === 'http:') return true;
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) return true;
    if (host.includes('xn--')) return true;
    const target = `${host}${parsed.pathname}`.toLowerCase();
    if (/login|verify|secure-update|account|wallet|password|signin|auth/i.test(target) && parsed.protocol === 'http:') return true;
    return false;
  } catch {
    return false;
  }
};

const instinctAppraisal = (event, rawText, settings = DEFAULT_LEARNING_SETTINGS, model = null, options = {}) => {
  const learningSettings = withLearningSettings(settings);
  const delayed = options.delayed === true;
  const danger = countLexiconHits(rawText, instinctLexicon.danger);
  const malware = countLexiconHits(rawText, instinctLexicon.malware);
  const phishing = countLexiconHits(rawText, instinctLexicon.phishing);
  const trap = countLexiconHits(rawText, instinctLexicon.trap);
  const failure = countLexiconHits(rawText, instinctLexicon.failure);
  const shock = countLexiconHits(rawText, instinctLexicon.shock);
  const crisisScores = {
    danger: clamp(danger / 3, 0, 1),
    malware: clamp(malware / 2, 0, 1),
    phishing: clamp(phishing / 2, 0, 1),
    trap: clamp(trap / 2, 0, 1),
    failure: clamp(failure / 2, 0, 1),
    shock: clamp(shock / 2, 0, 1),
  };
  const urlRisk = isSuspiciousUrl(event.url) ? 0.58 : 0;
  const navigationTrap = event.change?.urlChanged
    && event.change?.priorUrl
    && !isSuspiciousUrl(event.change.priorUrl)
    && isSuspiciousUrl(event.url) ? 0.72 : 0;
  const threatLevel = clamp(
    (
      crisisScores.danger * 0.36
      + crisisScores.malware * 0.52
      + crisisScores.phishing * 0.48
      + crisisScores.trap * 0.44
      + crisisScores.failure * 0.28
      + crisisScores.shock * 0.22
      + urlRisk * 0.62
      + navigationTrap * 0.68
    ) * learningSettings.crisisSensitivity,
    0,
    1
  );
  const instinctState = model?.instinctState || {};
  const avoidanceUrgency = clamp(
    threatLevel * (instinctState.threatAversion ?? 0.62)
    + (instinctState.escapeDrive ?? 0.55) * (threatLevel > 0 ? 0.12 : 0)
    + (delayed ? 0.08 : 0),
    0,
    1
  );
  const activeCrises = Object.entries(crisisScores)
    .filter(([, score]) => score > 0.15)
    .map(([kind]) => kind);
  const threatValence = threatLevel > 0 ? -clamp(threatLevel * avoidanceUrgency, 0, 1) : 0;
  return {
    crisisScores,
    activeCrises,
    threatLevel,
    avoidanceUrgency,
    threatValence,
    urlRisk,
    navigationTrap,
    evaluationMode: delayed ? 'delayed' : 'immediate',
    eventType: event.eventType,
  };
};

const instinctTokens = (appraisal) => {
  const tokens = [];
  if (appraisal.threatLevel <= 0) return tokens;
  tokens.push('instinct:threat-active');
  if (appraisal.avoidanceUrgency > 0.35) tokens.push('instinct:escape-urge');
  if (appraisal.evaluationMode === 'delayed') tokens.push('instinct:delayed-crisis');
  if (appraisal.urlRisk > 0) tokens.push('instinct:url-risk');
  if (appraisal.navigationTrap > 0) tokens.push('instinct:navigation-trap');
  for (const kind of appraisal.activeCrises) {
    tokens.push(`instinct:crisis:${kind}`);
  }
  return tokens;
};

const updateInstinctState = (model, instinct, settings = DEFAULT_LEARNING_SETTINGS) => {
  if (withLearningSettings(settings).instinctLearningMode === 'off') return;
  model.instinctState.threatAversion = clamp(
    (model.instinctState.threatAversion ?? 0.62) * 0.992 + instinct.threatLevel * 0.018,
    0.35,
    0.95
  );
  if (instinct.threatLevel > 0.35) {
    model.instinctState.escapeDrive = clamp(
      (model.instinctState.escapeDrive ?? 0.55) * 0.985 + instinct.avoidanceUrgency * 0.025,
      0.2,
      0.92
    );
  }
  if (instinct.navigationTrap > 0) {
    model.instinctState.noveltyCaution = clamp(
      (model.instinctState.noveltyCaution ?? 0.48) * 0.988 + instinct.navigationTrap * 0.02,
      0.2,
      0.9
    );
  }
};

const recordInstinctAction = (model, event, step) => {
  const actionable = ['click', 'autonomous_button', 'autonomous_link', 'autonomous_input'].includes(event.eventType);
  if (!actionable) return;
  const windowSize = model.config.instinctDelayedWindow || 48;
  model.instinct.actionTrace = [
    {
      step,
      eventType: event.eventType,
      url: event.url || '',
      elementText: String(event.elementText || '').slice(0, 200),
    },
    ...(model.instinct.actionTrace || []).filter((item) => step - item.step <= windowSize),
  ].slice(0, 32);
};

const crisisDelayedSignal = (model, event, settings = DEFAULT_LEARNING_SETTINGS) => {
  const learningSettings = withLearningSettings(settings);
  const eventType = String(event.eventType || '');
  let signal = 0;

  if (eventType.endsWith('_change')) {
    const change = event.change || {};
    const delayedInstinct = instinctAppraisal(
      event,
      String(change.addedText || ''),
      settings,
      model,
      { delayed: true }
    );
    if (delayedInstinct.threatLevel > 0.2) {
      signal -= delayedInstinct.threatLevel * delayedInstinct.avoidanceUrgency;
    }
    if (change.urlChanged && isSuspiciousUrl(event.url)) {
      signal -= 0.38 * learningSettings.crisisSensitivity;
    }
    if (change.priorUrl && !isSuspiciousUrl(change.priorUrl) && isSuspiciousUrl(event.url)) {
      signal -= 0.52 * learningSettings.crisisSensitivity;
    }
  }

  if (eventType === 'page_view') {
    const pageInstinct = instinctAppraisal(
      event,
      `${event.title || ''} ${event.visibleText || ''} ${event.url || ''}`,
      settings,
      model,
      { delayed: true }
    );
    if (pageInstinct.threatLevel > 0.32) {
      signal -= pageInstinct.threatLevel * 0.58;
    }
  }

  const nextStep = (model.stats.steps || 0) + 1;
  const windowSize = model.config.instinctDelayedWindow || 48;
  const recentAction = (model.instinct?.actionTrace || []).find((item) => nextStep - item.step <= windowSize);
  if (recentAction && signal < -0.06) {
    signal *= 1.28;
  }

  return clamp(signal * learningSettings.instinctPenaltyWeight, -1, 0);
};

export const tokenize = (text) => {
  const source = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!source) return [];
  if (segmenter) {
    return Array.from(segmenter.segment(source)).flatMap((part) => {
      if (!part.isWordLike) return [];
      const token = normalizeToken(part.segment);
      return token ? [token] : [];
    });
  }
  return source.split(/[^\p{L}\p{N}_-]+/u).flatMap((part) => {
    const token = normalizeToken(part);
    return token ? [token] : [];
  });
};

const semanticTokens = (text, limit = 160) => tokenize(text)
  .filter((token) => tokenWeight(token) >= 0.3)
  .slice(0, limit);

const lexicalPrefixLengths = (word) => {
  const normalized = normalizeToken(word);
  if (normalized.length < 3) return [normalized.length];
  const lengths = new Set([3, 4, Math.max(3, Math.floor(normalized.length * 0.45)), normalized.length]);
  return [...lengths].filter((len) => len >= 3 && len <= normalized.length);
};

const lexicalBridgeTokens = (word) => {
  const normalized = normalizeToken(word);
  if (!normalized || normalized.length < 2 || !isDiscoveryToken(normalized)) return [];
  const bridges = new Set([`lex:form:${normalized}`]);
  for (const len of lexicalPrefixLengths(normalized)) {
    bridges.add(`lex:prefix:${normalized.slice(0, len)}`);
  }
  return [...bridges];
};

const vocabularyLexicalBridges = (word, model) => {
  const normalized = normalizeToken(word);
  if (!normalized || normalized.length < 2) return [];
  const bridges = new Set();
  for (const neuron of model.vocabulary || []) {
    const existing = neuron.token;
    if (!existing || existing === normalized || !isDiscoveryToken(existing)) continue;
    if (existing.startsWith('lex:') || existing.includes(':')) continue;
    if (normalized.startsWith(existing) && normalized.length > existing.length + 1) {
      bridges.add(`lex:alias:${existing}`);
    }
    if (existing.startsWith(normalized) && existing.length > normalized.length + 1) {
      bridges.add(`lex:extends:${existing}`);
    }
  }
  return [...bridges].slice(0, 4);
};

const injectLexicalBridges = (tokens, model, limit = 160) => {
  const out = [];
  for (const token of tokens) {
    if (out.length >= limit) break;
    out.push(token);
    if (!isDiscoveryToken(token)) continue;
    for (const bridge of [...lexicalBridgeTokens(token), ...vocabularyLexicalBridges(token, model)]) {
      if (out.length >= limit) break;
      if (!out.includes(bridge)) out.push(bridge);
    }
  }
  return out;
};

const sharedPrefixLength = (a, b) => {
  const left = normalizeToken(a);
  const right = normalizeToken(b);
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return index;
};

const areLexicalRelatives = (a, b) => {
  const left = normalizeToken(a);
  const right = normalizeToken(b);
  if (!left || !right || left === right) return false;
  if (left.startsWith(right) || right.startsWith(left)) return true;
  const shared = sharedPrefixLength(left, right);
  return shared >= Math.min(4, Math.min(left.length, right.length));
};

const applyLexicalRelativeLinks = (model, words, step, reward, plasticityGate, runtimeIndex, trace) => {
  const semanticWords = [...new Set(words.filter((token) => isDiscoveryToken(token) && !String(token).startsWith('lex:')))];
  let links = 0;
  for (let i = 0; i < semanticWords.length; i += 1) {
    for (let j = i + 1; j < semanticWords.length; j += 1) {
      if (!areLexicalRelatives(semanticWords[i], semanticWords[j])) continue;
      const pre = findOrCreateNeuron(model, semanticWords[i], runtimeIndex);
      const post = findOrCreateNeuron(model, semanticWords[j], runtimeIndex);
      const association = findOrCreateAssociation(model, pre.id, post.id, runtimeIndex);
      const linkSignal = clamp(reward * 0.22 + 0.14, -0.35, 0.45);
      association.w = clamp(
        association.w + model.config.learningRate * plasticityGate * 0.42 * linkSignal,
        -1,
        1
      );
      association.lastUpdatedStep = step;
      links += 1;
      trace.push({
        step,
        kind: 'lexical_link',
        value: association.w,
        label: `${semanticWords[i]}~${semanticWords[j]}`,
        meta: { pre: semanticWords[i], post: semanticWords[j] },
      });
    }
  }
  return links;
};

export const estimateModelBytes = (model) => new TextEncoder().encode(JSON.stringify(model)).byteLength;

const findOrCreateNeuron = (model, token, runtimeIndex) => {
  let neuron = runtimeIndex?.tokenToNeuron?.get(token) || model.vocabulary.find((item) => item.token === token);
  if (neuron) return neuron;
  const maxVocabulary = Number(model.config.maxVocabulary || 0);
  if (maxVocabulary > 0 && model.vocabulary.length >= maxVocabulary) {
    neuron = model.vocabulary[0];
    for (const candidate of model.vocabulary) {
      const candidateScore = (candidate.count || 0) + (candidate.stability || 0) * 24 + Math.max(0, candidate.lastSeenStep || 0) * 0.002;
      const neuronScore = (neuron.count || 0) + (neuron.stability || 0) * 24 + Math.max(0, neuron.lastSeenStep || 0) * 0.002;
      if (candidateScore < neuronScore) {
        neuron = candidate;
      }
    }
    Object.assign(neuron, { token, count: 0, v: 0, positiveSpikeMass: 0, negativeSpikeMass: 0, stability: 0, threshold: 1 });
    neuron.importance = tokenWeight(token);
    neuron.role = isStructuralToken(token) ? 'structural' : 'semantic';
    runtimeIndex?.tokenToNeuron?.set(token, neuron);
    return neuron;
  }
  neuron = {
    id: model.vocabulary.length,
    token,
    count: 0,
    v: 0,
    threshold: 1,
    lastSeenStep: 0,
    positiveSpikeMass: 0,
    negativeSpikeMass: 0,
    stability: 0,
    importance: tokenWeight(token),
    role: isStructuralToken(token) ? 'structural' : 'semantic',
    lastConsolidatedStep: 0,
  };
  model.vocabulary.push(neuron);
  runtimeIndex?.tokenToNeuron?.set(token, neuron);
  return neuron;
};

const findOrCreateAssociation = (model, pre, post, runtimeIndex) => {
  const key = `${pre}:${post}`;
  let association = runtimeIndex?.associationByKey?.get(key) || model.associations.find((item) => item.pre === pre && item.post === post);
  if (association) return association;
  association = {
    id: model.associations.length,
    pre,
    post,
    w: 0.06,
    aPre: 0,
    aPost: 0,
    stability: 0,
    replayCount: 0,
    lastUpdatedStep: 0,
    d1Go: 0.08,
    d2NoGo: 0.02,
    rewardPrediction: 0,
  };
  model.associations.push(association);
  runtimeIndex?.associationByKey?.set(key, association);
  return association;
};

const findOrCreateCrossModalRelation = (model, left, right, leftModality, rightModality, runtimeIndex) => {
  const a = Math.min(left.id, right.id);
  const b = Math.max(left.id, right.id);
  const key = `${a}:${b}`;
  let relation = runtimeIndex?.crossModalByKey?.get(key)
    || model.crossModalRelations.find((item) => item.a === a && item.b === b);
  if (relation) return relation;
  relation = {
    id: model.crossModalRelations.length,
    a,
    b,
    modalities: left.id <= right.id ? [leftModality, rightModality] : [rightModality, leftModality],
    w: 0.04,
    coactivity: 0,
    stability: 0,
    lastUpdatedStep: 0,
    rewardPrediction: 0,
  };
  model.crossModalRelations.push(relation);
  runtimeIndex?.crossModalByKey?.set(key, relation);
  return relation;
};

const updateCrossModalRelations = (model, activeNeurons, step, reward, salience, plasticityGate, runtimeIndex, trace) => {
  const byModality = new Map();
  for (const item of activeNeurons) {
    for (const modality of item.modalities || []) {
      if (!['image', 'audio', 'video', 'text', 'body'].includes(modality)) continue;
      if (!byModality.has(modality)) byModality.set(modality, new Map());
      byModality.get(modality).set(item.neuron.id, item);
    }
  }
  const modalities = [...byModality.keys()];
  if (modalities.length < 2) return 0;
  let updates = 0;
  for (let i = 0; i < modalities.length; i += 1) {
    for (let j = i + 1; j < modalities.length; j += 1) {
      const leftItems = [...byModality.get(modalities[i]).values()]
        .sort((a, b) => b.drive - a.drive)
        .slice(0, 8);
      const rightItems = [...byModality.get(modalities[j]).values()]
        .sort((a, b) => b.drive - a.drive)
        .slice(0, 8);
      for (const left of leftItems) {
        for (const right of rightItems) {
          if (left.neuron.id === right.neuron.id) continue;
          const relation = findOrCreateCrossModalRelation(model, left.neuron, right.neuron, modalities[i], modalities[j], runtimeIndex);
          const coactivity = clamp(Math.sqrt(Math.max(0.01, left.drive) * Math.max(0.01, right.drive)), 0.01, 4);
          const predicted = relation.rewardPrediction || 0;
          const rpe = clamp(reward - predicted, -1, 1);
          relation.rewardPrediction = clamp(predicted * 0.94 + reward * 0.06, -1, 1);
          relation.coactivity = clamp((relation.coactivity || 0) * 0.88 + coactivity * 0.12, 0, 8);
          relation.w = clamp(
            (relation.w || 0)
              + model.config.learningRate
                * plasticityGate
                * (0.18 + salience * 0.24)
                * coactivity
                * (reward * 0.22 + rpe * 0.38),
            -1,
            1
          );
          relation.stability = clamp((relation.stability || 0) + Math.max(0, relation.w) * 0.002 + coactivity * 0.001, 0, 1);
          relation.lastUpdatedStep = step;
          updates += 1;
        }
      }
    }
  }
  if (model.crossModalRelations.length > 3000) {
    model.crossModalRelations = model.crossModalRelations
      .sort((a, b) => (Math.abs(b.w || 0) + (b.stability || 0) + (b.coactivity || 0) * 0.08) - (Math.abs(a.w || 0) + (a.stability || 0) + (a.coactivity || 0) * 0.08))
      .slice(0, 3000)
      .sort((a, b) => a.id - b.id);
    runtimeIndex.crossModalByKey = new Map(model.crossModalRelations.map((relation) => [`${Math.min(relation.a, relation.b)}:${Math.max(relation.a, relation.b)}`, relation]));
  }
  if (updates > 0) {
    model.stats.crossModalUpdates += updates;
    model.stats.crossModalRelations = model.crossModalRelations.length;
    trace.push({ step, kind: 'cross_modal', value: updates, label: 'modality_binding', meta: { modalities } });
  }
  return updates;
};

const noveltyScore = (model, tokens, runtimeIndex) => {
  if (tokens.length === 0) return 0;
  let novel = 0;
  for (const token of tokens) {
    if (!(runtimeIndex?.tokenToNeuron?.has(token) || model.vocabulary.some((neuron) => neuron.token === token))) novel += 1;
  }
  return novel / tokens.length;
};

const countNovelTokens = (model, tokens, runtimeIndex, filterFn = () => true) => {
  let novel = 0;
  let total = 0;
  for (const token of tokens) {
    if (!filterFn(token)) continue;
    total += 1;
    if (!runtimeIndex?.tokenToNeuron?.has(token)) novel += 1;
  }
  return { novel, total, ratio: total > 0 ? novel / total : 0 };
};

const discoveryAppraisal = (model, event, tokens, runtimeIndex) => {
  const pageKey = pageOrigin(event.url);
  const allNovelty = countNovelTokens(model, tokens, runtimeIndex);
  const semantic = countNovelTokens(model, tokens, runtimeIndex, isDiscoveryToken);
  const isNewPage = !model.pageStats[pageKey];
  const recentOnPage = (model.observations || []).filter((item) => pageOrigin(item.url) === pageKey);
  const isNewEventTypeHere = !recentOnPage.some((item) => item.eventType === event.eventType);
  const changeText = String(event.change?.addedText || '');
  const changeWords = changeText
    ? countNovelTokens(model, tokenize(changeText), runtimeIndex, isDiscoveryToken)
    : { novel: 0, total: 0, ratio: 0 };
  const wordDiscovery = Math.min(semantic.novel * 0.038, 0.45);
  const semanticNoveltyBonus = semantic.ratio * 0.48;
  const pageDiscovery = isNewPage ? 0.22 : 0;
  const eventDiscovery = isNewEventTypeHere && !isNewPage ? 0.1 : 0;
  const changeDiscovery = Math.min(changeWords.novel * 0.05, 0.38);
  const discoveryValence = clamp(
    wordDiscovery + semanticNoveltyBonus + pageDiscovery + eventDiscovery + changeDiscovery,
    0,
    0.85
  );
  return {
    novelty: allNovelty.ratio,
    semanticNovelty: semantic.ratio,
    novelWordCount: semantic.novel,
    changeNovelWords: changeWords.novel,
    newPage: isNewPage,
    newEventType: isNewEventTypeHere,
    discoveryValence,
  };
};

const eventSalience = (event, discovery) => {
  const novelty = discovery?.novelty ?? 0;
  const semanticNovelty = discovery?.semanticNovelty ?? novelty;
  const active = event.eventType === 'click' || event.eventType === 'input' || event.eventType === 'selection' ? 0.34 : 0;
  const autonomous = String(event.eventType || '').startsWith('autonomous_') ? 0.2 : 0;
  const change = String(event.eventType || '').endsWith('_change') ? 0.32 : 0;
  const autonomousButton = event.eventType === 'autonomous_button' ? 0.3 : 0;
  const autonomousInput = event.eventType === 'autonomous_input' ? 0.34 : 0;
  const media = isMediaEvent(event.eventType) ? 0.22 : 0;
  const visual = isVisualEvent(event.eventType) || event.visualTokens ? 0.26 : 0;
  const textRegion = event.textRegion ? 0.28 : 0;
  const cue = event.mediaCueText ? 0.18 : 0;
  const transcript = event.mediaTranscriptText ? 0.2 : 0;
  const movement = event.eventType === 'scroll' || event.eventType === 'media_seek' ? 0.16 : 0;
  return clamp(0.18 + semanticNovelty * 0.52 + novelty * 0.12 + active + autonomous + autonomousButton + autonomousInput + change + media + visual + textRegion + cue + transcript + movement, 0, 1);
};

const updateNeuromodulators = (model, reward, salience, novelty, event, instinctUrgency = 0) => {
  const n = model.neuromodulators;
  const error = reward - n.dopamine;
  n.dopamine = clamp(n.dopamine * 0.86 + error * 0.34, -1, 1);
  n.acetylcholine = clamp(n.acetylcholine * 0.82 + novelty * 0.55 + salience * 0.18, 0, 1);
  n.norepinephrine = clamp(
    n.norepinephrine * 0.86
    + salience * 0.42
    + instinctUrgency * 0.44
    + (event.eventType === 'media_seek' ? 0.18 : 0),
    0,
    1
  );
  n.serotonin = clamp(n.serotonin * 0.96 + Math.max(0, reward) * 0.08 - Math.max(0, -reward) * 0.06, 0, 1);
  n.fatigue = clamp(n.fatigue * 0.985 + Math.min(0.018, (model.stats.totalTokens || 0) / 800000), 0, 1);
};

const homeostaticRegulation = (model) => {
  const interval = model.config.homeostasisInterval || 16;
  if ((model.stats.steps || 0) % interval !== 0 || model.vocabulary.length === 0) return [];
  const avgPositive = (model.stats.positiveSpikes || 0) / Math.max(1, model.vocabulary.length);
  const target = Math.max(1.6, Math.log1p(model.stats.observations || 1) * 0.42);
  const trace = [];
  for (const neuron of model.vocabulary) {
    const localActivity = (neuron.positiveSpikeMass || 0) + (neuron.negativeSpikeMass || 0);
    const delta = localActivity > avgPositive + target ? 0.025 : -0.012;
    neuron.threshold = clamp((neuron.threshold || 1) + delta, 0.65, 2.4);
    neuron.v *= 0.94;
  }
  trace.push({ step: model.stats.steps, kind: 'homeostasis', value: avgPositive, label: 'threshold_balance' });
  return trace;
};

const consolidateMemory = (model) => {
  const interval = model.config.consolidationInterval || 48;
  if ((model.stats.steps || 0) % interval !== 0 || model.associations.length === 0) return [];
  const trace = [];
  const salientTokens = new Set(
    (model.observations || [])
      .filter((item) => (item.salience || 0) > 0.58 || (item.reward || 0) > 0.55)
      .flatMap((item) => semanticTokens([
        item.eventType || '',
        item.elementText || '',
        item.media?.cueText || '',
        item.media?.transcriptText || '',
        item.visual?.text || '',
        item.visibleText || '',
        ...(item.instinct?.activeCrises || []).map((kind) => `instinct:crisis:${kind}`),
      ].join(' '), 100))
  );

  for (const edge of model.associations) {
    const pre = model.vocabulary.find((item) => item.id === edge.pre);
    const post = model.vocabulary.find((item) => item.id === edge.post);
    const salient = salientTokens.has(pre?.token) || salientTokens.has(post?.token);
    const strong = Math.abs(edge.w || 0) > 0.36;
    if (strong || salient) {
      edge.stability = clamp((edge.stability || 0) + (salient ? 0.08 : 0.04), 0, 1);
      edge.replayCount = (edge.replayCount || 0) + 1;
      edge.w = clamp(edge.w + Math.sign(edge.w || 1) * edge.stability * 0.018, -1, 1);
      if (pre) {
        pre.stability = clamp((pre.stability || 0) + 0.025, 0, 1);
        pre.lastConsolidatedStep = model.stats.steps;
      }
      if (post) {
        post.stability = clamp((post.stability || 0) + 0.025, 0, 1);
        post.lastConsolidatedStep = model.stats.steps;
      }
      model.stats.stabilizedSynapses += 1;
    } else if ((edge.stability || 0) < 0.12) {
      edge.w *= 0.992;
    }
  }

  model.stats.consolidationCycles += 1;
  trace.push({ step: model.stats.steps, kind: 'consolidation', value: model.stats.stabilizedSynapses, label: 'sleep_replay' });
  return trace;
};

const delayedOutcomeSignal = (event, reward, salience, discovery) => {
  const novelty = discovery?.novelty ?? 0;
  const semanticNovelty = discovery?.semanticNovelty ?? novelty;
  const progress = clamp(Number(event.mediaProgress || 0), 0, 1);
  const completion = event.eventType === 'media_ended' || (event.eventType === 'media_pause' && progress > 0.78) ? 0.62 : 0;
  const activeChoice = event.eventType === 'click' ? 0.24
    : event.eventType === 'autonomous_button' ? 0.2
      : event.eventType === 'autonomous_input' ? 0.26
        : String(event.eventType || '').endsWith('_change') ? 0.34
    : event.eventType === 'selection' ? 0.46
      : event.eventType === 'input' ? 0.32
        : 0;
  const visualOutcome = isVisualEvent(event.eventType) && Number(event.visualVisibleRatio || 0) > 0.5 ? 0.12 : 0;
  const revisit = event.eventType === 'page_view' && semanticNovelty < 0.08 ? 0.12 : 0;
  const firstVisit = event.eventType === 'page_view' && discovery?.newPage ? 0.18 : 0;
  const newKnowledge = String(event.eventType || '').endsWith('_change') && (discovery?.changeNovelWords || 0) > 0
    ? Math.min(discovery.changeNovelWords * 0.04, 0.35)
    : 0;
  const frustration = event.eventType === 'media_pause' && progress < 0.08 ? -0.22 : 0;
  const noSignal = completion === 0 && activeChoice === 0 && visualOutcome === 0 && revisit === 0
    && firstVisit === 0 && newKnowledge === 0 && frustration === 0;
  if (noSignal) return 0;
  return clamp(
    completion + activeChoice + visualOutcome + revisit + firstVisit + newKnowledge
    + reward * 0.28 + salience * 0.12 - semanticNovelty * 0.04 + frustration,
    -1,
    1
  );
};

const applyDelayedReward = (model, outcome, step, options = {}) => {
  if (Math.abs(outcome) < 0.04 || model.eligibility.synapses.length === 0) return [];
  const trace = [];
  const window = options.window ?? model.config.delayedRewardWindow ?? 32;
  const decay = model.config.eligibilityDecay || 0.88;
  const lr = model.config.delayedLearningRate || 0.045;
  let assignments = 0;

  model.eligibility.synapses = model.eligibility.synapses.flatMap((item) => {
    const age = step - (item.lastStep || step);
    if (age > window) return [];
    const credit = (item.e || 0) * Math.pow(decay, Math.max(0, age));
    if (Math.abs(credit) < 0.002) return [];
    const association = model.associations.find((edge) => edge.pre === item.pre && edge.post === item.post);
    if (association) {
      const stabilityBrake = 1 - (association.stability || 0) * 0.42;
      association.w = clamp(association.w + lr * outcome * credit * stabilityBrake, -1, 1);
      association.delayedCredit = (association.delayedCredit || 0) + outcome * credit;
      association.lastDelayedRewardStep = step;
      if (outcome < 0) {
        association.d2NoGo = clamp((association.d2NoGo || 0) + Math.abs(outcome) * credit * 0.018, 0, 1);
      }
      assignments += 1;
    }
    return [{ ...item, e: credit * 0.82 }];
  }).slice(-240);

  if (assignments > 0) {
    if (options.kind === 'instinct_delayed') {
      model.stats.instinctDelayedAvoidance += 1;
    } else {
      model.stats.delayedRewards += 1;
    }
    model.stats.delayedCreditAssignments += assignments;
    trace.push({
      step,
      kind: options.kind ?? 'delayed_reward',
      value: outcome,
      label: options.label ?? 'eligibility_credit',
      meta: { assignments },
    });
  }
  return trace;
};

const rememberEligibility = (model, association, step, signedSpike, salience, eventType) => {
  const existing = model.eligibility.synapses.find((item) => item.pre === association.pre && item.post === association.post);
  const spikeCredit = Math.max(0.35, Math.abs(signedSpike || 0));
  const e = clamp((existing?.e || 0) * 0.72 + spikeCredit * (0.35 + salience), 0, 4);
  if (existing) {
    Object.assign(existing, { e, lastStep: step, eventType });
  } else {
    model.eligibility.synapses.push({ pre: association.pre, post: association.post, e, lastStep: step, eventType });
  }
  model.eligibility.synapses = model.eligibility.synapses
    .filter((item) => step - (item.lastStep || step) <= (model.config.delayedRewardWindow || 32))
    .slice(-240);
};

const synapseImportance = (edge) =>
  Math.abs(edge.w || 0)
  + (edge.stability || 0) * 0.8
  + Math.max(0, edge.d1Go || 0) * 0.45
  + Math.abs(edge.delayedCredit || 0) * 0.16
  - Math.max(0, edge.d2NoGo || 0) * 0.24;

const applySynapticScaling = (model) => {
  const maxOutgoing = model.config.maxOutgoingWeight || 4.5;
  const byPre = new Map();
  for (const edge of model.associations || []) {
    if (!byPre.has(edge.pre)) byPre.set(edge.pre, []);
    byPre.get(edge.pre).push(edge);
  }
  let scaled = 0;
  for (const edges of byPre.values()) {
    const total = edges.reduce((sum, edge) => sum + Math.abs(edge.w || 0), 0);
    if (total <= maxOutgoing) continue;
    const factor = maxOutgoing / Math.max(0.001, total);
    for (const edge of edges) {
      edge.w = clamp((edge.w || 0) * factor, -1, 1);
      scaled += 1;
    }
  }
  return scaled;
};

const applyLateralInhibition = (model, replayed) => {
  const replayedKeys = new Set(replayed.map((edge) => `${edge.pre}:${edge.post}`));
  const activePres = new Set(replayed.map((edge) => edge.pre));
  let inhibited = 0;
  for (const edge of model.associations || []) {
    if (!activePres.has(edge.pre) || replayedKeys.has(`${edge.pre}:${edge.post}`)) continue;
    const weakBrake = 1 - (edge.stability || 0) * 0.4;
    edge.w = clamp((edge.w || 0) * (1 - 0.018 * weakBrake), -1, 1);
    edge.d2NoGo = clamp((edge.d2NoGo || 0) + 0.004 * weakBrake, 0, 1);
    inhibited += 1;
  }
  return inhibited;
};

export const sleepConsolidateModel = (model, options = {}) => {
  ensureBrainFields(model);
  const cycles = Math.max(1, Math.min(12, Math.floor(options.cycles ?? model.config.sleepReplayCycles ?? 3)));
  const trace = [];
  let replayedCount = 0;
  let scaledCount = 0;
  let inhibitedCount = 0;
  const step = model.stats.steps || 0;
  const neuronsById = new Map((model.vocabulary || []).map((neuron) => [neuron.id, neuron]));
  const replayLimit = Math.min(220, Math.max(16, Math.floor((model.associations || []).length * 0.18)));

  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const replayed = [...(model.associations || [])]
      .sort((a, b) => synapseImportance(b) - synapseImportance(a))
      .slice(0, replayLimit);
    for (const edge of replayed) {
      const go = edge.d1Go || 0;
      const noGo = edge.d2NoGo || 0;
      const replayGain = clamp(0.006 + go * 0.022 - noGo * 0.018 + Math.max(0, edge.delayedCredit || 0) * 0.004, -0.035, 0.04);
      edge.w = clamp((edge.w || 0) + Math.sign(edge.w || 1) * replayGain, -1, 1);
      edge.stability = clamp((edge.stability || 0) + Math.max(0.004, go * 0.015) - noGo * 0.006, 0, 1);
      edge.replayCount = (edge.replayCount || 0) + 1;
      edge.lastConsolidatedStep = step;
      edge.delayedCredit = (edge.delayedCredit || 0) * 0.86;
      for (const id of [edge.pre, edge.post]) {
        const neuron = neuronsById.get(id);
        if (!neuron) continue;
        neuron.stability = clamp((neuron.stability || 0) + 0.006 + edge.stability * 0.004, 0, 1);
        neuron.lastConsolidatedStep = step;
      }
      replayedCount += 1;
    }
    inhibitedCount += applyLateralInhibition(model, replayed);
    scaledCount += applySynapticScaling(model);
  }

  const targetRate = model.config.targetFiringRate || 0.04;
  const steps = Math.max(1, model.stats.steps || 1);
  for (const neuron of model.vocabulary || []) {
    const firingRate = ((neuron.positiveSpikeMass || 0) + (neuron.negativeSpikeMass || 0)) / steps;
    const thresholdDelta = firingRate > targetRate ? 0.035 : -0.015;
    neuron.threshold = clamp((neuron.threshold || 1) + thresholdDelta, 0.65, 2.6);
    neuron.v *= 0.72;
  }

  model.neuromodulators.fatigue = clamp((model.neuromodulators.fatigue || 0) * 0.42, 0, 1);
  model.neuromodulators.acetylcholine = clamp((model.neuromodulators.acetylcholine || 0) * 0.72, 0, 1);
  model.neuromodulators.serotonin = clamp((model.neuromodulators.serotonin || 0) + 0.08, 0, 1);
  model.stats.sleepCycles += cycles;
  model.stats.consolidationCycles += cycles;
  model.stats.sleepReplayedSynapses += replayedCount;
  model.stats.sleepScaledSynapses += scaledCount;
  model.stats.sleepInhibitedSynapses += inhibitedCount;
  model.stats.stabilizedSynapses += replayedCount;
  model.stats.modelBytes = estimateModelBytes(model);
  model.savedAt = Date.now();
  trace.push({
    step,
    kind: 'sleep_consolidation',
    value: replayedCount,
    label: 'manual_sleep_replay',
    meta: { cycles, scaled: scaledCount, inhibited: inhibitedCount },
  });
  return {
    model,
    trace,
    summary: { cycles, replayed: replayedCount, scaled: scaledCount, inhibited: inhibitedCount },
  };
};

const safeClone = (value) => {
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
};

const cleanImportToken = (token) => normalizeToken(String(token ?? '')).slice(0, 220);

const pickImportSnapshot = (payload) => {
  if (payload?.kind === 'dst-snn-chat-model' && payload.model) return payload.model;
  if (payload?.model?.vocabulary) return payload.model;
  if (payload?.snapshot?.vocabulary) return payload.snapshot;
  if (payload?.vocabulary) return payload;
  if (Array.isArray(payload?.models)) {
    const entry = payload.models.find((item) =>
      item?.snapshot && (
        item.modelKind === 'browser-language'
        || item.modelKind === 'dst-web'
        || item.snapshot.domain === 'browser-language'
        || item.snapshot.domain === 'dst-web'
      )
    );
    return entry?.snapshot || null;
  }
  return null;
};

const addImportedAssociation = (model, associationMap, pre, post, edge = {}) => {
  if (!Number.isFinite(pre) || !Number.isFinite(post) || pre === post) return null;
  const key = `${pre}:${post}`;
  let association = associationMap.get(key);
  const incomingWeight = Number(edge.w ?? edge.weight ?? 0.06);
  if (association) {
    association.w = clamp((association.w || 0) + incomingWeight * 0.72, -1, 1);
    association.stability = clamp(Math.max(association.stability || 0, Number(edge.stability || 0)), 0, 1);
    association.replayCount = Math.max(association.replayCount || 0, Number(edge.replayCount || 0));
    association.d1Go = clamp(Math.max(association.d1Go || 0, Number(edge.d1Go ?? Math.max(0, incomingWeight))), 0, 1);
    association.d2NoGo = clamp(Math.max(association.d2NoGo || 0, Number(edge.d2NoGo ?? Math.max(0, -incomingWeight))), 0, 1);
    return association;
  }
  association = {
    id: model.associations.length,
    pre,
    post,
    w: clamp(incomingWeight, -1, 1),
    aPre: Number(edge.aPre || 0),
    aPost: Number(edge.aPost || 0),
    stability: clamp(Number(edge.stability || 0), 0, 1),
    replayCount: Number(edge.replayCount || 0),
    lastUpdatedStep: Number(edge.lastUpdatedStep || 0),
    d1Go: clamp(Number(edge.d1Go ?? Math.max(0.02, incomingWeight)), 0, 1),
    d2NoGo: clamp(Number(edge.d2NoGo ?? Math.max(0.01, -incomingWeight)), 0, 1),
    rewardPrediction: Number(edge.rewardPrediction || 0),
    semanticGate: Number(edge.semanticGate || 1),
    relationKind: edge.relationKind,
  };
  model.associations.push(association);
  associationMap.set(key, association);
  return association;
};

const addImportedCrossModalRelation = (model, relationMap, a, b, relation = {}) => {
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return null;
  const left = Math.min(a, b);
  const right = Math.max(a, b);
  const key = `${left}:${right}`;
  let existing = relationMap.get(key);
  const incomingWeight = Number(relation.w ?? relation.weight ?? 0.04);
  if (existing) {
    existing.w = clamp((existing.w || 0) + incomingWeight * 0.72, -1, 1);
    existing.coactivity = Math.max(existing.coactivity || 0, Number(relation.coactivity || 0));
    existing.stability = clamp(Math.max(existing.stability || 0, Number(relation.stability || 0)), 0, 1);
    existing.modalities = [...new Set([...(existing.modalities || []), ...(relation.modalities || [])])].slice(0, 6);
    return existing;
  }
  existing = {
    id: model.crossModalRelations.length,
    a: left,
    b: right,
    modalities: Array.isArray(relation.modalities) ? relation.modalities.slice(0, 6) : ['unknown', 'unknown'],
    w: clamp(incomingWeight, -1, 1),
    coactivity: Number(relation.coactivity || 0),
    stability: clamp(Number(relation.stability || 0), 0, 1),
    lastUpdatedStep: Number(relation.lastUpdatedStep || 0),
    rewardPrediction: Number(relation.rewardPrediction || 0),
  };
  model.crossModalRelations.push(existing);
  relationMap.set(key, existing);
  return existing;
};

export const normalizeImportedModel = (sourceModel, options = {}) => {
  const source = safeClone(sourceModel || {});
  const model = createModel();
  model.domain = 'browser-language';
  model.source = source.source ? `import:${source.source}` : 'imported';
  model.importedAt = Date.now();
  model.importedFrom = options.sourceName || source.sourceCheckpoint || '';
  model.config = { ...model.config, ...(source.config || {}) };
  model.moralState = { ...model.moralState, ...(source.moralState || {}) };
  model.instinctState = { ...model.instinctState, ...(source.instinctState || {}) };
  model.neuromodulators = { ...model.neuromodulators, ...(source.neuromodulators || {}) };
  model.stats = { ...model.stats, ...(source.stats || {}) };
  model.stats.importedModels = (model.stats.importedModels || 0) + 1;
  model.stats.steps = Number(source.stats?.steps || 0);
  model.stats.observations = Number(source.stats?.observations || source.observations?.length || 0);

  const oldIdToNewId = new Map();
  const tokenToNeuron = new Map();
  const rawTokenToId = new Map();
  const importedVocabulary = normalizeCollection(source.vocabulary);
  for (const neuron of importedVocabulary) {
    const token = cleanImportToken(neuron.token);
    if (!token) continue;
    let target = tokenToNeuron.get(token);
    if (!target) {
      target = {
        ...neuron,
        id: model.vocabulary.length,
        token,
        count: Number(neuron.count || 0),
        v: Number(neuron.v || 0),
        threshold: Number(neuron.threshold || 1),
        lastSeenStep: Number(neuron.lastSeenStep || 0),
        positiveSpikeMass: Number(neuron.positiveSpikeMass || 0),
        negativeSpikeMass: Number(neuron.negativeSpikeMass || 0),
        stability: clamp(Number(neuron.stability || 0), 0, 1),
        importance: Number(neuron.importance ?? tokenWeight(token)),
        role: neuron.role || (isStructuralToken(token) ? 'structural' : 'semantic'),
        modalities: Array.isArray(neuron.modalities) ? neuron.modalities.slice(0, 6) : undefined,
        primaryModality: neuron.primaryModality,
      };
      model.vocabulary.push(target);
      tokenToNeuron.set(token, target);
    } else {
      target.count += Number(neuron.count || 0);
      target.positiveSpikeMass += Number(neuron.positiveSpikeMass || 0);
      target.negativeSpikeMass += Number(neuron.negativeSpikeMass || 0);
      target.stability = clamp(Math.max(target.stability || 0, Number(neuron.stability || 0)), 0, 1);
      target.modalities = [...new Set([...(target.modalities || []), ...(neuron.modalities || [])])].slice(0, 6);
    }
    if (neuron.id !== undefined) oldIdToNewId.set(Number(neuron.id), target.id);
    rawTokenToId.set(token, target.id);
    for (const raw of neuron.rawTokens || []) rawTokenToId.set(String(raw).toLowerCase(), target.id);
  }

  const associationMap = new Map();
  for (const edge of normalizeCollection(source.associations)) {
    const pre = oldIdToNewId.get(Number(edge.pre));
    const post = oldIdToNewId.get(Number(edge.post));
    addImportedAssociation(model, associationMap, pre, post, edge);
  }

  const relationMap = new Map();
  for (const relation of normalizeCollection(source.crossModalRelations)) {
    let a = oldIdToNewId.get(Number(relation.a));
    let b = oldIdToNewId.get(Number(relation.b));
    if ((!Number.isFinite(a) || !Number.isFinite(b)) && Array.isArray(relation.tokens)) {
      a = rawTokenToId.get(String(relation.tokens[0] || '').toLowerCase());
      b = rawTokenToId.get(String(relation.tokens[1] || '').toLowerCase());
    }
    addImportedCrossModalRelation(model, relationMap, a, b, relation);
  }

  model.observations = normalizeCollection(source.observations).slice(0, model.config.maxObservationMemory || 180);
  ensureBrainFields(model);
  model.stats.totalTokens = Math.max(model.stats.totalTokens || 0, model.vocabulary.length);
  model.stats.associations = model.associations.length;
  model.stats.crossModalRelations = model.crossModalRelations.length;
  model.stats.modelBytes = estimateModelBytes(model);
  model.savedAt = Date.now();
  return model;
};

export const mergeImportedModel = (baseModel, importedModel, options = {}) => {
  const model = safeClone(baseModel || createModel());
  ensureBrainFields(model);
  const incoming = normalizeImportedModel(importedModel, options);
  const runtimeIndex = createRuntimeIndex(model);
  const idMap = new Map();

  for (const neuron of incoming.vocabulary || []) {
    const token = cleanImportToken(neuron.token);
    if (!token) continue;
    let target = runtimeIndex.tokenToNeuron.get(token);
    if (!target) {
      target = {
        ...neuron,
        id: model.vocabulary.length,
        token,
      };
      model.vocabulary.push(target);
      runtimeIndex.tokenToNeuron.set(token, target);
    } else {
      target.count = Number(target.count || 0) + Number(neuron.count || 0);
      target.positiveSpikeMass = Number(target.positiveSpikeMass || 0) + Number(neuron.positiveSpikeMass || 0);
      target.negativeSpikeMass = Number(target.negativeSpikeMass || 0) + Number(neuron.negativeSpikeMass || 0);
      target.stability = clamp(Math.max(target.stability || 0, Number(neuron.stability || 0)), 0, 1);
      target.importance = Math.max(Number(target.importance || 0), Number(neuron.importance || tokenWeight(token)));
      target.modalities = [...new Set([...(target.modalities || []), ...(neuron.modalities || [])])].slice(0, 6);
      target.primaryModality ??= neuron.primaryModality;
    }
    idMap.set(neuron.id, target.id);
  }

  const associationMap = new Map((model.associations || []).map((edge) => [`${edge.pre}:${edge.post}`, edge]));
  for (const edge of incoming.associations || []) {
    addImportedAssociation(model, associationMap, idMap.get(edge.pre), idMap.get(edge.post), edge);
  }

  const relationMap = new Map((model.crossModalRelations || []).map((relation) => [`${Math.min(relation.a, relation.b)}:${Math.max(relation.a, relation.b)}`, relation]));
  for (const relation of incoming.crossModalRelations || []) {
    addImportedCrossModalRelation(model, relationMap, idMap.get(relation.a), idMap.get(relation.b), relation);
  }

  model.observations = [
    ...(incoming.observations || []).map((item) => ({ ...item, imported: true })),
    ...(model.observations || []),
  ].slice(0, model.config.maxObservationMemory || 180);
  model.stats.importedModels = (model.stats.importedModels || 0) + 1;
  model.stats.importedNeurons = (model.stats.importedNeurons || 0) + incoming.vocabulary.length;
  model.stats.importedAssociations = (model.stats.importedAssociations || 0) + incoming.associations.length;
  model.stats.crossModalRelations = model.crossModalRelations.length;
  model.stats.modelBytes = estimateModelBytes(model);
  model.savedAt = Date.now();
  ensureBrainFields(model);
  return model;
};

export const decodeModelPayloadBytes = (buffer) => {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  const hasMagic = bytes.length >= magic.length + 4 && magic.every((value, index) => bytes[index] === value);
  if (!hasMagic) {
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(magic.length, true);
  const start = magic.length + 4;
  const end = start + length;
  if (end > bytes.length) throw new Error('Truncated .edensnn payload.');
  return JSON.parse(new TextDecoder().decode(bytes.slice(start, end)));
};

export const importedModelFromPayload = (payload, options = {}) => {
  const snapshot = pickImportSnapshot(payload);
  if (!snapshot) throw new Error('No compatible SNN model found in imported file.');
  return normalizeImportedModel(snapshot, options);
};

export const observeEvent = (model, event, settings = DEFAULT_LEARNING_SETTINGS, runtimeHints = {}) => {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  ensureBrainFields(model);
  const learningSettings = withLearningSettings(settings);
  const runtimeIndex = createRuntimeIndex(model);
  model.config.maxVocabulary = Math.max(0, Math.floor(Number(learningSettings.maxVocabulary ?? model.config.maxVocabulary ?? 0)));
  const mediaTokens = mediaStateTokens(event);
  const visualTokens = visualStateTokens(event);
  const hasVisualBinding = visualTokens.length > 0 || isVisualEvent(event.eventType) || Boolean(event.visualKind);
  const visualBindingRaw = hasVisualBinding
    ? String(event.visualText || event.elementText || '').trim()
    : '';
  const pageTextParts = [String(event.eventType || '')];
  const elementText = String(event.elementText ?? '').trim();
  const looseVisualText = String(event.visualText ?? '').trim();
  if (!hasVisualBinding) {
    if (elementText) pageTextParts.push(elementText);
    if (looseVisualText) pageTextParts.push(looseVisualText);
  } else if (elementText && elementText !== visualBindingRaw) {
    pageTextParts.push(elementText);
  }
  if (event.mediaCueText) pageTextParts.push(event.mediaCueText);
  if (event.mediaTranscriptText) pageTextParts.push(event.mediaTranscriptText);
  if (event.visibleText) pageTextParts.push(event.visibleText);
  const rawLearningText = pageTextParts.join(' ');
  const morality = moralAppraisal(event, [
    rawLearningText,
    visualBindingRaw,
  ].filter(Boolean).join(' '), learningSettings);
  const instinctSourceText = [
    rawLearningText,
    visualBindingRaw,
    event.url || '',
    event.title || '',
  ].filter(Boolean).join(' ');
  const instinct = instinctAppraisal(event, instinctSourceText, learningSettings, model);
  const safeLearningText = learningSettings.sensitiveTextMode === 'full' ? rawLearningText
    : learningSettings.sensitiveTextMode === 'abstract' ? morality.redactedText.replaceAll('[redacted-sensitive]', 'sensitive-value')
      : morality.redactedText;
  const safeVisualText = !visualBindingRaw ? ''
    : learningSettings.sensitiveTextMode === 'full' ? visualBindingRaw
      : learningSettings.sensitiveTextMode === 'abstract'
        ? redactSensitiveText(visualBindingRaw).text.replaceAll('[redacted-sensitive]', 'sensitive-value')
        : redactSensitiveText(visualBindingRaw).text;
  const visualTextTokens = hasVisualBinding
    ? injectLexicalBridges(semanticTokens(safeVisualText, 80), model, 120)
    : [];
  const visualTextAnchor = event.textRegion ? 'source:text-region' : 'source:visual-text';
  const cueTokens = injectLexicalBridges(semanticTokens(event.mediaCueText || '', 80), model, 100);
  const transcriptTokens = injectLexicalBridges(semanticTokens(event.mediaTranscriptText || '', 120), model, 140);
  const pageTokens = injectLexicalBridges(tokenize(safeLearningText), model, 220);
  const eventSemanticWords = [
    ...visualTextTokens,
    ...cueTokens,
    ...transcriptTokens,
    ...pageTokens,
  ].filter((token) => isDiscoveryToken(token) && !String(token).startsWith('lex:'));
  const tokens = [
    ...contextTokens(event),
    ...mediaTokens,
    ...visualTokens,
    ...(visualTextTokens.length ? [visualTextAnchor, ...visualTextTokens] : []),
    ...(learningSettings.instinctLearningMode === 'off' ? [] : instinctTokens(instinct)),
    ...(cueTokens.length ? ['source:caption', ...cueTokens] : []),
    ...(transcriptTokens.length ? ['source:transcript', ...transcriptTokens] : []),
    ...(learningSettings.moralLearningMode === 'off' ? [] : moralTokens(morality)),
    ...pageTokens
  ].slice(0, 480);
  const vocabularyBefore = model.vocabulary.length;
  const pageKey = pageOrigin(event.url);
  const discovery = discoveryAppraisal(model, event, tokens, runtimeIndex);
  const novelty = discovery.novelty;
  const semanticNovelty = discovery.semanticNovelty;
  const salience = eventSalience(event, discovery);
  const activityReward = event.eventType === 'click' ? 0.36
    : event.eventType === 'scroll' ? 0.18
      : event.eventType === 'input' ? 0.28
        : event.eventType === 'page_view' ? 0.22
          : event.eventType === 'autonomous_link' ? 0.3
            : event.eventType === 'autonomous_button' ? 0.28
              : event.eventType === 'autonomous_input' ? 0.3
                : event.eventType === 'autonomous_visual' ? 0.26
                  : String(event.eventType || '').startsWith('text_visual') ? 0.28
                  : isVisualEvent(event.eventType) ? 0.22
                : String(event.eventType || '').endsWith('_change') ? 0.3
                  : event.eventType === 'autonomous_scroll' ? 0.2
                    : event.eventType === 'autonomous_scan' ? 0.16
          : isMediaEvent(event.eventType) ? 0.24
            : 0.12;
  const continuityReward = event.eventType === 'media_sample' && !event.mediaPaused ? 0.12 : 0;
  const moralAdjustment = morality.moralValence >= 0
    ? morality.moralValence * learningSettings.prosocialWeight
    : morality.moralValence * learningSettings.moralPenaltyWeight;
  const instinctAdjustment = instinct.threatValence * learningSettings.instinctPenaltyWeight;
  const effectiveMoralAdjustment = (learningSettings.moralLearningMode === 'shape' || learningSettings.moralLearningMode === 'constraint')
    ? moralAdjustment : 0;
  const effectiveInstinctAdjustment = (learningSettings.instinctLearningMode === 'shape' || learningSettings.instinctLearningMode === 'constraint')
    ? instinctAdjustment : 0;
  const baseActivityReward = activityReward + continuityReward
    + novelty * 0.28
    + semanticNovelty * 0.52
    + discovery.discoveryValence
    + salience * 0.12
    - (tokens.length === 0 ? 0.6 : 0);
  const unconstrainedReward = clamp(baseActivityReward + effectiveMoralAdjustment + effectiveInstinctAdjustment, -1, 1);
  const moralRewardCap = learningSettings.moralLearningMode === 'constraint' && morality.moralRisk > 0.55 ? 0.12 - morality.moralRisk * 0.72 : 1;
  const instinctRewardCap = learningSettings.instinctLearningMode === 'constraint' && instinct.threatLevel > 0.45
    ? -0.08 - instinct.threatLevel * 0.55
    : 1;
  const rewardCap = Math.min(moralRewardCap, instinctRewardCap);
  const usesEthicalShaping = learningSettings.moralLearningMode === 'shape'
    || learningSettings.moralLearningMode === 'constraint'
    || learningSettings.instinctLearningMode === 'shape'
    || learningSettings.instinctLearningMode === 'constraint';
  const reward = usesEthicalShaping
    ? clamp(Math.min(unconstrainedReward, rewardCap), -1, 1)
    : clamp(baseActivityReward, -1, 1);
  updateNeuromodulators(model, reward, salience, semanticNovelty, event, instinct.avoidanceUrgency);
  const step = model.stats.steps + 1;
  recordInstinctAction(model, event, step);
  updateInstinctState(model, instinct, learningSettings);
  const instinctDelayed = (learningSettings.instinctLearningMode === 'shape' || learningSettings.instinctLearningMode === 'constraint')
    ? crisisDelayedSignal(model, event, learningSettings)
    : 0;
  const delayedOutcome = clamp(
    delayedOutcomeSignal(event, reward, salience, discovery)
    + ((learningSettings.moralLearningMode === 'shape' || learningSettings.moralLearningMode === 'constraint') ? morality.moralValence * 0.24 : 0)
    + ((learningSettings.instinctLearningMode === 'shape' || learningSettings.instinctLearningMode === 'constraint') ? instinct.threatValence * 0.18 : 0),
    -1,
    1
  );
  const pageStats = model.pageStats[pageKey] ?? { observations: 0, clicks: 0, scrolls: 0, mediaEvents: 0, tokens: 0 };

  model.stats.steps = step;
  model.stats.observations += 1;
  model.stats.userEvents += event.eventType === 'page_view' ? 0 : 1;
  model.stats.autonomousEvents += String(event.eventType || '').startsWith('autonomous_') ? 1 : 0;
  model.stats.mediaEvents += isMediaEvent(event.eventType) ? 1 : 0;
  model.stats.visualEvents += isVisualEvent(event.eventType) || visualTokens.length > 0 ? 1 : 0;
  model.stats.moralEvents += learningSettings.moralLearningMode !== 'off' && (morality.moralRisk > 0 || morality.prosocialSignal > 0) ? 1 : 0;
  model.stats.moralPenalties += learningSettings.moralLearningMode !== 'off' && morality.moralValence < 0 ? 1 : 0;
  model.stats.instinctEvents += learningSettings.instinctLearningMode !== 'off' && instinct.threatLevel > 0 ? 1 : 0;
  model.stats.instinctAvoidances += learningSettings.instinctLearningMode !== 'off' && instinct.threatValence < 0 ? 1 : 0;
  model.stats.discoveryRewards += discovery.discoveryValence > 0.05 ? 1 : 0;
  model.stats.novelWordsRewarded += discovery.novelWordCount;
  model.stats.newPageDiscoveries += discovery.newPage ? 1 : 0;
  model.stats.privacyRedactions += learningSettings.sensitiveTextMode === 'full' ? 0 : morality.redactionCount;
  model.stats.totalTokens += tokens.length;
  model.stats.denseMacOps += Math.max(1, model.associations.length);
  model.savedAt = Date.now();
  if (!model.pageStats[pageKey]) model.stats.pages += 1;
  pageStats.observations += 1;
  pageStats.clicks += event.eventType === 'click' ? 1 : 0;
  pageStats.scrolls += event.eventType === 'scroll' ? 1 : 0;
  pageStats.mediaEvents += isMediaEvent(event.eventType) ? 1 : 0;
  pageStats.visualEvents = (pageStats.visualEvents || 0) + (isVisualEvent(event.eventType) || visualTokens.length > 0 ? 1 : 0);
  pageStats.tokens += tokens.length;
  model.pageStats[pageKey] = pageStats;

  const modulation = model.neuromodulators;
  const plasticityGate = clamp(
    0.45
      + modulation.dopamine * 0.38
      + modulation.acetylcholine * 0.32
      + modulation.norepinephrine * 0.22
      - modulation.fatigue * 0.28,
    0.08,
    1.45
  );
  const trace = [
    { step, kind: 'event', value: tokens.length, label: event.eventType, meta: { url: event.url ?? '', salience } },
    { step, kind: 'neuromodulator', value: modulation.dopamine, label: 'dopamine' },
    { step, kind: 'neuromodulator', value: modulation.acetylcholine, label: 'acetylcholine' },
    { step, kind: 'neuromodulator', value: modulation.norepinephrine, label: 'norepinephrine' },
    ...(learningSettings.moralLearningMode === 'off' ? [] : [{ step, kind: 'moral_appraisal', value: morality.moralValence, label: morality.moralRisk > 0 ? 'moral_risk' : 'moral_context' }]),
    ...(learningSettings.instinctLearningMode === 'off' ? [] : [{
      step,
      kind: 'instinct_appraisal',
      value: instinct.threatValence,
      label: instinctDelayed < -0.04 ? 'instinct_delayed_crisis' : instinct.threatLevel > 0.3 ? 'instinct_threat' : 'instinct_scan',
    }]),
    ...(discovery.discoveryValence > 0.04 ? [{
      step,
      kind: 'discovery',
      value: discovery.discoveryValence,
      label: discovery.novelWordCount > 0 ? 'novel_words' : discovery.newPage ? 'new_page' : discovery.newEventType ? 'new_experience' : 'semantic_novelty',
      meta: { novelWords: discovery.novelWordCount, semanticNovelty },
    }] : []),
    ...applyDelayedReward(model, delayedOutcome, step),
    ...(instinctDelayed < -0.04 ? applyDelayedReward(model, instinctDelayed, step, {
      window: model.config.instinctDelayedWindow || 48,
      label: 'instinct_crisis_delayed',
      kind: 'instinct_delayed',
    }) : []),
  ];
  let previous = null;
  const gpuVoltages = Array.isArray(runtimeHints.gpuVoltages) ? runtimeHints.gpuVoltages : [];
  let activeTokenIndex = 0;
  let analogDecisionMass = 0;
  const activeNeurons = [];
  for (const token of tokens) {
    const neuron = findOrCreateNeuron(model, token, runtimeIndex);
    const currentTokenWeight = tokenWeight(token);
    const modalities = tokenModalities(token, event);
    neuron.importance = currentTokenWeight;
    neuron.role = isStructuralToken(token) ? 'structural' : 'semantic';
    if (modalities.length > 0) {
      neuron.modalities = [...new Set([...(neuron.modalities || []), ...modalities])].slice(0, 4);
      neuron.primaryModality ??= modalities[0];
    }
    neuron.count += currentTokenWeight;
    neuron.lastSeenStep = step;
    const gpuVoltage = gpuVoltages[activeTokenIndex];
    activeTokenIndex += 1;
    neuron.v = Number.isFinite(gpuVoltage)
      ? gpuVoltage * currentTokenWeight
      : neuron.v * 0.72 + (0.48 + salience * 0.24 + Math.abs(reward) * 0.18) * currentTokenWeight;
    const threshold = Math.max(0.55, neuron.threshold || 1);
    const analogMembrane = neuron.v / threshold;
    const signedSpike = clamp(Math.round(analogMembrane * (reward < 0 ? -1 : 1)), -model.config.spikeRangeD, model.config.spikeRangeD);
    const analogDecision = clamp(analogMembrane * currentTokenWeight * (reward < 0 ? -1 : 1), -model.config.spikeRangeD, model.config.spikeRangeD);
    analogDecisionMass += Math.abs(analogDecision);
    if (modalities.length > 0) {
      activeNeurons.push({
        neuron,
        token,
        modalities,
        drive: Math.abs(analogDecision) + Math.abs(signedSpike) + currentTokenWeight * 0.25,
      });
    }
    if (signedSpike !== 0) {
      neuron.v = 0;
      if (signedSpike > 0) {
        neuron.positiveSpikeMass += signedSpike;
        model.stats.positiveSpikes += signedSpike;
      } else {
        neuron.negativeSpikeMass += Math.abs(signedSpike);
        model.stats.negativeSpikes += Math.abs(signedSpike);
      }
      trace.push({
        step,
        kind: 'token_spike',
        value: signedSpike,
        label: token,
        meta: {
          neuron: neuron.id,
          token,
          routeKey: `neuron:${token}`,
          moduleTrigger: /^(module|skill|action|tool|media|visual|moral|instinct|source|lex):/.test(token) ? token : '',
          importance: currentTokenWeight,
        },
      });
    }
    if (previous) {
      const association = findOrCreateAssociation(model, previous.id, neuron.id, runtimeIndex);
      const previousWeight = tokenWeight(previous.token);
      const associationTokenGate = clamp(Math.sqrt(previousWeight * currentTokenWeight), 0.08, 1);
      association.aPre = association.aPre * 0.82 + 0.08;
      association.aPost = association.aPost * 0.82 + 0.08;
      const stabilityBrake = 1 - (association.stability || 0) * 0.55;
      const predictedReward = association.rewardPrediction || 0;
      const rpe = clamp(reward - predictedReward, -1, 1);
      association.rewardPrediction = clamp(predictedReward * 0.94 + reward * 0.06, -1, 1);
      if (rpe >= 0) {
        association.d1Go = clamp((association.d1Go || 0) + rpe * 0.08 + Math.max(0, signedSpike) * 0.006, 0, 1);
        association.d2NoGo = clamp((association.d2NoGo || 0) * (1 - rpe * 0.035), 0, 1);
      } else {
        association.d1Go = clamp((association.d1Go || 0) * (1 + rpe * 0.035), 0, 1);
        association.d2NoGo = clamp((association.d2NoGo || 0) + Math.abs(rpe) * 0.08 + Math.max(0, -signedSpike) * 0.006, 0, 1);
      }
      const goNoGoGate = clamp(0.28 + (association.d1Go || 0) * 0.86 - (association.d2NoGo || 0) * 0.58, 0.04, 1.35);
      const weightedAnalogDrive = clamp(
        analogDecision * (0.45 + Math.abs(association.w || 0)) + signedSpike * 0.38,
        -model.config.spikeRangeD,
        model.config.spikeRangeD
      );
      const moralBrake = learningSettings.moralLearningMode === 'constraint'
        ? clamp(1 - morality.privacyRisk * model.moralState.privacyRespect * 0.72 - morality.harmRisk * model.moralState.harmAversion * 0.44, 0.08, 1)
        : 1;
      const instinctBrake = learningSettings.instinctLearningMode === 'constraint'
        ? clamp(1 - instinct.threatLevel * (model.instinctState.threatAversion ?? 0.62) * 0.78, 0.06, 1)
        : 1;
      const analogGate = clamp(0.42 + Math.abs(weightedAnalogDrive) * 0.22, 0.18, 1.3);
      const learningSignal = clamp(reward * 0.26 + rpe * 0.56 + weightedAnalogDrive * 0.18, -1, 1);
      const hebbian = model.config.learningRate * plasticityGate * stabilityBrake * moralBrake * instinctBrake * goNoGoGate * associationTokenGate * analogGate * (0.3 + association.aPre + association.aPost) * learningSignal;
      association.w = clamp(association.w + hebbian, -1, 1);
      if (learningSettings.instinctLearningMode !== 'off' && instinct.avoidanceUrgency > 0.35 && learningSignal < 0) {
        association.d2NoGo = clamp((association.d2NoGo || 0) + instinct.avoidanceUrgency * 0.014, 0, 1);
      }
      association.lastUpdatedStep = step;
      association.semanticGate = associationTokenGate;
      association.analogDrive = weightedAnalogDrive;
      model.stats.rewardPredictionError = model.stats.rewardPredictionError * 0.94 + Math.abs(rpe) * 0.06;
      rememberEligibility(model, association, step, signedSpike || weightedAnalogDrive, salience, event.eventType);
      model.stats.sparseAcOps += Math.max(1, Math.ceil(Math.abs(signedSpike || weightedAnalogDrive)));
    }
    previous = neuron;
  }
  const lexicalLinks = applyLexicalRelativeLinks(
    model,
    eventSemanticWords,
    step,
    reward,
    plasticityGate,
    runtimeIndex,
    trace
  );
  model.stats.lexicalLinks += lexicalLinks;
  updateCrossModalRelations(model, activeNeurons, step, reward, salience, plasticityGate, runtimeIndex, trace);
  model.stats.analogDecisions += 1;
  model.stats.analogDrive = model.stats.analogDrive * 0.92 + analogDecisionMass * 0.08;
  model.stats.synapticScaled += applySynapticScaling(model);

  model.observations.unshift({
    step,
    eventType: event.eventType,
    url: event.url,
    title: event.title,
    elementText: learningSettings.sensitiveTextMode === 'full' ? event.elementText : redactSensitiveText(event.elementText).text,
    media: isMediaEvent(event.eventType) ? {
      id: event.mediaId,
      kind: event.mediaKind,
      src: event.mediaSrc,
      currentTime: event.mediaCurrentTime,
      duration: event.mediaDuration,
      progress: event.mediaProgress,
      paused: event.mediaPaused,
      muted: event.mediaMuted,
      volume: event.mediaVolume,
      playbackRate: event.mediaPlaybackRate,
      cueText: learningSettings.sensitiveTextMode === 'full' ? event.mediaCueText : redactSensitiveText(event.mediaCueText).text,
      transcriptText: learningSettings.sensitiveTextMode === 'full' ? event.mediaTranscriptText : redactSensitiveText(event.mediaTranscriptText).text,
    } : undefined,
    textRegion: event.textRegion ? true : undefined,
    visual: isVisualEvent(event.eventType) || visualTokens.length > 0 ? {
      id: event.visualId,
      kind: event.visualKind,
      width: event.visualWidth,
      height: event.visualHeight,
      visibleRatio: event.visualVisibleRatio,
      readablePixels: event.visualReadablePixels,
      tokenCount: visualTokens.length,
      text: safeVisualText || visualBindingRaw,
      textRegion: Boolean(event.textRegion),
    } : undefined,
    visibleText: (learningSettings.sensitiveTextMode === 'full' ? String(event.visibleText ?? '') : redactSensitiveText(event.visibleText).text).slice(0, 1000),
    reward,
    delayedOutcome,
    instinctDelayed,
    salience,
    novelty,
    semanticNovelty,
    discovery: {
      novelWordCount: discovery.novelWordCount,
      changeNovelWords: discovery.changeNovelWords,
      discoveryValence: discovery.discoveryValence,
      newPage: discovery.newPage,
      newEventType: discovery.newEventType,
    },
    morality: {
      harmRisk: morality.harmRisk,
      privacyRisk: morality.privacyRisk,
      deceptionRisk: morality.deceptionRisk,
      consentRisk: morality.consentRisk,
      prosocialSignal: morality.prosocialSignal,
      moralRisk: morality.moralRisk,
      moralValence: morality.moralValence,
      redactionCount: morality.redactionCount,
      mode: learningSettings.moralLearningMode,
    },
    instinct: {
      threatLevel: instinct.threatLevel,
      avoidanceUrgency: instinct.avoidanceUrgency,
      threatValence: instinct.threatValence,
      activeCrises: instinct.activeCrises,
      evaluationMode: instinct.evaluationMode,
      urlRisk: instinct.urlRisk,
      navigationTrap: instinct.navigationTrap,
      mode: learningSettings.instinctLearningMode,
    },
    neuromodulators: { ...model.neuromodulators },
    tokenCount: tokens.length,
    vocabularyBefore,
    vocabularyAfter: model.vocabulary.length,
    newTokenCount: Math.max(0, model.vocabulary.length - vocabularyBefore),
    vocabularyFull: Number(model.config.maxVocabulary || 0) > 0 && model.vocabulary.length >= model.config.maxVocabulary,
    at: event.at ?? Date.now(),
  });
  model.observations = model.observations
    .sort((a, b) => ((b.salience || 0) + Math.max(0, b.reward || 0) * 0.4 + (b.step || 0) * 0.0001) - ((a.salience || 0) + Math.max(0, a.reward || 0) * 0.4 + (a.step || 0) * 0.0001))
    .slice(0, model.config.maxObservationMemory || 180)
    .sort((a, b) => (b.step || 0) - (a.step || 0));
  trace.push({ step, kind: 'reward', value: reward, label: 'browser_reward' });
  trace.push(...homeostaticRegulation(model));
  trace.push(...consolidateMemory(model));
  const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startedAt;
  model.stats.lastStepMs = elapsedMs;
  model.stats.avgStepMs = model.stats.avgStepMs ? model.stats.avgStepMs * 0.92 + elapsedMs * 0.08 : elapsedMs;
  model.stats.cpuLoadEstimate = clamp((model.stats.avgStepMs || 0) / 50, 0, 1);
  model.stats.computeBackend = runtimeHints.backend || 'cpu';
  model.stats.performanceBudgetPercent = Number(runtimeHints.performanceBudgetPercent ?? learningSettings.performanceBudgetPercent ?? 65);
  model.stats.learningThrottleMs = Number(runtimeHints.learningThrottleMs || 0);
  model.stats.gpuTokenLimit = Number(runtimeHints.gpuTokenLimit || 0);
  model.stats.gpuThrottleMs = Number(runtimeHints.gpuThrottleMs || 0);
  model.stats.gpuThrottledSteps = (model.stats.gpuThrottledSteps || 0) + (runtimeHints.gpuThrottled ? 1 : 0);
  if (runtimeHints.webgpuAvailable !== undefined) model.stats.webgpuAvailable = Boolean(runtimeHints.webgpuAvailable);
  model.stats.modelBytes = estimateModelBytes(model);
  return { model, trace, reward };
};

export const encodeModelFile = (model) => {
  const payload = {
    kind: 'eden14-snn-life-model',
    version: 2,
    container: EDEN_SNN_MAGIC,
    exportedAt: new Date().toISOString(),
    modelCount: 1,
    models: [{
      creatureId: 'chrome-browser-language-snn',
      modelKind: 'browser-language',
      snapshot: model,
    }],
  };
  const encoded = new TextEncoder().encode(JSON.stringify(payload));
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  const header = new ArrayBuffer(magic.length + 4);
  const headerBytes = new Uint8Array(header);
  headerBytes.set(magic, 0);
  new DataView(header).setUint32(magic.length, encoded.byteLength, true);
  return new Blob([header, encoded], { type: 'application/x-edensnn' });
};
