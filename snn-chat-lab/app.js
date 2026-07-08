const EDEN_SNN_MAGIC = 'EDENSNN1';
const SUPPORTED_MODEL_DOMAINS = new Set(['browser-language', 'dst-web']);
const MODULE_TRIGGER_PATTERN = /^(module|skill|action|tool|media|moral|image|audio|video|text|body|visual|lex|source|instinct):/;

const state = {
  file: null,
  model: null,
  modelEntry: null,
  tokenById: new Map(),
  tokenByText: new Map(),
  aliasByText: new Map(),
  outgoing: new Map(),
  consolidatedMemories: [],
  lastActivations: [],
  lastResponse: null,
  lastPayload: null,
  lastSnnNode: null,
  spikeThreshold: 0.18,
};

const $ = (id) => document.getElementById(id);

const escapeHtml = (value) =>
  String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const format = (value, digits = 0) =>
  Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '-';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const STRUCTURAL_TOKENS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'so', 'because', 'if', 'then', 'than', 'to', 'of', 'in', 'on', 'at',
  'for', 'from', 'by', 'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that',
  'these', 'those', 'it', 'its', 'i', 'you', 'we', 'they', 'he', 'she',
  'の', 'に', 'は', 'を', 'が', 'と', 'で', 'も', 'へ', 'や', 'から', 'まで', 'より', 'です', 'ます',
  'そして', 'しかし', 'また', 'つまり', 'だから', 'ので', 'ため', 'こと', 'これ', 'それ', 'あれ'
]);

function tokenWeight(token) {
  const value = String(token || '');
  if (value.startsWith('image:') || value.startsWith('audio:') || value.startsWith('video:')) return 0.86;
  if (value.startsWith('text:') || value.startsWith('body:') || value.startsWith('visual:')) return 0.8;
  if (value.startsWith('lex:') || value.startsWith('instinct:')) return 0.74;
  if (value.startsWith('media:') || value.startsWith('moral:') || value.startsWith('source:')) return 0.72;
  if (STRUCTURAL_TOKENS.has(value)) return 0.16;
  if (value.length === 1 && /[\p{L}]/u.test(value)) return 0.28;
  return 1;
}

const isStructuralToken = (token) => tokenWeight(token) < 0.3;
const MODALITY_ALIASES = {
  image: ['image', 'visual', '画像', '視覚'],
  audio: ['audio', 'sound', '音声', '音'],
  video: ['video', 'movie', '動画', '映像'],
  text: ['text', '文章', 'テキスト', '言葉'],
  body: ['body', 'action', '操作', '身体', 'ブラウザ'],
};

function tokenHash(token) {
  let hash = 2166136261;
  for (const char of String(token || '')) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

const segmenter = typeof Intl !== 'undefined' && 'Segmenter' in Intl
  ? new Intl.Segmenter(['ja', 'en'], { granularity: 'word' })
  : null;

function tokenize(text) {
  const source = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!source) return [];
  if (segmenter) {
    return Array.from(segmenter.segment(source)).flatMap((part) => {
      if (!part.isWordLike) return [];
      const token = part.segment.trim().toLowerCase();
      return token ? [token] : [];
    });
  }
  return source.split(/[^\p{L}\p{N}_-]+/u).flatMap((part) => {
    const token = part.trim().toLowerCase();
    return token ? [token] : [];
  });
}

function pickModelEntry(container) {
  if (!Array.isArray(container.models)) return null;
  return container.models.find((entry) =>
    entry.snapshot && (
      entry.modelKind === 'browser-language'
      || entry.modelKind === 'dst-web'
      || SUPPORTED_MODEL_DOMAINS.has(entry.snapshot?.domain)
    )
  );
}

function buildExportContainer(model, sourceName) {
  const entry = {
    creatureId: model.source === 'dst-snn-pytorch' ? 'dst-web-snn' : 'snn-chat-model',
    modelKind: 'browser-language',
    importedFrom: sourceName,
    snapshot: model,
  };
  return {
    file: {
      kind: 'eden14-snn-life-model',
      version: 2,
      container: EDEN_SNN_MAGIC,
      exportedAt: new Date().toISOString(),
      modelCount: 1,
      models: [entry],
    },
    entry,
  };
}

function normalizePayload(parsed, sourceName) {
  if (parsed?.kind === 'dst-snn-chat-model' && parsed.model) {
    const model = normalizeModel(parsed.model);
    const container = buildExportContainer(model, sourceName);
    return { ...container, model };
  }

  if (Array.isArray(parsed?.models)) {
    const entry = pickModelEntry(parsed);
    if (!entry?.snapshot) throw new Error('No compatible SNN model found.');
    return { file: parsed, entry, model: normalizeModel(entry.snapshot) };
  }

  const model = parsed?.model?.vocabulary ? parsed.model : parsed?.snapshot?.vocabulary ? parsed.snapshot : parsed;
  if (!Array.isArray(model?.vocabulary)) throw new Error('No compatible SNN vocabulary found.');
  const normalized = normalizeModel(model);
  const container = buildExportContainer(normalized, sourceName);
  return { ...container, model: normalized };
}

async function decodeEdenSnnFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  if (bytes.length < magic.length + 4) throw new Error('File is too small to be .edensnn.');
  for (let index = 0; index < magic.length; index += 1) {
    if (bytes[index] !== magic[index]) throw new Error('Invalid .edensnn header.');
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(magic.length, true);
  const start = magic.length + 4;
  const end = start + length;
  if (end > bytes.length) throw new Error('Truncated .edensnn payload.');
  const parsed = JSON.parse(new TextDecoder().decode(bytes.slice(start, end)));
  if (!Array.isArray(parsed.models)) throw new Error('No models found in .edensnn.');
  return normalizePayload(parsed, file.name);
}

async function decodeJsonModelFile(file) {
  const parsed = JSON.parse(await file.text());
  return normalizePayload(parsed, file.name);
}

async function decodePtModelFile(file) {
  let response;
  try {
    response = await fetch('/api/convert-pt', {
      method: 'POST',
      headers: {
        'content-type': 'application/octet-stream',
        'x-filename': encodeURIComponent(file.name),
      },
      body: await file.arrayBuffer(),
    });
  } catch (error) {
    throw new Error(`.pt を直接読み込むにはローカル変換サーバが必要です。python scripts/serve_snn_chat_lab.py を実行して http://127.0.0.1:8765 を開くか、python scripts/export_dst_chat_model.py ${file.name} で .chat.json に変換してください。`);
  }

  const parsed = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = parsed?.error ? ` (${parsed.error})` : '';
    throw new Error(`.pt の変換に失敗しました${detail}。python scripts/serve_snn_chat_lab.py から起動しているか確認してください。`);
  }
  return normalizePayload(parsed, file.name);
}

async function decodeModelFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith('.pt')) return decodePtModelFile(file);
  if (name.endsWith('.json') || name.endsWith('.chat.json')) return decodeJsonModelFile(file);
  if (name.endsWith('.edensnn')) return decodeEdenSnnFile(file);
  try {
    return await decodeJsonModelFile(file);
  } catch {
    return decodeEdenSnnFile(file);
  }
}

function normalizeModel(model) {
  const source = model || {};
  const vocabulary = Array.isArray(source.vocabulary)
    ? source.vocabulary
      .map((neuron, index) => ({
        ...neuron,
        id: Number.isFinite(Number(neuron.id)) ? Number(neuron.id) : index,
        token: String(neuron.token ?? '').toLowerCase(),
        count: Number(neuron.count || 0),
      }))
      .filter((neuron) => neuron.token)
    : [];
  const associations = Array.isArray(source.associations)
    ? source.associations
      .filter((edge) => Number.isFinite(Number(edge.pre)) && Number.isFinite(Number(edge.post)))
      .map((edge, index) => ({
        ...edge,
        id: Number.isFinite(Number(edge.id)) ? Number(edge.id) : index,
        pre: Number(edge.pre),
        post: Number(edge.post),
        w: Number(edge.w || 0),
      }))
    : [];
  const observations = Array.isArray(source.observations) ? source.observations : [];
  const crossModalRelations = Array.isArray(source.crossModalRelations) ? source.crossModalRelations : [];
  return {
    ...source,
    stats: source.stats || {},
    vocabulary,
    associations,
    observations,
    crossModalRelations,
  };
}

function addAlias(alias, neuron) {
  const normalized = String(alias || '').toLowerCase();
  if (normalized.length < 2 || isStructuralToken(normalized)) return;
  if (!state.aliasByText.has(normalized)) state.aliasByText.set(normalized, []);
  const bucket = state.aliasByText.get(normalized);
  if (!bucket.some((item) => item.id === neuron.id)) bucket.push(neuron);
}

function indexAliases(neuron) {
  const token = String(neuron.token || '').toLowerCase();
  const segments = token.split(':').filter(Boolean);
  if (segments.length < 2) return;
  addAlias(segments.at(-1), neuron);
  if (segments.length >= 3) addAlias(segments.at(-2), neuron);
  for (const alias of MODALITY_ALIASES[segments[0]] || []) addAlias(alias, neuron);
}

function promptMatches(token) {
  const normalized = String(token || '').toLowerCase();
  const matches = new Map();
  const exact = state.tokenByText.get(normalized);
  if (exact) matches.set(exact.id, { neuron: exact, boost: 1 });
  for (const neuron of state.aliasByText.get(normalized) || []) {
    if (!matches.has(neuron.id)) matches.set(neuron.id, { neuron, boost: 0.72 });
  }
  return [...matches.values()]
    .sort((a, b) => ((b.neuron.count || 0) + (b.neuron.stability || 0) * 20) - ((a.neuron.count || 0) + (a.neuron.stability || 0) * 20))
    .slice(0, 20);
}

function indexModel() {
  state.tokenById = new Map();
  state.tokenByText = new Map();
  state.aliasByText = new Map();
  state.outgoing = new Map();

  for (const neuron of state.model.vocabulary) {
    neuron.importance ??= tokenWeight(neuron.token);
    neuron.role ??= isStructuralToken(neuron.token) ? 'structural' : 'semantic';
    state.tokenById.set(neuron.id, neuron);
    state.tokenByText.set(String(neuron.token).toLowerCase(), neuron);
    indexAliases(neuron);
  }

  for (const edge of state.model.associations) {
    if (!state.outgoing.has(edge.pre)) state.outgoing.set(edge.pre, []);
    state.outgoing.get(edge.pre).push(edge);
  }

  state.consolidatedMemories = [];
}

function findOrCreateNeuron(token) {
  const normalized = String(token || '').toLowerCase();
  let neuron = state.tokenByText.get(normalized);
  if (neuron) return neuron;
  neuron = {
    id: state.model.vocabulary.length,
    token: normalized,
    count: 0,
    v: 0,
    threshold: 1,
    positiveSpikeMass: 0,
    negativeSpikeMass: 0,
    stability: 0,
    importance: tokenWeight(normalized),
    role: isStructuralToken(normalized) ? 'structural' : 'semantic',
    lastSeenStep: state.model.stats?.steps || 0,
  };
  state.model.vocabulary.push(neuron);
  state.tokenById.set(neuron.id, neuron);
  state.tokenByText.set(normalized, neuron);
  indexAliases(neuron);
  return neuron;
}

function findOrCreateAssociation(pre, post) {
  const edges = state.outgoing.get(pre) || [];
  let association = edges.find((edge) => edge.pre === pre && edge.post === post)
    || state.model.associations.find((edge) => edge.pre === pre && edge.post === post);
  if (association) return association;
  association = {
    id: state.model.associations.length,
    pre,
    post,
    w: 0.08,
    aPre: 0,
    aPost: 0,
    stability: 0,
    replayCount: 0,
    d1Go: 0.1,
    d2NoGo: 0.01,
    rewardPrediction: 0,
    semanticGate: 1,
  };
  state.model.associations.push(association);
  if (!state.outgoing.has(pre)) state.outgoing.set(pre, []);
  state.outgoing.get(pre).push(association);
  return association;
}

function reinforcePrompt(prompt, ranked) {
  if (!state.model) return [];
  state.model.stats ??= {};
  state.model.stats.chatReinforcements = (state.model.stats.chatReinforcements || 0) + 1;
  const promptTokens = tokenize(prompt).filter((token) => tokenWeight(token) >= 0.3).slice(0, 80);
  const promptNeurons = promptTokens.map(findOrCreateNeuron);
  const activeNeurons = ranked
    .slice(0, 12)
    .map((item) => item.neuron)
    .filter((neuron) => neuron && tokenWeight(neuron.token) >= 0.3);
  const targets = [...promptNeurons, ...activeNeurons];
  const uniqueTargets = [...new Map(targets.map((neuron) => [neuron.id, neuron])).values()];

  for (const neuron of promptNeurons) {
    const weight = tokenWeight(neuron.token);
    neuron.count = (neuron.count || 0) + 2.4 * weight;
    neuron.positiveSpikeMass = (neuron.positiveSpikeMass || 0) + 1.5 * weight;
    neuron.stability = clamp((neuron.stability || 0) + 0.018 * weight, 0, 1);
    neuron.importance = weight;
    neuron.role = isStructuralToken(neuron.token) ? 'structural' : 'semantic';
  }

  for (const pre of promptNeurons) {
    for (const post of uniqueTargets) {
      if (pre.id === post.id) continue;
      const gate = clamp(Math.sqrt(tokenWeight(pre.token) * tokenWeight(post.token)), 0.08, 1);
      const association = findOrCreateAssociation(pre.id, post.id);
      association.w = clamp((association.w || 0) + 0.12 * gate, -1, 1);
      association.stability = clamp((association.stability || 0) + 0.035 * gate, 0, 1);
      association.d1Go = clamp((association.d1Go || 0) + 0.05 * gate, 0, 1);
      association.d2NoGo = clamp((association.d2NoGo || 0) * 0.96, 0, 1);
      association.semanticGate = gate;
      association.replayCount = (association.replayCount || 0) + 1;
      association.lastUpdatedStep = state.model.stats.steps || 0;
    }
  }

  state.model.stats.modelBytes = new TextEncoder().encode(JSON.stringify(state.model)).byteLength;
  return promptTokens;
}

function activate(prompt) {
  const promptTokens = tokenize(prompt);
  const activations = new Map();

  for (const token of promptTokens) {
    for (const match of promptMatches(token)) {
      const neuron = match.neuron;
      const weight = Math.sqrt(tokenWeight(token) * tokenWeight(neuron.token));
      const base = (1 + Math.log1p(neuron.count || 0) * 0.18 + (neuron.positiveSpikeMass || 0) * 0.015) * weight * match.boost;
      activations.set(neuron.id, (activations.get(neuron.id) || 0) + base);
    }
  }

  for (let depth = 0; depth < 2; depth += 1) {
    const next = new Map(activations);
    for (const [id, value] of activations.entries()) {
      const edges = state.outgoing.get(id) || [];
      for (const edge of edges) {
        if ((edge.w || 0) <= 0) continue;
        const post = state.tokenById.get(edge.post);
        const propagated = value * edge.w * tokenWeight(post?.token) * (depth === 0 ? 0.72 : 0.36);
        next.set(edge.post, (next.get(edge.post) || 0) + propagated);
      }
    }
    for (const [id, value] of next.entries()) activations.set(id, value);
  }

  const ranked = [...activations.entries()]
    .map(([id, score]) => ({ neuron: state.tokenById.get(id), score }))
    .filter((item) => item.neuron)
    .sort((a, b) => b.score - a.score);
  state.lastActivations = ranked.slice(0, 40);
  return { promptTokens, ranked };
}

function strongestLinks(ranked) {
  const activeIds = new Set(ranked.slice(0, 18).map((item) => item.neuron.id));
  return state.model.associations
    .filter((edge) => activeIds.has(edge.pre) || activeIds.has(edge.post))
    .sort((a, b) => Math.abs(b.w || 0) - Math.abs(a.w || 0))
    .slice(0, 8);
}

function buildActivationGraph(ranked) {
  const activeIds = new Set(ranked.slice(0, 24).map((item) => item.neuron.id));
  const scoreById = new Map(ranked.map((item) => [item.neuron.id, item.score]));
  return state.model.associations
    .filter((edge) => activeIds.has(edge.pre) && activeIds.has(edge.post) && (edge.w || 0) > 0)
    .map((edge) => {
      const pre = state.tokenById.get(edge.pre);
      const post = state.tokenById.get(edge.post);
      const score = (scoreById.get(edge.pre) || 0) + (scoreById.get(edge.post) || 0) + Math.abs(edge.w || 0) * 2;
      return { edge, pre, post, score };
    })
    .filter((item) => item.pre && item.post)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
}

function buildNeuronOutputPayload(promptTokens, ranked, graph) {
  const maxScore = Math.max(0.001, ...ranked.map((item) => item.score));
  const outputNeurons = ranked.slice(0, 24).map((item, index) => {
    const normalized = clamp(item.score / maxScore, 0, 1);
    return {
      rank: index + 1,
      id: item.neuron.id,
      token: item.neuron.token,
      activation: Number(item.score.toFixed(6)),
      normalized: Number(normalized.toFixed(6)),
      spike: Number(normalized.toFixed(6)),
      count: item.neuron.count || 0,
      stability: Number((item.neuron.stability || 0).toFixed(6)),
      voltage: Number((item.neuron.v || 0).toFixed(6)),
      positiveSpikeMass: Number((item.neuron.positiveSpikeMass || 0).toFixed(6)),
      negativeSpikeMass: Number((item.neuron.negativeSpikeMass || 0).toFixed(6)),
      routeKey: `neuron:${item.neuron.token}`,
    };
  });
  const synapsePaths = graph.slice(0, 16).map((item) => ({
    pre: {
      id: item.pre.id,
      token: item.pre.token,
    },
    post: {
      id: item.post.id,
      token: item.post.token,
    },
    weight: Number((item.edge.w || 0).toFixed(6)),
    stability: Number((item.edge.stability || 0).toFixed(6)),
    delayedCredit: Number((item.edge.delayedCredit || 0).toFixed(6)),
    pathScore: Number(item.score.toFixed(6)),
  }));
  const activationMass = outputNeurons.reduce((sum, neuron) => sum + neuron.activation, 0);
  return {
    kind: 'snn_neuron_output',
    promptTokens,
    firedCount: outputNeurons.length,
    activationMass: Number(activationMass.toFixed(6)),
    outputNeurons,
    spikeEvents: outputNeurons.map((neuron) => ({
      type: 'snn.spike',
      neuronId: neuron.id,
      token: neuron.token,
      routeKey: neuron.routeKey,
      strength: neuron.spike,
      activation: neuron.activation,
    })),
    moduleTriggers: outputNeurons
      .filter((neuron) => MODULE_TRIGGER_PATTERN.test(String(neuron.token)))
      .map((neuron) => ({
        trigger: neuron.token,
        strength: neuron.spike,
        source: 'snn.spike',
      })),
    synapsePaths,
  };
}

function thresholdPayload(payload) {
  const threshold = state.spikeThreshold;
  const outputNeurons = payload.outputNeurons.map((neuron) => ({
    ...neuron,
    masked: neuron.spike < threshold,
  }));
  const fullText = outputNeurons
    .filter((neuron) => !neuron.masked)
    .map((neuron) => String(neuron.token ?? ''))
    .join('');
  return {
    ...payload,
    spikeThreshold: Number(threshold.toFixed(2)),
    fullText,
    outputNeurons,
  };
}

function renderNeuronOutput(payload) {
  const thresholded = thresholdPayload(payload);
  const rows = thresholded.outputNeurons.length
    ? thresholded.outputNeurons.map((neuron) => `
      <tr class="${neuron.masked ? 'masked-neuron' : ''}">
        <td>${escapeHtml(neuron.rank)}</td>
        <td>${escapeHtml(neuron.id)}</td>
        <td>${escapeHtml(neuron.token)}</td>
        <td>${escapeHtml(format(neuron.activation, 4))}</td>
        <td>${escapeHtml(format(neuron.spike, 4))}</td>
        <td>${escapeHtml(format(neuron.stability, 4))}</td>
        <td>${escapeHtml(format(neuron.count, 0))}</td>
        <td>${neuron.masked ? 'MASKED' : 'PASS'}</td>
      </tr>
    `).join('')
    : '<tr><td colspan="8">NO_SPIKE</td></tr>';
  const paths = thresholded.synapsePaths.length
    ? thresholded.synapsePaths.map((path) => `
      <tr>
        <td>${escapeHtml(path.pre.token)}</td>
        <td>${escapeHtml(path.post.token)}</td>
        <td>${escapeHtml(format(path.weight, 4))}</td>
        <td>${escapeHtml(format(path.stability, 4))}</td>
        <td>${escapeHtml(format(path.delayedCredit, 4))}</td>
      </tr>
    `).join('')
    : '<tr><td colspan="5">NO_ACTIVE_PATH</td></tr>';

  return `
    <div class="neuron-output">
      <h3>SNN_NEURON_TEXT</h3>
      <p class="neuron-fulltext">${thresholded.fullText ? escapeHtml(thresholded.fullText) : 'NO_SPIKE'}</p>
      <h3>SNN_NEURON_OUTPUT</h3>
      <table class="neuron-table">
        <thead>
          <tr>
            <th>#</th>
            <th>ID</th>
            <th>Neuron</th>
            <th>Activation</th>
            <th>Spike</th>
            <th>Stability</th>
            <th>Count</th>
            <th>Mask</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <details>
        <summary>Active synapse paths</summary>
        <table class="neuron-table">
          <thead>
            <tr>
              <th>Pre</th>
              <th>Post</th>
              <th>Weight</th>
              <th>Stability</th>
              <th>Delayed</th>
            </tr>
          </thead>
          <tbody>${paths}</tbody>
        </table>
      </details>
      <details>
        <summary>Raw vector</summary>
        <pre class="vector-block">${escapeHtml(JSON.stringify(thresholded, null, 2))}</pre>
      </details>
    </div>
  `;
}

function generateAnswer(prompt) {
  if (!state.model) {
    return {
      html: '<p>先に `.edensnn` / `.pt` / `.chat.json` を読み込んでください。</p>',
      confidence: 0,
    };
  }

  const { promptTokens, ranked } = activate(prompt);
  reinforcePrompt(prompt, ranked);
  const activated = activate(prompt);
  const reinforcedPromptTokens = activated.promptTokens.length ? activated.promptTokens : promptTokens;
  const reinforcedRanked = activated.ranked;
  const graph = buildActivationGraph(reinforcedRanked);
  const payload = buildNeuronOutputPayload(reinforcedPromptTokens, reinforcedRanked, graph);
  const confidence = Math.min(1, payload.activationMass / 12);
  state.lastResponse = { prompt, promptTokens: reinforcedPromptTokens, ranked: reinforcedRanked.slice(0, 48), memories: [], consolidated: [], graph };
  state.lastPayload = payload;

  return {
    html: renderNeuronOutput(payload),
    confidence,
  };
}

function wavBlobFromFloat32(samples, sampleRate) {
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (const sample of samples) {
    view.setInt16(offset, clamp(sample, -1, 1) * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function downloadBlob(blob, filename, keepUrl = false) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  if (!keepUrl) setTimeout(() => URL.revokeObjectURL(url), 1000);
  return url;
}

function encodeCurrentModelFile() {
  if (!state.model || !state.file || !state.modelEntry) return null;
  state.model.savedAt = Date.now();
  state.modelEntry.snapshot = state.model;
  state.file.exportedAt = new Date().toISOString();
  state.file.models = state.file.models.map((entry) => entry === state.modelEntry ? state.modelEntry : entry);
  const payload = new TextEncoder().encode(JSON.stringify(state.file));
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  const header = new ArrayBuffer(magic.length + 4);
  const headerBytes = new Uint8Array(header);
  headerBytes.set(magic, 0);
  new DataView(header).setUint32(magic.length, payload.byteLength, true);
  return new Blob([header, payload], { type: 'application/x-edensnn' });
}

function exportUpdatedModel() {
  const blob = encodeCurrentModelFile();
  if (!blob) {
    addMessage('snn', '<p>先に `.edensnn` / `.pt` / `.chat.json` を読み込んでください。</p>');
    return;
  }
  const timestamp = new Date().toISOString().replaceAll(':', '-').replace(/\.\d+Z$/, 'Z');
  downloadBlob(blob, `snn-chat-reinforced-${timestamp}.edensnn`);
}

function activeRenderData() {
  const response = state.lastResponse;
  if (!response || response.ranked.length === 0) return null;
  const neurons = response.ranked.slice(0, 16).map((item, index) => {
    const hash = tokenHash(item.neuron.token);
    return {
      index,
      token: item.neuron.token,
      score: item.score,
      frequency: 120 + (hash % 920),
      phase: ((hash >>> 8) % 628) / 100,
      hue: hash % 360,
      stability: item.neuron.stability || 0,
      count: item.neuron.count || 0,
    };
  });
  const maxScore = Math.max(0.001, ...neurons.map((item) => item.score));
  for (const neuron of neurons) neuron.energy = clamp(neuron.score / maxScore, 0.08, 1);
  return { ...response, neurons };
}

function renderNeuronAudio() {
  const data = activeRenderData();
  if (!data) {
    $('renderStatus').textContent = 'Ask the model before rendering.';
    return;
  }
  const sampleRate = 44100;
  const duration = 5;
  const samples = new Float32Array(sampleRate * duration);
  const graphBoost = Math.min(0.35, data.graph.length * 0.025);

  for (let i = 0; i < samples.length; i += 1) {
    const t = i / sampleRate;
    let value = 0;
    for (const neuron of data.neurons) {
      const burst = 0.58 + 0.42 * Math.sin(Math.PI * 2 * (0.5 + neuron.energy) * t + neuron.phase);
      const carrier = Math.sin(Math.PI * 2 * neuron.frequency * t + neuron.phase);
      const overtone = Math.sin(Math.PI * 2 * neuron.frequency * 1.5 * t) * 0.22;
      value += (carrier + overtone) * burst * neuron.energy;
    }
    const envelope = Math.sin(Math.PI * Math.min(1, t / duration));
    samples[i] = (value / Math.max(1, data.neurons.length)) * (0.62 + graphBoost) * envelope;
  }

  const blob = wavBlobFromFloat32(samples, sampleRate);
  const url = downloadBlob(blob, `snn-neuron-response-${Date.now()}.wav`, true);
  $('audioPreview').src = url;
  $('renderStatus').textContent = `Rendered ${data.neurons.length} active neurons as WAV.`;
}

async function renderNeuronVideo() {
  const data = activeRenderData();
  if (!data) {
    $('renderStatus').textContent = 'Ask the model before rendering.';
    return;
  }
  if (!HTMLCanvasElement.prototype.captureStream || typeof MediaRecorder === 'undefined') {
    $('renderStatus').textContent = 'Video rendering is not supported in this browser.';
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = 960;
  canvas.height = 540;
  const ctx = canvas.getContext('2d');
  const stream = canvas.captureStream(30);
  const recorder = new MediaRecorder(stream, {
    mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm',
  });
  const chunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  const done = new Promise((resolve) => {
    recorder.onstop = resolve;
  });

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = 180;
  const positions = data.neurons.map((neuron, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, data.neurons.length) - Math.PI / 2;
    return { ...neuron, x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius };
  });
  const byToken = new Map(positions.map((item) => [item.token, item]));

  recorder.start();
  $('renderStatus').textContent = 'Rendering WebM from neuron activity...';
  for (let frame = 0; frame < 150; frame += 1) {
    const t = frame / 30;
    ctx.fillStyle = '#071018';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#dff7ff';
    ctx.font = '22px system-ui, sans-serif';
    ctx.fillText('SNN Neuron Response', 28, 40);
    ctx.font = '14px system-ui, sans-serif';
    ctx.fillStyle = '#8fb0c4';
    ctx.fillText(data.prompt.slice(0, 90), 28, 66);

    for (const item of data.graph.slice(0, 18)) {
      const a = byToken.get(item.pre.token);
      const b = byToken.get(item.post.token);
      if (!a || !b) continue;
      const pulse = 0.35 + 0.65 * Math.max(0, Math.sin(t * 4 + item.score));
      ctx.strokeStyle = `rgba(130, 220, 190, ${0.14 + pulse * 0.34})`;
      ctx.lineWidth = 1 + Math.abs(item.edge.w || 0) * 4;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (const neuron of positions) {
      const spike = 0.55 + 0.45 * Math.sin(t * (2 + neuron.energy * 5) + neuron.phase);
      const size = 10 + neuron.energy * 24 + spike * 8;
      ctx.fillStyle = `hsla(${neuron.hue}, 72%, 58%, ${0.58 + neuron.energy * 0.32})`;
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#e8f6ff';
      ctx.font = '13px system-ui, sans-serif';
      ctx.fillText(neuron.token.slice(0, 18), neuron.x + size + 5, neuron.y + 4);
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  recorder.stop();
  await done;
  const blob = new Blob(chunks, { type: 'video/webm' });
  downloadBlob(blob, `snn-neuron-response-${Date.now()}.webm`);
  $('renderStatus').textContent = `Rendered ${positions.length} active neurons as WebM.`;
}

function updateRenderControls() {
  const enabled = Boolean(state.lastResponse?.ranked?.length);
  $('audioRenderButton').disabled = !enabled;
  $('videoRenderButton').disabled = !enabled;
}

function renderMetrics() {
  const model = state.model;
  const stats = model?.stats || {};
  const rows = model
    ? [
      ['Vocabulary', model.vocabulary.length],
      ['Associations', model.associations.length],
      ['Observations', stats.observations || model.observations.length],
      ['Cross-modal', stats.crossModalRelations || model.crossModalRelations.length],
      ['Positive spikes', stats.positiveSpikes || 0],
      ['Negative spikes', stats.negativeSpikes || 0],
      ['Sparse ops', stats.sparseAcOps || 0],
    ]
    : [
      ['Vocabulary', 0],
      ['Associations', 0],
      ['Observations', 0],
      ['Cross-modal', 0],
      ['Positive spikes', 0],
      ['Negative spikes', 0],
      ['Sparse ops', 0],
    ];

  $('metrics').innerHTML = rows.map(([label, value]) => `
    <article class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(format(value, 1))}</strong>
    </article>
  `).join('');
}

function renderModelPanels() {
  if (!state.model) {
    $('activeNeurons').innerHTML = '<span class="token">no model</span>';
    $('strongLinks').innerHTML = '<div class="link">no model</div>';
    return;
  }

  const tokens = (state.lastActivations.length ? state.lastActivations.map((item) => item.neuron) : [...state.model.vocabulary].sort((a, b) => (b.count || 0) - (a.count || 0)))
    .slice(0, 36);
  $('activeNeurons').innerHTML = tokens.length
    ? tokens.map((neuron) => `<span class="token">${escapeHtml(neuron.token)} ${escapeHtml(format(neuron.count || 0))}</span>`).join('')
    : '<span class="token">no tokens</span>';

  const links = (state.lastActivations.length ? strongestLinks(state.lastActivations) : [...state.model.associations].sort((a, b) => Math.abs(b.w || 0) - Math.abs(a.w || 0)).slice(0, 12));
  $('strongLinks').innerHTML = links.length
    ? links.map((edge) => {
      const pre = state.tokenById.get(edge.pre)?.token ?? edge.pre;
      const post = state.tokenById.get(edge.post)?.token ?? edge.post;
      return `<div class="link"><span>${escapeHtml(pre)} -> ${escapeHtml(post)}</span><strong>${escapeHtml(format(edge.w || 0, 3))}</strong></div>`;
    }).join('')
    : '<div class="link">no links</div>';
}

function addMessage(kind, html) {
  const node = document.createElement('article');
  node.className = `message ${kind}`;
  node.innerHTML = `<span class="label">${kind === 'user' ? 'Input' : 'SNN'}</span>${html}`;
  $('chatLog').appendChild(node);
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
  return node;
}

function setMessageHtml(node, kind, html) {
  if (!node) return;
  node.innerHTML = `<span class="label">${kind === 'user' ? 'Input' : 'SNN'}</span>${html}`;
}

function updateThresholdLabel() {
  $('spikeThresholdValue').textContent = state.spikeThreshold.toFixed(2);
}

function rerenderLastNeuronOutput() {
  if (!state.lastPayload || !state.lastSnnNode) return;
  setMessageHtml(state.lastSnnNode, 'snn', renderNeuronOutput(state.lastPayload));
}

async function importModel(file) {
  const decoded = await decodeModelFile(file);
  state.file = decoded.file;
  state.modelEntry = decoded.entry;
  state.model = decoded.model;
  state.lastActivations = [];
  state.lastResponse = null;
  state.lastPayload = null;
  state.lastSnnNode = null;
  indexModel();
  $('modelStatus').textContent = `${file.name} / ${state.model.vocabulary.length} neurons / ${state.model.associations.length} links`;
  renderMetrics();
  renderModelPanels();
  updateRenderControls();
  addMessage('snn', `<p>${escapeHtml(file.name)} を読み込みました。入力に対するニューロン出力を検査できます。</p>`);
}

function clearModel() {
  state.file = null;
  state.model = null;
  state.modelEntry = null;
  state.tokenById = new Map();
  state.tokenByText = new Map();
  state.aliasByText = new Map();
  state.outgoing = new Map();
  state.lastActivations = [];
  state.lastResponse = null;
  state.lastPayload = null;
  state.lastSnnNode = null;
  $('modelStatus').textContent = 'No model loaded.';
  $('chatLog').innerHTML = '';
  renderMetrics();
  renderModelPanels();
  updateRenderControls();
  addMessage('snn', '<p>学習済み `.edensnn` / `.pt` / `.chat.json` を読み込むと、入力に対するニューロン出力を確認できます。</p>');
}

$('importButton').addEventListener('click', () => $('modelFileInput').click());
$('clearButton').addEventListener('click', clearModel);
$('exportButton').addEventListener('click', exportUpdatedModel);
$('modelFileInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await importModel(file);
  } catch (error) {
    addMessage('snn', `<p>読み込みに失敗しました: ${escapeHtml(error.message || error)}</p>`);
  } finally {
    event.target.value = '';
  }
});

$('chatForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const prompt = $('promptInput').value.trim();
  if (!prompt) return;
  $('promptInput').value = '';
  addMessage('user', `<p>${escapeHtml(prompt)}</p>`);
  const answer = generateAnswer(prompt);
  state.lastSnnNode = addMessage('snn', answer.html);
  renderMetrics();
  renderModelPanels();
  updateRenderControls();
});

$('spikeThresholdInput').addEventListener('input', (event) => {
  state.spikeThreshold = Number(event.target.value);
  updateThresholdLabel();
  rerenderLastNeuronOutput();
});

$('audioRenderButton').addEventListener('click', renderNeuronAudio);
$('videoRenderButton').addEventListener('click', () => {
  void renderNeuronVideo();
});

clearModel();
updateThresholdLabel();
