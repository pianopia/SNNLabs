const EDEN_SNN_MAGIC = 'EDENSNN1';
const STORAGE_KEY = 'browser-snn-language-lab:v2';

const pages = {
  habitat: {
    title: 'Browser Language Habitat',
    links: ['reward', 'tools', 'memory'],
    blocks: [
      ['h1', 'Browser Language Habitat'],
      ['p', 'SNN agents observe browser actions, move the cursor, scroll the viewport, click links, and learn language from visible text.'],
      ['p', '探索、クリック、スクロール、読解の結果として得た言語を、符号付きスパイクの刺激として学習します。'],
      ['card', 'The current goal is curiosity. New words and unseen sections produce positive reward. Repeated or empty views reduce reward.'],
      ['h2', 'Autonomous operation'],
      ['p', 'The agent chooses between reading, scrolling, clicking, and searching. Each action creates an observation event.'],
      ['link', 'reward', 'Open reward and memory'],
      ['link', 'tools', 'Open browser tools'],
      ['link', 'memory', 'Open language memory'],
    ],
  },
  reward: {
    title: 'Reward and Memory',
    links: ['habitat', 'tools', 'memory'],
    blocks: [
      ['h1', 'Reward and Memory'],
      ['p', 'Useful discoveries receive positive reward. Distracting, repeated, or uninformative views receive weaker or negative reward.'],
      ['p', '報酬は単語ニューロンと隣接関連シナプスの可塑性を変調します。'],
      ['card', 'Positive reward strengthens associations between operation context and visible language. Negative reward weakens unstable routes.'],
      ['h2', 'Sparse spikes'],
      ['p', 'Signed integer spikes encode positive and negative language evidence while keeping sparse operation statistics.'],
      ['link', 'habitat', 'Back to habitat'],
      ['link', 'tools', 'Open browser tools'],
    ],
  },
  tools: {
    title: 'Browser Tools',
    links: ['habitat', 'reward', 'memory'],
    blocks: [
      ['h1', 'Browser Tools'],
      ['p', 'A browser body includes pointer position, scroll depth, click targets, visible text, and operation history.'],
      ['search', 'Search memory, reward, sensory input, cursor, scroll, language, and association.'],
      ['card', 'Search actions create intent language. Click actions bind target text to subsequent reading. Scroll actions expose hidden text.'],
      ['h2', 'Mouse and scroll'],
      ['p', 'The cursor is part of the body. Moving it toward a target is an embodied precursor to click and read operations.'],
      ['link', 'memory', 'Open language memory'],
      ['link', 'reward', 'Open reward and memory'],
    ],
  },
  memory: {
    title: 'Language Memory',
    links: ['habitat', 'reward', 'tools'],
    blocks: [
      ['h1', 'Language Memory'],
      ['p', 'Language memory is built from token neurons and directional associations between neighboring words.'],
      ['p', '繰り返し出現する語彙は安定し、新しい語彙は探索報酬によって強く記録されます。'],
      ['card', 'A learned .edensnn model can be exported and inspected in the EDEN SNN Dashboard without depending on the EDEN runtime.'],
      ['h2', 'Next behavior'],
      ['p', 'Future agents can use this memory to choose which page to open, which link to click, and which text to ignore.'],
      ['link', 'habitat', 'Back to habitat'],
      ['link', 'tools', 'Open browser tools'],
    ],
  },
};

const $ = (id) => document.getElementById(id);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const format = (value) => Number.isFinite(value) ? value.toFixed(3) : '-';
const normalizeToken = (token) => token.trim().toLowerCase();

const segmenter = typeof Intl !== 'undefined' && 'Segmenter' in Intl
  ? new Intl.Segmenter(['ja', 'en'], { granularity: 'word' })
  : null;

const tokenize = (text) => {
  const source = text.replace(/\s+/g, ' ').trim();
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

const createModel = () => ({
  version: 1,
  domain: 'browser-language',
  savedAt: Date.now(),
  config: {
    neuronType: 'si-lif',
    spikeRangeD: 4,
    maxVocabulary: 220,
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

let model = loadModel();
let traceEvents = [];
let operationLog = [];
let currentPageId = 'habitat';
let autoTimer = null;
let agent = {
  cursorX: 42,
  cursorY: 42,
  visitedSections: new Set(),
  lastAction: 'observe',
  searchQuery: 'language memory',
};

function loadModel() {
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return createModel();
  try {
    const parsed = JSON.parse(raw);
    if (parsed?.domain !== 'browser-language') return createModel();
    return parsed;
  } catch {
    return createModel();
  }
}

function saveModel() {
  model.savedAt = Date.now();
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(model));
}

function pushLog(message) {
  operationLog = [`${new Date().toLocaleTimeString()} ${message}`, ...operationLog].slice(0, 36);
}

function findOrCreateNeuron(token) {
  let neuron = model.vocabulary.find((item) => item.token === token);
  if (neuron) return neuron;
  if (model.vocabulary.length >= model.config.maxVocabulary) {
    neuron = model.vocabulary[0];
    for (const candidate of model.vocabulary) {
      if (candidate.count < neuron.count || (candidate.count === neuron.count && candidate.lastSeenStep < neuron.lastSeenStep)) {
        neuron = candidate;
      }
    }
    Object.assign(neuron, { token, count: 0, v: 0, positiveSpikeMass: 0, negativeSpikeMass: 0 });
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
  };
  model.vocabulary.push(neuron);
  return neuron;
}

function findOrCreateAssociation(pre, post) {
  let association = model.associations.find((item) => item.pre === pre && item.post === post);
  if (association) return association;
  association = { id: model.associations.length, pre, post, w: 0.08, aPre: 0, aPost: 0 };
  model.associations.push(association);
  return association;
}

function observeLanguage({ operation, source, text, reward }) {
  const tokens = tokenize(text).slice(0, 360);
  const boundedReward = clamp(Number(reward) || 0, -1, 1);
  const step = model.stats.steps + 1;
  const events = [];

  model.stats.steps = step;
  model.stats.observations += 1;
  model.stats.totalTokens += tokens.length;
  model.stats.denseMacOps += Math.max(1, model.associations.length);
  model.savedAt = Date.now();
  model.observations.unshift({ step, operation, source, text: text.slice(0, 1200), reward: boundedReward, tokenCount: tokens.length });
  model.observations = model.observations.slice(0, 80);
  events.push({ step, kind: 'operation', value: tokens.length, label: operation, meta: { source } });

  let previous = null;
  for (const token of tokens) {
    const neuron = findOrCreateNeuron(token);
    neuron.count += 1;
    neuron.lastSeenStep = step;
    neuron.v = neuron.v * 0.72 + 0.55 + Math.abs(boundedReward) * 0.25;
    const signedSpike = clamp(Math.round(neuron.v * (boundedReward < 0 ? -1 : 1)), -model.config.spikeRangeD, model.config.spikeRangeD);
    if (signedSpike !== 0) {
      neuron.v = 0;
      if (signedSpike > 0) {
        neuron.positiveSpikeMass += signedSpike;
        model.stats.positiveSpikes += signedSpike;
      } else {
        neuron.negativeSpikeMass += Math.abs(signedSpike);
        model.stats.negativeSpikes += Math.abs(signedSpike);
      }
      events.push({ step, kind: 'token_spike', value: signedSpike, label: token, meta: { neuron: neuron.id } });
    }
    if (previous) {
      const association = findOrCreateAssociation(previous.id, neuron.id);
      association.aPre = association.aPre * 0.82 + 0.08;
      association.aPost = association.aPost * 0.82 + 0.08;
      association.w = clamp(association.w + model.config.learningRate * (0.3 + association.aPre + association.aPost) * boundedReward, -1, 1);
      model.stats.sparseAcOps += Math.max(1, Math.abs(signedSpike));
      events.push({ step, kind: 'association', value: association.w, label: `${previous.token} -> ${neuron.token}`, meta: { pre: previous.id, post: neuron.id } });
    }
    previous = neuron;
  }
  events.push({ step, kind: 'reward', value: boundedReward, label: 'language_reward' });
  traceEvents = [...events, ...traceEvents].slice(0, 160);
  saveModel();
  return events;
}

function renderPage(pageId) {
  currentPageId = pageId;
  const page = pages[pageId] ?? pages.habitat;
  const html = page.blocks.map((block, index) => {
    const [kind, ...rest] = block;
    if (kind === 'h1') return `<h1 data-learn-text data-block="${index}">${escapeHtml(rest[0])}</h1>`;
    if (kind === 'h2') return `<h2 data-learn-text data-block="${index}">${escapeHtml(rest[0])}</h2>`;
    if (kind === 'p') return `<p data-learn-text data-block="${index}">${escapeHtml(rest[0])}</p>`;
    if (kind === 'card') return `<div class="browser-card" data-learn-text data-block="${index}">${escapeHtml(rest[0])}</div>`;
    if (kind === 'link') return `<button class="browser-link" data-click-target="${escapeHtml(rest[0])}" data-learn-text data-block="${index}" type="button">${escapeHtml(rest[1])}</button>`;
    if (kind === 'search') {
      return `<div class="browser-search" data-learn-text data-block="${index}"><input value="${escapeHtml(rest[0])}" aria-label="Search text" readonly><button class="browser-button" data-search-action type="button">Search</button></div>`;
    }
    return '';
  }).join('');
  $('browserDocument').innerHTML = html;
  $('browserViewport').scrollTop = 0;
  $('pageInput').value = pageId;
  $('agentStatus').textContent = `page:${pageId}`;
  bindPageClicks();
}

function bindPageClicks() {
  document.querySelectorAll('[data-click-target]').forEach((node) => {
    node.addEventListener('click', () => {
      moveCursorToElement(node, true);
      renderPage(node.dataset.clickTarget);
      learnFromVisible('click', 0.45, `click:${node.textContent.trim()}`);
      pushLog(`manual click ${node.textContent.trim()}`);
      render();
    });
  });
}

function visibleTextBlocks() {
  const viewport = $('browserViewport');
  const viewportRect = viewport.getBoundingClientRect();
  return Array.from($('browserDocument').querySelectorAll('[data-learn-text]')).filter((node) => {
    const rect = node.getBoundingClientRect();
    return rect.bottom > viewportRect.top + 20 && rect.top < viewportRect.bottom - 20;
  });
}

function getVisibleText() {
  return visibleTextBlocks().map((node) => node.textContent.trim()).filter(Boolean).join('\n');
}

function noveltyScore(text) {
  const tokens = tokenize(text);
  if (tokens.length === 0) return 0;
  let novel = 0;
  for (const token of tokens) {
    if (!model.vocabulary.some((neuron) => neuron.token === token)) novel += 1;
  }
  return novel / tokens.length;
}

function learnFromVisible(operation, baseReward, source) {
  const text = getVisibleText();
  const novelty = noveltyScore(text);
  const reward = clamp(baseReward + novelty * 0.65 - (text.length === 0 ? 0.8 : 0), -1, 1);
  $('textInput').value = text;
  $('operationInput').value = operation;
  $('rewardInput').value = reward.toFixed(2);
  observeLanguage({ operation, source: `${currentPageId}:${source}`, text, reward });
  pushLog(`${operation} reward=${reward.toFixed(2)} text=${text.length}`);
}

function chooseAction() {
  const viewport = $('browserViewport');
  const policy = $('agentPolicyInput').value;
  const visible = visibleTextBlocks();
  const links = visible.filter((node) => node.matches('[data-click-target]'));
  const atBottom = viewport.scrollTop + viewport.clientHeight >= viewport.scrollHeight - 8;
  const text = getVisibleText();
  const novelty = noveltyScore(text);

  if (policy === 'link' && links.length > 0) return { type: 'click', target: links[0] };
  if (policy === 'depth' && !atBottom) return { type: 'scroll' };
  if (novelty > 0.22 && agent.lastAction !== 'read') return { type: 'read' };
  if (links.length > 0 && Math.random() < 0.32) return { type: 'click', target: links[Math.floor(Math.random() * links.length)] };
  if (!atBottom) return { type: 'scroll' };
  return { type: 'navigate', target: pages[currentPageId].links[Math.floor(Math.random() * pages[currentPageId].links.length)] };
}

function agentStep() {
  const action = chooseAction();
  agent.lastAction = action.type;
  if (action.type === 'read') {
    const first = visibleTextBlocks()[0];
    if (first) moveCursorToElement(first, false);
    learnFromVisible('read', 0.25, 'visible');
  } else if (action.type === 'scroll') {
    const viewport = $('browserViewport');
    viewport.scrollBy({ top: Math.round(viewport.clientHeight * 0.58), behavior: 'smooth' });
    moveCursor(34 + Math.random() * 120, viewport.clientHeight - 40, false);
    setTimeout(() => {
      learnFromVisible('scroll', 0.16, `scroll:${Math.round(viewport.scrollTop)}`);
      render();
    }, 240);
  } else if (action.type === 'click') {
    moveCursorToElement(action.target, true);
    const target = action.target.dataset.clickTarget;
    setTimeout(() => {
      renderPage(target);
      learnFromVisible('click', 0.42, `click:${action.target.textContent.trim()}`);
      render();
    }, 260);
  } else if (action.type === 'navigate') {
    renderPage(action.target);
    moveCursor(44, 44, false);
    learnFromVisible('navigate', 0.22, `navigate:${action.target}`);
  }
  render();
}

function startAgent() {
  stopAgent();
  $('agentStatus').textContent = 'running';
  const loop = () => {
    agentStep();
    autoTimer = window.setTimeout(loop, Number($('agentSpeedInput').value));
  };
  loop();
}

function stopAgent() {
  if (autoTimer) window.clearTimeout(autoTimer);
  autoTimer = null;
  $('agentStatus').textContent = `idle page:${currentPageId}`;
}

function moveCursorToElement(node, clicking) {
  const viewportRect = $('browserViewport').getBoundingClientRect();
  const rect = node.getBoundingClientRect();
  moveCursor(rect.left - viewportRect.left + Math.min(32, rect.width / 2), rect.top - viewportRect.top + Math.min(22, rect.height / 2), clicking);
}

function moveCursor(x, y, clicking) {
  agent.cursorX = clamp(x, 12, $('browserViewport').clientWidth - 18);
  agent.cursorY = clamp(y, 12, $('browserViewport').clientHeight - 18);
  const cursor = $('agentCursor');
  cursor.style.transform = `translate(${agent.cursorX}px, ${agent.cursorY}px) rotate(-18deg)`;
  cursor.classList.toggle('clicking', Boolean(clicking));
  if (clicking) window.setTimeout(() => cursor.classList.remove('clicking'), 420);
}

function encodeModelFile(payload) {
  const json = JSON.stringify({ kind: 'eden14-snn-life-model', version: 2, container: EDEN_SNN_MAGIC, exportedAt: new Date().toISOString(), modelCount: payload.models.length, models: payload.models });
  const encoded = new TextEncoder().encode(json);
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  const header = new ArrayBuffer(magic.length + 4);
  const headerBytes = new Uint8Array(header);
  headerBytes.set(magic, 0);
  new DataView(header).setUint32(magic.length, encoded.byteLength, true);
  return new Blob([header, encoded], { type: 'application/x-edensnn' });
}

async function decodeModelFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const magic = new TextEncoder().encode(EDEN_SNN_MAGIC);
  if (bytes.length < magic.length + 4) throw new Error('File is too small.');
  for (let i = 0; i < magic.length; i += 1) if (bytes[i] !== magic[i]) throw new Error('Invalid .edensnn header.');
  const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(magic.length, true);
  const start = magic.length + 4;
  const parsed = JSON.parse(new TextDecoder().decode(bytes.slice(start, start + length)));
  const languageModel = parsed.models?.find((entry) => entry.modelKind === 'browser-language' || entry.snapshot?.domain === 'browser-language');
  if (!languageModel?.snapshot) throw new Error('No browser-language model found.');
  return languageModel.snapshot;
}

function exportModel() {
  const blob = encodeModelFile({ models: [{ creatureId: 'autonomous-browser-language-snn', modelKind: 'browser-language', snapshot: model }] });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `autonomous-browser-language-snn-${new Date().toISOString().replace(/[:.]/g, '-')}.edensnn`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  pushLog('exported .edensnn model');
  render();
}

function render() {
  $('vocabMetric').textContent = model.vocabulary.length;
  $('assocMetric').textContent = model.associations.length;
  $('obsMetric').textContent = model.stats.observations;
  $('posMetric').textContent = format(model.stats.positiveSpikes);
  $('negMetric').textContent = format(model.stats.negativeSpikes);
  $('energyMetric').textContent = model.stats.sparseAcOps > 0 ? `${format(model.stats.denseMacOps / model.stats.sparseAcOps)}x` : '-';
  $('modelStatus').textContent = `${model.stats.observations} observations`;
  $('pageInput').value = currentPageId;

  const topTokens = [...model.vocabulary].sort((a, b) => b.count - a.count).slice(0, 40);
  $('tokenList').innerHTML = topTokens.length
    ? topTokens.map((neuron) => `<span class="token ${neuron.negativeSpikeMass > neuron.positiveSpikeMass ? 'negative' : ''}">${escapeHtml(neuron.token)} ${neuron.count}</span>`).join('')
    : '<span class="token">no tokens</span>';

  const strongest = [...model.associations].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 32);
  $('associationList').innerHTML = strongest.length
    ? strongest.map((association) => {
      const pre = model.vocabulary.find((neuron) => neuron.id === association.pre)?.token ?? association.pre;
      const post = model.vocabulary.find((neuron) => neuron.id === association.post)?.token ?? association.post;
      return `<div class="row"><span>${escapeHtml(String(pre))} -> ${escapeHtml(String(post))}</span><span class="${association.w >= 0 ? 'positive' : 'negative-text'}">${format(association.w)}</span></div>`;
    }).join('')
    : '<div>no associations</div>';

  $('traceList').innerHTML = traceEvents.length
    ? traceEvents.slice(0, 40).map((event) => `<div class="row"><span>${event.kind} ${escapeHtml(event.label)}</span><span>${format(event.value)}</span></div>`).join('')
    : '<div>no trace events</div>';

  $('operationLog').innerHTML = operationLog.length
    ? operationLog.map((line) => `<div>${escapeHtml(line)}</div>`).join('')
    : '<div>no operations</div>';
}

function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

$('startAgentButton').addEventListener('click', startAgent);
$('stopAgentButton').addEventListener('click', stopAgent);
$('stepAgentButton').addEventListener('click', agentStep);
$('resetButton').addEventListener('click', () => {
  stopAgent();
  model = createModel();
  traceEvents = [];
  operationLog = [];
  saveModel();
  renderPage('habitat');
  render();
});
$('exportModelButton').addEventListener('click', exportModel);
$('importModelButton').addEventListener('click', () => $('modelFileInput').click());
$('modelFileInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    model = await decodeModelFile(file);
    saveModel();
    pushLog(`imported ${file.name}`);
  } catch (error) {
    pushLog(`import failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  render();
});

renderPage('habitat');
learnFromVisible('observe', 0.2, 'initial');
render();
