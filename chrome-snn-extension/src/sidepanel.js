import {
  createModel,
  DEFAULT_LEARNING_SETTINGS,
  DEFAULT_MODEL_ID,
  decodeModelPayloadBytes,
  encodeModelFile,
  EVENT_STORAGE_KEY,
  eventStorageKey,
  importedModelFromPayload,
  MODEL_INDEX_STORAGE_KEY,
  MODEL_STORAGE_KEY,
  modelStorageKey,
  SETTINGS_STORAGE_KEY,
  TAB_STATE_STORAGE_KEY
} from './snnCore.js';

const els = {
  importButton: document.getElementById('importButton'),
  importFileInput: document.getElementById('importFileInput'),
  importStatus: document.getElementById('importStatus'),
  exportButton: document.getElementById('exportButton'),
  resetButton: document.getElementById('resetButton'),
  modelSelect: document.getElementById('modelSelect'),
  newModelButton: document.getElementById('newModelButton'),
  learningToggleButton: document.getElementById('learningToggleButton'),
  learningStatus: document.getElementById('learningStatus'),
  sleepButton: document.getElementById('sleepButton'),
  sleepStatus: document.getElementById('sleepStatus'),
  autonomousToggleButton: document.getElementById('autonomousToggleButton'),
  autonomousStatus: document.getElementById('autonomousStatus'),
  inputCapture: document.getElementById('inputCapture'),
  sensitiveTextMode: document.getElementById('sensitiveTextMode'),
  moralLearningMode: document.getElementById('moralLearningMode'),
  instinctLearningMode: document.getElementById('instinctLearningMode'),
  crisisSensitivity: document.getElementById('crisisSensitivity'),
  maxVocabulary: document.getElementById('maxVocabulary'),
  autonomousIntervalSeconds: document.getElementById('autonomousIntervalSeconds'),
  autonomousMaxNavigations: document.getElementById('autonomousMaxNavigations'),
  computeBackend: document.getElementById('computeBackend'),
  performanceBudgetPercent: document.getElementById('performanceBudgetPercent'),
  performanceLimitValue: document.getElementById('performanceLimitValue'),
  privacySensitivity: document.getElementById('privacySensitivity'),
  sensitiveValueReward: document.getElementById('sensitiveValueReward'),
  metrics: document.getElementById('metrics'),
  learningTabList: document.getElementById('learningTabList'),
  observationList: document.getElementById('observationList'),
  tokenList: document.getElementById('tokenList'),
  associationList: document.getElementById('associationList'),
  crossModalList: document.getElementById('crossModalList'),
  traceList: document.getElementById('traceList')
};

const PT_CONVERTER_URL = 'http://127.0.0.1:8765/api/convert-pt';
const SIDEPANEL_REFRESH_INTERVAL_MS = 10000;

let currentModel = createModel();
let currentEvents = [];
let learningSettings = { ...DEFAULT_LEARNING_SETTINGS };
let modelIndex = { selectedModelId: DEFAULT_MODEL_ID, models: [{ id: DEFAULT_MODEL_ID, name: 'Default Browser SNN' }] };
let tabStates = {};
let activeTab = null;
let activeTabState = {
  learningEnabled: DEFAULT_LEARNING_SETTINGS.learningEnabled !== false,
  autonomousExploreEnabled: DEFAULT_LEARNING_SETTINGS.autonomousExploreEnabled === true,
  modelId: DEFAULT_MODEL_ID,
};

const escapeHtml = (value) =>
  String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const formatNumber = (value, digits = 0) =>
  Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '-';

const formatBytes = (bytes) => {
  const value = Number(bytes || 0);
  if (value < 1024) return `${formatNumber(value)} B`;
  if (value < 1024 * 1024) return `${formatNumber(value / 1024, 1)} KB`;
  return `${formatNumber(value / 1024 / 1024, 2)} MB`;
};

const shortUrl = (url) => {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname === '/' ? '' : parsed.pathname}`;
  } catch {
    return url || 'unknown page';
  }
};

const energyRatio = (stats) => {
  const sparse = Math.max(1, stats.sparseAcOps || 0);
  const dense = Math.max(sparse, stats.denseMacOps || sparse);
  return `${formatNumber(dense / sparse, 1)}x sparse`;
};

const normalizePerformanceLimit = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_LEARNING_SETTINGS.performanceBudgetPercent;
  return Math.min(100, Math.max(10, Math.round(numeric)));
};

const renderPerformanceLimitValue = () => {
  const value = normalizePerformanceLimit(els.performanceBudgetPercent.value || learningSettings.performanceBudgetPercent);
  els.performanceLimitValue.textContent = `${value}%`;
};

const activeModelId = () => activeTabState.modelId || modelIndex.selectedModelId || DEFAULT_MODEL_ID;

const getActiveTab = async () => {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
};

const getDefaultTabState = () => ({
  learningEnabled: DEFAULT_LEARNING_SETTINGS.learningEnabled !== false,
  autonomousExploreEnabled: DEFAULT_LEARNING_SETTINGS.autonomousExploreEnabled === true,
  modelId: modelIndex.selectedModelId || DEFAULT_MODEL_ID,
  title: activeTab?.title || '',
  url: activeTab?.url || '',
});

async function refresh() {
  activeTab = await getActiveTab();
  const stored = await chrome.storage.local.get([MODEL_INDEX_STORAGE_KEY, TAB_STATE_STORAGE_KEY, SETTINGS_STORAGE_KEY]);
  modelIndex = stored[MODEL_INDEX_STORAGE_KEY] || modelIndex;
  if (!Array.isArray(modelIndex.models) || modelIndex.models.length === 0) {
    modelIndex = { selectedModelId: DEFAULT_MODEL_ID, models: [{ id: DEFAULT_MODEL_ID, name: 'Default Browser SNN' }] };
  }
  tabStates = stored[TAB_STATE_STORAGE_KEY] || {};
  activeTabState = {
    ...getDefaultTabState(),
    ...(activeTab?.id !== undefined ? tabStates[String(activeTab.id)] || {} : {}),
  };
  const modelId = activeModelId();
  const modelKey = modelStorageKey(modelId);
  const eventKey = eventStorageKey(modelId);
  const modelStored = await chrome.storage.local.get([modelKey, eventKey]);
  currentModel = modelStored[modelKey] || createModel();
  currentEvents = modelStored[eventKey] || [];
  learningSettings = { ...DEFAULT_LEARNING_SETTINGS, ...(stored[SETTINGS_STORAGE_KEY] || {}) };
  render();
}

function render() {
  renderMetrics();
  renderObservations();
  renderTokens();
  renderAssociations();
  renderCrossModalRelations();
  renderTrace();
  renderSettings();
  renderModels();
  renderLearningTabs();
}

function renderSettings() {
  const enabled = activeTabState.learningEnabled !== false;
  els.learningToggleButton.textContent = enabled ? 'STOP Learning' : 'START Learning';
  els.learningToggleButton.classList.toggle('stopped', !enabled);
  els.learningStatus.textContent = enabled ? 'Tab Learning: ON' : 'Tab Learning: OFF';
  els.learningStatus.classList.toggle('stopped', !enabled);
  const autonomousEnabled = activeTabState.autonomousExploreEnabled === true;
  els.autonomousToggleButton.textContent = autonomousEnabled ? 'STOP Explore' : 'START Explore';
  els.autonomousToggleButton.classList.toggle('stopped', !autonomousEnabled);
  els.autonomousStatus.textContent = autonomousEnabled ? 'Explore: ON' : 'Explore: OFF';
  els.autonomousStatus.classList.toggle('stopped', !autonomousEnabled);
  els.inputCapture.value = learningSettings.inputCapture;
  els.sensitiveTextMode.value = learningSettings.sensitiveTextMode;
  els.moralLearningMode.value = learningSettings.moralLearningMode;
  els.instinctLearningMode.value = learningSettings.instinctLearningMode || 'shape';
  els.crisisSensitivity.value = learningSettings.crisisSensitivity ?? 0.65;
  els.maxVocabulary.value = learningSettings.maxVocabulary ?? 0;
  els.autonomousIntervalSeconds.value = learningSettings.autonomousIntervalSeconds ?? 18;
  els.autonomousMaxNavigations.value = learningSettings.autonomousMaxNavigations ?? 24;
  els.computeBackend.value = learningSettings.computeBackend || 'auto';
  els.performanceBudgetPercent.value = normalizePerformanceLimit(learningSettings.performanceBudgetPercent);
  renderPerformanceLimitValue();
  els.privacySensitivity.value = learningSettings.privacySensitivity;
  els.sensitiveValueReward.value = learningSettings.sensitiveValueReward;
}

function renderModels() {
  const currentId = activeModelId();
  els.modelSelect.innerHTML = modelIndex.models
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name || model.id)}</option>`)
    .join('');
  els.modelSelect.value = currentId;
}

async function getTabsById() {
  try {
    const tabs = await chrome.tabs.query({});
    return new Map(tabs.map((tab) => [String(tab.id), tab]));
  } catch {
    return new Map();
  }
}

async function renderLearningTabs() {
  const tabsById = await getTabsById();
  const rows = Object.entries(tabStates)
    .filter(([, state]) => state.learningEnabled !== false || state.autonomousExploreEnabled === true)
    .map(([tabId, state]) => {
      const tab = tabsById.get(tabId);
      const model = modelIndex.models.find((item) => item.id === state.modelId);
      return {
        tabId,
        title: tab?.title || state.title || 'Untitled tab',
        url: tab?.url || state.url || '',
        learning: state.learningEnabled !== false,
        explore: state.autonomousExploreEnabled === true,
        modelName: model?.name || state.modelId || DEFAULT_MODEL_ID,
      };
    });
  if (rows.length === 0) {
    els.learningTabList.innerHTML = '<li class="empty">No tabs are learning or exploring.</li>';
    return;
  }
  els.learningTabList.innerHTML = rows
    .map((row) => `
      <li>
        <strong>${escapeHtml(row.title)}</strong>
        <span>${escapeHtml(row.learning ? 'Learning' : 'Paused')} / ${escapeHtml(row.explore ? 'Explore ON' : 'Explore OFF')} / ${escapeHtml(row.modelName)}</span>
        <small>${escapeHtml(shortUrl(row.url))}</small>
      </li>
    `)
    .join('');
}

async function saveSettings(patch) {
  learningSettings = { ...learningSettings, ...patch };
  await chrome.storage.local.set({ [SETTINGS_STORAGE_KEY]: learningSettings });
  renderSettings();
}

async function saveActiveTabState(patch) {
  if (activeTab?.id === undefined) return;
  const result = await chrome.runtime.sendMessage({
    type: 'BROWSER_SNN_SET_TAB_STATE',
    tabId: activeTab.id,
    patch: {
      title: activeTab.title || activeTabState.title,
      url: activeTab.url || activeTabState.url,
      ...patch,
    },
  });
  if (result?.ok) {
    activeTabState = { ...activeTabState, ...result.tabState };
    tabStates[String(activeTab.id)] = activeTabState;
  }
  await refresh();
}

function renderMetrics() {
  const stats = currentModel.stats || {};
  const mods = currentModel.neuromodulators || {};
  const latestObservation = currentModel.observations?.[0] || {};
  const modelBytes = stats.modelBytes || new TextEncoder().encode(JSON.stringify(currentModel)).byteLength;
  const metricRows = [
    ['Model size', formatBytes(modelBytes)],
    ['CPU load', `${formatNumber((stats.cpuLoadEstimate || 0) * 100, 0)}%`],
    ['Step time', `${formatNumber(stats.lastStepMs || 0, 2)} ms`],
    ['Avg step', `${formatNumber(stats.avgStepMs || 0, 2)} ms`],
    ['Backend', stats.computeBackend || 'cpu'],
    ['WebGPU', stats.webgpuAvailable ? 'available' : 'unavailable'],
    ['Perf limit', `${normalizePerformanceLimit(learningSettings.performanceBudgetPercent)}%`],
    ['Learn throttle', stats.learningThrottleMs ? `${formatNumber(stats.learningThrottleMs)} ms` : 'off'],
    ['GPU token cap', stats.gpuTokenLimit || '-'],
    ['GPU throttle', stats.gpuThrottleMs ? `${formatNumber(stats.gpuThrottleMs)} ms` : 'off'],
    ['GPU throttled', stats.gpuThrottledSteps || 0],
    ['Vocabulary', Object.keys(currentModel.vocabulary || {}).length],
    ['New tokens', latestObservation.newTokenCount || 0],
    ['Max vocab', currentModel.config?.maxVocabulary ? currentModel.config.maxVocabulary : 'unlimited'],
    ['Associations', (currentModel.associations || []).length],
    ['Cross-modal', (currentModel.crossModalRelations || []).length],
    ['Cross-modal updates', stats.crossModalUpdates || 0],
    ['Observations', stats.observations || 0],
    ['User events', stats.userEvents || 0],
    ['Autonomous events', stats.autonomousEvents || 0],
    ['Media events', stats.mediaEvents || 0],
    ['Visual events', stats.visualEvents || 0],
    ['Dopamine', mods.dopamine || 0],
    ['ACh', mods.acetylcholine || 0],
    ['Fatigue', mods.fatigue || 0],
    ['RPE', stats.rewardPredictionError || 0],
    ['Analog drive', stats.analogDrive || 0],
    ['Analog decisions', stats.analogDecisions || 0],
    ['Synaptic scaled', stats.synapticScaled || 0],
    ['Delayed rewards', stats.delayedRewards || 0],
    ['Credit assigns', stats.delayedCreditAssignments || 0],
    ['Sleep cycles', stats.sleepCycles || 0],
    ['Sleep replayed', stats.sleepReplayedSynapses || 0],
    ['Sleep scaled', stats.sleepScaledSynapses || 0],
    ['Sleep inhibited', stats.sleepInhibitedSynapses || 0],
    ['Moral events', stats.moralEvents || 0],
    ['Moral penalties', stats.moralPenalties || 0],
    ['Instinct events', stats.instinctEvents || 0],
    ['Instinct avoidances', stats.instinctAvoidances || 0],
    ['Instinct delayed', stats.instinctDelayedAvoidance || 0],
    ['Discovery rewards', stats.discoveryRewards || 0],
    ['Novel words rewarded', stats.novelWordsRewarded || 0],
    ['New page discoveries', stats.newPageDiscoveries || 0],
    ['Lexical links', stats.lexicalLinks || 0],
    ['Redactions', stats.privacyRedactions || 0],
    ['Consolidations', stats.consolidationCycles || 0],
    ['Stable synapses', stats.stabilizedSynapses || 0],
    ['Pages', stats.pages || 0],
    ['Energy est.', energyRatio(stats)]
  ];

  els.metrics.innerHTML = metricRows
    .map(([label, value]) => `
      <div class="metric">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(typeof value === 'number' ? formatNumber(value, 1) : value)}</strong>
      </div>
    `)
    .join('');
}

function renderObservations() {
  const observations = [...(currentModel.observations || [])].slice(-12).reverse();
  if (observations.length === 0) {
    els.observationList.innerHTML = '<li class="empty">No browser observations yet.</li>';
    return;
  }

  els.observationList.innerHTML = observations
    .map((item) => `
      <li>
        <strong>${escapeHtml(item.eventType || 'event')}</strong>
        <span>${escapeHtml(item.title || shortUrl(item.url))}</span>
        <small>${escapeHtml(item.media ? `${item.media.kind || 'media'} ${formatNumber(item.media.progress || 0, 2)}` : shortUrl(item.url))} / tokens ${escapeHtml(formatNumber(item.tokenCount || 0))} / new ${escapeHtml(formatNumber(item.newTokenCount || 0))} / reward ${escapeHtml(formatNumber(item.reward || 0, 3))}${item.discovery?.novelWordCount ? ` / discovery +${escapeHtml(formatNumber(item.discovery.novelWordCount))}w` : ''}</small>
      </li>
    `)
    .join('');
}

function renderTokens() {
  const tokens = Object.values(currentModel.vocabulary || {})
    .sort((a, b) => (b.count || 0) - (a.count || 0))
    .slice(0, 40);

  if (tokens.length === 0) {
    els.tokenList.innerHTML = '<li class="empty">Visible page language will appear here.</li>';
    return;
  }

  els.tokenList.innerHTML = tokens
    .map((token) => {
      const positive = token.positiveSpikeMass || 0;
      const negative = token.negativeSpikeMass || 0;
      const cls = negative > positive ? 'negative' : 'positive';
      return `
        <span class="token ${cls}">
          <span>${escapeHtml(token.token)}</span>
          <small>${escapeHtml(formatNumber(token.count || 0))} / v=${escapeHtml(formatNumber(token.v || 0, 2))} / stable=${escapeHtml(formatNumber(token.stability || 0, 2))}</small>
        </span>
      `;
    })
    .join('');
}

function renderAssociations() {
  const tokenById = Object.fromEntries(
    Object.values(currentModel.vocabulary || {}).map((token) => [token.id, token.token])
  );
  const associations = [...(currentModel.associations || [])]
    .sort((a, b) => Math.abs(b.w || 0) - Math.abs(a.w || 0))
    .slice(0, 28);

  if (associations.length === 0) {
    els.associationList.innerHTML = '<li class="empty">Token links grow after repeated behavior.</li>';
    return;
  }

  els.associationList.innerHTML = associations
    .map((edge) => {
      const cls = edge.w < 0 ? 'negative' : 'positive';
      return `
        <li class="${cls}">
          <strong>${escapeHtml(tokenById[edge.pre] || edge.pre)} -> ${escapeHtml(tokenById[edge.post] || edge.post)}</strong>
          <span>w=${escapeHtml(formatNumber(edge.w || 0, 3))} / D1=${escapeHtml(formatNumber(edge.d1Go || 0, 2))} / D2=${escapeHtml(formatNumber(edge.d2NoGo || 0, 2))} / RPE=${escapeHtml(formatNumber(edge.rewardPrediction || 0, 2))} / stable=${escapeHtml(formatNumber(edge.stability || 0, 2))}</span>
        </li>
      `;
    })
    .join('');
}

function renderCrossModalRelations() {
  const tokenById = Object.fromEntries(
    Object.values(currentModel.vocabulary || {}).map((token) => [token.id, token.token])
  );
  const relations = [...(currentModel.crossModalRelations || [])]
    .sort((a, b) => Math.abs(b.w || 0) + (b.coactivity || 0) * 0.08 - (Math.abs(a.w || 0) + (a.coactivity || 0) * 0.08))
    .slice(0, 24);

  if (relations.length === 0) {
    els.crossModalList.innerHTML = '<li class="empty">Image, audio, video, and text bindings will appear here.</li>';
    return;
  }

  els.crossModalList.innerHTML = relations
    .map((relation) => {
      const cls = relation.w < 0 ? 'negative' : 'positive';
      const modalities = relation.modalities || [];
      return `
        <li class="${cls}">
          <strong>${escapeHtml(tokenById[relation.a] || relation.a)} <-> ${escapeHtml(tokenById[relation.b] || relation.b)}</strong>
          <span>${escapeHtml(modalities.join(' <-> ') || 'modalities')} / w=${escapeHtml(formatNumber(relation.w || 0, 3))} / co=${escapeHtml(formatNumber(relation.coactivity || 0, 2))} / stable=${escapeHtml(formatNumber(relation.stability || 0, 2))}</span>
        </li>
      `;
    })
    .join('');
}

function renderTrace() {
  const events = [...currentEvents].slice(-18).reverse();
  if (events.length === 0) {
    els.traceList.innerHTML = '<li class="empty">User movement traces are stored locally.</li>';
    return;
  }

  els.traceList.innerHTML = events
    .map((event) => `
      <li>
        <strong>${escapeHtml(event.label || event.kind || 'event')}</strong>
        <span>${escapeHtml(event.kind || 'trace')} / ${escapeHtml(formatNumber(event.value || 0, 3))}</span>
        <small>${escapeHtml(shortUrl(event.meta?.url))}</small>
      </li>
    `)
    .join('');
}

async function resetModel() {
  currentModel = createModel();
  currentEvents = [];
  const modelId = activeModelId();
  await chrome.storage.local.set({
    [modelStorageKey(modelId)]: currentModel,
    [eventStorageKey(modelId)]: currentEvents
  });
  render();
}

function exportModel() {
  const blob = encodeModelFile(currentModel);
  const url = URL.createObjectURL(blob);
  const timestamp = new Date().toISOString().replaceAll(':', '-').replace(/\.\d+Z$/, 'Z');
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `chrome-browser-language-snn-${timestamp}.edensnn`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function payloadFromPt(file) {
  let response;
  try {
    response = await fetch(PT_CONVERTER_URL, {
      method: 'POST',
      headers: {
        'content-type': 'application/octet-stream',
        'x-filename': encodeURIComponent(file.name),
      },
      body: await file.arrayBuffer(),
    });
  } catch (error) {
    throw new Error(`Raw .pt import needs the local converter. Run: python scripts/serve_snn_chat_lab.py (${String(error?.message || error)})`);
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error ? `.pt conversion failed: ${payload.error}` : '.pt conversion failed.');
  }
  return payload;
}

async function modelFromImportFile(file) {
  const name = file.name.toLowerCase();
  const payload = name.endsWith('.pt')
    ? await payloadFromPt(file)
    : decodeModelPayloadBytes(await file.arrayBuffer());
  return importedModelFromPayload(payload, { sourceName: file.name });
}

async function importModelFile(file) {
  els.importButton.disabled = true;
  els.importStatus.textContent = `Importing ${file.name}...`;
  try {
    const importedModel = await modelFromImportFile(file);
    const hasCurrentModel = (currentModel.vocabulary || []).length > 0 || (currentModel.associations || []).length > 0;
    const mode = hasCurrentModel && window.confirm('現在選択中のモデルへ統合しますか？\nOK: 統合して追加学習 / Cancel: 新規モデルとして取り込み')
      ? 'merge'
      : 'new';
    const result = await chrome.runtime.sendMessage({
      type: 'BROWSER_SNN_IMPORT_MODEL',
      mode,
      modelId: activeModelId(),
      sourceName: file.name,
      model: importedModel,
      name: file.name.replace(/\.(edensnn|chat\.json|json|pt)$/i, ''),
    });
    if (!result?.ok) throw new Error(result?.error || 'import failed');
    if (result.mode === 'new') {
      modelIndex = result.modelIndex || modelIndex;
      await saveActiveTabState({ modelId: result.modelId });
    } else {
      await refresh();
    }
    els.importStatus.textContent = `Import: ${result.mode === 'merge' ? 'merged into current model' : 'loaded as active model'}`;
  } catch (error) {
    els.importStatus.textContent = `Import failed: ${String(error?.message || error)}`;
  } finally {
    els.importButton.disabled = false;
  }
}

async function sleepConsolidate() {
  els.sleepButton.disabled = true;
  els.sleepStatus.textContent = 'Manual sleep: running...';
  try {
    const result = await chrome.runtime.sendMessage({ type: 'BROWSER_SNN_SLEEP', modelId: activeModelId(), options: { cycles: 3 } });
    if (!result?.ok) throw new Error(result?.error || 'sleep failed');
    const summary = result.summary || {};
    els.sleepStatus.textContent = `Manual sleep: replay ${formatNumber(summary.replayed || 0)} / scale ${formatNumber(summary.scaled || 0)} / inhibit ${formatNumber(summary.inhibited || 0)}`;
    await refresh();
  } catch (error) {
    els.sleepStatus.textContent = `Manual sleep failed: ${String(error?.message || error)}`;
  } finally {
    els.sleepButton.disabled = false;
  }
}

async function createNewModel() {
  const name = window.prompt('Model name', `Browser SNN ${modelIndex.models.length + 1}`);
  if (!name) return;
  const result = await chrome.runtime.sendMessage({ type: 'BROWSER_SNN_CREATE_MODEL', name });
  if (!result?.ok) return;
  modelIndex = result.modelIndex;
  await saveActiveTabState({ modelId: result.modelId });
}

els.importButton.addEventListener('click', () => els.importFileInput.click());
els.importFileInput.addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  await importModelFile(file);
  event.target.value = '';
});
els.exportButton.addEventListener('click', exportModel);
els.resetButton.addEventListener('click', resetModel);
els.newModelButton.addEventListener('click', createNewModel);
els.modelSelect.addEventListener('change', () => saveActiveTabState({ modelId: els.modelSelect.value }));
els.learningToggleButton.addEventListener('click', () => saveActiveTabState({ learningEnabled: activeTabState.learningEnabled === false }));
els.autonomousToggleButton.addEventListener('click', () => saveActiveTabState({ autonomousExploreEnabled: activeTabState.autonomousExploreEnabled !== true }));
els.sleepButton.addEventListener('click', sleepConsolidate);
els.inputCapture.addEventListener('change', () => saveSettings({ inputCapture: els.inputCapture.value }));
els.sensitiveTextMode.addEventListener('change', () => saveSettings({ sensitiveTextMode: els.sensitiveTextMode.value }));
els.moralLearningMode.addEventListener('change', () => saveSettings({ moralLearningMode: els.moralLearningMode.value }));
els.instinctLearningMode.addEventListener('change', () => saveSettings({ instinctLearningMode: els.instinctLearningMode.value }));
els.crisisSensitivity.addEventListener('input', () => saveSettings({ crisisSensitivity: Number(els.crisisSensitivity.value) }));
els.maxVocabulary.addEventListener('change', () => saveSettings({ maxVocabulary: Math.max(0, Number(els.maxVocabulary.value || 0)) }));
els.autonomousIntervalSeconds.addEventListener('change', () => saveSettings({ autonomousIntervalSeconds: Math.max(8, Number(els.autonomousIntervalSeconds.value || 18)) }));
els.autonomousMaxNavigations.addEventListener('change', () => saveSettings({ autonomousMaxNavigations: Math.max(0, Number(els.autonomousMaxNavigations.value || 0)) }));
els.computeBackend.addEventListener('change', () => saveSettings({ computeBackend: els.computeBackend.value }));
els.performanceBudgetPercent.addEventListener('input', renderPerformanceLimitValue);
els.performanceBudgetPercent.addEventListener('change', () => saveSettings({ performanceBudgetPercent: normalizePerformanceLimit(els.performanceBudgetPercent.value) }));
els.privacySensitivity.addEventListener('input', () => saveSettings({ privacySensitivity: Number(els.privacySensitivity.value) }));
els.sensitiveValueReward.addEventListener('input', () => saveSettings({ sensitiveValueReward: Number(els.sensitiveValueReward.value) }));

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local') return;
  const modelId = activeModelId();
  if (
    changes[modelStorageKey(modelId)]
    || changes[eventStorageKey(modelId)]
    || changes[MODEL_STORAGE_KEY]
    || changes[EVENT_STORAGE_KEY]
    || changes[SETTINGS_STORAGE_KEY]
    || changes[MODEL_INDEX_STORAGE_KEY]
    || changes[TAB_STATE_STORAGE_KEY]
  ) {
    refresh();
  }
});

refresh();
setInterval(() => {
  if (document.visibilityState === 'visible') refresh();
}, SIDEPANEL_REFRESH_INTERVAL_MS);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') refresh();
});
