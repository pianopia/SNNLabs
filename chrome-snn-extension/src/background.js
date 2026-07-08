import {
  createModel,
  DEFAULT_LEARNING_SETTINGS,
  DEFAULT_MODEL_ID,
  eventStorageKey,
  MODEL_INDEX_STORAGE_KEY,
  modelStorageKey,
  tokenize,
  observeEvent,
  sleepConsolidateModel,
  SETTINGS_STORAGE_KEY,
  TAB_STATE_STORAGE_KEY,
  mergeImportedModel,
  normalizeImportedModel
} from './snnCore.js';
import { computeVoltagesOnGpu, getWebGpuStatus } from './webgpuLearner.js';

const DEFAULT_MODEL_INDEX = {
  selectedModelId: DEFAULT_MODEL_ID,
  models: [{ id: DEFAULT_MODEL_ID, name: 'Default Browser SNN', createdAt: Date.now() }],
};

const performanceState = {
  lastLearnAtByModel: new Map(),
  lastGpuAtByModel: new Map(),
};

const defaultTabState = () => ({
  learningEnabled: DEFAULT_LEARNING_SETTINGS.learningEnabled !== false,
  autonomousExploreEnabled: DEFAULT_LEARNING_SETTINGS.autonomousExploreEnabled === true,
  modelId: DEFAULT_MODEL_ID,
  title: '',
  url: '',
  lastEventAt: 0,
});

const getModelIndex = async () => {
  const stored = await chrome.storage.local.get([MODEL_INDEX_STORAGE_KEY]);
  const index = stored[MODEL_INDEX_STORAGE_KEY] || DEFAULT_MODEL_INDEX;
  const models = Array.isArray(index.models) && index.models.length ? index.models : DEFAULT_MODEL_INDEX.models;
  if (!models.some((model) => model.id === DEFAULT_MODEL_ID)) models.unshift(DEFAULT_MODEL_INDEX.models[0]);
  return {
    selectedModelId: index.selectedModelId || DEFAULT_MODEL_ID,
    models,
  };
};

const setModelIndex = async (index) => {
  await chrome.storage.local.set({ [MODEL_INDEX_STORAGE_KEY]: index });
};

const getTabStates = async () => {
  const stored = await chrome.storage.local.get([TAB_STATE_STORAGE_KEY]);
  return stored[TAB_STATE_STORAGE_KEY] || {};
};

const setTabStates = async (states) => {
  await chrome.storage.local.set({ [TAB_STATE_STORAGE_KEY]: states });
};

const getTabState = async (tabId) => {
  const states = await getTabStates();
  return { ...defaultTabState(), ...(states[String(tabId)] || {}) };
};

const patchTabState = async (tabId, patch) => {
  const states = await getTabStates();
  const key = String(tabId);
  states[key] = { ...defaultTabState(), ...(states[key] || {}), ...patch, updatedAt: Date.now() };
  await setTabStates(states);
  return states[key];
};

const notifyTabState = async (tabId, tabState) => {
  try {
    await chrome.tabs.sendMessage(tabId, { type: 'BROWSER_SNN_TAB_STATE', tabState });
  } catch {
    // The content script may not be present on chrome:// pages or during navigation.
  }
};

const getStoredModel = async (modelId = DEFAULT_MODEL_ID) => {
  const key = modelStorageKey(modelId);
  const stored = await chrome.storage.local.get([key]);
  return stored[key] ?? createModel();
};

const compactModelForStorage = (model) => {
  const compacted = structuredClone(model);
  compacted.observations = (compacted.observations || []).slice(0, 80).map((item) => ({
    ...item,
    visibleText: String(item.visibleText || '').slice(0, 420),
    elementText: String(item.elementText || '').slice(0, 220),
    media: item.media ? {
      ...item.media,
      cueText: String(item.media.cueText || '').slice(0, 300),
      transcriptText: String(item.media.transcriptText || '').slice(0, 420),
      src: String(item.media.src || '').slice(0, 220),
    } : undefined,
  }));
  const maxAssociations = 8500;
  if ((compacted.associations || []).length > maxAssociations) {
    compacted.associations = [...compacted.associations]
      .sort((a, b) =>
        (Math.abs(b.w || 0) + (b.stability || 0) * 0.8 + (b.replayCount || 0) * 0.002)
        - (Math.abs(a.w || 0) + (a.stability || 0) * 0.8 + (a.replayCount || 0) * 0.002)
      )
      .slice(0, maxAssociations);
  }
  const maxCrossModalRelations = 3000;
  if ((compacted.crossModalRelations || []).length > maxCrossModalRelations) {
    compacted.crossModalRelations = [...compacted.crossModalRelations]
      .sort((a, b) =>
        (Math.abs(b.w || 0) + (b.stability || 0) + (b.coactivity || 0) * 0.08)
        - (Math.abs(a.w || 0) + (a.stability || 0) + (a.coactivity || 0) * 0.08)
      )
      .slice(0, maxCrossModalRelations);
  }
  compacted.eligibility ??= {};
  compacted.eligibility.synapses = (compacted.eligibility.synapses || []).slice(-160);
  compacted.stats ??= {};
  compacted.stats.storageCompactions = (compacted.stats.storageCompactions || 0) + 1;
  compacted.stats.modelBytes = new TextEncoder().encode(JSON.stringify(compacted)).byteLength;
  return compacted;
};

const setStoredModel = async (model, modelId = DEFAULT_MODEL_ID) => {
  const key = modelStorageKey(modelId);
  try {
    await chrome.storage.local.set({ [key]: model });
  } catch (error) {
    const compacted = compactModelForStorage(model);
    await chrome.storage.local.set({ [key]: compacted });
  }
};

const getLearningSettings = async () => {
  const stored = await chrome.storage.local.get([SETTINGS_STORAGE_KEY]);
  const settings = { ...DEFAULT_LEARNING_SETTINGS, ...(stored[SETTINGS_STORAGE_KEY] || {}) };
  if (settings.maxVocabulary === undefined || settings.maxVocabulary === null) settings.maxVocabulary = 0;
  settings.performanceBudgetPercent = clampPerformanceBudget(settings.performanceBudgetPercent);
  return settings;
};

const modelVocabulary = (model) => Array.isArray(model.vocabulary)
  ? model.vocabulary
  : Object.values(model.vocabulary || {});

const clampPerformanceBudget = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_LEARNING_SETTINGS.performanceBudgetPercent;
  return Math.min(100, Math.max(10, Math.round(numeric)));
};

const learningThrottleMsForBudget = (settings) => {
  const budget = clampPerformanceBudget(settings.performanceBudgetPercent);
  return Math.max(0, Math.round((100 - budget) * 8));
};

const gpuThrottleMsForBudget = (settings) => {
  const budget = clampPerformanceBudget(settings.performanceBudgetPercent);
  return Math.max(0, Math.round((100 - budget) * 18));
};

const gpuTokenLimitForBudget = (settings) => {
  const budget = clampPerformanceBudget(settings.performanceBudgetPercent);
  return Math.max(32, Math.round(480 * (budget / 100)));
};

const isPriorityLearningEvent = (eventType) => {
  const type = String(eventType || '');
  return type === 'page_view'
    || type === 'click'
    || type === 'input'
    || type === 'selection'
    || type.endsWith('_change')
    || ['media_play', 'media_pause', 'media_ended', 'media_seek', 'media_rate', 'media_volume'].includes(type)
    || ['autonomous_link', 'autonomous_button', 'autonomous_input'].includes(type);
};

const shouldThrottleLearning = (modelId, payload, settings) => {
  const budget = clampPerformanceBudget(settings.performanceBudgetPercent);
  if (budget >= 100) return false;
  const baseThrottleMs = learningThrottleMsForBudget(settings);
  if (baseThrottleMs <= 0) return false;
  const eventType = payload?.eventType;
  const throttleMs = isPriorityLearningEvent(eventType)
    ? Math.round(baseThrottleMs * 0.25)
    : baseThrottleMs;
  const now = Date.now();
  const lastLearnAt = performanceState.lastLearnAtByModel.get(modelId) || 0;
  if (now - lastLearnAt < throttleMs) return true;
  performanceState.lastLearnAtByModel.set(modelId, now);
  return false;
};

const appendEvents = async (events, modelId = DEFAULT_MODEL_ID) => {
  const key = eventStorageKey(modelId);
  const stored = await chrome.storage.local.get([key]);
  const existing = stored[key] ?? [];
  const nextEvents = [...events, ...existing].slice(0, 220);
  try {
    await chrome.storage.local.set({ [key]: nextEvents });
  } catch {
    await chrome.storage.local.set({
      [key]: nextEvents.slice(0, 90).map((event) => ({
        step: event.step,
        kind: event.kind,
        value: event.value,
        label: event.label,
        meta: event.meta ? { url: event.meta.url || '', assignments: event.meta.assignments || 0 } : undefined,
      })),
    });
  }
};

const shouldUseWebGpu = (model, settings, tokenCount) => {
  if (settings.computeBackend === 'cpu') return false;
  if (settings.computeBackend === 'webgpu') return true;
  return (model.stats?.avgStepMs || 0) > 12 || modelVocabulary(model).length > 4000 || tokenCount > 160;
};

const buildGpuHints = async (model, payload, settings, modelId = DEFAULT_MODEL_ID) => {
  const performanceBudgetPercent = clampPerformanceBudget(settings.performanceBudgetPercent);
  const gpuTokenLimit = gpuTokenLimitForBudget(settings);
  const gpuThrottleMs = gpuThrottleMsForBudget(settings);
  const learningThrottleMs = learningThrottleMsForBudget(settings);
  const baseHints = {
    performanceBudgetPercent,
    learningThrottleMs,
    gpuTokenLimit,
    gpuThrottleMs,
  };
  if (settings.computeBackend === 'cpu') {
    return { ...baseHints, backend: 'cpu', webgpuAvailable: getWebGpuStatus().available };
  }
  const source = `${payload.eventType} ${payload.elementText ?? ''} ${payload.mediaCueText ?? ''} ${payload.visibleText ?? ''}`;
  const tokens = tokenize(source).slice(0, gpuTokenLimit);
  if (!shouldUseWebGpu(model, settings, tokens.length)) {
    return { ...baseHints, backend: 'cpu', webgpuAvailable: getWebGpuStatus().available };
  }
  const now = Date.now();
  const lastGpuAt = performanceState.lastGpuAtByModel.get(modelId) || 0;
  if (gpuThrottleMs > 0 && now - lastGpuAt < gpuThrottleMs) {
    return { ...baseHints, backend: 'cpu-throttled', webgpuAvailable: getWebGpuStatus().available, gpuThrottled: true };
  }
  const tokenMap = new Map(modelVocabulary(model).map((neuron) => [neuron.token, neuron]));
  const voltages = tokens.map((token) => tokenMap.get(token)?.v || 0);
  const result = await computeVoltagesOnGpu({
    voltages,
    salience: 0.6,
    reward: 0.5,
  });
  if (result.backend === 'webgpu') performanceState.lastGpuAtByModel.set(modelId, now);
  return {
    ...baseHints,
    backend: result.backend,
    webgpuAvailable: result.status.available,
    gpuVoltages: result.voltages,
  };
};

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  const index = await getModelIndex();
  await setModelIndex(index);
  const model = await getStoredModel(DEFAULT_MODEL_ID);
  await setStoredModel(model, DEFAULT_MODEL_ID);
  const settings = await getLearningSettings();
  await chrome.storage.local.set({ [SETTINGS_STORAGE_KEY]: settings });
});

chrome.action.onClicked.addListener(async (tab) => {
  if (tab?.windowId !== undefined) {
    if (tab.id !== undefined) {
      const states = await getTabStates();
      const state = states[String(tab.id)];
      if (state && (state.learningEnabled !== false || state.autonomousExploreEnabled === true)) {
        await patchTabState(tab.id, { title: tab.title || state.title, url: tab.url || state.url });
      }
    }
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  const states = await getTabStates();
  if (states[String(tabId)]) {
    delete states[String(tabId)];
    await setTabStates(states);
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (![
    'BROWSER_SNN_EVENT',
    'BROWSER_SNN_SLEEP',
    'BROWSER_SNN_GET_TAB_STATE',
    'BROWSER_SNN_SET_TAB_STATE',
    'BROWSER_SNN_CREATE_MODEL',
    'BROWSER_SNN_IMPORT_MODEL',
    'BROWSER_SNN_CAPTURE_TAB',
  ].includes(message?.type)) return false;
  void (async () => {
    try {
      if (message?.type === 'BROWSER_SNN_GET_TAB_STATE') {
        const tabId = message.tabId ?? _sender.tab?.id;
        if (tabId === undefined) {
          sendResponse({ ok: false, error: 'tab id unavailable' });
          return;
        }
        sendResponse({ ok: true, tabState: await getTabState(tabId), modelIndex: await getModelIndex() });
        return;
      }
      if (message?.type === 'BROWSER_SNN_SET_TAB_STATE') {
        const tabId = message.tabId;
        if (tabId === undefined) {
          sendResponse({ ok: false, error: 'tab id unavailable' });
          return;
        }
        const tabState = await patchTabState(tabId, message.patch || {});
        void notifyTabState(tabId, tabState);
        sendResponse({ ok: true, tabState });
        return;
      }
      if (message?.type === 'BROWSER_SNN_CREATE_MODEL') {
        const index = await getModelIndex();
        const id = `model-${Date.now().toString(36)}`;
        const name = String(message.name || `Browser SNN ${index.models.length + 1}`).slice(0, 80);
        index.models.push({ id, name, createdAt: Date.now() });
        index.selectedModelId = id;
        await setModelIndex(index);
        await setStoredModel(createModel(), id);
        await appendEvents([], id);
        sendResponse({ ok: true, modelIndex: index, modelId: id });
        return;
      }
      if (message?.type === 'BROWSER_SNN_IMPORT_MODEL') {
        const index = await getModelIndex();
        const sourceName = String(message.sourceName || 'imported model').slice(0, 120);
        const mode = message.mode === 'merge' ? 'merge' : 'new';
        let modelId = message.modelId || DEFAULT_MODEL_ID;
        let model;
        if (mode === 'merge') {
          const base = await getStoredModel(modelId);
          model = mergeImportedModel(base, message.model || {}, { sourceName });
        } else {
          modelId = `import-${Date.now().toString(36)}`;
          const name = String(message.name || sourceName.replace(/\.(edensnn|chat\.json|json|pt)$/i, '') || `Imported SNN ${index.models.length + 1}`).slice(0, 80);
          model = normalizeImportedModel(message.model || {}, { sourceName });
          index.models.push({ id: modelId, name, createdAt: Date.now(), importedFrom: sourceName });
          index.selectedModelId = modelId;
          await setModelIndex(index);
        }
        await setStoredModel(model, modelId);
        await appendEvents([{
          step: model.stats?.steps || 0,
          kind: 'model_import',
          value: model.vocabulary?.length || 0,
          label: mode === 'merge' ? 'merge_imported_model' : 'new_imported_model',
          meta: {
            sourceName,
            modelId,
            associations: model.associations?.length || 0,
            crossModalRelations: model.crossModalRelations?.length || 0,
          },
        }], modelId);
        sendResponse({ ok: true, modelIndex: index, modelId, mode });
        return;
      }
      if (message?.type === 'BROWSER_SNN_SLEEP') {
        const modelId = message.modelId || DEFAULT_MODEL_ID;
        const model = await getStoredModel(modelId);
        const result = sleepConsolidateModel(model, message.options || {});
        await setStoredModel(result.model, modelId);
        await appendEvents(result.trace, modelId);
        sendResponse({ ok: true, summary: result.summary });
        return;
      }
      if (message?.type === 'BROWSER_SNN_CAPTURE_TAB') {
        const windowId = _sender.tab?.windowId;
        if (windowId === undefined) {
          sendResponse({ ok: false, error: 'window id unavailable' });
          return;
        }
        const dataUrl = await chrome.tabs.captureVisibleTab(windowId, { format: 'png' });
        sendResponse({ ok: true, dataUrl });
        return;
      }
      const tabId = _sender.tab?.id;
      const tabState = tabId === undefined ? defaultTabState() : await getTabState(tabId);
      const modelId = tabState.modelId || DEFAULT_MODEL_ID;
      if (!tabState.learningEnabled) {
        sendResponse({ ok: true, paused: true, reward: 0 });
        return;
      }
      const settings = await getLearningSettings();
      if (tabId !== undefined) {
        await patchTabState(tabId, {
          title: message.payload?.title || tabState.title,
          url: message.payload?.url || tabState.url,
          lastEventAt: Date.now(),
        });
      }
      if (shouldThrottleLearning(modelId, message.payload, settings)) {
        sendResponse({
          ok: true,
          skipped: true,
          reason: 'performance_limit',
          reward: 0,
          performanceBudgetPercent: clampPerformanceBudget(settings.performanceBudgetPercent),
        });
        return;
      }
      const model = await getStoredModel(modelId);
      const gpuHints = await buildGpuHints(model, message.payload, settings, modelId);
      const result = observeEvent(model, message.payload, settings, gpuHints);
      await setStoredModel(result.model, modelId);
      await appendEvents(result.trace.map((event) => ({
        ...event,
        meta: { ...(event.meta || {}), tabId, modelId },
      })), modelId);
      sendResponse({ ok: true, reward: result.reward });
    } catch (error) {
      sendResponse({ ok: false, error: String(error?.message || error) });
    }
  })();
  return true;
});
