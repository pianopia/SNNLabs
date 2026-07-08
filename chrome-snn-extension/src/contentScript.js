const MAX_VISIBLE_TEXT = 2200;
const MEDIA_SAMPLE_INTERVAL_MS = 3200;
const VISUAL_SAMPLE_INTERVAL_MS = 4200;
const TEXT_REGION_SAMPLE_INTERVAL_MS = 5200;
const TEXT_REGION_MAX_PER_SCAN = 10;
const SETTINGS_STORAGE_KEY = 'elfentierBrowserSnnLearningSettings';
const DEFAULT_LEARNING_SETTINGS = {
  inputCapture: 'metadata',
  learningEnabled: false,
  autonomousExploreEnabled: false,
  autonomousIntervalSeconds: 18,
  autonomousMaxNavigations: 24,
};
let lastScrollAt = 0;
let lastInputAt = 0;
let mediaIndex = 0;
let learningSettings = { ...DEFAULT_LEARNING_SETTINGS };
let tabLearningState = {
  learningEnabled: DEFAULT_LEARNING_SETTINGS.learningEnabled !== false,
  autonomousExploreEnabled: DEFAULT_LEARNING_SETTINGS.autonomousExploreEnabled === true,
  modelId: 'default',
};
let autonomousTimer = null;
let autonomousBusy = false;
let samplingTimer = null;
let mutationObserver = null;
const mediaIds = new WeakMap();
const mediaLastSampleAt = new WeakMap();
const visualIds = new WeakMap();
const visualLastSampleAt = new WeakMap();
const textRegionLastSampleAt = new WeakMap();
let textRegionScanBusy = false;
let lastTextRegionScanAt = 0;
let lastSelectionScanAt = 0;
let lastPageViewHref = '';

const ACTION_CHANGE_DELAY_MS = 900;
const CHANGE_TEXT_LIMIT = 1200;

const isLearningActive = () => tabLearningState.learningEnabled === true;
const isExploreActive = () => tabLearningState.autonomousExploreEnabled === true;

const refreshLearningSettings = () => new Promise((resolve) => {
  try {
    let pending = 2;
    const done = () => {
      pending -= 1;
      if (pending <= 0) resolve();
    };
    chrome.storage.local.get([SETTINGS_STORAGE_KEY], (stored) => {
      learningSettings = { ...DEFAULT_LEARNING_SETTINGS, ...(stored?.[SETTINGS_STORAGE_KEY] || {}) };
      done();
    });
    const sent = chrome.runtime.sendMessage({ type: 'BROWSER_SNN_GET_TAB_STATE' }, (response) => {
      if (!chrome.runtime.lastError && response?.ok) {
        tabLearningState = { ...tabLearningState, ...(response.tabState || {}) };
      }
      done();
    });
    if (sent?.catch) sent.catch(() => done());
  } catch {
    learningSettings = { ...DEFAULT_LEARNING_SETTINGS };
    resolve();
  }
});

const autonomousKey = () => `elfentierAutonomous:${location.origin}`;

const autonomousState = () => {
  try {
    return JSON.parse(sessionStorage.getItem(autonomousKey()) || '{"visited":[],"navigations":0}');
  } catch {
    return { visited: [], navigations: 0 };
  }
};

const saveAutonomousState = (state) => {
  try {
    sessionStorage.setItem(autonomousKey(), JSON.stringify({
      visited: [...new Set(state.visited || [])].slice(-120),
      navigations: Math.max(0, Number(state.navigations || 0)),
    }));
  } catch {
    // Some pages restrict sessionStorage. Exploration still works without persistence.
  }
};

const visibleText = () => {
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = node.nodeValue?.replace(/\s+/g, ' ').trim();
      if (!text || text.length < 2) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      const style = getComputedStyle(parent);
      if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) return NodeFilter.FILTER_REJECT;
      const rect = parent.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const chunks = [];
  let node = walker.nextNode();
  while (node && chunks.join(' ').length < MAX_VISIBLE_TEXT) {
    chunks.push(node.nodeValue.replace(/\s+/g, ' ').trim());
    node = walker.nextNode();
  }
  return chunks.join(' ').slice(0, MAX_VISIBLE_TEXT);
};

const textWords = (text) => new Set(
  String(text || '')
    .toLowerCase()
    .split(/[^\p{L}\p{N}_-]+/u)
    .map((part) => part.trim())
    .filter((part) => part.length >= 2)
);

const pageSnapshot = () => ({
  url: location.href,
  title: document.title,
  visibleText: visibleText(),
  scrollY: Math.round(window.scrollY),
  viewportHeight: window.innerHeight,
});

const diffSnapshot = (before, after) => {
  const beforeWords = textWords(before.visibleText);
  const afterWords = textWords(after.visibleText);
  const added = [];
  const removed = [];
  for (const word of afterWords) {
    if (!beforeWords.has(word)) added.push(word);
    if (added.length >= 80) break;
  }
  for (const word of beforeWords) {
    if (!afterWords.has(word)) removed.push(word);
    if (removed.length >= 60) break;
  }
  const urlChanged = before.url !== after.url;
  const titleChanged = before.title !== after.title;
  const textChanged = added.length > 0 || removed.length > 0;
  const scrollChanged = Math.abs((after.scrollY || 0) - (before.scrollY || 0)) > 80;
  return {
    changed: urlChanged || titleChanged || textChanged || scrollChanged,
    addedText: added.join(' ').slice(0, CHANGE_TEXT_LIMIT),
    removedText: removed.join(' ').slice(0, 700),
    urlChanged,
    titleChanged,
    priorUrl: before.url,
    scrollDelta: (after.scrollY || 0) - (before.scrollY || 0),
  };
};

const scheduleChangeObservation = (eventType, before, context = {}) => {
  window.setTimeout(() => {
    if (!isLearningActive()) return;
    const after = pageSnapshot();
    const diff = diffSnapshot(before, after);
    if (!diff.changed) return;
    emit(eventType, {
      ...context,
      url: after.url,
      title: after.title,
      visibleText: after.visibleText,
      scrollY: after.scrollY,
      viewportHeight: after.viewportHeight,
      elementText: [
        context.elementText || '',
        diff.addedText ? `added ${diff.addedText}` : '',
        diff.removedText ? `removed ${diff.removedText}` : '',
      ].filter(Boolean).join(' ').slice(0, CHANGE_TEXT_LIMIT),
      change: diff,
    });
  }, ACTION_CHANGE_DELAY_MS);
};

const elementLabel = (element) => {
  if (!element) return '';
  const labelled = element.getAttribute?.('aria-label') || element.getAttribute?.('title') || element.getAttribute?.('alt');
  const tagName = element.tagName?.toLowerCase?.() || '';
  const inputType = element.getAttribute?.('type') || '';
  const isSensitiveInput = tagName === 'input' || tagName === 'textarea' || element.isContentEditable;
  const inputValue = String(element.value || '').replace(/\s+/g, ' ').trim();
  const valueHint = isSensitiveInput && learningSettings.inputCapture === 'full' ? inputValue
    : isSensitiveInput && learningSettings.inputCapture === 'text' && inputType !== 'password' ? inputValue
      : isSensitiveInput ? `[${tagName || 'input'}:${inputType || 'text'}]`
        : element.value;
  const text = element.innerText || element.textContent || valueHint || '';
  const label = String(labelled || '').replace(/\s+/g, ' ').trim();
  const body = String(text || '').replace(/\s+/g, ' ').trim();
  return isSensitiveInput && body ? `${label} ${body}`.trim().slice(0, 500) : String(label || body).slice(0, 500);
};

const nearbyText = (element) => {
  if (!element) return '';
  const chunks = [
    elementLabel(element),
    elementLabel(element.closest?.('figure')),
    elementLabel(element.closest?.('article')),
    elementLabel(element.parentElement),
  ].filter(Boolean);
  return [...new Set(chunks)].join(' ').slice(0, 900);
};

const activeCueText = (media) => {
  try {
    return Array.from(media.textTracks || []).flatMap((track) =>
      Array.from(track.activeCues || []).map((cue) => cue.text)
    ).join(' ').replace(/\s+/g, ' ').trim().slice(0, 900);
  } catch {
    return '';
  }
};

const youtubeCaptionText = () => {
  const selectors = [
    '.ytp-caption-segment',
    '.caption-window .captions-text',
    'yt-formatted-string.ytd-transcript-segment-renderer',
  ];
  return selectors
    .flatMap((selector) => Array.from(document.querySelectorAll(selector)))
    .map((node) => node.textContent?.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(' ')
    .slice(0, 1200);
};

const youtubeTranscriptText = () => {
  const transcript = Array.from(document.querySelectorAll('ytd-transcript-segment-renderer, ytd-transcript-segment-list-renderer'))
    .map((node) => node.textContent?.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(' ');
  if (transcript) return transcript.slice(0, 1800);
  const description = document.querySelector('#description-inline-expander, ytd-watch-metadata #description')?.textContent || '';
  const title = document.querySelector('h1 yt-formatted-string, h1.title')?.textContent || document.title || '';
  return `${title} ${description}`.replace(/\s+/g, ' ').trim().slice(0, 1200);
};

const mediaId = (media) => {
  if (!mediaIds.has(media)) {
    mediaIndex += 1;
    mediaIds.set(media, `media-${mediaIndex}`);
  }
  return mediaIds.get(media);
};

const mediaKind = (media) => media.tagName?.toLowerCase() === 'video' ? 'video' : 'audio';

const visualId = (element) => {
  if (!visualIds.has(element)) {
    mediaIndex += 1;
    visualIds.set(element, `visual-${mediaIndex}`);
  }
  return visualIds.get(element);
};

const visualElementKind = (element) => {
  const tag = element.tagName?.toLowerCase?.() || '';
  if (tag === 'video') return element.srcObject ? 'camera-or-stream' : 'video-frame';
  if (tag === 'img') return 'image';
  return 'background';
};

const visibleRatio = (rect) => {
  const left = Math.max(0, rect.left);
  const right = Math.min(window.innerWidth, rect.right);
  const top = Math.max(0, rect.top);
  const bottom = Math.min(window.innerHeight, rect.bottom);
  const visibleArea = Math.max(0, right - left) * Math.max(0, bottom - top);
  const totalArea = Math.max(1, rect.width * rect.height);
  return Math.max(0, Math.min(1, visibleArea / totalArea));
};

const visualPositionToken = (rect) => {
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const x = cx < window.innerWidth * 0.33 ? 'left' : cx > window.innerWidth * 0.66 ? 'right' : 'center';
  const y = cy < window.innerHeight * 0.33 ? 'top' : cy > window.innerHeight * 0.66 ? 'bottom' : 'middle';
  return `visual:position:${y}-${x}`;
};

const visualSizeToken = (rect) => {
  const area = rect.width * rect.height;
  const viewportArea = Math.max(1, window.innerWidth * window.innerHeight);
  const ratio = area / viewportArea;
  if (ratio > 0.45) return 'visual:size:hero';
  if (ratio > 0.16) return 'visual:size:large';
  if (ratio > 0.04) return 'visual:size:medium';
  return 'visual:size:small';
};

const quantize = (value, buckets = 5) => Math.max(0, Math.min(buckets - 1, Math.floor(value * buckets)));

const hueName = (r, g, b) => {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  if (max - min < 18) return 'neutral';
  if (r >= g && r >= b) return g > b ? 'warm' : 'red';
  if (g >= r && g >= b) return r > b ? 'yellow-green' : 'green';
  return r > g ? 'purple-blue' : 'blue';
};

const canvasVisualFeatures = (element) => {
  const canvas = document.createElement('canvas');
  const width = 16;
  const height = 16;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  try {
    ctx.drawImage(element, 0, 0, width, height);
    return extractVisualFeaturesFromPixels(ctx.getImageData(0, 0, width, height).data);
  } catch {
    return { readable: false };
  }
};

const extractVisualFeaturesFromPixels = (data) => {
  let brightness = 0;
  let saturation = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  let edge = 0;
  let previousLuma = null;
  const cells = data.length / 4;
  if (cells === 0) return { readable: false };
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const luma = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255;
    brightness += luma;
    saturation += max === 0 ? 0 : (max - min) / max;
    red += r;
    green += g;
    blue += b;
    if (previousLuma !== null) edge += Math.abs(luma - previousLuma);
    previousLuma = luma;
  }
  brightness /= cells;
  saturation /= cells;
  red /= cells;
  green /= cells;
  blue /= cells;
  edge /= cells;
  return {
    readable: true,
    brightness,
    saturation,
    edge,
    dominantHue: hueName(red, green, blue),
  };
};

const captureTabScreenshot = () => new Promise((resolve) => {
  try {
    const sent = chrome.runtime.sendMessage({ type: 'BROWSER_SNN_CAPTURE_TAB' }, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { ok: false });
    });
    if (sent?.catch) sent.catch((error) => resolve({ ok: false, error: String(error) }));
  } catch (error) {
    resolve({ ok: false, error: String(error) });
  }
});

const cropCaptureToFeatures = (dataUrl, rect) => new Promise((resolve) => {
  const dpr = window.devicePixelRatio || 1;
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    const sampleWidth = 16;
    const sampleHeight = 16;
    canvas.width = sampleWidth;
    canvas.height = sampleHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      resolve({ readable: false });
      return;
    }
    const sx = Math.max(0, Math.floor(rect.left * dpr));
    const sy = Math.max(0, Math.floor(rect.top * dpr));
    const sw = Math.max(1, Math.floor(rect.width * dpr));
    const sh = Math.max(1, Math.floor(rect.height * dpr));
    try {
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sampleWidth, sampleHeight);
      resolve(extractVisualFeaturesFromPixels(ctx.getImageData(0, 0, sampleWidth, sampleHeight).data));
    } catch {
      resolve({ readable: false });
    }
  };
  img.onerror = () => resolve({ readable: false });
  img.src = dataUrl;
});

const elementVisibleText = (element) => {
  if (!element) return '';
  const text = (element.innerText || element.textContent || elementLabel(element) || '')
    .replace(/\s+/g, ' ')
    .trim();
  return text.slice(0, 900);
};

const isTextRegionElement = (element) => {
  if (!element?.getBoundingClientRect) return false;
  const tag = element.tagName?.toLowerCase?.() || '';
  if (['script', 'style', 'noscript', 'svg', 'path', 'iframe'].includes(tag)) return false;
  if (element.closest?.('video,img')) return false;
  return elementVisibleText(element).length >= 2;
};

const textRegionCandidates = () => {
  const seen = new Set();
  const results = [];
  const root = document.body || document.documentElement;
  if (!root) return results;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = node.nodeValue?.replace(/\s+/g, ' ').trim();
      if (!text || text.length < 2) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (!parent || !isTextRegionElement(parent)) return NodeFilter.FILTER_REJECT;
      const style = getComputedStyle(parent);
      if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) {
        return NodeFilter.FILTER_REJECT;
      }
      const rect = parent.getBoundingClientRect();
      if (rect.width < 16 || rect.height < 10) return NodeFilter.FILTER_REJECT;
      if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  let node = walker.nextNode();
  while (node) {
    const parent = node.parentElement;
    if (parent && !seen.has(parent)) {
      seen.add(parent);
      results.push({
        element: parent,
        rect: parent.getBoundingClientRect(),
        text: elementVisibleText(parent),
      });
    }
    if (results.length >= TEXT_REGION_MAX_PER_SCAN * 3) break;
    node = walker.nextNode();
  }
  return results
    .sort((a, b) => a.rect.top - b.rect.top || a.rect.left - b.rect.left)
    .slice(0, TEXT_REGION_MAX_PER_SCAN);
};

const videoCaptionElements = () => Array.from(document.querySelectorAll([
  '.ytp-caption-segment',
  '.caption-window .captions-text',
  '.ytp-caption-window-container .captions-text',
  '[class*="caption-segment"]',
  '[class*="subtitle"]',
].join(',')))
  .filter((element) => isTextRegionElement(element) && isVisibleActionElement(element));

const textRegionPayload = (element, features, rect, textOverride = '') => {
  const text = String(textOverride || elementVisibleText(element)).trim();
  const tokens = [
    'visual:cortex:v1',
    'visual:kind:text-region',
    visualSizeToken(rect),
    visualPositionToken(rect),
    `visual:visible:${Math.round(visibleRatio(rect) * 10)}`,
    ...visualFeatureTokens(features),
  ];
  return {
    visualId: visualId(element),
    visualKind: 'text-region',
    visualWidth: Math.round(rect.width),
    visualHeight: Math.round(rect.height),
    visualVisibleRatio: visibleRatio(rect),
    visualTokens: tokens,
    visualText: text,
    visualReadablePixels: Boolean(features?.readable),
    elementText: text,
    textRegion: true,
  };
};

const emitTextRegionEvents = async (eventType, items, force = false) => {
  if (!items.length) return;
  if (!isLearningActive()) return;
  const capture = await captureTabScreenshot();
  if (!capture?.ok || !capture.dataUrl) return;
  const now = Date.now();
  for (const item of items) {
    const element = item.element || item;
    const rect = item.rect || element.getBoundingClientRect();
    const text = item.text || elementVisibleText(element);
    if (!isLearningActive()) return;
    if (!text || rect.width < 12 || rect.height < 8) continue;
    if (!force && now - (textRegionLastSampleAt.get(element) || 0) < TEXT_REGION_SAMPLE_INTERVAL_MS) continue;
    textRegionLastSampleAt.set(element, now);
    const features = await cropCaptureToFeatures(capture.dataUrl, rect);
    emit(eventType, textRegionPayload(element, features, rect, text));
  }
};

const scanTextVisualRegions = async (eventType = 'text_visual', force = false) => {
  if (!isLearningActive()) return;
  const now = Date.now();
  if (!force && now - lastTextRegionScanAt < TEXT_REGION_SAMPLE_INTERVAL_MS) return;
  if (textRegionScanBusy) return;
  textRegionScanBusy = true;
  lastTextRegionScanAt = now;
  try {
    await emitTextRegionEvents(eventType, textRegionCandidates(), force);
  } finally {
    textRegionScanBusy = false;
  }
};

const scanVideoTextRegions = async (media, force = false) => {
  if (mediaKind(media) !== 'video') return;
  const captions = videoCaptionElements().map((element) => ({
    element,
    rect: element.getBoundingClientRect(),
    text: elementVisibleText(element),
  }));
  if (captions.length > 0) {
    await emitTextRegionEvents('text_visual_caption', captions.slice(0, 6), force);
    return;
  }
  const cueText = [activeCueText(media), youtubeCaptionText()].filter(Boolean).join(' ').trim();
  if (!cueText) return;
  const rect = media.getBoundingClientRect();
  if (rect.width < 24 || rect.height < 24) return;
  const captionBand = {
    left: rect.left,
    top: rect.top + rect.height * 0.78,
    width: rect.width,
    height: Math.max(18, rect.height * 0.22),
    right: rect.right,
    bottom: rect.bottom,
  };
  await emitTextRegionEvents('text_visual_caption', [{
    element: media,
    rect: captionBand,
    text: cueText.slice(0, 900),
  }], force);
};

const scanTextRegionForElement = async (eventType, element, force = true) => {
  if (!element || !isTextRegionElement(element)) return;
  await emitTextRegionEvents(eventType, [{
    element,
    rect: element.getBoundingClientRect(),
    text: elementVisibleText(element),
  }], force);
};

const scanTextRegionForSelection = async (eventType = 'text_visual_selection') => {
  if (!isLearningActive()) return;
  const selection = document.getSelection();
  const text = selection?.toString().replace(/\s+/g, ' ').trim();
  if (!text || text.length < 4 || !selection?.rangeCount) return;
  const rect = selection.getRangeAt(0).getBoundingClientRect();
  if (rect.width < 8 || rect.height < 6) return;
  const anchor = selection.anchorNode?.parentElement || document.body;
  await emitTextRegionEvents(eventType, [{
    element: anchor,
    rect,
    text: text.slice(0, 900),
  }], true);
};

const visualFeatureTokens = (features) => {
  if (!features?.readable) return ['visual:pixels:unreadable'];
  const brightness = features.brightness < 0.28 ? 'dark' : features.brightness > 0.68 ? 'bright' : 'mid';
  const saturation = features.saturation < 0.22 ? 'muted' : features.saturation > 0.58 ? 'vivid' : 'balanced';
  const edge = features.edge > 0.095 ? 'high' : features.edge > 0.045 ? 'medium' : 'low';
  return [
    `visual:brightness:${brightness}`,
    `visual:saturation:${saturation}`,
    `visual:edge:${edge}`,
    `visual:hue:${features.dominantHue}`,
    `visual:brightness-bucket:${quantize(features.brightness)}`,
    `visual:saturation-bucket:${quantize(features.saturation)}`,
    `visual:edge-bucket:${quantize(Math.min(1, features.edge * 6))}`,
  ];
};

const visualPayload = (element) => {
  const rect = element.getBoundingClientRect();
  const kind = visualElementKind(element);
  let features = null;
  try {
    features = canvasVisualFeatures(element);
  } catch {
    features = { readable: false };
  }
  const tokens = [
    'visual:cortex:v1',
    `visual:kind:${kind}`,
    visualSizeToken(rect),
    visualPositionToken(rect),
    `visual:visible:${Math.round(visibleRatio(rect) * 10)}`,
    ...visualFeatureTokens(features),
  ];
  return {
    visualId: visualId(element),
    visualKind: kind,
    visualWidth: element.naturalWidth || element.videoWidth || Math.round(rect.width) || 0,
    visualHeight: element.naturalHeight || element.videoHeight || Math.round(rect.height) || 0,
    visualVisibleRatio: visibleRatio(rect),
    visualTokens: tokens,
    visualText: nearbyText(element),
    visualReadablePixels: Boolean(features?.readable),
    elementText: nearbyText(element),
  };
};

const visibleVisualElements = () => Array.from(document.querySelectorAll('img,video'))
  .filter((element) => {
    if (!isVisibleActionElement(element)) return false;
    const rect = element.getBoundingClientRect();
    if (rect.width < 24 || rect.height < 24) return false;
    if (element.tagName?.toLowerCase() === 'img' && (!element.complete || element.naturalWidth === 0)) return false;
    if (element.tagName?.toLowerCase() === 'video' && element.readyState < 2) return false;
    return true;
  });

const emitVisual = (eventType, element, force = false) => {
  if (!isLearningActive()) return;
  const now = Date.now();
  const lastAt = visualLastSampleAt.get(element) || 0;
  if (!force && now - lastAt < VISUAL_SAMPLE_INTERVAL_MS) return;
  visualLastSampleAt.set(element, now);
  emit(eventType, visualPayload(element));
};

const mediaPayload = (media) => ({
  mediaId: mediaId(media),
  mediaKind: mediaKind(media),
  mediaSrc: media.currentSrc || media.src || '',
  mediaDuration: Number.isFinite(media.duration) ? media.duration : 0,
  mediaCurrentTime: Number.isFinite(media.currentTime) ? media.currentTime : 0,
  mediaProgress: Number.isFinite(media.duration) && media.duration > 0 ? media.currentTime / media.duration : 0,
  mediaPaused: media.paused,
  mediaEnded: media.ended,
  mediaMuted: media.muted,
  mediaVolume: Number.isFinite(media.volume) ? media.volume : 0,
  mediaPlaybackRate: Number.isFinite(media.playbackRate) ? media.playbackRate : 1,
  mediaReadyState: media.readyState,
  mediaNetworkState: media.networkState,
  mediaWidth: media.videoWidth || media.clientWidth || 0,
  mediaHeight: media.videoHeight || media.clientHeight || 0,
  elementText: nearbyText(media),
  mediaCueText: [activeCueText(media), youtubeCaptionText()].filter(Boolean).join(' ').slice(0, 1400),
  mediaTranscriptText: youtubeTranscriptText(),
  ...(mediaKind(media) === 'video' ? visualPayload(media) : {}),
});

const emit = (eventType, extra = {}) => {
  if (!isLearningActive()) return;
  try {
    const sent = chrome.runtime.sendMessage({
      type: 'BROWSER_SNN_EVENT',
      payload: {
        eventType,
        url: location.href,
        title: document.title,
        visibleText: visibleText(),
        scrollY: Math.round(window.scrollY),
        viewportHeight: window.innerHeight,
        at: Date.now(),
        ...extra,
      },
    });
    if (sent?.catch) sent.catch(() => {});
  } catch {
    // The page may outlive the extension context during reloads or updates.
  }
};

const safeAutonomousHref = (anchor) => {
  const href = anchor.getAttribute('href') || '';
  if (!href || href.startsWith('#') || href.startsWith('javascript:') || href.startsWith('mailto:') || href.startsWith('tel:')) return '';
  if (anchor.hasAttribute('download')) return '';
  const label = elementLabel(anchor).toLowerCase();
  if (/(logout|log out|sign out|delete|remove|unsubscribe|checkout|cart|購入|削除|退会|ログアウト|支払い|精算)/i.test(label)) return '';
  try {
    const url = new URL(href, location.href);
    if (url.href === location.href || url.hash && `${url.origin}${url.pathname}${url.search}` === `${location.origin}${location.pathname}${location.search}`) return '';
    return url.href;
  } catch {
    return '';
  }
};

const visibleAutonomousLinks = () => Array.from(document.querySelectorAll('a[href]'))
  .map((anchor) => {
    const href = safeAutonomousHref(anchor);
    if (!href) return null;
    const rect = anchor.getBoundingClientRect();
    const style = getComputedStyle(anchor);
    if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) return null;
    if (rect.width < 16 || rect.height < 8) return null;
    if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) return null;
    return { href, label: elementLabel(anchor), rect };
  })
  .filter(Boolean);

const isVisibleActionElement = (element) => {
  const rect = element.getBoundingClientRect();
  const style = getComputedStyle(element);
  if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) return false;
  if (rect.width < 16 || rect.height < 8) return false;
  if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) return false;
  return true;
};

const safeAutonomousButton = (element) => {
  if (!isVisibleActionElement(element)) return null;
  if (element.disabled || element.getAttribute?.('aria-disabled') === 'true') return null;
  const label = elementLabel(element);
  if (!label && !element.id && !element.name) return null;
  if (/(logout|log out|sign out|delete|remove|unsubscribe|checkout|cart|purchase|buy|submit payment|購入|削除|退会|ログアウト|支払い|精算|注文|送信)/i.test(label)) return null;
  const tagName = element.tagName?.toLowerCase?.() || '';
  const type = element.getAttribute?.('type') || '';
  return {
    element,
    label: label || `${tagName}:${type || element.id || element.name || 'button'}`,
  };
};

const visibleAutonomousButtons = () => Array.from(document.querySelectorAll([
  'button',
  '[role="button"]',
  'input[type="button"]',
  'input[type="submit"]',
  'input[type="reset"]',
  'summary',
].join(',')))
  .map(safeAutonomousButton)
  .filter(Boolean);

const autonomousInputValue = (element) => {
  const label = elementLabel(element).toLowerCase();
  const pageWords = Array.from(textWords(`${document.title} ${visibleText()}`))
    .filter((word) => word.length >= 3 && !/(password|token|secret|email|phone|card|住所|電話|パスワード|秘密)/i.test(word))
    .slice(0, 3);
  if (/search|find|query|検索|探す/i.test(label)) {
    return pageWords.length ? pageWords.join(' ') : 'elfentier exploration';
  }
  if (/email|mail|phone|tel|password|pass|card|address|token|secret|メール|電話|住所|パスワード|秘密|カード/i.test(label)) {
    return '';
  }
  return pageWords.length ? `explore ${pageWords.join(' ')}` : 'elfentier exploration';
};

const safeAutonomousInput = (element) => {
  if (!isVisibleActionElement(element)) return null;
  if (element.disabled || element.readOnly || element.getAttribute?.('aria-disabled') === 'true') return null;
  const tagName = element.tagName?.toLowerCase?.() || '';
  const type = String(element.getAttribute?.('type') || 'text').toLowerCase();
  const editable = element.isContentEditable;
  const allowedInputTypes = new Set(['text', 'search', 'url', 'email', 'tel', 'number']);
  if (tagName === 'input' && !allowedInputTypes.has(type)) return null;
  if (tagName !== 'input' && tagName !== 'textarea' && !editable) return null;
  const value = autonomousInputValue(element);
  if (!value) return null;
  const label = elementLabel(element) || element.getAttribute?.('placeholder') || `${tagName}:${type}`;
  return { element, label, value };
};

const visibleAutonomousInputs = () => Array.from(document.querySelectorAll([
  'input',
  'textarea',
  '[contenteditable="true"]',
  '[contenteditable="plaintext-only"]',
].join(',')))
  .map(safeAutonomousInput)
  .filter(Boolean);

const setNativeInputValue = (element, value) => {
  if (element.isContentEditable) {
    element.focus();
    element.textContent = value;
  } else {
    element.focus();
    const proto = element.tagName?.toLowerCase() === 'textarea' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
    if (descriptor?.set) descriptor.set.call(element, value);
    else element.value = value;
  }
  element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
  element.dispatchEvent(new Event('change', { bubbles: true }));
};

const autonomousScroll = () => {
  const maxScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
  const current = Math.max(window.scrollY, document.documentElement.scrollTop || 0);
  const nearBottom = maxScroll > 0 && current / maxScroll > 0.82;
  const delta = nearBottom ? -Math.round(window.innerHeight * 0.62) : Math.round(window.innerHeight * (0.42 + Math.random() * 0.32));
  window.scrollBy({ top: delta, left: 0, behavior: 'smooth' });
  emit('autonomous_scroll', {
    elementText: nearBottom ? 'autonomous scroll up' : 'autonomous scroll down',
    autonomous: true,
  });
};

const autonomousExploreStep = async () => {
  if (autonomousBusy) return;
  if (!isExploreActive()) return;
  autonomousBusy = true;
  try {
    if (isLearningActive()) {
      emit('autonomous_scan', {
        elementText: 'autonomous browser exploration scan',
        autonomous: true,
      });
      scanMediaElements();
      scanVisualElements();
      void scanTextVisualRegions('text_visual', true);
      document.querySelectorAll('video,audio').forEach((media) => {
        if (!media.paused && !media.ended) emitMedia('media_sample', media, true);
      });
      visibleVisualElements().slice(0, 4).forEach((element) => emitVisual('autonomous_visual', element));
    }

    const state = autonomousState();
    const maxNavigations = Math.max(0, Number(learningSettings.autonomousMaxNavigations ?? 24));
    const links = visibleAutonomousLinks()
      .filter((item) => !(state.visited || []).includes(item.href));
    const buttons = visibleAutonomousButtons();
    const inputs = visibleAutonomousInputs();
    const shouldTypeInput = inputs.length > 0 && Math.random() < 0.2;
    const shouldClickButton = buttons.length > 0 && Math.random() < 0.22;
    const shouldNavigate = links.length > 0
      && maxNavigations > 0
      && (state.navigations || 0) < maxNavigations
      && Math.random() < 0.28;

    if (shouldTypeInput) {
      const chosen = inputs[Math.floor(Math.random() * Math.min(inputs.length, 8))];
      const before = pageSnapshot();
      emit('autonomous_input', {
        elementText: `${chosen.label} ${learningSettings.inputCapture === 'metadata' ? '[autonomous-input]' : chosen.value}`.slice(0, 500),
        tagName: chosen.element.tagName ?? '',
        autonomous: true,
      });
      setNativeInputValue(chosen.element, chosen.value);
      scheduleChangeObservation('autonomous_change', before, {
        elementText: `${chosen.label} ${learningSettings.inputCapture === 'metadata' ? '[autonomous-input]' : chosen.value}`.slice(0, 500),
        tagName: chosen.element.tagName ?? '',
        autonomous: true,
        actionType: 'input',
      });
      return;
    }

    if (shouldClickButton) {
      const chosen = buttons[Math.floor(Math.random() * Math.min(buttons.length, 8))];
      const before = pageSnapshot();
      emit('autonomous_button', {
        elementText: chosen.label,
        tagName: chosen.element.tagName ?? '',
        autonomous: true,
      });
      chosen.element.click();
      scheduleChangeObservation('autonomous_change', before, {
        elementText: chosen.label,
        tagName: chosen.element.tagName ?? '',
        autonomous: true,
        actionType: 'button',
      });
      return;
    }

    if (shouldNavigate) {
      const chosen = links[Math.floor(Math.random() * Math.min(links.length, 8))];
      state.visited = [...(state.visited || []), location.href, chosen.href];
      state.navigations = (state.navigations || 0) + 1;
      saveAutonomousState(state);
      emit('autonomous_link', {
        elementText: chosen.label || chosen.href,
        autonomous: true,
        targetUrl: chosen.href,
      });
      window.setTimeout(() => {
        location.assign(chosen.href);
      }, 450);
      return;
    }

    autonomousScroll();
  } finally {
    window.setTimeout(() => {
      autonomousBusy = false;
    }, 1200);
  }
};

const restartAutonomousExplorer = () => {
  if (autonomousTimer) window.clearInterval(autonomousTimer);
  autonomousTimer = null;
  if (!isExploreActive()) return;
  const seconds = Math.max(8, Math.min(180, Number(learningSettings.autonomousIntervalSeconds || 18)));
  autonomousTimer = window.setInterval(() => {
    void autonomousExploreStep();
  }, seconds * 1000);
  void autonomousExploreStep();
};

const emitMedia = (eventType, media, force = false) => {
  if (!isLearningActive()) return;
  const now = Date.now();
  const lastAt = mediaLastSampleAt.get(media) || 0;
  if (!force && now - lastAt < MEDIA_SAMPLE_INTERVAL_MS) return;
  mediaLastSampleAt.set(media, now);
  emit(eventType, mediaPayload(media));
  if (mediaKind(media) === 'video') {
    void scanVideoTextRegions(media, force);
  }
};

const bindMediaElement = (media) => {
  if (media.dataset?.elfentierSnnMediaBound === '1') return;
  if (media.dataset) media.dataset.elfentierSnnMediaBound = '1';

  media.addEventListener('play', () => emitMedia('media_play', media, true), true);
  media.addEventListener('pause', () => emitMedia('media_pause', media, true), true);
  media.addEventListener('ended', () => emitMedia('media_ended', media, true), true);
  media.addEventListener('seeked', () => emitMedia('media_seek', media, true), true);
  media.addEventListener('ratechange', () => emitMedia('media_rate', media, true), true);
  media.addEventListener('volumechange', () => emitMedia('media_volume', media, true), true);
  media.addEventListener('timeupdate', () => {
    if (!media.paused && !media.ended) emitMedia('media_sample', media);
  }, true);

  if (!media.paused || media.currentTime > 0) emitMedia('media_detected', media, true);
  if (mediaKind(media) === 'video' && (!media.paused || media.currentTime > 0)) emitVisual('visual_scan', media, true);
};

const scanMediaElements = () => {
  if (!isLearningActive()) return;
  document.querySelectorAll('video,audio').forEach(bindMediaElement);
};

const bindVisualElement = (element) => {
  if (element.dataset?.elfentierSnnVisualBound === '1') return;
  if (element.dataset) element.dataset.elfentierSnnVisualBound = '1';
  if (element.tagName?.toLowerCase() === 'img') {
    element.addEventListener('load', () => emitVisual('visual_scan', element, true), true);
    if (element.complete && element.naturalWidth > 0) emitVisual('visual_scan', element, true);
  }
};

const scanVisualElements = () => {
  if (!isLearningActive()) return;
  visibleVisualElements().slice(0, 8).forEach((element) => {
    bindVisualElement(element);
    emitVisual('visual_scan', element);
  });
};

window.addEventListener('click', (event) => {
  if (!isLearningActive()) return;
  const before = pageSnapshot();
  const target = event.target;
  const tagName = target?.tagName?.toLowerCase?.() || '';
  const clickPayload = {
    x: Math.round(event.clientX),
    y: Math.round(event.clientY),
    elementText: elementLabel(target),
    tagName: target?.tagName ?? '',
  };
  if (tagName === 'img' || tagName === 'video') {
    Object.assign(clickPayload, visualPayload(target));
  } else if (isTextRegionElement(target)) {
    void scanTextRegionForElement('text_visual_click', target, true);
  }
  emit('click', clickPayload);
  scheduleChangeObservation('click_change', before, {
    elementText: elementLabel(event.target),
    tagName: event.target?.tagName ?? '',
    actionType: 'user_click',
  });
}, true);

window.addEventListener('scroll', () => {
  if (!isLearningActive()) return;
  const now = Date.now();
  if (now - lastScrollAt < 750) return;
  lastScrollAt = now;
  emit('scroll');
  void scanTextVisualRegions('text_visual');
}, { passive: true });

window.addEventListener('input', (event) => {
  if (!isLearningActive()) return;
  const now = Date.now();
  if (now - lastInputAt < 650) return;
  lastInputAt = now;
  emit('input', {
    elementText: elementLabel(event.target),
    tagName: event.target?.tagName ?? '',
  });
}, true);

document.addEventListener('selectionchange', () => {
  if (!isLearningActive()) return;
  const selection = document.getSelection()?.toString().replace(/\s+/g, ' ').trim();
  if (!selection || selection.length < 8) return;
  const now = Date.now();
  if (now - lastSelectionScanAt < 1200) return;
  lastSelectionScanAt = now;
  emit('selection', { elementText: selection.slice(0, 500) });
  void scanTextRegionForSelection('text_visual_selection');
});

const emitPageViewIfLearning = (force = false) => {
  if (!isLearningActive()) return;
  if (!force && lastPageViewHref === location.href) return;
  lastPageViewHref = location.href;
  emit('page_view');
};

const runLearningSample = () => {
  if (!isLearningActive()) return;
  document.querySelectorAll('video,audio').forEach((media) => {
    if (!media.paused && !media.ended) emitMedia('media_sample', media);
  });
  visibleVisualElements().slice(0, 4).forEach((element) => emitVisual('visual_scan', element));
  void scanTextVisualRegions('text_visual');
};

const restartLearningSampler = () => {
  if (samplingTimer) window.clearInterval(samplingTimer);
  samplingTimer = null;
  if (!isLearningActive()) return;
  samplingTimer = window.setInterval(runLearningSample, MEDIA_SAMPLE_INTERVAL_MS);
};

const restartMutationObserver = () => {
  if (mutationObserver && !isLearningActive()) {
    mutationObserver.disconnect();
    mutationObserver = null;
    return;
  }
  if (mutationObserver || !isLearningActive() || !document.documentElement) return;
  mutationObserver = new MutationObserver(() => {
    if (!isLearningActive()) return;
    scanMediaElements();
    scanVisualElements();
    void scanTextVisualRegions('text_visual');
  });
  mutationObserver.observe(document.documentElement, { childList: true, subtree: true });
};

const syncRuntimeWork = ({ sendPageView = false } = {}) => {
  restartLearningSampler();
  restartMutationObserver();
  restartAutonomousExplorer();
  if (!isLearningActive()) return;
  if (sendPageView) emitPageViewIfLearning(true);
  scanMediaElements();
  scanVisualElements();
  window.setTimeout(() => {
    void scanTextVisualRegions('text_visual', true);
  }, 900);
};

chrome.runtime?.onMessage?.addListener((message) => {
  if (message?.type !== 'BROWSER_SNN_TAB_STATE') return false;
  tabLearningState = { ...tabLearningState, ...(message.tabState || {}) };
  syncRuntimeWork({ sendPageView: true });
  return false;
});

chrome.storage?.onChanged?.addListener((changes, areaName) => {
  if (areaName === 'local' && changes[SETTINGS_STORAGE_KEY]) {
    void refreshLearningSettings().then(() => {
      syncRuntimeWork();
    });
  }
});

void refreshLearningSettings().then(() => {
  syncRuntimeWork({ sendPageView: true });
});
