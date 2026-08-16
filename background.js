// background.js — MV3 service worker.
//
// Owns the single offscreen document that performs actual model inference
// (service workers can't reliably do WASM+canvas+WebGPU work), and relays
// classification requests from content scripts to it. Also tracks model
// load status/settings so the popup can show something useful.

const OFFSCREEN_URL = 'offscreen/offscreen.html';

const state = {
  enabled: true,
  threshold: 65, // percent, matches the bounty's required evaluation threshold
  // Off by default: Xenova/yolos-tiny (the model behind this signal) has
  // two independently reported real performance problems in
  // transformers.js -- ~15s/image under WASM and UI-blocking sluggishness
  // under WebGPU (see offscreen.js for issue links). It's a genuine
  // accuracy improvement for portrait-style false positives, but that's a
  // real speed tradeoff a person should opt into, not one we should
  // silently impose on every image on every page.
  personDampening: false,
  modelStatus: { status: 'idle' }, // idle | loading | downloading | ready | error
  stats: { analyzed: 0, flaggedAi: 0 },
};

let creatingOffscreen = null;

async function hasOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  return contexts.length > 0;
}

async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) return;
  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }
  creatingOffscreen = chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['BLOBS'],
    justification:
      'Decode fetched image bytes and run a local ONNX vision model (WebGPU/WASM) to classify them as real or AI-generated. All processing stays on-device.',
  });
  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

async function loadSettings() {
  const stored = await chrome.storage.local.get(['enabled', 'threshold', 'personDampening']);
  if (typeof stored.enabled === 'boolean') state.enabled = stored.enabled;
  if (typeof stored.threshold === 'number') state.threshold = stored.threshold;
  if (typeof stored.personDampening === 'boolean') state.personDampening = stored.personDampening;
}
loadSettings();

// --- Messaging -----------------------------------------------------------
// Content scripts & popup -> background: plain messages (no `target`).
// Background -> offscreen: messages with target:'offscreen'.
// Offscreen -> background: MODEL_PROGRESS messages with target:'background'.

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return undefined;

  // Progress updates coming from the offscreen document.
  if (message.target === 'background' && message.type === 'MODEL_PROGRESS') {
    state.modelStatus = message.payload;
    chrome.runtime.sendMessage({ type: 'MODEL_STATUS_UPDATE', payload: state.modelStatus }).catch(() => {});
    return undefined;
  }

  if (message.type === 'GET_STATE') {
    (async () => {
      // state.modelStatus lives in this service worker's memory, which
      // Chrome discards whenever it terminates the (ephemeral, MV3)
      // service worker after ~30s idle -- resetting it back to "idle"
      // even though the model is still fully loaded in the
      // longer-lived offscreen document and cached in the browser. Ask
      // the offscreen document for its actual current status rather
      // than trusting our own possibly-stale copy.
      try {
        await ensureOffscreenDocument();
        const res = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'GET_STATUS' });
        if (res && res.ok && res.status) {
          state.modelStatus = res.status;
        }
        // Genuinely idle (fresh offscreen doc, nothing loaded yet): kick
        // off loading now so opening the popup doesn't just report "idle"
        // and leave the person waiting for the first image to trigger it.
        if (!res || !res.ok || !res.status || res.status.status === 'idle') {
          chrome.runtime.sendMessage({ target: 'offscreen', type: 'WARM_UP' }).catch(() => {});
        }
      } catch (_) {
        /* offscreen not ready yet; fall back to whatever we have */
      }
      sendResponse({ ...state });
    })();
    return true;
  }

  if (message.type === 'SET_SETTINGS') {
    if (typeof message.enabled === 'boolean') state.enabled = message.enabled;
    if (typeof message.threshold === 'number') state.threshold = message.threshold;
    if (typeof message.personDampening === 'boolean') state.personDampening = message.personDampening;
    chrome.storage.local.set({ enabled: state.enabled, threshold: state.threshold, personDampening: state.personDampening });
    // Let any open content scripts know the settings changed.
    chrome.tabs.query({}).then((tabs) => {
      for (const tab of tabs) {
        if (tab.id != null) {
          chrome.tabs.sendMessage(tab.id, { type: 'SETTINGS_CHANGED', enabled: state.enabled, threshold: state.threshold }).catch(() => {});
        }
      }
    });
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === 'CLASSIFY_IMAGE') {
    (async () => {
      if (!state.enabled) {
        sendResponse({ ok: false, error: 'disabled' });
        return;
      }
      try {
        await ensureOffscreenDocument();
        const requestId = `${sender.tab ? sender.tab.id : 'popup'}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const result = await chrome.runtime.sendMessage({
          target: 'offscreen',
          type: 'CLASSIFY',
          requestId,
          url: message.url,
          imageBytes: message.imageBytes, // present when content.js could read pixels locally (no network needed)
          hostname: message.hostname,
          tierThreshold: state.threshold,
          enablePersonDampening: state.personDampening,
        });
        if (result && result.ok) {
          state.stats.analyzed += 1;
          // The three-tier verdict (ai / uncertain / real) is already
          // computed against state.threshold inside offscreen.js, so no
          // second threshold check is needed here.
          if (result.result.verdict === 'ai') {
            state.stats.flaggedAi += 1;
          }
          sendResponse({ ok: true, result: result.result });
        } else {
          sendResponse({ ok: false, error: (result && result.error) || 'unknown offscreen error' });
        }
      } catch (err) {
        sendResponse({ ok: false, error: String(err && err.message ? err.message : err) });
      }
    })();
    return true; // async
  }

  return undefined;
});

// Warm the model up as soon as the browser starts / extension loads so the
// first image on the first page doesn't pay the full load latency (the
// one-time weight download still only happens once regardless).
chrome.runtime.onInstalled.addListener(() => {
  ensureOffscreenDocument()
    .then(() => chrome.runtime.sendMessage({ target: 'offscreen', type: 'WARM_UP' }))
    .catch(() => {});
});
chrome.runtime.onStartup.addListener(() => {
  ensureOffscreenDocument()
    .then(() => chrome.runtime.sendMessage({ target: 'offscreen', type: 'WARM_UP' }))
    .catch(() => {});
});
