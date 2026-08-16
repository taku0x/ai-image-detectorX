// offscreen.js — runs inside the hidden offscreen document.
//
// Responsibilities:
//   1. Load a Vision-Transformer image classifier fully client-side via
//      transformers.js (ONNX Runtime Web, WebGPU with WASM fallback).
//      Model weights are fetched once from Hugging Face Hub and then
//      persisted by transformers.js in the browser's Cache Storage, so
//      every subsequent load (including after Chrome restarts) is 100%
//      offline.
//   2. Fetch candidate image bytes (host_permissions on the extension let
//      this bypass the *page's* CORS restrctions — nothing is sent
//      anywhere except the one-time model download to huggingface.co).
//   3. Run the classifier + a metadata heuristic scan, combine them, and
//      return a verdict to the background script.

import { pipeline, env, RawImage } from '../vendor/transformers.web.min.js';
import { scanImageMetadata, scoreSourceDomain } from '../lib/metadata-heuristics.js';
import { computePixelSignals } from '../lib/pixel-signals.js';
import { scoresToAiProbability, combineSignals, applyPixelAdjustment } from '../lib/combine-signals.js';
import { MODEL_ID, PERSON_DETECTOR_ID, PERSON_DETECTOR_THRESHOLD, PERSON_DOMINANT_FRACTION } from '../lib/model-config.js';

// --- transformers.js environment configuration -----------------------
// No local model *files* ship in the extension (only the JS runtime and
// the ONNX Runtime WASM binaries do). The classifier weights are fetched
// once from the Hugging Face Hub on first use and cached by the browser.
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true; // persists in Cache Storage -> true one-time download
env.useCustomCache = false;

// Point ONNX Runtime Web at the WASM binaries we vendor inside the
// extension so nothing is fetched from a CDN at runtime.
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('vendor/ort/');
// Force a deterministic, broadly-compatible execution mode: single
// threaded WASM (no SharedArrayBuffer / cross-origin-isolation required).
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = true;

// Publicly available ONNX vision classifier fine-tuned specifically for
// general real-vs-AI-generated image classification (photos, art, web
// graphics -- NOT restricted to human faces), already packaged for
// transformers.js (see README for provenance + license details).
//
// NOTE: this extension previously used
// onnx-community/Deep-Fake-Detector-v2-Model-ONNX (face-swap-only, poor
// fit -- see git history), then Organika/sdxl-detector (general-purpose,
// but CC-BY-NC-3.0 licensed and only ships an unquantized fp32 onnx).
// amrita-detectly/detect-ai-image-v1 is a quantized build of
// Smogy/SMOGY-Ai-images-detector (itself a further fine-tune of
// Organika/sdxl-detector on a broader real+AI dataset), MIT-licensed, and
// explicitly published with transformers.js/WebGPU support in mind --
// it ships model.onnx, model_quantized.onnx, model_fp16.onnx, and
// model_fp32.onnx, so we can let transformers.js auto-pick the right one
// per device instead of forcing fp32 the way Organika required.
//
// MODEL_ID and the person-detector settings below now live in
// lib/model-config.js, shared with scripts/benchmark.mjs, so the model we
// benchmark can never silently drift from the model we actually ship.

// Secondary, best-effort signal: a small COCO-trained object detector used
// only to check whether a human figure fills most of the frame (i.e. this
// looks like a portrait/headshot, not full face-landmark analysis -- that
// would need a much less battle-tested model than this one, which has an
// actual test in transformers.js's own test suite). This targets a
// specific, observed failure mode: professionally-lit portrait photography
// reads as "AI" to the main classifier far more often than it should.
// Entirely optional -- if it fails to load or errors on a given image, we
// silently skip the dampening it would have applied rather than block
// classification on it.
// FACE_DAMPING_FACTOR itself now lives in lib/combine-signals.js (shared
// with scripts/benchmark.mjs) -- see that file for the constant and its
// tuning-history comment.

let classifierPromise = null;
let modelDevice = 'unknown';
// Mirrors the last-reported status locally too, so a GET_STATUS query can
// answer instantly without needing background.js's (potentially stale,
// see below) copy.
let lastStatus = { status: 'idle' };

async function reportProgress(payload) {
  lastStatus = payload;
  try {
    await chrome.runtime.sendMessage({ target: 'background', type: 'MODEL_PROGRESS', payload });
  } catch (_) {
    /* background may not be listening yet; safe to ignore */
  }
}

async function getClassifier() {
  if (classifierPromise) return classifierPromise;

  classifierPromise = (async () => {
    const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    modelDevice = hasWebGPU ? 'webgpu' : 'wasm';

    await reportProgress({ status: 'loading', device: modelDevice });

    const classifier = await pipeline('image-classification', MODEL_ID, {
      device: modelDevice,
      // amrita-detectly/detect-ai-image-v1 ships properly-named
      // model_quantized.onnx / model_fp16.onnx / model_fp32.onnx files, so
      // we let transformers.js pick the right one for the active device
      // (q8 for wasm, fp32 for webgpu) instead of forcing a dtype.
      progress_callback: (p) => {
        if (p && p.status === 'progress') {
          reportProgress({
            status: 'downloading',
            file: p.file,
            loaded: p.loaded,
            total: p.total,
            progress: p.total ? p.loaded / p.total : undefined,
          });
        }
      },
    });

    await reportProgress({ status: 'ready', device: modelDevice });
    return classifier;
  })();

  try {
    return await classifierPromise;
  } catch (err) {
    classifierPromise = null; // allow retry on next request
    await reportProgress({ status: 'error', message: String(err && err.message ? err.message : err) });
    throw err;
  }
}

let personDetectorPromise = null;

// Lazily loaded and never allowed to block/fail the main classification --
// this is a bonus signal, not core functionality. Forced onto 'wasm'
// rather than following the main classifier's device choice: this specific
// model is known to run correctly but unusually slowly under WebGPU
// (see huggingface/transformers.js#983), and correctness/stability matter
// more than speed for a secondary dampening signal.
async function getPersonDetector() {
  if (personDetectorPromise) return personDetectorPromise;
  personDetectorPromise = pipeline('object-detection', PERSON_DETECTOR_ID, { device: 'wasm' }).catch((err) => {
    personDetectorPromise = null;
    throw err;
  });
  return personDetectorPromise;
}

/**
 * Best-effort check for whether a human figure dominates the frame (i.e.
 * this looks like a portrait/headshot). Never throws -- any failure just
 * means no dampening is applied, since this is a bonus signal layered on
 * top of the core classifier + metadata pipeline, not a dependency of it.
 *
 * IMPORTANT PERFORMANCE CAVEAT: Xenova/yolos-tiny has two independently
 * reported, real performance problems in transformers.js -- slow (~15s/
 * image on an M1) under WASM (huggingface/transformers.js#533), and
 * "abnormally sluggish... often blocking the UI" under WebGPU
 * (huggingface/transformers.js#983). Both execution paths are affected;
 * this is not a device-selection problem we can route around. Because of
 * this, callers MUST treat this as opt-in and gated (see classifyOne),
 * not something to run unconditionally on every image.
 */
async function detectDominantPerson(rawImage) {
  try {
    const detector = await getPersonDetector();
    const results = await detector(rawImage, { threshold: PERSON_DETECTOR_THRESHOLD, percentage: true });
    let maxFraction = 0;
    for (const r of results || []) {
      if (r.label !== 'person') continue;
      const { xmin, ymin, xmax, ymax } = r.box;
      const fraction = Math.max(0, xmax - xmin) * Math.max(0, ymax - ymin);
      if (fraction > maxFraction) maxFraction = fraction;
    }
    return { hasDominantPerson: maxFraction >= PERSON_DOMINANT_FRACTION, personFraction: maxFraction };
  } catch (_) {
    return { hasDominantPerson: false, personFraction: 0 };
  }
}

async function fetchImageBytes(url) {
  const res = await fetch(url, { credentials: 'omit', mode: 'cors' }).catch(() => null);
  if (!res || !res.ok) {
    // Retry once without explicit CORS mode for permissive same-origin cases.
    const res2 = await fetch(url, { credentials: 'omit' });
    if (!res2.ok) throw new Error(`fetch failed: ${res2.status}`);
    return await res2.arrayBuffer();
  }
  return await res.arrayBuffer();
}

async function classifyOne(url, { hostname, tierThreshold, enablePersonDampening, imageBytes } = {}) {
  // Prefer bytes the content script already extracted locally from the
  // rendered <img> element (zero network requests) over fetching the URL
  // ourselves. This matters beyond raw efficiency: if network access is
  // ever restricted after the page has loaded (e.g. a locked-down
  // evaluation environment), classification can still proceed as long as
  // the image was already visible on the page -- fetchImageBytes() below
  // is a fallback for when the content script couldn't read the pixels
  // locally (tainted canvas on a cross-origin image without CORS), not
  // the primary path.
  const buf = imageBytes instanceof ArrayBuffer ? imageBytes : await fetchImageBytes(url);
  const blob = new Blob([buf]);

  const [classifier, metaResult, rawImage] = await Promise.all([
    getClassifier(),
    Promise.resolve(scanImageMetadata(buf)),
    RawImage.fromBlob(blob),
  ]);

  const scores = await classifier(rawImage, { topk: null });
  let aiProb = scoresToAiProbability(scores);

  // Weak, bounded pixel-statistics nudge (noise-residual/edge/entropy) --
  // cheap enough to run unconditionally on every image (pure arithmetic
  // over a subsampled pixel grid, no model). Applied as a small logit-space
  // calibration adjustment to the classifier's own probability -- it can
  // shift the model's confidence a little, never flip its verdict outright
  // (bounded to +-0.06 pre-scaling, so at most a modest logit shift).
  const pixelSignal = computePixelSignals(rawImage);
  aiProb = applyPixelAdjustment(aiProb, pixelSignal.adjustment);

  // The person-dampener is opt-in (see README/popup) and, even when
  // enabled, only actually invoked when it could plausibly change the
  // outcome: skipped entirely whenever a generator fingerprint already
  // fired (dampening wouldn't apply anyway, see below) or the classifier
  // is already leaning "real" on its own (aiProb < 0.4 -- dampening it
  // further isn't going to move the needle). This keeps the expensive
  // model out of the hot path for the common case and only pays its cost
  // on the images it actually stands to help: exactly the
  // confidently-wrong portrait-photo cases.
  const domainSignal = scoreSourceDomain(hostname);
  const shouldCheckPerson = enablePersonDampening && !metaResult.isLikelyAI && aiProb >= 0.4;
  const personSignal = shouldCheckPerson
    ? await detectDominantPerson(rawImage)
    : { hasDominantPerson: false, personFraction: 0 };

  // Combine the classifier probability with every heuristic signal.
  // Delegated to lib/combine-signals.js so the exact same formula is used
  // here and in scripts/benchmark.mjs -- see that file for the full
  // reasoning behind the combination order (generator fingerprints
  // override; real-camera/person/domain signals proportionally dampen).
  const combined = combineSignals(aiProb, { metaResult, personSignal, domainSignal, tierThreshold });

  return {
    ...combined, // verdict, confidencePct, aiProbability, classifierAiProbability
    metadataSignals: metaResult.signals,
    realCameraSignals: metaResult.realSignals,
    personFraction: personSignal.personFraction,
    dampedForDominantPerson: personSignal.hasDominantPerson && !metaResult.isLikelyAI,
    domainTier: domainSignal.tier,
    pixelSignals: pixelSignal.signals,
    device: modelDevice,
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.target !== 'offscreen') return undefined;

  if (message.type === 'GET_STATUS') {
    sendResponse({ ok: true, status: lastStatus });
    return true;
  }

  if (message.type === 'WARM_UP') {
    getClassifier().then(() => sendResponse({ ok: true })).catch((err) =>
      sendResponse({ ok: false, error: String(err && err.message ? err.message : err) })
    );
    return true;
  }

  if (message.type === 'CLASSIFY') {
    classifyOne(message.url, {
      hostname: message.hostname,
      tierThreshold: message.tierThreshold,
      enablePersonDampening: message.enablePersonDampening,
      imageBytes: message.imageBytes,
    })
      .then((result) => sendResponse({ ok: true, requestId: message.requestId, result }))
      .catch((err) =>
        sendResponse({
          ok: false,
          requestId: message.requestId,
          error: String(err && err.message ? err.message : err),
        })
      );
    return true; // keep the message channel open for the async response
  }

  return undefined;
});
