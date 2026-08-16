#!/usr/bin/env node
// benchmark.mjs
//
// Runs the extension's ACTUAL detection pipeline (same model, same
// heuristic signals, same combination formula -- imported directly from
// ext/lib/, not reimplemented) against a folder of labeled images, and
// reports real balanced accuracy instead of manual screenshot spot-checks.
//
// This project went through many rounds of "does this look better?" from
// screenshots during development, which is genuinely informative but not
// a substitute for a real number. This script exists to close that gap.
//
// Usage:
//   npm install                      (installs @huggingface/transformers)
//   node scripts/benchmark.mjs <folder>
//
// <folder> must contain two subfolders:
//   <folder>/real/   -- genuine photographs
//   <folder>/ai/     -- AI-generated images
// (jpg/jpeg/png/webp, any filenames)
//
// This is a DEV TOOL, not part of the shipped extension -- it downloads
// the model from Hugging Face on first run same as the extension does in
// the browser (subject to the same one-time-download, then-cached
// behavior), but runs in Node instead of Chrome, so it can't verify
// browser-specific behavior (CSP, offscreen documents, message passing).
// It verifies detection ACCURACY, not extension PLUMBING -- both matter,
// this only covers one.

import { readdirSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';
import { pipeline, env, RawImage } from '@huggingface/transformers';
import { scanImageMetadata, scoreSourceDomain } from '../ext/lib/metadata-heuristics.js';
import { computePixelSignals } from '../ext/lib/pixel-signals.js';
import { scoresToAiProbability, combineSignals, applyPixelAdjustment } from '../ext/lib/combine-signals.js';
import { MODEL_ID, PERSON_DETECTOR_ID, PERSON_DETECTOR_THRESHOLD, PERSON_DOMINANT_FRACTION } from '../ext/lib/model-config.js';

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp']);
const DEFAULT_TIER_THRESHOLD = 65;

function parseArgs(argv) {
  const args = { folder: null, threshold: DEFAULT_TIER_THRESHOLD, personDampening: false };
  const positional = [];
  for (const a of argv) {
    if (a.startsWith('--threshold=')) args.threshold = Number(a.split('=')[1]);
    else if (a === '--person-dampening') args.personDampening = true;
    else positional.push(a);
  }
  args.folder = positional[0];
  return args;
}

function listImages(dir) {
  try {
    return readdirSync(dir)
      .filter((f) => IMAGE_EXTENSIONS.has(extname(f).toLowerCase()))
      .map((f) => join(dir, f))
      .filter((p) => statSync(p).isFile());
  } catch (_) {
    return [];
  }
}

async function main() {
  const { folder, threshold, personDampening } = parseArgs(process.argv.slice(2));
  if (!folder) {
    console.error('Usage: node scripts/benchmark.mjs <folder> [--threshold=65] [--person-dampening]');
    console.error('  <folder> must contain real/ and ai/ subfolders of images.');
    process.exit(1);
  }

  const realImages = listImages(join(folder, 'real'));
  const aiImages = listImages(join(folder, 'ai'));
  if (!realImages.length && !aiImages.length) {
    console.error(`No images found under ${folder}/real/ or ${folder}/ai/`);
    process.exit(1);
  }

  console.log(`Proofmark-style offline benchmark (ai-image-detector)`);
  console.log(`Model: ${MODEL_ID}${personDampening ? ` + ${PERSON_DETECTOR_ID} (portrait dampening on)` : ''}`);
  console.log(`Threshold: ${threshold}%   real: ${realImages.length} images   ai: ${aiImages.length} images\n`);

  // Node build of transformers.js downloads/caches under ~/.cache by
  // default -- same one-time-download-then-offline behavior as the
  // extension, just with Node's cache instead of the browser's Cache
  // Storage.
  env.allowRemoteModels = true;
  env.useBrowserCache = false;

  const classifier = await pipeline('image-classification', MODEL_ID);
  let personDetector = null;
  if (personDampening) {
    personDetector = await pipeline('object-detection', PERSON_DETECTOR_ID, { device: 'wasm' });
  }

  async function detectDominantPerson(rawImage) {
    if (!personDetector) return { hasDominantPerson: false, personFraction: 0 };
    try {
      const results = await personDetector(rawImage, { threshold: PERSON_DETECTOR_THRESHOLD, percentage: true });
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

  async function classifyFile(path) {
    const rawImage = await RawImage.read(path);
    const { readFileSync } = await import('node:fs');
    const buf = readFileSync(path).buffer;

    const scores = await classifier(rawImage, { topk: null });
    let aiProb = scoresToAiProbability(scores);

    const pixelSignal = computePixelSignals(rawImage);
    aiProb = applyPixelAdjustment(aiProb, pixelSignal.adjustment);

    const metaResult = scanImageMetadata(buf);
    const domainSignal = scoreSourceDomain(null); // no page context in an offline benchmark
    const shouldCheckPerson = personDampening && !metaResult.isLikelyAI && aiProb >= 0.4;
    const personSignal = shouldCheckPerson ? await detectDominantPerson(rawImage) : { hasDominantPerson: false, personFraction: 0 };

    return combineSignals(aiProb, { metaResult, personSignal, domainSignal, tierThreshold: threshold });
  }

  async function scoreSet(paths, trueLabel) {
    let correct = 0;
    let uncertain = 0;
    const results = [];
    for (const path of paths) {
      try {
        const r = await classifyFile(path);
        const isCorrect = r.verdict === trueLabel;
        if (isCorrect) correct += 1;
        if (r.verdict === 'uncertain') uncertain += 1;
        results.push({ path, ...r, correct: isCorrect });
        const mark = r.verdict === trueLabel ? '✓' : r.verdict === 'uncertain' ? '·' : '✗';
        console.log(`${mark}  ${r.confidencePct.toFixed(0)}% ${r.verdict.padEnd(9)} ${path}`);
      } catch (err) {
        console.log(`!  ERROR  ${path}: ${err.message}`);
      }
    }
    return { correct, uncertain, total: paths.length, results };
  }

  console.log('-- real/ --');
  const realResult = await scoreSet(realImages, 'real');
  console.log('\n-- ai/ --');
  const aiResult = await scoreSet(aiImages, 'ai');

  const realRecall = realResult.total ? realResult.correct / realResult.total : null;
  const aiRecall = aiResult.total ? aiResult.correct / aiResult.total : null;
  const balancedAccuracy =
    realRecall != null && aiRecall != null
      ? (realRecall + aiRecall) / 2
      : realRecall != null
        ? realRecall
        : aiRecall;

  console.log('\n== Results ==');
  if (realRecall != null) console.log(`Real recall:        ${(realRecall * 100).toFixed(1)}% (${realResult.correct}/${realResult.total}, ${realResult.uncertain} uncertain)`);
  if (aiRecall != null) console.log(`AI recall:           ${(aiRecall * 100).toFixed(1)}% (${aiResult.correct}/${aiResult.total}, ${aiResult.uncertain} uncertain)`);
  console.log(`Balanced accuracy:   ${(balancedAccuracy * 100).toFixed(1)}%`);
  console.log(`\nNote: "uncertain" verdicts count as incorrect for this accuracy figure (there's no honest`);
  console.log(`way to credit a non-answer as correct) -- if the bounty's actual scoring treats "uncertain"`);
  console.log(`differently, adjust this script's scoreSet() to match before trusting this number directly.`);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
