// combine-signals.js
//
// Pure, environment-agnostic combination logic: takes plain data (already
// computed classifier scores + heuristic signal results) and produces the
// final three-tier verdict. No DOM, no chrome.*, no network -- this can run
// identically inside the browser extension (offscreen.js) or a Node CLI
// (scripts/benchmark.mjs), which matters beyond convenience: if the
// scoring formula only existed once, inline in offscreen.js, a benchmark
// script would have to re-implement it by hand and could silently drift
// out of sync with what actually ships. Both now import this file, so
// "what we benchmark" and "what we ship" are provably the same code.

const AI_LABEL_HINTS = ['fake', 'deepfake', 'ai', 'synthetic', 'generated', 'artificial'];
const REAL_LABEL_HINTS = ['real', 'realism', 'authentic', 'photo', 'genuine', 'human'];
const FACE_DAMPING_FACTOR = 0.6; // keep in sync with offscreen.js's constant of the same name
// ^ Deliberately aggressive, on request, after confirming via live testing
// that a milder 40% setting was correctly applied but still left most
// studio headshots in "Uncertain" rather than "Likely Real" -- at 60%,
// observed 98%-confidence false positives land around 39% ("Uncertain")
// and an already-borderline 66% case flips to ~26% ("Likely Real"). Full
// tuning history in scripts/tune-damping.md.
//
// REAL COST, not just a tuning footnote: this dampener can't distinguish
// "a real photo of a person" from "a genuinely AI-generated photo of a
// person" -- it only detects "a person fills the frame." A stronger cut
// helps the false-positive case (real portraits wrongly flagged) but
// applies equally to true positives (deepfakes, synthetic profile
// photos, fake identities) which are arguably the single most
// consequential category of AI image this extension could ever catch.
// 0.6 was chosen specifically to stop short of ever fully clearing a
// 90%+ confidence AI call to "Likely Real" outright (it still lands
// "Uncertain," not "Real") -- but it does meaningfully reduce
// sensitivity to real synthetic-portrait cases in the 60-85% confidence
// range. If this extension is ever positioned around catching fake
// profile photos / deepfakes specifically, this value should come back
// down.

export function labelLooksAI(label) {
  const l = String(label).toLowerCase();
  if (AI_LABEL_HINTS.some((h) => l.includes(h))) return true;
  if (REAL_LABEL_HINTS.some((h) => l.includes(h))) return false;
  return true; // unknown vocabulary: default to the "positive"/first class
}

/** Reduces a transformers.js image-classification pipeline's raw `scores`
 * array to a single AI-probability in [0, 1]. */
export function scoresToAiProbability(scores) {
  if (!Array.isArray(scores) || !scores.length) return 0.5;
  if (scores.length === 1) {
    return labelLooksAI(scores[0].label) ? scores[0].score : 1 - scores[0].score;
  }
  let aiMass = 0;
  let total = 0;
  for (const s of scores) {
    total += s.score;
    if (labelLooksAI(s.label)) aiMass += s.score;
  }
  return total > 0 ? aiMass / total : 0.5;
}

/**
 * Combines the classifier's AI probability with every heuristic signal
 * into a final three-tier verdict. Mirrors offscreen.js's classifyOne
 * combination logic exactly (that function now delegates here).
 *
 * @param {number} aiProb - classifier's raw AI probability (already
 *   pixel-signal-adjusted by the caller, see offscreen.js)
 * @param {object} metaResult - return value of scanImageMetadata()
 * @param {object} personSignal - { hasDominantPerson, personFraction }
 * @param {object} domainSignal - return value of scoreSourceDomain()
 * @param {number} [tierThreshold=65] - percent, from the popup's slider
 */
export function combineSignals(aiProb, { metaResult, personSignal, domainSignal, tierThreshold }) {
  let combinedAiProb = aiProb;
  if (metaResult.isLikelyAI) {
    combinedAiProb = 1 - (1 - aiProb) * (1 - metaResult.confidence);
  } else {
    if (metaResult.isLikelyRealCamera) {
      combinedAiProb *= 1 - metaResult.realConfidence;
    }
    if (personSignal && personSignal.hasDominantPerson) {
      combinedAiProb *= 1 - FACE_DAMPING_FACTOR;
    }
    if (domainSignal && domainSignal.tier === 'real') {
      combinedAiProb *= 1 - domainSignal.weight;
    } else if (domainSignal && domainSignal.tier === 'ai') {
      combinedAiProb = 1 - (1 - combinedAiProb) * (1 - domainSignal.weight);
    }
  }
  combinedAiProb = Math.min(1, Math.max(0, combinedAiProb));

  const t = typeof tierThreshold === 'number' ? tierThreshold / 100 : 0.65;
  let verdict;
  if (combinedAiProb >= t) verdict = 'ai';
  else if (combinedAiProb <= 1 - t) verdict = 'real';
  else verdict = 'uncertain';

  const rawConfidence = verdict === 'ai' ? combinedAiProb : verdict === 'real' ? 1 - combinedAiProb : 0.5;
  const displayConfidence = Math.min(0.98, Math.max(0.5, rawConfidence));

  return {
    verdict,
    confidencePct: Math.round(displayConfidence * 1000) / 10,
    aiProbability: combinedAiProb,
    classifierAiProbability: aiProb,
  };
}

/** Applies the weak pixel-statistics adjustment in logit space. Shared so
 * offscreen.js and the benchmark CLI apply it identically. */
export function applyPixelAdjustment(aiProb, adjustment) {
  if (!adjustment) return aiProb;
  const clamped = Math.min(0.999, Math.max(0.001, aiProb));
  const shifted = Math.log(clamped / (1 - clamped)) + adjustment * 4;
  return 1 / (1 + Math.exp(-shifted));
}
