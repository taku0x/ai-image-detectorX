// model-config.js
//
// Single source of truth for model identifiers and tunable thresholds,
// imported by both offscreen.js (the shipped extension) and
// scripts/benchmark.mjs (the offline accuracy tool). Keeping these in one
// place means "the model we benchmark" and "the model we ship" can never
// silently drift apart from a forgotten edit in one file but not the
// other.

// General-purpose real-vs-AI image classifier. See README "Model history"
// for why this specific model was chosen over two earlier attempts.
export const MODEL_ID = 'amrita-detectly/detect-ai-image-v1';

// Secondary, opt-in object detector used only to check whether a human
// figure fills most of the frame (portrait-dampening signal). See
// offscreen.js and README for the real performance/accuracy tradeoffs
// this specific model carries.
export const PERSON_DETECTOR_ID = 'Xenova/yolos-tiny';
export const PERSON_DETECTOR_THRESHOLD = 0.6; // box confidence
export const PERSON_DOMINANT_FRACTION = 0.35; // person's box covers >=35% of frame
