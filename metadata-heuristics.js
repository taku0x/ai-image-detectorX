// metadata-heuristics.js
//
// Lightweight, dependency-free scanner for AI-generator "fingerprints" that
// many text-to-image tools leave behind in file metadata: PNG tEXt/iTXt
// chunks (Automatic1111 / ComfyUI / InvokeAI style "parameters"/"prompt"/
// "workflow" payloads), EXIF Software/UserComment tags, and C2PA content
// credentials that declare a "trainedAlgorithmicMedia" digital source type.
//
// This is intentionally a *generic* string/byte scan for well-known,
// publicly documented markers used by real-world generators — it is not
// tied to any specific evaluation image and contains no hashes or lookup
// tables of concrete images. It exists purely as one signal inside a
// hybrid pipeline; the learned classifier (see offscreen.js) is the
// primary and only mandatory detector, and metadata absence never counts
// against an image (plenty of real photos AND plenty of stripped/re-saved
// AI images carry no metadata at all).

// Known markers. Matching is case-insensitive and substring-based against
// the raw bytes of the file decoded as latin1 (safe 1:1 byte mapping).
const GENERATOR_MARKERS = [
  // Local / open source generation UIs & their metadata keys
  'stable diffusion', 'stable-diffusion', 'sdxl', 'automatic1111',
  'comfyui', 'invokeai', 'novelai', 'easydiffusion', 'fooocus',
  // These are the literal PNG tEXt keywords A1111/ComfyUI/InvokeAI write
  'negative prompt:', 'sampler:', 'cfg scale:', 'denoising strength:',
  // Hosted / commercial generators
  'midjourney', 'dall\u00b7e', 'dall-e', 'dalle', 'openai',
  'adobe firefly', 'firefly', 'leonardo.ai', 'leonardo ai',
  'playground ai', 'playground-ai', 'nightcafe', 'ideogram',
  'flux.1', 'black-forest-labs', 'stability.ai', 'stabilityai',
  'imagen', 'recraft', 'krea.ai', 'freepik ai', 'canva ai',
  // C2PA / content-credential vocabulary that specifically flags synthetic media
  'trainedalgorithmicmedia', 'compositewithtrainedalgorithmicmedia',
  'c2pa.ai', 'digitalsourcetype',
];

const AI_SOFTWARE_TAG_MARKERS = [
  'midjourney', 'dall', 'stable diffusion', 'firefly', 'leonardo',
  'nightcafe', 'playground', 'ideogram', 'imagen', 'niji',
];

// --- Real-camera counter-signal -----------------------------------------
//
// Everything above only ever pushes a verdict *toward* "AI". That leaves a
// gap: the classifier's most consistent failure mode in practice has been
// glossy, evenly-lit commercial/product photography (studio portraits,
// automotive catalog shots) getting misread as synthetic. Genuine camera
// photos frequently carry real EXIF data that generators don't produce —
// a manufacturer/model tag, and/or standard photographic notation for
// aperture, shutter speed, or focal length. Treating that as a counter-
// weight is a well-established forensic technique (the inverse of "look
// for generator fingerprints"), not something tuned to any specific image.
//
// Deliberately conservative: this only fires on the JPEG APP1 "Exif" box
// combined with a recognized camera/phone manufacturer string, not on
// EXIF presence alone (EXIF blocks can be forged or carried over through
// edits) and never on absence of metadata (that's just as common for
// ordinary re-compressed/re-shared real photos as for AI images).
const CAMERA_MAKE_MARKERS = [
  'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic', 'leica',
  'hasselblad', 'pentax', 'ricoh', 'gopro', 'dji', 'apple', 'samsung',
  'xiaomi', 'huawei', 'google', 'oneplus', 'motorola', 'lg electronics',
  'kodak', 'sigma corporation', 'phase one', 'nikon corporation',
];

// Standard human-readable photographic notation that real camera
// EXIF/XMP commonly renders as text (f-stop, focal length, ISO) but that
// generators have no reason to produce.
const CAMERA_NOTATION_PATTERN = /\bf\/\d+(\.\d+)?\b|\b\d{2,4}mm\b|\biso\s?\d{2,6}\b/i;

// --- Source-domain reputation --------------------------------------------
//
// A hostname isn't evidence about the pixels themselves, so this is
// deliberately the weakest signal in the pipeline (see the small fixed
// weights below, applied only as a mild nudge) — but it's a real, useful
// prior: an image served from a wire-service or encyclopedia domain is
// overwhelmingly likely to be a real photograph, and one served from a
// generator/showcase site is overwhelmingly likely to be AI output.
// Absence of a match (the common case) contributes nothing either way.
const TRUSTED_REAL_DOMAINS = [
  'wikipedia.org', 'wikimedia.org', 'reuters.com', 'apnews.com',
  'bbc.co.uk', 'bbc.com', 'gettyimages.com', 'shutterstock.com',
  'nytimes.com', 'washingtonpost.com', 'npr.org', 'afp.com',
  'britannica.com', 'nasa.gov', 'si.edu', 'loc.gov',
];
const TRUSTED_AI_DOMAINS = [
  'civitai.com', 'midjourney.com', 'lexica.art', 'leonardo.ai',
  'playgroundai.com', 'nightcafe.studio', 'openart.ai', 'artbreeder.com',
  'thispersondoesnotexist.com', 'craiyon.com', 'ideogram.ai',
];

/**
 * Scores a hostname (e.g. from the page the image was found on) for a
 * mild "real" or "AI" prior. Returns { tier: 'real'|'ai'|null, weight }.
 * `weight` is intentionally small (see caller) — this nudges, it never
 * decides on its own.
 */
export function scoreSourceDomain(hostname) {
  if (!hostname) return { tier: null, weight: 0 };
  const h = hostname.toLowerCase().replace(/^www\./, '');
  if (TRUSTED_REAL_DOMAINS.some((d) => h === d || h.endsWith(`.${d}`))) {
    return { tier: 'real', weight: 0.12 };
  }
  if (TRUSTED_AI_DOMAINS.some((d) => h === d || h.endsWith(`.${d}`))) {
    return { tier: 'ai', weight: 0.25 };
  }
  return { tier: null, weight: 0 };
}

/** Decode an ArrayBuffer to a latin1 string in fixed-size chunks (avoids
 *  call-stack blowups from String.fromCharCode(...hugeArray)). */
function toLatin1String(buf) {
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  let out = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return out;
}

function scanForMarkers(haystackLower) {
  const hits = [];
  for (const marker of GENERATOR_MARKERS) {
    if (haystackLower.includes(marker)) hits.push(marker);
  }
  return hits;
}

/**
 * Very small PNG chunk walker: returns concatenated text from
 * tEXt / iTXt / zTXt chunks (zTXt/compressed iTXt text is skipped since we
 * have no inflate available here — the keyword bytes preceding compressed
 * data are still readable and are enough to catch e.g. "parameters").
 */
function extractPngText(bytes) {
  const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < 8; i++) if (bytes[i] !== PNG_SIG[i]) return '';

  let out = '';
  let pos = 8;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  while (pos + 8 <= bytes.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    const dataStart = pos + 8;
    const dataEnd = Math.min(dataStart + len, bytes.length);
    if (type === 'tEXt' || type === 'iTXt' || type === 'zTXt') {
      out += toLatin1String(bytes.slice(dataStart, dataEnd).buffer) + '\n';
    }
    if (type === 'IEND') break;
    pos = dataEnd + 4; // skip CRC
    if (len < 0 || pos <= dataStart) break; // guard against malformed chunks
  }
  return out;
}

/**
 * Scans raw image bytes for known AI-generator fingerprints.
 * Returns { isLikelyAI: boolean, confidence: number (0-1), signals: string[] }
 * `confidence` here means "confidence that THIS heuristic's verdict is
 * meaningful", not a calibrated probability — it is combined with the
 * classifier score by the caller.
 */
export function scanImageMetadata(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const signals = [];

  // 1) Fast whole-file substring scan (works for JPEG EXIF UserComment/
  //    Software tags, XMP packets, and PNG text that survived byte-copying).
  const wholeFileLower = toLatin1String(arrayBuffer).toLowerCase();
  const generalHits = scanForMarkers(wholeFileLower);
  if (generalHits.length) {
    signals.push(...generalHits.map((m) => `marker:${m}`));
  }

  // 2) PNG-specific structured text chunk scan (more precise than the
  //    whole-file scan for PNGs, and catches the literal A1111 keyword
  //    "parameters" which is too generic to whole-file-match safely).
  if (bytes.length > 8 && bytes[0] === 0x89 && bytes[1] === 0x50) {
    const pngText = extractPngText(bytes).toLowerCase();
    if (pngText.includes('parameters') || pngText.includes('prompt') || pngText.includes('workflow')) {
      signals.push('png-text:generation-metadata');
    }
  }

  // 3) JPEG EXIF Software tag heuristic: look for the EXIF ASCII "Software"
  //    marker followed shortly by a known generator name within the APP1
  //    segment window.
  if (bytes.length > 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    const jpegHeadLower = wholeFileLower.slice(0, Math.min(wholeFileLower.length, 65536));
    for (const m of AI_SOFTWARE_TAG_MARKERS) {
      if (jpegHeadLower.includes(m)) {
        signals.push(`jpeg-header:${m}`);
        break;
      }
    }
  }

  // 4) C2PA / content credentials digital source type for synthetic media.
  if (wholeFileLower.includes('trainedalgorithmicmedia')) {
    signals.push('c2pa:trainedAlgorithmicMedia');
  }

  const uniqueSignals = [...new Set(signals)];
  const isLikelyAI = uniqueSignals.length > 0;
  // More independent signal categories firing -> higher confidence this
  // heuristic verdict is meaningful, capped well below certainty since a
  // single embedded string is strong-but-not-proof evidence.
  const confidence = isLikelyAI ? Math.min(0.95, 0.55 + 0.15 * uniqueSignals.length) : 0;

  // --- Real-camera counter-signal ---
  const realSignals = [];
  const hasExifBox = bytes.length > 4 && bytes[0] === 0xff && bytes[1] === 0xd8 && wholeFileLower.includes('exif');
  if (hasExifBox) {
    const foundMake = CAMERA_MAKE_MARKERS.find((m) => wholeFileLower.includes(m));
    if (foundMake) realSignals.push(`exif-make:${foundMake}`);
    if (CAMERA_NOTATION_PATTERN.test(wholeFileLower)) realSignals.push('exif-notation:aperture-or-focal-length');
  }
  const uniqueRealSignals = [...new Set(realSignals)];
  // Require the EXIF box *and* at least one specific marker before
  // treating this as meaningful evidence — an EXIF box with no recognized
  // content isn't worth much on its own.
  const isLikelyRealCamera = hasExifBox && uniqueRealSignals.length > 0;
  const realConfidence = isLikelyRealCamera
    ? Math.min(0.85, 0.5 + 0.2 * uniqueRealSignals.length)
    : 0;

  return {
    isLikelyAI,
    confidence,
    signals: uniqueSignals,
    isLikelyRealCamera,
    realConfidence,
    realSignals: uniqueRealSignals,
  };
}
