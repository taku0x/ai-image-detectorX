// pixel-signals.js
//
// Weak, bounded pixel-level statistics that nudge the final score but can
// never decide it alone. This is deliberately NOT an attempt at the
// "detect malformed hands / impossible reflections" idea that was
// considered and rejected earlier in this project (see README "Known
// limitations") — that requires its own trained model and isn't
// realistically hand-codable. This is a much smaller claim: cheap,
// general image statistics (local noise-residual smoothness, edge
// density, saturation, luminance entropy) that correlate *weakly* with
// common AI-generation artifacts (over-smooth texture, unusual color
// distributions) but are also produced by ordinary JPEG compression,
// blur, and editing -- so the adjustment this contributes is small and
// bounded, same spirit as the real-camera/domain-reputation signals
// elsewhere in this pipeline.
//
// Input: an ImageData-like object ({ data: Uint8ClampedArray, width,
// height }) such as what a canvas 2D context's getImageData() returns.

function luminanceAt(data, width, x, y, channels) {
  const i = (y * width + x) * channels;
  return 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
}

/**
 * Computes weak pixel-level statistics and a small bounded score
 * adjustment (added in logit space by the caller, same pattern as the
 * other heuristic signals in this pipeline).
 *
 * Accepts either a canvas-style ImageData ({ data, width, height }, always
 * 4-channel RGBA) or a transformers.js RawImage-shaped object ({ data,
 * width, height, channels }, which may be 3-channel RGB) -- `channels` is
 * inferred from data.length when not given explicitly.
 *
 * Returns { adjustment: number (-0.04..0.06), signals: string[] }.
 */
export function computePixelSignals(imageData) {
  const { data, width, height } = imageData || {};
  if (!data || !width || !height || width < 8 || height < 8) {
    return { adjustment: 0, signals: [] };
  }
  const channels = imageData.channels || Math.max(1, Math.round(data.length / (width * height)));
  if (channels < 3) return { adjustment: 0, signals: [] }; // grayscale: not enough signal to bother

  let residualTotal = 0;
  let residualSquared = 0;
  let edgeTotal = 0;
  let saturationCount = 0;
  let samples = 0;
  const histogram = new Uint32Array(32);

  // Sample every other pixel (both axes) to keep this cheap on large
  // images -- this is a coarse statistical signal, not something that
  // needs full-resolution precision.
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const i = (y * width + x) * channels;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      histogram[Math.min(31, Math.floor(luminance / 8))] += 1;

      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      if (max - min > 150 && max > 210) saturationCount += 1;

      const left = luminanceAt(data, width, x - 1, y, channels);
      const right = luminanceAt(data, width, x + 1, y, channels);
      const up = luminanceAt(data, width, x, y - 1, channels);
      const down = luminanceAt(data, width, x, y + 1, channels);
      const localMean = (left + right + up + down) / 4;
      const residual = Math.abs(luminance - localMean);
      residualTotal += residual;
      residualSquared += residual * residual;
      edgeTotal += Math.abs(right - left) + Math.abs(down - up);
      samples += 1;
    }
  }

  if (!samples) return { adjustment: 0, signals: [] };

  let luminanceEntropy = 0;
  for (const count of histogram) {
    if (!count) continue;
    const p = count / samples;
    luminanceEntropy -= p * Math.log2(p);
  }

  const residualMean = residualTotal / samples;
  const residualVariance = residualSquared / samples - residualMean * residualMean;
  const edgeDensity = edgeTotal / samples / 510;
  const saturation = saturationCount / samples;

  const signals = [];
  let adjustment = 0;

  // Over-smooth texture in high-detail (high-edge) regions: common in
  // diffusion-model output, but also produced by heavy JPEG compression
  // or noise-reduction -- hence the small weight.
  if (residualMean < 4.2 && edgeDensity > 0.055) {
    adjustment += 0.025;
    signals.push('smooth-high-detail-texture');
  }
  // Irregular synthetic-looking noise residual with otherwise low edge
  // content.
  if (residualVariance > 230 && edgeDensity < 0.045) {
    adjustment += 0.02;
    signals.push('irregular-texture-residual');
  }
  // Unusually saturated, high-entropy color distribution -- weakly
  // associated with some generator styles, also true of ordinary
  // vibrant/stylized real photography, hence small weight only.
  if (saturation > 0.13 && luminanceEntropy > 4.4) {
    adjustment += 0.015;
    signals.push('high-chroma-distribution');
  }
  // The inverse case: real camera sensor noise is well-documented to
  // produce moderate residual noise alongside genuine high-frequency
  // detail. This is a small *negative* adjustment (push toward real).
  if (residualMean > 13 && edgeDensity > 0.12) {
    adjustment -= 0.02;
    signals.push('camera-like-sensor-texture');
  }

  // Same bound as the source this was adapted from: never a large enough
  // swing to decide a verdict by itself.
  adjustment = Math.min(0.06, Math.max(-0.04, adjustment));

  return { adjustment, signals };
}
