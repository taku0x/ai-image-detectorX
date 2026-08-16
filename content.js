// content.js — runs on every page. Finds <img> elements, asks the
// background/offscreen pipeline to classify them, and overlays a small
// floating confidence badge. Never touches the page's own DOM structure
// (badges are appended to <body> and positioned with `position: fixed`)
// so it can't break page layout/CSS.

(() => {
  const MIN_RENDER_SIZE = 64; // px — skip icons/avatars/spacer gifs
  const MAX_CONCURRENT = 2; // offscreen inference is effectively single-threaded

  let enabled = true;
  let threshold = 65;

  const processed = new WeakSet();
  const badges = new Map(); // img -> badge element
  const queue = [];
  let inFlight = 0;

  // Many sites (Google Images results very much included) serve a small,
  // heavily recompressed proxy/thumbnail as the <img> src, while the true
  // full-resolution source is available nearby: a larger candidate in
  // `srcset`, a lazy-load `data-src`/`data-original` attribute, or an
  // enclosing <a> that links straight to the original image file. Feeding
  // the classifier a recompressed thumbnail instead of the original can
  // introduce its own artifacts and hurt accuracy, so we prefer the best
  // available source whenever we can find one.
  function bestImageUrl(img) {
    const candidates = [];

    if (img.srcset) {
      const parsed = img.srcset
        .split(',')
        .map((entry) => entry.trim().split(/\s+/))
        .map(([url, descriptor]) => ({
          url,
          width: descriptor && descriptor.endsWith('w') ? parseInt(descriptor, 10) : 0,
        }))
        .filter((c) => c.url);
      if (parsed.length) {
        parsed.sort((a, b) => b.width - a.width);
        candidates.push(parsed[0].url);
      }
    }

    for (const attr of ['data-src', 'data-original', 'data-lazy-src', 'data-srcset']) {
      const v = img.getAttribute(attr);
      if (v) candidates.push(v.split(',')[0].trim().split(/\s+/)[0]);
    }

    const link = img.closest('a[href]');
    if (link) {
      // Google Images wraps result thumbnails in a link to
      // google.com/imgres?...&imgurl=<original full-res URL>&... — that
      // original is far better classifier input than the gstatic.com
      // thumbnail proxy actually sitting in the <img> tag.
      try {
        const linkUrl = new URL(link.href, document.baseURI);
        const imgurl = linkUrl.searchParams.get('imgurl');
        if (imgurl) candidates.push(imgurl);
      } catch (_) {
        /* ignore malformed link URLs */
      }
      if (/\.(jpe?g|png|webp|gif|avif)(\?|#|$)/i.test(link.href)) {
        candidates.push(link.href);
      }
    }

    candidates.push(img.currentSrc || img.src || '');

    // Resolve relative URLs against the document and return the first
    // usable (non-empty) absolute candidate.
    for (const c of candidates) {
      if (!c) continue;
      try {
        return new URL(c, document.baseURI).href;
      } catch (_) {
        continue;
      }
    }
    return img.currentSrc || img.src || '';
  }

  function isEligible(img) {
    if (processed.has(img)) return false;
    const src = bestImageUrl(img);
    if (!/^https?:/i.test(src) && !src.startsWith('data:')) return false;
    const w = img.naturalWidth || img.width || 0;
    const h = img.naturalHeight || img.height || 0;
    if (w && h && (w < MIN_RENDER_SIZE || h < MIN_RENDER_SIZE)) return false;
    return true;
  }

  function enqueue(img) {
    if (processed.has(img)) return;
    processed.add(img);
    queue.push(img);
    pump();
  }

  function pump() {
    while (inFlight < MAX_CONCURRENT && queue.length) {
      const img = queue.shift();
      inFlight += 1;
      classify(img).finally(() => {
        inFlight -= 1;
        pump();
      });
    }
  }

  /**
   * Tries to read the image's bytes directly from the already-rendered
   * <img> element via canvas, with zero additional network requests --
   * the browser already downloaded and decoded this image to display it
   * on the page, so re-fetching it separately (as the fallback path
   * below does) is both wasteful and a real liability if network access
   * is ever restricted after the initial page load (e.g. a benchmark
   * harness that locks down connectivity once test pages are loaded).
   *
   * This only succeeds for same-origin, data:, or blob: images, or
   * cross-origin images the page explicitly loaded with
   * crossorigin="anonymous" AND a permissive CORS response -- anything
   * else taints the canvas and throws, which we treat as "try the
   * network fallback instead," not an error.
   */
  async function tryExtractImageBytesLocally(img) {
    try {
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (!w || !h) return null;
      const canvas = new OffscreenCanvas(w, h);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const blob = await canvas.convertToBlob({ type: 'image/png' });
      return await blob.arrayBuffer();
    } catch (_) {
      return null; // tainted canvas or other failure -- fall back to fetch
    }
  }

  function classify(img) {
    setBadge(img, { loading: true });
    const resolvedUrl = bestImageUrl(img);
    // Domain reputation should reflect where the *image* actually came
    // from, not the page you're browsing -- on a Google Images results
    // page, location.hostname is always "google.com" (not in either
    // trusted list), which silently made this signal permanently
    // impossible to trigger. bestImageUrl() already resolves through to
    // the original source (including unwrapping Google's imgurl= param),
    // so use that host instead.
    let imageHostname = location.hostname;
    try {
      imageHostname = new URL(resolvedUrl, document.baseURI).hostname || location.hostname;
    } catch (_) {
      /* data: URLs etc. have no hostname; fall back to the page's */
    }
    return tryExtractImageBytesLocally(img).then((localBytes) => {
      const payload = { type: 'CLASSIFY_IMAGE', hostname: imageHostname };
      if (localBytes) {
        // Zero-network path: hand the already-decoded bytes straight
        // through. structured-clone messaging supports ArrayBuffer
        // directly, no base64/serialization needed.
        payload.imageBytes = localBytes;
        payload.url = resolvedUrl; // kept for logging/tooltip purposes only
      } else {
        payload.url = resolvedUrl; // network fallback: offscreen.js will fetch() this
      }
      return chrome.runtime.sendMessage(payload);
    }).then((res) => {
        if (!res) return;
        if (res.ok) {
          setBadge(img, { result: res.result });
        } else if (res.error && res.error !== 'disabled') {
          removeBadge(img);
        } else {
          removeBadge(img);
        }
      })
      .catch(() => removeBadge(img));
  }

  // --- Badge rendering -----------------------------------------------

  function ensureBadge(img) {
    let badge = badges.get(img);
    if (badge) return badge;
    badge = document.createElement('div');
    badge.className = 'aidet-badge aidet-loading';
    badge.textContent = '…';
    document.body.appendChild(badge);
    badges.set(img, badge);
    positionBadge(img, badge);
    return badge;
  }

  function setBadge(img, { loading, result }) {
    if (!document.body.contains(img)) {
      removeBadge(img);
      return;
    }
    const badge = ensureBadge(img);
    if (loading) {
      badge.className = 'aidet-badge aidet-loading';
      badge.textContent = '…';
      return;
    }
    if (!result) {
      removeBadge(img);
      return;
    }
    // Three-tier verdict: 'ai' | 'uncertain' | 'real'. Most everyday
    // images should land in "uncertain" rather than a confident guess --
    // a detector willing to say "I don't know" is more trustworthy than
    // one that always picks a side. Confidence is never shown for
    // "uncertain" (there's nothing meaningful to report) and is otherwise
    // already capped well under 100% by offscreen.js.
    const { verdict, confidencePct } = result;
    if (verdict === 'ai') {
      badge.className = 'aidet-badge aidet-ai';
      badge.textContent = `Likely AI ${confidencePct.toFixed(0)}%`;
    } else if (verdict === 'real') {
      badge.className = 'aidet-badge aidet-real';
      badge.textContent = `Likely Real ${confidencePct.toFixed(0)}%`;
    } else {
      badge.className = 'aidet-badge aidet-uncertain';
      badge.textContent = 'Uncertain';
    }
    badge.title = buildTooltip(result);
  }

  function buildTooltip(result) {
    const verdictLabel =
      result.verdict === 'ai' ? 'Likely AI-generated' : result.verdict === 'real' ? 'Likely real' : 'Uncertain — no confident signal either way';
    const lines = [`Verdict: ${verdictLabel}`];
    if (result.verdict !== 'uncertain') {
      lines.push(`Confidence: ${result.confidencePct.toFixed(1)}%`);
    }
    lines.push(`Model device: ${result.device}`);
    if (result.metadataSignals && result.metadataSignals.length) {
      lines.push(`AI-generator signals: ${result.metadataSignals.join(', ')}`);
    }
    if (result.realCameraSignals && result.realCameraSignals.length) {
      lines.push(`Real-camera signals: ${result.realCameraSignals.join(', ')}`);
    }
    if (result.dampedForDominantPerson) {
      lines.push('Dampened: dominant human subject in frame (portrait-style)');
    }
    if (result.domainTier) {
      lines.push(`Source-domain prior: ${result.domainTier === 'real' ? 'trusted real-photo source' : 'trusted AI-showcase source'}`);
    }
    if (result.pixelSignals && result.pixelSignals.length) {
      lines.push(`Pixel-statistics signals: ${result.pixelSignals.join(', ')}`);
    }
    return lines.join('\n');
  }

  function removeBadge(img) {
    const badge = badges.get(img);
    if (badge) {
      badge.remove();
      badges.delete(img);
    }
  }

  function positionBadge(img, badge) {
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0 || rect.bottom < 0 || rect.top > window.innerHeight) {
      badge.style.display = 'none';
      return;
    }
    badge.style.display = 'block';
    const top = Math.max(0, rect.top) + 4;
    const left = Math.min(window.innerWidth - 8, rect.left + rect.width) - 8;
    badge.style.top = `${top}px`;
    badge.style.left = `${left}px`;
    badge.style.transform = 'translateX(-100%)';
  }

  let rafScheduled = false;
  function repositionAll() {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
      rafScheduled = false;
      for (const [img, badge] of badges) {
        if (!document.body.contains(img)) {
          removeBadge(img);
          continue;
        }
        positionBadge(img, badge);
      }
    });
  }
  window.addEventListener('scroll', repositionAll, { passive: true, capture: true });
  window.addEventListener('resize', repositionAll, { passive: true });

  // --- Discovery: IntersectionObserver so we only analyze visible images

  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && enabled) {
          const img = entry.target;
          io.unobserve(img);
          if (isEligible(img)) enqueue(img);
        }
      }
    },
    { rootMargin: '200px' }
  );

  function observeImg(img) {
    if (img.__aidetObserved) return;
    img.__aidetObserved = true;
    if (img.complete && (img.naturalWidth || img.naturalHeight)) {
      io.observe(img);
    } else {
      img.addEventListener(
        'load',
        () => io.observe(img),
        { once: true }
      );
    }
  }

  function scanAll(root = document) {
    if (!enabled) return;
    root.querySelectorAll('img').forEach(observeImg);
  }

  const mo = new MutationObserver((mutations) => {
    if (!enabled) return;
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (!(node instanceof Element)) continue;
        if (node.tagName === 'IMG') observeImg(node);
        else node.querySelectorAll?.('img').forEach(observeImg);
      }
    }
  });

  function start() {
    scanAll();
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  function clearAllBadges() {
    for (const img of Array.from(badges.keys())) removeBadge(img);
  }

  // --- Settings sync ----------------------------------------------------

  chrome.runtime
    .sendMessage({ type: 'GET_STATE' })
    .then((s) => {
      if (s) {
        enabled = s.enabled;
        threshold = s.threshold;
      }
      if (enabled) start();
    })
    .catch(() => start());

  chrome.runtime.onMessage.addListener((message) => {
    if (message && message.type === 'SETTINGS_CHANGED') {
      const wasEnabled = enabled;
      enabled = message.enabled;
      threshold = message.threshold;
      if (enabled && !wasEnabled) start();
      if (!enabled && wasEnabled) clearAllBadges();
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {}, { once: true });
  }
})();
