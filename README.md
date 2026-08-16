AI Image Detector

Privacy-first Chrome extension for detecting AI-generated images directly in the browser.

The extension performs all analysis locally and does not upload images to external servers.

Features

- Local browser inference
- AI image detection using ONNX/Transformers.js
- Metadata analysis
- C2PA / Content Credentials detection
- Camera EXIF verification
- Domain reputation scoring
- Portrait-aware confidence adjustment
- Three-state classification:
  - Likely AI
  - Uncertain
  - Likely Real
- Google Images support
- Bing Images support
- No cloud APIs
- No image uploads
- Privacy-focused architecture

---

How It Works

The detector combines multiple signals instead of relying solely on a single AI classifier.

Visual Model

A local AI model analyzes image content and produces an AI probability score.

Metadata Analysis

The extension checks for:

- Stable Diffusion metadata
- ComfyUI metadata
- Generated image markers
- Camera EXIF information

C2PA Detection

When available, Content Credentials and C2PA information are used as strong evidence.

Domain Reputation

Known trusted sources and known AI-content sources contribute to the final score.

Portrait-Aware Scoring

Portrait images are treated differently to reduce false positives on professional photography and public figures.

Signal Fusion

All evidence is combined into a final confidence score and verdict.

---

Verdict Types

Likely AI

Strong evidence that the image was generated or significantly synthesized.

Uncertain

Insufficient evidence to confidently classify the image.

Likely Real

Strong evidence supporting a real photograph or camera-captured image.

---

Privacy

All processing happens locally inside the browser.

The extension:

- Does not upload images
- Does not require an account
- Does not send image data to external APIs
- Does not require a backend server

---

Installation

1. Download or clone this repository.
2. Open Chrome.
3. Navigate to:

chrome://extensions

4. Enable Developer Mode.
5. Click Load unpacked.
6. Select the extension folder.

The extension is now installed.

---

Limitations

No AI image detector is perfectly accurate.

Modern image generation models continue to improve, and some real photographs may occasionally be flagged incorrectly while some AI-generated images may evade detection.

Results should be interpreted as confidence estimates rather than absolute proof.

---

Project Goals

- Privacy-first image analysis
- Browser-native inference
- Explainable scoring
- Reduced false positives
- Support for modern AI-generated content

---

License

MIT License
