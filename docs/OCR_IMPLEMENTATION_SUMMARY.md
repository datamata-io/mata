# OCR Implementation Summary — MATA v1.9.0

> Introduced in **v1.9.0**. Optical Character Recognition is a first-class task alongside
> detection, classification, segmentation, depth, and tracking.

---

## Architecture Diagram

```
mata.load("ocr", ...) / mata.run("ocr", ...)
        │
        ▼
   api.py  ─────────────────── UniversalLoader
        │                    (5-strategy detection)
        │                             │
        │          ┌──────────────────┼──────────────────────┐
        │          ▼                  ▼                       ▼
        │    Config alias        HuggingFace ID          Local file
        │          └──────────────────┼───────────────────────┘
        │                             ▼
        │                    OCR Adapter selection
        │          ┌──────────────────┼──────────────────────┐
        ▼          ▼                  ▼                       ▼
  HuggingFaceOCRAdapter    EasyOCRAdapter   PaddleOCRAdapter  TesseractAdapter
  (GOT-OCR2 / TrOCR)     (80+ languages)  (80+ languages)   (system binary)
        │                       │                │                  │
        └───────────────────────┴────────────────┴──────────────────┘
                                         │
                                         ▼
                                    OCRResult
                              (regions: list[TextRegion]
                               full_text: str,  meta: dict)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                  Export System                  Graph System
            (.json / .csv / .txt /             OCRWrapper
             .png/.jpg overlay)                     │
                                                    ▼
                                               OCRText artifact
                                         (text_blocks, full_text,
                                          instance_ids, meta)
```

---

## Supported Backends

| Backend       | Class                   | Models / Source               | Runtime                             | Bbox support         | GPU                   | Notes                                                      |
| ------------- | ----------------------- | ----------------------------- | ----------------------------------- | -------------------- | --------------------- | ---------------------------------------------------------- |
| **GOT-OCR2**  | `HuggingFaceOCRAdapter` | `stepfun-ai/GOT-OCR-2.0-hf`   | PyTorch (AutoModelForCausalLM)      | ❌ (whole-image)     | ✅ via `device`       | ⚠️ Known hallucination issues — avoid until further notice |
| **TrOCR**     | `HuggingFaceOCRAdapter` | `microsoft/trocr-*`           | PyTorch (VisionEncoderDecoderModel) | ❌ (whole-image)     | ✅ via `device`       | Single text-line crops only                                |
| **EasyOCR**   | `EasyOCRAdapter`        | local engine (80+ languages)  | PyTorch (internal)                  | ✅ xyxy polygon→bbox | ✅ via `gpu=True`     |                                                            |
| **PaddleOCR** | `PaddleOCRAdapter`      | local engine (80+ languages)  | PaddlePaddle (internal)             | ✅ xyxy polygon→bbox | ✅ via `use_gpu=True` | paddleocr + paddlepaddle major versions must match         |
| **Tesseract** | `TesseractAdapter`      | system binary via pytesseract | External process                    | ✅ xyxy (x+w, y+h)   | ❌                    |                                                            |

### Installation

```bash
# HuggingFace models (TrOCR)  — see GOT-OCR2 warning in Known Limitations
pip install transformers accelerate

# EasyOCR
pip install easyocr
# or: pip install mata[ocr]

# PaddleOCR — IMPORTANT: paddleocr and paddlepaddle MAJOR versions must match.
# paddleocr 3.x requires paddlepaddle 3.x (mismatched installs cause native crashes).
pip install paddleocr paddlepaddle                 # CPU (latest)

# GPU — choose the wheel matching your CUDA version:
pip install paddleocr
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/  # CUDA 11.8
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/  # CUDA 12.3
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/  # CUDA 12.6 (recommended)
# Wheel is ~1–1.5 GB.  See https://www.paddlepaddle.org.cn/install/quick for all options.
# or: pip install mata[ocr-paddle]  (installs CPU wheel only)

# Tesseract
pip install pytesseract
# or: pip install mata[ocr-tesseract]
# + system binary:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
#   Windows:       https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Result Types

### `OCRResult` (core type — returned by every adapter)

```python
@dataclass(frozen=True)
class TextRegion:
    text:  str
    score: float                                      # [0.0, 1.0]
    bbox:  tuple[float, float, float, float] | None  # xyxy absolute pixels
    label: str | None                                # e.g. "en", "handwritten"

@dataclass(frozen=True)
class OCRResult:
    regions: list[TextRegion]
    meta:    dict[str, Any]

    # Convenience
    full_text: str                                   # "\n".join(region.text …)
    filter_by_score(threshold) -> OCRResult
    to_dict() / to_json() / from_dict() / from_json()
    save(output_path)                                # auto-format by extension
```

### `OCRText` (graph artifact — produced by `OCRWrapper` / `OCR` node)

```python
@dataclass(frozen=True)
class TextBlock:
    text:       str
    confidence: float
    bbox:       tuple[float, float, float, float] | None
    language:   str | None

@dataclass(frozen=True)
class OCRText(Artifact):
    text_blocks:  tuple[TextBlock, ...]
    full_text:    str
    instance_ids: tuple[str, ...]   # aligns with ROI instance IDs for Fuse node
    meta:         dict[str, Any]
```

The separation between `OCRResult` (thin, serializable, adapter-facing) and `OCRText`
(graph-wired, ROI-correlatable, artifact-protocol) is deliberate — it mirrors the
pattern used for detection (`VisionResult` vs `Detections` artifact).

---

## Public API

```python
import mata

# Load an OCR adapter
adapter = mata.load("ocr", "easyocr")
adapter = mata.load("ocr", "paddleocr", lang="zh")
adapter = mata.load("ocr", "stepfun-ai/GOT-OCR-2.0-hf")   # HuggingFace ID
adapter = mata.load("ocr", "microsoft/trocr-base-handwritten")
adapter = mata.load("ocr", "tesseract", lang="eng+fra")
adapter = mata.load("ocr", "my-alias")                      # config alias

# Run OCR directly
result = mata.run("ocr", "document.jpg", model="easyocr")
result = mata.run("ocr", "document.jpg", model="stepfun-ai/GOT-OCR-2.0-hf")

# Work with results
print(result.full_text)
for region in result.regions:
    print(f"{region.text!r}  conf={region.score:.2f}  bbox={region.bbox}")

# Filter by confidence
high_conf = result.filter_by_score(0.85)

# Export
result.save("output.json")     # structured JSON
result.save("output.csv")      # CSV: text, score, x1, y1, x2, y2
result.save("output.txt")      # plain text
result.save("overlay.png")     # image with bbox overlays
```

---

## Graph Composition Recipes

### Pattern 1: Standalone OCR

```python
from mata.nodes import OCR
from mata.core.graph import Graph

# Simple end-to-end OCR on a full image
graph = Graph([
    OCR(using="ocr_engine", out="text"),
])

result = mata.infer(graph, image="document.jpg", providers={
    "ocr_engine": mata.load("ocr", "easyocr"),
})
print(result["text"].full_text)
```

### Pattern 2: Detect → Extract ROIs → OCR Pipeline

```python
from mata.nodes import Detect, ExtractROIs, OCR, Fuse

# Detect signs/license plates, then read text from each crop
graph = (
    Detect(using="detector", out="dets")
    >> Filter(src="dets", label_in=["sign", "license_plate"], out="filtered")
    >> ExtractROIs(src_dets="filtered", out="rois")
    >> OCR(using="ocr_engine", src="rois", out="ocr_result")
    >> Fuse(out="final", dets="filtered", ocr="ocr_result")
)

result = mata.infer(graph, image="street.jpg", providers={
    "detector":  mata.load("detect", "facebook/detr-resnet-50"),
    "ocr_engine": mata.load("ocr", "easyocr"),
})

for item in result["final"].items:
    print(f"{item.label}: {item.ocr_text!r}")
```

When `OCR` receives a `ROIs` artifact each crop is processed independently and
`OCRText.instance_ids` aligns one-to-one with the source ROI IDs, enabling the
`Fuse` node to correlate detections with their extracted text by ID.

### Pattern 3: Conditional OCR (quality gate)

```python
from mata.nodes import OCR, Conditional

# Only run expensive OCR when image quality is sufficient
graph = Graph([
    Conditional(
        condition=lambda ctx: ctx["quality_score"] > 0.7,
        if_true=OCR(using="high_quality_ocr", out="text"),
        if_false=OCR(using="fast_ocr", out="text"),
    ),
])

result = mata.infer(graph, image="scan.jpg", providers={
    "high_quality_ocr": mata.load("ocr", "stepfun-ai/GOT-OCR-2.0-hf"),
    "fast_ocr":         mata.load("ocr", "easyocr"),
}, quality_score=0.85)
```

### Pattern 4: VLM Tool-Calling Integration

```python
from mata.nodes import VLMQuery

# VLM can dispatch OCR as a tool during its agent loop
node = VLMQuery(
    using="vlm_provider",
    prompt="Extract all visible text from this document and summarize it.",
    tools=["ocr", "zoom"],    # OCR exposed as a callable tool
    max_iterations=4,
)

result = mata.infer(graph, image="form.jpg", providers={
    "vlm_provider": mata.load("vlm", "qwen3-vl"),
    "ocr":          mata.load("ocr", "easyocr"),
})
```

When `tools=["ocr", ...]` is set, the VLM agent loop dispatches OCR by calling
`OCRWrapper.predict()` which is the VLM-tool-dispatch-compatible entry point on
the wrapper.

---

## Design Decisions

### 1. End-to-End Only (no two-stage detect-then-recognize)

All five supported backends perform text detection **and** recognition in a
single `predict()` call. A two-stage pipeline (e.g. EAST detector + CRNN
recognizer) requires external coordination and is deferred to a future release.
Users who need two-stage OCR today can compose a `Detect` node (for text
region detection) with a per-crop `OCR` node on the extracted `ROIs`.

### 2. Dual result types — `OCRResult` and `OCRText`

`OCRResult` is the thin, adapter-level type. It is serializable, JSON-roundtrippable,
and carries no graph or artifact protocol dependencies. This keeps adapters free
of graph coupling (the same pattern used for detection's `VisionResult` / `Detections`).

`OCRText` is the graph artifact. It adds `instance_ids` for ROI correlation and
implements the `Artifact` protocol so it can flow through typed graph edges.
The conversion is done once inside `OCRWrapper.recognize()`.

### 3. External engine routing via adapter class, not model ID

Unlike HuggingFace adapters (where model IDs drive dispatch), the external engines
(EasyOCR, PaddleOCR, Tesseract) are selected by passing their alias strings
(`"easyocr"`, `"paddleocr"`, `"tesseract"`) which resolve to their respective
adapter classes via the `UniversalLoader`. No model ID download is attempted for
these backends.

### 4. Lazy imports for all optional backends

All five backends use the `_ensure_X()` lazy-import pattern so that MATA starts
up without error even when none of the OCR packages are installed. Import failures
surface as clear `ImportError` messages with `pip install` instructions only at
`.predict()` or `mata.load()` time.

### 5. Confidence normalization

- **EasyOCR**: returns float [0.0, 1.0] — used directly.
- **PaddleOCR**: returns float [0.0, 1.0] — used directly.
- **Tesseract**: returns int [0, 100] (with −1 meaning "no text") — divided by 100 and
  −1 entries are filtered out.
- **TrOCR / GOT-OCR2**: no per-token confidence; `score` is set to `1.0` as a
  placeholder indicating the model produced output.

### 6. Bbox coordinate contract

All adapters normalize spatial output to **xyxy absolute pixel coordinates**
regardless of the backend's native format:

| Backend          | Native format                  | Conversion                     |
| ---------------- | ------------------------------ | ------------------------------ |
| EasyOCR          | 4-point polygon `[[x,y], ...]` | `min/max` of all points → xyxy |
| PaddleOCR        | 4-point polygon `[[x,y], ...]` | `min/max` of all points → xyxy |
| Tesseract        | `(x, y, width, height)`        | `(x, y, x+w, y+h)`             |
| TrOCR / GOT-OCR2 | none                           | `bbox=None`                    |

---

## Known Limitations

### TrOCR — single-line focus

TrOCR (`microsoft/trocr-*`) uses a vision encoder + text decoder architecture
fine-tuned on **pre-cropped single text-line images**. When a full-page or
multi-line document is passed, performance degrades significantly. Use GOT-OCR2
or an external engine (EasyOCR / PaddleOCR / Tesseract) for full-page documents.

If per-line TrOCR is needed, detect text lines first with a text-detection model,
extract `ROIs`, and then run `OCR(using="trocr")` on the crops.

### PaddleOCR — version compatibility (paddleocr vs paddlepaddle)

> **Critical:** `paddleocr` and `paddlepaddle` (or `paddlepaddle-gpu`) **major versions
> must match**. Installing `paddleocr 3.x` with `paddlepaddle 2.x` (or vice versa)
> causes native Windows crashes (`STATUS_ACCESS_VIOLATION`, exit code `-1073741819`),
> missing-symbol `ImportError`s deep inside paddle's C++ runtime, and broken
> `AnalysisConfig` API calls that cannot be caught from Python.
>
> MATA detects this at load time and raises a clear `ImportError` with install
> instructions before any native code runs.

```bash
# Check installed versions
pip show paddleocr paddlepaddle paddlepaddle-gpu

# Correct GPU setup for paddleocr 3.x (CUDA 12.6 recommended):
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

**Additional Windows-specific quirks** patched automatically by MATA
(in `paddleocr_adapter.py`):

| Symptom                                                                           | Root cause                                                  | Fix applied                                                 |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| `ImportError: cannot import name 'forward_complete_op_role'`                      | `pipeline_scheduler_pass` package broken in some GPU wheels | Stub module injected into `sys.modules` before paddle loads |
| `AttributeError: 'AnalysisConfig' … 'set_optimization_level'`                     | Method renamed/read-only in GPU wheel                       | No-op shim patched onto `AnalysisConfig` after load         |
| `AttributeError: partially initialized module 'paddle' has no attribute 'tensor'` | Circular import on second `import paddle` call              | Use `sys.modules.get('paddle')` instead of fresh import     |

### PaddleOCR — large wheel size

`paddlepaddle-gpu` 3.x is approximately **1–1.5 GB**. In container or CI environments
this can be prohibitive. Use `paddlepaddle` (CPU) or a lighter alternative
(EasyOCR, Tesseract) if wheel size is a constraint.

### Tesseract — system binary dependency

`pytesseract` is only a thin Python wrapper. The actual `tesseract` binary must
be installed separately on the host OS. This breaks out-of-the-box in Docker
images that do not include the binary. Installations in restricted environments
may require custom `tesseract_cmd` configuration.

### GOT-OCR2 — ⚠️ Avoid until further notice

> **Warning:** GOT-OCR2 (`stepfun-ai/GOT-OCR-2.0-hf`) is **currently disabled in
> MATA** due to consistent hallucination behaviour observed with recent `transformers`
> versions. The model generates plausible-looking but incorrect text even on clean
> document images. Do **not** use it in production.
>
> Status: under investigation. Will be re-enabled once a stable, non-hallucinating
> configuration is identified. Track progress in the repo issues.

Additional hard requirement: `trust_remote_code=True` is needed, which executes
model-specific Python code downloaded from the HuggingFace Hub. Only use this
model from trusted, vetted sources once the hallucination issue is resolved.

**Recommended alternatives** while GOT-OCR2 is unusable:

- Full-page documents: **EasyOCR** or **PaddleOCR** (both support 80+ languages).
- Single text-line crops: **TrOCR** (`microsoft/trocr-base-printed` / `…-handwritten`).

### No confidence values for HuggingFace models

Neither TrOCR nor GOT-OCR2 expose per-region confidence scores through their
standard inference APIs. The `score` field is set to `1.0` as a placeholder.
Applications that rely on confidence-based filtering should prefer EasyOCR,
PaddleOCR, or Tesseract.

### No ONNX / TorchScript OCR runtime

The OCR task does not yet have an ONNX adapter. All inference runs through the
native backend runtimes (PyTorch, PaddlePaddle, the Tesseract binary).

---

## Future Roadmap

| Item                                        | Notes                                                                                                                           |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **ONNX OCR runtime**                        | Export TrOCR / CRNN to ONNX for faster CPU inference and consistent cross-platform behavior.                                    |
| **Two-stage detect + recognize**            | Add a `TwoStageOCRAdapter` that wraps a text detector (e.g. CRAFT, DBNet) and a recognizer (e.g. CRNN) for maximum flexibility. |
| **Document layout analysis**                | Pre-segment documents into regions (title, paragraph, table, figure) before OCR to improve reading order and accuracy.          |
| **Per-token confidence (GOT-OCR2)**         | Expose token-level log-probabilities where available.                                                                           |
| **Streaming / page-level batching**         | Efficient multi-page PDF processing without loading all pages into memory.                                                      |
| **Azure Document Intelligence integration** | Cloud backend adapter for high-accuracy handwriting and form understanding.                                                     |

---

## File Map

```
src/mata/
  api.py                                ← mata.load("ocr", ...) / mata.run("ocr", ...)
  core/
    types.py                            ← OCRResult, TextRegion
    model_loader.py                     ← UniversalLoader (OCR strategy)
    artifacts/
      ocr_text.py                       ← OCRText, TextBlock (graph artifact)
  adapters/
    ocr/
      __init__.py                       ← exports all four adapter classes
      huggingface_ocr_adapter.py        ← GOT-OCR2 + TrOCR
      easyocr_adapter.py                ← EasyOCR
      paddleocr_adapter.py              ← PaddleOCR
      tesseract_adapter.py              ← Tesseract via pytesseract
    wrappers/
      ocr_wrapper.py                    ← OCRWrapper (graph + VLM tool protocol)
  nodes/
    ocr.py                              ← OCR graph node (Image / ROIs → OCRText)

tests/
  test_ocr_adapters.py                  ← adapter-level unit tests (56 tests)
  test_ocr_api.py                       ← public API integration tests (8 tests)
```
