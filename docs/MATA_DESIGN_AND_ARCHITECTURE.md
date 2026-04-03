# MATA Multi‑Task Vision Architecture & Code Structure (Practical)

This document describes a **strongly‑typed, mnemonic, model‑agnostic** architecture for **multi‑task vision** in MATA (Model‑Agnostic Task Architecture).

> Goal: **One input frame → multiple integrated outputs** (detections, masks, keypoints, tracks, attributes) via a **typed task graph** that is readable, safe, and extensible.

---

## 1) Design Principles

### 1.1 Model‑agnostic core

- Nodes express **intent** (`Detect`, `PromptBoxes`, `RefineMask`, `Track`), not vendor/model names.
- Models are plugged in as **providers/adapters** that implement capabilities.

### 1.2 Strong types at the artifact boundary

- Every node has explicit **input/output artifact types** (e.g., `Image -> Detections`).
- Graph validation prevents invalid chains before runtime.

### 1.3 Mnemonic graph DSL

- Graph reads like a sentence:
  - `Detect → Filter → PromptBoxes → RefineMask → Fuse`
- Support both:
  - **Pipe style** (`graph.then(Node(...))`)
  - **Explicit wiring** (`out = Node()(a, b)`)

### 1.4 Deterministic “multi‑task result bundle”

- Unified schema with named **channels**:
  - `detections`, `masks`, `keypoints`, `tracks`, `overlays`, `metrics`, `provenance`

---

## 2) Core Concepts

### 2.1 Artifact Types (Strongly‑Typed Data Contracts)

Artifacts are the unit of graph wiring.

| Artifact      | Represents                 | Typical fields                                |
| ------------- | -------------------------- | --------------------------------------------- |
| `Image`       | frame tensor + metadata    | `width`, `height`, `color_space`, `timestamp` |
| `Detections`  | boxes + labels + scores    | `boxes`, `labels`, `scores`, `instance_ids`   |
| `ROIs`        | crops/views based on boxes | `roi_images`, `roi_map`                       |
| `Masks`       | per‑instance masks         | `mask_rle`/`polygons`, `instance_ids`         |
| `Keypoints`   | skeleton points            | `points`, `scores`, `instance_ids`            |
| `Tracks`      | temporal association       | `track_id`, `history`, `velocity`             |
| `MultiResult` | result bundle              | all channels + provenance                     |

### 2.2 Capability Interfaces (Provider Contracts)

A model is wrapped by an adapter implementing one or more capabilities:

- `Detector`: `predict(Image) -> Detections`
- `Segmenter`: `segment(Image, prompts) -> Masks`
- `PoseEstimator`: `pose(Image, rois/boxes) -> Keypoints`
- `Embedder`: `embed(Image/ROIs) -> Embeddings` (optional)

Providers may run locally, remotely, or in containerized services, but expose the same typed contract.

### 2.3 Nodes (Task Units)

Nodes are typed transformations with signatures like:

- `Detect: (Image) -> Detections`
- `Filter: (Detections) -> Detections`
- `PromptBoxes: (Image, Detections) -> Masks`
- `RefineMask: (Masks) -> Masks`
- `ExpandBoxes: (Detections, Masks) -> Detections`
- `Fuse: (...) -> MultiResult`

---

## 3) Reference Architecture

### 3.1 Logical Architecture (High‑Level)

```
        ┌─────────────────────────────┐
        │           Client            │
        │  UI: overlays + inspector   │
        └──────────────┬──────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                     MATA Runtime                   │
│                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │ Graph       │   │ Type Checker  │   │ Scheduler│ │
│  │ Compiler    │──▶│ + Validator   │──▶│ (parallel│ │
│  └─────┬───────┘   └──────────────┘   │  + async)│ │
│        │                               └────┬─────┘ │
│        ▼                                    ▼       │
│  ┌─────────────┐     ┌────────────────────────────┐ │
│  │ Artifact Bus │◀──▶│  Providers (Model Adapters) │ │
│  └─────┬───────┘     │  detector/segmenter/pose    │ │
│        │             └────────────────────────────┘ │
│        ▼                                            │
│  ┌─────────────┐                                   │
│  │ MultiResult  │  (channels + provenance + metrics)│
│  └─────────────┘                                   │
└────────────────────────────────────────────────────┘
```

### 3.2 Execution Patterns Supported

**Pattern A — Single multi‑head model** (optional later)

- One provider implements multiple capabilities with shared backbone.

**Pattern B — Parallel models** (common)

- `Detect` + `Pose` + `Segment` run in parallel where possible.

**Pattern C — Cascade/Conditional**

- `Detect -> Pose(person only)` and `Detect -> Segment(targets only)`.

---

## 4) Minimal Public API (Ergonomic + Typed)

### 4.1 User‑Facing Example (transformer_or_pytorch detect + SAM3 refine)

```python
results = mata.infer(
    image=image,
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.35, out="targets"),
        PromptBoxes(using="segmenter", image="image", dets="targets", out="masks"),
        RefineMask(src="masks", method="morph_close", radius=3, out="masks_ref"),
        Fuse(detections="targets", masks="masks_ref", out="final"),
    ],
    providers={
        "detector": transformer_or_pytorch_provider,
        "segmenter": sam3_provider,
    },
)
```

### 4.2 Result “Channels” (multimodal‑like UX)

```python
results.final.detections   # Detections
results.final.masks        # Masks
results.final.keypoints    # Keypoints (optional)
results.final.tracks       # Tracks (optional)
results.final.overlay()    # rendered overlay (optional)
results.final.provenance   # model hashes, params, versions
```

---

## 5) Strong Typing Strategy (Python‑Practical)

### 5.1 Type system layers

1. **Runtime‑enforced artifact dataclasses**
2. **Static typing via Protocols/Generics** (mypy/pyright friendly)
3. **Graph validator** checks:
   - Required inputs available
   - Input/output artifact compatibility
   - Dependency wiring correctness
   - Naming collisions, missing artifacts

### 5.2 Example artifact dataclasses (sketch)

```python
@dataclass(frozen=True)
class Image(Artifact):
    data: "np.ndarray | torch.Tensor"
    width: int
    height: int
    color_space: str = "BGR"
    timestamp_ms: int | None = None

@dataclass(frozen=True)
class Detections(Artifact):
    instances: list[Instance]  # spatial detections with bbox/mask
    instance_ids: list[str]    # stable per-frame ids (auto-generated)
    entities: list[Entity]     # semantic detections from VLM
    entity_ids: list[str]      # stable entity IDs (auto-generated)
    meta: dict[str, Any]       # optional metadata

    # Convenience property accessors:
    # dets.boxes   -> (N,4) xyxy numpy array
    # dets.scores  -> (N,) numpy array
    # dets.labels  -> list[str]

@dataclass(frozen=True)
class Masks(Artifact):
    instances: list[Instance]  # per-instance masks (RLE, polygon, or binary)
    instance_ids: list[str]    # aligned to source detections when prompted by boxes
    meta: dict[str, Any]       # optional metadata

    # Supports RLE, polygon, and binary mask formats with conversion methods
```

---

## 6) Graph Model

### 6.1 Node signatures (strong typing)

Each node declares:

- `inputs: dict[str, type[Artifact]]`
- `outputs: dict[str, type[Artifact]]`
- `requires: set[str]` (named artifact keys)
- `provides: str` (artifact key)

Example:

- `Detect` provides `"dets"` of type `Detections`
- `PromptBoxes` requires `"image"` + `"targets"` and provides `"masks"` of type `Masks`

### 6.2 Scheduler / Execution engine

- DAG compilation (topological sort)
- Parallel execution when nodes have no dependencies
- Latest‑frame policy optional (for streaming)
- Deterministic execution order for reproducibility

---

## 7) Folder / Code Structure (Recommended)

```
mata/
  __init__.py
  __main__.py
  api.py                   # Public API (mata.load/run/track/infer/val)
  cli.py                   # CLI entry point
  notebook.py              # Notebook display helpers
  visualization.py         # Overlay rendering
  visualization_cv2.py     # OpenCV visualization backend

  core/
    artifacts/
      base.py              # Artifact base + validation helpers
      image.py             # Image (PIL/numpy/torch, lazy conversion)
      detections.py        # Detections (Instance-based, VLM entity support)
      masks.py             # Masks (RLE/polygons/bitmaps)
      keypoints.py         # Keypoints
      tracks.py            # Tracks
      rois.py              # ROIs (cropped regions)
      result.py            # MultiResult bundle
      embeddings.py        # Embeddings artifact
      classifications.py   # Classification artifact
      depth_map.py         # Depth map artifact
      matches.py           # Recognition matches
      cross_matches.py     # Cross-camera / cross-gallery matches
      barcode_data.py      # Barcode data
      ocr_text.py          # OCR text artifact
      converters.py        # Cross-artifact converters

    graph/
      node.py              # Node base class (typed IO signature)
      graph.py             # Graph builder + DAG compile
      validator.py         # type+dependency checks
      scheduler.py         # SyncScheduler + ParallelScheduler
      context.py           # execution context, caching, device selection
      conditionals.py      # If / EarlyExit / While control flow
      temporal.py          # FramePolicy*, VideoProcessor, Window
      dsl.py               # Graph DSL helpers

    registry/
      providers.py         # Provider registry, capability lookup
      protocols.py         # Protocols: Detector, Segmenter, PoseEstimator, Embedder

    observability/
      metrics.py           # per-node latency, GPU stats hooks
      tracing.py           # spans/events
      provenance.py        # model hashes, config fingerprints

    model_loader.py        # UniversalLoader (5-strategy detection)
    model_registry.py      # YAML config model registry
    types.py               # VisionResult, Instance, Entity, ClassifyResult, etc.
    mask_utils.py          # RLE/polygon encoding/conversion
    exceptions.py          # Custom exception hierarchy
    exporters/             # JSON/CSV/image export

  nodes/                   # One file per node
    annotate.py            # Annotate overlay output
    barcode.py             # Barcode / QR decoding
    classify.py            # Classify
    depth.py               # EstimateDepth
    detect.py              # Detect
    embed.py               # Embed
    expand_boxes.py        # ExpandBoxes
    filter.py              # Filter
    fuse.py                # Fuse / bundle outputs
    gallery_match.py       # GalleryMatchNode
    mask_to_box.py         # MaskToBox
    nms.py                 # NMS
    ocr.py                 # OCR
    prompt_boxes.py        # PromptBoxes
    prompt_points.py       # PromptPoints
    refine_mask.py         # RefineMask
    roi.py                 # ExtractROIs
    segment.py             # SegmentImage
    segment_everything.py  # SegmentEverything
    topk.py                # TopK
    track.py               # Track
    vlm_query.py           # VLMQuery (agent tool-calling)
    vlm_detect.py          # VLMDetect
    vlm_describe.py        # VLMDescribe
    ...                    # + annotate_rt, merge, promote_entities, reid, valkey_*

  adapters/                # Model adapters (flat structure)
    huggingface_adapter.py          # HuggingFace detection
    huggingface_sam_adapter.py      # SAM segmentation
    huggingface_classify_adapter.py # HuggingFace classification
    huggingface_depth_adapter.py    # Depth estimation
    huggingface_vlm_adapter.py      # VLM backends
    clip_adapter.py                 # CLIP zero-shot
    embed_adapter.py                # Embedding extraction
    reid_adapter.py                 # ReID for tracking
    onnx_adapter.py                 # ONNX runtime
    pytorch_adapter.py              # PyTorch runtime
    torchscript_adapter.py          # TorchScript runtime
    tracking_adapter.py             # Tracker orchestration
    ...                             # + task/runtime-specific adapters

  recognition/             # Gallery-based identity matching
  trackers/                # Vendored ByteTrack/BotSort + ReID bridge
  training/                # Fine-tuning support
  eval/                    # Validation/evaluation pipeline
```

---

## 8) Recommended Node Set (MVP → Pro)

### MVP nodes (ship first)

- `Detect`
- `Filter` / `TopK`
- `PromptBoxes` (segment-from-box)
- `RefineMask`
- `Fuse` (bundle output)

### Next wave (implemented)

- `Classify` for full images and ROI-based classification flows
- `SegmentEverything` for SAM-style whole-image segmentation
- `MaskToBox` / `ExpandBoxes` for mask-box conversion utilities
- `Track` for vendored BYTETrack / BotSort workflows
- `EstimateDepth` for monocular depth estimation
- `Embed` for feature embedding extraction
- `GalleryMatchNode` for gallery-based identity matching

### Pro / advanced (implemented in selected modules)

- Conditional branches via `If` plus control-flow primitives `EarlyExit` and `While`
- Temporal windows via `Window(n=8)` and related temporal processing helpers
- Multi-camera ReID via `ReIDBridge` in the tracking stack
- VLM agent tool-calling via `VLMQuery(tools=[...])`

### Not yet implemented as public node exports

- `PoseFromDetections`
- `Switch`
- Shared backbone providers as a first-class public abstraction

---

## 9) MultiResult Output Schema (API-friendly)

### 9.1 Canonical shape

```json
{
  "frame_id": "cam1:000123",
  "inputs": {
    "image_size": [1920, 1080],
    "timestamp_ms": 1730000000000
  },
  "channels": {
    "detections": { "... typed payload ..." },
    "masks": { "... typed payload ..." },
    "keypoints": null,
    "tracks": null,
    "overlays": {
      "rgba_png": "artifact://overlay.png"
    }
  },
  "metrics": {
    "latency_ms": {
      "Detect": 18,
      "PromptBoxes": 62,
      "total": 92
    }
  },
  "provenance": {
    "models": {
      "detector": {"name":"detector_v1","hash":"...","params_fingerprint":"..."},
      "segmenter": {"name":"segmenter_v1","hash":"...","params_fingerprint":"..."}
    },
    "graph_hash": "..."
  }
}
```

### 9.2 Important rule: stable instance identity

- `instance_ids` unify artifacts across channels within a frame:
  - detection `obj_1` ↔ mask `obj_1` ↔ keypoints `obj_1`

---

## 10) Practical Validation Rules (must‑have)

1. `PromptBoxes(Image, Detections) -> Masks` requires `Detections.instance_ids` to exist
2. `Fuse` requires consistent `instance_ids` across artifacts (or explicit mapping)
3. Graph compilation fails if:
   - required artifact keys are missing
   - incompatible artifact types are wired
   - nodes produce duplicate keys unless explicitly allowed
4. Provider selection fails fast if the bound provider does not implement required capability

---

## 11) Streaming Considerations (RTSP/Webcam)

For real‑time use:

- Add `FramePolicyLatest()` (drop old frames) vs `FramePolicyQueue(n)`
- Tracking nodes should optionally run even if segmentation drops frames (decouple)

Suggested streaming pipeline:

- `Decode -> Detect -> Track` (fast lane)
- `Decode -> Detect -> Segment` (slow lane, best-effort)
- `JoinByTrackId/NearestTimestamp`

---

## 12) Next Steps (Implementation Order)

1. Implement `Artifact` base + canonical artifacts (`Image`, `Detections`, `Masks`, `MultiResult`)
2. Implement node base + graph compiler + validator
3. Add minimal scheduler (sync first, then parallel)
4. Build 2 providers:
   - Detector provider (transformer_or_pytorch/RT-DETR adapter)
   - Segmenter provider (SAM3 adapter)
5. Add `Fuse` + JSON serializer
6. Add 2 example scripts (image + stream)

---

## Appendix A — Minimal Node Signature Spec (for docs)

A node must declare:

- `name: str`
- `inputs: dict[str, type[Artifact]]`
- `outputs: dict[str, type[Artifact]]`
- `run(ctx, **inputs) -> dict[str, Artifact]`

Example:

```python
class Detect(Node):
    name = "Detect"
    inputs = {"image": Image}
    outputs = {"dets": Detections}

    def __init__(self, using: str, out: str = "dets"):
        self.using = using
        self.out = out

    def run(self, ctx, image: Image):
        det = ctx.providers.detector(self.using).predict(image)
        return {self.out: det}
```

---

**End.**
