# MATA Multi‑Task Vision Architecture & Code Structure (Practical)

This document proposes a **strongly‑typed, mnemonic, model‑agnostic** architecture for **multi‑task vision** in MATA (Model‑Agnostic Task Architecture).

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

| Artifact | Represents | Typical fields |
|---|---|---|
| `Image` | frame tensor + metadata | `width`, `height`, `color_space`, `timestamp` |
| `Detections` | boxes + labels + scores | `boxes`, `labels`, `scores`, `instance_ids` |
| `ROIs` | crops/views based on boxes | `roi_images`, `roi_map` |
| `Masks` | per‑instance masks | `mask_rle`/`polygons`, `instance_ids` |
| `Keypoints` | skeleton points | `points`, `scores`, `instance_ids` |
| `Tracks` | temporal association | `track_id`, `history`, `velocity` |
| `MultiResult` | result bundle | all channels + provenance |

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
    models={
        "detector": transformer_or_pytorch_provider,
        "segmenter": sam3_provider,
    },
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.35, out="targets"),
        PromptBoxes(using="segmenter", image="image", dets="targets", out="masks"),
        RefineMask(src="masks", method="morph_close", radius=3, out="masks_ref"),
        Fuse(dets="targets", masks="masks_ref", out="final"),
    ],
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

1) **Runtime‑enforced artifact dataclasses**  
2) **Static typing via Protocols/Generics** (mypy/pyright friendly)  
3) **Graph validator** checks:
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
    boxes_xyxy: "np.ndarray"   # (N,4)
    labels: list[str]
    scores: "np.ndarray"       # (N,)
    instance_ids: list[str]    # stable per-frame ids

@dataclass(frozen=True)
class Masks(Artifact):
    mask_rle: list[dict]       # per instance RLE (or polygons)
    instance_ids: list[str]    # must align to detections when prompted by boxes
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

  core/
    artifacts/
      base.py              # Artifact base + validation helpers
      image.py             # Image, FrameMeta
      detections.py        # Detections, Boxes, Labels
      masks.py             # Masks (RLE/polygons/bitmaps)
      keypoints.py         # Keypoints
      tracks.py            # Tracks
      result.py            # MultiResult bundle

    graph/
      node.py              # Node base class (typed IO signature)
      graph.py             # Graph builder + DAG compile
      validator.py         # type+dependency checks
      scheduler.py         # parallel/async runner
      context.py           # execution context, caching, device selection

    io/
      codecs.py            # RLE/poly encoding, overlay render
      serializers.py       # JSON schema export for API responses

    registry/
      providers.py         # Provider registry, capability lookup
      typing.py            # Protocols: Detector, Segmenter, PoseEstimator

    observability/
      metrics.py           # per-node latency, GPU stats hooks
      tracing.py           # spans/events
      provenance.py        # model hashes, config fingerprints

  nodes/
    detect.py              # Detect node
    filter.py              # Filter/TopK
    roi.py                 # Crop/ExtractROIs
    prompts.py             # PromptBoxes/PromptPoints/SegmentEverything
    mask_ops.py            # RefineMask/MaskToBox/ExpandBoxes
    pose.py                # PoseFromDetections
    tracking.py            # Track/Associate
    fuse.py                # Fuse/Bundle

  providers/
    transformer_or_pytorch/
      adapter.py           # implements Detector (and optional Segmenter/Pose)
      config.py
    sam3/
      adapter.py           # implements Segmenter
      config.py
    rtdetr/
      adapter.py           # implements Detector
      config.py

  examples/
    image_multitask.py
    video_stream_multitask.py

  tests/
    test_graph_validation.py
    test_node_signatures.py
    test_result_schema.py
```

---

## 8) Recommended Node Set (MVP → Pro)

### MVP nodes (ship first)
- `Detect`
- `Filter` / `TopK`
- `PromptBoxes` (segment-from-box)
- `RefineMask`
- `Fuse` (bundle output)

### Next wave
- `PoseFromDetections`
- `SegmentEverything` (if supported)
- `MaskToBox` / `ExpandBoxes`
- `Track` (BYTETrack/DeepSORT-like adapter)
- `ClassifyROIs` (attributes)

### Pro / advanced
- Conditional branches (`If`, `Switch`)
- Temporal windows (`Window(n=8)`)
- Multi-camera joins (re-id)
- Shared backbone providers

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

1) `PromptBoxes(Image, Detections) -> Masks` requires `Detections.instance_ids` to exist  
2) `Fuse` requires consistent `instance_ids` across artifacts (or explicit mapping)  
3) Graph compilation fails if:
   - required artifact keys are missing
   - incompatible artifact types are wired
   - nodes produce duplicate keys unless explicitly allowed
4) Provider selection fails fast if the bound provider does not implement required capability

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

1) Implement `Artifact` base + canonical artifacts (`Image`, `Detections`, `Masks`, `MultiResult`)  
2) Implement node base + graph compiler + validator  
3) Add minimal scheduler (sync first, then parallel)  
4) Build 2 providers:
   - Detector provider (transformer_or_pytorch/RT-DETR adapter)
   - Segmenter provider (SAM3 adapter)
5) Add `Fuse` + JSON serializer  
6) Add 2 example scripts (image + stream)

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
