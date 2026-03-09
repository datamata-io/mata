# MATA Graph System — API Reference

> **Version**: 1.6.0 | **Last Updated**: February 12, 2026

---

## Table of Contents

1. [Public API](#public-api)
2. [Artifacts](#artifacts)
3. [Built-in Nodes](#built-in-nodes)
4. [Storage Nodes](#storage-nodes)
5. [Graph Builder](#graph-builder)
6. [Schedulers](#schedulers)
7. [Execution Context](#execution-context)
8. [Providers & Protocols](#providers--protocols)
9. [Conditional Execution](#conditional-execution)
10. [Temporal / Video](#temporal--video)
11. [Observability](#observability)
12. [DSL Helpers](#dsl-helpers)
13. [Presets](#presets)
14. [Converters & Utilities](#converters--utilities)

---

## Public API

### `mata.infer()`

Execute a multi-task graph on an image.

```python
def infer(
    image: Union[str, Path, PIL.Image.Image, np.ndarray],
    graph: Union[Graph, List[Node]],
    providers: Dict[str, Any],
    scheduler: Optional[Scheduler] = None,
    device: str = "auto",
    **kwargs,
) -> MultiResult
```

**Parameters:**

| Parameter   | Type                                     | Default           | Description                                                                                              |
| ----------- | ---------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| `image`     | `str \| Path \| PIL.Image \| np.ndarray` | _required_        | Input image (file path, PIL image, or numpy array)                                                       |
| `graph`     | `Graph \| list[Node]`                    | _required_        | Graph object or list of nodes (auto-wrapped into Graph)                                                  |
| `providers` | `dict[str, Any]`                         | _required_        | Provider instances keyed by name. Flat `{"name": adapter}` or nested `{"capability": {"name": adapter}}` |
| `scheduler` | `Scheduler \| None`                      | `SyncScheduler()` | Execution strategy. Use `ParallelScheduler()` for concurrent stages                                      |
| `device`    | `str`                                    | `"auto"`          | Device placement: `"auto"`, `"cuda"`, `"cpu"`                                                            |

**Returns:** `MultiResult` with all task outputs accessible as attributes.

**Raises:** `ValueError` (bad input), `ValidationError` (compilation failure), `RuntimeError` (execution failure).

**Example:**

```python
import mata
from mata.nodes import Detect, Filter, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")
result = mata.infer(
    image="test.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.3, out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"detector": detector},
)
print(result.final)
```

---

## Artifacts

All artifacts inherit from `Artifact` (frozen dataclass, immutable). Every artifact implements `to_dict()`, `from_dict()`, and optionally `validate()`.

### `Image`

Multi-format image container with lazy conversion.

```python
from mata.core.artifacts import Image
```

| Attribute      | Type                                      | Description                                 |
| -------------- | ----------------------------------------- | ------------------------------------------- |
| `data`         | `PIL.Image \| np.ndarray \| torch.Tensor` | Raw image data                              |
| `width`        | `int`                                     | Image width in pixels                       |
| `height`       | `int`                                     | Image height in pixels                      |
| `color_space`  | `str`                                     | `"RGB"`, `"BGR"`, `"GRAY"`, `"L"`, `"RGBA"` |
| `timestamp_ms` | `int \| None`                             | Unix timestamp (for video frames)           |
| `frame_id`     | `str \| None`                             | Frame identifier (for video)                |
| `source_path`  | `str \| None`                             | Original file path                          |

**Factory methods:**

| Method                                         | Description         |
| ---------------------------------------------- | ------------------- |
| `Image.from_path(path, color_space="RGB")`     | Load from disk      |
| `Image.from_pil(pil_image)`                    | Wrap PIL image      |
| `Image.from_numpy(array, color_space="RGB")`   | Wrap numpy array    |
| `Image.from_tensor(tensor, color_space="RGB")` | Wrap PyTorch tensor |

**Conversion methods:**

| Method        | Returns        | Description                 |
| ------------- | -------------- | --------------------------- |
| `to_pil()`    | `PIL.Image`    | Convert to PIL (lazy)       |
| `to_numpy()`  | `np.ndarray`   | Convert to numpy HWC uint8  |
| `to_tensor()` | `torch.Tensor` | Convert to CHW float tensor |

---

### `Detections`

Detection results with instance IDs and entity support.

```python
from mata.core.artifacts import Detections
```

| Attribute      | Type             | Description                                  |
| -------------- | ---------------- | -------------------------------------------- |
| `instances`    | `list[Instance]` | Spatial detections (bbox + label + score)    |
| `instance_ids` | `list[str]`      | Stable IDs (auto-generated if empty)         |
| `entities`     | `list[Entity]`   | Semantic detections from VLM (label + score) |
| `entity_ids`   | `list[str]`      | Entity IDs (auto-generated if empty)         |
| `meta`         | `dict`           | Optional metadata                            |

**Properties:**

| Property | Type                | Description                   |
| -------- | ------------------- | ----------------------------- |
| `boxes`  | `np.ndarray (N, 4)` | Bounding boxes in xyxy format |
| `scores` | `np.ndarray (N,)`   | Confidence scores             |
| `labels` | `list[str]`         | Label names                   |

**Methods:**

| Method                                | Returns        | Description                     |
| ------------------------------------- | -------------- | ------------------------------- |
| `from_vision_result(result)`          | `Detections`   | Convert from VisionResult       |
| `to_vision_result()`                  | `VisionResult` | Convert back to VisionResult    |
| `filter_by_score(threshold)`          | `Detections`   | Keep detections above threshold |
| `filter_by_label(labels)`             | `Detections`   | Keep specified labels           |
| `top_k(k)`                            | `Detections`   | Keep top K by score             |
| `promote_entities(spatial, strategy)` | `Detections`   | Promote entities to instances   |

---

### `Masks`

Per-instance segmentation masks with format conversion.

```python
from mata.core.artifacts import Masks
```

| Attribute      | Type             | Description                      |
| -------------- | ---------------- | -------------------------------- |
| `instances`    | `list[Instance]` | Instances with mask data         |
| `instance_ids` | `list[str]`      | Stable IDs for cross-referencing |
| `meta`         | `dict`           | Optional metadata                |

**Methods:**

| Method                       | Returns        | Description                     |
| ---------------------------- | -------------- | ------------------------------- |
| `from_vision_result(result)` | `Masks`        | Convert from VisionResult       |
| `to_vision_result()`         | `VisionResult` | Convert back                    |
| `to_rle()`                   | `Masks`        | Convert masks to RLE encoding   |
| `to_polygons()`              | `Masks`        | Convert masks to polygon format |
| `to_binary()`                | `Masks`        | Convert masks to binary arrays  |

---

### `Classifications`

Sorted classification predictions.

```python
from mata.core.artifacts import Classifications
```

| Attribute     | Type                    | Description                        |
| ------------- | ----------------------- | ---------------------------------- |
| `predictions` | `tuple[Classification]` | Sorted predictions (label + score) |
| `meta`        | `dict`                  | Optional metadata                  |

**Properties:**

| Property | Type                   | Description                   |
| -------- | ---------------------- | ----------------------------- |
| `top1`   | `Classification`       | Highest confidence prediction |
| `top5`   | `list[Classification]` | Top 5 predictions             |
| `labels` | `list[str]`            | All label names               |
| `scores` | `list[float]`          | All scores                    |

**Methods:**

| Method                         | Returns           | Description                 |
| ------------------------------ | ----------------- | --------------------------- |
| `from_classify_result(result)` | `Classifications` | Convert from ClassifyResult |
| `to_classify_result()`         | `ClassifyResult`  | Convert back                |
| `to_json()`                    | `str`             | JSON serialization          |

---

### `DepthMap`

Per-pixel depth estimation result.

```python
from mata.core.artifacts import DepthMap
```

| Attribute    | Type                 | Description                        |
| ------------ | -------------------- | ---------------------------------- |
| `depth`      | `np.ndarray (H, W)`  | Raw depth values                   |
| `normalized` | `np.ndarray \| None` | Normalized [0, 1] depth (optional) |
| `meta`       | `dict`               | Optional metadata                  |

**Properties:**

| Property | Type    | Description  |
| -------- | ------- | ------------ |
| `height` | `int`   | Map height   |
| `width`  | `int`   | Map width    |
| `shape`  | `tuple` | (H, W) shape |

**Methods:**

| Method                      | Returns       | Description              |
| --------------------------- | ------------- | ------------------------ |
| `from_depth_result(result)` | `DepthMap`    | Convert from DepthResult |
| `to_depth_result()`         | `DepthResult` | Convert back             |

---

### `Keypoints`

Per-instance keypoint arrays.

```python
from mata.core.artifacts import Keypoints
```

| Attribute      | Type                            | Description                         |
| -------------- | ------------------------------- | ----------------------------------- |
| `keypoints`    | `list[np.ndarray]`              | Each shape `(N, 3)` for x, y, score |
| `instance_ids` | `list[str]`                     | Instance IDs                        |
| `skeleton`     | `list[tuple[int, int]] \| None` | Skeleton connections                |
| `meta`         | `dict`                          | Optional metadata                   |

**Methods:**

| Method                                  | Returns      | Description                     |
| --------------------------------------- | ------------ | ------------------------------- |
| `filter_by_visibility(threshold)`       | `Keypoints`  | Keep visible keypoints          |
| `get_visible_keypoints(idx, threshold)` | `np.ndarray` | Get visible points for instance |
| `count_visible(threshold)`              | `list[int]`  | Count visible per instance      |

---

### `Tracks`

Temporal tracking results.

```python
from mata.core.artifacts import Tracks, Track
```

| `Track` Attribute | Type                  | Description                          |
| ----------------- | --------------------- | ------------------------------------ |
| `track_id`        | `int`                 | Unique track identifier              |
| `bbox`            | `tuple[float, ...]`   | Bounding box (xyxy)                  |
| `score`           | `float`               | Confidence score                     |
| `label`           | `str`                 | Object label                         |
| `age`             | `int`                 | Frames since creation                |
| `state`           | `str`                 | `"active"`, `"lost"`, `"terminated"` |
| `history`         | `list[tuple] \| None` | Historical positions                 |

| `Tracks` Attribute | Type          | Description          |
| ------------------ | ------------- | -------------------- |
| `tracks`           | `list[Track]` | Current frame tracks |
| `frame_id`         | `str`         | Frame identifier     |
| `meta`             | `dict`        | Optional metadata    |

**Methods:**

| Method                      | Returns         | Description        |
| --------------------------- | --------------- | ------------------ |
| `get_active_tracks()`       | `list[Track]`   | Active tracks only |
| `get_lost_tracks()`         | `list[Track]`   | Lost tracks        |
| `get_terminated_tracks()`   | `list[Track]`   | Terminated tracks  |
| `get_track_by_id(track_id)` | `Track \| None` | Find by ID         |

---

### `ROIs`

Cropped image regions.

```python
from mata.core.artifacts import ROIs
```

| Attribute      | Type                              | Description           |
| -------------- | --------------------------------- | --------------------- |
| `roi_images`   | `list[PIL.Image \| np.ndarray]`   | Cropped image regions |
| `instance_ids` | `list[str]`                       | Source instance IDs   |
| `source_boxes` | `list[tuple[int, int, int, int]]` | Source bounding boxes |
| `meta`         | `dict`                            | Optional metadata     |

**Methods:**

| Method                         | Returns            | Description             |
| ------------------------------ | ------------------ | ----------------------- |
| `get_roi_sizes()`              | `list[tuple]`      | (width, height) per ROI |
| `get_roi_areas()`              | `list[int]`        | Area per ROI            |
| `filter_by_size(min_w, min_h)` | `ROIs`             | Filter small ROIs       |
| `to_numpy_list()`              | `list[np.ndarray]` | All as numpy            |
| `to_pil_list()`                | `list[PIL.Image]`  | All as PIL              |

---

### `MultiResult`

Channel-based result bundle from graph execution.

```python
from mata.core.artifacts import MultiResult
```

| Attribute    | Type                  | Description                            |
| ------------ | --------------------- | -------------------------------------- |
| `channels`   | `dict[str, Artifact]` | Named artifact channels                |
| `provenance` | `dict`                | Model versions, graph hash, timestamps |
| `metrics`    | `dict`                | Per-node timing and resource metrics   |
| `meta`       | `dict`                | Additional metadata                    |

**Dynamic attribute access:** `result.detections` → `result.channels["detections"]`

**Methods:**

| Method                       | Returns            | Description                 |
| ---------------------------- | ------------------ | --------------------------- |
| `has_channel(name)`          | `bool`             | Check channel existence     |
| `get_channel(name, default)` | `Artifact \| None` | Get with default            |
| `get_instance_artifacts(id)` | `dict`             | Cross-channel instance data |
| `to_json()`                  | `str`              | Full JSON serialization     |
| `to_dict()`                  | `dict`             | Dictionary representation   |

---

## Built-in Nodes

All nodes are importable from `mata.nodes`.

### Task Nodes

#### `Detect`

Run object detection via a provider.

```python
Detect(using: str, out: str = "dets", name: str = None, **kwargs)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `image`      | `Image`      |
| Output | `detections` | `Detections` |

Provider capability: `Detector.predict()`. Extra kwargs forwarded to predict (e.g., `threshold`, `nms_iou`, `text_prompts`).

---

#### `Classify`

Run image classification via a provider.

```python
Classify(using: str, out: str = "classifications", name: str = None, **kwargs)
```

| I/O    | Name              | Type              |
| ------ | ----------------- | ----------------- |
| Input  | `image`           | `Image`           |
| Output | `classifications` | `Classifications` |

Provider capability: `Classifier.classify()`.

---

#### `SegmentImage`

Run image segmentation via a provider.

```python
SegmentImage(using: str, out: str = "masks", name: str = None, **kwargs)
```

| I/O    | Name    | Type    |
| ------ | ------- | ------- |
| Input  | `image` | `Image` |
| Output | `masks` | `Masks` |

Provider capability: `Segmenter.segment()`.

---

#### `EstimateDepth`

Run monocular depth estimation via a provider.

```python
EstimateDepth(using: str, out: str = "depth", name: str = None, **kwargs)
```

| I/O    | Name    | Type       |
| ------ | ------- | ---------- |
| Input  | `image` | `Image`    |
| Output | `depth` | `DepthMap` |

Provider capability: `DepthEstimator.estimate()`.

---

### Data Transform Nodes

#### `Filter`

Filter detections by score threshold and/or labels.

```python
Filter(
    src: str = "dets",
    out: str = "filtered",
    score_gt: float = None,
    label_in: list[str] = None,
    label_not_in: list[str] = None,
    fuzzy: bool = False,
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `detections` | `Detections` |
| Output | `detections` | `Detections` |

Filtering order: score → label_in → label_not_in. Instance IDs are preserved.

---

#### `TopK`

Keep top K detections by confidence score.

```python
TopK(k: int, src: str = "dets", out: str = "topk", name: str = None)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `detections` | `Detections` |
| Output | `detections` | `Detections` |

---

#### `ExtractROIs`

Crop image regions from detection bounding boxes.

```python
ExtractROIs(
    src_image: str = "image",
    src_dets: str = "dets",
    out: str = "rois",
    padding: int = 0,
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `image`      | `Image`      |
| Input  | `detections` | `Detections` |
| Output | `rois`       | `ROIs`       |

---

#### `ExpandBoxes`

Recompute bounding boxes from segmentation masks.

```python
ExpandBoxes(
    src_dets: str = "dets",
    src_masks: str = "masks",
    out: str = "expanded",
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `detections` | `Detections` |
| Input  | `masks`      | `Masks`      |
| Output | `detections` | `Detections` |

---

### Prompt Nodes

#### `PromptBoxes`

Segment regions using detection bounding boxes as SAM prompts.

```python
PromptBoxes(
    using: str,
    image_src: str = "image",
    dets_src: str = "dets",
    out: str = "masks",
    name: str = None,
    **kwargs,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `image`      | `Image`      |
| Input  | `detections` | `Detections` |
| Output | `masks`      | `Masks`      |

Provider capability: `Segmenter.segment(mode="boxes")`. Preserves instance IDs from input detections.

---

#### `PromptPoints`

Segment regions using explicit point prompts.

```python
PromptPoints(
    using: str,
    points: list[tuple[int, int, int]] = None,  # (x, y, label)
    image_src: str = "image",
    out: str = "masks",
    name: str = None,
    **kwargs,
)
```

| I/O    | Name    | Type    |
| ------ | ------- | ------- |
| Input  | `image` | `Image` |
| Output | `masks` | `Masks` |

---

#### `SegmentEverything`

Automatic segmentation without prompts (SAM everything mode).

```python
SegmentEverything(
    using: str,
    image_src: str = "image",
    out: str = "masks",
    name: str = None,
    **kwargs,
)
```

| I/O    | Name    | Type    |
| ------ | ------- | ------- |
| Input  | `image` | `Image` |
| Output | `masks` | `Masks` |

---

### Mask Refinement Nodes

#### `RefineMask`

Apply morphological operations to segmentation masks.

```python
RefineMask(
    src: str = "masks",
    out: str = "masks_ref",
    method: str = "morph_close",
    radius: int = 3,
    name: str = None,
)
```

| I/O    | Name    | Type    |
| ------ | ------- | ------- |
| Input  | `masks` | `Masks` |
| Output | `masks` | `Masks` |

**Methods:** `"morph_close"`, `"morph_open"`, `"dilate"`, `"erode"`

---

#### `MaskToBox`

Extract bounding boxes from masks.

```python
MaskToBox(
    src: str = "masks",
    out: str = "detections",
    filter_empty: bool = True,
    expand_px: int = 0,
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `masks`      | `Masks`      |
| Output | `detections` | `Detections` |

---

### Tracking Node

#### `Track`

Temporal object tracking across frames.

```python
Track(
    using: str,
    dets: str = "dets",
    out: str = "tracks",
    frame_id: str = None,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `detections` | `Detections` |
| Output | `tracks`     | `Tracks`     |

Provider capability: `Tracker.update()`. Supports BYTETrack or built-in IoU-based tracker.

---

### Fusion Nodes

#### `Fuse`

Bundle artifacts into a `MultiResult` with provenance and metrics.

```python
Fuse(out: str = "final", **channel_sources)
```

| I/O    | Name        | Type             |
| ------ | ----------- | ---------------- |
| Input  | _(dynamic)_ | _(any artifact)_ |
| Output | `result`    | `MultiResult`    |

The `**channel_sources` maps channel names to artifact names in context:

```python
Fuse(detections="dets", masks="refined_masks", out="final")
```

---

#### `Merge`

Merge multiple artifacts by `instance_id` alignment.

```python
Merge(
    dets: str = "dets",
    masks: str = "masks",
    keypoints: str = "keypoints",
    out: str = "merged",
)
```

| I/O    | Name            | Type                     |
| ------ | --------------- | ------------------------ |
| Input  | `detections`    | `Detections`             |
| Input  | `masks`         | `Masks` _(optional)_     |
| Input  | `keypoints`     | `Keypoints` _(optional)_ |
| Output | `vision_result` | `VisionResult`           |

Aligns instances across modalities using stable `instance_id` values.

---

### VLM Nodes

#### `VLMDescribe`

Generate natural language image description.

```python
VLMDescribe(
    using: str,
    prompt: str = "Describe this image in detail.",
    out: str = "description",
    name: str = None,
    **vlm_kwargs,
)
```

| I/O    | Name          | Type         |
| ------ | ------------- | ------------ |
| Input  | `image`       | `Image`      |
| Output | `description` | `Detections` |

Provider capability: `VisionLanguageModel.query(output_mode="describe")`.

---

#### `VLMDetect`

Detect objects using a VLM with structured output.

```python
VLMDetect(
    using: str,
    prompt: str = "List all objects you can identify.",
    out: str = "vlm_dets",
    auto_promote: bool = True,
    name: str = None,
    **vlm_kwargs,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `image`      | `Image`      |
| Output | `detections` | `Detections` |

When `auto_promote=True`, entities with spatial data are automatically promoted to instances.

---

#### `VLMQuery`

Generic VLM query with multi-image support.

```python
VLMQuery(
    using: str,
    prompt: str,
    output_mode: str = None,
    out: str = "vlm_result",
    name: str = None,
    **vlm_kwargs,
)
```

| I/O    | Name     | Type         |
| ------ | -------- | ------------ |
| Input  | `image`  | `Image`      |
| Output | `result` | `Detections` |

Supports `output_mode`: `None`, `"json"`, `"detect"`, `"classify"`, `"describe"`.

---

#### `PromoteEntities`

Promote VLM entities to spatial instances via label matching.

```python
PromoteEntities(
    entities_src: str = "vlm_dets",
    spatial_src: str = "dino_dets",
    match_strategy: str = "label_fuzzy",
    out: str = "promoted",
    name: str = None,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `entities`   | `Detections` |
| Input  | `spatial`    | `Detections` |
| Output | `detections` | `Detections` |

**Match strategies:**

- `"label_exact"` — Case-sensitive exact match
- `"label_fuzzy"` — Case-insensitive, handles plurals and articles (recommended)

---

### Visualization & Analysis Nodes

#### `Annotate`

Render detections/masks onto images.

```python
Annotate(
    using: str = "pil",
    show_boxes: bool = True,
    show_labels: bool = True,
    show_masks: bool = True,
    show_scores: bool = True,
    alpha: float = 0.5,
    line_width: int = 2,
    out: str = "annotated",
    image_src: str = "image",
    detections_src: str = "detections",
    **kwargs,
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `image`      | `Image`      |
| Input  | `detections` | `Detections` |
| Output | `annotated`  | `Image`      |

**Backends:** `"pil"` (fast, default), `"matplotlib"` (publication-quality).

---

#### `NMS`

Non-maximum suppression to remove redundant overlapping detections.

```python
NMS(
    iou_threshold: float = 0.5,
    out: str = "nms_dets",
    detections_src: str = "detections",
)
```

| I/O    | Name         | Type         |
| ------ | ------------ | ------------ |
| Input  | `detections` | `Detections` |
| Output | `detections` | `Detections` |

Uses `torchvision.ops.nms` internally.

---

## Storage Nodes

Storage nodes connect graph pipelines to [Valkey](https://valkey.io/) (or wire-compatible Redis) for distributed result sharing, cross-pipeline caching, and event-driven architectures.

```python
from mata.nodes import ValkeyStore, ValkeyLoad
```

**Installation:**

```bash
pip install mata[valkey]   # valkey-py client
pip install mata[redis]    # redis-py client (fallback)
```

Both nodes lazy-import the client library — `import mata` succeeds without either installed. An `ImportError` with an actionable message is raised only when a storage node actually executes.

---

### URI Scheme Format

Storage nodes and `result.save()` accept Valkey/Redis URIs in the following formats:

| Format         | Example                                 |
| -------------- | --------------------------------------- |
| Basic          | `valkey://localhost:6379/my_key`        |
| With DB number | `valkey://localhost:6379/0/my_key`      |
| With password  | `valkey://user:pass@host:6379/0/my_key` |
| Redis fallback | `redis://localhost:6379/my_key`         |
| TLS (Redis)    | `rediss://host:6379/my_key`             |

The **key** is the last path segment (or everything after `db/` when a numeric DB is present). Passwords in URIs are passed through to the client and are **never logged**.

**Direct save from result objects:**

```python
# Any MATA result type supports valkey:// URIs in save()
result.save("valkey://localhost:6379/pipeline:detections:latest")
result.save("valkey://localhost:6379/0/detections:frame_042")  # DB 0
result.save("redis://localhost:6379/detections")               # redis-py
```

---

### Key Template Syntax

`ValkeyStore` accepts a `key` parameter that supports safe placeholder substitution:

| Placeholder   | Resolved value               | Example output |
| ------------- | ---------------------------- | -------------- |
| `{node}`      | Node's `name` attribute      | `ValkeyStore`  |
| `{timestamp}` | Unix epoch (integer seconds) | `1741478400`   |

Placeholders are resolved with `str.format()` using only these two predefined variables — user-controlled input is **never** interpolated directly.

```python
ValkeyStore(
    src="filtered",
    url="valkey://localhost:6379",
    key="pipeline:{node}:{timestamp}",  # → "pipeline:ValkeyStore:1741478400"
    ttl=3600,
)
```

---

### `ValkeyStore`

Sink node that writes an artifact to Valkey during graph execution. The artifact passes through unchanged, so downstream nodes can still consume it.

```python
ValkeyStore(
    src: str,
    url: str,
    key: str,
    ttl: int | None = None,
    serializer: str = "json",
    out: str | None = None,
)
```

**Parameters:**

| Parameter    | Type          | Default       | Description                                                    |
| ------------ | ------------- | ------------- | -------------------------------------------------------------- |
| `src`        | `str`         | _required_    | Name of the input artifact in the graph context                |
| `url`        | `str`         | _required_    | Valkey/Redis connection URL (see URI formats above)            |
| `key`        | `str`         | _required_    | Key name or template (`{node}`, `{timestamp}` supported)       |
| `ttl`        | `int \| None` | `None`        | Key expiration in seconds; `None` = no expiry                  |
| `serializer` | `str`         | `"json"`      | `"json"` (default) or `"msgpack"` (requires `msgpack` package) |
| `out`        | `str \| None` | same as `src` | Output artifact name (pass-through)                            |

**I/O:**

| I/O    | Name       | Type                   |
| ------ | ---------- | ---------------------- |
| Input  | `artifact` | `Artifact` (any)       |
| Output | `artifact` | `Artifact` (unchanged) |

**Supported artifact types:** `Detections`, `Masks`, `Classifications`, `DepthMap`. Other `Artifact` subclasses are stored as-is (best-effort).

**Example:**

```python
from mata.nodes import Detect, Filter, ValkeyStore
from mata.core.graph import Graph

graph = (
    Graph()
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(ValkeyStore(
        src="filtered",
        url="valkey://localhost:6379",
        key="pipeline:detections:{timestamp}",
        ttl=3600,
    ))
)

result = mata.infer("frame.jpg", graph=graph, providers={"detr": detector})
# result.filtered is still available — ValkeyStore is a pass-through sink
```

---

### `ValkeyLoad`

Source node that loads a previously stored result from Valkey and injects it into the graph as a typed artifact. Use this as an **entry node** to consume results produced by another pipeline.

```python
ValkeyLoad(
    url: str,
    key: str,
    result_type: str = "auto",
    out: str = "loaded",
)
```

**Parameters:**

| Parameter     | Type  | Default    | Description                                               |
| ------------- | ----- | ---------- | --------------------------------------------------------- |
| `url`         | `str` | _required_ | Valkey/Redis connection URL                               |
| `key`         | `str` | _required_ | Key name to load from                                     |
| `result_type` | `str` | `"auto"`   | `"auto"`, `"vision"`, `"classify"`, `"depth"`, or `"ocr"` |
| `out`         | `str` | `"loaded"` | Output artifact name in the graph context                 |

**I/O:**

| I/O    | Name                   | Type       |
| ------ | ---------------------- | ---------- |
| Input  | _(none — source node)_ | —          |
| Output | `artifact`             | `Artifact` |

**Auto-detection logic** (`result_type="auto"`):

| Key present in stored data | Detected type | Output artifact   |
| -------------------------- | ------------- | ----------------- |
| `instances`                | `vision`      | `Detections`      |
| `predictions`              | `classify`    | `Classifications` |
| `depth`                    | `depth`       | `DepthMap`        |
| `regions`                  | `ocr`         | _(raw dict)_      |

**Raises:**

- `KeyError` — key does not exist in Valkey
- `ValueError` — stored data cannot be auto-detected or `result_type` is unrecognized
- `ImportError` — valkey/redis client not installed

**Example:**

```python
from mata.nodes import ValkeyLoad, Filter, Fuse
from mata.core.graph import Graph

graph = (
    Graph()
    .then(ValkeyLoad(
        url="valkey://localhost:6379",
        key="upstream:detections:latest",
        result_type="vision",
        out="dets",
    ))
    .then(Filter(src="dets", score_gt=0.7, out="filtered"))
    .then(Fuse(detections="filtered"))
)

result = mata.infer("frame.jpg", graph=graph, providers={})
```

---

### Complete Store → Load Pipeline Example

This pattern enables two independent pipelines to share detection results through Valkey:

```python
import mata
from mata.nodes import Detect, Filter, ValkeyStore, ValkeyLoad, Fuse
from mata.core.graph import Graph

detector = mata.load("detect", "facebook/detr-resnet-50")

# --- Pipeline A: run detection and persist to Valkey ---
store_graph = (
    Graph()
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(ValkeyStore(
        src="filtered",
        url="valkey://localhost:6379",
        key="shared:detections:latest",
        ttl=60,  # expires after 60 seconds
    ))
)

mata.infer("frame_001.jpg", graph=store_graph, providers={"detr": detector})

# --- Pipeline B: load persisted results and annotate ---
load_graph = (
    Graph()
    .then(ValkeyLoad(
        url="valkey://localhost:6379",
        key="shared:detections:latest",
        out="dets",
    ))
    .then(Fuse(detections="dets", out="annotated"))
)

annotated = mata.infer("frame_001.jpg", graph=load_graph, providers={})
```

**Named connections via YAML config:**

```yaml
# .mata/models.yaml
storage:
  valkey:
    default:
      url: "valkey://localhost:6379"
      db: 0
      ttl: 3600
    production:
      url: "valkey://prod-cluster:6379"
      password_env: "VALKEY_PASSWORD" # read from env var, never stored in plaintext
      db: 1
      tls: true
```

```python
from mata.core.model_registry import ModelRegistry

registry = ModelRegistry()
conn = registry.get_valkey_connection("production")  # resolves password from env
```

---

## Graph Builder

### `Graph`

```python
from mata.core.graph import Graph
```

#### Constructor

```python
Graph(name: str = None)
```

#### Methods

| Method                                                    | Returns         | Description                                          |
| --------------------------------------------------------- | --------------- | ---------------------------------------------------- |
| `then(node)`                                              | `Graph`         | Add node sequentially with auto-wiring               |
| `add(node, inputs=None)`                                  | `Graph`         | Add node with optional explicit wiring               |
| `parallel(nodes)`                                         | `Graph`         | Add nodes for parallel execution                     |
| `conditional(predicate, then_branch, else_branch=None)`   | `Graph`         | Add conditional branch                               |
| `compile(providers)`                                      | `CompiledGraph` | Validate and compile to executable DAG               |
| `run(image, providers, *, scheduler=None, device="auto")` | `MultiResult`   | Execute graph on image (delegates to `mata.infer()`) |
| `visualize(output_path)`                                  | `None`          | Generate graph visualization (DOT/PNG/PDF)           |

#### `run()`

Convenience method that delegates to `mata.infer()`. Allows fluent build-and-execute in a single expression.

```python
def run(
    image: Union[str, Path, PIL.Image.Image, np.ndarray],
    providers: Dict[str, Any],
    *,
    scheduler: Optional[Scheduler] = None,
    device: str = "auto",
    **kwargs,
) -> MultiResult
```

**Parameters:**

| Parameter   | Type                                     | Default           | Description                                       |
| ----------- | ---------------------------------------- | ----------------- | ------------------------------------------------- |
| `image`     | `str \| Path \| PIL.Image \| np.ndarray` | _required_        | Input image                                       |
| `providers` | `dict[str, Any]`                         | _required_        | Provider instances keyed by name (flat or nested) |
| `scheduler` | `Scheduler \| None`                      | `SyncScheduler()` | Execution strategy                                |
| `device`    | `str`                                    | `"auto"`          | Device placement                                  |

**Example — Fluent chained execution:**

```python
result = (Graph("top5")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(TopK(k=5, src="filtered", out="top5"))
    .then(Fuse(dets="top5", out="final"))
    .run("photo.jpg", providers={"detector": detector})
)
print(result.final)
```

**Example — Separate build and run:**

```python
graph = (Graph("pipeline")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(Fuse(dets="filtered", out="final"))
)
result = graph.run("photo.jpg", providers={"detector": detector})
```

**Example — With parallel scheduler:**

```python
result = graph.run(
    "scene.jpg",
    providers={"detector": detector, "depth": depth_model},
    scheduler=ParallelScheduler(),
)
```

---

### `CompiledGraph`

```python
from mata.core.graph import CompiledGraph
```

| Attribute           | Type                 | Description                 |
| ------------------- | -------------------- | --------------------------- |
| `name`              | `str`                | Graph name                  |
| `nodes`             | `list[Node]`         | Validated nodes             |
| `wiring`            | `dict[str, str]`     | Wiring connections          |
| `dag`               | `nx.DiGraph \| None` | NetworkX DAG (if available) |
| `validation_result` | `ValidationResult`   | Validation outcome          |
| `execution_order`   | `list[list[Node]]`   | Stages of parallel nodes    |

| Method                   | Returns            | Description        |
| ------------------------ | ------------------ | ------------------ |
| `get_parallel_stages()`  | `list[list[Node]]` | Execution stages   |
| `visualize(output_path)` | `None`             | Save visualization |

---

## Schedulers

### `SyncScheduler`

Sequential execution in topological order.

```python
from mata.core.graph import SyncScheduler

scheduler = SyncScheduler()
result = scheduler.execute(compiled, context, {"input.image": image})
```

### `ParallelScheduler`

Concurrent execution using `ThreadPoolExecutor` for independent stages.

```python
from mata.core.graph import ParallelScheduler

scheduler = ParallelScheduler(max_workers=4)
result = scheduler.execute(compiled, context, {"input.image": image})
```

### `OptimizedParallelScheduler`

Advanced scheduler with device placement and multi-GPU support.

```python
from mata.core.graph import OptimizedParallelScheduler

scheduler = OptimizedParallelScheduler(
    strategy="memory_aware",  # or "auto", "round_robin"
    unload_models=True,
)
result = scheduler.execute(compiled, context, {"input.image": image})
```

**Device strategies:**

- `"auto"` — Automatic device selection
- `"round_robin"` — Distribute across GPUs
- `"memory_aware"` — Place based on available GPU memory

---

## Execution Context

```python
from mata.core.graph import ExecutionContext

ctx = ExecutionContext(
    providers={"detect": {"detr": adapter}},
    device="auto",
    cache_artifacts=False,
)
```

| Method                               | Description                       |
| ------------------------------------ | --------------------------------- |
| `store(name, artifact)`              | Store artifact in context         |
| `retrieve(name)`                     | Retrieve artifact by name         |
| `has(name)`                          | Check artifact existence          |
| `get_provider(capability, name)`     | Get provider by capability + name |
| `record_metric(node, metric, value)` | Record execution metric           |
| `get_metrics()`                      | Get all collected metrics         |

| Attribute            | Type                | Description                           |
| -------------------- | ------------------- | ------------------------------------- |
| `device`             | `str`               | Resolved device (`"cuda"` or `"cpu"`) |
| `providers`          | `dict`              | Nested provider registry              |
| `metrics_collector`  | `MetricsCollector`  | Metrics subsystem                     |
| `tracer`             | `ExecutionTracer`   | Tracing subsystem                     |
| `provenance_tracker` | `ProvenanceTracker` | Provenance subsystem                  |

---

## Providers & Protocols

### Capability Protocols

All protocols are in `mata.core.registry.protocols` and are `@runtime_checkable`.

| Protocol              | Method Signature                                        | Returns           |
| --------------------- | ------------------------------------------------------- | ----------------- |
| `Detector`            | `predict(image: Image, **kw)`                           | `Detections`      |
| `Segmenter`           | `segment(image: Image, **kw)`                           | `Masks`           |
| `Classifier`          | `classify(image: Image, **kw)`                          | `Classifications` |
| `DepthEstimator`      | `estimate(image: Image, **kw)`                          | `DepthResult`     |
| `PoseEstimator`       | `estimate(image: Image, rois: ROIs = None, **kw)`       | `Keypoints`       |
| `Tracker`             | `update(dets: Detections, frame_id: str, **kw)`         | `Tracks`          |
| `Embedder`            | `embed(input: Image \| ROIs, **kw)`                     | `np.ndarray`      |
| `VisionLanguageModel` | `query(image: Image \| list[Image], prompt: str, **kw)` | `VisionResult`    |

### ProviderRegistry

```python
from mata.core.registry.providers import ProviderRegistry

registry = ProviderRegistry()
registry.register("detr", Detector, factory_fn, lazy=True)
detector = registry.get(Detector, "detr")
registry.list_providers(Detector)  # ["detr"]
registry.unregister(Detector, "detr")
```

| Method                                           | Description                   |
| ------------------------------------------------ | ----------------------------- |
| `register(name, capability, factory, lazy=True)` | Register provider             |
| `get(capability, name)`                          | Retrieve (triggers lazy load) |
| `list_providers(capability=None)`                | List by capability or all     |
| `unregister(capability, name)`                   | Remove provider               |
| `has(capability, name)`                          | Check existence               |

---

## Conditional Execution

```python
from mata.core.graph import If, Pass, HasLabel, CountAbove, ScoreAbove
```

### `If` Node

```python
If(
    predicate: Callable[[ExecutionContext], bool],
    then_branch: Node,
    else_branch: Node = Pass(),
)
```

### Built-in Predicates

| Predicate    | Signature                    | Condition                                      |
| ------------ | ---------------------------- | ---------------------------------------------- |
| `HasLabel`   | `HasLabel(src, label)`       | Label present in detections (case-insensitive) |
| `CountAbove` | `CountAbove(src, n)`         | Detection count > n                            |
| `ScoreAbove` | `ScoreAbove(src, threshold)` | Max score > threshold                          |

**Functional helpers:** `has_label(src, label)`, `count_above(src, n)`, `score_above(src, threshold)` — return predicate instances.

### `Pass` Node

No-op node for else branches that should do nothing.

---

## Temporal / Video

```python
from mata.core.graph import (
    VideoProcessor, Window,
    FramePolicyEveryN, FramePolicyLatest, FramePolicyQueue,
)
```

### Frame Policies

| Class               | Constructor                      | Description                            |
| ------------------- | -------------------------------- | -------------------------------------- |
| `FramePolicyEveryN` | `FramePolicyEveryN(n=5)`         | Process every N-th frame               |
| `FramePolicyLatest` | `FramePolicyLatest()`            | Keep only the latest frame (real-time) |
| `FramePolicyQueue`  | `FramePolicyQueue(max_queue=10)` | Queue up to N frames                   |

### `VideoProcessor`

```python
processor = VideoProcessor(
    graph=compiled_graph,
    providers={"detect": {"detr": adapter}},
    frame_policy=FramePolicyEveryN(n=5),
    scheduler=SyncScheduler(),
)
```

| Method                                  | Returns             | Description                |
| --------------------------------------- | ------------------- | -------------------------- |
| `process_video(path, output_path=None)` | `list[MultiResult]` | Process video file         |
| `process_stream(source, callback)`      | `None`              | Process RTSP/camera stream |

### `Window` Node

Buffers N frames for temporal operations.

```python
Window(n: int = 8)
```

| I/O    | Name     | Type          |
| ------ | -------- | ------------- |
| Input  | `image`  | `Image`       |
| Output | `images` | `list[Image]` |

---

## Observability

### `MetricsCollector`

```python
from mata.core.graph import MetricsCollector
```

| Method                        | Description                   |
| ----------------------------- | ----------------------------- |
| `start()`                     | Begin timing                  |
| `stop()`                      | End timing                    |
| `record_latency(node, ms)`    | Record node latency           |
| `record_memory(node, mb)`     | Record memory usage           |
| `record_gpu_memory(node, mb)` | Record GPU memory             |
| `record(node, metric, value)` | Record custom metric          |
| `get_node_metrics(node)`      | Get metrics for one node      |
| `get_summary()`               | Get aggregated summary        |
| `export(format)`              | Export as `"json"` or `"csv"` |

**Property:** `wall_time_ms` — Total wall clock time.

### `ExecutionTracer`

```python
from mata.core.graph import ExecutionTracer, Span
```

| Method                                              | Description     |
| --------------------------------------------------- | --------------- |
| `start_span(name, parent_id=None, attributes=None)` | Create span     |
| `end_span(span, status="ok", error_message=None)`   | End span        |
| `add_span_attribute(span, key, value)`              | Attach metadata |
| `get_span(span_id)`                                 | Retrieve span   |
| `export_trace(format)`                              | Export as JSON  |

**Properties:** `spans` (all spans), `active_spans` (unfinished spans).

**`Span` attributes:** `span_id`, `name`, `parent_id`, `start_time`, `end_time`, `status`, `duration_ms`.

### `ProvenanceTracker`

```python
from mata.core.graph import ProvenanceTracker
```

| Method                                   | Description                    |
| ---------------------------------------- | ------------------------------ |
| `record_model(name, adapter)`            | Record model info from adapter |
| `record_model_info(name, model_id, ...)` | Record model info manually     |
| `record_graph(compiled_graph)`           | Record graph structure         |
| `get_provenance()`                       | Get full provenance dict       |
| `export(format)`                         | Export as JSON                 |

---

## DSL Helpers

```python
from mata.core.graph.dsl import NodePipe, out, bind, sequential, parallel_tasks, pipeline
```

### `NodePipe`

Chain nodes with `>>` operator:

```python
pipe = NodePipe(Detect(using="detr")) >> Filter(score_gt=0.5) >> Fuse()
graph = pipe.build(name="my_graph")
```

### Helper Functions

| Function                           | Description                                    |
| ---------------------------------- | ---------------------------------------------- |
| `out(node, name)`                  | Override node's output artifact name           |
| `bind(node, **inputs)`             | Bind specific input artifact sources to a node |
| `sequential(nodes, name=None)`     | Build graph from sequential list               |
| `parallel_tasks(nodes, name=None)` | Build graph with parallel nodes                |
| `pipeline(stages, name=None)`      | Build multi-stage graph                        |

---

## Presets

```python
from mata.presets import (
    grounding_dino_sam,
    segment_and_refine,
    detection_pose,
    full_scene_analysis,
    detect_and_track,
    vlm_grounded_detection,
    vlm_scene_understanding,
    vlm_multi_image_comparison,
)
```

### Traditional CV Presets

#### `grounding_dino_sam()`

```python
grounding_dino_sam(
    detection_threshold: float = 0.3,
    nms_iou_threshold: float = None,
    refine_method: str = "morph_close",
    refine_radius: int = 3,
) -> Graph
```

**Pipeline:** Detect → Filter → [NMS] → PromptBoxes → RefineMask → Fuse  
**Providers:** `"detector"`, `"segmenter"`

#### `segment_and_refine()`

```python
segment_and_refine(
    segmentation_method: str = "everything",
    refine_method: str = "morph_close",
    refine_radius: int = 3,
) -> Graph
```

**Pipeline:** SegmentEverything → RefineMask → Fuse  
**Providers:** `"segmenter"`

#### `detection_pose()`

```python
detection_pose(
    detection_threshold: float = 0.5,
    person_only: bool = True,
    top_k: int = None,
    nms_iou_threshold: float = 0.5,
) -> Graph
```

**Pipeline:** Detect → Filter [person] → [NMS] → [TopK] → Fuse  
**Providers:** `"detector"`, `"pose"`

#### `full_scene_analysis()`

```python
full_scene_analysis(
    detection_threshold: float = 0.3,
    classification_labels: list[str] = None,
) -> Graph
```

**Pipeline:** Parallel(Detect, Classify, EstimateDepth) → Filter → Fuse  
**Providers:** `"detector"`, `"classifier"`, `"depth"`

#### `detect_and_track()`

```python
detect_and_track(
    detection_threshold: float = 0.5,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph
```

**Pipeline:** Detect → Filter → Track → Fuse  
**Providers:** `"detector"`, `"tracker"`

### VLM Presets

#### `vlm_grounded_detection()`

```python
vlm_grounded_detection(
    vlm_prompt: str = ...,
    detection_threshold: float = 0.3,
    match_strategy: str = "label_fuzzy",
    auto_promote: bool = False,
) -> Graph
```

**Pipeline:** Parallel(VLMDetect, Detect) → Filter → PromoteEntities → Fuse  
**Providers:** `"vlm"`, `"detector"`

#### `vlm_scene_understanding()`

```python
vlm_scene_understanding(
    describe_prompt: str = ...,
    detection_threshold: float = 0.3,
    classification_labels: list[str] = None,
) -> Graph
```

**Pipeline:** Parallel(VLMDescribe, Detect, EstimateDepth, [Classify]) → Fuse  
**Providers:** `"vlm"`, `"detector"`, `"depth"`, optional `"classifier"`

#### `vlm_multi_image_comparison()`

```python
vlm_multi_image_comparison(
    prompt: str = ...,
    output_mode: str = None,
) -> Graph
```

**Pipeline:** VLMQuery → Fuse  
**Provider:** `"vlm"`

---

## Converters & Utilities

```python
from mata.core.artifacts import (
    # VisionResult conversions
    vision_result_to_detections,
    detections_to_vision_result,
    vision_result_to_masks,
    masks_to_vision_result,
    classify_result_to_artifact,
    artifact_to_classify_result,
    depth_result_to_artifact,
    artifact_to_depth_result,
    # Instance ID management
    generate_instance_ids,
    ensure_instance_ids,
    align_instance_ids,
    # Entity promotion
    promote_entities_to_instances,
    match_entity_to_instance,
    merge_entity_attributes,
    auto_promote_vision_result,
)
```

### VisionResult ↔ Detections

| Function                            | Description                                |
| ----------------------------------- | ------------------------------------------ |
| `vision_result_to_detections(vr)`   | Convert VisionResult → Detections artifact |
| `detections_to_vision_result(dets)` | Convert Detections → VisionResult          |
| `vision_result_to_masks(vr)`        | Convert VisionResult → Masks artifact      |
| `masks_to_vision_result(masks)`     | Convert Masks → VisionResult               |

### Instance ID Management

| Function                                     | Description                        |
| -------------------------------------------- | ---------------------------------- |
| `generate_instance_ids(n, prefix="inst")`    | Generate N stable instance IDs     |
| `ensure_instance_ids(instances)`             | Add IDs if missing                 |
| `align_instance_ids(artifacts1, artifacts2)` | Align IDs across two artifact sets |

### Entity Promotion (VLM)

| Function                                                | Description                                             |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `promote_entities_to_instances(entities, spatial)`      | Batch promote entities using spatial data               |
| `match_entity_to_instance(entity, instances, strategy)` | Find best match for one entity                          |
| `merge_entity_attributes(instance, entity)`             | Merge entity attributes into instance                   |
| `auto_promote_vision_result(result)`                    | Auto-promote entities with spatial data in VisionResult |
