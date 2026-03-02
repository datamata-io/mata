# MATA Graph System Architecture Guide

> **Version**: 1.6.0 | **Last Updated**: February 12, 2026

## Overview

The MATA Graph System is a **strongly-typed, task-centric** framework for building multi-task computer vision workflows. It enables composing detection, segmentation, classification, depth estimation, tracking, and VLM tasks into validated, executable directed acyclic graphs (DAGs).

### Design Principles

1. **Task contracts over model specifics** — Nodes and providers communicate through typed artifacts, not model-specific formats
2. **Immutability** — All artifacts are frozen dataclasses; operations return new instances
3. **Compile-time validation** — Graphs are validated at compile time (type safety, cycle detection, dependency resolution) before execution
4. **Zero overhead philosophy** — Graph orchestration adds <10% overhead vs direct adapter calls
5. **Backward compatible** — The existing `mata.load()` / `mata.run()` APIs remain unchanged; the graph system is additive

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Artifact** | Strongly-typed, immutable data container (Image, Detections, Masks, etc.) |
| **Node** | Processing unit with declared typed inputs/outputs and a `run()` method |
| **Graph** | Fluent builder that chains nodes into a validated DAG |
| **Provider** | Task-specific adapter (detector, segmenter, etc.) accessed via capability protocols |
| **Scheduler** | Executes compiled graphs (sync, parallel, or optimized) |
| **ExecutionContext** | Runtime state: artifact storage, provider access, metrics, tracing |

---

## Architecture

```
                       ┌─────────────────────┐
                       │   mata.infer() API   │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      Graph Builder          │
                    │  .then() .parallel()        │
                    │  .conditional()             │
                    └─────────────┬──────────────┘
                                  │ compile()
                    ┌─────────────▼──────────────┐
                    │     GraphValidator          │
                    │  • Type checking            │
                    │  • Cycle detection          │
                    │  • Dependency resolution    │
                    │  • Provider capability check│
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     CompiledGraph (DAG)     │
                    │  • Topological ordering     │
                    │  • Parallel stage groups    │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       Scheduler             │
                    │  Sync │ Parallel │ Optimized│
                    └─────────────┬──────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
   ┌──────▼──────┐       ┌───────▼──────┐       ┌───────▼──────┐
   │   Node A     │       │   Node B     │       │   Node C     │
   │ (Detect)     │       │ (Classify)   │       │ (Depth)      │
   └──────┬──────┘       └───────┬──────┘       └───────┬──────┘
          │                       │                       │
          │          ExecutionContext                      │
          │  ┌─────────────────────────────────────┐     │
          └──► Artifact Store │ Providers │ Metrics ◄────┘
             └─────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       MultiResult           │
                    │  channels + provenance +    │
                    │  metrics                    │
                    └────────────────────────────┘
```

---

## Artifact Type System

Artifacts are the strongly-typed data containers that flow between nodes. All artifacts inherit from the `Artifact` base class and are **frozen dataclasses** (immutable).

### Artifact Hierarchy

```
Artifact (ABC)
├── Image          — Multi-format image (PIL/numpy/tensor) with lazy conversion
├── Detections     — Bounding boxes + entities with instance IDs
├── Masks          — Per-instance segmentation masks (RLE/polygon/binary)
├── Classifications — Sorted class predictions (label + score)
├── DepthMap       — Per-pixel depth values (H×W float array)
├── Keypoints      — Per-instance keypoint arrays (N×3: x, y, score)
├── Tracks         — Temporal track objects with state management
├── ROIs           — Cropped image regions mapped to instance IDs
└── MultiResult    — Channel-based result bundle with provenance
```

### Core Properties

| Property | Description |
|----------|-------------|
| **Immutable** | All artifacts use `frozen=True` dataclasses. Operations return new instances. |
| **Serializable** | Every artifact implements `to_dict()` / `from_dict()` for JSON serialization. |
| **Validatable** | Optional `validate()` hook for custom assertion logic. |
| **ID-tracked** | Detections, Masks, Keypoints, Tracks carry stable `instance_ids` for cross-referencing. |

### Entity vs Instance (VLM Integration)

The graph system distinguishes between two types of detections:

| | Entity | Instance |
|---|--------|----------|
| **Source** | VLM text output | Traditional CV models |
| **Data** | Label + score + attributes | Label + score + bbox + mask |
| **Spatial?** | No (semantic only) | Yes (bounding box required) |
| **Use case** | Semantic understanding | Spatial localization |

The `Detections` artifact carries **both** `instances` and `entities` fields, enabling mixed VLM→spatial fusion workflows via the `PromoteEntities` node.

---

## Node Base Class

Nodes are the processing units of the graph. Each node declares its input/output types and implements a `run()` method.

### Anatomy of a Node

```python
from mata.core.graph import Node
from mata.core.artifacts import Image, Detections

class Detect(Node):
    # 1. Declare typed I/O at class level
    inputs = {"image": Image}
    outputs = {"detections": Detections}

    # 2. Configure via __init__
    def __init__(self, using: str, out: str = "dets", **kwargs):
        super().__init__(name="Detect")
        self.provider_name = using
        self.output_name = out     # Dynamic output key
        self.kwargs = kwargs

    # 3. Implement execution logic
    def run(self, ctx, image: Image) -> dict[str, Artifact]:
        detector = ctx.get_provider("detect", self.provider_name)
        result = detector.predict(image, **self.kwargs)
        detections = Detections.from_vision_result(result)
        return {self.output_name: detections}
```

### Key Node Design Patterns

- **`using` parameter** — Names the provider to look up from `ExecutionContext`
- **`out` parameter** — Dynamic output artifact name (overrides class-level key)
- **`src` parameter** — Names the input artifact to read from context (for transform nodes)
- **Input validation** — `validate_inputs()` checks types match declared signature
- **Output validation** — `validate_outputs()` checks returned artifacts match types

### Node Categories

| Category | Nodes | Description |
|----------|-------|-------------|
| **Task** | Detect, Classify, SegmentImage, EstimateDepth | Run model inference via providers |
| **Transform** | Filter, TopK, ExtractROIs, ExpandBoxes | Transform artifacts without models |
| **Prompt** | PromptBoxes, PromptPoints, SegmentEverything | Feed prompts to SAM-style models |
| **Refinement** | RefineMask, MaskToBox | Post-process masks and boxes |
| **Tracking** | Track | Temporal tracking across frames |
| **Fusion** | Fuse, Merge | Bundle or merge artifacts |
| **VLM** | VLMDescribe, VLMDetect, VLMQuery, PromoteEntities | Vision-language model nodes |
| **Visualization** | Annotate, NMS | Render results and filter overlaps |
| **Control** | If, Pass | Conditional execution branches |
| **Temporal** | Window | Frame buffering for video |

---

## Graph Building

The `Graph` class provides a fluent API for composing nodes into validated workflows.

### Sequential Pipeline

```python
from mata.core.graph import Graph
from mata.nodes import Detect, Filter, Fuse

graph = (Graph("detect_pipeline")
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(Fuse(dets="filtered", out="final"))
)
```

Nodes added with `.then()` are **auto-wired**: each node's inputs are connected to the previous node's outputs when types match.

### Parallel Execution

```python
from mata.nodes import Detect, Classify, EstimateDepth

graph = (Graph("parallel_tasks")
    .parallel([
        Detect(using="detr", out="dets"),
        Classify(using="clip", out="cls"),
        EstimateDepth(using="depth", out="depth"),
    ])
    .then(Fuse(dets="dets", classification="cls", depth="depth", out="scene"))
)
```

Parallel nodes have no dependencies between them and can execute simultaneously when using `ParallelScheduler`.

### Explicit Wiring

```python
graph = Graph()
graph.add(Detect(using="detr", out="dets"), inputs={"image": "input.image"})
graph.add(Filter(src="dets", out="filtered"), inputs={"detections": "Detect.dets"})
```

Use `.add()` with explicit `inputs` when auto-wiring is insufficient. Sources follow the format `"NodeName.output_name"` or `"input.artifact_name"` for graph inputs.

### Conditional Branching

```python
from mata.core.graph import If, HasLabel, Pass

graph = (Graph("conditional")
    .then(Detect(using="detr", out="dets"))
    .then(If(
        predicate=HasLabel("dets", "person"),
        then_branch=SegmentImage(using="sam", out="masks"),
        else_branch=Pass()
    ))
)
```

Built-in predicates: `HasLabel`, `CountAbove`, `ScoreAbove`. Custom predicates are any callable `(ctx) -> bool`.

### DSL Helpers

```python
from mata.core.graph.dsl import NodePipe, out, bind, sequential, parallel_tasks

# Pipe syntax
pipeline = (
    NodePipe(Detect(using="detr", out="dets"))
    >> Filter(src="dets", score_gt=0.5, out="filtered")
    >> Fuse(dets="filtered", out="final")
)
graph = pipeline.build(name="my_pipeline")

# Sequential helper
graph = sequential([
    Detect(using="detr", out="dets"),
    Filter(src="dets", score_gt=0.5, out="filtered"),
], name="seq_graph")

# Parallel helper
graph = parallel_tasks([
    Detect(using="detr", out="dets"),
    EstimateDepth(using="depth", out="depth"),
], name="par_graph")
```

---

## Graph Compilation

Calling `graph.compile(providers)` validates the graph and produces a `CompiledGraph`:

```python
compiled = graph.compile(providers={
    "detect": {"detr": detr_adapter},
    "segment": {"sam": sam_adapter},
})
```

### Validation Checks

The `GraphValidator` runs the following checks at compile time:

| Check | Description |
|-------|-------------|
| **Name collisions** | All node names must be unique |
| **Dependency resolution** | All input artifacts must be provided by upstream nodes or graph inputs |
| **Type compatibility** | Connected artifacts must have compatible types |
| **Cycle detection** | Graph must be a DAG (no circular dependencies) |
| **Provider capabilities** | Referenced providers must exist in the registry |

If validation fails, a `ValidationError` is raised with detailed error messages:

```
Graph validation failed:
  1. Node 'Filter' requires input 'detections' but no upstream node produces it
  2. Type mismatch: 'Detect.dets' produces Detections but 'Depth.image' expects Image
```

### Execution Order

The `CompiledGraph` computes an optimal execution order by performing topological sort on the DAG. Nodes with no inter-dependencies are grouped into parallel stages:

```
Stage 0: [Detect, EstimateDepth]  ← Can run in parallel
Stage 1: [Filter]                  ← Depends on Detect
Stage 2: [Fuse]                    ← Depends on Filter + EstimateDepth
```

Uses NetworkX (if available) or a built-in Kahn's algorithm fallback.

---

## Execution & Scheduling

### Schedulers

| Scheduler | Description | Use Case |
|-----------|-------------|----------|
| `SyncScheduler` | Sequential execution in topological order | Default, simplest, debugging |
| `ParallelScheduler` | ThreadPoolExecutor for parallel stages | Independent tasks (detect + classify + depth) |
| `OptimizedParallelScheduler` | Device placement, multi-GPU, model unloading | Production with multiple GPUs |

### Execution Flow

1. **Store initial artifacts** — Input image stored as `"input.image"` in context
2. **Execute stages** — Each stage's nodes run with inputs resolved from context
3. **Resolve inputs** — For each node, follow wiring to find source artifacts
4. **Validate** — Input and output types checked against declarations
5. **Store outputs** — Outputs stored in context under both qualified (`Node.name`) and unqualified names
6. **Collect metrics** — Per-node latency, memory usage recorded
7. **Build result** — Final `MultiResult` assembled with channels, provenance, and metrics

```python
from mata.core.graph import SyncScheduler, ExecutionContext

ctx = ExecutionContext(
    providers={"detect": {"detr": adapter}},
    device="auto",
)
scheduler = SyncScheduler()
result = scheduler.execute(compiled, ctx, {"input.image": image_artifact})
```

### ExecutionContext

The `ExecutionContext` is the runtime backbone, providing:

- **Artifact storage**: `ctx.store(name, artifact)` / `ctx.retrieve(name)`
- **Provider access**: `ctx.get_provider(capability, name)`
- **Metrics**: `ctx.record_metric(node, metric, value)`
- **Device**: `ctx.device` — resolved device string (`"cuda"` or `"cpu"`)
- **Observability**: Built-in `MetricsCollector`, `ExecutionTracer`, `ProvenanceTracker`

---

## Provider System

Providers are task-specific adapters accessed through **capability protocols** (PEP 544 runtime-checkable protocols).

### Capability Protocols

| Protocol | Method | Returns | Task |
|----------|--------|---------|------|
| `Detector` | `predict(image, **kw)` | `Detections` | Object detection |
| `Segmenter` | `segment(image, **kw)` | `Masks` | Image segmentation |
| `Classifier` | `classify(image, **kw)` | `Classifications` | Image classification |
| `DepthEstimator` | `estimate(image, **kw)` | `DepthResult` | Depth estimation |
| `PoseEstimator` | `estimate(image, rois, **kw)` | `Keypoints` | Pose estimation |
| `Tracker` | `update(dets, frame_id, **kw)` | `Tracks` | Object tracking |
| `Embedder` | `embed(input, **kw)` | `np.ndarray` | Feature extraction |
| `VisionLanguageModel` | `query(image, prompt, **kw)` | `VisionResult` | VLM queries |

Any object implementing the required method signature is a valid provider (duck typing with runtime verification):

```python
from mata.core.registry.protocols import Detector

class MyCustomDetector:
    def predict(self, image, **kwargs):
        # Your detection logic
        return detections

detector = MyCustomDetector()
assert isinstance(detector, Detector)  # ✅ Runtime check passes
```

### ProviderRegistry

The `ProviderRegistry` manages providers with:

- **Capability-based indexing**: Organized by protocol type
- **Lazy loading**: Factory functions called on first access
- **Thread-safe**: All operations protected by lock
- **Protocol verification**: Optional runtime isinstance check

```python
from mata.core.registry.providers import ProviderRegistry
from mata.core.registry.protocols import Detector

registry = ProviderRegistry()
registry.register("detr", Detector, lambda: load_model(), lazy=True)
detector = registry.get(Detector, "detr")  # Factory called here
```

---

## Observability

The graph system includes three observability subsystems, all automatically integrated into the execution flow.

### MetricsCollector

Collects per-node execution metrics:

```python
collector = MetricsCollector()
collector.start()
collector.record_latency("Detect", 45.2)
collector.record_memory("Detect", 512.0)
collector.record_gpu_memory("Detect", 1024.0)
collector.stop()

summary = collector.get_summary()
# {"total_latency_ms": 45.2, "wall_time_ms": 46.1, ...}
```

### ExecutionTracer

Records hierarchical execution spans:

```python
tracer = ExecutionTracer()
span = tracer.start_span("graph:pipeline", attributes={"num_nodes": 3})
child = tracer.start_span("node:Detect", parent_id=span.span_id)
tracer.end_span(child)
tracer.end_span(span)

trace_json = tracer.export_trace("json")
```

### ProvenanceTracker

Tracks model versions and graph configuration for reproducibility:

```python
tracker = ProvenanceTracker()
tracker.record_model("detector", adapter)
tracker.record_graph(compiled_graph)
provenance = tracker.get_provenance()
# {"models": {"detector": {"model_type": "HuggingFace", "hash": "abc123"}}, ...}
```

---

## Video / Temporal Processing

The `VideoProcessor` drives a compiled graph over video frames with configurable frame policies.

### Frame Policies

| Policy | Description |
|--------|-------------|
| `FramePolicyEveryN(n=5)` | Process every N-th frame |
| `FramePolicyLatest()` | Drop old frames for real-time (camera/RTSP) |
| `FramePolicyQueue(max_queue=10)` | Queue up to N frames, drop when full |

### VideoProcessor

```python
from mata.core.graph.temporal import VideoProcessor, FramePolicyEveryN

processor = VideoProcessor(
    graph=compiled,
    providers={"detect": {"detr": adapter}},
    frame_policy=FramePolicyEveryN(n=5),
)

# Process video file
results = processor.process_video("input.mp4", output_path="output.mp4")

# Process live stream
processor.process_stream("rtsp://camera/stream", callback=handle_result)
```

### Window Node

Buffers N frames for temporal operations (for tracking or temporal aggregation):

```python
from mata.core.graph.temporal import Window

graph = (Graph("tracking")
    .then(Window(n=8))            # Buffer 8 frames
    .then(Detect(using="detr"))
    .then(Track(using="tracker"))
)
```

---

## Multi-Task Result Access

All graph executions return a `MultiResult` — a channel-based result bundle:

```python
result = mata.infer("image.jpg", graph, providers=providers)

# Attribute access for channels
dets = result.dets              # or result.channels["dets"]
masks = result.masks
depth = result.depth

# Check channel existence
if result.has_channel("keypoints"):
    kpts = result.keypoints

# Cross-reference by instance ID
instance_data = result.get_instance_artifacts("inst_0000")
# {"dets": Instance(...), "masks": Instance(...)}

# Metadata
result.metrics     # Per-node timing
result.provenance  # Model versions, graph hash, timestamps

# Serialization
json_str = result.to_json()
```

---

## Presets

Pre-built graph factories for common workflows:

| Preset | Description | Providers Needed |
|--------|-------------|------------------|
| `grounding_dino_sam()` | Detect → Filter → SAM segment → Refine | detector, segmenter |
| `segment_and_refine()` | Segment everything → Refine masks | segmenter |
| `detection_pose()` | Detect → Filter persons → Pose estimate | detector, pose |
| `full_scene_analysis()` | Parallel detect + classify + depth | detector, classifier, depth |
| `detect_and_track()` | Detect → Filter → Track | detector, tracker |
| `vlm_grounded_detection()` | VLM → Detect → Promote entities | vlm, detector |
| `vlm_scene_understanding()` | Parallel VLM + detect + depth | vlm, detector, depth |
| `vlm_multi_image_comparison()` | VLM multi-image query | vlm |

```python
from mata.presets import grounding_dino_sam

graph = grounding_dino_sam(detection_threshold=0.3, refine_method="morph_close")
result = mata.infer("image.jpg", graph, providers={
    "detector": mata.load("detect", "IDEA-Research/grounding-dino-tiny"),
    "segmenter": mata.load("segment", "facebook/sam-vit-base"),
})
```

---

## Type Safety

### How Type Checking Works

1. **Declaration** — Nodes declare `inputs` and `outputs` as `Dict[str, Type[Artifact]]`
2. **Compile-time** — `GraphValidator.check_type_compatibility()` verifies connected artifact types
3. **Runtime** — `Node.validate_inputs()` / `validate_outputs()` check actual artifact isinstance
4. **Dynamic outputs** — When nodes use `out` parameters, the validator registers both static and dynamic names

### Common Type Errors

```
# Connecting detection output to image input
Type mismatch: 'Detect.dets' produces Detections but 'SegmentImage.image' expects Image

# Missing dependency
Node 'Filter' requires input 'detections' but no upstream node produces it

# Cycle
Circular dependency detected: A → B → C → A
```

### Type Compatibility Rules

- Exact type match is required (e.g., `Detections` → `Detections`)
- Subclass matching is supported (any `Artifact` subclass accepted where `Artifact` is declared)
- Dynamic output names are resolved at compile time from node `output_name` / `out` attributes

---

## Error Handling

### Compile-Time Errors

Raised by `GraphValidator` during `graph.compile()`:

- `ValidationError` with numbered list of all errors and warnings
- Includes suggested fixes in error messages

### Runtime Errors

The scheduler catches and re-raises node execution errors with context:

- Node name and type included in traceback
- Execution tracer records error spans for debugging
- Partial results may be available in the execution context

### Custom Exceptions

```python
from mata.core.exceptions import (
    ValidationError,     # Graph validation failures
    ModelNotFoundError,  # Provider not found
    MATAError,           # Base exception for all MATA errors
)
```

---

## Performance Considerations

| Aspect | Target | Notes |
|--------|--------|-------|
| Graph compilation | <100ms | Topological sort + validation |
| Graph overhead | <10% | vs direct adapter calls |
| Parallel speedup | 1.5–3× | For independent tasks (model dependent) |
| Video processing | >20 FPS | RT-DETR + BYTETrack on GPU |
| Memory | Efficient | Model unloading support in OptimizedParallelScheduler |

### Optimization Tips

1. **Use `ParallelScheduler`** for graphs with independent branches
2. **Use `OptimizedParallelScheduler`** for multi-GPU workloads with device placement
3. **Apply `FramePolicyEveryN`** for video to skip frames
4. **Use `Filter` early** in the pipeline to reduce work for downstream nodes
5. **Enable `cache_artifacts=True`** in `ExecutionContext` for repeated queries

---

## Further Reading

- [API Reference](GRAPH_API_REFERENCE.md) — Complete reference for all nodes, artifacts, and APIs
- [Cookbook](GRAPH_COOKBOOK.md) — Recipes and patterns for common workflows
- [Migration Guide](MIGRATION_GUIDE.md) — Migrating from v1.5 standalone to v1.6 graph system
- [Status](STATUS.md) — Current implementation status
