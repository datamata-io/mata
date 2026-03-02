# MATA Graph Cookbook

> **Version**: 1.6.0 | **Last Updated**: February 12, 2026

Practical recipes and patterns for common computer vision workflows using the MATA graph system.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Detection Workflows](#detection-workflows)
3. [Segmentation Workflows](#segmentation-workflows)
4. [Classification & Depth](#classification--depth)
5. [Multi-Task Parallel Pipelines](#multi-task-parallel-pipelines)
6. [Conditional Execution](#conditional-execution)
7. [Video & Tracking](#video--tracking)
8. [VLM Workflows](#vlm-workflows)
9. [Custom Nodes & Providers](#custom-nodes--providers)
10. [Performance Optimization](#performance-optimization)
11. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Getting Started

### Recipe 1: Minimal Detection Pipeline

The simplest graph — detect objects and collect results.

```python
import mata
from mata.nodes import Detect, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Fuse(dets="dets", out="final"),
    ],
    providers={"detector": detector},
)

for inst in result.final.instances:
    print(f"{inst.label_name}: {inst.score:.2f} at {inst.bbox}")
```

### Recipe 2: Detection with Filtering

Filter low-confidence detections and keep only specific classes.

```python
from mata.nodes import Detect, Filter, Fuse

result = mata.infer(
    image="street.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.5, label_in=["person", "car"], out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"detector": detector},
)

print(f"Found {len(result.final.instances)} objects above 0.5 confidence")
```

### Recipe 3: Using the Graph Builder

Build graphs with the fluent `Graph` API instead of node lists.

```python
from mata.core.graph import Graph
from mata.nodes import Detect, Filter, TopK, Fuse

graph = (Graph("top5_detections")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(TopK(k=5, src="filtered", out="top5"))
    .then(Fuse(dets="top5", out="final"))
)

result = mata.infer("photo.jpg", graph, providers={"detector": detector})
```

### Recipe 3b: Fluent Build-and-Run with `Graph.run()`

Execute a graph directly without importing `mata.infer()` — build and run in a single expression.

```python
from mata.core.graph import Graph
from mata.nodes import Detect, Filter, TopK, Fuse

# Single expression: build and run
result = (Graph("top5_detections")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(TopK(k=5, src="filtered", out="top5"))
    .then(Fuse(dets="top5", out="final"))
    .run("photo.jpg", providers={"detector": detector})
)

for inst in result.final.instances:
    print(f"{inst.label_name}: {inst.score:.2f}")
```

Or build separately and run later:

```python
graph = (Graph("pipeline")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(Fuse(dets="filtered", out="final"))
)

# Run with different images
result1 = graph.run("photo1.jpg", providers={"detector": detector})
result2 = graph.run("photo2.jpg", providers={"detector": detector})
```

> **Note:** `Graph.run()` delegates to `mata.infer()` internally. Both APIs are equivalent — use whichever style you prefer.

### Recipe 4: Using the Pipe DSL

Chain nodes with the `>>` operator for compact syntax.

```python
from mata.core.graph.dsl import NodePipe
from mata.nodes import Detect, Filter, Fuse

graph = (
    NodePipe(Detect(using="detector", out="dets"))
    >> Filter(src="dets", score_gt=0.5, out="filtered")
    >> Fuse(dets="filtered", out="final")
).build(name="pipe_graph")

result = mata.infer("photo.jpg", graph, providers={"detector": detector})
```

---

## Detection Workflows

### Recipe 5: Zero-Shot Detection with GroundingDINO

Detect objects by text description — no class labels required.

```python
from mata.nodes import Detect, Filter, Fuse

detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

result = mata.infer(
    image="kitchen.jpg",
    graph=[
        Detect(using="detector", text_prompts="coffee mug . plate . fork", out="dets"),
        Filter(src="dets", score_gt=0.3, out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"detector": detector},
)
```

### Recipe 6: NMS Filtering for Dense Detections

Remove redundant overlapping boxes.

```python
from mata.nodes import Detect, NMS, Filter, Fuse

result = mata.infer(
    image="crowd.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        NMS(iou_threshold=0.5, out="nms_dets"),
        Filter(src="nms_dets", score_gt=0.4, out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"detector": detector},
)
```

### Recipe 7: Extract Object Crops (ROIs)

Crop detected objects for per-region analysis.

```python
from mata.nodes import Detect, Filter, ExtractROIs, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.5, out="filtered"),
        ExtractROIs(src_dets="filtered", padding=10, out="rois"),
        Fuse(dets="filtered", rois="rois", out="final"),
    ],
    providers={"detector": detector},
)

# Access cropped regions
for roi, inst_id in zip(result.final.roi_images, result.final.instance_ids):
    print(f"Crop for {inst_id}: {roi.size}")
```

---

## Segmentation Workflows

### Recipe 8: Detection → SAM Segmentation

Use detection boxes as prompts for SAM.

```python
from mata.nodes import Detect, Filter, PromptBoxes, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")
segmenter = mata.load("segment", "facebook/sam-vit-base")

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.5, out="filtered"),
        PromptBoxes(using="segmenter", dets_src="filtered", out="masks"),
        Fuse(dets="filtered", masks="masks", out="final"),
    ],
    providers={"detector": detector, "segmenter": segmenter},
)

print(f"Segmented {len(result.final.instances)} objects")
```

### Recipe 9: GroundingDINO + SAM (Preset)

Use the pre-built preset for the most common detect+segment workflow.

```python
from mata.presets import grounding_dino_sam

detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
segmenter = mata.load("segment", "facebook/sam-vit-base")

result = mata.infer(
    image="photo.jpg",
    grounding_dino_sam(detection_threshold=0.3, refine_method="morph_close"),
    providers={"detector": detector, "segmenter": segmenter},
)
```

### Recipe 10: Segment Everything

Generate all possible masks without prompts.

```python
from mata.nodes import SegmentEverything, Fuse

segmenter = mata.load("segment", "facebook/sam-vit-base")

result = mata.infer(
    image="photo.jpg",
    graph=[
        SegmentEverything(using="segmenter", out="masks"),
        Fuse(masks="masks", out="final"),
    ],
    providers={"segmenter": segmenter},
)
```

### Recipe 11: Segment + Refine Masks

Apply morphological operations to clean up segmentation masks.

```python
from mata.nodes import SegmentEverything, RefineMask, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        SegmentEverything(using="segmenter", out="masks"),
        RefineMask(src="masks", method="morph_close", radius=5, out="refined"),
        Fuse(masks="refined", out="final"),
    ],
    providers={"segmenter": segmenter},
)
```

### Recipe 12: Masks → Bounding Boxes

Extract tight bounding boxes from segmentation masks.

```python
from mata.nodes import SegmentEverything, MaskToBox, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        SegmentEverything(using="segmenter", out="masks"),
        MaskToBox(src="masks", out="box_dets"),
        Fuse(detections="box_dets", masks="masks", out="final"),
    ],
    providers={"segmenter": segmenter},
)
```

### Recipe 13: Point-Prompted Segmentation

Segment specific regions using explicit point coordinates.

```python
from mata.nodes import PromptPoints, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        PromptPoints(
            using="segmenter",
            points=[(320, 240, 1), (100, 100, 0)],  # (x, y, label) — 1=foreground, 0=background
            out="masks",
        ),
        Fuse(masks="masks", out="final"),
    ],
    providers={"segmenter": segmenter},
)
```

---

## Classification & Depth

### Recipe 14: Zero-Shot Classification with CLIP

Classify images using text prompts — no predefined labels needed.

```python
from mata.nodes import Classify, Fuse

classifier = mata.load("classify", "openai/clip-vit-base-patch32")

result = mata.infer(
    image="photo.jpg",
    graph=[
        Classify(
            using="classifier",
            text_prompts=["cat", "dog", "bird", "car", "house"],
            out="cls",
        ),
        Fuse(classification="cls", out="final"),
    ],
    providers={"classifier": classifier},
)

print(f"Top prediction: {result.final.top1.label_name} ({result.final.top1.score:.2f})")
```

### Recipe 15: Monocular Depth Estimation

Estimate per-pixel depth from a single image.

```python
from mata.nodes import EstimateDepth, Fuse

depth_model = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

result = mata.infer(
    image="photo.jpg",
    graph=[
        EstimateDepth(using="depth", out="depth"),
        Fuse(depth="depth", out="final"),
    ],
    providers={"depth": depth_model},
)

depth_map = result.final.depth  # np.ndarray (H, W)
print(f"Depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")
```

---

## Multi-Task Parallel Pipelines

### Recipe 16: Parallel Detection + Classification + Depth

Run independent tasks simultaneously for 1.5–3× speedup.

```python
from mata.core.graph import Graph, ParallelScheduler
from mata.nodes import Detect, Classify, EstimateDepth, Filter, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")
classifier = mata.load("classify", "openai/clip-vit-base-patch32")
depth_model = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

graph = (Graph("full_scene")
    .parallel([
        Detect(using="detector", out="dets"),
        Classify(using="classifier", text_prompts=["indoor", "outdoor"], out="cls"),
        EstimateDepth(using="depth", out="depth"),
    ])
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(Fuse(dets="filtered", classification="cls", depth="depth", out="scene"))
)

result = mata.infer(
    image="room.jpg",
    graph=graph,
    providers={
        "detector": detector,
        "classifier": classifier,
        "depth": depth_model,
    },
    scheduler=ParallelScheduler(),
)

print(f"Scene: {result.scene.classification.top1.label_name}")
print(f"Objects: {len(result.scene.detections.instances)}")
```

### Recipe 17: Full Scene Analysis (Preset)

Use the pre-built preset for comprehensive scene understanding.

```python
from mata.presets import full_scene_analysis
from mata.core.graph import ParallelScheduler

graph = full_scene_analysis(
    detection_threshold=0.3,
    classification_labels=["indoor", "outdoor", "urban", "nature"],
)

result = mata.infer(
    image="landscape.jpg",
    graph=graph,
    providers={
        "detector": detector,
        "classifier": classifier,
        "depth": depth_model,
    },
    scheduler=ParallelScheduler(),
)
```

### Recipe 18: Detection + Segmentation + Depth (Three-Task Pipeline)

```python
graph = (Graph("three_task")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .parallel([
        PromptBoxes(using="segmenter", dets_src="filtered", out="masks"),
        EstimateDepth(using="depth", out="depth"),
    ])
    .then(Fuse(dets="filtered", masks="masks", depth="depth", out="complete"))
)
```

---

## Conditional Execution

### Recipe 19: Segment Only If Objects Detected

Skip expensive segmentation when detection returns empty.

```python
from mata.core.graph import Graph, If, CountAbove, Pass
from mata.nodes import Detect, Filter, PromptBoxes, Fuse

graph = (Graph("conditional_segment")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(If(
        predicate=CountAbove("filtered", 0),
        then_branch=PromptBoxes(using="segmenter", dets_src="filtered", out="masks"),
        else_branch=Pass(),
    ))
    .then(Fuse(dets="filtered", out="final"))
)
```

### Recipe 20: Different Processing for High vs Low Confidence

Route execution based on detection quality.

```python
from mata.core.graph import If, ScoreAbove
from mata.nodes import TopK, Filter

graph = (Graph("quality_routing")
    .then(Detect(using="detector", out="dets"))
    .then(If(
        predicate=ScoreAbove("dets", 0.8),
        then_branch=TopK(k=5, src="dets", out="final_dets"),
        else_branch=Filter(src="dets", score_gt=0.3, out="final_dets"),
    ))
    .then(Fuse(dets="final_dets", out="final"))
)
```

### Recipe 21: Label-Conditional Segmentation

Only segment when a specific object class is detected.

```python
from mata.core.graph import If, HasLabel

graph = (Graph("cat_segmenter")
    .then(Detect(using="detector", out="dets"))
    .then(If(
        predicate=HasLabel("dets", "cat"),
        then_branch=PromptBoxes(using="segmenter", dets_src="dets", out="masks"),
        else_branch=Pass(),
    ))
    .then(Fuse(dets="dets", out="final"))
)
```

### Recipe 22: Custom Predicate Function

Write your own condition as a simple callable.

```python
def has_large_objects(ctx):
    """Check if any detection covers >20% of image area."""
    dets = ctx.retrieve("dets")
    image = ctx.retrieve("input.image")
    img_area = image.width * image.height

    for inst in dets.instances:
        if inst.bbox is not None:
            x1, y1, x2, y2 = inst.bbox
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / img_area > 0.2:
                return True
    return False

graph = (Graph("large_object_handler")
    .then(Detect(using="detector", out="dets"))
    .then(If(
        predicate=has_large_objects,
        then_branch=PromptBoxes(using="segmenter", dets_src="dets", out="masks"),
        else_branch=Pass(),
    ))
    .then(Fuse(dets="dets", out="final"))
)
```

---

## Video & Tracking

### Recipe 23: Detection + Tracking Across Video

Track objects across video frames using BYTETrack.

```python
from mata.presets import detect_and_track
from mata.core.graph.temporal import VideoProcessor, FramePolicyEveryN

detector = mata.load("detect", "facebook/detr-resnet-50")
# BYTETrack or IoU-based tracker (built-in)
tracker = ...  # Your tracker instance

graph = detect_and_track(detection_threshold=0.5, track_buffer=30)
compiled = graph.compile(providers={
    "detect": {"detector": detector},
    "track": {"tracker": tracker},
})

processor = VideoProcessor(
    graph=compiled,
    providers={"detect": {"detector": detector}, "track": {"tracker": tracker}},
    frame_policy=FramePolicyEveryN(n=3),  # Process every 3rd frame
)

results = processor.process_video("input.mp4", output_path="tracked.mp4")

for frame_result in results:
    tracks = frame_result.tracks
    print(f"Frame: {len(tracks.get_active_tracks())} active tracks")
```

### Recipe 24: Real-Time Stream Processing

Process live camera or RTSP feed.

```python
from mata.core.graph.temporal import VideoProcessor, FramePolicyLatest

def handle_result(result):
    """Callback for each processed frame."""
    dets = result.dets
    print(f"Detected {len(dets.instances)} objects")

processor = VideoProcessor(
    graph=compiled,
    providers=providers,
    frame_policy=FramePolicyLatest(),  # Drop stale frames
)

# RTSP camera stream
processor.process_stream("rtsp://192.168.1.100/stream", callback=handle_result)

# Or local webcam
processor.process_stream("0", callback=handle_result)
```

### Recipe 25: Frame Skipping for Performance

Process only every N-th frame to hit target FPS.

```python
from mata.core.graph.temporal import FramePolicyEveryN

# Process every 5th frame → 6 FPS from 30 FPS video
policy = FramePolicyEveryN(n=5)

processor = VideoProcessor(
    graph=compiled,
    providers=providers,
    frame_policy=policy,
)

results = processor.process_video("input.mp4")
print(f"Processed {len(results)} frames")
```

---

## VLM Workflows

### Recipe 26: Image Description

Generate natural language descriptions using a VLM.

```python
from mata.nodes import VLMDescribe, Fuse

vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

result = mata.infer(
    image="photo.jpg",
    graph=[
        VLMDescribe(using="vlm", prompt="What do you see in this image?", out="desc"),
        Fuse(description="desc", out="final"),
    ],
    providers={"vlm": vlm},
)
```

### Recipe 27: VLM Zero-Shot Detection

Detect objects using VLM with auto-promotion.

```python
from mata.nodes import VLMDetect, Filter, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        VLMDetect(
            using="vlm",
            prompt="Identify all objects with their locations.",
            auto_promote=True,
            out="dets",
        ),
        Filter(src="dets", score_gt=0.5, out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"vlm": vlm},
)
```

### Recipe 28: VLM → GroundingDINO Grounded Detection

Use VLM for semantic understanding, then ground with spatial detector.

```python
from mata.presets import vlm_grounded_detection

vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

graph = vlm_grounded_detection(
    vlm_prompt="What objects are in this kitchen?",
    detection_threshold=0.3,
    match_strategy="label_fuzzy",
)

result = mata.infer(
    image="kitchen.jpg",
    graph=graph,
    providers={"vlm": vlm, "detector": detector},
)
```

### Recipe 29: VLM Scene Understanding (Parallel)

Combine VLM description with detection and depth in parallel.

```python
from mata.presets import vlm_scene_understanding
from mata.core.graph import ParallelScheduler

graph = vlm_scene_understanding(
    describe_prompt="Describe this scene in detail.",
    detection_threshold=0.3,
)

result = mata.infer(
    image="scene.jpg",
    graph=graph,
    providers={
        "vlm": vlm,
        "detector": detector,
        "depth": depth_model,
    },
    scheduler=ParallelScheduler(),
)
```

### Recipe 30: Multi-Image Comparison

Compare multiple images using VLM reasoning.

```python
from mata.presets import vlm_multi_image_comparison

graph = vlm_multi_image_comparison(
    prompt="What are the differences between these two images?",
)

# Note: multi-image support via VLMQuery node
result = mata.infer(
    image="before.jpg",  # Primary image
    graph=graph,
    providers={"vlm": vlm},
)
```

### Recipe 31: Entity → Instance Promotion (Manual)

Manually promote VLM entities to spatial instances.

```python
from mata.nodes import VLMDetect, Detect, Filter, PromoteEntities, Fuse

graph = (Graph("entity_promotion")
    .parallel([
        VLMDetect(using="vlm", auto_promote=False, out="vlm_dets"),
        Detect(using="detector", out="spatial_dets"),
    ])
    .then(Filter(src="spatial_dets", score_gt=0.3, out="filtered_spatial"))
    .then(PromoteEntities(
        entities_src="vlm_dets",
        spatial_src="filtered_spatial",
        match_strategy="label_fuzzy",
        out="promoted",
    ))
    .then(Fuse(dets="promoted", out="final"))
)
```

---

## Custom Nodes & Providers

### Recipe 32: Custom Node (Blur Detection)

Create a node that computes image blur score.

```python
from mata.core.graph import Node
from mata.core.artifacts import Image, Detections

class BlurDetection(Node):
    """Detect if image is blurry."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, threshold: float = 100.0, out: str = "blur_score"):
        super().__init__(name="BlurDetection")
        self.threshold = threshold
        self.output_name = out

    def run(self, ctx, image: Image) -> dict:
        import cv2
        gray = cv2.cvtColor(image.to_numpy(), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        is_blurry = laplacian_var < self.threshold
        ctx.record_metric(self.name, "laplacian_variance", laplacian_var)

        # Store as a Detections artifact with metadata
        from mata.core.artifacts.detections import Detections
        result = Detections(
            meta={"blur_score": laplacian_var, "is_blurry": is_blurry}
        )
        return {self.output_name: result}

# Use in graph
graph = (Graph("blur_check")
    .then(BlurDetection(threshold=100.0, out="blur"))
    .then(Fuse(blur="blur", out="final"))
)
```

### Recipe 33: Custom Provider (Wrapping a PyTorch Model)

Wrap any PyTorch model as a MATA provider.

```python
from mata.core.registry.protocols import Detector
from mata.core.artifacts import Image, Detections
from mata.core.types import Instance

class MyPyTorchDetector:
    """Wrap a custom PyTorch detection model."""

    def __init__(self, model, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, image: Image, threshold: float = 0.5, **kwargs) -> Detections:
        import torch
        tensor = image.to_tensor().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        instances = []
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score > threshold:
                instances.append(Instance(
                    bbox=tuple(box.cpu().tolist()),
                    score=float(score),
                    label_name=str(label),
                ))

        return Detections(instances=instances)

# Register and use
my_detector = MyPyTorchDetector(my_model, device="cuda")
assert isinstance(my_detector, Detector)  # Protocol check

result = mata.infer(
    image="test.jpg",
    graph=[Detect(using="my_det", out="dets"), Fuse(dets="dets", out="final")],
    providers={"my_det": my_detector},
)
```

### Recipe 34: Custom Predicate for Conditional Branching

```python
from mata.core.graph.conditionals import Predicate
from mata.core.artifacts.detections import Detections

class HasMinArea(Predicate):
    """Check if any detection exceeds minimum pixel area."""

    def __init__(self, src: str, min_area: int):
        self.src = src
        self.min_area = min_area

    def __call__(self, ctx) -> bool:
        try:
            dets = ctx.retrieve(self.src)
        except KeyError:
            return False

        for inst in dets.instances:
            if inst.bbox:
                x1, y1, x2, y2 = inst.bbox
                area = (x2 - x1) * (y2 - y1)
                if area >= self.min_area:
                    return True
        return False

# Usage
graph.then(If(
    predicate=HasMinArea("dets", min_area=10000),
    then_branch=PromptBoxes(using="sam", out="masks"),
    else_branch=Pass(),
))
```

---

## Performance Optimization

### Recipe 35: Parallel Scheduler for Independent Tasks

```python
from mata.core.graph import ParallelScheduler

# 1.5-3x speedup for graphs with parallel branches
result = mata.infer(
    image="photo.jpg",
    graph=graph,
    providers=providers,
    scheduler=ParallelScheduler(max_workers=4),
)
```

### Recipe 36: Optimized Multi-GPU Execution

```python
from mata.core.graph import OptimizedParallelScheduler

scheduler = OptimizedParallelScheduler(
    strategy="memory_aware",   # Place models on GPU with most free memory
    unload_models=True,        # Free GPU memory after each node completes
)

result = mata.infer(
    image="photo.jpg",
    graph=graph,
    providers=providers,
    scheduler=scheduler,
    device="cuda",             # Enable GPU
)
```

### Recipe 37: Early Filtering for Pipeline Efficiency

Place `Filter` early to reduce work for downstream nodes.

```python
# ❌ Slow: segment ALL detections, then filter
graph_slow = (Graph()
    .then(Detect(using="detector", out="dets"))
    .then(PromptBoxes(using="segmenter", dets_src="dets", out="masks"))  # segments all
    .then(Filter(src="dets", score_gt=0.7, out="filtered"))              # too late!
)

# ✅ Fast: filter first, segment only what we need
graph_fast = (Graph()
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.7, out="filtered"))              # filter early
    .then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))  # segment less
)
```

### Recipe 38: Video Frame Skipping

Balance quality vs speed for video processing.

```python
from mata.core.graph.temporal import VideoProcessor, FramePolicyEveryN

# At 30 FPS source video:
# n=1  → process all frames (30 FPS output, GPU-intensive)
# n=3  → process every 3rd frame (10 processed FPS)
# n=5  → process every 5th frame (6 processed FPS, fast)
# n=10 → process every 10th frame (3 processed FPS, surveillance mode)

processor = VideoProcessor(
    graph=compiled,
    providers=providers,
    frame_policy=FramePolicyEveryN(n=5),
)
```

---

## Debugging & Troubleshooting

### Recipe 39: Inspect Execution Metrics

```python
result = mata.infer("photo.jpg", graph, providers=providers)

# Per-node timing
for node_name, metrics in result.metrics.items():
    print(f"{node_name}: {metrics.get('latency_ms', 0):.1f}ms")

# Total execution time
print(f"Total: {result.metrics.get('total_time_ms', 0):.1f}ms")

# Provenance (model info, graph hash)
print(f"Models: {result.provenance.get('models', {})}")
print(f"Graph: {result.provenance.get('graph_name', 'unknown')}")
```

### Recipe 40: Visualize Results

Render detections on the image using built-in Annotate node.

```python
from mata.nodes import Detect, Filter, Annotate, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.5, out="filtered"),
        Annotate(
            using="pil",
            show_boxes=True,
            show_labels=True,
            show_scores=True,
            line_width=3,
            out="annotated",
        ),
        Fuse(dets="filtered", annotated="annotated", out="final"),
    ],
    providers={"detector": detector},
)

# Save annotated image
annotated_img = result.final.to_pil()
annotated_img.save("annotated_output.jpg")
```

### Recipe 41: Visualize Graph Structure

Generate a visual diagram of your graph.

```python
graph = (Graph("my_pipeline")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.5, out="filtered"))
    .then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))
    .then(Fuse(dets="filtered", masks="masks", out="final"))
)

compiled = graph.compile(providers=providers)
graph.visualize("pipeline.png")  # Requires networkx + pydot
```

### Recipe 42: Debug Validation Errors

```python
from mata.core.graph import GraphValidator

validator = GraphValidator()
result = validator.validate(
    nodes=graph._nodes,
    wiring=graph._wiring,
    providers=providers,
)

if not result.valid:
    print("Validation failed:")
    for error in result.errors:
        print(f"  ❌ {error}")
    for warning in result.warnings:
        print(f"  ⚠️ {warning}")
else:
    print("✅ Graph is valid")
```

### Recipe 43: Access Intermediate Artifacts

After execution, all intermediate artifacts are in the context.

```python
from mata.core.graph import SyncScheduler, ExecutionContext

ctx = ExecutionContext(providers=nested_providers, device="auto")
scheduler = SyncScheduler()
result = scheduler.execute(compiled, ctx, {"input.image": image})

# Access any intermediate artifact
raw_dets = ctx.retrieve("dets")
filtered = ctx.retrieve("filtered")
masks = ctx.retrieve("masks")
```

---

## Common Patterns Summary

| Pattern           | When to Use             | Example                                                       |
| ----------------- | ----------------------- | ------------------------------------------------------------- |
| **Sequential**    | Step-by-step processing | `Detect → Filter → Fuse`                                      |
| **Parallel**      | Independent tasks       | `Detect ∥ Classify ∥ Depth → Fuse`                            |
| **Conditional**   | Skip expensive ops      | `If(has_objects) → Segment`                                   |
| **Preset**        | Common workflows        | `grounding_dino_sam()`                                        |
| **VLM → Spatial** | Semantic grounding      | `VLMDetect → Detect → PromoteEntities`                        |
| **Early filter**  | Performance boost       | `Detect → Filter → Segment` (not `Detect → Segment → Filter`) |
| **Frame skip**    | Video performance       | `FramePolicyEveryN(n=5)`                                      |
| **NMS**           | Dense detections        | `Detect → NMS → Filter`                                       |

---

## Further Reading

- [Architecture Guide](GRAPH_SYSTEM_GUIDE.md) — System design and concepts
- [API Reference](GRAPH_API_REFERENCE.md) — Complete API documentation
- [Migration Guide](MIGRATION_GUIDE.md) — Upgrading from v1.5
