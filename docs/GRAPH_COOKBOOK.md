# MATA Graph Cookbook

> **Version**: 1.9.7 | **Last Updated**: April 3, 2026

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
   - [Recipe 26: Video Processing with Per-Frame Callback](#recipe-26-video-processing-with-per-frame-callback)
   - [Recipe 27: Cross-Camera ReID Pipeline](#recipe-27-cross-camera-reid-pipeline)
   - [Recipe 28: Real-Time Annotated Video](#recipe-28-real-time-annotated-video)
   - [Recipe 29: Multi-Camera Dashboard](#recipe-29-multi-camera-dashboard)
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

for inst in result.final.dets.instances:
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

print(f"Found {len(result.final.dets.instances)} objects above 0.5 confidence")
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

for inst in result.final.dets.instances:
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
for roi, inst_id in zip(result.final.rois.roi_images, result.final.rois.instance_ids):
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

print(f"Segmented {len(result.final.dets.instances)} objects")
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

print(
    f"Top prediction: {result.final.classification.top1.label_name} "
    f"({result.final.classification.top1.score:.2f})"
)
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
print(f"Objects: {len(result.scene.dets.instances)}")
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
from mata.core.graph.temporal import FramePolicyEveryN

detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = ...  # Your tracker instance

graph = detect_and_track(detection_threshold=0.5, track_buffer=30)

results = graph.run(
    "input.mp4",
    providers={"detector": detector, "tracker": tracker},
    frame_policy=FramePolicyEveryN(n=3),  # Process every 3rd frame
    output_path="tracked.mp4",
)

for frame_result in results:
    tracks = frame_result.tracks
    print(f"Frame: {len(tracks.get_active_tracks())} active tracks")
```

> **Advanced:** For fine-grained control (runtime graph swaps, custom frame producers) use
> `VideoProcessor` directly — see [Recipe 23b](#recipe-23b-videoprocessor-direct-usage).

### Recipe 23b: VideoProcessor Direct Usage

Use `VideoProcessor` directly when you need full control over compilation and the frame loop.

```python
from mata.presets import detect_and_track
from mata.core.graph.temporal import VideoProcessor, FramePolicyEveryN

detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = ...

graph = detect_and_track(detection_threshold=0.5, track_buffer=30)
providers = {"detector": detector, "tracker": tracker}
compiled = graph.compile(providers=providers)

processor = VideoProcessor(
    graph=compiled,
    providers=providers,
    frame_policy=FramePolicyEveryN(n=3),
)
results = processor.process_video("input.mp4", output_path="tracked.mp4")
```

### Recipe 24: Real-Time Stream Processing

Process live camera or RTSP feed.

```python
from mata.core.graph.temporal import FramePolicyLatest

# Generator mode — constant memory, iterate lazily
for result in graph.run(
    "rtsp://192.168.1.100/stream",
    providers=providers,
    frame_policy=FramePolicyLatest(),  # Drop stale frames
):
    dets = result.dets
    print(f"Detected {len(dets.instances)} objects")

# Callback mode — blocking, useful for background threads
import threading

stop = threading.Event()
graph.run(
    "rtsp://192.168.1.100/stream",
    providers=providers,
    frame_policy=FramePolicyLatest(),
    callback=lambda result, frame_num: print(f"Frame {frame_num}: {result}"),
    stop_event=stop,
)

# Or local webcam (integer index)
for result in graph.run(0, providers=providers, frame_policy=FramePolicyLatest()):
    process(result)
```

### Recipe 25: Frame Skipping for Performance

Process only every N-th frame to hit target FPS.

```python
from mata.core.graph.temporal import FramePolicyEveryN

# At 30 FPS source video:
# n=1  → process all frames (30 FPS output, GPU-intensive)
# n=3  → process every 3rd frame (10 processed FPS)
# n=5  → process every 5th frame (6 processed FPS, fast)
# n=10 → process every 10th frame (3 processed FPS, surveillance mode)

results = graph.run(
    "input.mp4",
    providers=providers,
    frame_policy=FramePolicyEveryN(n=5),
)
print(f"Processed {len(results)} frames")
```

### Recipe 26: Video Processing with Per-Frame Callback

Process a video file frame-by-frame while displaying or saving each annotated frame in real time. The `callback` receives `(result, frame_num, frame_bgr)` for every processed frame; results are still returned as a list for post-processing.

```python
import threading
import cv2
import mata
from mata.core.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN
from mata.nodes import Detect, Filter, Track, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = mata.load("track", "facebook/detr-resnet-50", tracker="bytetrack")

graph = (
    Graph("video_callback")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(Track(using="tracker", dets="filtered", out="tracks"))
    .then(Fuse(tracks="tracks", out="final"))
)

stop = threading.Event()

def on_frame(result, frame_num, frame_bgr):
    tracks = result.channels.get("tracks")
    n_active = len(tracks.get_active_tracks()) if tracks else 0
    print(f"\rFrame {frame_num:5d} | active tracks: {n_active}", end="", flush=True)

    # Optionally display with OpenCV
    cv2.imshow("Preview", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop.set()

results = graph.run(
    "input.mp4",
    providers={"detector": detector, "tracker": tracker},
    frame_policy=FramePolicyEveryN(n=2),
    max_frames=300,
    callback=on_frame,
    stop_event=stop,
)
print(f"\nProcessed {len(results)} frames in total")
cv2.destroyAllWindows()
```

> **Tip:** `callback` works for video files (returns `list`) and streams (returns `None`, blocking). For streams the callback signature is `(result, frame_num)` — the raw BGR frame is only provided for video files.

---

### Recipe 27: Cross-Camera ReID Pipeline

Detect → Track → extract ROIs → embed → publish/query Valkey for cross-camera identity matching. Requires a running Valkey (or Redis) instance and the `mata.trackers.ReIDBridge`.

```python
import mata
from mata.core.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN
from mata.nodes import Detect, Filter, Track, ExtractROIs, Embed, ReID, Fuse
from mata.trackers import ReIDBridge

# Load models
detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = mata.load("track", "facebook/detr-resnet-50", tracker="botsort")
encoder = mata.load("embed", "openai/clip-vit-base-patch32")

# Cross-camera bridge (unique camera_id per process)
bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")

graph = (
    Graph("cross_camera_reid")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(Track(using="tracker", dets="filtered", out="tracks"))
    .then(ExtractROIs(src_dets="tracks", src_image="image", out="rois"))
    .then(Embed(using="encoder", src="rois", out="embeddings"))
    .then(ReID(using="bridge", tracks_src="tracks",
               embeddings_src="embeddings", out="cross_matches"))
    .then(Fuse(tracks="tracks", cross_matches="cross_matches", out="final"))
)

results = graph.run(
    "rtsp://192.168.1.100/stream",
    providers={"detector": detector, "tracker": tracker,
               "encoder": encoder, "bridge": bridge},
    frame_policy=FramePolicyEveryN(n=3),
)

for result in results:
    cm = result.channels.get("cross_matches")
    if cm and len(cm.matches) > 0:
        for match in cm.matches:
            print(f"Local track #{match.local_track_id} "
                  f"→ {match.remote_camera_id}#{match.remote_track_id} "
                  f"(similarity={match.similarity:.2f})")
```

> **Multi-camera:** Run a second process with the same `valkey://` URL and `camera_id="cam-2"`. Both processes share Valkey; `ReIDBridge` handles TTL eviction and per-camera exclusion automatically.

---

### Recipe 28: Real-Time Annotated Video

Full pipeline with `AnnotateRT` — draws bounding boxes, track IDs, trajectory trails, and cross-camera highlights directly onto each frame using OpenCV.

```python
import mata
import cv2
from mata.core.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN
from mata.nodes import Detect, Filter, Track, ExtractROIs, Embed, ReID, AnnotateRT, Fuse
from mata.trackers import ReIDBridge

detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = mata.load("track", "facebook/detr-resnet-50", tracker="botsort")
encoder = mata.load("embed", "openai/clip-vit-base-patch32")
bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")

annotator = AnnotateRT(
    show_track_ids=True,
    show_trails=True,
    trail_length=40,
    camera_label="CAM-1",
    camera_color=(255, 100, 60),  # BGR orange
    out="annotated",
    detections_src="tracks",
    tracks_src="tracks",
    cross_matches_src="cross_matches",
)

graph = (
    Graph("annotated_video")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(Track(using="tracker", dets="filtered", out="tracks"))
    .then(ExtractROIs(src_dets="tracks", src_image="image", out="rois"))
    .then(Embed(using="encoder", src="rois", out="embeddings"))
    .then(ReID(using="bridge", tracks_src="tracks",
               embeddings_src="embeddings", out="cross_matches"))
    .then(annotator)
)

# Set up video writer
cap_probe = cv2.VideoCapture("input.mp4")
fps = cap_probe.get(cv2.CAP_PROP_FPS)
cap_probe.release()
writer = cv2.VideoWriter("output_annotated.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / 2,  # matches FramePolicyEveryN(n=2)
                         (1280, 720))

def on_frame(result, frame_num, frame_bgr):
    annotated = result.channels.get("annotated")
    frame_out = annotated.to_numpy() if annotated is not None else frame_bgr
    writer.write(cv2.resize(frame_out, (1280, 720)))

graph.run(
    "input.mp4",
    providers={"detector": detector, "tracker": tracker,
               "encoder": encoder, "bridge": bridge},
    frame_policy=FramePolicyEveryN(n=2),
    callback=on_frame,
)
writer.release()
print("Saved output_annotated.mp4")
```

> **Stateful trails:** `AnnotateRT` maintains per-track centre-point history across frames. Call `annotator.reset()` between clips to clear trail state when switching videos.

---

### Recipe 29: Multi-Camera Dashboard

Run two independent `graph.run()` instances in separate threads sharing a single Valkey backend. Each camera contributes to the cross-camera identity pool.

```python
import threading
import mata
from mata.core.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN
from mata.nodes import Detect, Filter, Track, ExtractROIs, Embed, ReID, AnnotateRT
from mata.trackers import ReIDBridge

def make_graph(cam_id: str) -> tuple[Graph, AnnotateRT]:
    annotator = AnnotateRT(
        show_track_ids=True,
        show_trails=True,
        camera_label=cam_id.upper(),
        out="annotated",
        detections_src="tracks",
        tracks_src="tracks",
        cross_matches_src="cross_matches",
    )
    g = (
        Graph(f"pipeline_{cam_id}")
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=0.4, out="filtered"))
        .then(Track(using="tracker", dets="filtered", out="tracks"))
        .then(ExtractROIs(src_dets="tracks", src_image="image", out="rois"))
        .then(Embed(using="encoder", src="rois", out="embeddings"))
        .then(ReID(using="bridge", tracks_src="tracks",
                   embeddings_src="embeddings", out="cross_matches"))
        .then(annotator)
    )
    return g, annotator

# Shared models (thread-safe read-only inference)
detector = mata.load("detect", "facebook/detr-resnet-50")
tracker1 = mata.load("track", "facebook/detr-resnet-50", tracker="botsort")
tracker2 = mata.load("track", "facebook/detr-resnet-50", tracker="botsort")
encoder = mata.load("embed", "openai/clip-vit-base-patch32")

VALKEY_URL = "valkey://localhost:6379"
bridge1 = ReIDBridge(VALKEY_URL, camera_id="cam-1")
bridge2 = ReIDBridge(VALKEY_URL, camera_id="cam-2")

graph1, annotator1 = make_graph("cam-1")
graph2, annotator2 = make_graph("cam-2")

stop = threading.Event()

def run_camera(graph, providers, source, stop_event):
    graph.run(
        source,
        providers=providers,
        frame_policy=FramePolicyEveryN(n=3),
        stop_event=stop_event,
    )

t1 = threading.Thread(target=run_camera, args=(
    graph1,
    {"detector": detector, "tracker": tracker1,
     "encoder": encoder, "bridge": bridge1},
    "rtsp://192.168.1.101/stream",
    stop,
))
t2 = threading.Thread(target=run_camera, args=(
    graph2,
    {"detector": detector, "tracker": tracker2,
     "encoder": encoder, "bridge": bridge2},
    "rtsp://192.168.1.102/stream",
    stop,
))

t1.start()
t2.start()

try:
    t1.join()
    t2.join()
except KeyboardInterrupt:
    stop.set()  # Signal both cameras to stop
    t1.join()
    t2.join()
    print("Dashboard stopped.")
```

> **Scaling:** Each `ReIDBridge` instance needs a unique `camera_id`. The Valkey backend handles concurrent writes safely. For production deployments, use a Valkey cluster and set `ttl_seconds` on the bridge to control how long embeddings persist.

---

## VLM Workflows

### Recipe 30: Image Description

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

### Recipe 31: VLM Zero-Shot Detection

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

### Recipe 32: VLM → GroundingDINO Grounded Detection

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

### Recipe 33: VLM Scene Understanding (Parallel)

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

### Recipe 34: Multi-Image Comparison

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

### Recipe 35: Entity → Instance Promotion (Manual)

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

### Recipe 36: Custom Node (Blur Detection)

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

### Recipe 37: Custom Provider (Wrapping a PyTorch Model)

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

### Recipe 38: Custom Predicate for Conditional Branching

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

### Recipe 39: Parallel Scheduler for Independent Tasks

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

### Recipe 40: Optimized Multi-GPU Execution

```python
from mata.core.graph import OptimizedParallelScheduler

scheduler = OptimizedParallelScheduler(
    device_placement="memory_aware",  # Place models on GPU with most free memory
    unload_unused=True,                # Free GPU memory after each node completes
)

result = mata.infer(
    image="photo.jpg",
    graph=graph,
    providers=providers,
    scheduler=scheduler,
    device="cuda",             # Enable GPU
)
```

### Recipe 41: Early Filtering for Pipeline Efficiency

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

### Recipe 42: Video Frame Skipping

Balance quality vs speed for video processing.

```python
from mata.core.graph.temporal import FramePolicyEveryN

# At 30 FPS source video:
# n=1  → process all frames (30 FPS output, GPU-intensive)
# n=3  → process every 3rd frame (10 processed FPS)
# n=5  → process every 5th frame (6 processed FPS, fast)
# n=10 → process every 10th frame (3 processed FPS, surveillance mode)

results = graph.run(
    "input.mp4",
    providers=providers,
    frame_policy=FramePolicyEveryN(n=5),
)
```

---

## Debugging & Troubleshooting

### Recipe 43: Inspect Execution Metrics

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

### Recipe 44: Visualize Results

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
annotated_img = result.final.annotated.to_pil()
annotated_img.save("annotated_output.jpg")
```

### Recipe 45: Visualize Graph Structure

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

### Recipe 46: Debug Validation Errors

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

### Recipe 47: Access Intermediate Artifacts

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
| **EarlyExit**     | Quality gate            | `Detect → EarlyExit(no_objects) → Segment`                    |
| **While**         | Iterative refinement    | `While([Detect], condition=low_confidence)`                   |
| **Recognition**   | Identity matching       | `Detect → ExtractROIs → Embed → GalleryMatchNode`             |

---

## Further Reading

- [Architecture Guide](GRAPH_SYSTEM_GUIDE.md) — System design and concepts
- [API Reference](GRAPH_API_REFERENCE.md) — Complete API documentation
- [Migration Guide](MIGRATION_GUIDE.md) — Upgrading from v1.5

---

## Recipe: Quality Gate with `EarlyExit` (v1.9.5)

Stop expensive downstream inference the moment a fast guard determines there is nothing to process — zero wasted compute.

**Scenario:** Run object detection, then skip segmentation and OCR if no objects were found.

```python
import mata
from mata.nodes import Detect, EarlyExit, PromptBoxes
from mata.core.graph import Graph

detector  = mata.load("detect",  "facebook/detr-resnet-50")
segmenter = mata.load("segment", "facebook/sam3")

def no_objects(ctx):
    """True when detection found nothing."""
    return len(ctx.retrieve("dets").instances) == 0

graph = (
    Graph("quality_gate")
    .then(Detect(using="detector", out="dets"))
    .then(EarlyExit(
        predicate=no_objects,
        reason="No detections — skipping segmentation",
        name="gate",
    ))
    .then(PromptBoxes(using="segmenter", dets_src="dets", out="masks"))
)

try:
    result = mata.infer(
        "frame.jpg",
        graph=graph,
        providers={"detector": detector, "segmenter": segmenter},
    )
    # result.masks is available only when objects were found
except Exception:
    pass  # EarlyExitException is caught inside mata.infer — never raised to caller
```

**Key points:**

- `EarlyExitException` is caught by the scheduler; `mata.infer()` returns a partial `MultiResult` rather than raising.
- Nodes **before** the gate always execute; nodes **after** are skipped.
- Combine with `condition=` guards for fine-grained per-node control.

---

## Recipe: Iterative Detection with `While` (v1.9.5)

Re-run a detection node inside a bounded do-while loop until the result meets a quality threshold — useful when the underlying model supports iterative refinement (e.g., updating the ROI or confidence threshold each pass).

**Scenario:** Re-detect if the highest-confidence detection is below 0.8, up to 4 attempts.

```python
import mata
from mata.nodes import Detect, While
from mata.core.graph import Graph

detector = mata.load("detect", "facebook/detr-resnet-50")

def low_confidence(ctx):
    instances = ctx.retrieve("dets").instances
    if not instances:
        return False   # Nothing found; stop looping
    return max(i.score for i in instances) < 0.8

graph = (
    Graph("iterative_detect")
    .then(While(
        body=[Detect(using="detector", out="dets")],
        condition=low_confidence,
        max_iterations=4,
        name="refine_loop",
    ))
)

result = mata.infer(
    "image.jpg",
    graph=graph,
    providers={"detector": detector},
)
print(result.dets)          # Final detection result after loop
```

**Key points:**

- `body` is a `list[Node]` — chain multiple nodes inside the loop by listing them all.
- The body always executes **at least once** (do-while semantics).
- `max_iterations` cannot be disabled. If the cap is reached the loop exits cleanly.
- Per-iteration latency and the `max_iterations_reached` flag are recorded in graph metrics.

---

## Recipe: Gallery Recognition Pipeline (v1.9.5)

Identify known individuals in a scene by comparing their embeddings against a pre-built gallery.

**Scenario:** Detect faces/persons, extract per-ROI embeddings, and match against a known-identity gallery.

```python
import mata
from mata import Gallery
from mata.nodes import Detect, GalleryMatchNode
from mata.nodes.embed import Embed
from mata.nodes.extract_rois import ExtractROIs
from mata.core.graph import Graph

# ── Step 1: Build and save gallery (run once) ──────────────────────────
encoder = mata.load("embed", "openai/clip-vit-base-patch32")

gallery = Gallery(threshold=0.6)
for name, image_path in [("alice", "alice.jpg"), ("bob", "bob.jpg")]:
    emb_result = mata.run("embed", image_path, model="openai/clip-vit-base-patch32")
    gallery.add(name, emb_result.embeddings[0])

gallery.save("people_gallery.npz")

# ── Step 2: Build recognition graph ────────────────────────────────────
detector = mata.load("detect", "facebook/detr-resnet-50")
gallery  = Gallery.load("people_gallery.npz")

graph = (
    Graph("recognition")
    .then(Detect(using="detector", out="dets"))
    .then(ExtractROIs(src_dets="dets", out="rois"))
    .then(Embed(using="encoder", src="rois", out="embeddings"))
    .then(GalleryMatchNode(
        gallery=gallery,
        top_k=1,
        threshold=0.6,
        out="matches",
    ))
)

# ── Step 3: Run inference ───────────────────────────────────────────────
result = mata.infer(
    "group_photo.jpg",
    graph=graph,
    providers={"detector": detector, "encoder": encoder},
)

for match_entry in result.matches:
    print(f"Instance {match_entry.instance_id}: {match_entry.label}"
          f" ({match_entry.similarity:.2f})")
```

**Key points:**

- `GalleryMatchNode` does **not** need a `providers` entry — the gallery is injected at construction.
- `top_k=1` returns the single best identity per embedding; increase for top-N ranked results.
- `threshold` filters out low-confidence matches; unmatched instances have `match_entry.label == "unknown"`.
- Use `mata.run("recognize", image, gallery=gallery, model="openai/clip-vit-base-patch32")` for the one-liner equivalent.
- See [Recognition API Reference](GRAPH_API_REFERENCE.md#recognition-nodes-v195) for full `GalleryMatchNode` docs.

---

## Video Semantic Search (v1.9.7)

### Recipe 40: Index a Video and Search by Natural Language

Build a searchable index from a video file, then find moments matching free-text descriptions.

```python
import mata
from mata.nodes import IndexVideo, EmbeddingSearch
from mata.core.graph import Graph

# Load an embed-capable model
embedder = mata.load("embed", "openai/clip-vit-base-patch32")

graph = (
    Graph("video_search")
    .then(IndexVideo(using="embedder", sample_fps=1.0))
    .then(EmbeddingSearch(using="embedder", text="red car", top_k=5))
)

result = graph.run(video="traffic.mp4", providers={"embedder": embedder})

for qr in result["search_results"]:
    for rank, m in enumerate(qr.matches, 1):
        print(f"#{rank}  sim={m.similarity:.4f}  @ {m.start_s:.1f}s–{m.end_s:.1f}s")
```

**Key points:**

- `IndexVideo` samples the video at `sample_fps` frames per second, embeds each frame, and stores the gallery as a `VideoIndexData` artifact.
- `EmbeddingSearch` embeds the query text(s) and runs cosine nearest-neighbour search against the gallery.
- The same `"embedder"` provider is shared by both nodes.

---

### Recipe 41: Multi-Query Video Search with Threshold

Filter low-confidence matches by supplying `threshold`:

```python
embedder = mata.load("embed", "openai/clip-vit-base-patch32")

graph = (
    Graph("multi_query_search")
    .then(IndexVideo(using="embedder", sample_fps=2.0))
    .then(EmbeddingSearch(
        using="embedder",
        text=["pedestrian crossing", "red bus", "cyclist"],
        top_k=3,
        threshold=0.20,
    ))
)

result = graph.run(video="dashcam.mp4", providers={"embedder": embedder})

for qr in result["search_results"].results:
    print(f'\nQuery: "{qr.query}"')
    for rank, m in enumerate(qr.matches, 1):
        mm, ss = int(m.start_s) // 60, int(m.start_s) % 60
        print(f"  #{rank}  sim={m.similarity:.4f}  @ {mm:02d}m{ss:02d}s")
```

---

### Recipe 42: High-Quality Search with Qwen3-VL-Embedding

Qwen3-VL-Embedding is a multimodal encoder that understands visual content at a deeper semantic level:

```python
embedder = mata.load("embed", "Qwen/Qwen3-VL-Embedding-2B", dtype="bfloat16")

graph = (
    Graph("semantic_search")
    .then(IndexVideo(using="embedder", mode="frame", sample_fps=1.0))
    .then(EmbeddingSearch(
        using="embedder",
        text=["vehicle collision", "jaywalking pedestrian"],
        top_k=5,
        threshold=0.18,
    ))
)

result = graph.run(video="cctv_footage.mp4", providers={"embedder": embedder})
```

---

### Recipe 43: Reuse a Saved Video Index

Index once, search many times without re-encoding:

```python
from mata.recognition import VideoIndex, index_video

# Step 1 — build and persist the index
embedder = mata.load("embed", "openai/clip-vit-base-patch32")
vidx = index_video("long_video.mp4", embedder, sample_fps=1.0)
vidx.save("long_video.npz")

# Step 2 — load and search later (no re-encoding)
vidx = VideoIndex.load("long_video.npz")
matches = vidx.search("fire and smoke", top_k=5)
for m in matches:
    print(f"sim={m.similarity:.4f}  @ {m.start_s:.1f}s")
```

---

### Recipe 44: Combined Detect + Video Search Pipeline

Run object detection on frames while simultaneously building a searchable text index:

```python
from mata.nodes import Detect, IndexVideo, EmbeddingSearch
from mata.core.graph import Graph

detector = mata.load("detect", "facebook/detr-resnet-50")
embedder = mata.load("embed", "openai/clip-vit-base-patch32")

graph = Graph("detect_and_search")
graph.add(Detect(using="detector"), inputs={"image": "input.image"})
graph.add(IndexVideo(using="embedder", sample_fps=1.0), inputs={"video": "input.video"})
graph.add(
    EmbeddingSearch(using="embedder", text="person running", top_k=5),
    inputs={"video_index": "video_index"},
)

result = graph.run(
    video="scene.mp4",
    image="thumbnail.jpg",
    providers={"detector": detector, "embedder": embedder},
)

print("Detections:", result["detections"])
print("Search matches:", result["search_results"][0].matches)
```

**Key points:**

- `IndexVideo` and `EmbeddingSearch` consume `VideoPath` / `VideoIndexData` artifacts; `Detect` consumes image artifacts — they can share a graph without conflict.
- Use `mata.infer(video=..., image=..., ...)` when supplying both video and image inputs.
- See [Video Search Nodes API Reference](GRAPH_API_REFERENCE.md#video-search-nodes-v197) for full parameter tables.
