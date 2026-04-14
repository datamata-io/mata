---
title: "Tracking"
description: "Track objects across frames, manage identities, and integrate temporal state into MATA workflows."
sidebar_position: 3
---

# Object Tracking Guide

Track objects across video frames with persistent IDs using ByteTrack or BotSort. MATA's tracking system is fully vendored — no external tracking dependencies required.

## Quick Start

```python
import mata

# Track objects in a video file
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    tracker="botsort",   # or "bytetrack"
    conf=0.3,
    save=True,
    show_track_ids=True,
)

for frame_idx, result in enumerate(results):
    for inst in result.instances:
        print(f"Frame {frame_idx}: Track #{inst.track_id} "
              f"{inst.label_name} ({inst.score:.2f}) @ {inst.bbox}")
```

## Streaming Mode

For long videos and RTSP streams, use `stream=True` to process frames with constant memory:

```python
for result in mata.track("rtsp://camera/stream",
                         model="facebook/detr-resnet-50",
                         stream=True):
    active = [i for i in result.instances if i.track_id is not None]
    print(f"Active tracks: {len(active)}")
```

## Persistent Frame-by-Frame Tracking

For custom processing loops, use `persist=True` to maintain track state across frames:

```python
import cv2
import mata

tracker = mata.load("track", "facebook/detr-resnet-50", tracker="bytetrack")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = tracker.update(frame, persist=True)
    # result.instances have .track_id set
cap.release()
```

## Graph-Based Tracking Pipelines

Integrate tracking into multi-task graph workflows:

```python
from mata.core.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN, VideoProcessor
from mata.nodes import Detect, Filter, Fuse, Track
from mata.nodes.track import SimpleIOUTracker

detector = mata.load("detect", "facebook/detr-resnet-50")
tracker = SimpleIOUTracker()

graph = (
    Graph("detect_and_track")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", label_in=["person", "car"], out="filtered"))
    .then(Track(using="tracker", dets="filtered", out="tracks"))
    .then(Fuse(detections="filtered", tracks="tracks", out="frame_result"))
)

flat_providers = {"detector": detector, "tracker": tracker}
compiled = graph.compile(providers=flat_providers)

processor = VideoProcessor(
    graph=compiled,
    providers={
        "detect": {"detector": detector},
        "track": {"tracker": tracker},
    },
    frame_policy=FramePolicyEveryN(n=1),
)
results = processor.process_video(video_path="video.mp4")
```

Add `Annotate` afterward if you want a rendered output stage.

## Supported Source Types

| Source          | Example           | Notes                                  |
| --------------- | ----------------- | -------------------------------------- |
| Video file      | `"video.mp4"`     | `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv` |
| RTSP stream     | `"rtsp://..."`    | Live camera feeds                      |
| HTTP stream     | `"http://..."`    | IP cameras, web streams                |
| Webcam          | `0` (int)         | Local camera by device index           |
| Image directory | `"frames/"`       | Sorted by filename                     |
| Single image    | `"image.jpg"`     | Returns 1-frame result                 |
| numpy array     | `np.ndarray`      | Direct frame input                     |
| PIL Image       | `Image.open(...)` | Direct frame input                     |

## ByteTrack vs BotSort

| Feature       | ByteTrack                    | BotSort                                            |
| ------------- | ---------------------------- | -------------------------------------------------- |
| Algorithm     | Two-stage IoU association    | IoU + Global Motion Compensation (GMC)             |
| Camera motion | No                           | Sparse optical flow compensation                   |
| Speed         | Typically lower overhead     | Typically higher overhead (GMC + optional ReID)    |
| Accuracy      | Strong baseline IoU tracking | Often more robust under camera motion (GMC)        |
| Default       | No                           | **Yes** (MATA default, matches Ultralytics)        |
| ReID          | No                           | Yes (v1.9.2+, supply `reid_model=` to auto-enable) |

## YAML Configuration

```yaml
# .mata/models.yaml
models:
  track:
    highway-cam:
      source: "facebook/detr-resnet-50"
      tracker: botsort
      tracker_config:
        track_high_thresh: 0.6
        track_buffer: 60
      frame_rate: 30
```

```python
tracker = mata.load("track", "highway-cam")
```

## Appearance-Based ReID (v1.9.2+)

Enable appearance re-identification with BotSort to recover track IDs after occlusion or re-entry. Pass any HuggingFace image encoder (ViT, CLIP, OSNet, etc.) as a ReID model:

```python
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    tracker="botsort",
    reid_model="openai/clip-vit-base-patch32",
    conf=0.3,
    save=True,
)
```

### ONNX ReID Models

For production deployment where ONNX Runtime fits your serving stack:

```python
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    reid_model="osnet_x1_0.onnx",
)
```

### ReID via Config Alias

```yaml
# .mata/models.yaml
models:
  track:
    smart-cam:
      source: "facebook/detr-resnet-50"
      tracker: botsort
      reid_model: "openai/clip-vit-base-patch32"
      with_reid: true # Optional here; auto-enabled when reid_model is provided
      tracker_config:
        track_high_thresh: 0.6
        appearance_thresh: 0.25
```

```python
tracker = mata.load("track", "smart-cam")  # ReID loads and activates automatically
```

## Cross-Camera Re-Identification (v1.9.2+)

Use `ReIDBridge` with Valkey/Redis to share embeddings across cameras:

```python
from mata.trackers import ReIDBridge

# Camera 1 — publish embeddings
bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")
results = mata.track("rtsp://cam1/stream", model="facebook/detr-resnet-50",
                     reid_model="openai/clip-vit-base-patch32",
                     reid_bridge=bridge, stream=True)

# Camera 2 — query nearest identity
bridge2 = ReIDBridge("valkey://localhost:6379", camera_id="cam-2")
# Embeddings from cam-1 are queryable cross-camera with cosine similarity
```

See [Valkey Guide](./VALKEY_GUIDE.md) for Valkey/Redis setup and configuration.

## CLI

```bash
# Basic tracking
mata track video.mp4 --model facebook/detr-resnet-50 --tracker botsort --save

# With ReID (--reid-model auto-enables appearance matching)
mata track video.mp4 --model facebook/detr-resnet-50 --reid-model openai/clip-vit-base-patch32
```

## Examples

- [Basic Tracking](../examples/track/basic_tracking.py) — Video file tracking with save output
- [Persistent Tracking](../examples/track/persist_tracking.py) — Frame-by-frame with `persist=True`
- [ReID Tracking](../examples/track/reid_tracking.py) — Single-camera BotSort ReID with `reid_model=`
- [Cross-Camera ReID](../examples/track/cross_camera_reid.py) — Valkey-backed identity sharing across cameras
- [Stream Tracking](../examples/track/stream_tracking.py) — Memory-efficient generator mode
- [Graph Tracking](../examples/graph/video_tracking.py) — Graph pipeline integration
