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
from mata.nodes import Detect, Filter, Track, Annotate

graph = [
    Detect(using="detr", out="dets"),
    Filter(src="dets", labels=["person", "car"], out="filtered"),
    Track(using="botsort", dets="filtered", out="tracks"),
    Annotate(using="drawer", dets="tracks", show_track_ids=True, out="annotated"),
]
result = mata.infer(graph=graph, video="video.mp4", providers={...})
```

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

| Feature       | ByteTrack                 | BotSort                                     |
| ------------- | ------------------------- | ------------------------------------------- |
| Algorithm     | Two-stage IoU association | IoU + Global Motion Compensation (GMC)      |
| Camera motion | No                        | Sparse optical flow compensation            |
| Speed         | Faster                    | Slightly slower                             |
| Accuracy      | Good                      | Better (especially for panning cameras)     |
| Default       | No                        | **Yes** (MATA default, matches Ultralytics) |
| ReID          | No                        | Yes (v1.9.2+, `reid_model=` kwarg)          |

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

For production deployment with lower latency:

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
      tracker_config:
        track_high_thresh: 0.6
        appearance_thresh: 0.25
```

```python
tracker = mata.load("track", "smart-cam")  # ReID loaded automatically
```

## Cross-Camera Re-Identification (v1.9.2+)

Use `ReIDBridge` with Valkey/Redis to share embeddings across cameras:

```python
from mata.trackers import ReIDBridge

# Camera 1 — publish embeddings
bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")
results = mata.track("rtsp://cam1/stream", model="detr",
                     reid_model="openai/clip-vit-base-patch32",
                     reid_bridge=bridge, stream=True)

# Camera 2 — query nearest identity
bridge2 = ReIDBridge("valkey://localhost:6379", camera_id="cam-2")
# Embeddings from cam-1 are queryable cross-camera with cosine similarity
```

See [Valkey Guide](VALKEY_GUIDE.md) for Valkey/Redis setup and configuration.

## CLI

```bash
# Basic tracking
mata track video.mp4 --model facebook/detr-resnet-50 --tracker botsort --save

# With ReID
mata track video.mp4 --model facebook/detr-resnet-50 --reid-model openai/clip-vit-base-patch32
```

## Examples

- [Basic Tracking](../examples/track/basic_tracking.py) — Video file tracking with save output
- [Persistent Tracking](../examples/track/persist_tracking.py) — Frame-by-frame with `persist=True`
- [Stream Tracking](../examples/track/stream_tracking.py) — Memory-efficient generator mode
- [Graph Tracking](../examples/graph/video_tracking.py) — Graph pipeline integration
