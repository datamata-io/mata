---
title: "CLI"
description: "Use the mata CLI for one-shot inference, recognition, tracking, validation, and export workflows."
sidebar_position: 1
---

# MATA CLI Reference

> **Version**: 1.9.5 | **Last Updated**: April 3, 2026

MATA ships a `mata` command installed alongside the Python package. It exposes the five most common workflows directly in a terminal without writing any Python.

```
mata <command> [options]
```

---

## Table of Contents

1. [Global Options](#global-options)
2. [mata run](#mata-run)
3. [mata recognize](#mata-recognize)
4. [mata track](#mata-track)
5. [mata val](#mata-val)
6. [mata export](#mata-export)
7. [Output Formats](#output-formats)
8. [Examples](#examples)

---

## Global Options

| Flag             | Description                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `--version`      | Print the installed MATA version and exit                                                       |
| `-v / --verbose` | Suppress third-party library noise (`-v`); show everything including third-party output (`-vv`) |

---

## mata run

Run one-shot inference on a single image. Wraps `mata.run()`.

```
mata run <task> <input> [options]
```

### Positional Arguments

| Argument | Description                                                                           |
| -------- | ------------------------------------------------------------------------------------- |
| `task`   | Task type: `detect`, `segment`, `classify`, `depth`, `embed`, `ocr`, `vlm`, `barcode` |
| `input`  | Path to the input image (local file or URL)                                           |

### Options

| Flag           | Default          | Description                                                                |
| -------------- | ---------------- | -------------------------------------------------------------------------- |
| `--model / -m` | registry default | Model ID (HuggingFace), local path, or config alias                        |
| `--conf`       | `None`           | Confidence threshold for detection / segmentation                          |
| `--device`     | `None`           | Compute device: `cpu`, `cuda`, `cuda:0`, `mps`                             |
| `--text`       | `None`           | Comma-separated text prompts for zero-shot tasks (classify/detect/segment) |
| `--prompt`     | `None`           | Free-text prompt for VLM tasks                                             |
| `--save`       | `False`          | Save the annotated result image to `--save-dir`                            |
| `--save-dir`   | `runs/`          | Directory to write saved results                                           |
| `--json`       | `False`          | Print the raw result as JSON to stdout                                     |

### Examples

```bash
# Object detection
mata run detect image.jpg --model facebook/detr-resnet-50 --save

# Zero-shot classification with CLIP
mata run classify image.jpg \
    --model openai/clip-vit-base-patch32 \
    --text "cat,dog,bird"

# Depth estimation
mata run depth image.jpg --model depth-anything/Depth-Anything-V2-Small-hf --save

# Feature embedding (outputs JSON vector)
mata run embed image.jpg --model openai/clip-vit-base-patch32 --json

# OCR
mata run ocr document.jpg --save

# VLM visual question answering
mata run vlm image.jpg \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --prompt "What objects are visible in this image?"

# Barcode / QR decoding
mata run barcode package.jpg --model pyzbar --json

# Save annotated segment output
mata run segment image.jpg --model facebook/sam3 --save --save-dir ./output/
```

---

## mata recognize

Gallery-based recognition. Embeds the input image and performs cosine similarity search against a pre-built `.npz` gallery file.

```
mata recognize <input> --gallery <file> [options]
```

### Positional Arguments

| Argument | Description      |
| -------- | ---------------- |
| `input`  | Input image path |

### Options

| Flag             | Default    | Description                                                  |
| ---------------- | ---------- | ------------------------------------------------------------ |
| `--gallery / -g` | _required_ | Path to `.npz` gallery file (created with `Gallery.save()`)  |
| `--model / -m`   | `None`     | Embed model ID, local path, or config alias                  |
| `--top-k`        | `1`        | Number of top matches to return                              |
| `--threshold`    | `None`     | Minimum cosine similarity; uses gallery default when omitted |
| `--device`       | `None`     | Compute device: `cpu`, `cuda`, `cuda:0`, `mps`               |
| `--json`         | `False`    | Print raw JSON result to stdout                              |

### Building a Gallery

```python
import mata

gallery = mata.Gallery()
gallery.add("alice", "alice_1.jpg", model="openai/clip-vit-base-patch32")
gallery.add("alice", "alice_2.jpg", model="openai/clip-vit-base-patch32")
gallery.add("bob",   "bob_1.jpg",   model="openai/clip-vit-base-patch32")
gallery.save("people.npz")
```

### Examples

```bash
# Identify a person in a photo
mata recognize query.jpg --gallery people.npz

# Return top-3 matches with similarity scores
mata recognize query.jpg --gallery people.npz --top-k 3

# Use CLIP model explicitly, filter by threshold
mata recognize query.jpg \
    --gallery products.npz \
    --model openai/clip-vit-base-patch32 \
    --threshold 0.6 \
    --json
```

### Output (console)

```
Best match:  alice  (similarity=0.8234)
```

---

## mata track

Run multi-object tracking on a video file, RTSP stream, or webcam. Wraps `mata.track()`.

```
mata track <source> [options]
```

### Positional Arguments

| Argument | Description                                              |
| -------- | -------------------------------------------------------- |
| `source` | Video file path, RTSP URL, or webcam index (`0`, `1`, …) |

### Options

| Flag           | Default   | Description                                                       |
| -------------- | --------- | ----------------------------------------------------------------- |
| `--model / -m` | `None`    | Detection model ID, local path, or config alias                   |
| `--tracker`    | `botsort` | Tracker algorithm: `botsort` or `bytetrack`                       |
| `--conf`       | `0.25`    | Confidence threshold                                              |
| `--iou`        | `0.7`     | IoU threshold for NMS                                             |
| `--device`     | `None`    | Compute device: `cpu`, `cuda`, `cuda:0`, `mps`                    |
| `--save`       | `False`   | Save annotated output video                                       |
| `--show`       | `False`   | Display results in a live window                                  |
| `--save-dir`   | `runs/`   | Directory to write output video                                   |
| `--reid-model` | `None`    | ReID embedding model for appearance-based identity re-association |
| `--json`       | `False`   | Print per-frame JSON detection data to stdout                     |

### Examples

```bash
# Basic tracking with BotSort
mata track video.mp4 --model facebook/detr-resnet-50 --save

# Live webcam with ByteTrack
mata track 0 --model facebook/detr-resnet-50 --tracker bytetrack --show

# RTSP stream with ReID
mata track rtsp://192.168.1.10/stream \
    --model facebook/detr-resnet-50 \
    --reid-model openai/clip-vit-base-patch32

# High-confidence tracking, save output
mata track dashcam.mp4 \
    --model facebook/detr-resnet-50 \
    --conf 0.5 --iou 0.6 \
    --save --save-dir runs/track/
```

---

## mata val

Evaluate a model on a labelled dataset. Wraps `mata.val()`.

```
mata val <task> --data <yaml> [options]
```

### Positional Arguments

| Argument | Description                                                |
| -------- | ---------------------------------------------------------- |
| `task`   | Task type: `detect`, `segment`, `classify`, `depth`, `ocr` |

### Options

| Flag           | Default     | Description                                             |
| -------------- | ----------- | ------------------------------------------------------- |
| `--model / -m` | `None`      | Model ID, local path, or config alias                   |
| `--data`       | _required_  | Path to dataset YAML configuration file                 |
| `--conf`       | `0.001`     | Confidence threshold                                    |
| `--iou`        | `0.5`       | IoU threshold for mAP calculation                       |
| `--device`     | `None`      | Compute device: `cpu`, `cuda`, `cuda:0`, `mps`          |
| `--split`      | `val`       | Dataset split to evaluate (`train`, `val`, `test`)      |
| `--save-dir`   | `runs/val/` | Directory for plots, CSV metrics, and per-image results |
| `--plots`      | `False`     | Save PR curve, F1 curve, and confusion matrix plots     |
| `--json`       | `False`     | Print all metrics as JSON to stdout                     |

### Dataset YAML Format

```yaml
# coco.yaml
path: data/coco
train: images/train
val: images/val
test: images/test

nc: 80
names: ["person", "bicycle", "car", ...]
```

### Examples

```bash
# Evaluate DETR on COCO validation split
mata val detect \
    --model facebook/detr-resnet-50 \
    --data coco.yaml \
    --plots

# OCR evaluation on COCO-Text
mata val ocr \
    --model easyocr \
    --data coco_text.yaml \
    --json

# Classification evaluation with strict threshold
mata val classify \
    --model openai/clip-vit-base-patch32 \
    --data imagenet.yaml \
    --conf 0.5 \
    --split test
```

---

## mata export

Export a model to a portable format. **Stub — full support arrives in v2.0.**

```
mata export <task> <model> [options]
```

### Positional Arguments

| Argument | Description                                |
| -------- | ------------------------------------------ |
| `task`   | Task type: `detect`, `segment`, `classify` |
| `model`  | Model ID or local path                     |

### Options

| Flag            | Default | Description                              |
| --------------- | ------- | ---------------------------------------- |
| `--format`      | `onnx`  | Export format: `onnx` or `torchscript`   |
| `--quantize`    | `None`  | Quantization: `int8` or `fp16`           |
| `--output / -o` | `None`  | Output file path (auto-named if omitted) |

### Example

```bash
mata export detect ./model.pt --format onnx --quantize int8
```

> Full export functionality is planned for v2.0. This command currently prints a stub notice.

---

## Output Formats

### Console Output (default)

Each command prints a human-readable summary:

```
# mata run detect
Detected 3 objects:
  person  conf=0.92  [120, 45, 340, 380]
  car     conf=0.87  [450, 100, 800, 420]
  bicycle conf=0.65  [60, 200, 180, 410]

# mata recognize
Best match:  alice  (similarity=0.8234)

# mata track
Tracked 450 frames | total detections: 1823
```

### JSON Output (`--json`)

Use `--json` to receive machine-readable output on stdout:

```bash
# Detection JSON
mata run detect image.jpg --model facebook/detr-resnet-50 --json
```

```json
{
  "instances": [
    {
      "score": 0.92,
      "label": 0,
      "label_name": "person",
      "bbox": [120, 45, 340, 380],
      "area": null,
      "is_stuff": null,
      "track_id": null,
      "mask": null,
      "embedding": null
    }
  ],
  "entities": [],
  "meta": {},
  "text": null,
  "prompt": null
}
```

### Exit Codes

| Code | Meaning                                                     |
| ---- | ----------------------------------------------------------- |
| `0`  | Success                                                     |
| `1`  | Runtime error (model load failure, file not found, etc.)    |
| `2`  | Argument parsing error (wrong flags / missing required arg) |

---

## Examples

### Quick Detection Pipeline

```bash
# Download and detect, print JSON
mata run detect https://example.com/street.jpg \
    --model facebook/detr-resnet-50 \
    --conf 0.4 \
    --json

# Save annotated image
mata run detect street.jpg \
    --model facebook/detr-resnet-50 \
    --save --save-dir ./output
```

### Gallery-Based Recognition Workflow

```bash
# 1. Build the gallery in Python
python - <<'EOF'
import mata
g = mata.Gallery()
for name, path in [("alice", "a.jpg"), ("bob", "b.jpg")]:
    g.add(name, path, model="openai/clip-vit-base-patch32")
g.save("gallery.npz")
EOF

# 2. Run recognition from the CLI
mata recognize query.jpg --gallery gallery.npz --top-k 3
```

### Tracking + ReID

```bash
mata track footage.mp4 \
    --tracker botsort \
    --model facebook/detr-resnet-50 \
    --reid-model openai/clip-vit-base-patch32 \
    --conf 0.3 \
    --save --save-dir runs/tracked/
```

### Batch Validation with Plots

```bash
mata val detect \
    --model facebook/detr-resnet-50 \
    --data coco_mini.yaml \
    --plots \
    --save-dir runs/val/detr/
```

---

**Version:** 1.9.5
**Date:** April 3, 2026
**Status:** ✅ Production Ready
