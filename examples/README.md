# MATA Examples

> Quick-start examples for every MATA task. Pick a task folder and run.

## Quick Start

Your very first detection in 3 lines:

```python
import mata
result = mata.run("detect", "examples/images/000000039769.jpg", model="facebook/detr-resnet-50")
print(result)
```

See [`detect/basic_detection.py`](detect/basic_detection.py) for the full progressive walkthrough.

## By Task

### Detection ([`detect/`](detect/))

| File                                                          | What it shows                                 |
| ------------------------------------------------------------- | --------------------------------------------- |
| [`basic_detection.py`](detect/basic_detection.py)             | One-shot, load/reuse, model switching, export |
| [`zeroshot_detection.py`](detect/zeroshot_detection.py)       | GroundingDINO, OWL-ViT with text prompts      |
| [`torchvision_detection.py`](detect/torchvision_detection.py) | CNN detection with RetinaNet, Faster R-CNN    |

### Classification ([`classify/`](classify/))

| File                                                          | What it shows                                                      |
| ------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`basic_classification.py`](classify/basic_classification.py) | One-shot, load/reuse, `.top1`/`.top5`, model comparison, filtering |
| [`clip_zeroshot.py`](classify/clip_zeroshot.py)               | CLIP zero-shot with text prompts, templates, threshold, batch      |

### Segmentation ([`segment/`](segment/))

| File                                                             | What it shows                                                 |
| ---------------------------------------------------------------- | ------------------------------------------------------------- |
| [`basic_segmentation.py`](segment/basic_segmentation.py)         | One-shot, instance vs panoptic, mask access, save overlay     |
| [`sam_segment.py`](segment/sam_segment.py)                       | SAM3 text/point/box prompts, load-once batch, post-processing |
| [`grounding_sam_pipeline.py`](segment/grounding_sam_pipeline.py) | GroundingDINO + SAM pipeline, custom thresholds               |

### Depth Estimation ([`depth/`](depth/))

| File                                     | What it shows                                   |
| ---------------------------------------- | ----------------------------------------------- |
| [`basic_depth.py`](depth/basic_depth.py) | Depth Anything V1/V2, load-once, save depth map |

### VLM & OCR ([`vlm/`](vlm/))

| File                               | What it shows                                                            |
| ---------------------------------- | ------------------------------------------------------------------------ |
| [`basic_vlm.py`](vlm/basic_vlm.py) | Description, VQA, system prompts, load-once, metadata, structured output |
| [`ocr.py`](vlm/ocr.py)             | EasyOCR, PaddleOCR, Tesseract, GOT-OCR2, TrOCR, export, filtering        |

### Object Tracking ([`track/`](track/))

| File                                               | What it shows                                                |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [`basic_tracking.py`](track/basic_tracking.py)     | `mata.track()` one-liner, BotSort/ByteTrack, save output     |
| [`persist_tracking.py`](track/persist_tracking.py) | Per-frame tracking with `tracker.update()` YOLO-like pattern |
| [`stream_tracking.py`](track/stream_tracking.py)   | Constant-memory stream mode for video/RTSP                   |

## Pipelines & Graphs ([`graph/`](graph/))

5 core examples + 20 industry scenarios.  
See [graph/README.md](graph/README.md) for the full guide.

## Tools & Utilities ([`tools/`](tools/))

| File                                           | What it shows                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| [`save_results.py`](tools/save_results.py)     | Export to JSON, CSV, image crops, segmentation overlays                      |
| [`config_aliases.py`](tools/config_aliases.py) | Define and use model aliases via `.mata/models.yaml`                         |
| [`onnx_inference.py`](tools/onnx_inference.py) | Detection & classification with local `.onnx` files, explicit ModelType, GPU |

## Validation

[`validation.py`](validation.py) — Evaluate models against COCO, ImageNet, and custom datasets.

## Dataset Configs ([`configs/`](configs/))

| File                                                               | Dataset                       |
| ------------------------------------------------------------------ | ----------------------------- |
| [`coco.yaml`](configs/coco.yaml)                                   | COCO detection / segmentation |
| [`coco_text.yaml`](configs/coco_text.yaml)                         | COCO-Text OCR evaluation      |
| [`imagenet.yaml`](configs/imagenet.yaml)                           | ImageNet classification       |
| [`diode.yaml`](configs/diode.yaml)                                 | DIODE depth estimation        |
| [`torchvision_aliases.yaml`](configs/torchvision_aliases.yaml)     | Torchvision model aliases     |
| [`torchvision_detection.yaml`](configs/torchvision_detection.yaml) | Torchvision detection config  |
