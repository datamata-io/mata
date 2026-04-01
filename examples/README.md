# MATA Examples

> Quick-start examples for every MATA task. Pick a task folder and run.

## Quick Start

Your very first detection in 3 lines:

```python
import mata
result = mata.run("detect", "examples/images/000000039769.jpg", model="facebook/detr-resnet-50")
print(result)
```

```shell
Note: Download the test image (COCO 000000039769.jpg) if not already present:
wget -P examples/images http://images.cocodataset.org/val2017/000000039769.jpg (linux/macOS)
curl -o examples/images/000000039769.jpg http://images.cocodataset.org/val2017/000000039769.jpg (Windows)
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

| File                               | What it shows                                                              |
| ---------------------------------- | -------------------------------------------------------------------------- |
| [`basic_vlm.py`](vlm/basic_vlm.py) | Description, VQA, system prompts, load-once, metadata, structured output   |
| [`ocr.py`](vlm/ocr.py)             | EasyOCR, GLM-OCR, TrOCR, GOT-OCR2, PaddleOCR, Tesseract, export, filtering |

### Object Tracking ([`track/`](track/))

| File                                               | What it shows                                                |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [`basic_tracking.py`](track/basic_tracking.py)     | `mata.track()` one-liner, BotSort/ByteTrack, save output     |
| [`persist_tracking.py`](track/persist_tracking.py) | Per-frame tracking with `tracker.update()` YOLO-like pattern |
| [`stream_tracking.py`](track/stream_tracking.py)   | Constant-memory stream mode for video/RTSP                   |

### Barcode & QR Code ([`barcode/`](barcode/)) _(v1.9.3)_

| File                                     | What it shows                                                    |
| ---------------------------------------- | ---------------------------------------------------------------- |
| [`basic_scan.py`](barcode/basic_scan.py) | One-shot scan, load/reuse, pyzbar vs zxing, export, ROI pipeline |

### Feature Embedding ([`embed/`](embed/)) _(v1.9.6)_

Requires `pip install datamata[xclip]` (for `microsoft/xclip-base-patch32`). Videos not bundled — supply your own `.mp4` file.

| File                                                                    | What it shows                                                              |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`embed/video_semantic_search.py`](embed/video_semantic_search.py)      | Index a video with X-CLIP, search by text queries; multi-query comparison  |
| [`embed/video_search_by_image.py`](embed/video_search_by_image.py)      | Image-to-video search: index a video, find clips matching a query frame    |

### Inference Utilities ([`inference/`](inference/))

| File                                                                              | What it shows                                                                |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [`inference/embed_example.py`](inference/embed_example.py)                        | `mata.run("embed", ...)`, batch crops, pre-loaded adapter, save/load — mock + `--real` |
| [`inference/gallery_match_example.py`](inference/gallery_match_example.py)        | Gallery build, cosine search, `mata.run("recognize", ...)`, persistence, batch search — mock + `--real` |
| [`inference/embedding.py`](inference/embedding.py)                                | Low-level `EmbedAdapter` + `Embeddings` artifact; graph pipeline simulation  |

## Pipelines & Graphs ([`graph/`](graph/))

6 core examples + 20 industry scenarios.  
See [graph/README.md](graph/README.md) for the full guide.

Notable graph example:

| File                                                           | What it shows                                      |
| -------------------------------------------------------------- | -------------------------------------------------- |
| [`graph/graph_reid_pipeline.py`](graph/graph_reid_pipeline.py) | Detect → Track → Embed → ReID (cross-camera graph) |

## Tools & Utilities ([`tools/`](tools/))

| File                                           | What it shows                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| [`save_results.py`](tools/save_results.py)     | Export to JSON, CSV, image crops, segmentation overlays                      |
| [`config_aliases.py`](tools/config_aliases.py) | Define and use model aliases via `.mata/models.yaml`                         |
| [`onnx_inference.py`](tools/onnx_inference.py) | Detection & classification with local `.onnx` files, explicit ModelType, GPU |

## CLI Examples ([`cli/`](cli/)) _(v1.9.5)_

Shell and PowerShell scripts for every `mata` subcommand. See [cli/README.md](cli/README.md) for the full guide.

| Bash                                                     | PowerShell                                                 | Covers                                           |
| -------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| [`cli/getting_started.sh`](cli/getting_started.sh)       | [`cli/getting_started.ps1`](cli/getting_started.ps1)       | First steps: version, help, detect, classify     |
| [`cli/run_examples.sh`](cli/run_examples.sh)             | [`cli/run_examples.ps1`](cli/run_examples.ps1)             | `mata run` — all tasks                           |
| [`cli/track_examples.sh`](cli/track_examples.sh)         | [`cli/track_examples.ps1`](cli/track_examples.ps1)         | `mata track` — BotSort/ByteTrack, ReID           |
| [`cli/val_examples.sh`](cli/val_examples.sh)             | [`cli/val_examples.ps1`](cli/val_examples.ps1)             | `mata val` — dataset evaluation                  |
| [`cli/recognize_examples.sh`](cli/recognize_examples.sh) | [`cli/recognize_examples.ps1`](cli/recognize_examples.ps1) | `mata recognize` — gallery matching              |
| [`cli/export_examples.sh`](cli/export_examples.sh)       | [`cli/export_examples.ps1`](cli/export_examples.ps1)       | `mata export` — ONNX / TorchScript _(v2.0 stub)_ |

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

## Changelog

| Date       | Change                                                                                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-04-01 | Added `embed/video_semantic_search.py` and `embed/video_search_by_image.py` — X-CLIP semantic video search (v1.9.6)                                                |
| 2026-04-01 | Added `inference/embed_example.py`, `inference/gallery_match_example.py`, `inference/embedding.py` to README                                                       |
| 2026-04-01 | Fixed `inference/embed_example.py` real mode: `mata.run("embed")` returns `np.ndarray`; wrap in `EmbedResult`; use `embedder.embed()` not `embedder.predict()`     |
| 2026-04-01 | Fixed `inference/gallery_match_example.py` real mode: `result.matches` → `result.entries[0].all_matches` (dict access)                                             |
| 2026-03-19 | Fixed `MultiResult.__getitem__` and `__contains__` — `result['key']` and `'key' in result` now work correctly (was `TypeError` at runtime)                         |
| 2026-03-19 | Fixed `grounding_sam_pipeline.py` — moved private `_mask_to_binary` import to lazy in-place usage                                                                  |
| 2026-03-19 | Removed `graph/valkey_rtsp_pipeline.py` — required live RTSP + Valkey with no mock fallback; see `graph/valkey_pipeline.py` and `track/stream_tracking.py` instead |
| 2026-03-19 | Known issues in scenario `--real` mode documented in [`docs/2026-03-19_Bugs_Report.md`](../docs/2026-03-19_Bugs_Report.md)                                         |
