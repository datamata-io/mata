# MATA Example Notebooks

Interactive Jupyter notebooks demonstrating MATA's rich notebook display integration.

## Prerequisites

```bash
pip install datamata[notebook]
```

For each notebook, the required extras are listed in the first markdown cell.

## Notebooks

| # | Notebook | Task | Model |
|---|----------|------|-------|
| 01 | [Detection](01_detection.ipynb) | Object detection | `facebook/detr-resnet-50` |
| 02 | [Classification](02_classification.ipynb) | Zero-shot image classification | `openai/clip-vit-base-patch32` |
| 03 | [Segmentation](03_segmentation.ipynb) | Instance + panoptic segmentation | `facebook/mask2former-swin-tiny-coco-instance` |
| 04 | [Depth Estimation](04_depth_estimation.ipynb) | Monocular depth with magma colormap | `depth-anything/Depth-Anything-V2-Small-hf` |
| 05 | [Object Tracking](05_tracking.ipynb) | Multi-object tracking with track IDs | `facebook/detr-resnet-50` + BotSort |
| 06 | [VLM Q&A](06_vlm_query.ipynb) | Visual question answering | `Qwen/Qwen3-VL-2B-Instruct` |

## Rich Display Protocol

MATA result types implement the standard IPython rich display protocol:

| Result Type | Method | Output |
|-------------|--------|--------|
| `VisionResult` | `_repr_html_()` | HTML table with label / score / bbox |
| `ClassifyResult` | `_repr_html_()` | SVG bar chart + score table |
| `DepthResult` | `_repr_png_()` | Colormap PNG (magma) |
| `OCRResult` | `_repr_html_()` | HTML text region table |
| `BarcodeResult` | `_repr_html_()` | HTML decoded barcode table |
| `Embeddings` | `_repr_html_()` | Shape/dtype summary table |

Results auto-render when you evaluate them in a notebook cell:

```python
result = mata.run("detect", "image.jpg", model="facebook/detr-resnet-50")
result  # ← rich HTML table displays automatically
```

Use `mata.show()` for explicit display with extra options:

```python
mata.show(result, image="image.jpg")  # with image overlay
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for notebook contribution guidelines,
including how to set up `nbstripout` to keep cell outputs out of Git.
