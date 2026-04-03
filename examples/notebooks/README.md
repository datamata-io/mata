# MATA Example Notebooks

Interactive Jupyter notebooks demonstrating MATA's rich notebook display integration.

## Prerequisites

```bash
pip install datamata[notebook]
```

For each notebook, the required extras are listed in the first markdown cell.

## Notebooks

| #   | Notebook                                            | Task                                                             | Model                                              |
| --- | --------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| 01  | [Detection](01_detection.ipynb)                     | Object detection                                                 | `facebook/detr-resnet-50`                          |
| 02  | [Classification](02_classification.ipynb)           | Zero-shot image classification                                   | `openai/clip-vit-base-patch32`                     |
| 03  | [Segmentation](03_segmentation.ipynb)               | Instance + panoptic segmentation                                 | `facebook/mask2former-swin-tiny-coco-instance`     |
| 04  | [Depth Estimation](04_depth_estimation.ipynb)       | Monocular depth with magma colormap                              | `depth-anything/Depth-Anything-V2-Small-hf`        |
| 05  | [Object Tracking](05_tracking.ipynb)                | Multi-object tracking with track IDs                             | `facebook/detr-resnet-50` + BotSort                |
| 06  | [VLM Q&A](06_vlm_query.ipynb)                       | Visual question answering                                        | `Qwen/Qwen3-VL-2B-Instruct`                        |
| 07  | [Barcode & QR Code](08_barcode_qr.ipynb)            | Barcode/QR scanning with pyzbar & zxing                          | `pyzbar` / `zxing`                                 |
| 08  | [OCR](09_ocr.ipynb)                                 | Text recognition — EasyOCR, PaddleOCR, Tesseract, TrOCR, GLM-OCR | `easyocr`, `zai-org/GLM-OCR`, …                    |
| 09  | [Graph — Simple](10_graph_simple.ipynb)             | Node lists, Graph builder, Pipe DSL, filtering, detect→segment   | `facebook/detr-resnet-50`, `facebook/sam-vit-base` |
| 10  | [Graph — Advanced](11_graph_advanced.ipynb)         | Parallel, conditional, VLM, presets, custom predicates           | Multiple (detect + classify + depth + VLM)         |
| 11  | [Graph — Control Flow](12_graph_control_flow.ipynb) | EarlyExit, While loops, conditional edges                        | Mock nodes (no download required)                  |
| 12  | [Graph — Video Search](13_graph_video_search.ipynb) | IndexVideo + EmbeddingSearch, semantic video retrieval           | `Qwen/Qwen3-VL-Embedding-2B`                       |

## Rich Display Protocol

MATA result types implement the standard IPython rich display protocol:

| Result Type      | Method          | Output                               |
| ---------------- | --------------- | ------------------------------------ |
| `VisionResult`   | `_repr_html_()` | HTML table with label / score / bbox |
| `ClassifyResult` | `_repr_html_()` | SVG bar chart + score table          |
| `DepthResult`    | `_repr_png_()`  | Colormap PNG (magma)                 |
| `OCRResult`      | `_repr_html_()` | HTML text region table               |
| `BarcodeResult`  | `_repr_html_()` | HTML decoded barcode table           |
| `Embeddings`     | `_repr_html_()` | Shape/dtype summary table            |

Results auto-render when you evaluate them in a notebook cell:

```python
result = mata.run("detect", "image.jpg", model="facebook/detr-resnet-50")
result  # ← rich HTML table displays automatically
```

Use `mata.show()` for explicit display with extra options:

```python
mata.show(result, image="image.jpg")  # with image overlay
```

## Notes

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for notebook contribution guidelines,
including how to set up `nbstripout` to keep cell outputs out of Git.
