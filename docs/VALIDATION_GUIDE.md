# MATA Validation & Evaluation Guide

**Version:** 1.9.0  
**Updated:** March 2026

`mata.val()` evaluates any supported model against a labeled dataset and returns structured metrics. It follows the YOLO-style validation pattern — one function call, per-class breakdowns, PR/F1 curves, confusion matrices, and formatted console tables.

**Supported tasks:** detection, segmentation, classification, depth estimation, OCR / text recognition.

---

## Table of Contents

- [Quick Start](#quick-start)
- [API Reference — `mata.val()`](#api-reference--mataval)
- [Dataset YAML Format](#dataset-yaml-format)
- [Dataset Download & Setup](#dataset-download--setup)
- [Metrics Reference by Task](#metrics-reference-by-task)
  - [DetMetrics (Detection)](#detmetrics-detection)
  - [SegmentMetrics (Segmentation)](#segmentmetrics-segmentation)
  - [ClassifyMetrics (Classification)](#classifymetrics-classification)
  - [DepthMetrics (Depth Estimation)](#depthmetrics-depth-estimation)
  - [OCRMetrics (OCR / Text Recognition)](#ocrmetrics-ocr--text-recognition)
- [Supporting Classes](#supporting-classes)
  - [Metric (Base AP Accumulator)](#metric-base-ap-accumulator)
  - [ConfusionMatrix](#confusionmatrix)
  - [DatasetLoader & GroundTruth](#datasetloader--groundtruth)
- [Plots & Visualization](#plots--visualization)
- [Standalone Mode](#standalone-mode)
- [Serialization & Export](#serialization--export)
- [Console Output](#console-output)
- [Architecture (For Contributors)](#architecture-for-contributors)
- [Low-Level Functions](#low-level-functions)
- [Known Limitations](#known-limitations)

---

## Quick Start

```python
import mata

# Detection — COCO mAP
metrics = mata.val(
    "detect",
    model="facebook/detr-resnet-50",
    data="examples/configs/coco.yaml",
    verbose=True,
    plots=True,
    save_dir="runs/val/detect",
)
print(f"mAP@50:    {metrics.box.map50:.3f}")   # e.g. 0.644
print(f"mAP@50-95: {metrics.box.map:.3f}")     # e.g. 0.456

# Segmentation — box + mask mAP
metrics = mata.val(
    "segment",
    model="shi-labs/oneformer_coco_swin_large",
    data="examples/configs/coco.yaml",
    verbose=True,
)
print(f"Box  mAP@50: {metrics.box.map50:.3f}")
print(f"Mask mAP@50: {metrics.seg.map50:.3f}")

# Classification — top-k accuracy
metrics = mata.val(
    "classify",
    model="microsoft/resnet-101",
    data="examples/configs/imagenet.yaml",
    verbose=True,
)
print(f"Top-1: {metrics.top1:.1%}")   # e.g. 81.9%
print(f"Top-5: {metrics.top5:.1%}")   # e.g. 95.7%

# Depth estimation
metrics = mata.val(
    "depth",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    data="examples/configs/diode.yaml",
    verbose=True,
)
print(f"AbsRel:   {metrics.abs_rel:.4f}")    # e.g. 0.3930
print(f"δ < 1.25: {metrics.delta_1:.1%}")    # e.g. 66.9%

# OCR — recognition metrics
metrics = mata.val(
    "ocr",
    model="easyocr",
    data="examples/configs/coco_text.yaml",
    verbose=True,
)
print(f"CER:      {metrics.cer:.4f}")         # e.g. 0.1523
print(f"WER:      {metrics.wer:.4f}")         # e.g. 0.2347
print(f"Accuracy: {metrics.accuracy:.1%}")    # e.g. 62.4%
```

---

## API Reference — `mata.val()`

```python
def val(
    task: str,
    *,
    model: str | Any | None = None,
    data: str | dict | None = None,
    predictions: list | None = None,
    ground_truth: str | list | None = None,
    conf: float = 0.001,
    iou: float = 0.50,
    device: str | None = None,
    verbose: bool = True,
    plots: bool = False,
    save_dir: str = "",
    split: str = "val",
    **kwargs,
) -> DetMetrics | SegmentMetrics | ClassifyMetrics | DepthMetrics | OCRMetrics
```

### Parameters

| Parameter      | Type                     | Default    | Description                                                                                                  |
| -------------- | ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------ |
| `task`         | `str`                    | _required_ | One of `"detect"`, `"segment"`, `"classify"`, `"depth"`, `"ocr"`                                             |
| `model`        | `str \| adapter \| None` | `None`     | HuggingFace ID, local path, config alias, or a pre-loaded adapter. Required when `predictions` is not given. |
| `data`         | `str \| dict \| None`    | `None`     | Path to a dataset YAML file, or dict with equivalent keys. Required when `predictions` is not given.         |
| `predictions`  | `list \| None`           | `None`     | Pre-computed `VisionResult`/`ClassifyResult`/`DepthResult` list (standalone mode).                           |
| `ground_truth` | `str \| list \| None`    | `None`     | COCO-format JSON path, or list of `GroundTruth` objects (standalone mode).                                   |
| `conf`         | `float`                  | `0.001`    | Minimum confidence threshold for predictions.                                                                |
| `iou`          | `float`                  | `0.50`     | IoU threshold for true-positive / false-positive matching.                                                   |
| `device`       | `str \| None`            | `None`     | Device for inference (`"cpu"`, `"cuda"`, `"cuda:0"`). Auto-detected if omitted.                              |
| `verbose`      | `bool`                   | `True`     | Print per-class metrics table to stdout.                                                                     |
| `plots`        | `bool`                   | `False`    | Save PR curve, F1 curve, and confusion matrix plots.                                                         |
| `save_dir`     | `str`                    | `""`       | Directory for plots and `metrics.json`. Empty string disables saving.                                        |
| `split`        | `str`                    | `"val"`    | Dataset split key within the YAML (`"val"`, `"test"`, `"train"`).                                            |

### Usage Modes

**Dataset-driven** — provide `model` + `data`; the `Validator` loads the model, iterates over images, runs inference, and computes metrics automatically:

```python
metrics = mata.val("detect", model="facebook/detr-resnet-50", data="coco.yaml")
```

**Standalone** — provide `predictions` + `ground_truth`; no inference is performed:

```python
metrics = mata.val("detect", predictions=my_preds, ground_truth="annotations.json")
```

### Return Type

Returns one of `DetMetrics`, `SegmentMetrics`, `ClassifyMetrics`, `DepthMetrics`, or `OCRMetrics` depending on `task`.

---

## Dataset YAML Format

Create a YAML file pointing to your dataset:

```yaml
# examples/configs/coco.yaml
path: /data/coco # dataset root (absolute or relative to CWD)
val: val2017 # sub-directory containing validation images
annotations: annotations/instances_val2017.json # COCO-format annotation JSON (relative to path)
names: # optional: class-index → name mapping
  0: person
  1: bicycle
  2: car
  # ... up to class 79
```

**Key fields:**

| Field         | Required | Description                                                           |
| ------------- | -------- | --------------------------------------------------------------------- |
| `path`        | Yes      | Root directory of the dataset                                         |
| `val`         | Yes      | Sub-directory containing images for the split being evaluated         |
| `annotations` | Yes      | Path to COCO-format JSON file (relative to `path`)                    |
| `names`       | No       | Class index → name mapping. Auto-extracted from COCO JSON if omitted. |

### Classification YAML

```yaml
# examples/configs/imagenet.yaml
path: /data/imagenet
val: val
annotations: imagenet_val_labels.json # {filename: class_index} mapping
```

### Depth YAML

```yaml
# examples/configs/diode.yaml
path: /data/diode
val: val/indoors # or val/outdoors
annotations: diode_val_annotations.json # {rgb_path: depth_npy_path} mapping
```

### OCR (COCO-Text) YAML

```yaml
# examples/configs/coco_text.yaml
path: /data/coco-text
val: val2014 # COCO 2014 validation images directory
annotations: cocotext_v2.json # COCO-Text annotations with "text" field per annotation
names:
  0: text
```

The annotation JSON must follow COCO-Text format — each annotation dict contains a `"text"` key with the transcription string:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "COCO_val2014_000001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "categories": [{ "id": 1, "name": "text" }],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 50, 20],
      "text": "STOP"
    }
  ]
}
```

---

## Dataset Download & Setup

### COCO 2017 (Detection & Segmentation)

The [COCO 2017](https://cocodataset.org/#download) validation split is ~1 GB images + ~240 MB annotations.

```bash
mkdir -p /data/coco && cd /data/coco

# Validation images (1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Annotations (241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

Expected layout:

```
/data/coco/
├── val2017/                     # ~5,000 validation images
└── annotations/
    └── instances_val2017.json
```

### ImageNet ILSVRC 2012 (Classification)

Requires a free account on [image-net.org](https://image-net.org/download).

```bash
mkdir -p /data/imagenet && cd /data/imagenet

tar -xvf ILSVRC2012_img_val.tar           # → val/  (50,000 images)
tar -xvf ILSVRC2012_bbox_val_v3.tgz       # → bbox_val/  (50,000 XMLs)

# Generate MATA-compatible annotation file
cd /path/to/MATA
python scripts/generate_imagenet_val_labels.py
```

Expected layout:

```
/data/imagenet/
├── val/                          # 50,000 validation images
└── imagenet_val_labels.json      # generated {filename: class_index}
```

### DIODE (Depth Estimation)

[DIODE](https://diode-dataset.org/) provides indoor and outdoor depth maps (~5 GB).

```bash
mkdir -p /data/diode && cd /data/diode
wget http://diode-dataset.s3.amazonaws.com/val.tar.gz
tar -xvf val.tar.gz
```

Expected layout:

```
/data/diode/
├── val/
│   ├── indoors/          # scene_*/scan_*/ — RGB .png + depth .npy pairs
│   └── outdoors/
└── diode_val_annotations.json
```

---

## Metrics Reference by Task

### DetMetrics (Detection)

`DetMetrics` holds bounding-box AP results computed over 10 COCO IoU thresholds (0.50–0.95).

```python
from mata import DetMetrics  # or: from mata.eval import DetMetrics
```

#### Fields

| Field              | Type                      | Description                                                        |
| ------------------ | ------------------------- | ------------------------------------------------------------------ |
| `names`            | `dict[int, str]`          | Class ID → name mapping                                            |
| `box`              | `Metric`                  | Inner AP accumulator (see [Metric](#metric-base-ap-accumulator))   |
| `speed`            | `dict[str, float]`        | `{"preprocess": ms, "inference": ms, "postprocess": ms}` per image |
| `confusion_matrix` | `ConfusionMatrix \| None` | Optional confusion matrix                                          |
| `save_dir`         | `str`                     | Output directory for plots                                         |

#### Key Properties

| Property                 | Returns                            | Example                      |
| ------------------------ | ---------------------------------- | ---------------------------- |
| `metrics.box.map`        | `float` — mAP at IoU 0.50–0.95     | `0.456`                      |
| `metrics.box.map50`      | `float` — mAP at IoU 0.50          | `0.644`                      |
| `metrics.box.map75`      | `float` — mAP at IoU 0.75          | `0.487`                      |
| `metrics.box.mp`         | `float` — mean precision           | `0.726`                      |
| `metrics.box.mr`         | `float` — mean recall              | `0.586`                      |
| `metrics.box.maps`       | `ndarray` — per-class AP@0.50-0.95 | shape `(nc,)`                |
| `metrics.speed`          | `dict` — timing per image          | `{"preprocess": 0.001, ...}` |
| `metrics.maps`           | `ndarray` — alias for `box.maps`   | shape `(nc,)`                |
| `metrics.ap_class_index` | `ndarray` — class IDs with AP data | shape `(nc,)`                |

#### Methods

| Method            | Returns             | Description                                                  |
| ----------------- | ------------------- | ------------------------------------------------------------ |
| `mean_results()`  | `tuple[float, ...]` | `(precision, recall, mAP50, mAP50-95)` averaged over classes |
| `class_result(i)` | `tuple[float, ...]` | Same metrics for class index `i`                             |
| `fitness()`       | `float`             | Weighted fitness score: `0.1 * mAP50 + 0.9 * mAP50-95`       |
| `summary()`       | `list[dict]`        | Per-class breakdown as list of dicts                         |
| `to_dict()`       | `dict`              | Full results as nested dict                                  |
| `to_json()`       | `str`               | JSON string of `to_dict()`                                   |
| `to_csv()`        | `str`               | CSV-formatted string (one row per class)                     |

#### Real-World Example Output

Evaluated with `facebook/detr-resnet-50` on COCO val2017 (5,000 images, 80 classes):

```
metrics.box.map50   = 0.644     # mAP at IoU 0.50
metrics.box.map     = 0.456     # mAP at IoU 0.50-0.95
metrics.box.mp      = 0.726     # mean precision
metrics.box.mr      = 0.586     # mean recall
metrics.fitness()   = 0.474
Speed: preprocess 0.001ms, inference 75.0ms, postprocess 0.011ms per image
```

---

### SegmentMetrics (Segmentation)

`SegmentMetrics` extends `DetMetrics` with a separate `seg` namespace for mask-level AP.

```python
from mata import SegmentMetrics  # or: from mata.eval import SegmentMetrics
```

#### Additional Field

| Field | Type     | Description               |
| ----- | -------- | ------------------------- |
| `seg` | `Metric` | Mask-level AP accumulator |

#### Key Properties

Box metrics are accessed via `metrics.box.*` (same as `DetMetrics`). Mask metrics use `metrics.seg.*`:

| Property            | Returns   | Description               |
| ------------------- | --------- | ------------------------- |
| `metrics.seg.map50` | `float`   | Mask mAP at IoU 0.50      |
| `metrics.seg.map`   | `float`   | Mask mAP at IoU 0.50-0.95 |
| `metrics.seg.map75` | `float`   | Mask mAP at IoU 0.75      |
| `metrics.seg.mp`    | `float`   | Mask mean precision       |
| `metrics.seg.mr`    | `float`   | Mask mean recall          |
| `metrics.seg.maps`  | `ndarray` | Per-class mask AP         |

#### Fitness & Aggregation

```python
# Fitness combines box and mask equally:
metrics.fitness()  # 0.5 * box.fitness() + 0.5 * seg.fitness()

# Maps combines both namespaces:
metrics.maps  # mean of box.maps and seg.maps
```

#### Methods

Same interface as `DetMetrics` — `mean_results()`, `class_result(i)`, `summary()`, `to_dict()`, `to_json()`, `to_csv()`.

For `SegmentMetrics`, `mean_results()` returns an 8-tuple: `(box_P, box_R, box_mAP50, box_mAP50-95, mask_P, mask_R, mask_mAP50, mask_mAP50-95)`.

#### Metrics Keys

`SegmentMetrics` exports 8 keys in `results_dict`:

```
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M)
```

---

### ClassifyMetrics (Classification)

`ClassifyMetrics` accumulates top-1 and top-5 accuracy across batches.

```python
from mata import ClassifyMetrics  # or: from mata.eval import ClassifyMetrics
```

#### Fields

| Field              | Type                      | Description                                    |
| ------------------ | ------------------------- | ---------------------------------------------- |
| `names`            | `dict[int, str]`          | Class ID → name mapping                        |
| `nc`               | `int`                     | Number of classes (auto-inferred from `names`) |
| `top1`             | `float`                   | Running top-1 accuracy (0.0–1.0)               |
| `top5`             | `float`                   | Running top-5 accuracy (0.0–1.0)               |
| `speed`            | `dict[str, float]`        | Timing breakdown per image                     |
| `confusion_matrix` | `ConfusionMatrix \| None` | Optional confusion matrix                      |

#### Key Method — `process_predictions()`

```python
metrics.process_predictions(
    pred_labels,    # (N,) predicted class indices
    target_labels,  # (N,) ground-truth class indices
    pred_top5=None, # (N, 5) top-5 predicted indices (optional)
)
```

Updates `top1` and `top5` incrementally. Can be called multiple times for batched evaluation.

#### Properties

| Property          | Returns | Example                    |
| ----------------- | ------- | -------------------------- |
| `metrics.top1`    | `float` | `0.8192`                   |
| `metrics.top5`    | `float` | `0.9569`                   |
| `metrics.fitness` | `float` | mean(top1, top5) = `0.888` |

#### Methods

| Method            | Returns               | Description                                              |
| ----------------- | --------------------- | -------------------------------------------------------- |
| `mean_results()`  | `tuple[float, float]` | `(top1, top5)`                                           |
| `class_result(i)` | `tuple`               | Per-class accuracy for class `i`                         |
| `summary()`       | `list[dict]`          | `[{"top1_acc": ..., "top5_acc": ..., "n_samples": ...}]` |
| `to_dict()`       | `dict`                | Full results dict                                        |
| `to_json()`       | `str`                 | JSON string                                              |
| `to_csv()`        | `str`                 | CSV string                                               |

#### Real-World Example Output

Evaluated with `microsoft/resnet-101` on ImageNet val (50,000 images, 1,000 classes):

```
metrics.top1    = 0.8192    # 81.92% top-1 accuracy
metrics.top5    = 0.9569    # 95.69% top-5 accuracy
metrics.fitness = 0.888
Speed: preprocess 0.002ms, inference 53.1ms, postprocess 0.006ms per image
```

---

### DepthMetrics (Depth Estimation)

`DepthMetrics` implements standard depth estimation metrics from Eigen et al. (2014).

```python
from mata import DepthMetrics  # or: from mata.eval import DepthMetrics
```

#### Configuration Fields

| Field          | Type   | Default | Description                                                                                                         |
| -------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `align_scale`  | `bool` | `True`  | Apply median scaling alignment before computing metrics. Compensates for scale ambiguity in monocular depth models. |
| `align_affine` | `bool` | `False` | Apply least-squares affine (scale + shift) alignment. Mutually exclusive with `align_scale`.                        |

#### Metric Fields

| Field      | Type               | Description                                                             |
| ---------- | ------------------ | ----------------------------------------------------------------------- | ----------- | ----- |
| `abs_rel`  | `float`            | Mean absolute relative error: $\frac{1}{n}\sum\frac{                    | d - \hat{d} | }{d}$ |
| `sq_rel`   | `float`            | Mean squared relative error: $\frac{1}{n}\sum\frac{(d - \hat{d})^2}{d}$ |
| `rmse`     | `float`            | Root mean squared error                                                 |
| `log_rmse` | `float`            | RMSE in log space                                                       |
| `delta_1`  | `float`            | % of pixels where $\max(\frac{\hat{d}}{d}, \frac{d}{\hat{d}}) < 1.25$   |
| `delta_2`  | `float`            | $\delta < 1.25^2$                                                       |
| `delta_3`  | `float`            | $\delta < 1.25^3$                                                       |
| `speed`    | `dict[str, float]` | Timing breakdown                                                        |

#### Key Methods

```python
# Per-image accumulation
metrics.process_batch(
    pred_depth,           # (H, W) float array
    gt_depth,             # (H, W) float array
    valid_mask=None,      # (H, W) bool array (optional)
)

# Finalize averages after all images
metrics.finalize()

# update() is an alias for process_batch()
metrics.update(pred_depth, gt_depth)
```

#### Properties

| Property               | Returns     | Description                            |
| ---------------------- | ----------- | -------------------------------------- |
| `metrics.fitness`      | `float`     | `delta_1 - abs_rel` (higher is better) |
| `metrics.keys`         | `list[str]` | 7 metric key names                     |
| `metrics.results_dict` | `dict`      | 8 entries (7 metrics + fitness)        |

#### Methods

| Method           | Returns             | Description                         |
| ---------------- | ------------------- | ----------------------------------- |
| `mean_results()` | `tuple[float, ...]` | All 7 metric values as a tuple      |
| `summary()`      | `list[dict]`        | Single-row summary with all metrics |
| `to_dict()`      | `dict`              | Full results dict                   |
| `to_json()`      | `str`               | JSON string                         |
| `to_csv()`       | `str`               | CSV string                          |

#### Real-World Example Output

Evaluated with `depth-anything/Depth-Anything-V2-Small-hf` on DIODE indoor (771 images):

```
metrics.abs_rel  = 0.3930    # absolute relative error
metrics.sq_rel   = 4.4949    # squared relative error
metrics.rmse     = 4.3978    # root mean squared error
metrics.log_rmse = 1.9102    # log-space RMSE
metrics.delta_1  = 0.6694    # 66.94% pixels within 1.25× ratio
metrics.delta_2  = 0.8158    # 81.58% within 1.25²
metrics.delta_3  = 0.8834    # 88.34% within 1.25³
metrics.fitness  = 0.2765
Speed: preprocess 0.002ms, inference 145.4ms, postprocess 0.013ms per image
```

---

### OCRMetrics (OCR / Text Recognition)

`OCRMetrics` implements recognition-only evaluation metrics: CER, WER, and exact-match accuracy.
Evaluation is image-level: all predicted text regions are concatenated and compared against all
ground-truth transcriptions (also concatenated).

```python
from mata import OCRMetrics  # or: from mata.eval import OCRMetrics
```

#### Configuration Fields

| Field            | Type   | Default | Description                                                                                                    |
| ---------------- | ------ | ------- | -------------------------------------------------------------------------------------------------------------- |
| `case_sensitive` | `bool` | `False` | When `False` (default), both predicted and GT text are lowercased before comparison. Matches ICDAR convention. |

#### Metric Fields

| Field      | Type               | Description                                                                                                 |
| ---------- | ------------------ | ----------------------------------------------------------------------------------------------------------- | ----------- | ------ |
| `cer`      | `float`            | Mean Character Error Rate: $\text{CER} = \frac{\text{Levenshtein}(pred, gt)}{\max(                          | gt          | , 1)}$ |
| `wer`      | `float`            | Mean Word Error Rate: $\text{WER} = \frac{\text{Levenshtein}(pred*{words}, gt*{words})}{\max(               | gt\_{words} | , 1)}$ |
| `accuracy` | `float`            | Exact-match accuracy: $\frac{\text{count}(pred = gt)}{N}$ — fraction of images with identical transcription |
| `speed`    | `dict[str, float]` | Timing breakdown per image                                                                                  |

#### Key Methods

```python
# Per-image accumulation
metrics.process_batch(
    pred_text,    # str — full predicted transcription for this image
    gt_text,      # str — full ground-truth transcription for this image
)

# Finalize averages after all images
metrics.finalize()

# update() is an alias for process_batch()
metrics.update(pred_text, gt_text)
```

#### Properties

| Property               | Returns     | Description                     |
| ---------------------- | ----------- | ------------------------------- |
| `metrics.fitness`      | `float`     | `accuracy` (exact-match ratio)  |
| `metrics.keys`         | `list[str]` | 3 metric key names              |
| `metrics.results_dict` | `dict`      | 4 entries (3 metrics + fitness) |

#### Methods

| Method           | Returns       | Description                         |
| ---------------- | ------------- | ----------------------------------- |
| `mean_results()` | `list[float]` | `[cer, wer, accuracy]`              |
| `summary()`      | `list[dict]`  | Single-row summary with all metrics |
| `to_dict()`      | `dict`        | Full results dict                   |
| `to_json()`      | `str`         | JSON string                         |
| `to_csv()`       | `str`         | CSV string (header + one data row)  |

#### `case_sensitive` Parameter

By default, text comparison is case-insensitive (`case_sensitive=False`), matching the ICDAR benchmark convention. Pass `case_sensitive=True` to `mata.val()` to enable strict casing:

```python
metrics = mata.val(
    "ocr",
    model="easyocr",
    data="examples/configs/coco_text.yaml",
    case_sensitive=True,  # “Stop” ≠ “stop”
)
```

#### Real-World Example Output

Evaluated with EasyOCR on COCO-Text v2 val split (~9,000 images):

```
metrics.cer       = 0.1523    # mean character error rate
metrics.wer       = 0.2347    # mean word error rate
metrics.accuracy  = 0.6240    # 62.4% exact-match accuracy
metrics.fitness   = 0.6240
Speed: preprocess 0.001ms, inference 210.3ms, postprocess 0.002ms per image
```

---

## Supporting Classes

### Metric (Base AP Accumulator)

`Metric` is the internal dataclass that stores per-class AP results. It powers the `.box` and `.seg` fields of `DetMetrics` and `SegmentMetrics`.

```python
from mata.eval.metrics import Metric
```

#### Properties

| Property         | Returns            | Description                                         |
| ---------------- | ------------------ | --------------------------------------------------- |
| `ap50`           | `ndarray (nc,)`    | Per-class AP at IoU 0.50                            |
| `ap`             | `ndarray (nc, 10)` | Per-class AP at each of 10 COCO thresholds          |
| `map50`          | `float`            | Mean AP at IoU 0.50                                 |
| `map75`          | `float`            | Mean AP at IoU 0.75                                 |
| `map`            | `float`            | Mean AP at IoU 0.50–0.95                            |
| `maps`           | `ndarray (nc,)`    | Per-class mAP (mean over 10 thresholds)             |
| `mp`             | `float`            | Mean precision (at optimal F1 threshold)            |
| `mr`             | `float`            | Mean recall (at optimal F1 threshold)               |
| `curves`         | `tuple`            | `(precision_curve, recall_curve, f1_curve, x_axis)` |
| `curves_results` | `tuple`            | 12-element tuple from `ap_per_class()`              |

#### Methods

| Method         | Signature                 | Description                                 |
| -------------- | ------------------------- | ------------------------------------------- |
| `update`       | `(results: tuple) → None` | Populate from `ap_per_class()` output       |
| `class_result` | `(i: int) → tuple`        | `(p, r, ap50, ap50-95)` for class index `i` |
| `mean_results` | `() → tuple`              | `(mean_p, mean_r, mean_ap50, mean_ap50-95)` |
| `fitness`      | `() → float`              | `0.1 * map50 + 0.9 * map`                   |

---

### ConfusionMatrix

`ConfusionMatrix` accumulates predictions batch by batch and supports both detection and classification tasks.

```python
from mata.eval.confusion_matrix import ConfusionMatrix
```

#### Constructor

```python
cm = ConfusionMatrix(
    nc,                    # Number of foreground classes
    names=None,            # {class_id: "name"} mapping
    task="detect",         # "detect" or "classify"
    conf_threshold=0.25,   # Min confidence for detection predictions
    iou_threshold=0.45,    # IoU threshold for positive match
)
```

**Matrix dimensions:**

- Detection: `(nc+1) × (nc+1)` — extra row/column for background (unmatched)
- Classification: `nc × nc`

#### Accumulation Methods

```python
# Detection: greedy highest-confidence-first matching
cm.process_batch(
    detections,   # (N, 6) array: [x1, y1, x2, y2, conf, class_id]
    labels,       # (M, 5) array: [class_id, x1, y1, x2, y2]
)

# Classification: simple pred→true tally
cm.process_cls_preds(
    preds,        # (N,) predicted class indices
    targets,      # (N,) ground-truth class indices
)
```

#### Properties & Methods

| Name                 | Returns      | Description                                         |
| -------------------- | ------------ | --------------------------------------------------- |
| `matrix`             | `ndarray`    | Raw confusion matrix                                |
| `tp_fp()`            | `(tp, fp)`   | True/false positive counts per class, shape `(nc,)` |
| `plot(...)`          | `None`       | Save matplotlib heatmap as `confusion_matrix.png`   |
| `print()`            | `None`       | Print raw matrix to stdout                          |
| `summary(normalize)` | `list[dict]` | Per-class TP/FP/FN breakdown                        |
| `to_json()`          | `str`        | JSON export                                         |
| `to_csv()`           | `str`        | CSV export                                          |

#### Plot Example

```python
# After validation completes:
metrics = mata.val("detect", model=..., data=..., plots=True, save_dir="runs/val/")
# → saves runs/val/confusion_matrix.png (when plots=True)

# Or manually from ConfusionMatrix:
metrics.confusion_matrix.plot(
    normalize=True,         # Row-normalize (percentages)
    save_dir="my_output/",  # Output directory
    names=class_names,      # Class name dict
)
```

---

### DatasetLoader & GroundTruth

`DatasetLoader` parses a dataset YAML config and provides an iterator over `(image_path, GroundTruth)` pairs.

```python
from mata.eval import DatasetLoader, GroundTruth
```

#### GroundTruth Dataclass

```python
@dataclass
class GroundTruth:
    image_id: int | str           # Unique image identifier
    image_path: str               # Absolute path to the image file
    boxes: np.ndarray             # (N, 4) xyxy bounding boxes
    labels: np.ndarray            # (N,) class indices
    masks: list | None = None     # Optional list of masks (RLE, binary, polygon)
    depth: np.ndarray | None      # Optional (H, W) ground-truth depth map
    image_size: tuple[int, int]   # (width, height)
    text: list[str] | None = None # Optional N transcription strings (OCR datasets only)
```

#### DatasetLoader Construction

```python
# From YAML config file
loader = DatasetLoader.from_yaml("examples/configs/coco.yaml")

# From COCO JSON directly
loader = DatasetLoader.from_coco_json(
    images_dir="/data/coco/val2017",
    json_path="/data/coco/annotations/instances_val2017.json",
)
```

#### Properties

| Property          | Returns          | Description                               |
| ----------------- | ---------------- | ----------------------------------------- |
| `names`           | `dict[int, str]` | Class ID → name mapping                   |
| `cat_id_to_label` | `dict[int, int]` | COCO category ID → contiguous label index |
| `class_names`     | `list[str]`      | Ordered class name list                   |

#### Iteration

```python
loader = DatasetLoader.from_yaml("coco.yaml")
for image_path, gt in loader:
    print(image_path)         # "/data/coco/val2017/000001.jpg"
    print(gt.boxes.shape)     # (N, 4) xyxy
    print(gt.labels.shape)    # (N,)
    print(gt.image_size)      # (640, 480)
```

---

## Plots & Visualization

Enable `plots=True` in `mata.val()` to generate visual outputs in `save_dir`:

```python
metrics = mata.val(
    "detect",
    model="facebook/detr-resnet-50",
    data="coco.yaml",
    plots=True,
    save_dir="runs/val/detect",
)
```

### Auto-Generated Plots

| File                   | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `PR_curve.png`         | Precision-Recall curve (top-5 classes + bold mean)            |
| `F1_curve.png`         | F1 score vs confidence threshold                              |
| `confusion_matrix.png` | Class confusion heatmap (when `ConfusionMatrix` is available) |

### Manual Plot Functions

The following functions are also available for custom plotting:

```python
from mata.eval.plots import plot_pr_curve, plot_f1_curve, plot_p_curve, plot_r_curve

# All follow the same signature pattern:
plot_pr_curve(
    px,          # (1000,) x-axis (recall values)
    py,          # (nc, 1000) y-axis (precision per class)
    ap,          # (nc,) AP per class (for legend)
    save_dir,    # Output directory
    names=None,  # {class_id: "name"} or list
    save_path=None,  # Override filename
)
```

**Visual style:** Thin gray lines for individual classes (top-5 highlighted if nc > 10), bold blue line for the mean, legend with AP@0.50 or max F1 value.

> **Note:** The Validator auto-generates only `PR_curve.png` and `F1_curve.png`. The `plot_p_curve()` and `plot_r_curve()` functions are available for manual use but are not called automatically.

---

## Standalone Mode

Standalone mode lets you score pre-computed predictions against a COCO-format ground-truth file without re-running inference. This is useful when inference runs on a separate machine (e.g., a GPU cluster).

```python
import mata

# Step 1: Collect predictions (can be done separately)
predictions = [
    mata.run("detect", img, model="facebook/detr-resnet-50")
    for img in image_list
]

# Step 2: Score against ground truth
metrics = mata.val(
    "detect",
    predictions=predictions,
    ground_truth="annotations/instances_val2017.json",
    conf=0.001,
    iou=0.50,
    verbose=True,
)
print(f"mAP@50: {metrics.box.map50:.3f}")
```

**Requirements:**

- `predictions` must be a list of result objects matching the task type
- `ground_truth` is a path to a COCO-format JSON file (detection/segmentation) or task-specific JSON
- Images are matched by filename between predictions and ground truth

See [`examples/validation.py`](../examples/validation.py) for complete, runnable examples of all four tasks plus the standalone workflow.

---

## Serialization & Export

All metrics classes support consistent serialization:

### Methods Available on All Metric Classes

```python
metrics.to_dict()   # → dict   — full results as nested dictionary
metrics.to_json()   # → str    — JSON string of to_dict()
metrics.to_csv()    # → str    — CSV-formatted string (one row per class)
metrics.summary()   # → list[dict] — per-class breakdown
```

### `metrics.json` Output Structure

When `save_dir` is set, `mata.val()` writes a `metrics.json` file. Detection example:

```json
{
  "results": {
    "metrics/precision(B)": 0.726,
    "metrics/recall(B)": 0.586,
    "metrics/mAP50(B)": 0.644,
    "metrics/mAP50-95(B)": 0.456,
    "fitness": 0.474
  },
  "speed": {
    "preprocess": 0.001,
    "inference": 75.036,
    "postprocess": 0.011
  },
  "per_class": [
    {
      "class_id": 0,
      "class_name": "person",
      "precision": 0.83,
      "recall": 0.695,
      "ap50": 0.795,
      "ap50_95": 0.548
    }
  ]
}
```

### `results_dict` Keys by Task

| Task     | Keys                                                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| detect   | `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `fitness`                                           |
| segment  | All detect keys + `metrics/precision(M)`, `metrics/recall(M)`, `metrics/mAP50(M)`, `metrics/mAP50-95(M)`                                    |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5`, `fitness`                                                                                 |
| depth    | `metrics/abs_rel`, `metrics/sq_rel`, `metrics/rmse`, `metrics/log_rmse`, `metrics/delta_1`, `metrics/delta_2`, `metrics/delta_3`, `fitness` |
| ocr      | `metrics/cer`, `metrics/wer`, `metrics/accuracy`, `fitness`                                                                                 |

---

## Console Output

When `verbose=True` (default), `mata.val()` prints a table to stdout.

### Detection / Segmentation Table

```
Class          Images  Instances  P       R       mAP50   mAP50-95
all              5000     36335  0.726   0.586   0.644    0.456
person           5000     10777  0.830   0.695   0.795    0.548
bicycle          5000       314  0.782   0.478   0.606    0.348
car              5000      1918  0.708   0.612   0.675    0.426
...

Speed: pre-process 0.0ms, inference 75.0ms, post-process 0.0ms per image
```

### Classification Table

```
top1_acc     top5_acc
0.819        0.957
```

### Depth Table

```
abs_rel  sq_rel   RMSE    log_RMSE  δ<1.25  δ<1.25²  δ<1.25³
0.3930   4.4949   4.3978  1.9102    0.6694  0.8158   0.8834
```

### OCR Table

```
         CER         WER    Accuracy
      0.1523      0.2347      0.6240

Speed: pre-process 0.0ms, inference 210.3ms, post-process 0.0ms per image
```

---

## Architecture (For Contributors)

### Module Map

```
src/mata/eval/
├── __init__.py           # Re-exports: Validator, DatasetLoader, GroundTruth + all metric classes
├── validator.py          # Validator — end-to-end evaluation pipeline (855 lines)
├── dataset.py            # DatasetLoader + GroundTruth — YAML/COCO JSON ingestion (501 lines)
├── confusion_matrix.py   # ConfusionMatrix — detect + classify modes (364 lines)
├── plots.py              # PR/F1/P/R curve plotting (385 lines)
├── printer.py            # Console table output (306 lines)
└── metrics/
    ├── __init__.py       # Re-exports all metric classes + low-level functions
    ├── base.py           # Metric dataclass + ap_per_class() — core 101-point COCO AP (404 lines)
    ├── detect.py         # DetMetrics — bounding-box AP container (195 lines)
    ├── segment.py        # SegmentMetrics — extends DetMetrics with mask AP (218 lines)
    ├── classify.py       # ClassifyMetrics — top-1/top-5 accumulator (203 lines)
    ├── depth.py          # DepthMetrics — Eigen et al. depth metrics (384 lines)
    ├── ocr.py            # OCRMetrics — CER/WER/accuracy recognition metrics (297 lines)
    └── iou.py            # box_iou, mask_iou, COCO_IOU_THRESHOLDS (227 lines)
```

### Validator Pipeline

```
Validator.run()
  │
  ├── _build_loader()           → DatasetLoader (from YAML or standalone GT)
  ├── _load_adapter()           → mata.load(task, model) — lazy adapter loading
  │
  ├── _iterate_images()         → for each (image, gt):
  │   ├── adapter.predict()        run inference with timing
  │   ├── _match_detections()      greedy IoU matching (detect)
  │   ├── _match_segments()        box + mask IoU matching (segment)
  │   ├── ClassifyMetrics.process_predictions()  (classify)
  │   └── DepthMetrics.process_batch()           (depth)
  │
  ├── _compute_metrics()        → ap_per_class() → Metric.update()
  ├── _build_label_remap()      → align adapter labels with GT labels
  ├── _build_confusion_matrix() → optional ConfusionMatrix
  ├── _save_plots()             → PR_curve.png, F1_curve.png, metrics.json
  └── _print_table()            → Printer.print_results()
```

### Label Remapping

The Validator automatically handles label mismatches between model predictions and ground-truth annotations. It builds a remapping table by matching class names between the adapter's label vocabulary and the dataset's `names` dict. This handles:

- DETR-style models (1-indexed COCO category IDs)
- OneFormer-style models (contiguous 0-indexed labels)
- Custom label sets (matched by string name)

### Mask IoU Fallback

When computing mask IoU, the system tries:

1. **pycocotools** (C-accelerated) — fast path for RLE masks
2. **Numpy matrix multiply** — fallback for binary/polygon masks or when pycocotools is unavailable

Both paths produce identical results. The fallback is ~3× slower but requires no extra dependencies.

### Extension Points

To add a new evaluation task:

1. Create a new dataclass in `src/mata/eval/metrics/` implementing `mean_results()`, `class_result(i)`, `fitness()`, `summary()`, `to_dict()`, `to_json()`, `to_csv()`
2. Add accumulation logic (like `process_predictions()` or `process_batch()`)
3. Register the task string in `Validator._SUPPORTED_TASKS`
4. Add a matching branch in `Validator._iterate_images()` and `Validator._compute_metrics()`
5. Add console header tuple in `Printer`

---

## Low-Level Functions

These functions from `mata.eval.metrics` are used internally but available for custom evaluation workflows.

### `ap_per_class()`

```python
from mata.eval.metrics import ap_per_class

result = ap_per_class(
    tp,               # (N, T) bool — true positive flags per IoU threshold
    conf,             # (N,) float — confidence scores
    pred_cls,         # (N,) int — predicted class IDs
    target_cls,       # (M,) int — ground-truth class IDs
    iou_thresholds=COCO_IOU_THRESHOLDS,  # 10 thresholds by default
    eps=1e-16,
)
# Returns 12-element tuple:
# (tp_at_f1, fp_at_f1, precision, recall, f1, all_ap,
#  unique_classes, p_curve, r_curve, f1_curve, x_axis, prec_values)
```

Computes 101-point interpolated average precision (matching `pycocotools.COCOeval` within 0.01 tolerance).

### `box_iou()`

```python
from mata.eval.metrics import box_iou

iou_matrix = box_iou(boxes1, boxes2)  # (N, M) pairwise IoU
# boxes1: (N, 4) xyxy, boxes2: (M, 4) xyxy
```

### `box_iou_batch()`

```python
from mata.eval.metrics import box_iou_batch

matches = box_iou_batch(pred_boxes, gt_boxes, iou_thresholds)
# Returns: (T, N, M) boolean match matrix
```

### `mask_iou()`

```python
from mata.eval.metrics import mask_iou

iou_matrix = mask_iou(masks1, masks2, image_shape=(H, W))
# Supports: RLE dicts, binary (H,W) arrays, polygon coordinate lists
```

### `COCO_IOU_THRESHOLDS`

```python
from mata.eval.metrics import COCO_IOU_THRESHOLDS
# [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

---

## Known Limitations

1. **Mask IoU fallback:** When `pycocotools` is not installed, mask IoU uses a numpy fallback that is functionally identical but ~3× slower.

2. **`num_targets` in JSON export:** Per-class instance counts appear as `-1` in the `metrics.json` `per_class` entries. The console table (verbose mode) shows correct instance counts, but they are not propagated to the serialized output.

3. **Auto-generated plots:** The Validator only auto-generates `PR_curve.png` and `F1_curve.png` when `plots=True`. The `plot_p_curve()` and `plot_r_curve()` functions exist but must be called manually if needed.

4. **Segmentation mask matching:** If the model does not produce masks (or mask format is incompatible), mask AP metrics will be zero while box metrics remain valid.

5. **Single-IoU confusion matrix:** The `ConfusionMatrix` operates at a single IoU threshold (default 0.45) and confidence threshold (default 0.25), regardless of the `iou` parameter passed to `mata.val()`.
6. **OCR recognition-only:** `OCRMetrics` computes recognition metrics only (CER, WER, exact-match accuracy). Text detection metrics (H-mean precision/recall/F1 on bounding-box matching) and end-to-end evaluation (combined detection + recognition) are not yet supported. Pass `mode="e2e"` is reserved for a future release.

7. **OCR image-level comparison:** Ground-truth transcriptions for all text regions in an image are concatenated with spaces before CER/WER computation. This avoids the hard problem of pairing predicted regions to GT regions (which requires IoU matching), but means per-region error rates are not available.

---

## ReID Tracking Notes (v1.9.2)

Appearance-Based Re-Identification (ReID) in MATA enhances BotSort's track-recovery capability after occlusion or target re-entry. This section covers how to enable ReID, inspect its outputs, and reason about tracking quality.

### What ReID Adds

Without ReID, BotSort associates detections to tracks using two cues:

1. **IoU** — spatial overlap between predicted and detected bounding boxes
2. **GMC** — global motion compensation (sparse optical flow) for camera motion

With ReID enabled (`reid_model=...`), a third cue is added: 3. **Cosine appearance distance** — L2-normalised embedding vectors extracted from detection crops are compared against cached track features (`smooth_feat`)

This allows BotSort to re-associate tracks even when the predicted position drifts significantly due to occlusion gaps.

### Enabling ReID

```python
import mata

# Single-camera tracking with ReID
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    tracker="botsort",
    reid_model="openai/clip-vit-base-patch32",  # any HF image encoder
    conf=0.3,
)

# Inspect per-instance embedding vectors
for frame_result in results:
    for inst in frame_result.instances:
        print(f"Track #{inst.track_id} embedding shape: "
              f"{inst.embedding.shape if inst.embedding is not None else 'N/A'}")
```

ONNX models are also supported:

```python
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    reid_model="osnet_x1_0.onnx",     # local ONNX ReID model
)
```

### Inspecting Embedding Quality

Each tracked instance with an active ReID encoder will have `Instance.embedding` populated with an L2-normalised float32 vector of shape `(D,)`:

```python
import numpy as np

for result in results:
    for inst in result.instances:
        if inst.embedding is not None:
            emb = inst.embedding
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-5, "Not unit norm"
            print(f"Track #{inst.track_id}: {emb.shape}, norm={np.linalg.norm(emb):.4f}")
```

### Cross-Camera ReID with `ReIDBridge`

`ReIDBridge` publishes confirmed-track embeddings to a shared Valkey store so independent tracker instances can resolve the same physical identity across feeds.

```python
from mata.trackers import ReIDBridge

# Camera A
bridge_a = ReIDBridge(
    "valkey://localhost:6379",
    camera_id="cam-a",
    ttl=300,                  # embeddings expire after 5 min
    similarity_thresh=0.25,   # cosine similarity cutoff
)

# mata.track() with reid_bridge: each confirmed track is published after update()
for result in mata.track(
    "rtsp://cam-a/stream",
    model="facebook/detr-resnet-50",
    reid_model="openai/clip-vit-base-patch32",
    reid_bridge=bridge_a,
    stream=True,
):
    active = [i for i in result.instances if i.track_id is not None]
    print(f"Active tracks: {len(active)}")

# Camera B — query nearest identity from cam-a
bridge_b = ReIDBridge("valkey://localhost:6379", camera_id="cam-b")
query_embedding = ...  # np.ndarray shape (D,), L2-normalised
matches = bridge_b.query(query_embedding, exclude_camera="cam-b", top_k=1)
if matches:
    print(f"Best cross-camera match: {matches[0]}")
    # {'track_id': 7, 'camera_id': 'cam-a', 'similarity': 0.83, ...}
```

### ReID Validation Tips

| Scenario                        | Recommended Approach                                                            |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Verify embeddings are populated | Check `inst.embedding is not None` after `update()`                             |
| Measure track-recovery rate     | Count frames where a lost track recovers its original ID                        |
| Tune appearance threshold       | Adjust `appearance_thresh` in `tracker_config` (BotSort default: 0.25)          |
| Reduce false re-associations    | Increase `reid_model` → use a more discriminative encoder (e.g., OSNet vs CLIP) |
| GPU inference for ReID          | Pass `device="cuda"` at `mata.load("track", ..., device="cuda")`                |
| ONNX production deployment      | Export your ReID model to ONNX and pass the `.onnx` path as `reid_model`        |

### Known Limitations (ReID)

1. **BotSort only:** ReID is integrated into BotSort's `get_dists()` method. ByteTrack does not support appearance-distance matching — `reid_model` is silently ignored when `tracker="bytetrack"`.

2. **No detection-level alignment:** ReID embeddings are computed for all detections that pass the confidence threshold, not only those that fail IoU association. For very dense scenes this may increase latency. Future work: skip ReID for IoU-matched detections (40–60% latency reduction).

3. **Cross-camera ID namespace:** Each tracker process maintains an independent `STrack._count` — cross-camera track IDs are not globally unique. `ReIDBridge` resolves this at the application layer by storing `(camera_id, track_id)` pairs.

4. **Embedding warm-up:** BotSort's `smooth_feat` is a running average that stabilises after ~5 frames. Track re-association quality may be lower for newly initialised tracks.

5. **Valkey dependency for `ReIDBridge`:** `ReIDBridge` requires `pip install datamata[valkey]` (or `datamata[redis]`). If the server is unreachable, `publish()` / `query()` log a warning and return gracefully — tracking continues unaffected.
