# MATA Training Guide

> **Version**: v2.0  
> **Module**: `mata.training`  
> **API**: `mata.train()`, `mata.finetune()`

> [!WARNING]
> **Beta Feature** — `mata.train()` and `mata.finetune()` are beta in v2.0.0.
> The API surface is stable but internal behavior may change in v2.1.0.
> Report issues at [github.com/datamata-io/mata/issues](https://github.com/datamata-io/mata/issues).

Train and fine-tune computer vision models directly within the MATA framework — no third-party training loops required.

---

## Table of Contents

1. [Overview & Quickstart](#1-overview--quickstart)
2. [Supported Tasks & Models](#2-supported-tasks--models)
3. [Dataset Formats](#3-dataset-formats)
4. [Data Augmentation](#4-data-augmentation)
5. [Training API Reference](#5-training-api-reference)
6. [Training Configuration](#6-training-configuration)
7. [Fine-Tuning Guide](#7-fine-tuning-guide)
8. [Checkpoint Management](#8-checkpoint-management)
9. [Evaluation Integration](#9-evaluation-integration)
10. [Reloading Trained Models](#10-reloading-trained-models)
11. [HuggingFace vs Torchvision Training](#11-huggingface-vs-torchvision-training)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)

---

## 1. Overview & Quickstart

The MATA training module provides a single, consistent API for training and fine-tuning models across detection, classification, and segmentation tasks. It automatically routes to the correct backend engine depending on the model source:

- **HuggingFace models** (`facebook/detr-resnet-50`, `microsoft/resnet-50`, …) → `transformers.Trainer`
- **Torchvision models** (`torchvision/fasterrcnn_resnet50_fpn`, …) → custom PyTorch loop

### Installation

```bash
# Core training (no extra deps beyond PyTorch + transformers)
pip install datamata

# Optional: richer augmentations + progress bars
pip install "datamata[training]"
```

### 30-Second Quickstart

```python
import mata

# --- Fine-tune a detection model ---
result = mata.finetune(
    "detect",
    model="facebook/detr-resnet-50",
    data="data/my_coco_dataset.yaml",   # COCO-format YAML
    epochs=10,
    batch_size=4,
)
print(result.summary())

# --- Reload best checkpoint and run inference ---
detector = mata.load("detect", result.best_checkpoint)
detections = mata.run("detect", "test.jpg", model=detector)
```

---

## 2. Supported Tasks & Models

### Tasks

| Task                  | `mata.train()` task arg | Engines supported        |
| --------------------- | ----------------------- | ------------------------ |
| Object Detection      | `"detect"`              | HuggingFace, Torchvision |
| Image Classification  | `"classify"`            | HuggingFace              |
| Instance Segmentation | `"segment"`             | HuggingFace              |

> Depth estimation, VLM, OCR, object tracking (`track`), and pose estimation (`pose`) training are out of scope for v2.0 (inference-only tasks).

### HuggingFace Detection Models

Any model loadable via `AutoModelForObjectDetection` is supported:

```
facebook/detr-resnet-50
facebook/detr-resnet-101
PekingU/rtdetr_v2_r18vd
PekingU/rtdetr_v2_r50vd
IDEA-Research/grounding-dino-tiny    # zero-shot (no label list needed)
```

### HuggingFace Classification Models

Any model loadable via `AutoModelForImageClassification`:

```
microsoft/resnet-50
google/vit-base-patch16-224
facebook/convnext-base-224
openai/clip-vit-base-patch32         # zero-shot classification
```

### HuggingFace Segmentation Models

Models loadable via `Mask2FormerForUniversalSegmentation`:

```
facebook/mask2former-swin-large-ade-semantic
facebook/mask2former-swin-base-coco-instance
```

### Torchvision Detection Models

All seven models from `TorchvisionDetectAdapter` are supported:

| Model key                  | `model=` argument                           |
| -------------------------- | ------------------------------------------- |
| Faster R-CNN ResNet-50 FPN | `torchvision/fasterrcnn_resnet50_fpn`       |
| Faster R-CNN v2            | `torchvision/fasterrcnn_resnet50_fpn_v2`    |
| RetinaNet ResNet-50 FPN    | `torchvision/retinanet_resnet50_fpn`        |
| RetinaNet v2               | `torchvision/retinanet_resnet50_fpn_v2`     |
| FCOS ResNet-50 FPN         | `torchvision/fcos_resnet50_fpn`             |
| SSD300 VGG16               | `torchvision/ssd300_vgg16`                  |
| SSDLite320 MobileNetV3     | `torchvision/ssdlite320_mobilenet_v3_large` |

---

## 3. Dataset Formats

### COCO Format (Detection & Segmentation)

The standard COCO JSON format is fully supported. Use a YAML config file to point to your splits:

**`examples/configs/coco.yaml`:**

```yaml
path: /data/coco # Root directory (optional prefix)
train: train2017 # Images directory for training
val: val2017 # Images directory for validation
train_annotations: annotations/instances_train2017.json
val_annotations: annotations/instances_val2017.json
names:
  0: person
  1: bicycle
  2: car
  # ...
```

**Or use explicit paths:**

```python
from mata.training.datasets import COCODetectionDataset

dataset = COCODetectionDataset(
    root="/data/coco/train2017",
    annotation_file="/data/coco/annotations/instances_train2017.json",
)
print(f"Loaded {len(dataset)} images")
print(f"Classes: {dataset.class_names}")
```

**Key behaviors:**

- Bounding boxes are always `xyxy` format (absolute pixel coordinates)
- Labels are 0-indexed
- Crowd annotations (`iscrowd=1`) are automatically excluded
- Images with zero annotations return empty tensors (not skipped)

### COCO Segmentation Format

For instance segmentation, use `COCOSegmentationDataset`. The YAML format is identical; the dataset returns binary masks in addition to bounding boxes:

```python
from mata.training.datasets import COCOSegmentationDataset

dataset = COCOSegmentationDataset("coco.yaml", split="train")
image, target = dataset[0]
# target["masks"]: torch.Tensor of shape (N, H, W)  — binary masks
# target["boxes"]: torch.Tensor of shape (N, 4)     — xyxy coords
# target["labels"]: torch.Tensor of shape (N,)      — class indices
```

### VOC Format (Detection)

Pascal VOC XML annotations are supported for detection:

```
data/voc/
├── JPEGImages/          ← images (.jpg)
├── Annotations/         ← one .xml per image
└── ImageSets/
    └── Main/
        ├── train.txt
        └── val.txt
```

```python
from mata.training.datasets import VOCDetectionDataset

dataset = VOCDetectionDataset(
    root="/data/voc",
    split="train",            # reads ImageSets/Main/train.txt
    skip_difficult=True,      # exclude "difficult" objects
)
```

### ImageFolder Format (Classification)

For classification, organize images into class sub-directories:

```
data/flowers/
├── train/
│   ├── roses/        ← all training images for class "roses"
│   ├── sunflowers/
│   └── tulips/
└── val/
    ├── roses/
    ├── sunflowers/
    └── tulips/
```

```python
# Direct usage in mata.train()
result = mata.train(
    "classify",
    model="microsoft/resnet-50",
    data="data/flowers/train",       # train split directory
    val_data="data/flowers/val",     # validation split directory
)
```

Class names are **auto-discovered** from sub-directory names, sorted alphabetically.

### Custom PyTorch Dataset

Pass any `torch.utils.data.Dataset` directly — no YAML or format conversion needed:

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = ...      # torch.Tensor (C, H, W)
        target = {
            "boxes":  torch.tensor([[x1, y1, x2, y2], ...]),   # xyxy
            "labels": torch.tensor([class_id, ...]),
        }
        return image, target

    def __len__(self):
        return 1000

# Pass-through — DatasetFactory wraps it transparently
result = mata.train("detect", model="torchvision/fasterrcnn_resnet50_fpn",
                    data=MyDataset())
```

### DatasetFactory Auto-Detection

When a path string is passed to `mata.train()`, the `DatasetFactory` auto-detects the format:

| Input                                                   | Detected format             |
| ------------------------------------------------------- | --------------------------- |
| `*.yaml` with `annotations:` / `train_annotations:` key | COCO JSON (via YAML)        |
| Directory with `Annotations/*.xml`                      | VOC XML                     |
| Directory with `*.json` files                           | COCO JSON (first JSON used) |
| Directory whose children are all sub-directories        | ImageFolder                 |
| `torch.utils.data.Dataset` instance                     | Pass-through                |

---

## 4. Data Augmentation

### Built-in Augmentations

MATA includes task-specific augmentation pipelines based on `torchvision.transforms.v2`. They are enabled by default (`augment=True`) and require no extra dependencies.

**Detection augmentations** (training):

- Random horizontal flip (mirrors bounding boxes)
- Random resize with aspect-ratio preservation
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

**Classification augmentations** (training):

- `RandomResizedCrop(224)`
- `RandomHorizontalFlip`
- `ColorJitter`
- ImageNet normalization

**Validation mode** (resize + center crop + normalize only):

```python
from mata.training.augmentations import AugmentationFactory

val_aug = AugmentationFactory.create("classify", train=False)
```

**Custom size:**

```python
aug = AugmentationFactory.create("detect", config={"size": 640})
```

### Albumentations Integration

For advanced augmentations (elastic transforms, mosaic, grid distortion, etc.), wrap an `albumentations.Compose` pipeline:

```bash
pip install "datamata[training]"   # installs albumentations
```

```python
import albumentations as A
from mata.training.augmentations import AugmentationFactory

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.GaussNoise(p=0.1),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)

aug = AugmentationFactory.create(
    "detect",
    config={
        "type": "albumentations",
        "transform": transform,
    },
)
```

Pass the augmentation config dict to `mata.train()`:

```python
result = mata.train(
    "detect",
    model="facebook/detr-resnet-50",
    data="data/coco.yaml",
    augment_config={
        "type": "albumentations",
        "transform": transform,
    },
)
```

---

## 5. Training API Reference

### `mata.train()`

Train a model from scratch or continue from a checkpoint.

```python
mata.train(
    task: str,
    *,
    model: str,
    data: str | dict,
    val_data: str | dict | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    optimizer: str = "adamw",           # "adamw" | "adam" | "sgd"
    weight_decay: float = 0.01,
    scheduler: str = "cosine",          # "cosine" | "linear" | "step" | "none"
    warmup_epochs: int = 1,
    device: str = "auto",               # "auto" | "cpu" | "cuda" | "cuda:0"
    amp: bool = True,                   # automatic mixed precision
    save_dir: str = "runs/train",
    save_every: int = 0,                # 0 = best + last only
    val_every: int = 1,                 # validate every N epochs
    patience: int = 0,                  # 0 = disabled
    freeze_backbone: bool = False,
    freeze_layers: list[str] | None = None,
    augment: bool = True,
    augment_config: dict | None = None,
    resume: str | None = None,          # checkpoint path to resume from
    num_workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
) -> TrainingResult
```

**Example:**

```python
result = mata.train(
    "detect",
    model="facebook/detr-resnet-50",
    data="examples/configs/coco.yaml",
    epochs=20,
    batch_size=4,
    lr=1e-4,
    device="cuda",
    val_every=2,
    save_every=5,
    patience=10,
    verbose=True,
)
```

### `mata.finetune()`

Fine-tune a pre-trained model. Identical to `mata.train()` but with fine-tuning defaults:

| Parameter         | `train()` default | `finetune()` default |
| ----------------- | ----------------- | -------------------- |
| `lr`              | `1e-4`            | `1e-5`               |
| `epochs`          | `10`              | `5`                  |
| `batch_size`      | `8`               | `16`                 |
| `freeze_backbone` | `False`           | `True`               |

```python
result = mata.finetune(
    "classify",
    model="microsoft/resnet-50",
    data="data/flowers/train",
    val_data="data/flowers/val",
    epochs=10,            # override the default of 5
)
```

### `TrainingResult`

Both `mata.train()` and `mata.finetune()` return a `TrainingResult`:

```python
result.best_metrics       # Best validation metrics (DetMetrics / ClassifyMetrics / SegmentMetrics)
result.final_metrics      # Metrics from the final epoch
result.best_checkpoint    # str: path to the best model checkpoint directory
result.last_checkpoint    # str: path to the last checkpoint directory
result.history            # dict[str, list[float]]: per-epoch metrics
result.config             # TrainingConfig used for the run
result.epochs_completed   # int: actual epochs run (may be < config.epochs with early stopping)

print(result.summary())   # Human-readable training summary
result.plot_loss()        # Plot training/validation loss curve (requires matplotlib)

# Access training history
import matplotlib.pyplot as plt
plt.plot(result.history["train_loss"], label="train")
plt.plot(result.history.get("val_map50", []), label="val mAP50")
plt.legend(); plt.show()
```

---

## 6. Training Configuration

### YAML Configuration Files

All training parameters can be defined in a YAML file and loaded with `TrainingConfig.from_yaml()`:

```yaml
# examples/configs/training_detect.yaml

task: detect
model: facebook/detr-resnet-50
data: examples/configs/coco.yaml
val_data: null # null → use 'val' split from data config

epochs: 20
batch_size: 4
lr: 1.0e-4
optimizer: adamw # adamw | adam | sgd
weight_decay: 0.01
scheduler: cosine # cosine | linear | step | none
warmup_epochs: 2

device: auto # auto | cpu | cuda | cuda:0
amp: true # automatic mixed precision (CUDA only)

save_dir: runs/train/detect
save_every: 5 # checkpoint every 5 epochs; 0 = best + last only
val_every: 1 # validate every epoch
patience: 10 # early stopping after 10 no-improvement epochs; 0 = disabled

freeze_backbone: false
augment: true

num_workers: 4
gradient_accumulation_steps: 1 # effective batch = batch_size × N
gradient_checkpointing: false # true = less VRAM, ~20-30% slower
max_grad_norm: 1.0 # 0.0 = disabled; use 0.1 for DETR
seed: 42
verbose: true
```

**Usage:**

```python
from mata.training import TrainingConfig
import mata

config = TrainingConfig.from_yaml("examples/configs/training_detect.yaml")
result = mata.train(config.task, **config.__dict__)
```

### All Hyperparameters

| Parameter                     | Type          | Default        | Description                                                              |
| ----------------------------- | ------------- | -------------- | ------------------------------------------------------------------------ |
| `task`                        | `str`         | —              | Task type: `"detect"`, `"classify"`, `"segment"`                         |
| `model`                       | `str`         | —              | Model source: HF ID, `torchvision/*`, alias, or checkpoint path          |
| `data`                        | `str \| dict` | —              | Training dataset (YAML, directory, or COCO JSON)                         |
| `val_data`                    | `str \| None` | `None`         | Validation dataset (uses `val` split from `data` if `None`)              |
| `epochs`                      | `int`         | `10`           | Number of training epochs                                                |
| `batch_size`                  | `int`         | `8`            | Training batch size                                                      |
| `lr`                          | `float`       | `1e-4`         | Initial learning rate                                                    |
| `optimizer`                   | `str`         | `"adamw"`      | Optimizer: `"adamw"`, `"adam"`, `"sgd"`                                  |
| `weight_decay`                | `float`       | `0.01`         | L2 regularization coefficient                                            |
| `scheduler`                   | `str`         | `"cosine"`     | LR scheduler: `"cosine"`, `"linear"`, `"step"`, `"none"`                 |
| `warmup_epochs`               | `int`         | `1`            | Linear LR warmup from 0 → `lr` (must be < `epochs`)                      |
| `device`                      | `str`         | `"auto"`       | Device: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`                          |
| `amp`                         | `bool`        | `True`         | Automatic mixed precision (CUDA only; silently disabled on CPU)          |
| `save_dir`                    | `str`         | `"runs/train"` | Root directory for checkpoints — auto-incremented                        |
| `save_every`                  | `int`         | `0`            | Periodic checkpoint every N epochs; `0` = best + last only               |
| `val_every`                   | `int`         | `1`            | Run validation every N epochs                                            |
| `patience`                    | `int`         | `0`            | Early stopping; `0` = disabled                                           |
| `freeze_backbone`             | `bool`        | `False`        | Freeze all backbone parameters                                           |
| `freeze_layers`               | `list[str]`   | `None`         | Freeze specific layers by name pattern                                   |
| `augment`                     | `bool`        | `True`         | Enable built-in data augmentation                                        |
| `augment_config`              | `dict`        | `None`         | Custom augmentation config (e.g., albumentations)                        |
| `resume`                      | `str`         | `None`         | Path to a checkpoint directory to resume from                            |
| `num_workers`                 | `int`         | `4`            | DataLoader worker processes                                              |
| `gradient_accumulation_steps` | `int`         | `1`            | Accumulate gradients over N steps (effective batch = `batch_size × N`)   |
| `gradient_checkpointing`      | `bool`        | `False`        | Recompute activations on the backward pass to save VRAM (~20–30% slower) |
| `max_grad_norm`               | `float`       | `1.0`          | Gradient clipping max norm (`0.0` = disabled; try `0.1` for DETR)        |
| `seed`                        | `int`         | `42`           | Random seed for reproducibility                                          |
| `verbose`                     | `bool`        | `True`         | Print progress table to console                                          |

### Validation

`TrainingConfig.validate()` is called automatically before training and raises `ConfigurationError` with actionable messages on invalid inputs:

```python
from mata.training import TrainingConfig
from mata.core.exceptions import ConfigurationError

config = TrainingConfig(task="invalid", model="facebook/detr-resnet-50", data="coco.yaml")
try:
    config.validate()
except ConfigurationError as exc:
    print(exc)
# → Invalid task 'invalid'. Must be one of: classify, detect, segment.
```

---

## 7. Fine-Tuning Guide

### When to Fine-Tune vs. Train from Scratch

| Scenario                                | Recommendation                                           |
| --------------------------------------- | -------------------------------------------------------- |
| Custom dataset with ≥ 1k images         | `mata.finetune(freeze_backbone=True)`                    |
| Large custom dataset (≥ 10k)            | `mata.train(freeze_backbone=False)`                      |
| Very small dataset (< 500 images)       | `mata.finetune(freeze_backbone=True, epochs=3)`          |
| Domain shift (medical, satellite, etc.) | `mata.train()` or `mata.finetune(freeze_backbone=False)` |

### Backbone Freezing

Setting `freeze_backbone=True` freezes all backbone parameters and only updates the detection/classification head. This is faster, uses less memory, and prevents catastrophic forgetting when starting from a strong pre-trained model.

```python
# Only the detection head is trainable
result = mata.finetune(
    "detect",
    model="facebook/detr-resnet-50",
    data="coco_custom.yaml",
    freeze_backbone=True,     # default for finetune()
    lr=1e-5,                  # conservative LR — head only
    epochs=10,
)
```

### Partial Layer Freezing

Use `freeze_layers` to freeze specific layers by name pattern:

```python
result = mata.train(
    "classify",
    model="microsoft/resnet-50",
    data="data/flowers/train",
    freeze_layers=[
        "layer1",     # freezes all parameters matching "layer1"
        "layer2",
    ],
    lr=5e-5,
)
```

### Learning Rate Schedule Recommendations

| Scenario                      | `scheduler` | `warmup_epochs` |
| ----------------------------- | ----------- | --------------- |
| Fine-tuning (frozen backbone) | `"cosine"`  | `1`             |
| Training from scratch         | `"cosine"`  | `2-3`           |
| SGD with momentum             | `"step"`    | `0`             |
| Quick experiments             | `"none"`    | `0`             |

### Optimizer Recommendations

| Optimizer           | Best for                                             |
| ------------------- | ---------------------------------------------------- |
| `"adamw"` (default) | Transformer-based models (DETR, ViT)                 |
| `"adam"`            | CNN-based HuggingFace models                         |
| `"sgd"`             | Torchvision models (follows torchvision conventions) |

---

## 8. Checkpoint Management

### Automatic Saving

MATA automatically saves checkpoints during training:

```
runs/train/detect/
├── best/                    ← best validation metric (updated during training)
│   ├── model_state.pth
│   ├── optimizer_state.pth
│   ├── training_state.json
│   └── config.json
└── last/                    ← checkpoint from the final epoch
    ├── model_state.pth
    └── ...
```

If `save_every > 0`, periodic checkpoints are also saved:

```
runs/train/detect/
├── best/
├── last/
├── epoch_005/               ← saved at epoch 5 (save_every=5)
├── epoch_010/
└── ...
```

If the directory already exists, it is **auto-incremented** (`detect2`, `detect3`, …) to prevent overwrites.

### Resuming Training

```python
result = mata.train(
    "detect",
    model="facebook/detr-resnet-50",
    data="coco.yaml",
    resume="runs/train/detect/last",   # path to an existing checkpoint directory
    epochs=30,                          # total epochs (will continue from saved epoch)
)
```

### Checkpoint Files Explained

| File                  | Contents                                                                          |
| --------------------- | --------------------------------------------------------------------------------- |
| `model_state.pth`     | `torch.save(model.state_dict())` — model weights                                  |
| `optimizer_state.pth` | Optimizer + scheduler state dicts                                                 |
| `training_state.json` | `{"epoch": N, "best_metric": X, "history": {...}}`                                |
| `config.json`         | `{"model_source": "...", "task": "...", "engine": "..."}` — used by `mata.load()` |

### Low-Level Checkpoint API

```python
from mata.training.checkpoint import CheckpointManager

ckpt = CheckpointManager()

# Save
ckpt.save(model, optimizer, scheduler, epoch=5, metrics=metrics,
          config=config, path="runs/train/detect/epoch_005")

# Load
state = ckpt.load("runs/train/detect/epoch_005")
model.load_state_dict(state["model_state"])
optimizer.load_state_dict(state["optimizer_state"])
start_epoch = state["training_state"]["epoch"]

# Export for inference (removes optimizer state, prepares for deployment)
ckpt.export_for_inference(
    model=model,
    config=config,
    output_dir="runs/export/detect_v1",
)

# List all checkpoints in a run directory
checkpoints = ckpt.list_checkpoints("runs/train/detect")
```

---

## 9. Evaluation Integration

`mata.val()` is called automatically during training when `val_every > 0` and a validation dataset is available.

### During Training

```python
result = mata.train(
    "detect",
    model="facebook/detr-resnet-50",
    data="coco.yaml",
    val_every=2,       # run mata.val() every 2 epochs
    patience=5,        # stop early if mAP50 doesn't improve for 5 checks
)

# View validation metrics history
print(result.history.get("val_map50"))
```

### After Training

```python
import mata

# Evaluate the best checkpoint
metrics = mata.val(
    "detect",
    model=result.best_checkpoint,   # or any checkpoint path
    data="examples/configs/coco.yaml",
    split="val",
    verbose=True,
)

print(f"mAP50:   {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### Early Stopping

Set `patience > 0` to stop training when the primary validation metric doesn't improve:

```python
result = mata.train(
    "classify",
    model="microsoft/resnet-50",
    data="data/flowers/train",
    val_data="data/flowers/val",
    patience=5,       # stop after 5 epochs without top-1 accuracy improvement
    val_every=1,
)
print(f"Stopped at epoch {result.epochs_completed}")
```

Primary metric tracked per task:

- **detect** → `map50`
- **classify** → `top1` accuracy
- **segment** → `map50`

### ValidationCallback

For advanced use, you can use `ValidationCallback` directly in custom training loops:

```python
from mata.training.callbacks import ValidationCallback

callback = ValidationCallback(
    task="detect",
    val_data="coco.yaml",
    val_every=2,
    verbose=False,
)

for epoch in range(epochs):
    # ... training step ...
    metrics = callback.on_epoch_end(epoch, model=model)
    if metrics:
        print(f"Epoch {epoch}: mAP50={metrics.get('map50', 0):.3f}")
```

---

## 10. Reloading Trained Models

Trained checkpoints are reloadable via the standard `mata.load()` API. MATA detects checkpoint directories automatically.

### Loading a Checkpoint

```python
import mata

# mata.load() auto-detects that this is a trained checkpoint
detector = mata.load("detect", "runs/train/detect/best")

# Run inference exactly as with a pre-trained model
result = mata.run("detect", "test.jpg", model=detector)
print(result)
```

### Checkpoint Detection Logic

`mata.load()` recognizes a directory as a MATA checkpoint if it contains:

- `config.json` (metadata) **AND**
- `model_state.pth` (torchvision) **OR** `model.safetensors` (HuggingFace)

HuggingFace checkpoints are loaded via `from_pretrained(checkpoint_dir)`. Torchvision checkpoints use `torch.load(weights_only=True)` for security.

### Full Round-Trip Example

```python
import mata

# 1. Fine-tune
result = mata.finetune(
    "detect",
    model="facebook/detr-resnet-50",
    data="data/custom_coco.yaml",
    epochs=10,
    save_dir="runs/train",
)

# 2. Evaluate
metrics = mata.val("detect", model=result.best_checkpoint,
                   data="data/custom_coco.yaml", split="val")
print(f"Best mAP50: {metrics.box.map50:.3f}")

# 3. Reload for inference
detector = mata.load("detect", result.best_checkpoint)

# 4. Predict
prediction = mata.run("detect", "new_image.jpg", model=detector, threshold=0.5)
prediction.save("output/detections.jpg")
```

---

## 11. HuggingFace vs Torchvision Training

### When to Use Each

|                          | HuggingFace (`transformers.Trainer`)       | Torchvision (custom loop)        |
| ------------------------ | ------------------------------------------ | -------------------------------- |
| **Model sources**        | `facebook/`, `microsoft/`, `google/`, etc. | `torchvision/fasterrcnn_*`, etc. |
| **Tasks**                | detect, classify, segment                  | detect only                      |
| **Backbone freezing**    | ✅ via `model.backbone`                    | ✅ via `model.backbone`          |
| **AMP**                  | ✅ via `TrainingArguments.fp16`            | ✅ via `torch.amp.autocast`      |
| **Checkpointing**        | HF Trainer native + MATA wrapper           | MATA `CheckpointManager`         |
| **Distributed training** | Future (v2.1)                              | Future (v2.1)                    |

### Engine Detection

The training engine is detected automatically using the same strategy as `mata.load()`:

```
source.startswith("torchvision/")      → TorchTrainingEngine
"/" in source (org/model pattern)      → HFTrainingEngine
Config alias                           → resolved via ModelRegistry, then re-detected
Local checkpoint directory             → engine read from config.json
```

```python
from mata.training.trainer import TrainingOrchestrator
from mata.training.config import TrainingConfig

config = TrainingConfig(task="detect", model="facebook/detr-resnet-50", data="coco.yaml")
orchestrator = TrainingOrchestrator(config)
print(orchestrator._detect_engine("facebook/detr-resnet-50"))   # → "huggingface"
print(orchestrator._detect_engine("torchvision/fasterrcnn_resnet50_fpn"))  # → "torchvision"
```

### HuggingFace Training Details

- Uses `transformers.Trainer` for the training loop
- `AutoModelForObjectDetection` / `AutoModelForImageClassification` / `Mask2FormerForUniversalSegmentation`
- Evaluation via `compute_metrics` callback that delegates to `mata.val()`
- Checkpoints saved in HuggingFace format (`model.safetensors` + `config.json`)

### Torchvision Training Details

- Custom PyTorch training loop (torchvision models return a loss dict in `.train()` mode)
- AMP via `torch.amp.autocast("cuda")` + `GradScaler` (silently skipped on CPU)
- Supports all 7 model families with automatic head replacement for custom class counts
- Checkpoints saved as `model_state.pth` (PyTorch state dict)

---

## 12. Troubleshooting & FAQ

### Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. Reduce `batch_size`: try `batch_size=2` or `batch_size=1`
2. Reduce image size: `augment_config={"size": 480}` (default 640 for detection)
3. Enable AMP: ensure `amp=True` (default)
4. Freeze backbone: `freeze_backbone=True` reduces gradient memory
5. Reduce `num_workers`: workers pre-load batches; try `num_workers=0`

### Training Loss Not Decreasing

**Possible causes:**

- Learning rate too high: try `lr=1e-5` or `lr=1e-6`
- Incorrect dataset labels: verify boxes are `xyxy` format and labels are 0-indexed
- Bad augmentation: disable augmentation with `augment=False` as a diagnostic step
- No warmup: add `warmup_epochs=2` for transformer models

### `ConfigurationError` on `mata.train()`

`TrainingConfig.validate()` provides actionable error messages. Common cases:

```
Invalid task 'detection'. Must be one of: classify, detect, segment.
Invalid optimizer 'adam-w'. Must be one of: adam, adamw, sgd.
warmup_epochs (10) must be less than epochs (5).
resume path 'runs/train/old_run' does not exist.
```

### Validation Metrics Are Always 0

- Ensure `val_data` is set or the `data` YAML contains a `val:` split key
- Ensure `val_every=1` (not `0` — which would never trigger validation)
- Check that the validation split has annotations

### `ImportError: transformers is required`

```bash
pip install transformers torch
```

### `ImportError: albumentations`

```bash
pip install "datamata[training]"
# or
pip install albumentations
```

### `ImportError: matplotlib` for `plot_loss()`

```bash
pip install matplotlib
```

### Resuming Fails with `TrainingError: missing model_state.pth`

The resume path must point to a **checkpoint directory** (containing `model_state.pth`), not the run root. Use `result.last_checkpoint` or `result.best_checkpoint` which are already pointing to the correct sub-directory.

```python
# ✅ Correct
result2 = mata.train("detect", ..., resume=result.last_checkpoint)

# ❌ Wrong — this is the run root, not a checkpoint
result2 = mata.train("detect", ..., resume="runs/train/detect")
```

### Checkpoint Not Found After Training

If training terminates unexpectedly (e.g., keyboard interrupt), the `last` checkpoint may not be written. Use the most recent periodic checkpoint from `save_every`:

```python
from mata.training.checkpoint import CheckpointManager

checkpoints = CheckpointManager().list_checkpoints("runs/train/detect")
print("Available checkpoints:", checkpoints)
latest = checkpoints[-1]  # sorted list
detector = mata.load("detect", str(latest))
```

### Class Count Mismatch

When using `torchvision/` models, the detection head is automatically replaced to match the number of classes in your dataset. If you see shape mismatch errors, ensure your annotation labels are **0-indexed** (class 0 is the first class, not background — MATA handles the +1 background offset internally).

### Slow Training on CPU

Training on CPU is supported but slow for large models. For CPU-only environments:

- Use small models: `torchvision/ssdlite320_mobilenet_v3_large`
- Reduce to `batch_size=1` and `num_workers=0`
- Use `amp=False` (AMP has no effect on CPU)
- Consider exporting to ONNX for CPU inference after training on a GPU machine

---

## Example Scripts

See the `examples/train/` directory for complete, runnable scripts:

| Script                                                                     | Description                                                      |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [finetune_detection.py](../examples/train/finetune_detection.py)           | Fine-tune DETR on COCO-format data → evaluate → reload → predict |
| [finetune_classification.py](../examples/train/finetune_classification.py) | Fine-tune ResNet-50 on ImageFolder → evaluate → export →         |

## Related Documentation

- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) — full `mata.val()` reference
- [examples/configs/training_detect.yaml](../examples/configs/training_detect.yaml) — detection config reference
- [examples/configs/training_classify.yaml](../examples/configs/training_classify.yaml) — classification config reference
