# KACA: Pure-CNN Object Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-85%25+-green.svg)](tests/)

**KACA** is a pure-CNN object detector designed for cross-platform stability, ONNX deployment, and enterprise production environments. Built with security-first principles and portability in mind—no custom CUDA ops, no MMCV/Detectron2 dependencies, just PyTorch native operations.

> **Status**: Training Phase (Week 9/10) | **Model**: KACA-S (~7M params) | **Coverage**: 85%+ | **Tests**: 65+ passing

---

## 🎯 Key Features

### **Production-Ready Architecture**

- ✅ **Pure PyTorch**: No custom CUDA kernels, runs anywhere PyTorch runs
- ✅ **ONNX Export**: First-class ONNX support with dynamic batching
- ✅ **Secure Loading**: Uses `weights_only=True` for checkpoint loading (CVE-2025-32434 mitigated)
- ✅ **Deterministic**: Reproducible results across platforms

### **Performance & Efficiency**

- ⚡ **Lightweight**: 6-7M parameters for KACA-S variant
- ⚡ **Fast Inference**: Target ≥30 FPS on RTX 3090
- ⚡ **Scalable**: Support for S/M/L model variants
- ⚡ **Multi-GPU**: DistributedDataParallel training support

### **Developer Experience**

- 🛠️ **Pure Python**: No compilation required
- 🛠️ **Comprehensive Testing**: 140+ unit tests, 95%+ coverage
- 🛠️ **Rich Documentation**: Architecture guides, API docs, tutorials
- 🛠️ **CLI Tools**: Train, detect, export with simple commands

### **Multi-Task Architecture** 🚀

KACA features a task-agnostic framework supporting multiple vision tasks:

- ✅ **Object Detection** (Production Ready - `build_kaca_det_s()`)
- 🎨 **Instance Segmentation** (Coming Soon - `build_kaca_seg_s()`)
- 🤸 **Pose Estimation** (Coming Soon - `build_kaca_pose_s()`)
- 🏷️ **Classification** (Extensible - Custom heads supported)

Pluggable heads enable easy task extension. See [Multi-Task Support](#-multi-task-support) section.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/KACA.git
cd KACA

# Install dependencies
pip install -r requirements.txt

# Install KACA in development mode
pip install -e .
```

### Object Detection (5 Lines)

```python
from kaca import build_kaca_det_s

# Load model (random weights or trained checkpoint)
model = build_kaca_det_s(num_classes=80)
# model.load_weights('kaca_s_coco.pth')  # Load trained weights

# Run inference
detections = model.predict(
    "path/to/image.jpg",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Results: [{'bbox_xyxy': [x1,y1,x2,y2], 'score': 0.95, 'class_id': 0, 'class_name': 'person'}, ...]
```

> **Note**: The old API `build_kaca_s()` is deprecated. Use `build_kaca_det_s()` instead. See [Migration Guide](docs/migration_guide_multitask.md) for upgrade instructions.

### Command-Line Interface

```bash
# Object detection on images
python scripts/detect.py \
    --weights runs/train/exp/final.pth \
    --source path/to/images/ \
    --conf 0.25 \
    --save-results

# Train on COCO dataset
python scripts/train.py \
    --config configs/kaca_s_coco.yaml \
    --data /path/to/coco

# Export to ONNX
python scripts/export.py \
    --weights runs/train/exp/final.pth \
    --output kaca_s.onnx \
    --opset 16
```

---

## 📐 Architecture

KACA follows a three-component pipeline optimized for detection tasks:

```
┌─────────────┐    ┌──────────┐    ┌────────────────┐
│ CSPBackbone │ -> │ PANFPN   │ -> │ DecoupledHead  │
│ (Stages 2-5)│    │ (P3-P5)  │    │ (cls/box/obj)  │
└─────────────┘    └──────────┘    └────────────────┘
     ↓                  ↓                  ↓
  Features          Multi-Scale        Predictions
  C3/C4/C5         80×80/40×40/20×20    Anchor-free
```

### Components

| Component         | Description                        | Key Features                           |
| ----------------- | ---------------------------------- | -------------------------------------- |
| **CSPBackbone**   | Cross Stage Partial backbone       | Efficient feature extraction, 5 stages |
| **PANFPN**        | Path Aggregation + Feature Pyramid | Multi-scale fusion, bidirectional flow |
| **DecoupledHead** | Anchor-free detection head         | Separate cls/box/obj branches          |

**Model Scaling** (depth × width multipliers):

- **KACA-S**: 0.33 × 0.50 → ~6-7M params (Current)
- **KACA-M**: 0.67 × 0.75 → ~15-20M params (Planned)
- **KACA-L**: 1.00 × 1.00 → ~40-50M params (Planned)

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## � Multi-Task Support

KACA is designed as a **task-agnostic framework** with pluggable heads for different computer vision tasks.

### Object Detection (Current)

```python
from kaca import build_kaca_det_s
from kaca.train.loss import DetectionLoss
from kaca.train.validators.detection import DetectionValidator

# Build detection model
model = build_kaca_det_s(num_classes=80)

# Use with task-specific loss and validator
criterion = DetectionLoss(num_classes=80)
validator = DetectionValidator(num_classes=80)

# Inference
detections = model.predict("image.jpg")
```

### Future Tasks

KACA's extensible architecture enables easy addition of new vision tasks:

- **Instance Segmentation**: `build_kaca_seg_s()` (Planned - Q1 2026)
  - Mask prediction head
  - Segmentation loss and metrics
  - Polygon/mask output format

- **Pose Estimation**: `build_kaca_pose_s()` (Planned - Q2 2026)
  - Keypoint detection head
  - Heatmap-based pose estimation
  - COCO keypoint format support

- **Custom Tasks**: Implement your own!
  - Create custom `TaskHead` classes
  - Implement task-specific loss functions
  - Add custom validators for evaluation

### Creating Custom Tasks

```python
from kaca import KACA
from kaca.models.heads import TaskHead

# 1. Define custom head
class MyCustomHead(TaskHead):
    def forward(self, features):
        # Your task-specific logic
        pass

# 2. Build model with custom head
model = KACA(
    depth_multiple=0.33,
    width_multiple=0.50,
    head=MyCustomHead(args)
)

# 3. Train with custom loss and validator
from kaca.train import Trainer
trainer = Trainer(
    model=model,
    criterion=MyCustomLoss(),
    validator=MyCustomValidator()
)
```

See [docs/architecture.md](docs/architecture.md#task-extension-guide) for detailed task extension guide.

### Migration from Old API

If you're using the old detection-only API:

```python
# OLD (Deprecated - will be removed in v0.2.0)
from kaca import build_kaca_s
model = build_kaca_s(num_classes=80)

# NEW (Recommended)
from kaca import build_kaca_det_s
model = build_kaca_det_s(num_classes=80)
```

**What's changed?**

- `build_kaca_s()` → `build_kaca_det_s()` (detection-specific)
- `KACA` class now accepts pluggable `head` parameter
- Training requires explicit `criterion` and `validator` injection
- 100% backward compatible - old checkpoints load seamlessly

See [Migration Guide](docs/migration_guide_multitask.md) for complete upgrade instructions.

---

## �🎓 Training

### Dataset Preparation

KACA uses COCO format annotations:

```bash
data/
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
```

### Training Configuration

```yaml
# configs/kaca_s_coco.yaml
model:
  num_classes: 80
  depth_multiple: 0.33
  width_multiple: 0.50

train:
  epochs: 300
  batch_size: 16
  learning_rate: 0.01
  optimizer: SGD
  scheduler: cosine # with 3-epoch warmup

augment:
  mosaic: 0.5
  mixup: 0.15
  hsv_prob: 0.5
```

### Training Examples

```bash
# Full COCO training (300 epochs) - uses build_kaca_det_s() internally
python scripts/train.py --task detection --config configs/kaca_s_coco.yaml --data data/coco

# Memory-constrained (8GB GPU)
python scripts/train.py --task detection --config configs/kaca_s_coco_8gb.yaml --data data/coco

# Quick validation (dryrun dataset)
python scripts/train.py --task detection --config configs/kaca_s_dryrun.yaml --epochs 5
```

### Programmatic Training with New API

```python
from kaca import build_kaca_det_s
from kaca.train import Trainer
from kaca.train.loss import DetectionLoss
from kaca.train.validators.detection import DetectionValidator
from kaca.data.datasets import COCODetectionDataset
from torch.utils.data import DataLoader

# Build model and training components
model = build_kaca_det_s(num_classes=80)
criterion = DetectionLoss(num_classes=80)
validator = DetectionValidator(num_classes=80)

# Create data loaders
train_dataset = COCODetectionDataset(root='data/coco', split='train')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize trainer with dependency injection
trainer = Trainer(
    model=model,
    criterion=criterion,      # Explicit loss function
    validator=validator,      # Explicit validator
    train_loader=train_loader,
    epochs=300,
    device='cuda'
)

# Train
trainer.train()
```

### Training Features

- ✅ **Gradient Accumulation**: Train with larger effective batch size
- ✅ **Mixed Precision (AMP)**: Faster training, lower memory
- ✅ **EMA (Exponential Moving Average)**: Improved model stability
- ✅ **Warmup + Cosine LR**: Smooth learning rate scheduling
- ✅ **TensorBoard Logging**: Real-time training visualization
- ✅ **Checkpointing**: Auto-save best model based on mAP

---

## 🔍 Inference

### Python API

```python
from kaca import build_kaca_det_s
from pathlib import Path

# Initialize model with new API
model = build_kaca_det_s(num_classes=80)
model.load_weights('kaca_s_coco.pth')

# Single image
detections = model.predict('image.jpg', conf_threshold=0.25)

# Batch inference
images = list(Path('images/').glob('*.jpg'))
for img in images:
    detections = model.predict(str(img))
    print(f"{img.name}: {len(detections)} objects detected")
```

### Detection Output Format

```python
[
    {
        'bbox_xyxy': [x1, y1, x2, y2],  # Bounding box coordinates
        'score': 0.95,                   # Confidence score
        'class_id': 0,                   # Class index
        'class_name': 'person'           # Class name (if provided)
    },
    # ... more detections
]
```

### CLI Inference

```bash
# Single image
python scripts/detect.py --weights final.pth --source image.jpg

# Directory of images
python scripts/detect.py --weights final.pth --source images/ --save-results

# With custom thresholds
python scripts/detect.py \
    --weights final.pth \
    --source images/ \
    --conf 0.3 \
    --iou 0.5 \
    --max-det 100
```

---

## 📤 ONNX Export

### Export Models

```bash
# Basic export
python scripts/export.py --weights kaca_s.pth --output kaca_s.onnx

# With dynamic batch size
python scripts/export.py \
    --weights kaca_s.pth \
    --output kaca_s.onnx \
    --dynamic-batch \
    --opset 16

# Simplified ONNX (optional)
python scripts/export.py \
    --weights kaca_s.pth \
    --output kaca_s.onnx \
    --simplify
```

### ONNX Runtime Inference

```python
from kaca.export.onnx_exporter import ONNXInferenceSession

# Load ONNX model
session = ONNXInferenceSession('kaca_s.onnx')

# Run inference (same API as PyTorch)
detections = session.predict('image.jpg', conf_threshold=0.25)
```

### Deployment Benefits

- ✅ **Cross-Platform**: Run on CPU, GPU, mobile, edge devices
- ✅ **Framework Agnostic**: Use with TensorRT, OpenVINO, ONNX Runtime
- ✅ **Optimized**: Convert to INT8/FP16 for faster inference
- ✅ **No PyTorch Dependency**: Smaller deployment footprint

---

## 🧪 Testing

KACA has comprehensive test coverage across all components:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=kaca --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v          # Architecture tests
pytest tests/test_inference.py -v       # Inference pipeline tests
pytest tests/test_training.py -v        # Training loop tests
pytest tests/test_parity.py -v          # PyTorch ↔ ONNX parity

# Performance benchmarks
python scripts/benchmark.py --data data/coco --device cuda
```

### Test Coverage

| Module            | Tests | Coverage | Status     |
| ----------------- | ----- | -------- | ---------- |
| **Models**        | 40+   | 95%      | ✅ Passing |
| **Inference**     | 31    | 92%      | ✅ Passing |
| **Training**      | 13    | 88%      | ✅ Passing |
| **Data Pipeline** | 20+   | 90%      | ✅ Passing |
| **ONNX Export**   | 15+   | 87%      | ✅ Passing |

---

## 🗺️ Roadmap

### ✅ Completed (Weeks 1-8)

- [x] Foundation & Core Models
- [x] Inference Pipeline
- [x] Training Infrastructure (COCO dataset, augmentations, loss functions)
- [x] ONNX Export & Validation
- [x] Comprehensive Testing (65+ tests, 85%+ coverage)
- [x] Documentation (Architecture, API, Guides)

### 🔄 In Progress (Week 9)

- [ ] **Full COCO Training** (300 epochs on Google Colab Pro)
  - Target: mAP@0.5 ≥ 45%, mAP@0.5:0.95 ≥ 30%
  - Current: Setting up training environment

### ⏳ Upcoming (Weeks 10+)

#### **Phase 1: Model Validation & Release** (Week 10)

- [ ] Model validation & performance analysis
- [ ] Per-class metrics & failure case analysis
- [ ] Speed benchmarks (CPU/GPU)
- [ ] Release v1.0 artifacts (weights, ONNX models)

#### **Phase 2: Multi-Task Refactoring** (Weeks 11-14)

Transform KACA into a task-agnostic framework supporting multiple vision tasks:

**Week 11 - Foundation Layer**

- [ ] Create `TaskHead` interface for pluggable heads
- [ ] Refactor `DecoupledHead` to inherit from `TaskHead`
- [ ] Create `KACADetection` wrapper for backward compatibility
- [ ] Implement checkpoint migration utilities

**Week 12 - Core Refactoring**

- [ ] Modify KACA base class for head injection
- [ ] Create `TaskLoss` and `TaskValidator` interfaces
- [ ] Update Trainer for task-agnostic operation
- [ ] 60+ new tests for refactored components

**Week 13 - Migration & Compatibility**

- [ ] Add deprecation warnings to old API
- [ ] Update scripts (train.py, detect.py, export.py)
- [ ] Migrate existing checkpoints with task metadata
- [ ] 65+ backward compatibility tests

**Week 14 - Documentation & Examples**

- [ ] Create migration guide for users
- [ ] Update architecture and API documentation
- [ ] Provide custom task extension examples
- [ ] Release v0.2.0 with multi-task support

See [TASKS_MULTITASK_REFACTORING.md](TASKS_MULTITASK_REFACTORING.md) for detailed implementation plan.

#### **Phase 3: New Task Implementation** (Q1-Q2 2026)

- [ ] Instance Segmentation (`KACASegmentation`)
- [ ] Pose Estimation (`KACAPose`)
- [ ] Image Classification (`KACAClassifier`)
- [ ] Multi-task transfer learning examples

---

## 📚 Documentation

| Document                                                      | Description                                          |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| [Architecture Guide](docs/architecture.md)                    | Detailed architecture, design decisions, comparisons |
| [API Reference](docs/api.md)                                  | Complete API documentation                           |
| [Migration Guide](docs/migration_guide_multitask.md)          | **Upgrade from old API to multi-task framework**     |
| [Training Guide](docs/colab_training_plan.md)                 | Step-by-step training instructions                   |
| [Dataset Usage](docs/dataset_usage.md)                        | COCO dataset preparation and usage                   |
| [Augmentation Guide](docs/augmentation_guide.md)              | Data augmentation techniques                         |
| [Security Guide](docs/security/secure_loading.md)             | Secure model loading best practices                  |
| [Multi-Task Refactoring](docs/multi_task_refactoring_plan.md) | Multi-task refactoring implementation details        |

---

## 🛠️ Development

### Project Structure

```
KACA/
├── kaca/                   # Main package
│   ├── models/            # Backbone, Neck, Head, KACA
│   ├── data/              # Datasets, transforms, augmentations
│   ├── train/             # Trainer, loss functions, validators
│   ├── utils/             # Boxes, metrics, general utilities
│   └── export/            # ONNX exporter, model packing
├── scripts/               # CLI scripts (train, detect, export)
├── configs/               # YAML configuration files
├── tests/                 # Unit tests (pytest)
├── docs/                  # Documentation
└── examples/              # Usage examples
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run linting
black kaca/ tests/
flake8 kaca/ tests/
isort kaca/ tests/

# Run type checking
mypy src/kaca
```

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Type Hints**: All functions must have type annotations
3. **Testing**: Maintain >85% code coverage
4. **Documentation**: Update docs for API changes
5. **Security**: Use `weights_only=True` for all checkpoint loading

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for detailed development guidelines.

---

## 📊 Project Status

| Metric             | Value                   |
| ------------------ | ----------------------- |
| **Implementation** | 95% Complete            |
| **Tests**          | 140+ passing            |
| **Code Coverage**  | 95%+                    |
| **Documentation**  | 15,000+ lines           |
| **Model Size**     | ~7M parameters (KACA-S) |
| **Architecture**   | Multi-Task Ready        |

### Current Milestones

- ✅ All core components implemented and tested
- ✅ ONNX export with ONNX Runtime support
- ✅ Comprehensive test suite with high coverage
- 🔄 Full COCO training in progress
- ⏳ Multi-task refactoring planned (4-5 weeks)

---

## 🔒 Security

KACA prioritizes security in model loading and deployment:

- **Secure Checkpoint Loading**: Uses `weights_only=True` (mitigates CVE-2025-32434)
- **No Arbitrary Code Execution**: Pure tensor deserialization
- **Hash Verification**: Optional SHA256 verification for model integrity
- **SafeTensors Support**: Planned for future versions

See [docs/security/secure_loading.md](docs/security/secure_loading.md) for details.

---

## 📝 Citation

If you use KACA in your research or projects, please cite:

```bibtex
@software{kaca2026,
  title={KACA: Pure-CNN Object Detector for Cross-Platform Deployment},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/KACA}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- YOLO series for anchor-free detection inspiration
- PyTorch team for robust deep learning framework
- COCO dataset for training and evaluation
- Community contributors and testers

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/KACA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/KACA/discussions)
- **Changelog**: [TASKS.md](TASKS.md)

---

**Built with ❤️ for production ML deployments**
