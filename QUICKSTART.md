---
title: "Quickstart"
description: "Run your first MATA tasks with one-shot inference and a simple graph pipeline."
sidebar_position: 3
---

# MATA Quick Start Guide

This guide will get you up and running with MATA in 5 minutes.

## Installation

```bash
pip install datamata
```

For GPU acceleration (requires NVIDIA GPU + CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datamata
```

> **From source** (contributors only): `git clone https://github.com/datamata-io/mata.git && cd mata && pip install -e .`

See [INSTALLATION.md](INSTALLATION.md) for all options, optional extras, and troubleshooting.

## Verify Installation

```bash
python verify_install.py  # Shows GPU/CPU status and runs test detection
```

## Command-Line Interface

Once installed, the `mata` CLI is available immediately:

```bash
# Run a task directly — no Python required
mata run detect image.jpg --model facebook/detr-resnet-50 --conf 0.4
mata run classify image.jpg --model microsoft/resnet-50 --json
mata run vlm image.jpg --model Qwen/Qwen3-VL-2B-Instruct --prompt "Describe this"

# Track objects in a video
mata track video.mp4 --model facebook/detr-resnet-50 --save

# Evaluate a model
mata val detect --data coco.yaml --model facebook/detr-resnet-50

# Show version
mata --version
```

See [CLI Examples](examples/cli/) for shell and PowerShell scripts covering every subcommand, and [QUICK_REFERENCE.md](QUICK_REFERENCE.md#-cli-quick-reference-v195) for the full flags reference.

Or check programmatically:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## First Detection

Create a file `test_detect.py`:

```python
import mata

# One-shot detection (simplest)
result = mata.run("detect", "path/to/your/image.jpg",
                  model="facebook/detr-resnet-50")

# Print results
print(f"Found {len(result.instances)} objects:")
for inst in result.instances:
    print(f"  - {inst.label_name}: {inst.score:.2%}")

# Get JSON output
print(result.to_json(indent=2))
```

Run it:

```bash
python test_detect.py
```

## First Depth Estimation

Create a file `test_depth.py`:

```python
import mata

result = mata.run(
    "depth",
    "path/to/your/image.jpg",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    normalize=True,
)

# Save depth visualization
result.save("depth_output.png", colormap="magma")
```

Run it:

```bash
python test_depth.py
```

## Try Different Models

```python
import mata

# List available models from HuggingFace Hub
models = mata.list_models("detect")
for model in models[:3]:
    print(f"{model['id']} ({model['downloads']} downloads)")

# Use a different model
result = mata.run("detect", "image.jpg",
                  model="IDEA-Research/grounding-dino-tiny", threshold=0.6)

# Or load adapter for repeated use
detector = mata.load("detect", "facebook/detr-resnet-50")
result1 = detector.predict("image1.jpg")
result2 = detector.predict("image2.jpg")
```

## Common Parameters

```python
# Adjust detection threshold
result = mata.run("detect", "image.jpg",
                  model="facebook/detr-resnet-50", threshold=0.7)

# Force CPU (default is auto)
detector = mata.load("detect", "facebook/detr-resnet-50", device="cpu")

# Use a larger model variant
detector = mata.load(
    "detect",
    "PekingU/rtdetr_v2_r50vd",
    threshold=0.5
)
```

## Working with Results

```python
result = mata.run("detect", "image.jpg", model="facebook/detr-resnet-50")

# Access individual detections
for inst in result.instances:
    x1, y1, x2, y2 = inst.bbox  # xyxy format
    label = inst.label           # integer label
    label_name = inst.label_name # human-readable name (if available)
    score = inst.score           # confidence [0.0, 1.0]

    print(f"Object: {label_name} ({score:.2%})")
    print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

# Serialize to JSON
json_str = result.to_json(indent=2)
with open("results.json", "w") as f:
    f.write(json_str)

# Deserialize from JSON
from mata import DetectResult
loaded_result = DetectResult.from_json(json_str)
```

## Performance Best Practices

### GPU vs CPU Selection

```python
# Option 1: Auto-detection (recommended)
detector = mata.load("detect", "facebook/detr-resnet-50", device="auto")
# Uses GPU if available, falls back to CPU

# Option 2: Explicit GPU
detector = mata.load("detect", "facebook/detr-resnet-50", device="cuda")
# Requires CUDA-capable GPU

# Option 3: Explicit CPU
detector = mata.load("detect", "facebook/detr-resnet-50", device="cpu")
# Useful for testing or non-GPU environments
```

### Device Verification

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available, using CPU")

# Verify model is on correct device
detector = mata.load("detect", "facebook/detr-resnet-50", device="auto")
print(f"Model device: {detector.device}")
```

### TorchScript Models (Optimized)

TorchScript models offer faster inference through pre-traced computation graphs:

```python
# Load TorchScript model (no config needed)
detector = mata.load(
    "detect",
    "examples/models/torchscript/rtv4_l.pt",
    device="cuda",  # Best performance on GPU
    input_size=640,
    threshold=0.5
)

# Benefits:
# ✓ Faster inference (pre-optimized)
# ✓ No architecture reconstruction
# ✓ Smaller memory footprint
# ✓ Better for production deployment
```

### Performance Tips

**GPU Optimization:**

- Use `device="cuda"` for batch inference
- TorchScript models leverage GPU acceleration better
- Keep model on GPU for repeated predictions
- Use larger batch sizes when possible

**CPU Optimization:**

- Use smaller models (rtv4_s.pt vs rtv4_x.pt)
- Reduce input_size (480 vs 640) for faster processing
- Consider ONNX models for CPU deployment
- Use threading for parallel image processing

**Memory Management:**

```python
import torch

# Clear GPU cache between large batches
detector = mata.load("detect", "facebook/detr-resnet-50", device="cuda")
result = detector.predict("large_image.jpg")
torch.cuda.empty_cache()  # Free unused GPU memory
```

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mata --cov-report=html
open htmlcov/index.html

# Run specific test file
pytest tests/test_api.py -v
```

## Run Examples

```bash
# Basic detection examples
python examples/detect/basic_detection.py
```

## Configuration

Create a config file `mata_config.json`:

```json
{
  "default_device": "cuda",
  "default_models": {
    "detect": "facebook/detr-resnet-50"
  },
  "log_level": "INFO"
}
```

Use it:

```python
from mata import MATAConfig, set_config

# Load from file
config = MATAConfig.from_file("mata_config.json")
set_config(config)

# Or set via code
config = MATAConfig(
    default_device="cpu",
    default_models={"detect": "facebook/detr-resnet-50"},
    log_level="DEBUG"
)
set_config(config)

# Now mata.load() will use your defaults
detector = mata.load("detect")  # Uses config defaults
```

## Troubleshooting

### No Detections Found (0 objects)

**This is the most common issue!** RT-DETR models are sensitive to the threshold parameter.

```python
# Default threshold (0.3) might be too high for your image
result = mata.run("detect", "image.jpg",
                  model="facebook/detr-resnet-50", threshold=0.3)  # 0 detections

# Try lowering it
result = mata.run("detect", "image.jpg",
                  model="facebook/detr-resnet-50", threshold=0.2)  # May find objects now
```

**Debug with the debug script**:

```bash
python verify_install.py
```

**More details**: See [Common Issues](README.md#common-issues)

### Import Error

```
ModuleNotFoundError: No module named 'mata'
```

**Solution**: Install in editable mode: `pip install -e .`

### Transformers Not Found

```
ImportError: transformers is required for RT-DETR adapter
```

**Solution**: Install dependencies: `pip install transformers torch pillow`

### CUDA Out of Memory

```python
# Use CPU instead
detector = mata.load("detect", "facebook/detr-resnet-50", device="cpu")

# Or use a smaller model
detector = mata.load("detect", "PekingU/rtdetr_r18vd")
```

### Model Not Found

```
ModelNotFoundError: Model 'my-model' not found for task 'detect'
```

**Solution**: Check your model ID or config alias. Run `python verify_install.py` to diagnose.

## Evaluate Your Model

After running inference, measure your model's accuracy against a labeled dataset with `mata.val()`:

```python
import mata

metrics = mata.val(
    "detect",
    model="facebook/detr-resnet-50",
    data="examples/configs/coco.yaml",
    verbose=True,            # print per-class table
    plots=True,              # save PR/F1 curve PNGs
    save_dir="runs/val/detect",
)
print(f"mAP@50:    {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
```

All four tasks are supported — detection, segmentation, classification, and depth.
See the [Validation Guide](docs/VALIDATION_GUIDE.md) for dataset setup, full API reference, and metrics details.

## Valkey / Redis Result Storage

Persist any MATA result to [Valkey](https://valkey.io/) or Redis for distributed pipelines, cross-process sharing, or caching.

### Install

```bash
pip install datamata[valkey]   # valkey-py (recommended)
pip install datamata[redis]    # redis-py (alternative)
```

### Save and load a result

```bash
# Start a local Valkey server (or use an existing Redis server)
docker run -d -p 6379:6379 valkey/valkey:latest
```

```python
import mata

# Run detection and save result to Valkey
result = mata.run("detect", "image.jpg", model="PekingU/rtdetr_r18vd")
result.save("valkey://localhost:6379/detections:frame_001")

# Load it back later (in a different process or service)
from mata.core.exporters import load_valkey
loaded = load_valkey(url="valkey://localhost:6379", key="detections:frame_001")
print(loaded)  # equivalent VisionResult
```

### Use in a graph pipeline with `ValkeyStore` / `ValkeyLoad`

```python
import mata
from mata.nodes import Detect, Filter, ValkeyStore, ValkeyLoad, Fuse
from mata.core.graph import Graph

detector = mata.load("detect", "PekingU/rtdetr_r18vd")

# Pipeline A — detect and persist
store_graph = (
    Graph()
    .then(Detect(using="detr", out="dets"))
    .then(Filter(src="dets", score_gt=0.4, out="filtered"))
    .then(ValkeyStore(
        src="filtered",
        url="valkey://localhost:6379",
        key="pipeline:detections:{timestamp}",
        ttl=60,  # expires after 60 s
    ))
)
mata.infer("frame.jpg", graph=store_graph, providers={"detr": detector})

# Pipeline B — load and annotate (in a separate service)
load_graph = (
    Graph()
    .then(ValkeyLoad(
        url="valkey://localhost:6379",
        key="pipeline:detections:latest",
        out="dets",
    ))
    .then(Fuse(detections="dets", out="annotated"))
)
result = mata.infer("frame.jpg", graph=load_graph, providers={})
```

See the [Graph API Reference](docs/GRAPH_API_REFERENCE.md#storage-nodes) for full parameter documentation.

## Notebook Display

MATA results render automatically in Jupyter notebooks:

```python
import mata

# Evaluate in a cell — rich HTML table renders inline
result = mata.run("detect", "image.jpg", model="facebook/detr-resnet-50")
result

# Explicit display with image overlay
mata.show(result, image="image.jpg")
```

Install notebook extras:

```bash
pip install datamata[notebook]
```

See [`examples/notebooks/`](examples/notebooks/) for ready-to-run starter notebooks.

## Gallery Matching / Recognition

Build a gallery of known identities, then match query images against it with cosine similarity:

```python
import mata

# Build once
gallery = mata.Gallery(similarity_thresh=0.7)

# Embed photos, then add them to the gallery
alice_emb = mata.run("embed", "alice_photo.jpg", model="openai/clip-vit-base-patch32")
gallery.add("alice", alice_emb.embedding)

bob_emb = mata.run("embed", "bob_photo.jpg", model="openai/clip-vit-base-patch32")
gallery.add("bob", bob_emb.embedding)

gallery.save("gallery.npz")

# Recognise later
gallery = mata.Gallery.load("gallery.npz")
matches = mata.run("recognize", "query.jpg",
    gallery=gallery,
    model="openai/clip-vit-base-patch32",
    top_k=3)

print(matches.entries[0].label)       # best match
print(matches.entries[0].similarity)  # cosine similarity score

# Or use CLI:
# mata recognize query.jpg --gallery gallery.npz --top-k 3
```

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md#-gallery--recognition-quick-reference-v195) for the full API and graph pipeline pattern.

1. **Read the full documentation**: [README.md](README.md)
2. **Understand the architecture**: [docs/MATA_DESIGN_AND_ARCHITECTURE.md](docs/MATA_DESIGN_AND_ARCHITECTURE.md)
3. **Explore examples**: Check `examples/` directory and [examples/README.md](examples/README.md)

## Getting Help

- Check error messages - they include troubleshooting guidance
- Run `python verify_install.py` to diagnose issues
- Read docstrings: `help(mata.load)`, `help(mata.run)`
- Review test files in `tests/` for usage patterns

Happy detecting! 🎯
