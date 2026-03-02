# MATA Quick Start Guide

This guide will get you up and running with MATA in 5 minutes.

## Installation

### Quick Install (CPU)

```bash
cd MATA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### GPU Installation (Faster, requires NVIDIA GPU)

```bash
cd MATA
# Check your CUDA version with: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

**See [INSTALLATION.md](INSTALLATION.md) for detailed GPU/CPU installation instructions and troubleshooting.**

## Verify Installation

```bash
python verify_install.py  # Shows GPU/CPU status and runs test detection
```

Or check programmatically:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## First Detection

Create a file `test_detect.py`:

```python
import mata

# Option 1: One-shot detection (simplest)
result = mata.run("detect", "path/to/your/image.jpg")

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

# List available models
print(mata.list_models("detect"))
# Output: ['rtdetr', 'dino', 'conditional_detr']

# List depth models
print(mata.list_models("depth"))

# Use DINO instead of default RT-DETR
result = mata.run("detect", "image.jpg", model="dino", threshold=0.6)

# Or load adapter for repeated use
detector = mata.load("detect", "dino")
result1 = detector.predict("image1.jpg")
result2 = detector.predict("image2.jpg")
```

## Common Parameters

```python
# Adjust detection threshold
result = mata.run("detect", "image.jpg", threshold=0.7)

# Force CPU (default is auto)
detector = mata.load("detect", "rtdetr", device="cpu")

# Use different model variant
detector = mata.load(
    "detect",
    "rtdetr",
    model_id="PekingU/rtdetr_v2_r50vd",  # Larger model
    threshold=0.5
)
```

## Working with Results

```python
result = mata.run("detect", "image.jpg")

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
detector = mata.load("detect", "rtdetr", device="auto")
# Uses GPU if available, falls back to CPU

# Option 2: Explicit GPU
detector = mata.load("detect", "rtdetr", device="cuda")
# Requires CUDA-capable GPU

# Option 3: Explicit CPU
detector = mata.load("detect", "rtdetr", device="cpu")
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
detector = mata.load("detect", "rtdetr", device="auto")
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
detector = mata.load("detect", "rtdetr", device="cuda")
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
    "detect": "dino"
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
    default_models={"detect": "rtdetr"},
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
result = mata.run("detect", "image.jpg", threshold=0.3)  # 0 detections

# Try lowering it
result = mata.run("detect", "image.jpg", threshold=0.2)  # May find objects now
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
detector = mata.load("detect", device="cpu")

# Or use smaller model
detector = mata.load("detect", "rtdetr", model_id="facebook/detr-resnet-50")
```

### Plugin Not Discovered

```
PluginNotFoundError: Plugin 'rtdetr' not found for task 'detect'
```

**Solution**: Run `python verify_install.py` to check plugin discovery

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

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Understand the architecture**: [MATA_architecture_and_code_structure.md](MATA_architecture_and_code_structure.md)
3. **See implementation details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
4. **Explore examples**: Check `examples/` directory and [examples/README.md](examples/README.md)
5. **Write your own plugin**: See README.md "Plugin Development" section

## Getting Help

- Check error messages - they include troubleshooting guidance
- Run `python verify_install.py` to diagnose issues
- Read docstrings: `help(mata.load)`, `help(mata.run)`
- Review test files in `tests/` for usage patterns

Happy detecting! 🎯
