# MATA Installation Guide

This guide covers installing MATA with GPU or CPU support.

## Prerequisites

- Python 3.10 or later
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

## Quick Install

### Option 1: CPU-Only (Recommended for testing)

Best for: Laptops, systems without NVIDIA GPU, or when you don't need maximum speed.

```bash
# Clone repository
git clone https://github.com/datamata-io/mata.git
cd MATA

# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install MATA
pip install -e .
```

**Pros**: Smaller download, works on any system  
**Cons**: Slower inference (~500-800ms per image)

### Option 2: GPU (Recommended for production)

Best for: Systems with NVIDIA GPU, production deployments, high-throughput processing.

```bash
# Clone repository
git clone https://github.com/datamata-io/mata.git
cd MATA

# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install MATA
pip install -e .
```

**Pros**: 5-10x faster inference (~50-100ms per image)  
**Cons**: Larger download (~2GB), requires NVIDIA GPU

## Choosing the Right PyTorch Version

### Step 1: Check Your CUDA Version

Run this command to see your CUDA version:

```bash
nvidia-smi
```

Look for the CUDA version in the output (e.g., "CUDA Version: 12.1").

If you get an error, you don't have an NVIDIA GPU or CUDA drivers installed - use CPU version.

### Step 2: Install Matching PyTorch

| CUDA Version      | PyTorch Installation Command                                                       |
| ----------------- | ---------------------------------------------------------------------------------- |
| 11.8              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 12.1              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 12.4              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.6              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| 13.0              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| No GPU / CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`   |

**Note**: PyTorch supports older CUDA versions than what you have. For example, if you have CUDA 12.4, you can use `cu121` (12.1) PyTorch - it will still work.

See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the latest versions.

## Switching from CPU to GPU (or vice versa)

### Already installed CPU version and want GPU?

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision

# Install GPU version (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MATA doesn't need reinstalling
```

### Already have GPU but want CPU (for testing/compatibility)?

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision

# Install CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# MATA doesn't need reinstalling
```

## Verifying Your Installation

### Check if GPU is available

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
```

### Test MATA with device detection

```bash
python verify_install.py
```

This script will:

- Show your PyTorch version
- Detect if CUDA/GPU is available
- Show which device is being used
- Provide installation instructions if GPU not found

### Quick detection test

```python
import mata

# MATA will auto-detect and use GPU if available
detector = mata.load("detect", "rtdetr")  # device="auto" is default

# Force specific device
detector_gpu = mata.load("detect", "rtdetr", device="cuda")  # GPU only
detector_cpu = mata.load("detect", "rtdetr", device="cpu")   # CPU only

# Run detection
result = detector.predict("image.jpg")
print(f"Found {len(result.detections)} objects")
```

## Troubleshooting

### "CUDA out of memory" error

**Solution 1**: Use CPU instead

```python
detector = mata.load("detect", "rtdetr", device="cpu")
```

**Solution 2**: Use smaller model

```python
# RT-DETRv2-R18 is the smallest (default)
detector = mata.load("detect", "rtdetr", model_id="facebook/detr-resnet-50")
```

**Solution 3**: Close other GPU applications

### "torch.cuda.is_available() returns False"

Possible causes:

1. **No NVIDIA GPU**: Check with `nvidia-smi`
2. **No CUDA drivers**: Install from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
3. **Wrong PyTorch version**: Reinstall with CUDA support (see above)
4. **Incompatible CUDA version**: Check `nvidia-smi` CUDA version matches PyTorch

### GPU slower than expected

1. **First run is always slow**: Model downloads and compiles
2. **Check GPU utilization**: Run `nvidia-smi` while inference running
3. **Verify GPU is actually being used**:
   ```python
   import mata
   detector = mata.load("detect", "rtdetr", device="cuda")
   # Check internal model device
   print(detector.device)  # Should show "cuda"
   ```

## Development Installation

If you want to contribute or modify MATA:

```bash
# Install with dev dependencies (includes pytest, black, etc.)
pip install -e ".[dev]"
```

## Virtual Environment (Recommended)

Using a virtual environment prevents dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Now install as normal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Package Sizes

For planning your installation:

| Component       | CPU Version                      | GPU Version (CUDA 12.1) |
| --------------- | -------------------------------- | ----------------------- |
| PyTorch         | ~150 MB                          | ~2.5 GB                 |
| Transformers    | ~10 MB                           | ~10 MB                  |
| RT-DETRv2 model | ~50 MB (downloaded on first run) | Same                    |
| Total           | ~200 MB + models                 | ~2.5 GB + models        |

## Next Steps

After installation:

1. Run `python verify_install.py` to verify GPU/CPU detection
2. Try `python examples/detect/basic_detection.py` for a quick test
3. Read [QUICKSTART.md](QUICKSTART.md) for API usage
4. Check the [Common Issues](README.md#common-issues) section if you encounter issues
