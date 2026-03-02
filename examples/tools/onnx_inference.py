"""ONNX Runtime Inference — MATA Framework

Detection and classification with local .onnx model files.
MATA auto-detects ONNX models from the .onnx file extension — no explicit
model_type needed in most cases.

Requirements:
    pip install onnxruntime        # CPU
    pip install onnxruntime-gpu    # GPU (requires CUDA)

Run: python examples/tools/onnx_inference.py
"""

import time
from pathlib import Path

import mata
from mata.core.types import ModelType


# === Section 1: ONNX Detection ===

def onnx_detection(model_path: str, image_path: str):
    """Load a detection model from a local .onnx file and run inference."""
    print("=" * 70)
    print("Section 1: ONNX Detection")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"  ❌ Model not found: {model_path}")
        print("     Export a detection model to ONNX first, then update model_path.")
        return

    if not Path(image_path).exists():
        print(f"  ❌ Image not found: {image_path}")
        return

    # Auto-detected from .onnx extension — no model_type needed
    load_start = time.perf_counter()
    detector = mata.load("detect", model_path, threshold=0.5, device="auto")
    print(f"  ✓ Loaded detector ({time.perf_counter() - load_start:.2f}s): {detector.__class__.__name__}")

    infer_start = time.perf_counter()
    result = detector.predict(image_path)
    print(f"  ✓ Detected {len(result.instances)} objects ({(time.perf_counter() - infer_start)*1000:.1f}ms)")

    for i, inst in enumerate(result.instances[:5], 1):
        print(f"    {i}. {inst.label_name:<20} {inst.score:>6.1%}")
    if len(result.instances) > 5:
        print(f"    ... and {len(result.instances) - 5} more")


# === Section 2: ONNX Classification ===

def onnx_classification(model_path: str, image_path: str):
    """Load a classification model from a local .onnx file and run inference."""
    print("\n" + "=" * 70)
    print("Section 2: ONNX Classification")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"  ❌ Model not found: {model_path}")
        print("     Export a classification model to ONNX first:")
        print("     See Section 4 (export snippet) or use optimum-cli.")
        return

    if not Path(image_path).exists():
        print(f"  ❌ Image not found: {image_path}")
        return

    # Auto-detected from .onnx extension
    classifier = mata.load("classify", model_path, top_k=5, device="auto")
    info = classifier.info()
    print(f"  ✓ Loaded classifier: {info.get('name', 'ONNX')}")
    print(f"    Backend: {info.get('backend', 'ONNX Runtime')}")
    print(f"    Device:  {info.get('device', 'auto')}")

    result = classifier.predict(image_path, top_k=5)

    print("\n  Top-5 predictions:")
    for i, pred in enumerate(result.predictions, 1):
        print(f"    {i}. {pred.label_name or pred.label:<30} {pred.score:.4f}")


# === Section 3: Explicit ModelType ===

def onnx_explicit_type(model_path: str, image_path: str):
    """Use ModelType.ONNX explicitly — useful when the extension is ambiguous."""
    print("\n" + "=" * 70)
    print("Section 3: Explicit ModelType.ONNX")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"  ❌ Model not found: {model_path} (skipping)")
        return

    # Force ONNX backend regardless of filename
    classifier = mata.load(
        "classify",
        model_path,
        model_type=ModelType.ONNX,
        top_k=10,
        device="cpu",
    )
    print(f"  ✓ Loaded with explicit ModelType.ONNX: {classifier.info().get('name')}")

    if Path(image_path).exists():
        result = classifier.predict(image_path)
        top1 = result.get_top1()
        if top1:
            print(f"  ✓ Top prediction: {top1.label_name} ({top1.score:.3f})")


# === Section 4: GPU Selection ===

def onnx_gpu_selection(model_path: str, image_path: str):
    """Select GPU explicitly (requires onnxruntime-gpu)."""
    print("\n" + "=" * 70)
    print("Section 4: GPU Selection")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"  ❌ Model not found: {model_path} (skipping)")
        return

    try:
        classifier = mata.load(
            "classify",
            model_path,
            device="cuda",  # Force CUDA execution provider
            top_k=5,
        )
        print(f"  ✓ Loaded on CUDA: {classifier.info().get('device')}")

        if Path(image_path).exists():
            result = classifier.predict(image_path)
            top1 = result.get_top1()
            if top1:
                print(f"  ✓ Top prediction: {top1.label_name} ({top1.score:.3f})")

    except Exception as e:
        print(f"  ⚠ CUDA not available: {e}")
        print("    Install onnxruntime-gpu and ensure CUDA is configured.")


def main():
    """Run ONNX inference examples."""
    print("\n" + "=" * 70)
    print("MATA — ONNX Runtime Inference")
    print("=" * 70)

    # Check ONNX Runtime is installed
    try:
        import onnxruntime
        print(f"✓ ONNX Runtime {onnxruntime.__version__} detected\n")
    except ImportError:
        print("❌ ONNX Runtime not installed.")
        print("   CPU:  pip install onnxruntime")
        print("   GPU:  pip install onnxruntime-gpu")
        return

    # --- Update these paths to point to your actual model files ---
    detect_model = "examples/models/onnx/rtv4_l.onnx"
    classify_model = "examples/models/onnx/resnet50.onnx"
    image_path = "examples/images/000000039769.jpg"
    # ---------------------------------------------------------------

    onnx_detection(detect_model, image_path)
    onnx_classification(classify_model, image_path)
    onnx_explicit_type(classify_model, image_path)
    onnx_gpu_selection(classify_model, image_path)

    print("\n" + "=" * 70)
    print("Tips:")
    print("  • ONNX models are platform-independent (export once, run anywhere)")
    print("  • Faster than PyTorch for fixed-shape inference")
    print("  • Export from HuggingFace: optimum-cli export onnx --model org/model .")
    print("  • Export from PyTorch:     torch.onnx.export(model, dummy, 'model.onnx')")
    print("=" * 70)


if __name__ == "__main__":
    main()
