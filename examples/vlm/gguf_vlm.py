"""GGUF VLM Example — MATA Framework

Demonstrates loading a quantized GGUF VLM (Vision-Language Model) via
llama-cpp-python for CPU-friendly inference.

Download a GGUF VLM from HuggingFace Hub, for example:
  - Qwen2-VL-7B: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF
    -> Download: Qwen2-VL-7B-Instruct-Q4_K_M.gguf
  - LLaVA-v1.5: https://huggingface.co/mys/ggml_llava-v1.5-7b
    -> Download llava-v1.5-7b-q4.gguf + mmproj-model-f16.gguf (projector)

Requirements:
  pip install datamata[gguf]

Run:
  python examples/vlm/gguf_vlm.py
"""

import sys
from pathlib import Path

import mata

# ---------------------------------------------------------------------------
# Configuration — update these paths to point to your downloaded GGUF files
# ---------------------------------------------------------------------------

MODEL_PATH = "model.gguf"          # Path to your GGUF VLM file
MMPROJ_PATH = None                 # Optional: path to LLaVA projector .gguf file

IMAGE_PATH = str(Path(__file__).parent.parent / "images" / "000000039769.jpg")


def check_prerequisites():
    """Verify model file exists before proceeding."""
    if not Path(MODEL_PATH).exists():
        print(f"Error: GGUF model not found at '{MODEL_PATH}'.")
        print("Download a GGUF VLM file first — see the docstring at the top of this file.")
        sys.exit(1)


def example_basic_description():
    """Describe image content with a GGUF VLM."""
    print("\n=== 1. Basic Image Description ===")

    result = mata.run(
        "vlm",
        IMAGE_PATH,
        model=MODEL_PATH,
        prompt="Describe what you see in this image.",
    )
    print(f"Response: {result.text}")


def example_vqa():
    """Visual question answering with a GGUF VLM."""
    print("\n=== 2. Visual Question Answering ===")

    result = mata.run(
        "vlm",
        IMAGE_PATH,
        model=MODEL_PATH,
        prompt="How many objects are visible? What are they?",
        max_new_tokens=256,
    )
    print(f"Response: {result.text}")


def example_persistent_model():
    """Load once, run multiple prompts — more efficient than re-loading."""
    print("\n=== 3. Load Once, Run Many Prompts ===")

    # CPU-only (default); for GPU: n_gpu_layers=-1
    vlm = mata.load("vlm", MODEL_PATH, n_gpu_layers=0)

    prompts = [
        "What is the main subject of this image?",
        "What colors are most prominent?",
        "Describe the background.",
    ]
    for prompt in prompts:
        result = vlm.predict(IMAGE_PATH, prompt=prompt)
        print(f"  Q: {prompt}")
        print(f"  A: {result.text}\n")


def example_llava_with_mmproj():
    """LLaVA-style model with separate multimodal projector file."""
    print("\n=== 4. LLaVA Model with mmproj ===")

    if MMPROJ_PATH is None or not Path(MMPROJ_PATH).exists():
        print("  (Skipped — set MMPROJ_PATH to your projector .gguf file)")
        return

    vlm = mata.load("vlm", MODEL_PATH, mmproj=MMPROJ_PATH)
    result = vlm.predict(IMAGE_PATH, prompt="Describe the scene in detail.")
    print(f"Response: {result.text}")


if __name__ == "__main__":
    check_prerequisites()
    example_basic_description()
    example_vqa()
    example_persistent_model()
    example_llava_with_mmproj()
