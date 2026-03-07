"""
CLIP Zero-Shot Classification — MATA Framework

Text-prompt-based image classification using CLIP models.
Classify images into arbitrary categories defined at runtime via text prompts —
no fine-tuning or pre-defined label set required.

Sections:
  1. Basic zero-shot classification
  2. Custom text template
  3. Threshold and top-k filtering
  4. Batch classification

Models supported:
  - OpenAI CLIP: openai/clip-vit-base-patch32, openai/clip-vit-large-patch14
  - LAION CLIP: laion/clip-vit-base-patch32-laion2b-s34b-b79k
  - Any HuggingFace CLIP-compatible model

Requirements:
    pip install mata transformers pillow

Usage:
    python examples/classify/clip_zeroshot.py
"""

import sys
from pathlib import Path

from PIL import Image

import mata

IMG_PATH = Path(__file__).parent.parent / "images"
MODEL_NAME = "openai/clip-vit-base-patch32"


# === Section 1: Basic Zero-Shot Classification ===

def example1_basic_usage():
    """One-shot and load/reuse patterns with custom text prompts."""
    print("\n" + "=" * 60)
    print("Section 1: Basic Zero-Shot Classification")
    print("=" * 60)

    # One-shot: classify without pre-loading
    print("\n[one-shot] Running direct classification...")
    result = mata.run(
        "classify",
        str(IMG_PATH / "000000039769.jpg"),
        model=MODEL_NAME,
        text_prompts=["cat", "dog", "remote control", "couch"],
    )
    print(f"  Top prediction: {result.top1.label_name}  ({result.top1.score:.4f})")

    # Load once, reuse for multiple images
    print("\n[load/reuse] Loading classifier once...")
    classifier = mata.load("classify", MODEL_NAME)

    text_prompts = ["cat", "sleeping cat", "remote", "couch"]
    image = Image.open(str(IMG_PATH / "000000039769.jpg")).convert("RGB")
    result = classifier.predict(image, text_prompts=text_prompts)

    print("\nClassification results:")
    for pred in result.predictions:
        print(f"  {pred.label_name:20s}: {pred.score:.4f}")
    print(f"\nTop-1: {result.top1.label_name}")
    print(f"Top-5: {[p.label_name for p in result.top5]}")


# === Section 2: Custom Text Templates ===

def example2_custom_template():
    """Use a text template to improve classification accuracy."""
    print("\n" + "=" * 60)
    print("Section 2: Custom Text Template")
    print("=" * 60)

    # Default template wraps prompts as "a photo of a {label}"
    # Custom templates can shift the model's attention
    classifier = mata.load(
        "classify",
        MODEL_NAME,
        template="this is a photo of a {}",
    )

    image = Image.open(str(IMG_PATH / "000000039769.jpg")).convert("RGB")
    text_prompts = ["airplane", "car", "boat", "train", "bicycle", "cat"]

    result = classifier.predict(image, text_prompts=text_prompts)

    print("\nWith template 'this is a photo of a {}':")
    for pred in result.predictions:
        print(f"  {pred.label_name:15s}: {pred.score:.4f}")
    print(f"\nTop prediction: {result.top1.label_name}")


# === Section 3: Threshold and Top-K Filtering ===

def example3_threshold_and_topk():
    """Filter predictions by confidence threshold and limit count with top-k."""
    print("\n" + "=" * 60)
    print("Section 3: Threshold + Top-K Filtering")
    print("=" * 60)

    # Set minimum confidence and maximum result count at load time
    classifier = mata.load(
        "classify",
        MODEL_NAME,
        threshold=0.05,  # drop predictions below 5% confidence
        top_k=3,         # return at most 3 results
    )

    image = Image.open(str(IMG_PATH / "000000039769.jpg")).convert("RGB")
    text_prompts = [
        "cat", "dog", "remote control", "couch",
        "painting", "photograph", "landscape", "animal",
    ]

    result = classifier.predict(image, text_prompts=text_prompts)

    print(f"\nAmong {len(text_prompts)} categories (threshold=0.05, top_k=3):")
    for pred in result.predictions:
        print(f"  {pred.label_name:20s}: {pred.score:.4f}")
    print(f"\nReturned {len(result.predictions)} prediction(s) after filtering")

    # Can also override at predict time
    result_top1 = classifier.predict(image, text_prompts=text_prompts, top_k=1)
    print(f"\nWith top_k=1 override: {result_top1.top1.label_name} ({result_top1.top1.score:.4f})")


# === Section 4: Batch Classification ===

def example4_batch_classification():
    """Classify multiple images in a loop using a single loaded classifier."""
    print("\n" + "=" * 60)
    print("Section 4: Batch Classification")
    print("=" * 60)

    classifier = mata.load("classify", MODEL_NAME)

    # Synthetic images for deterministic demo (solid colors)
    images = [
        ("red swatch",   Image.new("RGB", (224, 224), color="red")),
        ("green swatch", Image.new("RGB", (224, 224), color="green")),
        ("blue swatch",  Image.new("RGB", (224, 224), color="blue")),
    ]
    text_prompts = ["red color", "green color", "blue color", "yellow color"]

    print(f"\nClassifying {len(images)} images against: {text_prompts}\n")
    for name, img in images:
        result = classifier.predict(img, text_prompts=text_prompts, top_k=2)
        top2 = [(p.label_name, f"{p.score:.4f}") for p in result.predictions]
        print(f"  {name:15s} to {top2}")


def main():
    print("=" * 60)
    print("CLIP Zero-Shot Classification — MATA Framework")
    print("=" * 60)

    try:
        example1_basic_usage()
        example2_custom_template()
        example3_threshold_and_topk()
        example4_batch_classification()

        print("\n" + "=" * 60)
        print("All examples completed successfully.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
