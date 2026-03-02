"""
Zero-Shot Object Detection — MATA Framework

Demonstrates text-prompt-based object detection using GroundingDINO and OWL-ViT models.
Detects arbitrary objects specified via text descriptions without any training.

Models supported:
- GroundingDINO (IDEA-Research): State-of-the-art, slower
- OWL-ViT v1 (Google): Fast, good accuracy
- OWL-ViT v2 (Google): Best speed/accuracy tradeoff

Requirements:
    pip install mata transformers pillow

Usage:
    python examples/detect/zeroshot_detection.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import mata


def draw_detections(image, result, text_prompts):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]

    for idx, instance in enumerate(result.instances):
        x1, y1, x2, y2 = instance.bbox
        color = colors[idx % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"{instance.label_name}: {instance.score:.2f}"
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    return image


def example_grounding_dino():
    """Example 1: GroundingDINO - Best performance."""
    print("\n" + "=" * 70)
    print("Example 1: GroundingDINO - State-of-the-art zero-shot detection")
    print("=" * 70)

    print("\n[1/4] Loading GroundingDINO model...")
    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

    print("[2/4] Creating test image...")
    image_path = Path("examples/images/test_image.jpg")
    if not image_path.exists():
        image = Image.new("RGB", (800, 600), color="lightblue")
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 300, 300], fill="brown", outline="black", width=3)
        draw.rectangle([400, 200, 600, 450], fill="green", outline="black", width=3)
        draw.ellipse([150, 350, 250, 450], fill="red", outline="black", width=3)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        print(f"   Created test image: {image_path}")

    image = Image.open(image_path)

    # GroundingDINO uses space-dot separated prompts
    text_prompts = "square . rectangle . circle . object"
    print(f"[3/4] Running detection with prompts: '{text_prompts}'")

    result = detector.predict(image, text_prompts=text_prompts)

    print(f"[4/4] Found {len(result.instances)} objects:")
    for instance in result.instances:
        print(f"   - {instance.label_name}: {instance.score:.3f} at bbox {instance.bbox}")

    output_image = draw_detections(image.copy(), result, text_prompts)
    output_path = "examples/images/output_grounding_dino.jpg"
    output_image.save(output_path)
    print(f"\n✓ Saved visualization to: {output_path}")

    return result


def example_owlvit_v2():
    """Example 2: OWL-ViT v2 - Fast inference."""
    print("\n" + "=" * 70)
    print("Example 2: OWL-ViT v2 - Fast zero-shot detection")
    print("=" * 70)

    print("\n[1/4] Loading OWL-ViT v2 model...")
    detector = mata.load("detect", "google/owlv2-base-patch16")

    print("[2/4] Loading test image...")
    image_path = Path("examples/images/test_image.jpg")
    image = Image.open(image_path)

    # OWL-ViT accepts a list of prompts
    text_prompts = ["square", "rectangle", "circle", "object"]
    print(f"[3/4] Running detection with prompts: {text_prompts}")

    result = detector.predict(image, text_prompts=text_prompts)

    print(f"[4/4] Found {len(result.instances)} objects:")
    for instance in result.instances:
        print(f"   - {instance.label_name}: {instance.score:.3f} at bbox {instance.bbox}")

    output_image = draw_detections(image.copy(), result, text_prompts)
    output_path = "examples/images/output_owlvit_v2.jpg"
    output_image.save(output_path)
    print(f"\n✓ Saved visualization to: {output_path}")

    return result


def example_batch_processing():
    """Example 3: Batch processing multiple images."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing - Multiple images at once")
    print("=" * 70)

    print("\n[1/4] Loading GroundingDINO model...")
    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

    print("[2/4] Creating test image batch...")
    images = []
    for i in range(3):
        img = Image.new("RGB", (400, 300), color=["lightblue", "lightgreen", "lightyellow"][i])
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill="red", outline="black", width=2)
        draw.ellipse([200, 100, 350, 250], fill="blue", outline="black", width=2)
        images.append(img)

    text_prompts = "square . circle"
    print(f"[3/4] Running batch detection on {len(images)} images...")

    results = detector.predict(images, text_prompts=text_prompts)

    print("[4/4] Batch results:")
    for idx, result in enumerate(results):
        print(f"   Image {idx+1}: {len(result.instances)} objects detected")
        for instance in result.instances:
            print(f"      - {instance.label_name}: {instance.score:.3f}")

    print(f"\n✓ Processed {len(images)} images in batch")

    return results


def example_custom_threshold():
    """Example 4: Custom confidence threshold."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Threshold - Filter low-confidence detections")
    print("=" * 70)

    print("\n[1/4] Loading detector with threshold=0.5...")
    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny", threshold=0.5)

    print("[2/4] Loading test image...")
    image_path = Path("examples/images/test_image.jpg")
    image = Image.open(image_path)

    text_prompts = "square . rectangle . circle . object"
    print("[3/4] Running detection with high threshold...")
    result = detector.predict(image, text_prompts=text_prompts)

    print("[4/4] High-confidence objects (>0.5):")
    for instance in result.instances:
        print(f"   - {instance.label_name}: {instance.score:.3f} at bbox {instance.bbox}")

    print("\n[Bonus] Runtime threshold override to 0.2...")
    result_low = detector.predict(image, text_prompts=text_prompts, threshold=0.2)
    print(f"   Lower threshold found {len(result_low.instances)} objects (vs {len(result.instances)} at 0.5)")

    return result


def example_model_comparison():
    """Example 5: Compare different zero-shot models."""
    print("\n" + "=" * 70)
    print("Example 5: Model Comparison - GroundingDINO vs OWL-ViT")
    print("=" * 70)

    print("\n[1/4] Loading test image...")
    image_path = Path("examples/images/test_image.jpg")
    image = Image.open(image_path)

    text_prompts = "square . circle"

    print("[2/4] Testing GroundingDINO...")
    detector_gdino = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
    result_gdino = detector_gdino.predict(image, text_prompts=text_prompts)
    print(f"   GroundingDINO: {len(result_gdino.instances)} objects")

    print("[3/4] Testing OWL-ViT v1...")
    detector_owlv1 = mata.load("detect", "google/owlvit-base-patch32")
    result_owlv1 = detector_owlv1.predict(image, text_prompts=["square", "circle"])
    print(f"   OWL-ViT v1: {len(result_owlv1.instances)} objects")

    print("[4/4] Testing OWL-ViT v2...")
    detector_owlv2 = mata.load("detect", "google/owlv2-base-patch16")
    result_owlv2 = detector_owlv2.predict(image, text_prompts=["square", "circle"])
    print(f"   OWL-ViT v2: {len(result_owlv2.instances)} objects")

    print("\n[Results] Model comparison:")
    print(f"   ├─ GroundingDINO: {len(result_gdino.instances)} detections")
    print(f"   ├─ OWL-ViT v1:    {len(result_owlv1.instances)} detections")
    print(f"   └─ OWL-ViT v2:    {len(result_owlv2.instances)} detections")

    return result_gdino, result_owlv1, result_owlv2


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MATA Zero-Shot Object Detection Examples")
    print("=" * 70)
    print("\nThis script demonstrates text-prompt-based object detection")
    print("using GroundingDINO and OWL-ViT models.")
    print("\nNo training required — just describe what you want to detect!")

    try:
        example_grounding_dino()
        example_owlvit_v2()
        example_batch_processing()
        example_custom_threshold()
        example_model_comparison()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Check the output images in examples/images/")
        print("  2. Try with your own images")
        print("  3. Experiment with different text prompts")
        print("  4. Explore the GroundingDINO→SAM pipeline: examples/segment/grounding_sam_pipeline.py")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
