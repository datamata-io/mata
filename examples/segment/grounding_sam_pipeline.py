"""GroundingDINO + SAM Pipeline Examples — MATA Framework

Demonstrates text-prompt-based instance segmentation using the GroundingDINO>SAM
pipeline. Combines zero-shot object detection with zero-shot instance segmentation
for precise masks.

Pipeline Flow:
    Text Prompts > GroundingDINO (bboxes) > SAM (masks) > VisionResult

Run: python examples/segment/grounding_sam_pipeline.py
Requirements: pip install mata transformers pillow numpy
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mata
from mata.visualization import _mask_to_binary


def visualize_instances(image, result, output_path):
    """Visualize instances with both bboxes and masks."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_image = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    colors = [
        (255, 0, 0, 100),
        (0, 255, 0, 100),
        (0, 0, 255, 100),
        (255, 255, 0, 100),
        (255, 0, 255, 100),
        (0, 255, 255, 100),
        (255, 128, 0, 100),
        (128, 0, 255, 100),
    ]
    outline_colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]

    print(f"\nVisualizing {len(result.instances)} instances:")

    for idx, instance in enumerate(result.instances):
        color = colors[idx % len(colors)]
        outline_color = outline_colors[idx % len(outline_colors)]

        if instance.mask is not None:
            if hasattr(instance.mask, "to_binary"):
                mask_array = instance.mask.to_binary()
            else:
                mask_array = _mask_to_binary(instance.mask, image_size=image.size)

            mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
            colored_mask = Image.new("RGBA", image.size, color)
            overlay.paste(colored_mask, (0, 0), mask_img)

            print(f"  [{idx+1}] {instance.label_name}: bbox + mask (area={instance.area})")
        else:
            print(f"  [{idx+1}] {instance.label_name}: bbox only (no mask)")

        if instance.bbox is not None:
            x1, y1, x2, y2 = instance.bbox
            draw_image.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)

            label_text = f"{instance.label_name}: {instance.score:.2f}"
            bbox = draw_image.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw_image.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=outline_color)
            draw_image.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.save(output_path)
    print(f"Saved visualization to: {output_path}")

    return image


def example_basic_pipeline():
    """Example 1: Basic GroundingDINO>SAM pipeline with text prompts."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Pipeline - Text > BBox > Mask")
    print("=" * 70)

    print("\n[1/4] Loading GroundingDINO>SAM pipeline...")
    pipeline = mata.load(
        "pipeline", 
        detector_model_id="IDEA-Research/grounding-dino-tiny", 
        sam_model_id="facebook/sam-vit-base"
    )

    print("[2/4] Creating test image...")
    image_path = Path("examples/images/pipeline_test.jpg")
    if not image_path.exists():
        image = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 300, 300], fill="red", outline="darkred", width=3)
        draw.ellipse([400, 150, 650, 400], fill="blue", outline="darkblue", width=3)
        draw.polygon([(100, 450), (250, 350), (400, 450)], fill="green", outline="darkgreen", width=3)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        print(f"   Created test image: {image_path}")

    image = Image.open(image_path)

    text_prompts = "square . circle . triangle"
    print(f"[3/4] Running pipeline with prompts: '{text_prompts}'")
    result = pipeline.predict(image, text_prompts=text_prompts)

    print(f"[4/4] Pipeline complete! Found {len(result.instances)} instances:")
    for instance in result.instances:
        has_mask = "yes" if instance.mask is not None else "no"
        print(f"   - {instance.label_name}: score={instance.score:.3f}, mask={has_mask}")

    output_path = "examples/images/output_pipeline_basic.jpg"
    visualize_instances(image.copy(), result, output_path)

    return result


def example_custom_thresholds():
    """Example 2: Custom detection and segmentation thresholds."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Thresholds")
    print("=" * 70)

    pipeline = mata.load(
        "pipeline",
        detector_model_id="IDEA-Research/grounding-dino-tiny",
        sam_model_id="facebook/sam-vit-base",
        detection_threshold=0.4,
        segmentation_threshold=0.6,
    )

    image_path = Path("examples/images/pipeline_test.jpg")
    image = Image.open(image_path)

    text_prompts = "square . circle . triangle . shape"
    result = pipeline.predict(image, text_prompts=text_prompts)

    print(f"\nFound {len(result.instances)} high-confidence instances:")
    for instance in result.instances:
        print(f"   - {instance.label_name}: {instance.score:.3f}")

    output_path = "examples/images/output_pipeline_thresholds.jpg"
    visualize_instances(image.copy(), result, output_path)

    return result


def example_empty_detection_handling():
    """Example 3: Handling cases with no detections."""
    print("\n" + "=" * 70)
    print("Example 3: Empty Detection Handling")
    print("=" * 70)

    pipeline = mata.load(
        "pipeline", detector_model_id="IDEA-Research/grounding-dino-tiny", sam_model_id="facebook/sam-vit-base"
    )

    image = Image.new("RGB", (400, 300), color="white")

    text_prompts = "unicorn . dragon . phoenix"
    print(f"Searching for non-existent objects: '{text_prompts}'")
    result = pipeline.predict(image, text_prompts=text_prompts)

    if len(result.instances) == 0:
        print("No objects detected (as expected) — pipeline returns empty result")
    else:
        print(f"Unexpectedly found {len(result.instances)} objects")

    return result


def main():
    """Run all pipeline examples."""
    print("\n" + "=" * 70)
    print("MATA GroundingDINO + SAM Pipeline Examples")
    print("=" * 70)
    print("\nCombines zero-shot detection (GroundingDINO) with zero-shot")
    print("segmentation (SAM) for text-prompted instance segmentation.")

    try:
        example_basic_pipeline()
        example_custom_thresholds()
        example_empty_detection_handling()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("Output images saved to examples/images/")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
