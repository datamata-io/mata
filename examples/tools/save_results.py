#!/usr/bin/env python3
"""Save Results — MATA Framework

Comprehensive demonstration of all save() functionality across
detection, segmentation, and classification tasks.

Showcases:
- JSON export (structured data)
- CSV export (tabular data)
- Image overlays (visual results)
- Detection crop extraction
- Format auto-detection
- Customization options
- Error handling

Run: python examples/tools/save_results.py
"""

from pathlib import Path

from PIL import Image

import mata


def setup_outputs():
    """Create outputs directory structure."""
    base_dir = Path(__file__).parent / "outputs" / "save_demo"

    dirs = {
        "detection": base_dir / "detection",
        "segmentation": base_dir / "segmentation",
        "classification": base_dir / "classification",
        "crops": base_dir / "crops",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def demonstrate_detection_saves(image_path: Path, output_dir: Path):
    """Show all detection save formats."""
    print("\n" + "=" * 70)
    print("DETECTION SAVE DEMONSTRATION")
    print("=" * 70)

    print(f"\n1. Running detection on: {image_path.name}")
    result = mata.run("detect", str(image_path), model="facebook/detr-resnet-50", threshold=0.5)
    print(f"   ✓ Found {len(result.instances)} detections")

    # JSON export
    print("\n2. Saving JSON (structured data)")
    json_path = output_dir / "detections.json"
    result.save(str(json_path))
    print(f"   ✓ JSON: {json_path}")

    # CSV export
    print("\n3. Saving CSV (spreadsheet)")
    csv_path = output_dir / "detections.csv"
    result.save(str(csv_path))
    print(f"   ✓ CSV: {csv_path}")
    print("   Columns: id, label, label_name, score, x1, y1, x2, y2, area")

    # Image overlay
    print("\n4. Saving image overlay (default)")
    overlay_path = output_dir / "overlay_default.png"
    result.save(str(overlay_path))
    print(f"   ✓ Overlay: {overlay_path}")

    # Custom overlay (no scores)
    print("\n5. Saving custom overlay (no scores)")
    overlay_path = output_dir / "overlay_custom.png"
    result.save(str(overlay_path), show_boxes=True, show_labels=True, show_scores=False)
    print(f"   ✓ Custom overlay: {overlay_path}")

    # Crop extraction
    print("\n6. Extracting detection crops")
    crop_base = output_dir / "all_detections.png"
    result.save(str(crop_base), crop_dir="detection_crops", padding=10)
    crop_dir = output_dir / "detection_crops"
    crop_count = len(list(crop_dir.glob("*.png")))
    print(f"   ✓ Extracted {crop_count} crops to: {crop_dir}")

    return result


def demonstrate_segmentation_saves(image_path: Path, output_dir: Path):
    """Show all segmentation save formats."""
    print("\n" + "=" * 70)
    print("SEGMENTATION SAVE DEMONSTRATION")
    print("=" * 70)

    print(f"\n1. Running segmentation on: {image_path.name}")
    result = mata.run("segment", str(image_path), model="facebook/mask2former-swin-tiny-coco-instance", threshold=0.5)
    print(f"   ✓ Found {len(result.instances)} segments")

    # JSON export
    print("\n2. Saving JSON (structured data with masks)")
    json_path = output_dir / "segments.json"
    result.save(str(json_path))
    print(f"   ✓ JSON: {json_path}")

    # CSV export
    print("\n3. Saving CSV (mask metadata)")
    csv_path = output_dir / "segments.csv"
    result.save(str(csv_path))
    print(f"   ✓ CSV: {csv_path}")

    # Mask overlay
    print("\n4. Saving mask overlay (default)")
    overlay_path = output_dir / "overlay_with_masks.png"
    result.save(str(overlay_path))
    print(f"   ✓ Mask overlay: {overlay_path}")

    # Transparent masks
    print("\n5. Saving transparent masks (alpha=0.3)")
    overlay_path = output_dir / "overlay_transparent.png"
    result.save(str(overlay_path), show_masks=True, alpha=0.3)
    print(f"   ✓ Transparent overlay: {overlay_path}")

    # Boxes only
    print("\n6. Saving boxes only (no masks)")
    overlay_path = output_dir / "overlay_boxes_only.png"
    result.save(str(overlay_path), show_masks=False, show_boxes=True)
    print(f"   ✓ Boxes only: {overlay_path}")

    return result


def demonstrate_classification_saves(image_path: Path, output_dir: Path):
    """Show all classification save formats."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION SAVE DEMONSTRATION")
    print("=" * 70)

    print(f"\n1. Running classification on: {image_path.name}")
    result = mata.run("classify", str(image_path), model="google/vit-base-patch16-224", top_k=10)
    print(f"   ✓ Top prediction: {result.predictions[0].label_name}")
    print(f"   ✓ Confidence: {result.predictions[0].score:.2%}")

    # JSON export
    print("\n2. Saving JSON (all predictions)")
    json_path = output_dir / "predictions.json"
    result.save(str(json_path))
    print(f"   ✓ JSON: {json_path}")

    # CSV export
    print("\n3. Saving CSV (ranked predictions)")
    csv_path = output_dir / "predictions.csv"
    result.save(str(csv_path))
    print(f"   ✓ CSV: {csv_path}")

    # Bar chart
    print("\n4. Saving bar chart (top 5)")
    try:
        chart_path = output_dir / "chart_top5.png"
        result.save(str(chart_path), top_k=5)
        print(f"   ✓ Chart: {chart_path}")
    except ImportError:
        print("   ⚠ Matplotlib not installed — skipping chart")
        print("   Install with: pip install matplotlib")

    return result


def demonstrate_pil_input_handling(image_path: Path, output_dir: Path):
    """Show how to handle PIL Image inputs."""
    print("\n" + "=" * 70)
    print("PIL IMAGE INPUT HANDLING")
    print("=" * 70)

    print(f"\n1. Loading PIL Image: {image_path.name}")
    pil_image = Image.open(image_path)

    print("\n2. Running detection on PIL Image")
    result = mata.run("detect", pil_image, model="facebook/detr-resnet-50", threshold=0.5)
    print(f"   ✓ Found {len(result.instances)} detections")

    # JSON/CSV work without image path
    print("\n3. Saving JSON/CSV (no image path needed)")
    result.save(str(output_dir / "pil_detections.json"))
    result.save(str(output_dir / "pil_detections.csv"))
    print(f"   ✓ JSON + CSV saved")

    # Image overlay requires explicit image
    print("\n4. Saving overlay (explicit image required for PIL input)")
    overlay_path = output_dir / "pil_overlay.png"
    result.save(str(overlay_path), image=str(image_path))
    print(f"   ✓ Overlay: {overlay_path}")


def demonstrate_error_handling(image_path: Path, output_dir: Path):
    """Show error handling for common issues."""
    print("\n" + "=" * 70)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 70)

    result = mata.run("detect", str(image_path), model="facebook/detr-resnet-50", threshold=0.5)

    print("\n1. Unsupported file format")
    try:
        result.save("output.xyz")
    except ValueError as e:
        print(f"   ✓ Caught ValueError: {e}")

    print("\n2. Missing image for overlay (PIL input)")
    try:
        pil_result = mata.run("detect", Image.open(image_path))
        pil_result.save(str(output_dir / "overlay.png"))
    except Exception as e:
        print(f"   ✓ Caught {type(e).__name__}")
        print("   Solution: result.save('overlay.png', image='path.jpg')")


def main():
    """Run complete save functionality demonstration."""
    print("=" * 70)
    print("MATA — Save Results Demonstration")
    print("=" * 70)

    image_path = Path(__file__).parent.parent / "images" / "000000039769.jpg"
    if not image_path.exists():
        print(f"\n⚠ Image not found: {image_path}")
        print("Please ensure a test image exists at examples/images/000000039769.jpg")
        return

    dirs = setup_outputs()
    print(f"\n✓ Output directory: {dirs['detection'].parent}")

    demonstrate_detection_saves(image_path, dirs["detection"])
    demonstrate_segmentation_saves(image_path, dirs["segmentation"])
    demonstrate_classification_saves(image_path, dirs["classification"])
    demonstrate_pil_input_handling(image_path, dirs["detection"])
    demonstrate_error_handling(image_path, dirs["detection"])

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Extension auto-detection: .json → JSON, .csv → CSV, .png → image")
    print("  • Image path auto-stored when using file paths")
    print("  • PIL/numpy inputs require explicit image for overlay/crops")
    print("  • Customize overlays with show_boxes/show_labels/show_scores/alpha")
    print("  • Crop extraction creates a subdirectory with sequential naming")
    print(f"\n✓ All outputs saved to: {dirs['detection'].parent}")
    print("=" * 70)


if __name__ == "__main__":
    main()
