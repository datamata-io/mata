"""Basic Segmentation Examples — MATA Framework

Instance and panoptic segmentation with Mask2Former.
Run: python examples/segment/basic_segmentation.py
"""

from pathlib import Path

import mata

IMAGE = Path("examples/images/000000039769.jpg")


# === Section 1: One-Shot Segmentation ===
def one_shot_segmentation():
    """Simplest usage: run segmentation in a single call."""
    if not IMAGE.exists():
        print(f"  ⚠ Image not found: {IMAGE}")
        return

    result = mata.run(
        "segment",
        str(IMAGE),
        model="facebook/mask2former-swin-tiny-coco-instance",
        threshold=0.5,
    )
    print(f"  Found {len(result.masks)} segments")


# === Section 2: Load Once, Predict Many ===
def load_and_predict():
    """Load a model once and reuse it for multiple images."""
    if not IMAGE.exists():
        print(f"  ⚠ Image not found: {IMAGE}")
        return

    segmenter = mata.load("segment", "facebook/mask2former-swin-tiny-coco-instance", threshold=0.5)
    result = segmenter.predict(IMAGE)

    # Access top segments by confidence
    top = sorted(result.masks, key=lambda m: m.score, reverse=True)[:5]
    for i, mask in enumerate(top, 1):
        label = mask.label_name or f"class_{mask.label}"
        print(f"  {i}. {label}: score={mask.score:.3f}, area={mask.area} px")


# === Section 3: Instance vs Panoptic ===
def instance_vs_panoptic():
    """Panoptic mode separates countable instances from background stuff."""
    if not IMAGE.exists():
        print(f"  ⚠ Image not found: {IMAGE}")
        return

    segmenter = mata.load(
        "segment",
        "facebook/mask2former-swin-tiny-coco-panoptic",
        threshold=0.5,
        segment_mode="panoptic",
    )
    result = segmenter.predict(IMAGE)

    instances = result.get_instances()  # countable objects (person, cat, ...)
    stuff = result.get_stuff()          # background regions (sky, floor, ...)

    print(f"  Instances: {len(instances)}, Stuff regions: {len(stuff)}")
    for mask in sorted(instances, key=lambda m: m.score, reverse=True)[:3]:
        print(f"    instance — {mask.label_name}: score={mask.score:.3f}")
    for mask in sorted(stuff, key=lambda m: m.area, reverse=True)[:3]:
        print(f"    stuff    — {mask.label_name}: area={mask.area} px")


# === Section 4: Save Results ===
def save_results():
    """Save segmentation overlay and JSON export."""
    if not IMAGE.exists():
        print(f"  ⚠ Image not found: {IMAGE}")
        return

    result = mata.run(
        "segment",
        str(IMAGE),
        model="facebook/mask2former-swin-tiny-coco-instance",
        threshold=0.5,
    )
    result.save("runs/segment/overlay.png", show_masks=True, show_boxes=True)
    result.save("runs/segment/result.json")
    print("  Saved overlay.png and result.json to runs/segment/")


def main():
    sections = [
        ("1. One-Shot Segmentation", one_shot_segmentation),
        ("2. Load Once, Predict Many", load_and_predict),
        ("3. Instance vs Panoptic", instance_vs_panoptic),
        ("4. Save Results", save_results),
    ]
    for title, fn in sections:
        print(f"\n{title}")
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
