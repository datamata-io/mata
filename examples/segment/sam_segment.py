"""SAM Zero-Shot Segmentation Examples — MATA Framework

Demonstrates point prompts, box prompts, combined prompts, and SAM3 text prompts
for class-agnostic segmentation with the Segment Anything Model.

Run: python examples/segment/sam_segment.py
"""

from pathlib import Path

import mata

# Path to test image (examples/images/ relative to this file's parent)
IMG_PATH = Path(__file__).parent.parent / "images" / "000000039769.jpg"

# SAM model variants
# Original SAM (visual prompts only):
# - facebook/sam-vit-base: Fastest, good for prototyping
# - facebook/sam-vit-large: Better quality, slower
# - facebook/sam-vit-huge: Best quality, slowest
#
# SAM3 (text + visual prompts - Promptable Concept Segmentation):
# - facebook/sam3: Text-based zero-shot segmentation with 270K+ concepts
SAM_MODEL = "facebook/sam-vit-base"
SAM3_MODEL = "facebook/sam3"

print("=" * 70)
print("SAM Zero-Shot Segmentation Examples")
print("=" * 70)

# =================================================================
# Example 0: Text Prompts (SAM3 Only - Concept Segmentation)
# =================================================================
print("\n0. Text Prompt (SAM3 Only - Zero-Shot Concept)")
print("-" * 70)
print("SAM3 adds text-based segmentation for open-vocabulary concepts")

try:
    result_text = mata.run("segment", str(IMG_PATH), model=SAM3_MODEL, text_prompts="cat", threshold=0.5)

    print(f"Found {len(result_text.masks)} instances of 'cat'")
    print(f"Confidence scores: {[f'{m.score:.3f}' for m in result_text.masks]}")
    print(f"Mode: {result_text.meta['mode']}")
    print("\nSAM3 supports text prompts for any concept:")
    print("  - Objects: 'cat', 'dog', 'person', 'car', 'laptop'")
    print("  - Parts: 'ear', 'wheel', 'handle', 'keyboard'")
    print("  - Attributes: 'red car', 'open door', 'running person'")

except Exception as e:
    print(f"SAM3 not available: {e}")
    print("Install with: pip install -U transformers>=4.46.0")
    print("Continuing with original SAM examples (visual prompts)...")

# =================================================================
# Example 0b: Text + Negative Box (SAM3 Refinement)
# =================================================================
print("\n0b. Text + Negative Box (SAM3 Only - Exclude Regions)")
print("-" * 70)

try:
    result_text_refined = mata.run(
        "segment",
        str(IMG_PATH),
        model=SAM3_MODEL,
        text_prompts=["cat"],
        box_prompts=[(0, 0, 100, 100)],  # Exclude top-left region
        box_labels=[0],  # 0 = negative box (exclude)
        threshold=0.5,
    )

    print(f"Found {len(result_text_refined.masks)} cats (excluded top-left region)")
    print("Use case: Remove false positives by excluding specific areas")

except Exception as e:
    print(f"SAM3 not available: {e}")
    print("Install with: pip install -U transformers>=4.46.0")
    print("Continuing with original SAM examples (visual prompts)...")

# =================================================================
# Example 1: Basic Point Prompt (Foreground Click)
# =================================================================
print("\n\n1. Point Prompt (Foreground)")
print("-" * 70)

result = mata.run(
    "segment",
    str(IMG_PATH),
    model=SAM_MODEL,
    point_prompts=[(320, 240, 1)],  # (x, y, label) - label=1 for foreground
    threshold=0.0,  # Return all masks (SAM generates 3 per prompt)
)

print(f"Generated {len(result.masks)} masks")
print(f"Mask qualities (IoU scores): {[f'{m.score:.3f}' for m in result.masks]}")
print(f"Mode: {result.meta['mode']}")
print(f"Prompts: {result.meta['prompts']}")

# SAM returns 3 masks with different granularities - get the best one
best_mask = result.masks[0]  # Already sorted by IoU score
print(f"\nBest mask: IoU={best_mask.score:.3f}, Area={best_mask.area} pixels, BBox={best_mask.bbox}")

# =================================================================
# Example 2: Point Prompt with Threshold Filtering
# =================================================================
print("\n\n2. Point Prompt with Quality Filtering (threshold=0.8)")
print("-" * 70)

result_filtered = mata.run(
    "segment",
    str(IMG_PATH),
    model=SAM_MODEL,
    point_prompts=[(320, 240, 1)],
    threshold=0.8,  # Only return high-quality masks
)

print(f"High-quality masks: {len(result_filtered.masks)}")
print(f"All scores >= 0.8: {all(m.score >= 0.8 for m in result_filtered.masks)}")

# =================================================================
# Example 3: Box Prompt (Region of Interest)
# =================================================================
print("\n\n3. Box Prompt (Segment Specific Region)")
print("-" * 70)

result_box = mata.run(
    "segment", str(IMG_PATH), model=SAM_MODEL, box_prompts=[(100, 100, 400, 400)], threshold=0.5  # (x1, y1, x2, y2)
)

print(f"Generated {len(result_box.masks)} masks from box prompt")
print("Box region: (100, 100, 400, 400)")

# =================================================================
# Example 4: Combined Prompts (Foreground + Background)
# =================================================================
print("\n\n4. Combined Prompts (Foreground + Background Points)")
print("-" * 70)

result_combined = mata.run(
    "segment",
    str(IMG_PATH),
    model=SAM_MODEL,
    point_prompts=[
        (320, 240, 1),  # Foreground point (include this)
        (50, 50, 0),  # Background point (exclude this)
    ],
    threshold=0.7,
)

print(f"Generated {len(result_combined.masks)} masks with refinement")
print("Foreground point: (320, 240), Background point: (50, 50)")

# =================================================================
# Example 5: Multiple Objects with Box + Point Refinement
# =================================================================
print("\n\n5. Box + Point Prompts (Precise Segmentation)")
print("-" * 70)

result_refined = mata.run(
    "segment",
    str(IMG_PATH),
    model=SAM_MODEL,
    point_prompts=[(320, 240, 1)],  # Focus on this point
    box_prompts=[(200, 150, 450, 350)],  # Within this region
    threshold=0.85,
)

print(f"Generated {len(result_refined.masks)} high-precision masks")
print("Combined box + point prompting for better accuracy")

# =================================================================
# Example 6: Load Once, Predict Many (Efficient Workflow)
# =================================================================
print("\n\n6. Efficient Batch Processing (Load Once)")
print("-" * 70)

# Load model once
sam = mata.load("segment", SAM_MODEL, threshold=0.8)

# Predict multiple times with different prompts
prompts = [
    {"point_prompts": [(320, 240, 1)]},
    {"point_prompts": [(150, 180, 1)]},
    {"box_prompts": [(100, 100, 300, 300)]},
]

for i, prompt_kwargs in enumerate(prompts, 1):
    result = sam.predict(str(IMG_PATH), **prompt_kwargs)
    best = f"Best IoU: {result.masks[0].score:.3f}" if result.masks else "no masks above threshold"
    print(f"  Prompt {i}: {len(result.masks)} masks, {best}")

# =================================================================
# Example 7: Class-Agnostic Nature (No Labels)
# =================================================================
print("\n\n7. Class-Agnostic Segmentation (Generic 'object' Label)")
print("-" * 70)

result_agnostic = mata.run("segment", str(IMG_PATH), model=SAM_MODEL, point_prompts=[(320, 240, 1)], threshold=0.9)

# SAM doesn't predict classes - all masks are labeled "object"
for i, mask in enumerate(result_agnostic.masks, 1):
    print(f"  Mask {i}: label={mask.label}, label_name='{mask.label_name}', " f"is_stuff={mask.is_stuff}")

print("\nNote: SAM is class-agnostic. Use a separate classifier for class labels.")

# =================================================================
# Example 8: Comparison with Traditional Segmentation
# =================================================================
print("\n\n8. SAM vs Traditional Segmentation (Mode Comparison)")
print("-" * 70)

# Traditional instance segmentation (requires training on specific classes)
traditional_result = mata.run(
    "segment",
    str(IMG_PATH),
    model="facebook/mask2former-swin-tiny-coco-instance",
    segment_mode="instance",
    threshold=0.5,
)

print(f"Traditional (Mask2Former): {len(traditional_result.masks)} instances")
print("  - Trained on 80 COCO classes")
print("  - Auto-detects objects without prompts")
print(f"  - Classes: {set(m.label_name for m in traditional_result.masks)}")

# SAM zero-shot (works on any object with prompts)
sam_result = mata.run("segment", str(IMG_PATH), model=SAM_MODEL, point_prompts=[(320, 240, 1)], threshold=0.8)

print(f"\nSAM (Zero-shot): {len(sam_result.masks)} masks")
print("  - No training needed (universal)")
print("  - Requires user prompts (points/boxes)")
print("  - Class-agnostic (all 'object')")

# =================================================================
# Example 9: Filtering and Post-Processing
# =================================================================
print("\n\n9. Post-Processing Results")
print("-" * 70)

result_raw = mata.run(
    "segment", str(IMG_PATH), model=SAM_MODEL, point_prompts=[(320, 240, 1)], threshold=0.0  # Get all masks
)

print(f"Raw results: {len(result_raw.masks)} masks")

# Filter by score
high_quality = result_raw.filter_by_score(0.85)
print(f"After filtering (IoU >= 0.85): {len(high_quality.masks)} masks")

# Get mask with largest area
largest_mask = max(result_raw.masks, key=lambda m: m.area or 0)
print(f"Largest mask: {largest_mask.area} pixels, IoU={largest_mask.score:.3f}")

# =================================================================
# Summary
# =================================================================
print("\n" + "=" * 70)
print("Summary: When to Use SAM")
print("=" * 70)
print("""
Use SAM when:
  - You need to segment any object (not just COCO classes)
  - You can provide prompts (clicks, boxes)
  - You want universal segmentation without training
  - Class labels are not required

Use traditional segmentation when:
  - You need automatic detection (no prompts)
  - You need class labels (person, car, etc.)
  - You're working with well-defined categories
  - Speed is critical (SAM is slower)

Tips:
  - Start with point prompts for quick testing
  - Use threshold=0.8 to get only high-quality masks
  - SAM returns 3 masks per prompt - use the best one
  - Combine foreground + background points for refinement
  - Use box prompts for large regions
""")

print("=" * 70)
print("Example completed successfully!")
print("=" * 70)
