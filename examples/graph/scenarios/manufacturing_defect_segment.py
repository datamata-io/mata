#!/usr/bin/env python3
"""Manufacturing: Defect Segmentation & Area Measurement.

Real-World Problem:
    Quality control requires not just detecting defects, but measuring their
    precise size and shape. For example, a scratch's length or a dent's area
    determines whether a part is salvageable or scrapped.

Solution:
    GroundingDINO detects defect regions using text prompts, then SAM segments
    each defect with pixel-level precision. MaskToBox converts masks to bounding
    boxes for easy area calculation (width × height).

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Segmenter: facebook/sam-vit-base (prompt-based segmentation)

Graph Flow:
    Detect("scratch . crack . dent") → PromptBoxes(SAM) → RefineMask → MaskToBox

Usage:
    python manufacturing_defect_segment.py                    # Mock mode
    python manufacturing_defect_segment.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import grounding_dino_sam

        # GroundingDINO supports text prompts passed during inference
        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        segmenter = mata.load("segment", "facebook/sam-vit-base")

        # Reuse existing preset with defect-specific detection threshold
        result = mata.infer(
            image_path,
            grounding_dino_sam(
                detection_threshold=0.25,
                refine_method="morph_close",
                refine_radius=3,
            ),
            providers={"detector": detector, "segmenter": segmenter},
            # Text prompts are passed to the detector
            text_prompts="scratch . crack . dent . corrosion . damage",
        )

        print(f"Segmented {len(result['final'].instances)} defects:")
        for inst in result["final"].instances:
            if inst.mask is not None:
                # Calculate approximate area from bbox (width × height)
                x1, y1, x2, y2 = inst.bbox
                area_pixels = (x2 - x1) * (y2 - y1)
                print(f"  {inst.label_name}: {inst.score:.2f}, area ≈ {area_pixels:.0f} px²")
            else:
                print(f"  {inst.label_name}: {inst.score:.2f} at {inst.bbox}")
    else:
        print("=== Manufacturing: Defect Segmentation & Area Measurement (Mock) ===")
        print()
        print("Graph: Detect → PromptBoxes(SAM) → RefineMask → MaskToBox")
        print("Models: GroundingDINO (detector) + SAM (segmenter)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → list of defect instances")
        print("  Each instance has:")
        print("    - bbox: bounding box derived from segmentation mask")
        print("    - mask: pixel-precise segmentation mask (RLE format)")
        print("    - score: detection confidence")
        print("  Area calculation: (x2-x1) × (y2-y1) from bbox")
        print()

        # Verify preset construction
        from mata.presets import grounding_dino_sam
        graph = grounding_dino_sam(detection_threshold=0.25)
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Note: Text prompts ('scratch . crack . dent . corrosion . damage')")
        print("      are passed as kwargs to mata.infer(), not to the preset.")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
