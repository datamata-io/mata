#!/usr/bin/env python3
"""Manufacturing: Surface Defect Detection & Classification.

Real-World Problem:
    Manufacturing lines need automated visual inspection to detect surface
    defects (scratches, cracks, dents) on metal/plastic parts and classify
    each defect type for root-cause analysis and quality reporting.

Solution:
    GroundingDINO detects defect regions using text prompts (zero-shot,
    no training required). Each detected region is cropped and classified
    by CLIP into specific defect categories.

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Classifier: openai/clip-vit-base-patch32 (zero-shot classification)

Graph Flow:
    Detect("scratch . crack . dent") > Filter > ExtractROIs > Classify > Fuse

Usage:
    python manufacturing_defect_classify.py                    # Mock mode
    python manufacturing_defect_classify.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import defect_detect_classify

        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        classifier = mata.load("classify", "openai/clip-vit-base-patch32")

        result = mata.infer(
            image_path,
            defect_detect_classify(
                defect_prompts="scratch . crack . dent . corrosion",
                classification_labels=["scratch", "crack", "dent", "corrosion", "normal"],
            ),
            providers={"detector": detector, "classifier": classifier},
        )

        print(f"Detected {len(result['final'].instances)} defects:")
        for inst in result["final"].instances:
            print(f"  {inst.label_name}: {inst.score:.2f} at {inst.bbox}")
    else:
        print("=== Manufacturing: Defect Detection & Classification (Mock) ===")
        print()
        print("Graph: Detect > Filter > ExtractROIs > Classify > Fuse")
        print("Models: GroundingDINO (detector) + CLIP (classifier)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > list of defect instances with bboxes")
        print("  result['final'].rois > cropped images of each defect")
        print("  result['final'].classifications > defect type per crop")
        print()

        # Verify preset construction
        from mata.presets import defect_detect_classify
        graph = defect_detect_classify()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
