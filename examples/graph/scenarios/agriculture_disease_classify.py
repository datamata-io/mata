#!/usr/bin/env python3
"""Agriculture: Crop Disease Detection & Classification.

Real-World Problem:
    Agricultural monitoring requires early detection of crop diseases to
    prevent widespread infection and crop loss. Manual inspection is slow
    and inconsistent across large fields. Farmers need automated systems
    to identify diseased leaves and classify the disease type for targeted
    treatment.

Solution:
    GroundingDINO detects diseased leaf regions using text prompts
    (zero-shot, no training required). Each detected region is cropped
    and classified by CLIP into specific disease categories (bacterial spot,
    powdery mildew, leaf blight, rust, or healthy).

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Classifier: openai/clip-vit-base-patch32 (zero-shot classification)

Graph Flow:
    Detect("diseased leaf . pest damage . healthy leaf") > Filter > ExtractROIs > Classify > Fuse

Usage:
    python agriculture_disease_classify.py                    # Mock mode
    python agriculture_disease_classify.py --real image.jpg   # Real inference
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
                defect_prompts="diseased leaf . pest damage . healthy leaf",
                classification_labels=["healthy", "bacterial_spot", "powdery_mildew", "leaf_blight", "rust"],
            ),
            providers={"detector": detector, "classifier": classifier},
        )

        print(f"Detected {len(result['final'].instances)} leaf regions:")
        for inst in result["final"].instances:
            print(f"  {inst.label_name}: {inst.score:.2f} at {inst.bbox}")
    else:
        print("=== Agriculture: Crop Disease Detection & Classification (Mock) ===")
        print()
        print("Real-World Problem:")
        print("  Farmers need automated disease detection to prevent crop loss.")
        print("  Manual inspection is slow and inconsistent across large fields.")
        print()
        print("Graph: Detect > Filter > ExtractROIs > Classify > Fuse")
        print("Models: GroundingDINO (detector) + CLIP (classifier)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > list of leaf regions with disease classifications")
        print("  result['final'].rois > cropped images of each detected leaf")
        print("  result['final'].classifications > disease type per crop")
        print()

        # Verify preset construction
        from mata.presets import defect_detect_classify
        graph = defect_detect_classify(
            defect_prompts="diseased leaf . pest damage . healthy leaf",
            classification_labels=["healthy", "bacterial_spot", "powdery_mildew", "leaf_blight", "rust"],
        )
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
