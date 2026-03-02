#!/usr/bin/env python3
"""Agriculture: Pest Detection & Area Mapping.

Real-World Problem:
    Early detection of pest infestations is critical for crop protection.
    Farmers need to identify pest locations, measure affected areas, and
    prioritize treatment zones. Traditional manual scouting is labor-intensive
    and often misses early-stage infestations.

Solution:
    GroundingDINO detects pest insects using text prompts (zero-shot).
    SAM segments each detection to precisely measure the affected crop area.
    Refined masks provide accurate polygon boundaries for targeted pesticide
    application and infestation monitoring.

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Segmenter: facebook/sam-vit-base (prompt-based segmentation)

Graph Flow:
    Detect("insect . pest . caterpillar . aphid . beetle") → Filter → PromptBoxes(SAM) → RefineMask → Fuse

Usage:
    python agriculture_pest_detection.py                    # Mock mode
    python agriculture_pest_detection.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import grounding_dino_sam

        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        segmenter = mata.load("segment", "facebook/sam-vit-base")

        result = mata.infer(
            image_path,
            grounding_dino_sam(detection_threshold=0.25),
            providers={"detector": detector, "segmenter": segmenter},
            text_prompts="insect . pest . caterpillar . aphid . beetle",
        )

        print(f"Detected {len(result['final'].instances)} pest instances:")
        for inst in result["final"].instances:
            has_mask = inst.mask is not None
            print(f"  {inst.label_name}: {inst.score:.2f} at {inst.bbox}, mask: {has_mask}")
    else:
        print("=== Agriculture: Pest Detection & Area Mapping (Mock) ===")
        print()
        print("Real-World Problem:")
        print("  Early pest detection is critical for crop protection.")
        print("  Need to identify locations, measure affected areas, and prioritize treatment.")
        print()
        print("Graph: Detect → Filter → PromptBoxes(SAM) → RefineMask → Fuse")
        print("Models: GroundingDINO (detector) + SAM (segmenter)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → pest detections with precise segmentation masks")
        print("  Each instance has bbox + mask for area measurement")
        print("  Enables: targeted pesticide application, infestation monitoring")
        print()

        # Verify preset construction
        from mata.presets import grounding_dino_sam
        graph = grounding_dino_sam(detection_threshold=0.25)
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")
        print("(Text prompts 'insect . pest . caterpillar . aphid . beetle' passed via text_prompts parameter)")


if __name__ == "__main__":
    main()
