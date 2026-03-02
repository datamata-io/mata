#!/usr/bin/env python3
"""Security: Suspicious Unattended Object Detection & Analysis.

Real-World Problem:
    Security systems in airports, transit stations, and public venues need to
    automatically detect and assess potentially suspicious unattended objects
    (bags, packages, backpacks). Manual monitoring is slow, and false positives
    waste security resources. The challenge is not just detection, but contextual
    understanding — is the bag truly abandoned, or is its owner nearby?

Solution:
    Three-stage AI pipeline combining zero-shot detection, precise segmentation,
    and visual language reasoning:
    1. GroundingDINO detects suspicious object classes using text prompts
    2. SAM segments each object for precise boundary delineation
    3. VLM analyzes each object in context: Is it unattended? Abandoned? Suspicious?

    This is the most complex security scenario — the VLM provides human-like
    reasoning that goes beyond simple object detection.

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Segmenter: facebook/sam-vit-base (prompt-based segmentation)
    - VLM: Qwen/Qwen3-VL-2B-Instruct (contextual reasoning)

Graph Flow:
    Detect → Filter → PromptBoxes(SAM) → RefineMask → VLMQuery → Fuse

Usage:
    python security_suspicious_object.py                    # Mock mode
    python security_suspicious_object.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "surveillance_scene.jpg")

    if USE_REAL:
        import mata
        from mata.presets import suspicious_object_detection

        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        segmenter = mata.load("segment", "facebook/sam-vit-base")
        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

        result = mata.infer(
            image_path,
            suspicious_object_detection(
                object_prompts="backpack . suitcase . bag . package . luggage",
                vlm_prompt=(
                    "Analyze this object carefully. Is it unattended or abandoned? "
                    "Is there a person nearby who appears to be the owner? "
                    "Is the placement suspicious (e.g., hidden, unusual location)? "
                    "Describe the object's state and surrounding context."
                ),
            ),
            providers={
                "detector": detector,
                "segmenter": segmenter,
                "vlm": vlm,
            },
        )

        print(f"Detected {len(result['final'].instances)} suspicious objects:")
        print("=" * 60)
        for i, inst in enumerate(result["final"].instances, 1):
            print(f"\nObject {i}: {inst.label_name} (confidence: {inst.score:.2f})")
            print(f"  Location: {inst.bbox}")
            if hasattr(inst, "vlm_response"):
                print(f"  VLM Analysis: {inst.vlm_response}")
            elif "vlm_analysis" in result:
                print(f"  VLM Analysis: {result['vlm_analysis']}")

    else:
        print("=== Security: Suspicious Unattended Object Detection (Mock) ===")
        print()
        print("Graph: Detect → Filter → PromptBoxes → RefineMask → VLMQuery → Fuse")
        print("Models: GroundingDINO + SAM + Qwen3-VL (3-model chain)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → list of detected suspicious objects")
        print("  result['final'].masks → precise segmentation masks for each object")
        print("  result['final'].vlm_analysis → VLM reasoning about each object")
        print()
        print("Why 3 models?")
        print("  1. GroundingDINO: Zero-shot detection via text prompts")
        print("     - No training needed, just describe what to find")
        print("  2. SAM: Precise segmentation boundaries")
        print("     - Essential for distinguishing overlapping objects")
        print("  3. Qwen3-VLM: Contextual reasoning")
        print("     - Answers: Is it abandoned? Is owner nearby? Is it suspicious?")
        print("     - Reduces false positives through semantic understanding")
        print()
        print("Real-world deployment considerations:")
        print("  - Temporal analysis: Track if object remains unattended over time")
        print("  - Owner proximity detection: Use person detection + spatial proximity")
        print("  - Alert escalation: High-risk objects → immediate human review")
        print()

        # Verify preset construction
        from mata.presets import suspicious_object_detection

        graph = suspicious_object_detection()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
