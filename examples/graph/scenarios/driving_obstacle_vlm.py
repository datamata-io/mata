#!/usr/bin/env python3
"""Autonomous Driving: Obstacle Detection with VLM Reasoning.

Real-World Problem:
    Autonomous vehicles encounter unpredictable obstacles (debris, animals,
    construction equipment) that standard object detectors may miss or
    misclassify. VLMs can provide semantic understanding of unusual obstacles,
    assess road hazards, and make context-aware decisions beyond fixed
    category detection.

Solution:
    Combines three analysis streams for obstacle awareness:
    1. VLM describes the scene with focus on road hazards and obstacles
    2. Standard detector identifies known traffic participants
    3. Depth estimation provides spatial context for navigation

    This demonstrates REUSING the existing `vlm_scene_understanding()` preset
    with a domain-specific prompt rather than creating a new preset.

Models:
    - VLM: Qwen/Qwen3-VL-2B-Instruct (visual reasoning)
    - Detector: facebook/detr-resnet-50
    - Depth: depth-anything/Depth-Anything-V2-Small-hf

Graph Flow:
    Parallel(VLMDescribe, Detect, EstimateDepth) > Fuse
    (Reuses vlm_scene_understanding preset with custom prompt)

Usage:
    python driving_obstacle_vlm.py                    # Mock mode
    python driving_obstacle_vlm.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "road_obstacle.jpg")

    if USE_REAL:
        import mata
        from mata.presets import vlm_scene_understanding

        # Load models
        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        detector = mata.load("detect", "facebook/detr-resnet-50")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

        # REUSE vlm_scene_understanding with driving-specific prompt
        result = mata.infer(
            image_path,
            vlm_scene_understanding(
                describe_prompt=(
                    "Analyze this road scene for autonomous driving. Identify any obstacles, "
                    "road hazards, or unusual objects that could affect safe navigation. "
                    "Describe road conditions, visibility, and any potential safety concerns."
                ),
                detection_threshold=0.4,
            ),
            providers={"vlm": vlm, "detector": detector, "depth": depth},
        )

        # Display multi-modal analysis
        print("=== Obstacle Detection & Road Hazard Analysis ===")
        print("\nVLM Reasoning (Road Hazards & Obstacles):")
        if hasattr(result["scene"], "meta") and "text" in result["scene"].meta:
            print(f"  {result['scene'].meta['text']}")
        else:
            print("  Check result['scene'].description for VLM output")

        print(f"\nStandard Detections: {len(result['scene'].instances)}")
        for inst in result["scene"].instances:
            print(f"  {inst.label_name}: {inst.score:.2f} at {inst.bbox}")

        print(f"\nDepth map available: {result['scene'].depth.shape}")
        print("Combine VLM reasoning with depth for obstacle avoidance decisions.")

    else:
        print("=== Autonomous Driving: Obstacle Detection with VLM Reasoning (Mock) ===")
        print()
        print("Graph: Parallel(VLMDescribe, Detect, EstimateDepth) > Fuse")
        print("Models: Qwen3-VL (reasoning) + DETR (detection) + Depth Anything")
        print()
        print("Key Design Pattern: REUSING EXISTING PRESET")
        print("  This example demonstrates reusing `vlm_scene_understanding()`")
        print("  with a custom driving-specific prompt, rather than creating")
        print("  a new preset function. This is the recommended approach when")
        print("  only the prompt differs.")
        print()
        print("Expected output structure:")
        print("  result['scene'].description > VLM's road hazard analysis")
        print("  result['scene'].instances > detected objects")
        print("  result['scene'].depth > depth map for spatial awareness")
        print()
        print("VLM Capabilities for Driving:")
        print("  • Identify unusual obstacles (debris, animals, construction)")
        print("  • Assess road conditions (wet, icy, damaged)")
        print("  • Detect hazards outside standard detection categories")
        print("  • Provide context-aware reasoning for decision making")
        print()

        # Verify preset construction with custom prompt
        from mata.presets import vlm_scene_understanding
        graph = vlm_scene_understanding(
            describe_prompt=(
                "Analyze this road scene for autonomous driving. Identify any obstacles, "
                "road hazards, or unusual objects that could affect safe navigation."
            ),
            detection_threshold=0.4,
        )
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
