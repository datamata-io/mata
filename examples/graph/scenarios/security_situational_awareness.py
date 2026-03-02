#!/usr/bin/env python3
"""Security: Comprehensive Situational Awareness Analysis.

Real-World Problem:
    Security operations centers need holistic scene understanding beyond simple
    object detection. Operators must answer questions like: "Are there any
    security concerns?", "Is anyone behaving unusually?", "Are there restricted
    area violations?", "What's the overall security posture of this scene?"

    Traditional detection systems can count people or cars, but cannot provide
    semantic security assessment or detect subtle anomalies.

Solution:
    Reuse the `vlm_scene_understanding()` preset with security-specific prompts.
    This demonstrates MATA's preset reusability — the same multi-modal graph
    (VLM + detection + depth) can be repurposed for different domains simply
    by changing the VLM prompt.

    The system provides three complementary analysis channels:
    - VLM semantic assessment focused on security concerns
    - Object detection for crowd/vehicle counting
    - Depth estimation for spatial awareness and distance assessment

Models:
    - VLM: Qwen/Qwen3-VL-2B-Instruct (security-focused scene description)
    - Detector: facebook/detr-resnet-50 (multi-class object detection)
    - Depth: depth-anything/Depth-Anything-V2-Small-hf (spatial understanding)

Graph Flow:
    Parallel(VLMDescribe, Detect, EstimateDepth) → Fuse

Usage:
    python security_situational_awareness.py                    # Mock mode
    python security_situational_awareness.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "security_scene.jpg")

    if USE_REAL:
        import mata
        from mata.presets import vlm_scene_understanding

        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        detector = mata.load("detect", "facebook/detr-resnet-50")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

        # Security-specific VLM prompt
        security_prompt = (
            "Analyze this scene from a security perspective. "
            "Describe any security concerns, unusual behavior, or restricted area violations. "
            "Note crowd density, unattended objects, unauthorized access, or suspicious activities. "
            "If the scene appears normal, confirm this explicitly."
        )

        result = mata.infer(
            image_path,
            vlm_scene_understanding(describe_prompt=security_prompt, detection_threshold=0.4),
            providers={
                "vlm": vlm,
                "detector": detector,
                "depth": depth,
            },
        )

        print("=== Comprehensive Situational Awareness Report ===")
        print()
        print("VLM Security Assessment:")
        print("-" * 60)
        if "scene" in result and hasattr(result["scene"], "description"):
            print(result["scene"].description)
        elif "description" in result:
            print(result["description"])
        print()

        print("Object Detection Summary:")
        print("-" * 60)
        if "scene" in result and hasattr(result["scene"], "instances"):
            detections = result["scene"].instances
        elif "dets" in result:
            detections = result["dets"].instances
        else:
            detections = []

        print(f"Total objects detected: {len(detections)}")
        # Count by category
        from collections import Counter

        categories = Counter(inst.label_name for inst in detections)
        for label, count in categories.most_common():
            print(f"  {label}: {count}")
        print()

        print("Depth Analysis:")
        print("-" * 60)
        if "scene" in result and hasattr(result["scene"], "depth_map"):
            depth_map = result["scene"].depth_map
        elif "depth" in result and hasattr(result["depth"], "depth_map"):
            depth_map = result["depth"].depth_map
        else:
            depth_map = None

        if depth_map is not None:
            import numpy as np

            print(f"Depth map shape: {depth_map.shape}")
            print(f"Depth range: {np.min(depth_map):.2f} to {np.max(depth_map):.2f}")
            print("(Can be used to assess distances to detected objects)")

    else:
        print("=== Security: Comprehensive Situational Awareness (Mock) ===")
        print()
        print("Graph: Parallel(VLMDescribe, Detect, EstimateDepth) → Fuse")
        print("Models: Qwen3-VLM + DETR + Depth-Anything")
        print()
        print("Key Design Pattern: PRESET REUSABILITY")
        print("-" * 60)
        print("This example reuses vlm_scene_understanding() — a general-purpose")
        print("preset originally designed for broad scene analysis. By simply")
        print("changing the VLM prompt to a security-focused one, the same graph")
        print("architecture now provides security-specific insights.")
        print()
        print("Expected output structure:")
        print("  result['scene'].description   → VLM's security assessment")
        print("  result['scene'].instances      → Detected objects (people, vehicles, etc.)")
        print("  result['scene'].depth_map      → Spatial depth information")
        print()
        print("Security use cases:")
        print("  - Perimeter security: Detect intrusions, unusual activity")
        print("  - Access control: Identify unauthorized personnel/vehicles")
        print("  - Crowd management: Assess density and movement patterns")
        print("  - Incident investigation: Post-event analysis of security footage")
        print("  - Restricted areas: Detect violations of access policies")
        print()
        print("Benefits of multi-modal analysis:")
        print("  - VLM provides semantic understanding (unusual vs. normal behavior)")
        print("  - Detection provides quantitative metrics (person count, object types)")
        print("  - Depth enables distance assessment (how far is that person from the fence?)")
        print()

        # Verify preset construction
        from mata.presets import vlm_scene_understanding

        graph = vlm_scene_understanding()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
