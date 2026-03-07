#!/usr/bin/env python3
"""Medical: Automated Image Report Generation.

⚠️ DISCLAIMER: This example is for research and demonstration purposes
only. It is NOT intended for clinical diagnosis, treatment decisions,
or any medical application. Always consult qualified medical professionals
for clinical decisions. This tool has not been validated for clinical use.

Real-World Research Problem:
    Medical imaging research benefits from automated preliminary image
    descriptions that highlight potentially interesting features for
    further expert review. This can help researchers quickly triage
    large image datasets.

Solution:
    Multi-modal analysis combining VLM scene understanding, object detection,
    and depth estimation provides comprehensive image description. The VLM
    generates natural language descriptions while detection and depth provide
    structured spatial information for research documentation.

Models:
    - VLM: Qwen/Qwen3-VL-2B (vision-language understanding)
    - Detector: facebook/detr-resnet-50 (object detection)
    - Depth: depth-anything/Depth-Anything-V2-Small-hf (depth estimation)

Graph Flow:
    Parallel(VLMDescribe, Detect, EstimateDepth) > Filter > Fuse

Usage:
    python medical_report_generation.py                    # Mock mode
    python medical_report_generation.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import vlm_scene_understanding

        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B")
        detector = mata.load("detect", "facebook/detr-resnet-50")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

        # Reuse vlm_scene_understanding with medical-specific prompt
        result = mata.infer(
            image_path,
            vlm_scene_understanding(
                describe_prompt=(
                    "Describe any abnormalities, lesions, or notable features "
                    "in this medical image. Note colors, shapes, locations, and "
                    "any asymmetries or unusual patterns."
                ),
                detection_threshold=0.3,
            ),
            providers={"vlm": vlm, "detector": detector, "depth": depth},
        )

        print("⚠️  RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        print("\n=== Medical Image Analysis Report (Research Demo) ===")
        print()
        
        # VLM description
        if "description" in result:
            print("VLM Description:")
            print(f"  {result['description']}")
            print()
        
        # Detection results
        if "final" in result and hasattr(result["final"], "instances"):
            print(f"Detected {len(result['final'].instances)} objects:")
            for inst in result["final"].instances[:5]:  # Show top 5
                print(f"  - {inst.label_name}: {inst.score:.2f} at {inst.bbox}")
            if len(result["final"].instances) > 5:
                print(f"  ... and {len(result['final'].instances) - 5} more")
            print()
        
        # Depth information
        if "depth" in result:
            print(f"Depth map generated: {result['depth'].depth_map.shape}")
            print()
        
        print("⚠️  This output is for research demonstration only.")
    else:
        print("=== Medical: Automated Report Generation (Mock) ===")
        print()
        print("⚠️  DISCLAIMER: Research and demonstration purposes only.")
        print("    NOT for clinical diagnosis or treatment decisions.")
        print()
        print("Graph: Parallel(VLMDescribe, Detect, EstimateDepth) > Filter > Fuse")
        print("Models: Qwen3-VL + DETR + Depth Anything")
        print("VLM prompt: Medical abnormality description")
        print()
        print("Expected output structure:")
        print("  result['description'] > VLM natural language description")
        print("  result['final'].instances > detected objects with bboxes")
        print("  result['depth'] > depth map for spatial understanding")
        print()
        print("This preset demonstrates reusing vlm_scene_understanding()")
        print("with domain-specific medical prompts for research applications.")
        print()

        # Verify preset construction
        from mata.presets import vlm_scene_understanding
        graph = vlm_scene_understanding(
            describe_prompt="Describe any abnormalities in this medical image."
        )
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
