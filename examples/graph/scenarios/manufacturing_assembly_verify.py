#!/usr/bin/env python3
"""Manufacturing: Assembly Verification with VLM.

Real-World Problem:
    Complex assemblies (e.g., electronics, automotive parts) require verification
    that all components are present and correctly positioned. Human inspection is
    slow and error-prone. Traditional object detection counts parts but misses
    subtle orientation errors or missing fasteners.

Solution:
    Two-pronged approach: VLM provides holistic semantic inspection (missing parts,
    incorrect orientation, assembly errors), while standard detection counts and
    localizes individual components. Combined results give both semantic and
    spatial analysis.

Models:
    - VLM: Qwen/Qwen3-VL-2B-Instruct (holistic reasoning)
    - Detector: facebook/detr-resnet-50 (component detection)

Graph Flow:
    Parallel(VLMQuery, Detect) → Filter → Fuse

Usage:
    python manufacturing_assembly_verify.py                    # Mock mode
    python manufacturing_assembly_verify.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import assembly_verification

        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        detector = mata.load("detect", "facebook/detr-resnet-50")

        result = mata.infer(
            image_path,
            assembly_verification(
                vlm_prompt=(
                    "Inspect this assembly. Are all components present and correctly positioned? "
                    "List any missing parts, incorrect orientations, or installation errors."
                ),
                detection_threshold=0.4,
            ),
            providers={"vlm": vlm, "detector": detector},
        )

        # Print VLM's holistic assessment
        if "vlm_assessment" in result and result["vlm_assessment"]:
            print("=== VLM Assessment ===")
            print(result["vlm_assessment"])
            print()

        # Print detected components
        if "final" in result and result["final"].instances:
            print(f"=== Detected Components ({len(result['final'].instances)}) ===")
            component_counts = {}
            for inst in result["final"].instances:
                component_counts[inst.label_name] = component_counts.get(inst.label_name, 0) + 1

            for component, count in sorted(component_counts.items()):
                print(f"  {component}: {count}")
    else:
        print("=== Manufacturing: Assembly Verification with VLM (Mock) ===")
        print()
        print("Graph: Parallel(VLMQuery, Detect) → Filter → Fuse")
        print("Models: Qwen3-VL (VLM) + DETR (detector)")
        print()
        print("Expected output structure:")
        print("  result['vlm_assessment'] → VLM's holistic inspection report")
        print("  result['final'].instances → detected components with counts")
        print()
        print("Example VLM output:")
        print('  "Assembly appears complete. All 4 screws present and torqued correctly."')
        print('  "Warning: Missing rubber gasket in upper-left corner."')
        print()

        # Verify preset construction
        from mata.presets import assembly_verification
        graph = assembly_verification()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
