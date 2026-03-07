#!/usr/bin/env python3
"""Manufacturing: Per-Component Detailed Inspection.

Real-World Problem:
    In precision manufacturing (e.g., electronics, aerospace), each component
    needs individual close inspection for microscopic defects, wear patterns,
    or contamination that affect the entire assembly. Inspecting a single image
    holistically misses component-specific issues.

Solution:
    Detect all components, crop each one as a separate image, then use a VLM
    to inspect each crop individually with detailed prompts. This enables
    per-component defect analysis, wear assessment, and quality reporting.

Models:
    - Detector: facebook/detr-resnet-50 (component localization)
    - VLM: Qwen/Qwen3-VL-2B-Instruct (detailed per-component inspection)

Graph Flow:
    Detect > Filter > ExtractROIs > VLMQuery > Fuse

Usage:
    python manufacturing_component_inspect.py                    # Mock mode
    python manufacturing_component_inspect.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import component_inspection

        detector = mata.load("detect", "facebook/detr-resnet-50")
        vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

        result = mata.infer(
            image_path,
            component_inspection(
                detection_threshold=0.5,
                vlm_prompt=(
                    "Inspect this component closely. Describe any visible defects, "
                    "wear, scratches, corrosion, contamination, or anomalies. "
                    "Rate the component condition as: Excellent / Good / Fair / Poor."
                ),
                top_k=5,  # Only inspect top 5 most confident detections
            ),
            providers={"detector": detector, "vlm": vlm},
        )

        # Print per-component inspection reports
        if "final" in result and result["final"].instances:
            print(f"=== Component Inspection Report ({len(result['final'].instances)} components) ===")
            print()

            # Note: VLM responses are typically stored in result metadata
            # or could be fused into instances. Implementation may vary.
            for i, inst in enumerate(result["final"].instances, 1):
                print(f"Component {i}: {inst.label_name} (confidence: {inst.score:.2f})")
                print(f"  Location: {inst.bbox}")
                # VLM inspection result would be accessed here
                # print(f"  Inspection: {inst.inspection_report}")
                print()
    else:
        print("=== Manufacturing: Per-Component Detailed Inspection (Mock) ===")
        print()
        print("Graph: Detect > Filter > ExtractROIs > VLMQuery > Fuse")
        print("Models: DETR (detector) + Qwen3-VL (VLM)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > detected components")
        print("  result['final'].rois > cropped images of each component")
        print("  Each component gets individual VLM inspection report:")
        print('    "Component shows minor surface wear on upper edge. No critical defects."')
        print('    "Excellent condition. No visible defects or contamination."')
        print()
        print("Common use cases:")
        print("  - PCB component inspection for solder defects")
        print("  - Bearing surface wear assessment")
        print("  - Connector pin contamination detection")
        print("  - Fastener thread damage inspection")
        print()

        # Verify preset construction
        from mata.presets import component_inspection
        graph = component_inspection(top_k=5)
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
