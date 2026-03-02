#!/usr/bin/env python3
"""Medical: Region of Interest Segmentation & Area Measurement.

⚠️ DISCLAIMER: This example is for research and demonstration purposes
only. It is NOT intended for clinical diagnosis, treatment decisions,
or any medical application. Always consult qualified medical professionals
for clinical decisions. This tool has not been validated for clinical use.

Real-World Research Problem:
    Medical imaging research requires automated detection and segmentation
    of regions of interest (ROIs) such as lesions, nodules, or masses, with
    precise area/volume measurements for tracking disease progression or
    treatment response in research studies.

Solution:
    GroundingDINO detects medical ROIs using text prompts (zero-shot),
    then SAM segments each region with pixel-level precision. Mask area
    calculation enables quantitative analysis for research purposes.

Models:
    - Detector: IDEA-Research/grounding-dino-tiny (zero-shot text-prompted)
    - Segmenter: facebook/sam-vit-base (prompt-based segmentation)

Graph Flow:
    Detect("lesion . nodule . mass") → Filter → PromptBoxes(SAM) → RefineMask → Fuse

Usage:
    python medical_roi_segmentation.py                    # Mock mode
    python medical_roi_segmentation.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.core.graph.graph import Graph
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.prompt_boxes import PromptBoxes
        from mata.nodes.refine_mask import RefineMask
        from mata.nodes.fuse import Fuse

        detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        segmenter = mata.load("segment", "facebook/sam-vit-base")

        # Build custom graph with medical text prompts
        graph = (
            Graph("medical_roi_segmentation")
            .then(Detect(
                using="detector",
                out="dets",
                text_prompts="lesion . nodule . mass . abnormality"
            ))
            .then(Filter(src="dets", score_gt=0.25, out="filtered"))
            .then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))
            .then(RefineMask(src="masks", method="morph_close", radius=3, out="masks_ref"))
            .then(Fuse(out="final", dets="filtered", masks="masks_ref"))
        )

        result = mata.infer(
            image_path,
            graph,
            providers={"detector": detector, "segmenter": segmenter},
        )

        print(f"⚠️  RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        print(f"\nSegmented {len(result['final'].instances)} ROIs:")
        for i, inst in enumerate(result["final"].instances, 1):
            if inst.mask is not None:
                # Calculate approximate area from bbox (width × height)
                x1, y1, x2, y2 = inst.bbox
                area_pixels = (x2 - x1) * (y2 - y1)
                print(f"  ROI {i} ({inst.label_name}): confidence {inst.score:.2f}, area ≈ {area_pixels:.0f} px²")
            else:
                print(f"  ROI {i} ({inst.label_name}): confidence {inst.score:.2f} at {inst.bbox}")
        
        print(f"\n⚠️  This output is for research demonstration only.")
    else:
        print("=== Medical: ROI Segmentation & Area Measurement (Mock) ===")
        print()
        print("⚠️  DISCLAIMER: Research and demonstration purposes only.")
        print("    NOT for clinical diagnosis or treatment decisions.")
        print()
        print("Graph: Detect → Filter → PromptBoxes(SAM) → RefineMask → Fuse")
        print("Models: GroundingDINO (detector) + SAM (segmenter)")
        print("Text prompts: 'lesion . nodule . mass . abnormality'")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → list of ROI instances")
        print("  Each instance has:")
        print("    - bbox: bounding box derived from segmentation mask")
        print("    - mask: pixel-precise segmentation mask (RLE format)")
        print("    - score: detection confidence")
        print("  Area calculation: (x2-x1) × (y2-y1) from bbox")
        print()

        # Verify graph construction
        from mata.core.graph.graph import Graph
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.prompt_boxes import PromptBoxes
        from mata.nodes.refine_mask import RefineMask
        from mata.nodes.fuse import Fuse
        
        graph = (
            Graph("medical_roi_segmentation")
            .then(Detect(using="detector", out="dets", text_prompts="lesion . nodule . mass"))
            .then(Filter(src="dets", score_gt=0.25, out="filtered"))
            .then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))
            .then(RefineMask(src="masks", method="morph_close", radius=3, out="masks_ref"))
            .then(Fuse(out="final", dets="filtered", masks="masks_ref"))
        )
        
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
