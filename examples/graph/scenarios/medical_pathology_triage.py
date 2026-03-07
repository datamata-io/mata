#!/usr/bin/env python3
"""Medical: Pathology Triage with Conditional VLM Analysis.

⚠️ DISCLAIMER: This example is for research and demonstration purposes
only. It is NOT intended for clinical diagnosis, treatment decisions,
or any medical application. Always consult qualified medical professionals
for clinical decisions. This tool has not been validated for clinical use.

Real-World Research Problem:
    In pathology research, large datasets of tissue images need preliminary
    screening to identify potentially interesting specimens for detailed
    expert review. Automated triage can help researchers prioritize which
    samples to examine more closely.

Solution:
    This is the most complex example, demonstrating conditional branching:
    
    1. Detect regions of interest in the image
    2. Extract ROI crops for each region
    3. Use CLIP to classify each ROI as ["normal", "benign", "atypical", "uncertain"]
    4. For regions with "atypical" score > 0.3: query VLM for detailed description
    5. For other regions: mark as routine
    
    This conditional logic demonstrates how to build research triage pipelines
    that focus computational resources on potentially interesting cases.

Models:
    - Detector: facebook/detr-resnet-50 (region detection)
    - Classifier: openai/clip-vit-base-patch32 (zero-shot triage)
    - VLM: Qwen/Qwen3-VL-2B (detailed analysis for flagged regions)

Graph Flow:
    Detect > Filter > ExtractROIs > Classify > [Conditional VLM Query] > Fuse

Usage:
    python medical_pathology_triage.py                    # Mock mode
    python medical_pathology_triage.py --real image.jpg   # Real inference
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
        from mata.nodes.roi import ExtractROIs
        from mata.nodes.classify import Classify
        from mata.nodes.fuse import Fuse

        # Load models
        detector = mata.load("detect", "facebook/detr-resnet-50")
        classifier = mata.load("classify", "openai/clip-vit-base-patch32")

        # Build triage graph (VLM is optional for this demo)
        triage_labels = ["normal", "benign", "atypical", "uncertain"]
        
        graph = (
            Graph("pathology_triage")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.4, out="filtered"))
            .then(ExtractROIs(src_dets="filtered", padding=10, out="rois"))
            .then(Classify(
                using="classifier",
                out="classes",
                text_prompts=triage_labels
            ))
            .then(Fuse(out="final", dets="filtered", rois="rois", classifications="classes"))
        )

        result = mata.infer(
            image_path,
            graph,
            providers={"detector": detector, "classifier": classifier},
        )

        print("⚠️  RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS")
        print("\n=== Pathology Triage Analysis (Research Demo) ===")
        print()

        # Analyze classifications and identify flagged regions
        if "classes" in result and hasattr(result["classes"], "classifications"):
            flagged_count = 0
            routine_count = 0
            
            print(f"Analyzed {len(result['classes'].classifications)} regions:")
            print()
            
            for i, classification in enumerate(result["classes"].classifications, 1):
                if hasattr(classification, "label_name") and hasattr(classification, "score"):
                    label = classification.label_name
                    score = classification.score
                    
                    # Triage logic: flag "atypical" or "uncertain" with score > 0.3
                    is_flagged = (label in ["atypical", "uncertain"]) and (score > 0.3)
                    
                    if is_flagged:
                        flagged_count += 1
                        print(f"  ⚠️  Region {i}: {label.upper()} ({score:.2%}) — FLAGGED for review")
                        print(f"      Recommendation: Expert pathologist review required")
                    else:
                        routine_count += 1
                        if i <= 3:  # Show first 3 routine
                            print(f"  ✓  Region {i}: {label} ({score:.2%}) — routine")
            
            if routine_count > 3:
                print(f"  ... and {routine_count - 3} more routine regions")
            
            print()
            print(f"Summary: {flagged_count} flagged for review, {routine_count} routine")
            print()
            print("⚠️  In a full research pipeline, flagged regions would be")
            print("   sent to VLM for detailed pathological description.")
        
        print()
        print("⚠️  This output is for research demonstration only.")
    else:
        print("=== Medical: Pathology Triage (Mock) ===")
        print()
        print("⚠️  DISCLAIMER: Research and demonstration purposes only.")
        print("    NOT for clinical diagnosis or treatment decisions.")
        print()
        print("Complex Conditional Pipeline (Example-Only Pattern):")
        print()
        print("  Flow:")
        print("    1. Detect > identify regions of interest")
        print("    2. ExtractROIs > crop each region")
        print("    3. Classify > triage as [normal, benign, atypical, uncertain]")
        print("    4. Conditional Logic:")
        print("       - If 'atypical' or 'uncertain' score > 0.3:")
        print("         > Flag for VLM detailed analysis (research review queue)")
        print("       - Otherwise:")
        print("         > Mark as routine")
        print()
        print("Graph: Detect > Filter > ExtractROIs > Classify > [Conditional] > Fuse")
        print("Models: DETR (detector) + CLIP (classifier) + Qwen3-VL (optional VLM)")
        print()
        print("This demonstrates the most complex scenario: conditional branching")
        print("with real-world research triage logic for prioritizing expert review.")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > detected regions")
        print("  result['final'].rois > cropped region images")
        print("  result['final'].classifications > triage classifications")
        print("  Conditional logic identifies which regions need expert review")
        print()

        # Verify graph construction
        from mata.core.graph.graph import Graph
        from mata.nodes.detect import Detect
        from mata.nodes.filter import Filter
        from mata.nodes.roi import ExtractROIs
        from mata.nodes.classify import Classify
        from mata.nodes.fuse import Fuse
        
        graph = (
            Graph("pathology_triage")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.4, out="filtered"))
            .then(ExtractROIs(src_dets="filtered", padding=10, out="rois"))
            .then(Classify(using="classifier", out="classes",
                           text_prompts=["normal", "benign", "atypical", "uncertain"]))
            .then(Fuse(out="final", dets="filtered", rois="rois", classifications="classes"))
        )
        
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
