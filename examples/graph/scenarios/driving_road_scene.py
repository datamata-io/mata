#!/usr/bin/env python3
"""Autonomous Driving: Comprehensive Road Scene Analysis.

Real-World Problem:
    Self-driving systems require multi-modal scene understanding: detecting
    individual objects (cars, pedestrians), segmenting road infrastructure
    (lanes, sidewalks, sky), estimating depth for spatial awareness, and
    classifying the overall scene type (urban, highway, intersection) for
    context-aware decision making.

Solution:
    Four parallel vision tasks provide comprehensive analysis:
    1. Object detection identifies traffic participants
    2. Panoptic segmentation maps road structure (stuff + things)
    3. Depth estimation provides 3D spatial context
    4. Zero-shot CLIP classification categorizes the scene type

    All results are fused into a unified scene representation.

Models:
    - Detector: facebook/detr-resnet-50
    - Segmenter: facebook/mask2former-swin-tiny-coco-panoptic
    - Depth: depth-anything/Depth-Anything-V2-Small-hf
    - Classifier: openai/clip-vit-base-patch32 (zero-shot)

Graph Flow:
    Parallel(Detect, SegmentImage, EstimateDepth, Classify) → Filter → Fuse

Usage:
    python driving_road_scene.py                    # Mock mode
    python driving_road_scene.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "road.jpg")

    if USE_REAL:
        import mata
        from mata.presets import road_scene_analysis

        detector = mata.load("detect", "facebook/detr-resnet-50")
        segmenter = mata.load("segment", "facebook/mask2former-swin-tiny-coco-panoptic")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        classifier = mata.load("classify", "openai/clip-vit-base-patch32")

        result = mata.infer(
            image_path,
            road_scene_analysis(
                detection_threshold=0.4,
                scene_labels=["urban_road", "highway", "rural_road", "intersection", "parking_lot"],
            ),
            providers={
                "detector": detector,
                "segmenter": segmenter,
                "depth": depth,
                "classifier": classifier,
            },
        )

        # Display comprehensive analysis
        print("=== Road Scene Analysis ===")
        print(f"\nScene Type: {result['final'].classifications[0].label} "
              f"(confidence: {result['final'].classifications[0].score:.2f})")

        print(f"\nDetected Objects: {len(result['final'].instances)}")
        for inst in result["final"].instances:
            print(f"  {inst.label_name}: {inst.score:.2f}")

        print(f"\nSegmentation: {len([i for i in result['final'].instances if i.mask is not None])} regions")
        print(f"Depth map shape: {result['final'].depth.shape}")

    else:
        print("=== Autonomous Driving: Comprehensive Road Scene Analysis (Mock) ===")
        print()
        print("Graph: Parallel(Detect, SegmentImage, EstimateDepth, Classify) → Filter → Fuse")
        print("Models: DETR + Mask2Former + Depth Anything + CLIP")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → detected objects")
        print("  result['final'].masks → panoptic segmentation (road, sidewalk, sky)")
        print("  result['final'].depth → depth map for spatial context")
        print("  result['final'].classifications → scene type classification")
        print()
        print("This is the most comprehensive driving preset with 4 parallel tasks.")
        print()

        # Verify preset construction
        from mata.presets import road_scene_analysis
        graph = road_scene_analysis()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
