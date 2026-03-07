#!/usr/bin/env python3
"""Agriculture: Aerial Crop Field Analysis.

Real-World Problem:
    Precision agriculture requires detailed understanding of crop distribution
    and terrain topology from aerial imagery (drones/satellites). Farmers need
    to identify crop regions, measure coverage area, and understand terrain
    elevation for optimized irrigation, fertilization, and pesticide application.

Solution:
    Mask2Former performs panoptic segmentation to identify different crop
    regions and field boundaries. Simultaneously, Depth Anything estimates
    terrain depth/elevation. Combined results enable crop coverage analysis
    and terrain-aware resource planning.

Models:
    - Segmenter: facebook/mask2former-swin-large-coco-panoptic (panoptic segmentation)
    - Depth: depth-anything/Depth-Anything-V2-Small-hf (monocular depth estimation)

Graph Flow:
    Parallel(SegmentImage, EstimateDepth) > Fuse

Usage:
    python agriculture_aerial_crop.py                    # Mock mode
    python agriculture_aerial_crop.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "sample.jpg")

    if USE_REAL:
        import mata
        from mata.presets import aerial_crop_analysis

        segmenter = mata.load("segment", "facebook/mask2former-swin-large-coco-panoptic")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

        result = mata.infer(
            image_path,
            aerial_crop_analysis(),
            providers={"segmenter": segmenter, "depth": depth},
        )

        print(f"Segmented {len(result['final'].instances)} crop regions")
        print(f"Depth map shape: {result['final'].depth.shape}")
        print()
        print("Analysis results:")
        for inst in result["final"].instances:
            print(f"  Region {inst.label_name}: {inst.score:.2f}, area: {inst.bbox}")
    else:
        print("=== Agriculture: Aerial Crop Field Analysis (Mock) ===")
        print()
        print("Real-World Problem:")
        print("  Precision agriculture needs crop distribution + terrain topology")
        print("  from aerial imagery for optimized resource planning.")
        print()
        print("Graph: Parallel(SegmentImage, EstimateDepth) > Fuse")
        print("Models: Mask2Former (segmenter) + Depth Anything (depth)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances > segmented crop regions with masks")
        print("  result['final'].depth > terrain depth map (H, W) array")
        print("  Enables: crop coverage area, terrain-aware irrigation/spraying")
        print()

        # Verify preset construction
        from mata.presets import aerial_crop_analysis
        graph = aerial_crop_analysis()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
