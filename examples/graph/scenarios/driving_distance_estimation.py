#!/usr/bin/env python3
"""Autonomous Driving: Vehicle Distance Estimation.

Real-World Problem:
    Autonomous vehicles and ADAS systems need to detect nearby vehicles and
    pedestrians while simultaneously estimating their distance for collision
    avoidance, adaptive cruise control, and safe lane changes.

Solution:
    Combines object detection with monocular depth estimation. The detector
    identifies traffic participants (cars, trucks, pedestrians), and the depth
    model provides a pixel-wise distance map. Results are fused to correlate
    each detected object with its estimated distance.

Models:
    - Detector: facebook/detr-resnet-50 (multi-class object detection)
    - Depth: depth-anything/Depth-Anything-V2-Small-hf (monocular depth)

Graph Flow:
    Parallel(Detect, EstimateDepth) → Filter(vehicle classes) → Fuse

Usage:
    python driving_distance_estimation.py                    # Mock mode
    python driving_distance_estimation.py --real image.jpg   # Real inference
"""
from __future__ import annotations

import sys


def main():
    USE_REAL = "--real" in sys.argv
    image_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "street.jpg")

    if USE_REAL:
        import mata
        from mata.presets import vehicle_distance_estimation

        detector = mata.load("detect", "facebook/detr-resnet-50")
        depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")

        result = mata.infer(
            image_path,
            vehicle_distance_estimation(
                detection_threshold=0.5,
                vehicle_labels=["car", "truck", "bus", "person", "bicycle", "motorcycle"],
            ),
            providers={"detector": detector, "depth": depth},
        )

        print(f"Detected {len(result['final'].instances)} traffic participants:")
        for inst in result["final"].instances:
            bbox = inst.bbox
            # Calculate approximate distance from depth map at bbox center
            print(f"  {inst.label_name}: {inst.score:.2f} at {bbox}")

        # Depth map is available in result["final"].depth
        print(f"\nDepth map shape: {result['final'].depth.shape}")
        print("Distance analysis can correlate depth values with detected objects.")

    else:
        print("=== Autonomous Driving: Vehicle Distance Estimation (Mock) ===")
        print()
        print("Graph: Parallel(Detect, EstimateDepth) → Filter → Fuse")
        print("Models: DETR (detector) + Depth Anything (depth)")
        print()
        print("Expected output structure:")
        print("  result['final'].instances → detected vehicles/pedestrians with bboxes")
        print("  result['final'].depth → depth map (H, W) numpy array")
        print("  Distance correlation: sample depth values at bbox centers")
        print()

        # Verify preset construction
        from mata.presets import vehicle_distance_estimation
        graph = vehicle_distance_estimation()
        print(f"Graph '{graph.name}' constructed with {len(graph._nodes)} nodes")
        print(f"Node types: {[type(n).__name__ for n in graph._nodes]}")
        print()
        print("Run with --real <image.jpg> for actual inference.")


if __name__ == "__main__":
    main()
