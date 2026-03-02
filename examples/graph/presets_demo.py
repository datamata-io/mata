#!/usr/bin/env python3
"""Using pre-built graph presets for common workflows.

Demonstrates:
1. All 8 built-in presets (5 traditional CV + 3 VLM)
2. Customizing preset parameters
3. Executing presets with `mata.infer()`

Usage:
    python examples/graph/presets_demo.py
    python examples/graph/presets_demo.py --real
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create a full set of mock providers for all preset workflows."""
    from unittest.mock import Mock

    import numpy as np

    from mata.core.types import Classification, ClassifyResult, DepthResult, Instance, VisionResult

    # Detector
    mock_detector = Mock()
    mock_detector.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(100, 50, 300, 400), label=0, score=0.92, label_name="person"),
            Instance(bbox=(350, 100, 500, 350), label=1, score=0.85, label_name="car"),
        ],
        meta={"model": "mock-detr"},
    ))

    # Segmenter
    mock_segmenter = Mock()
    mock_segmenter.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(
                bbox=(100, 50, 300, 400), label=0, score=0.92,
                label_name="person",
                mask=np.ones((350, 200), dtype=np.uint8),
            ),
        ],
        meta={"model": "mock-sam"},
    ))

    # Classifier
    mock_classifier = Mock()
    mock_classifier.predict = Mock(return_value=ClassifyResult(
        classifications=[
            Classification(label="outdoor", score=0.82),
            Classification(label="indoor", score=0.18),
        ],
        meta={"model": "mock-clip"},
    ))

    # Depth estimator
    mock_depth = Mock()
    mock_depth.predict = Mock(return_value=DepthResult(
        depth_map=np.random.rand(480, 640).astype(np.float32),
        meta={"model": "mock-depth"},
    ))

    # Tracker
    mock_tracker = Mock()

    # VLM
    mock_vlm = Mock()
    mock_vlm.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(100, 50, 300, 400), label=0, score=0.88, label_name="person"),
        ],
        meta={"text": "A person standing next to a car.", "model": "mock-vlm"},
    ))

    return {
        "detector": mock_detector,
        "segmenter": mock_segmenter,
        "classifier": mock_classifier,
        "depth_estimator": mock_depth,
        "tracker": mock_tracker,
        "vlm": mock_vlm,
    }


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main():
    """Demonstrate all MATA graph presets."""
    from mata.presets import (
        detect_and_track,
        detection_pose,
        full_scene_analysis,
        # Traditional CV presets
        grounding_dino_sam,
        segment_and_refine,
        # VLM presets
        vlm_grounded_detection,
        vlm_multi_image_comparison,
        vlm_scene_understanding,
    )

    providers = create_mock_providers()

    # -----------------------------------------------------------------------
    # Preset 1: GroundingDINO + SAM (Detection + Segmentation)
    # -----------------------------------------------------------------------
    print("=== Preset 1: grounding_dino_sam ===")
    g1 = grounding_dino_sam(
        detection_threshold=0.3,
        nms_iou_threshold=0.5,         # Optional NMS
        refine_method="morph_close",    # Morphological refinement
        refine_radius=3,
    )
    print(f"  Graph: {g1.name}, Nodes: {len(g1._nodes)}")
    print("  Usage: detector + segmenter providers")
    print("  Flow: Detect → Filter → [NMS] → PromptBoxes → RefineMask → Fuse")

    # -----------------------------------------------------------------------
    # Preset 2: Segment and Refine (Segment Everything)
    # -----------------------------------------------------------------------
    print("\n=== Preset 2: segment_and_refine ===")
    g2 = segment_and_refine()
    print(f"  Graph: {g2.name}, Nodes: {len(g2._nodes)}")
    print("  Usage: segmenter provider")
    print("  Flow: SegmentEverything → RefineMask → Fuse")

    # -----------------------------------------------------------------------
    # Preset 3: Detection + Pose
    # -----------------------------------------------------------------------
    print("\n=== Preset 3: detection_pose ===")
    g3 = detection_pose(
        detection_threshold=0.5,
        nms_iou_threshold=0.5,
        top_k=10,
    )
    print(f"  Graph: {g3.name}, Nodes: {len(g3._nodes)}")
    print("  Usage: detector provider")
    print("  Flow: Detect → Filter → [NMS] → [TopK] → Fuse")

    # -----------------------------------------------------------------------
    # Preset 4: Full Scene Analysis (Parallel)
    # -----------------------------------------------------------------------
    print("\n=== Preset 4: full_scene_analysis ===")
    g4 = full_scene_analysis(
        classification_labels=["indoor", "outdoor", "urban", "nature"],
    )
    print(f"  Graph: {g4.name}, Nodes: {len(g4._nodes)}")
    print("  Usage: detector + classifier + depth_estimator providers")
    print("  Flow: parallel(Detect, Classify, EstimateDepth) → Filter → Fuse")

    # -----------------------------------------------------------------------
    # Preset 5: Detection + Tracking
    # -----------------------------------------------------------------------
    print("\n=== Preset 5: detect_and_track ===")
    g5 = detect_and_track(
        detection_threshold=0.5,
        track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.8,
    )
    print(f"  Graph: {g5.name}, Nodes: {len(g5._nodes)}")
    print("  Usage: detector + tracker providers")
    print("  Flow: Detect → Filter → Track → Fuse")

    # -----------------------------------------------------------------------
    # Preset 6: VLM Grounded Detection
    # -----------------------------------------------------------------------
    print("\n=== Preset 6: vlm_grounded_detection ===")
    g6 = vlm_grounded_detection()
    print(f"  Graph: {g6.name}, Nodes: {len(g6._nodes)}")
    print("  Usage: vlm + detector providers")
    print("  Flow: parallel(VLMDetect, Detect) → Filter → PromoteEntities → Fuse")

    # -----------------------------------------------------------------------
    # Preset 7: VLM Scene Understanding
    # -----------------------------------------------------------------------
    print("\n=== Preset 7: vlm_scene_understanding ===")
    g7 = vlm_scene_understanding()
    print(f"  Graph: {g7.name}, Nodes: {len(g7._nodes)}")
    print("  Usage: vlm + detector + depth_estimator providers")
    print("  Flow: parallel(VLMDescribe, Detect, EstimateDepth) → Fuse")

    # -----------------------------------------------------------------------
    # Preset 8: VLM Multi-Image Comparison
    # -----------------------------------------------------------------------
    print("\n=== Preset 8: vlm_multi_image_comparison ===")
    g8 = vlm_multi_image_comparison()
    print(f"  Graph: {g8.name}, Nodes: {len(g8._nodes)}")
    print("  Usage: vlm provider")
    print("  Flow: VLMQuery → Fuse")

    # -----------------------------------------------------------------------
    # Execute a preset with mata.infer()
    # -----------------------------------------------------------------------
    print("\n=== Executing a Preset ===")

    if "--real" not in sys.argv:
        import mata

        result = mata.infer(
            image="examples/images/000000039769.jpg",
            graph=g4,  # full_scene_analysis preset
            providers=providers,
        )

        print("full_scene_analysis result:")
        print(f"  Channels: {list(result.channels.keys())}")
        if result.has_channel("detections"):
            dets = result.get_channel("detections")
            print(f"  Detections: {len(dets.instances)} objects")
    else:
        print("  (Use --real flag with actual model providers)")

    # -----------------------------------------------------------------------
    # Quick reference
    # -----------------------------------------------------------------------
    print("\n=== Quick Reference ===")
    print("""
    from mata.presets import grounding_dino_sam, full_scene_analysis
    import mata

    # Load models
    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
    segmenter = mata.load("segment", "facebook/sam-vit-base")

    # Run preset
    result = mata.infer(
        "image.jpg",
        grounding_dino_sam(detection_threshold=0.3),
        providers={"detector": detector, "segmenter": segmenter},
    )

    # Access results
    result.detections   # Bounding boxes
    result.masks        # Segmentation masks
    """)

    print("✓ Presets demo complete!")


if __name__ == "__main__":
    main()
