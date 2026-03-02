#!/usr/bin/env python3
"""Parallel execution of independent tasks: Detect + Classify + Depth.

Demonstrates:
1. Running independent task nodes in parallel for 1.5–3× speedup
2. Using `Graph.parallel()` to create parallel stages
3. Using `ParallelScheduler` for multi-threaded execution
4. Fusing results from all tasks into a single `MultiResult`

Usage:
    python examples/graph/parallel_tasks.py
    python examples/graph/parallel_tasks.py --real
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Setup paths
IMAGE_DIR = Path(__file__).parent.parent / "images"
IMAGE_1 = IMAGE_DIR / "000000039769.jpg"
IMAGE_2 = IMAGE_DIR / "hanvin-cheong-tuR2XRPdtYI-unsplash.jpg"

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create mock detector, classifier, and depth estimator."""
    from unittest.mock import Mock

    import numpy as np

    from mata.core.types import Classification, ClassifyResult, Instance, VisionResult

    # Mock detector — returns bounding boxes
    mock_detector = Mock()
    mock_detector.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(100, 50, 300, 400), label=0, score=0.91, label_name="person"),
            Instance(bbox=(350, 100, 500, 350), label=1, score=0.85, label_name="car"),
        ],
        meta={"model": "mock-detr"},
    ))

    # Mock classifier — returns classifications
    mock_classifier = Mock()
    mock_classifier.predict = Mock(return_value=ClassifyResult(
        predictions=[
            Classification(label="outdoor", score=0.88),
            Classification(label="indoor", score=0.12),
        ],
        meta={"model": "mock-clip"},
    ))
    # Classify node calls .classify() not .predict()
    mock_classifier.classify = mock_classifier.predict

    # Mock depth estimator — returns a depth map
    mock_depth = Mock()
    depth_result = Mock()
    depth_result.depth_map = np.random.rand(480, 640).astype(np.float32)
    depth_result.meta = {"model": "mock-depth-anything"}
    mock_depth.predict = Mock(return_value=depth_result)
    # EstimateDepth node calls .estimate() not .predict()
    mock_depth.estimate = mock_depth.predict

    return {
        "detector": mock_detector,
        "classifier": mock_classifier,
        "depth_estimator": mock_depth,
    }


def create_real_providers():
    """Create real providers from HuggingFace."""
    import mata

    return {
        "detector": mata.load("detect", "facebook/detr-resnet-50"),
        "classifier": mata.load("classify", "openai/clip-vit-base-patch32"),
        "depth_estimator": mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf"),
    }


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main():
    """Run parallel detection + classification + depth estimation."""
    import mata
    from mata.core.graph import Graph, ParallelScheduler
    from mata.nodes import Classify, Detect, EstimateDepth, Filter, Fuse

    use_real = "--real" in sys.argv
    if use_real:
        print("Loading real models from HuggingFace...")
        providers = create_real_providers()
    else:
        print("Running with mock providers (use --real for actual models)")
        providers = create_mock_providers()

    # -----------------------------------------------------------------------
    # Build graph with parallel stage
    # -----------------------------------------------------------------------
    graph = (
        Graph("full_scene_analysis")
        # Three independent tasks run simultaneously
        .parallel([
            Detect(using="detector", out="dets"),
            Classify(
                using="classifier",
                out="classifications",
                text_prompts=["indoor", "outdoor", "urban", "nature"],
            ),
            EstimateDepth(using="depth_estimator", out="depth"),
        ])
        # Post-process detections
        .then(Filter(src="dets", score_gt=0.3, out="filtered"))
        # Bundle everything into a final result
        .then(Fuse(
            detections="filtered",
            classifications="classifications",
            depth="depth",
            out="scene",
        ))
    )

    # -----------------------------------------------------------------------
    # Execute with ParallelScheduler for multi-threaded speedup
    # -----------------------------------------------------------------------
    print("\n=== Running with ParallelScheduler (max_workers=4) ===")
    start = time.perf_counter()

    result = mata.infer(
        image=str(IMAGE_1),
        graph=graph,
        providers=providers,
        scheduler=ParallelScheduler(max_workers=4),
    )

    elapsed = time.perf_counter() - start
    print(f"Parallel execution: {elapsed:.3f}s")

    # -----------------------------------------------------------------------
    # Access results
    # -----------------------------------------------------------------------
    print(f"\nResult channels: {list(result.channels.keys())}")

    if result.has_channel("detections"):
        dets = result.get_channel("detections")
        print(f"Detections: {len(dets.instances)} objects")
        for inst in dets.instances:
            print(f"  - {inst.label_name}: score={inst.score:.2f}")

    if result.has_channel("classifications"):
        print("Scene classification available")

    if result.has_channel("depth"):
        print("Depth map available")

    # -----------------------------------------------------------------------
    # Compare with sequential execution
    # -----------------------------------------------------------------------
    from mata.core.graph import SyncScheduler

    print("\n=== Running with SyncScheduler (sequential) ===")
    start = time.perf_counter()

    result_seq = mata.infer(
        image=str(IMAGE_1),
        graph=graph,
        providers=providers,
        scheduler=SyncScheduler(),
    )

    elapsed_seq = time.perf_counter() - start
    print(f"Sequential execution: {elapsed_seq:.3f}s")

    if elapsed_seq > 0:
        print(f"Parallel speedup: {elapsed_seq / max(elapsed, 0.001):.1f}×")

    # -----------------------------------------------------------------------
    # Using the full_scene_analysis preset (same graph, less code)
    # -----------------------------------------------------------------------
    print("\n=== Using preset: full_scene_analysis() ===")
    from mata.presets import full_scene_analysis

    preset_graph = full_scene_analysis(
        classification_labels=["indoor", "outdoor"],
    )
    print(f"Preset graph name: {preset_graph.name}")
    print(f"Preset nodes: {len(preset_graph._nodes)}")

    print("\n✓ Parallel tasks example complete!")


if __name__ == "__main__":
    main()
