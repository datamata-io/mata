#!/usr/bin/env python3
"""Simple two-stage pipeline: Detect → Filter → Segment → Fuse.

Demonstrates the core MATA graph workflow:
1. Load provider models (detector + segmenter)
2. Build a sequential graph with `mata.infer()` and node list
3. Access results from the returned `MultiResult`

Usage:
    # Mock mode (no real models needed)
    python examples/graph/simple_pipeline.py

    # Real mode (downloads HuggingFace models)
    python examples/graph/simple_pipeline.py --real
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup paths
IMAGE_DIR = Path(__file__).parent.parent / "images"
IMAGE_1 = IMAGE_DIR / "000000039769.jpg"
IMAGE_2 = IMAGE_DIR / "hanvin-cheong-tuR2XRPdtYI-unsplash.jpg"

# ---------------------------------------------------------------------------
# Mock providers for standalone demo (no model downloads required)
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create mock detector and segmenter for demo purposes."""
    from unittest.mock import Mock

    import numpy as np

    from mata.core.types import Instance, VisionResult

    # -- Mock detector --
    mock_detector = Mock()
    mock_detector.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(50, 30, 220, 300), label=0, score=0.92, label_name="cat"),
            Instance(bbox=(280, 60, 450, 350), label=1, score=0.87, label_name="dog"),
            Instance(bbox=(10, 10, 50, 50), label=2, score=0.15, label_name="noise"),
        ],
        meta={"model": "mock-detr"},
    ))

    # -- Mock segmenter --
    mock_segmenter = Mock()
    # PromptBoxes node calls .segment() not .predict()
    mock_segmenter.segment = Mock(return_value=VisionResult(
        instances=[
            Instance(
                bbox=(50, 30, 220, 300),
                label=0,
                score=0.92,
                label_name="cat",
                mask=np.ones((270, 170), dtype=np.uint8),
            ),
            Instance(
                bbox=(280, 60, 450, 350),
                label=1,
                score=0.87,
                label_name="dog",
                mask=np.ones((290, 170), dtype=np.uint8),
            ),
        ],
        meta={"model": "mock-sam"},
    ))

    return {"detector": mock_detector, "segmenter": mock_segmenter}


# ---------------------------------------------------------------------------
# Real providers (uncomment or use --real flag)
# ---------------------------------------------------------------------------

def create_real_providers():
    """Create real detector and segmenter from HuggingFace."""
    import mata

    detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
    segmenter = mata.load("segment", "facebook/sam-vit-base")
    return {"detector": detector, "segmenter": segmenter}


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main():
    """Run the simple detection + segmentation pipeline."""
    import mata
    from mata.nodes import Detect, Filter, Fuse, PromptBoxes

    # Choose real or mock providers
    use_real = "--real" in sys.argv
    if use_real:
        print("Loading real models from HuggingFace...")
        providers = create_real_providers()
    else:
        print("Running with mock providers (use --real for actual models)")
        providers = create_mock_providers()

    # -----------------------------------------------------------------------
    # Option A: Pass a list of nodes (auto-wraps into a Graph)
    # -----------------------------------------------------------------------
    print("\n=== Option A: Node list ===")

    result = mata.infer(
        image=str(IMAGE_1),
        graph=[
            # Step 1: Detect objects (text_prompts required for GroundingDINO)
            Detect(using="detector", out="dets", text_prompts="cat . dog . person"),
            # Step 2: Keep only high-confidence detections
            Filter(src="dets", score_gt=0.3, out="filtered"),
            # Step 3: Segment each detected box with SAM
            PromptBoxes(using="segmenter", dets_src="filtered", out="masks"),
            # Step 4: Bundle everything into a MultiResult
            Fuse(detections="filtered", masks="masks", out="final"),
        ],
        providers=providers,
    )

    # Access results via channel names
    print(f"Result type: {type(result).__name__}")
    print(f"Channels: {list(result.channels.keys())}")
    if result.has_channel("detections"):
        dets = result.get_channel("detections")
        print(f"Detections: {len(dets.instances)} objects")
        for inst in dets.instances:
            print(f"  - {inst.label_name}: score={inst.score:.2f}, bbox={inst.bbox}")

    # -----------------------------------------------------------------------
    # Option B: Build a Graph with the fluent builder API
    # -----------------------------------------------------------------------
    print("\n=== Option B: Graph builder ===")

    from mata.core.graph import Graph

    graph = (
        Graph("detect_and_segment")
        .then(Detect(using="detector", out="dets", text_prompts="cat . dog . person"))
        .then(Filter(src="dets", score_gt=0.3, out="filtered"))
        .then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))
        .then(Fuse(detections="filtered", masks="masks", out="final"))
    )

    result_b = mata.infer(
        image=str(IMAGE_1),
        graph=graph,
        providers=providers,
    )

    print(f"Graph '{graph.name}' completed")
    print(f"Channels: {list(result_b.channels.keys())}")

    print("\n✓ Simple pipeline example complete!")


if __name__ == "__main__":
    main()
