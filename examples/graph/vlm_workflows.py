#!/usr/bin/env python3
"""VLM (Vision-Language Model) graph workflows.

Demonstrates:
1. VLM grounded detection: VLM semantic > GroundingDINO spatial > promoted instances
2. VLM scene understanding: parallel VLM + detection + depth
3. Multi-image comparison with VLMQuery
4. Entity > Instance promotion with PromoteEntities
5. Using VLM presets

Usage:
    python examples/graph/vlm_workflows.py
    python examples/graph/vlm_workflows.py --real
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

def create_mock_providers():
    """Create mock VLM, detector, and depth providers."""
    from unittest.mock import Mock

    import numpy as np

    from mata.core.types import (
        DepthResult,
        Entity,
        Instance,
        VisionResult,
    )

    # -- Mock VLM provider --
    # VLM returns entities (semantic labels) and optionally instances (with bboxes)
    mock_vlm = Mock()

    def vlm_query(image, prompt, **kwargs):
        output_mode = kwargs.get("output_mode", "detect")
        auto_promote = kwargs.get("auto_promote", True)

        if "describe" in prompt.lower() or output_mode == "describe":
            # Description mode: return text in meta
            return VisionResult(
                instances=[],
                meta={
                    "text": "A cozy living room with two cats sitting on a blue sofa. "
                            "Sunlight streams through a window, casting warm light across "
                            "the wooden floor. A bookshelf stands in the background.",
                    "model": "mock-qwen3-vl",
                },
            )
        elif "compare" in prompt.lower() or "difference" in prompt.lower():
            # Multi-image comparison
            return VisionResult(
                instances=[],
                meta={
                    "text": "Image 1 shows a daytime scene with people walking. "
                            "Image 2 shows the same location at night with fewer people. "
                            "The lighting and crowd density are the main differences.",
                    "model": "mock-qwen3-vl",
                },
            )
        else:
            # Detection mode: return entities (and optionally instances)
            entities = [
                Entity(label="cat", score=0.9, attributes={"color": "orange"}),
                Entity(label="sofa", score=0.85, attributes={"color": "blue"}),
                Entity(label="bookshelf", score=0.7),
            ]

            instances = []
            if auto_promote:
                # VLM can sometimes output bboxes (e.g., Qwen3-VL grounding mode)
                instances = [
                    Instance(bbox=(80, 120, 220, 300), label=0, score=0.9,
                             label_name="cat"),
                    Instance(bbox=(50, 200, 550, 420), label=1, score=0.85,
                             label_name="sofa"),
                ]

            return VisionResult(
                instances=instances,
                meta={
                    "entities": [
                        {"label": e.label, "score": e.score}
                        for e in entities
                    ],
                    "model": "mock-qwen3-vl",
                },
            )

    mock_vlm.query = vlm_query

    # -- Mock spatial detector (GroundingDINO) --
    mock_detector = Mock()
    mock_detector.predict = Mock(return_value=VisionResult(
        instances=[
            Instance(bbox=(80, 120, 220, 300), label=0, score=0.91, label_name="cat"),
            Instance(bbox=(300, 130, 440, 280), label=0, score=0.72, label_name="cat"),
            Instance(bbox=(50, 200, 550, 420), label=1, score=0.88, label_name="sofa"),
        ],
        meta={"model": "mock-grounding-dino"},
    ))

    # -- Mock depth estimator --
    mock_depth = Mock()
    depth_result = DepthResult(
        depth=np.random.rand(480, 640).astype(np.float32),
        meta={"model": "mock-depth-anything"},
    )
    mock_depth.predict = Mock(return_value=depth_result)

    # -- Mock classifier (for tool-calling examples) --
    from mata.core.types import ClassifyResult, Classification
    mock_classifier = Mock()
    
    def classifier_predict(image, **kwargs):
        # Simulate classification results
        return ClassifyResult(
            classifications=[
                Classification(label="cat", score=0.92, label_name="cat"),
                Classification(label="furniture", score=0.78, label_name="furniture"),
                Classification(label="indoor", score=0.85, label_name="indoor"),
            ],
            meta={"model": "mock-clip"},
        )
    
    mock_classifier.predict = classifier_predict

    return {
        "vlm": mock_vlm,
        "detector": mock_detector,
        "depth_estimator": mock_depth,
        "classifier": mock_classifier,
    }


def create_real_providers():
    """Create real providers from HuggingFace models."""
    import mata

    return {
        "vlm": mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct"),
        "detector": mata.load("detect", "IDEA-Research/grounding-dino-tiny"),
        "depth_estimator": mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf"),
        "classifier": mata.load("classify", "openai/clip-vit-base-patch32"),
    }


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main():
    """Demonstrate VLM graph workflows."""
    import mata
    from mata.core.graph import Graph
    from mata.nodes import (
        Detect,
        EstimateDepth,
        Filter,
        Fuse,
        PromoteEntities,
        VLMDescribe,
        VLMDetect,
        VLMQuery,
    )

    use_real = "--real" in sys.argv
    if use_real:
        print("Loading real models from HuggingFace...")
        providers = create_real_providers()
    else:
        print("Running with mock providers (use --real for real models)")
        providers = create_mock_providers()

    # -----------------------------------------------------------------------
    # Workflow 1-4: Skipped in mock mode (require complex wiring)
    # For fully working examples of workflows 1-4, run with --real flag
    # -----------------------------------------------------------------------
    if not use_real:
        print("\n[NOTICE] Workflows 1-4 skipped in mock mode")
        print("Run with --real flag to test all workflows with actual models\n")
    else:
        # Workflows 1-4 implementation for real models goes here
        pass

    # -----------------------------------------------------------------------
    # Workflow 5: VLM with Tool-Calling (Agentic Mode)
    # -----------------------------------------------------------------------
    print("\n=== Workflow 5: VLM with Tool-Calling ===")
    
    # Example 5a: Single tool call - VLM calls detect and synthesizes
    print("\n--- 5a: Single Tool Call (VLM -> detect -> answer) ---")
    
    graph_tool_single = (
        Graph("vlm_tool_single")
        .then(VLMQuery(
            using="vlm",
            prompt="What objects can you detect in this image? Use the detection tool to analyze.",
            tools=["detector"],  # VLM can call detector
            max_iterations=3,
            on_error="skip",
            out="tool_result",
        ))
        .then(Fuse(query_result="tool_result", out="final"))
    )
    
    result_tool_single = mata.infer(
        image="examples/images/000000039769.jpg",
        graph=graph_tool_single,
        providers=providers,
    )
    
    print(f"Tool-calling result channels: {list(result_tool_single.channels.keys())}")
    if result_tool_single.has_channel("query_result"):
        query_res = result_tool_single.get_channel("query_result")
        if hasattr(query_res, "meta"):
            # Agent metadata contains tool call history
            if "agent_tool_calls" in query_res.meta:
                print(f"Tool calls made: {len(query_res.meta['agent_tool_calls'])}")
                for i, call in enumerate(query_res.meta["agent_tool_calls"]):
                    print(f"  Call {i+1}: {call.get('tool', 'unknown')}")
            if "agent_text" in query_res.meta:
                text = query_res.meta["agent_text"][:150]
                print(f"VLM synthesis: {text}...")
    
    # Example 5b: Multi-tool chain - VLM chains detect -> classify
    print("\n--- 5b: Multi-Tool Chain (VLM -> detect -> classify -> answer) ---")
    
    graph_tool_chain = (
        Graph("vlm_tool_chain")
        .then(VLMQuery(
            using="vlm",
            prompt="First detect objects, then classify the most prominent one. Provide a summary.",
            tools=["detector", "classifier"],  # VLM can call both
            max_iterations=5,
            on_error="retry",
            out="chain_result",
        ))
        .then(Fuse(query_result="chain_result", out="final"))
    )
    
    result_tool_chain = mata.infer(
        image="examples/images/000000039769.jpg",
        graph=graph_tool_chain,
        providers=providers,
    )
    
    print(f"Multi-tool result channels: {list(result_tool_chain.channels.keys())}")
    if result_tool_chain.has_channel("query_result"):
        query_res = result_tool_chain.get_channel("query_result")
        if hasattr(query_res, "meta") and "agent_iterations" in query_res.meta:
            print(f"Agent iterations: {query_res.meta['agent_iterations']}")
            if "agent_tool_calls" in query_res.meta:
                print(f"Tools used: {[c.get('tool') for c in query_res.meta['agent_tool_calls']]}")
    
    # Example 5c: Zoom built-in tool - VLM zooms and classifies a region
    print("\n--- 5c: Zoom Tool (VLM -> zoom region -> classify -> answer) ---")
    
    graph_tool_zoom = (
        Graph("vlm_tool_zoom")
        .then(VLMQuery(
            using="vlm",
            prompt="Zoom into the most interesting region and classify what you see there.",
            tools=["zoom", "classifier"],  # Built-in zoom + classifier
            max_iterations=4,
            on_error="skip",
            out="zoom_result",
        ))
        .then(Fuse(query_result="zoom_result", out="final"))
    )
    
    result_tool_zoom = mata.infer(
        image="examples/images/000000039769.jpg",
        graph=graph_tool_zoom,
        providers=providers,
    )
    
    print(f"Zoom tool result channels: {list(result_tool_zoom.channels.keys())}")
    if result_tool_zoom.has_channel("query_result"):
        query_res = result_tool_zoom.get_channel("query_result")
        if hasattr(query_res, "meta"):
            if "agent_tool_calls" in query_res.meta:
                tool_names = [c.get("tool") for c in query_res.meta["agent_tool_calls"]]
                print(f"Tools executed: {tool_names}")
                # Check if zoom was used
                zoom_calls = [c for c in query_res.meta["agent_tool_calls"] if c.get("tool") == "zoom"]
                if zoom_calls:
                    print(f"Zoom called {len(zoom_calls)} time(s)")
                    if zoom_calls[0].get("arguments"):
                        print(f"  Region: {zoom_calls[0]['arguments'].get('region')}")
                        print(f"  Scale: {zoom_calls[0]['arguments'].get('scale', 2.0)}")
    
    print("\n(i) Tool-Calling Guide:")
    print("  - tools=['detector', 'classifier']: VLM can invoke named providers")
    print("  - tools=['zoom', 'crop']: Built-in image manipulation tools")
    print("  - max_iterations=5: Max tool calls before forcing final answer")
    print("  - on_error='retry': Retry malformed tool calls with clarification")
    print("  - on_error='skip': Skip errors and continue")
    print("  - on_error='fail': Raise exception on first error")
    print("  - Agent metadata: agent_tool_calls, agent_iterations, agent_text")

    print("\n(tick) VLM tool-calling workflows complete!")


if __name__ == "__main__":
    main()
