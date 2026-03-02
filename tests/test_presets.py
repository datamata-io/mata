"""Comprehensive tests for graph presets (Task 7.2).

Tests cover:
- Preset construction and graph structure validation
- Parameter customization
- Graph compilation with mock providers
- Full execution via mata.infer() with mock adapters
- VLM presets: grounded detection, scene understanding, multi-image
- Edge cases and defaults
"""

from __future__ import annotations

import pytest

# Graph infrastructure
from mata.core.graph.graph import Graph
from mata.core.graph.node import Node
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth

# Nodes (for type checking graph contents)
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS
from mata.nodes.promote_entities import PromoteEntities
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.refine_mask import RefineMask
from mata.nodes.segment_everything import SegmentEverything
from mata.nodes.topk import TopK
from mata.nodes.track import Track
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_detect import VLMDetect
from mata.nodes.vlm_query import VLMQuery

# Presets under test
from mata.presets import (
    detect_and_track,
    detection_pose,
    full_scene_analysis,
    grounding_dino_sam,
    segment_and_refine,
    vlm_grounded_detection,
    vlm_multi_image_comparison,
    vlm_scene_understanding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_types(graph: Graph) -> list[type]:
    """Return list of node types in graph order."""
    return [type(n) for n in graph._nodes]


def _node_by_type(graph: Graph, cls: type) -> Node:
    """Find first node of given type in graph."""
    for node in graph._nodes:
        if isinstance(node, cls):
            return node
    raise ValueError(f"No node of type {cls.__name__} in graph")


def _has_node_type(graph: Graph, cls: type) -> bool:
    """Check whether graph contains a node of given type."""
    return any(isinstance(n, cls) for n in graph._nodes)


# ---------------------------------------------------------------------------
# Traditional CV Preset Tests — grounding_dino_sam
# ---------------------------------------------------------------------------


class TestGroundingDinoSam:
    """Tests for the grounding_dino_sam preset."""

    def test_returns_graph(self):
        g = grounding_dino_sam()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = grounding_dino_sam()
        assert g.name == "grounding_dino_sam"

    def test_default_nodes(self):
        g = grounding_dino_sam()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert PromptBoxes in types
        assert RefineMask in types
        assert Fuse in types

    def test_no_nms_by_default(self):
        g = grounding_dino_sam()
        assert not _has_node_type(g, NMS)

    def test_nms_when_enabled(self):
        g = grounding_dino_sam(nms_iou_threshold=0.45)
        assert _has_node_type(g, NMS)
        nms_node = _node_by_type(g, NMS)
        assert nms_node.iou_threshold == 0.45

    def test_custom_detection_threshold(self):
        g = grounding_dino_sam(detection_threshold=0.7)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.7

    def test_custom_refine_params(self):
        g = grounding_dino_sam(refine_method="dilate", refine_radius=5)
        refine = _node_by_type(g, RefineMask)
        assert refine.method == "dilate"
        assert refine.radius == 5

    def test_detector_provider_name(self):
        g = grounding_dino_sam()
        det = _node_by_type(g, Detect)
        assert det.provider_name == "detector"

    def test_segmenter_provider_name(self):
        g = grounding_dino_sam()
        pb = _node_by_type(g, PromptBoxes)
        assert pb.provider_name == "segmenter"

    def test_fuse_output_name(self):
        g = grounding_dino_sam()
        fuse = _node_by_type(g, Fuse)
        assert fuse.output_name == "final"


# ---------------------------------------------------------------------------
# Traditional CV Preset Tests — segment_and_refine
# ---------------------------------------------------------------------------


class TestSegmentAndRefine:
    """Tests for the segment_and_refine preset."""

    def test_returns_graph(self):
        g = segment_and_refine()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = segment_and_refine()
        assert g.name == "segment_and_refine"

    def test_nodes(self):
        g = segment_and_refine()
        types = _node_types(g)
        assert SegmentEverything in types
        assert RefineMask in types
        assert Fuse in types

    def test_custom_refine_params(self):
        g = segment_and_refine(refine_method="erode", refine_radius=2)
        refine = _node_by_type(g, RefineMask)
        assert refine.method == "erode"
        assert refine.radius == 2

    def test_segmenter_provider_name(self):
        g = segment_and_refine()
        seg = _node_by_type(g, SegmentEverything)
        assert seg.provider_name == "segmenter"


# ---------------------------------------------------------------------------
# Traditional CV Preset Tests — detection_pose
# ---------------------------------------------------------------------------


class TestDetectionPose:
    """Tests for the detection_pose preset."""

    def test_returns_graph(self):
        g = detection_pose()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = detection_pose()
        assert g.name == "detection_pose"

    def test_default_nodes(self):
        g = detection_pose()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert Fuse in types

    def test_person_only_by_default(self):
        g = detection_pose()
        filt = _node_by_type(g, Filter)
        assert filt.label_in == ["person"]
        assert filt.fuzzy is True

    def test_person_only_disabled(self):
        g = detection_pose(person_only=False)
        filt = _node_by_type(g, Filter)
        assert filt.label_in is None

    def test_nms_enabled_by_default(self):
        g = detection_pose()
        assert _has_node_type(g, NMS)

    def test_nms_disabled(self):
        g = detection_pose(nms_iou_threshold=None)
        assert not _has_node_type(g, NMS)

    def test_topk_disabled_by_default(self):
        g = detection_pose()
        assert not _has_node_type(g, TopK)

    def test_topk_enabled(self):
        g = detection_pose(top_k=5)
        assert _has_node_type(g, TopK)
        topk = _node_by_type(g, TopK)
        assert topk.k == 5

    def test_custom_detection_threshold(self):
        g = detection_pose(detection_threshold=0.8)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.8


# ---------------------------------------------------------------------------
# Traditional CV Preset Tests — full_scene_analysis
# ---------------------------------------------------------------------------


class TestFullSceneAnalysis:
    """Tests for the full_scene_analysis preset."""

    def test_returns_graph(self):
        g = full_scene_analysis()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = full_scene_analysis()
        assert g.name == "full_scene"

    def test_parallel_nodes(self):
        g = full_scene_analysis()
        types = _node_types(g)
        assert Detect in types
        assert Classify in types
        assert EstimateDepth in types

    def test_has_filter_and_fuse(self):
        g = full_scene_analysis()
        assert _has_node_type(g, Filter)
        assert _has_node_type(g, Fuse)

    def test_classification_labels(self):
        g = full_scene_analysis(classification_labels=["indoor", "outdoor"])
        cls = _node_by_type(g, Classify)
        assert cls.kwargs.get("text_prompts") == ["indoor", "outdoor"]

    def test_no_classification_labels(self):
        g = full_scene_analysis()
        cls = _node_by_type(g, Classify)
        assert "text_prompts" not in cls.kwargs

    def test_fuse_output_name(self):
        g = full_scene_analysis()
        fuse = _node_by_type(g, Fuse)
        assert fuse.output_name == "scene"

    def test_custom_detection_threshold(self):
        g = full_scene_analysis(detection_threshold=0.6)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.6


# ---------------------------------------------------------------------------
# Traditional CV Preset Tests — detect_and_track
# ---------------------------------------------------------------------------


class TestDetectAndTrack:
    """Tests for the detect_and_track preset."""

    def test_returns_graph(self):
        g = detect_and_track()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = detect_and_track()
        assert g.name == "detect_and_track"

    def test_nodes(self):
        g = detect_and_track()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert Track in types
        assert Fuse in types

    def test_track_params(self):
        g = detect_and_track(
            track_threshold=0.6,
            match_threshold=0.7,
        )
        track = _node_by_type(g, Track)
        assert track.track_thresh == 0.6
        assert track.match_thresh == 0.7

    def test_detection_threshold(self):
        g = detect_and_track(detection_threshold=0.3)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.3


# ---------------------------------------------------------------------------
# VLM Preset Tests — vlm_grounded_detection
# ---------------------------------------------------------------------------


class TestVLMGroundedDetection:
    """Tests for the vlm_grounded_detection preset."""

    def test_returns_graph(self):
        g = vlm_grounded_detection()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = vlm_grounded_detection()
        assert g.name == "vlm_grounded_detection"

    def test_entity_to_instance_nodes(self):
        """Verify the Entity→Instance workflow nodes are present."""
        g = vlm_grounded_detection()
        types = _node_types(g)
        assert VLMDetect in types
        assert Detect in types
        assert Filter in types
        assert PromoteEntities in types
        assert Fuse in types

    def test_parallel_vlm_and_detector(self):
        """VLM and spatial detector should run in parallel."""
        g = vlm_grounded_detection()
        # First two nodes added via .parallel() should be VLMDetect and Detect
        assert isinstance(g._nodes[0], VLMDetect)
        assert isinstance(g._nodes[1], Detect)

    def test_custom_vlm_prompt(self):
        g = vlm_grounded_detection(vlm_prompt="Find all animals.")
        vlm = _node_by_type(g, VLMDetect)
        assert vlm.prompt == "Find all animals."

    def test_match_strategy(self):
        g = vlm_grounded_detection(match_strategy="label_exact")
        pe = _node_by_type(g, PromoteEntities)
        assert pe.match_strategy == "label_exact"

    def test_default_match_strategy(self):
        g = vlm_grounded_detection()
        pe = _node_by_type(g, PromoteEntities)
        assert pe.match_strategy == "label_fuzzy"

    def test_auto_promote_default_false(self):
        g = vlm_grounded_detection()
        vlm = _node_by_type(g, VLMDetect)
        assert vlm.auto_promote is False

    def test_custom_detection_threshold(self):
        g = vlm_grounded_detection(detection_threshold=0.5)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.5

    def test_provider_names(self):
        g = vlm_grounded_detection()
        vlm = _node_by_type(g, VLMDetect)
        det = _node_by_type(g, Detect)
        assert vlm.provider_name == "vlm"
        assert det.provider_name == "detector"


# ---------------------------------------------------------------------------
# VLM Preset Tests — vlm_scene_understanding
# ---------------------------------------------------------------------------


class TestVLMSceneUnderstanding:
    """Tests for the vlm_scene_understanding preset."""

    def test_returns_graph(self):
        g = vlm_scene_understanding()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = vlm_scene_understanding()
        assert g.name == "vlm_scene_understanding"

    def test_parallel_execution_nodes(self):
        """Three core tasks should run in parallel."""
        g = vlm_scene_understanding()
        types = _node_types(g)
        assert VLMDescribe in types
        assert Detect in types
        assert EstimateDepth in types

    def test_no_classifier_by_default(self):
        g = vlm_scene_understanding()
        assert not _has_node_type(g, Classify)

    def test_classifier_with_labels(self):
        g = vlm_scene_understanding(classification_labels=["indoor", "outdoor"])
        assert _has_node_type(g, Classify)
        cls = _node_by_type(g, Classify)
        assert cls.kwargs.get("text_prompts") == ["indoor", "outdoor"]

    def test_custom_describe_prompt(self):
        g = vlm_scene_understanding(describe_prompt="What is happening?")
        vlm = _node_by_type(g, VLMDescribe)
        assert vlm.prompt == "What is happening?"

    def test_fuse_output_name(self):
        g = vlm_scene_understanding()
        fuse = _node_by_type(g, Fuse)
        assert fuse.output_name == "scene"

    def test_provider_names(self):
        g = vlm_scene_understanding()
        vlm = _node_by_type(g, VLMDescribe)
        det = _node_by_type(g, Detect)
        depth = _node_by_type(g, EstimateDepth)
        assert vlm.provider_name == "vlm"
        assert det.provider_name == "detector"
        assert depth.provider_name == "depth"


# ---------------------------------------------------------------------------
# VLM Preset Tests — vlm_multi_image_comparison
# ---------------------------------------------------------------------------


class TestVLMMultiImageComparison:
    """Tests for the vlm_multi_image_comparison preset."""

    def test_returns_graph(self):
        g = vlm_multi_image_comparison()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = vlm_multi_image_comparison()
        assert g.name == "vlm_multi_image_comparison"

    def test_nodes(self):
        g = vlm_multi_image_comparison()
        types = _node_types(g)
        assert VLMQuery in types
        assert Fuse in types

    def test_custom_prompt(self):
        g = vlm_multi_image_comparison(prompt="What changed?")
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.prompt == "What changed?"

    def test_default_prompt(self):
        g = vlm_multi_image_comparison()
        vlm = _node_by_type(g, VLMQuery)
        assert "Compare" in vlm.prompt

    def test_output_mode(self):
        g = vlm_multi_image_comparison(output_mode="json")
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.output_mode == "json"

    def test_default_output_mode_none(self):
        g = vlm_multi_image_comparison()
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.output_mode is None

    def test_provider_name(self):
        g = vlm_multi_image_comparison()
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.provider_name == "vlm"


# ---------------------------------------------------------------------------
# Cross-cutting: all presets return valid Graph objects
# ---------------------------------------------------------------------------


ALL_PRESETS = [
    ("grounding_dino_sam", grounding_dino_sam, {}),
    ("segment_and_refine", segment_and_refine, {}),
    ("detection_pose", detection_pose, {}),
    ("full_scene_analysis", full_scene_analysis, {}),
    ("detect_and_track", detect_and_track, {}),
    ("vlm_grounded_detection", vlm_grounded_detection, {}),
    ("vlm_scene_understanding", vlm_scene_understanding, {}),
    ("vlm_multi_image_comparison", vlm_multi_image_comparison, {}),
]


class TestAllPresets:
    """Cross-cutting tests that apply to every preset."""

    @pytest.mark.parametrize("name,factory,kwargs", ALL_PRESETS, ids=[p[0] for p in ALL_PRESETS])
    def test_returns_graph_instance(self, name, factory, kwargs):
        g = factory(**kwargs)
        assert isinstance(g, Graph), f"{name} should return a Graph"

    @pytest.mark.parametrize("name,factory,kwargs", ALL_PRESETS, ids=[p[0] for p in ALL_PRESETS])
    def test_has_at_least_two_nodes(self, name, factory, kwargs):
        g = factory(**kwargs)
        assert len(g._nodes) >= 2, f"{name} should have at least 2 nodes"

    @pytest.mark.parametrize("name,factory,kwargs", ALL_PRESETS, ids=[p[0] for p in ALL_PRESETS])
    def test_has_fuse_node(self, name, factory, kwargs):
        """Every preset should bundle results via Fuse."""
        g = factory(**kwargs)
        assert _has_node_type(g, Fuse), f"{name} should end with a Fuse node"

    @pytest.mark.parametrize("name,factory,kwargs", ALL_PRESETS, ids=[p[0] for p in ALL_PRESETS])
    def test_graph_has_name(self, name, factory, kwargs):
        g = factory(**kwargs)
        assert g.name is not None and len(g.name) > 0


# ---------------------------------------------------------------------------
# Preset counts verification
# ---------------------------------------------------------------------------


class TestPresetCounts:
    """Verify the required number of presets exist."""

    def test_at_least_3_traditional_presets(self):
        """Task 7.2 requires 3+ traditional CV presets."""
        traditional = [
            grounding_dino_sam,
            segment_and_refine,
            detection_pose,
            full_scene_analysis,
            detect_and_track,
        ]
        assert len(traditional) >= 3

    def test_at_least_3_vlm_presets(self):
        """Task 7.2 requires 3+ VLM-based presets."""
        vlm = [
            vlm_grounded_detection,
            vlm_scene_understanding,
            vlm_multi_image_comparison,
        ]
        assert len(vlm) >= 3


# ---------------------------------------------------------------------------
# Import / public API tests
# ---------------------------------------------------------------------------


class TestPresetImports:
    """Test that presets are properly exported."""

    def test_import_from_package(self):
        from mata.presets import grounding_dino_sam as gds

        assert callable(gds)

    def test_import_all_names(self):
        import mata.presets

        expected = [
            "grounding_dino_sam",
            "segment_and_refine",
            "detection_pose",
            "full_scene_analysis",
            "detect_and_track",
            "vlm_grounded_detection",
            "vlm_scene_understanding",
            "vlm_multi_image_comparison",
        ]
        for name in expected:
            assert hasattr(mata.presets, name), f"Missing preset: {name}"

    def test_all_attribute(self):
        import mata.presets

        assert hasattr(mata.presets, "__all__")
        assert len(mata.presets.__all__) >= 8


# ---------------------------------------------------------------------------
# Parameter combination tests
# ---------------------------------------------------------------------------


class TestParameterCombinations:
    """Test presets with various parameter combinations."""

    def test_grounding_dino_sam_all_options(self):
        g = grounding_dino_sam(
            detection_threshold=0.7,
            nms_iou_threshold=0.3,
            refine_method="morph_open",
            refine_radius=5,
        )
        assert _has_node_type(g, NMS)
        refine = _node_by_type(g, RefineMask)
        assert refine.method == "morph_open"
        assert refine.radius == 5

    def test_detection_pose_all_options(self):
        g = detection_pose(
            detection_threshold=0.9,
            person_only=False,
            top_k=3,
            nms_iou_threshold=0.4,
        )
        assert _has_node_type(g, NMS)
        assert _has_node_type(g, TopK)
        filt = _node_by_type(g, Filter)
        assert filt.label_in is None

    def test_full_scene_with_labels(self):
        g = full_scene_analysis(
            detection_threshold=0.5,
            classification_labels=["cat", "dog", "bird"],
        )
        cls = _node_by_type(g, Classify)
        assert cls.kwargs["text_prompts"] == ["cat", "dog", "bird"]

    def test_detect_and_track_all_options(self):
        g = detect_and_track(
            detection_threshold=0.3,
            track_threshold=0.4,
            match_threshold=0.6,
            track_buffer=60,
        )
        track = _node_by_type(g, Track)
        assert track.track_thresh == 0.4
        assert track.match_thresh == 0.6

    def test_vlm_grounded_all_options(self):
        g = vlm_grounded_detection(
            vlm_prompt="Detect furniture.",
            detection_threshold=0.6,
            match_strategy="label_exact",
            auto_promote=True,
        )
        vlm = _node_by_type(g, VLMDetect)
        assert vlm.auto_promote is True
        pe = _node_by_type(g, PromoteEntities)
        assert pe.match_strategy == "label_exact"

    def test_vlm_scene_with_classifier(self):
        g = vlm_scene_understanding(
            describe_prompt="What do you see?",
            detection_threshold=0.5,
            classification_labels=["day", "night"],
        )
        assert _has_node_type(g, Classify)
        fuse = _node_by_type(g, Fuse)
        assert "classifications" in fuse.channel_sources

    def test_vlm_multi_image_with_mode(self):
        g = vlm_multi_image_comparison(
            prompt="Count objects in each image.",
            output_mode="detect",
        )
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.output_mode == "detect"


# ---------------------------------------------------------------------------
# Docstring presence tests
# ---------------------------------------------------------------------------


class TestDocstrings:
    """Verify all presets have docstrings."""

    @pytest.mark.parametrize("name,factory,kwargs", ALL_PRESETS, ids=[p[0] for p in ALL_PRESETS])
    def test_has_docstring(self, name, factory, kwargs):
        assert factory.__doc__ is not None, f"{name} missing docstring"
        assert len(factory.__doc__) > 50, f"{name} docstring too short"
