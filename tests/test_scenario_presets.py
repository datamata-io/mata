"""Tests for industry scenario graph presets.

Tests cover:
- Preset construction and graph structure validation
- Parameter customization
- Node type presence and configuration
- Provider name verification
- Edge cases and defaults
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.core.graph.node import Node

# Nodes (for type checking graph contents)
from mata.nodes.annotate import Annotate
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.refine_mask import RefineMask
from mata.nodes.roi import ExtractROIs
from mata.nodes.segment import SegmentImage
from mata.nodes.topk import TopK
from mata.nodes.track import Track
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_query import VLMQuery
from mata.presets.agriculture import aerial_crop_analysis
from mata.presets.driving import (
    road_scene_analysis,
    traffic_tracking,
    vehicle_distance_estimation,
)
from mata.presets.general import ensemble_detection

# Presets under test
from mata.presets.manufacturing import (
    assembly_verification,
    component_inspection,
    defect_detect_classify,
)
from mata.presets.retail import (
    shelf_product_analysis,
    stock_level_analysis,
)
from mata.presets.surveillance import (
    crowd_monitoring,
    suspicious_object_detection,
)

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_presets.py)
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
# Manufacturing Preset Tests
# ---------------------------------------------------------------------------


class TestDefectDetectClassify:
    """Tests for the defect_detect_classify preset."""

    def test_returns_graph(self):
        g = defect_detect_classify()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = defect_detect_classify()
        assert g.name == "defect_detect_classify"

    def test_default_nodes(self):
        g = defect_detect_classify()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert ExtractROIs in types
        assert Classify in types
        assert Fuse in types

    def test_detector_uses_text_prompts(self):
        g = defect_detect_classify(defect_prompts="scratch . crack")
        det = _node_by_type(g, Detect)
        assert det.kwargs.get("text_prompts") == "scratch . crack"

    def test_custom_classification_labels(self):
        custom_labels = ["scratch", "crack", "normal"]
        g = defect_detect_classify(classification_labels=custom_labels)
        clf = _node_by_type(g, Classify)
        assert clf.kwargs.get("text_prompts") == custom_labels

    def test_default_classification_labels(self):
        g = defect_detect_classify()
        clf = _node_by_type(g, Classify)
        text_prompts = clf.kwargs.get("text_prompts")
        assert len(text_prompts) == 7
        assert "scratch" in text_prompts
        assert "normal" in text_prompts

    def test_custom_detection_threshold(self):
        g = defect_detect_classify(detection_threshold=0.7)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.7

    def test_detector_provider_name(self):
        g = defect_detect_classify()
        det = _node_by_type(g, Detect)
        assert det.provider_name == "detector"

    def test_classifier_provider_name(self):
        g = defect_detect_classify()
        clf = _node_by_type(g, Classify)
        assert clf.provider_name == "classifier"

    def test_fuse_output_name(self):
        g = defect_detect_classify()
        fuse = _node_by_type(g, Fuse)
        assert fuse.output_name == "final"


class TestAssemblyVerification:
    """Tests for the assembly_verification preset."""

    def test_returns_graph(self):
        g = assembly_verification()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = assembly_verification()
        assert g.name == "assembly_verification"

    def test_has_vlm_and_detect_in_parallel(self):
        g = assembly_verification()
        assert _has_node_type(g, VLMQuery)
        assert _has_node_type(g, Detect)

    def test_has_filter_and_fuse(self):
        g = assembly_verification()
        types = _node_types(g)
        assert Filter in types
        assert Fuse in types

    def test_custom_vlm_prompt(self):
        custom_prompt = "Check if all screws are present"
        g = assembly_verification(vlm_prompt=custom_prompt)
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.prompt == custom_prompt

    def test_custom_detection_threshold(self):
        g = assembly_verification(detection_threshold=0.6)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.6

    def test_detector_provider_name(self):
        g = assembly_verification()
        det = _node_by_type(g, Detect)
        assert det.provider_name == "detector"

    def test_vlm_provider_name(self):
        g = assembly_verification()
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.provider_name == "vlm"


class TestComponentInspection:
    """Tests for the component_inspection preset."""

    def test_returns_graph(self):
        g = component_inspection()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = component_inspection()
        assert g.name == "component_inspection"

    def test_default_nodes(self):
        g = component_inspection()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert ExtractROIs in types
        assert VLMQuery in types
        assert Fuse in types

    def test_no_topk_by_default(self):
        g = component_inspection()
        assert not _has_node_type(g, TopK)

    def test_topk_when_enabled(self):
        g = component_inspection(top_k=5)
        assert _has_node_type(g, TopK)
        topk = _node_by_type(g, TopK)
        assert topk.k == 5

    def test_custom_vlm_prompt(self):
        custom_prompt = "Inspect for wear and tear"
        g = component_inspection(vlm_prompt=custom_prompt)
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.prompt == custom_prompt

    def test_custom_detection_threshold(self):
        g = component_inspection(detection_threshold=0.8)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.8


# ---------------------------------------------------------------------------
# General Preset Tests
# ---------------------------------------------------------------------------


class TestEnsembleDetection:
    """Tests for the ensemble_detection preset."""

    def test_returns_graph(self):
        g = ensemble_detection()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = ensemble_detection()
        assert g.name == "ensemble_detection"

    def test_has_two_detectors(self):
        g = ensemble_detection()
        detect_nodes = [n for n in g._nodes if isinstance(n, Detect)]
        assert len(detect_nodes) == 2

    def test_has_nms(self):
        g = ensemble_detection()
        assert _has_node_type(g, NMS)

    def test_has_filter_and_fuse(self):
        g = ensemble_detection()
        types = _node_types(g)
        assert Filter in types
        assert Fuse in types

    def test_detector_provider_names(self):
        g = ensemble_detection()
        detect_nodes = [n for n in g._nodes if isinstance(n, Detect)]
        provider_names = [n.provider_name for n in detect_nodes]
        assert "detector_a" in provider_names
        assert "detector_b" in provider_names

    def test_custom_nms_threshold(self):
        g = ensemble_detection(nms_iou_threshold=0.3)
        nms = _node_by_type(g, NMS)
        assert nms.iou_threshold == 0.3

    def test_custom_detection_threshold(self):
        g = ensemble_detection(detection_threshold=0.5)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.5


# ---------------------------------------------------------------------------
# Retail Preset Tests
# ---------------------------------------------------------------------------


class TestShelfProductAnalysis:
    """Tests for the shelf_product_analysis preset."""

    def test_returns_graph(self):
        g = shelf_product_analysis()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = shelf_product_analysis()
        assert g.name == "shelf_product_analysis"

    def test_default_nodes(self):
        g = shelf_product_analysis()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert ExtractROIs in types
        assert Classify in types
        assert Fuse in types

    def test_nms_enabled_by_default(self):
        g = shelf_product_analysis()
        assert _has_node_type(g, NMS)

    def test_nms_disabled_when_none(self):
        g = shelf_product_analysis(nms_iou_threshold=None)
        assert not _has_node_type(g, NMS)

    def test_custom_nms_threshold(self):
        g = shelf_product_analysis(nms_iou_threshold=0.4)
        nms = _node_by_type(g, NMS)
        assert nms.iou_threshold == 0.4

    def test_default_product_labels(self):
        g = shelf_product_analysis()
        clf = _node_by_type(g, Classify)
        text_prompts = clf.kwargs.get("text_prompts")
        assert len(text_prompts) == 7
        assert "beverage" in text_prompts
        assert "snack" in text_prompts

    def test_custom_classification_labels(self):
        custom_labels = ["cola", "water", "juice"]
        g = shelf_product_analysis(classification_labels=custom_labels)
        clf = _node_by_type(g, Classify)
        assert clf.kwargs.get("text_prompts") == custom_labels

    def test_custom_roi_padding(self):
        g = shelf_product_analysis(roi_padding=15)
        roi = _node_by_type(g, ExtractROIs)
        assert roi.padding == 15


class TestStockLevelAnalysis:
    """Tests for the stock_level_analysis preset."""

    def test_returns_graph(self):
        g = stock_level_analysis()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = stock_level_analysis()
        assert g.name == "stock_level_analysis"

    def test_has_parallel_triple_analysis(self):
        g = stock_level_analysis()
        assert _has_node_type(g, VLMDescribe)
        assert _has_node_type(g, Detect)
        assert _has_node_type(g, Classify)

    def test_has_filter_and_fuse(self):
        g = stock_level_analysis()
        types = _node_types(g)
        assert Filter in types
        assert Fuse in types

    def test_custom_vlm_prompt(self):
        custom_prompt = "Count products on each shelf"
        g = stock_level_analysis(vlm_prompt=custom_prompt)
        vlm = _node_by_type(g, VLMDescribe)
        assert vlm.prompt == custom_prompt

    def test_default_stock_labels(self):
        g = stock_level_analysis()
        clf = _node_by_type(g, Classify)
        text_prompts = clf.kwargs.get("text_prompts")
        assert len(text_prompts) == 4
        assert "fully_stocked" in text_prompts
        assert "empty_shelf" in text_prompts

    def test_custom_stock_labels(self):
        custom_labels = ["full", "half", "empty"]
        g = stock_level_analysis(classification_labels=custom_labels)
        clf = _node_by_type(g, Classify)
        assert clf.kwargs.get("text_prompts") == custom_labels


# ---------------------------------------------------------------------------
# Driving Preset Tests
# ---------------------------------------------------------------------------


class TestVehicleDistanceEstimation:
    """Tests for the vehicle_distance_estimation preset."""

    def test_returns_graph(self):
        g = vehicle_distance_estimation()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = vehicle_distance_estimation()
        assert g.name == "vehicle_distance_estimation"

    def test_has_detect_and_depth_parallel(self):
        g = vehicle_distance_estimation()
        assert _has_node_type(g, Detect)
        assert _has_node_type(g, EstimateDepth)

    def test_has_filter_with_vehicle_labels(self):
        g = vehicle_distance_estimation()
        filt = _node_by_type(g, Filter)
        assert filt.label_in is not None
        assert "car" in filt.label_in
        assert "person" in filt.label_in

    def test_default_vehicle_classes(self):
        g = vehicle_distance_estimation()
        filt = _node_by_type(g, Filter)
        assert len(filt.label_in) == 6
        assert "car" in filt.label_in
        assert "truck" in filt.label_in
        assert "motorcycle" in filt.label_in

    def test_custom_vehicle_labels(self):
        custom_labels = ["car", "truck"]
        g = vehicle_distance_estimation(vehicle_labels=custom_labels)
        filt = _node_by_type(g, Filter)
        assert filt.label_in == custom_labels

    def test_custom_detection_threshold(self):
        g = vehicle_distance_estimation(detection_threshold=0.6)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.6


class TestRoadSceneAnalysis:
    """Tests for the road_scene_analysis preset."""

    def test_returns_graph(self):
        g = road_scene_analysis()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = road_scene_analysis()
        assert g.name == "road_scene_analysis"

    def test_has_four_parallel_tasks(self):
        g = road_scene_analysis()
        assert _has_node_type(g, Detect)
        assert _has_node_type(g, SegmentImage)
        assert _has_node_type(g, EstimateDepth)
        assert _has_node_type(g, Classify)

    def test_has_filter_and_fuse(self):
        g = road_scene_analysis()
        types = _node_types(g)
        assert Filter in types
        assert Fuse in types

    def test_default_scene_labels(self):
        g = road_scene_analysis()
        clf = _node_by_type(g, Classify)
        text_prompts = clf.kwargs.get("text_prompts")
        assert len(text_prompts) == 5
        assert "urban_road" in text_prompts
        assert "highway" in text_prompts

    def test_custom_scene_labels(self):
        custom_labels = ["city", "country"]
        g = road_scene_analysis(scene_labels=custom_labels)
        clf = _node_by_type(g, Classify)
        assert clf.kwargs.get("text_prompts") == custom_labels

    def test_custom_detection_threshold(self):
        g = road_scene_analysis(detection_threshold=0.5)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.5


class TestTrafficTracking:
    """Tests for the traffic_tracking preset."""

    def test_returns_graph(self):
        g = traffic_tracking()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = traffic_tracking()
        assert g.name == "traffic_tracking"

    def test_has_tracking_pipeline(self):
        g = traffic_tracking()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert Track in types
        assert Annotate in types
        assert Fuse in types

    def test_filter_uses_vehicle_labels(self):
        g = traffic_tracking()
        filt = _node_by_type(g, Filter)
        assert filt.label_in is not None
        assert "car" in filt.label_in

    def test_track_params_propagate(self):
        g = traffic_tracking(
            track_threshold=0.6,
            match_threshold=0.7,
        )
        track = _node_by_type(g, Track)
        assert track.track_thresh == 0.6
        assert track.match_thresh == 0.7
        # Note: traffic_tracking preset doesn't pass track_buffer parameter
        assert track.track_buffer == 30  # default value

    def test_custom_vehicle_labels(self):
        custom_labels = ["car", "truck"]
        g = traffic_tracking(vehicle_labels=custom_labels)
        filt = _node_by_type(g, Filter)
        assert filt.label_in == custom_labels


# ---------------------------------------------------------------------------
# Surveillance Preset Tests
# ---------------------------------------------------------------------------


class TestCrowdMonitoring:
    """Tests for the crowd_monitoring preset."""

    def test_returns_graph(self):
        g = crowd_monitoring()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = crowd_monitoring()
        assert g.name == "crowd_monitoring"

    def test_has_tracking_pipeline(self):
        g = crowd_monitoring()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert Track in types
        assert Annotate in types
        assert Fuse in types

    def test_filter_person_only_fuzzy(self):
        g = crowd_monitoring()
        filt = _node_by_type(g, Filter)
        assert filt.label_in == ["person"]
        assert filt.fuzzy is True

    def test_track_params_propagate(self):
        g = crowd_monitoring(
            track_threshold=0.7,
            match_threshold=0.9,
            track_buffer=40,
        )
        track = _node_by_type(g, Track)
        assert track.track_thresh == 0.7
        assert track.match_thresh == 0.9
        assert track.track_buffer == 40

    def test_custom_detection_threshold(self):
        g = crowd_monitoring(detection_threshold=0.5)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.5


class TestSuspiciousObjectDetection:
    """Tests for the suspicious_object_detection preset."""

    def test_returns_graph(self):
        g = suspicious_object_detection()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = suspicious_object_detection()
        assert g.name == "suspicious_object_detection"

    def test_has_detection_segmentation_vlm_chain(self):
        g = suspicious_object_detection()
        types = _node_types(g)
        assert Detect in types
        assert Filter in types
        assert PromptBoxes in types
        assert RefineMask in types
        assert VLMQuery in types
        assert Fuse in types

    def test_custom_object_prompts(self):
        custom_prompts = "bag . briefcase"
        g = suspicious_object_detection(object_prompts=custom_prompts)
        det = _node_by_type(g, Detect)
        assert det.kwargs.get("text_prompts") == custom_prompts

    def test_custom_vlm_prompt(self):
        custom_prompt = "Is this object abandoned?"
        g = suspicious_object_detection(vlm_prompt=custom_prompt)
        vlm = _node_by_type(g, VLMQuery)
        assert vlm.prompt == custom_prompt

    def test_custom_refine_params(self):
        g = suspicious_object_detection(
            refine_method="dilate",
            refine_radius=5,
        )
        refine = _node_by_type(g, RefineMask)
        assert refine.method == "dilate"
        assert refine.radius == 5

    def test_custom_detection_threshold(self):
        g = suspicious_object_detection(detection_threshold=0.3)
        filt = _node_by_type(g, Filter)
        assert filt.score_gt == 0.3


# ---------------------------------------------------------------------------
# Agriculture Preset Tests
# ---------------------------------------------------------------------------


class TestAerialCropAnalysis:
    """Tests for the aerial_crop_analysis preset."""

    def test_returns_graph(self):
        g = aerial_crop_analysis()
        assert isinstance(g, Graph)

    def test_graph_name(self):
        g = aerial_crop_analysis()
        assert g.name == "aerial_crop_analysis"

    def test_has_segment_and_depth_parallel(self):
        g = aerial_crop_analysis()
        assert _has_node_type(g, SegmentImage)
        assert _has_node_type(g, EstimateDepth)

    def test_has_fuse_output(self):
        g = aerial_crop_analysis()
        assert _has_node_type(g, Fuse)
        fuse = _node_by_type(g, Fuse)
        assert fuse.output_name == "final"

    def test_segmenter_provider_name(self):
        g = aerial_crop_analysis()
        seg = _node_by_type(g, SegmentImage)
        assert seg.provider_name == "segmenter"

    def test_depth_provider_name(self):
        g = aerial_crop_analysis()
        depth = _node_by_type(g, EstimateDepth)
        assert depth.provider_name == "depth"
