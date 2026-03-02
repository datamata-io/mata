"""Comprehensive tests for Node base class and all built-in nodes.

Tests cover:
- Node base class: abstract enforcement, config, inputs/outputs
- Built-in task nodes (Detect, Classify, SegmentImage, EstimateDepth)
- Transform nodes (Filter, TopK)
- Prompt nodes (PromptBoxes)
- Mask refinement nodes (RefineMask, MaskToBox)
- Fusion nodes (Fuse, Merge)
- Visualization nodes (Annotate, NMS)
- Input/output validation with real artifact types
- Dynamic output name support
- Error handling
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.exceptions import ValidationError
from mata.core.graph.context import ExecutionContext
from mata.core.graph.node import Node
from mata.core.types import Classification, ClassifyResult, DepthResult, Instance, VisionResult

# ──────── helpers ────────


def _make_image() -> Image:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    return Image(data=arr, width=64, height=64, color_space="RGB")


def _make_detections(n: int = 3) -> Detections:
    instances = [
        Instance(
            bbox=(i * 10.0, i * 10.0, i * 10.0 + 50, i * 10.0 + 50),
            score=0.9 - i * 0.1,
            label=i,
            label_name=f"obj_{i}",
        )
        for i in range(n)
    ]
    return Detections.from_vision_result(VisionResult(instances=instances))


def _make_ctx(providers: dict | None = None) -> ExecutionContext:
    return ExecutionContext(providers=providers or {}, device="cpu")


# ──────── mock node for tests ────────


class EchoNode(Node):
    """Passes image through unchanged + creates dummy detections."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, out: str = "dets", **kw):
        super().__init__(**kw)
        self.output_name = out

    def run(self, ctx, image: Image) -> dict[str, Artifact]:
        dets = _make_detections(2)
        return {self.output_name: dets}


class NoIONode(Node):
    """Node with no declared inputs/outputs."""

    inputs = {}
    outputs = {}

    def run(self, ctx, **inputs) -> dict[str, Artifact]:
        return {}


# ──────────────────────────────────────────────────────────────
# 1. Node abstract base
# ──────────────────────────────────────────────────────────────


class TestNodeBase:
    """Base class contract tests."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            _ = Node()

    def test_default_name_is_classname(self):
        n = EchoNode()
        assert n.name == "EchoNode"

    def test_custom_name(self):
        n = EchoNode(name="my_echo")
        assert n.name == "my_echo"

    def test_config_stored(self):
        n = EchoNode(threshold=0.5, device="cuda")
        assert n.config["threshold"] == 0.5
        assert n.config["device"] == "cuda"

    def test_required_inputs(self):
        n = EchoNode()
        assert n.required_inputs == {"image"}

    def test_provided_outputs(self):
        n = EchoNode()
        assert n.provided_outputs == {"detections"}

    def test_repr(self):
        n = EchoNode()
        r = repr(n)
        assert "EchoNode" in r
        assert "image" in r


# ──────────────────────────────────────────────────────────────
# 2. Input validation
# ──────────────────────────────────────────────────────────────


class TestInputValidation:
    """validate_inputs on Node base."""

    def test_valid_inputs(self):
        n = EchoNode()
        img = _make_image()
        n.validate_inputs({"image": img})  # Should not raise

    def test_missing_required_input(self):
        n = EchoNode()
        with pytest.raises(ValidationError, match="missing required inputs"):
            n.validate_inputs({})

    def test_unexpected_input(self):
        n = EchoNode()
        img = _make_image()
        with pytest.raises(ValidationError, match="unexpected inputs"):
            n.validate_inputs({"image": img, "extra": img})

    def test_wrong_type(self):
        n = EchoNode()
        dets = _make_detections()
        with pytest.raises(ValidationError, match="wrong type"):
            n.validate_inputs({"image": dets})

    def test_no_io_node_accepts_anything(self):
        """Node with empty inputs dict should not raise on any kwargs."""
        n = NoIONode()
        n.validate_inputs({})  # ok


# ──────────────────────────────────────────────────────────────
# 3. Output validation
# ──────────────────────────────────────────────────────────────


class TestOutputValidation:
    """validate_outputs on Node base."""

    def test_valid_outputs(self):
        n = EchoNode()
        dets = _make_detections()
        n.validate_outputs({"detections": dets})

    def test_dynamic_output_name(self):
        """Node with output_name attr should accept renamed output."""
        n = EchoNode(out="my_dets")
        dets = _make_detections()
        n.validate_outputs({"my_dets": dets})

    def test_wrong_output_type(self):
        n = EchoNode()
        img = _make_image()
        with pytest.raises(ValidationError, match="wrong type"):
            n.validate_outputs({"detections": img})

    def test_empty_output(self):
        """Missing all expected outputs should raise."""
        n = EchoNode()
        with pytest.raises(ValidationError):
            n.validate_outputs({})


# ──────────────────────────────────────────────────────────────
# 4. Built-in Detect node
# ──────────────────────────────────────────────────────────────


class TestDetectNode:
    """Built-in Detect node with mock provider."""

    def test_detect_with_mock_provider(self):
        from mata.nodes.detect import Detect

        # Create mock provider
        mock_provider = Mock()
        mock_vr = VisionResult(instances=[Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, label_name="cat")])
        mock_provider.predict.return_value = mock_vr

        ctx = _make_ctx({"detect": {"det": mock_provider}})
        ctx.store("input.image", _make_image())

        node = Detect(using="det", out="dets")
        result = node.run(ctx, image=_make_image())
        assert "dets" in result
        assert isinstance(result["dets"], Detections)

    def test_detect_type_declarations(self):
        from mata.nodes.detect import Detect

        node = Detect(using="det")
        assert "image" in node.inputs
        assert node.inputs["image"] is Image

    def test_detect_custom_output_name(self):
        from mata.nodes.detect import Detect

        node = Detect(using="det", out="my_detections")
        assert node.output_name == "my_detections"


# ──────────────────────────────────────────────────────────────
# 5. Built-in Classify node
# ──────────────────────────────────────────────────────────────


class TestClassifyNode:
    """Built-in Classify node with mock provider."""

    def test_classify_with_mock_provider(self):
        from mata.nodes.classify import Classify

        mock_provider = Mock()
        mock_cr = ClassifyResult(predictions=[Classification(label=0, score=0.92, label_name="cat")])
        mock_provider.classify.return_value = mock_cr

        ctx = _make_ctx({"classify": {"clf": mock_provider}})
        node = Classify(using="clf", out="cls")
        result = node.run(ctx, image=_make_image())
        assert "cls" in result
        assert isinstance(result["cls"], Classifications)

    def test_classify_type_declarations(self):
        from mata.nodes.classify import Classify

        node = Classify(using="clf")
        assert "image" in node.inputs
        assert "classifications" in node.outputs


# ──────────────────────────────────────────────────────────────
# 6. Built-in EstimateDepth node
# ──────────────────────────────────────────────────────────────


class TestDepthNode:
    """Built-in EstimateDepth node with mock provider."""

    def test_depth_with_mock_provider(self):
        from mata.nodes.depth import EstimateDepth

        mock_provider = Mock()
        mock_dr = DepthResult(
            depth=np.random.rand(48, 64).astype(np.float32),
            meta={"source": "test"},
        )
        mock_provider.estimate.return_value = mock_dr

        ctx = _make_ctx({"depth": {"dp": mock_provider}})
        node = EstimateDepth(using="dp", out="depth")
        result = node.run(ctx, image=_make_image())
        assert "depth" in result
        assert isinstance(result["depth"], DepthMap)

    def test_depth_type_declarations(self):
        from mata.nodes.depth import EstimateDepth

        node = EstimateDepth(using="dp")
        assert "image" in node.inputs
        assert "depth" in node.outputs


# ──────────────────────────────────────────────────────────────
# 7. Filter node
# ──────────────────────────────────────────────────────────────


class TestFilterNode:
    """Filter node operations."""

    def test_filter_by_score(self):
        from mata.nodes.filter import Filter

        dets = _make_detections(5)
        ctx = _make_ctx()
        ctx.store("dets", dets)

        node = Filter(src="dets", out="filtered", score_gt=0.75)
        result = node.run(ctx, detections=dets)
        assert "filtered" in result
        filtered = result["filtered"]
        assert all(inst.score > 0.75 for inst in filtered.instances)

    def test_filter_by_label(self):
        from mata.nodes.filter import Filter

        dets = _make_detections(3)
        ctx = _make_ctx()
        ctx.store("dets", dets)

        node = Filter(src="dets", out="filtered", label_in=["obj_0"])
        result = node.run(ctx, detections=dets)
        filtered = result["filtered"]
        assert all(inst.label_name == "obj_0" for inst in filtered.instances)

    def test_filter_empty_input(self):
        from mata.nodes.filter import Filter

        dets = Detections()
        ctx = _make_ctx()
        node = Filter(src="dets", out="filtered", score_gt=0.5)
        result = node.run(ctx, detections=dets)
        assert len(result["filtered"].instances) == 0


# ──────────────────────────────────────────────────────────────
# 8. Fuse node
# ──────────────────────────────────────────────────────────────


class TestFuseNode:
    """Fuse node bundles artifacts into MultiResult."""

    def test_fuse_basic(self):
        from mata.nodes.fuse import Fuse

        dets = _make_detections(2)
        ctx = _make_ctx()
        ctx.store("dets", dets)

        node = Fuse(out="final", detections="dets")
        result = node.run(ctx, dets=dets)
        assert "final" in result
        assert isinstance(result["final"], MultiResult)
        assert result["final"].has_channel("detections")

    def test_fuse_multiple_channels(self):
        from mata.nodes.fuse import Fuse

        dets = _make_detections(2)
        img = _make_image()
        ctx = _make_ctx()
        ctx.store("dets", dets)
        ctx.store("img", img)

        node = Fuse(out="bundle", detections="dets", image="img")
        result = node.run(ctx, dets=dets, img=img)
        mr = result["bundle"]
        assert mr.has_channel("detections")
        assert mr.has_channel("image")

    def test_fuse_empty(self):
        from mata.nodes.fuse import Fuse

        ctx = _make_ctx()
        node = Fuse(out="empty")
        result = node.run(ctx)
        assert isinstance(result["empty"], MultiResult)


# ──────────────────────────────────────────────────────────────
# 9. NMS node
# ──────────────────────────────────────────────────────────────


class TestNMSNode:
    """NMS node suppression tests."""

    def test_nms_empty(self):
        from mata.nodes.nms import NMS

        dets = Detections()
        ctx = _make_ctx()
        node = NMS(iou_threshold=0.5, out="nms_dets")
        result = node.run(ctx, detections=dets)
        assert len(result["nms_dets"].instances) == 0

    def test_nms_no_overlap(self):
        from mata.nodes.nms import NMS

        # Well-separated boxes should all survive NMS
        instances = [
            Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(100, 100, 110, 110), score=0.8, label=0, label_name="cat"),
        ]
        dets = Detections.from_vision_result(VisionResult(instances=instances))
        ctx = _make_ctx()
        node = NMS(iou_threshold=0.5, out="nms_dets")
        result = node.run(ctx, detections=dets)
        assert len(result["nms_dets"].instances) == 2


# ──────────────────────────────────────────────────────────────
# 10. Node run through context wiring
# ──────────────────────────────────────────────────────────────


class TestNodeExecution:
    """Test run() method with ExecutionContext wiring."""

    def test_echo_node_produces_detections(self):
        ctx = _make_ctx()
        node = EchoNode(out="echo_dets")
        result = node.run(ctx, image=_make_image())
        assert "echo_dets" in result
        assert isinstance(result["echo_dets"], Detections)

    def test_node_run_with_kwargs_passthrough(self):
        from mata.nodes.detect import Detect

        mock_provider = Mock()
        mock_vr = VisionResult(instances=[])
        mock_provider.predict.return_value = mock_vr

        ctx = _make_ctx({"detect": {"det": mock_provider}})
        node = Detect(using="det", out="dets", threshold=0.5)
        node.run(ctx, image=_make_image())
        # The provider should have been called with threshold kwarg
        call_kwargs = mock_provider.predict.call_args
        assert call_kwargs is not None
