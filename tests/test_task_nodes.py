"""Unit tests for Task 5.1: Core Task Nodes.

Tests cover:
- Each node with mock provider
- Output naming
- Kwargs passthrough
- Error handling (missing provider)
- Type validation
- Metrics recording
- Artifact conversion from adapter results
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.context import ExecutionContext
from mata.core.graph.node import Node
from mata.core.types import (
    Classification,
    ClassifyResult,
    DepthResult,
    Instance,
    VisionResult,
)
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth
from mata.nodes.detect import Detect
from mata.nodes.segment import SegmentImage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    """Create a minimal Image artifact for testing."""
    data = np.zeros((100, 200, 3), dtype=np.uint8)
    return Image(data=data, width=200, height=100)


@pytest.fixture
def sample_vision_result():
    """VisionResult with two instances (what a detect adapter returns)."""
    instances = [
        Instance(
            bbox=(10, 20, 100, 150),
            score=0.95,
            label=0,
            label_name="cat",
        ),
        Instance(
            bbox=(200, 50, 300, 200),
            score=0.82,
            label=1,
            label_name="dog",
        ),
    ]
    return VisionResult(instances=instances, meta={"model": "detr"})


@pytest.fixture
def sample_detections(sample_vision_result):
    """Detections artifact from a VisionResult."""
    return Detections.from_vision_result(sample_vision_result)


@pytest.fixture
def sample_classify_result():
    """ClassifyResult with predictions."""
    preds = [
        Classification(label=0, score=0.92, label_name="cat"),
        Classification(label=1, score=0.05, label_name="dog"),
        Classification(label=2, score=0.03, label_name="bird"),
    ]
    return ClassifyResult(predictions=preds, meta={"model": "resnet50"})


@pytest.fixture
def sample_depth_result():
    """DepthResult with a dummy depth map."""
    depth = np.random.rand(100, 200).astype(np.float32)
    return DepthResult(depth=depth, meta={"model": "depth-anything"})


@pytest.fixture
def sample_masks_vision_result():
    """VisionResult with mask data for segmentation."""
    instances = [
        Instance(
            bbox=(10, 20, 100, 150),
            score=0.95,
            label=0,
            label_name="cat",
            mask={"size": [100, 200], "counts": "abc"},
        ),
    ]
    return VisionResult(instances=instances, meta={"model": "mask2former"})


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    """Helper to build an ExecutionContext with given providers."""
    return ExecutionContext(providers=providers or {}, device="cpu")


# ===========================================================================
# Detect node tests
# ===========================================================================


class TestDetectNode:
    """Tests for the Detect node."""

    def test_basic_detection_with_mock(self, sample_image, sample_vision_result):
        """Mock provider returns VisionResult, node converts to Detections."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_vision_result

        ctx = _make_ctx({"detect": {"detr": mock_detector}})
        node = Detect(using="detr")
        result = node.run(ctx, image=sample_image)

        assert "dets" in result
        dets = result["dets"]
        assert isinstance(dets, Detections)
        assert len(dets.instances) == 2
        assert dets.instances[0].label_name == "cat"
        mock_detector.predict.assert_called_once_with(sample_image)

    def test_custom_output_name(self, sample_image, sample_vision_result):
        """Output key respects the `out` parameter."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_vision_result

        ctx = _make_ctx({"detect": {"yolo": mock_detector}})
        node = Detect(using="yolo", out="my_detections")
        result = node.run(ctx, image=sample_image)

        assert "my_detections" in result
        assert "dets" not in result

    def test_kwargs_passthrough(self, sample_image, sample_vision_result):
        """Extra kwargs are forwarded to the provider."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_vision_result

        ctx = _make_ctx({"detect": {"detr": mock_detector}})
        node = Detect(using="detr", threshold=0.7, nms_iou=0.45)
        node.run(ctx, image=sample_image)

        mock_detector.predict.assert_called_once_with(sample_image, threshold=0.7, nms_iou=0.45)

    def test_missing_provider_raises(self, sample_image):
        """KeyError when provider is not registered."""
        ctx = _make_ctx({})
        node = Detect(using="nonexistent")
        with pytest.raises(KeyError, match="detect"):
            node.run(ctx, image=sample_image)

    def test_missing_provider_name_raises(self, sample_image, sample_vision_result):
        """KeyError when provider name is wrong."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_vision_result

        ctx = _make_ctx({"detect": {"detr": mock_detector}})
        node = Detect(using="yolo")
        with pytest.raises(KeyError, match="yolo"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, sample_vision_result):
        """Node records latency and detection count."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_vision_result

        ctx = _make_ctx({"detect": {"detr": mock_detector}})
        node = Detect(using="detr", name="det_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "det_node" in metrics
        assert "latency_ms" in metrics["det_node"]
        assert metrics["det_node"]["latency_ms"] >= 0
        assert metrics["det_node"]["num_detections"] == 2

    def test_passthrough_detections_artifact(self, sample_image, sample_detections):
        """If provider already returns Detections, no conversion needed."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = sample_detections

        ctx = _make_ctx({"detect": {"detr": mock_detector}})
        node = Detect(using="detr")
        result = node.run(ctx, image=sample_image)

        assert result["dets"] is sample_detections

    def test_type_declarations(self):
        """Verify class-level input/output type declarations."""
        assert Detect.inputs == {"image": Image}
        assert Detect.outputs == {"detections": Detections}

    def test_repr(self):
        """Node repr includes provider, output name, and kwargs."""
        node = Detect(using="detr", out="dets", threshold=0.5)
        r = repr(node)
        assert "detr" in r
        assert "dets" in r
        assert "threshold" in r

    def test_default_name(self):
        """Default name is the class name."""
        node = Detect(using="detr")
        assert node.name == "Detect"

    def test_custom_name(self):
        """Custom name overrides default."""
        node = Detect(using="detr", name="my_detector")
        assert node.name == "my_detector"


# ===========================================================================
# Classify node tests
# ===========================================================================


class TestClassifyNode:
    """Tests for the Classify node."""

    def test_basic_classification_with_mock(self, sample_image, sample_classify_result):
        """Mock provider returns ClassifyResult, node converts to Classifications."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = sample_classify_result

        ctx = _make_ctx({"classify": {"resnet": mock_classifier}})
        node = Classify(using="resnet")
        result = node.run(ctx, image=sample_image)

        assert "classifications" in result
        cls = result["classifications"]
        assert isinstance(cls, Classifications)
        assert len(cls) == 3
        assert cls.top1.label_name == "cat"
        mock_classifier.classify.assert_called_once_with(sample_image)

    def test_custom_output_name(self, sample_image, sample_classify_result):
        """Output key respects the `out` parameter."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = sample_classify_result

        ctx = _make_ctx({"classify": {"clip": mock_classifier}})
        node = Classify(using="clip", out="cls")
        result = node.run(ctx, image=sample_image)

        assert "cls" in result
        assert "classifications" not in result

    def test_kwargs_passthrough(self, sample_image, sample_classify_result):
        """Extra kwargs forwarded to provider."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = sample_classify_result

        ctx = _make_ctx({"classify": {"clip": mock_classifier}})
        node = Classify(using="clip", top_k=5, text_prompts=["cat", "dog"])
        node.run(ctx, image=sample_image)

        mock_classifier.classify.assert_called_once_with(sample_image, top_k=5, text_prompts=["cat", "dog"])

    def test_missing_provider_raises(self, sample_image):
        """KeyError when classify capability not registered."""
        ctx = _make_ctx({})
        node = Classify(using="resnet")
        with pytest.raises(KeyError, match="classify"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, sample_classify_result):
        """Node records latency and prediction count."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = sample_classify_result

        ctx = _make_ctx({"classify": {"resnet": mock_classifier}})
        node = Classify(using="resnet", name="cls_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "cls_node" in metrics
        assert metrics["cls_node"]["num_predictions"] == 3

    def test_passthrough_classifications_artifact(self, sample_image):
        """If provider already returns Classifications, no conversion needed."""
        preds = (Classification(label=0, score=0.9, label_name="cat"),)
        artifact = Classifications(predictions=preds)

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = artifact

        ctx = _make_ctx({"classify": {"resnet": mock_classifier}})
        node = Classify(using="resnet")
        result = node.run(ctx, image=sample_image)

        assert result["classifications"] is artifact

    def test_type_declarations(self):
        """Verify class-level input/output type declarations."""
        assert Classify.inputs == {"image": Image}
        assert Classify.outputs == {"classifications": Classifications}

    def test_repr(self):
        node = Classify(using="clip", out="cls", top_k=3)
        r = repr(node)
        assert "clip" in r
        assert "cls" in r
        assert "top_k" in r


# ===========================================================================
# SegmentImage node tests
# ===========================================================================


class TestSegmentImageNode:
    """Tests for the SegmentImage node."""

    def test_basic_segmentation_with_mock(self, sample_image, sample_masks_vision_result):
        """Mock provider returns VisionResult with masks, node converts to Masks."""
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = sample_masks_vision_result

        ctx = _make_ctx({"segment": {"mask2former": mock_segmenter}})
        node = SegmentImage(using="mask2former")
        result = node.run(ctx, image=sample_image)

        assert "masks" in result
        masks = result["masks"]
        assert isinstance(masks, Masks)
        assert len(masks.instances) == 1
        mock_segmenter.segment.assert_called_once_with(sample_image)

    def test_custom_output_name(self, sample_image, sample_masks_vision_result):
        """Output key respects the `out` parameter."""
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = sample_masks_vision_result

        ctx = _make_ctx({"segment": {"sam": mock_segmenter}})
        node = SegmentImage(using="sam", out="seg_masks")
        result = node.run(ctx, image=sample_image)

        assert "seg_masks" in result

    def test_kwargs_passthrough(self, sample_image, sample_masks_vision_result):
        """Extra kwargs forwarded to provider."""
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = sample_masks_vision_result

        ctx = _make_ctx({"segment": {"sam": mock_segmenter}})
        node = SegmentImage(using="sam", threshold=0.8, mode="everything")
        node.run(ctx, image=sample_image)

        mock_segmenter.segment.assert_called_once_with(sample_image, threshold=0.8, mode="everything")

    def test_missing_provider_raises(self, sample_image):
        """KeyError when segment capability not registered."""
        ctx = _make_ctx({})
        node = SegmentImage(using="sam")
        with pytest.raises(KeyError, match="segment"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, sample_masks_vision_result):
        """Node records latency and mask count."""
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = sample_masks_vision_result

        ctx = _make_ctx({"segment": {"mask2former": mock_segmenter}})
        node = SegmentImage(using="mask2former", name="seg_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "seg_node" in metrics
        assert metrics["seg_node"]["num_masks"] == 1

    def test_passthrough_masks_artifact(self, sample_image):
        """If provider already returns Masks, no conversion needed."""
        masks = Masks(instances=[], instance_ids=[], meta={})

        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = masks

        ctx = _make_ctx({"segment": {"sam": mock_segmenter}})
        node = SegmentImage(using="sam")
        result = node.run(ctx, image=sample_image)

        assert result["masks"] is masks

    def test_type_declarations(self):
        """Verify class-level input/output type declarations."""
        assert SegmentImage.inputs == {"image": Image}
        assert SegmentImage.outputs == {"masks": Masks}

    def test_repr(self):
        node = SegmentImage(using="sam", out="seg", mode="boxes")
        r = repr(node)
        assert "sam" in r
        assert "seg" in r


# ===========================================================================
# EstimateDepth node tests
# ===========================================================================


class TestEstimateDepthNode:
    """Tests for the EstimateDepth node."""

    def test_basic_depth_with_mock(self, sample_image, sample_depth_result):
        """Mock provider returns DepthResult, node converts to DepthMap."""
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = sample_depth_result

        ctx = _make_ctx({"depth": {"depth-anything": mock_estimator}})
        node = EstimateDepth(using="depth-anything")
        result = node.run(ctx, image=sample_image)

        assert "depth" in result
        dm = result["depth"]
        assert isinstance(dm, DepthMap)
        assert dm.height == 100
        assert dm.width == 200
        mock_estimator.estimate.assert_called_once_with(sample_image)

    def test_custom_output_name(self, sample_image, sample_depth_result):
        """Output key respects the `out` parameter."""
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = sample_depth_result

        ctx = _make_ctx({"depth": {"midas": mock_estimator}})
        node = EstimateDepth(using="midas", out="depth_map")
        result = node.run(ctx, image=sample_image)

        assert "depth_map" in result
        assert "depth" not in result

    def test_kwargs_passthrough(self, sample_image, sample_depth_result):
        """Extra kwargs forwarded to provider."""
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = sample_depth_result

        ctx = _make_ctx({"depth": {"midas": mock_estimator}})
        node = EstimateDepth(using="midas", output_type="normalized")
        node.run(ctx, image=sample_image)

        mock_estimator.estimate.assert_called_once_with(sample_image, output_type="normalized")

    def test_missing_provider_raises(self, sample_image):
        """KeyError when depth capability not registered."""
        ctx = _make_ctx({})
        node = EstimateDepth(using="midas")
        with pytest.raises(KeyError, match="depth"):
            node.run(ctx, image=sample_image)

    def test_records_metrics(self, sample_image, sample_depth_result):
        """Node records latency and map dimensions."""
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = sample_depth_result

        ctx = _make_ctx({"depth": {"depth-anything": mock_estimator}})
        node = EstimateDepth(using="depth-anything", name="depth_node")
        node.run(ctx, image=sample_image)

        metrics = ctx.get_metrics()
        assert "depth_node" in metrics
        assert metrics["depth_node"]["height"] == 100
        assert metrics["depth_node"]["width"] == 200

    def test_passthrough_depth_map_artifact(self, sample_image):
        """If provider already returns DepthMap, no conversion needed."""
        dm = DepthMap(depth=np.zeros((50, 50), dtype=np.float32))

        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = dm

        ctx = _make_ctx({"depth": {"midas": mock_estimator}})
        node = EstimateDepth(using="midas")
        result = node.run(ctx, image=sample_image)

        assert result["depth"] is dm

    def test_type_declarations(self):
        """Verify class-level input/output type declarations."""
        assert EstimateDepth.inputs == {"image": Image}
        assert EstimateDepth.outputs == {"depth": DepthMap}

    def test_repr(self):
        node = EstimateDepth(using="midas", out="d", colormap="magma")
        r = repr(node)
        assert "midas" in r
        assert "colormap" in r


# ===========================================================================
# Cross-cutting concerns
# ===========================================================================


class TestNodeInheritance:
    """All four nodes inherit from Node correctly."""

    @pytest.mark.parametrize("cls", [Detect, Classify, SegmentImage, EstimateDepth])
    def test_is_node_subclass(self, cls):
        assert issubclass(cls, Node)

    @pytest.mark.parametrize(
        "cls,using",
        [
            (Detect, "detr"),
            (Classify, "resnet"),
            (SegmentImage, "sam"),
            (EstimateDepth, "midas"),
        ],
    )
    def test_abstract_run_implemented(self, cls, using):
        """Sanity check that run() is concrete."""
        node = cls(using=using)
        # Should not raise TypeError about abstract methods
        assert hasattr(node, "run")


class TestEmptyResults:
    """Nodes handle empty results gracefully."""

    def test_detect_empty(self, sample_image):
        mock = MagicMock()
        mock.predict.return_value = VisionResult(instances=[], meta={})

        ctx = _make_ctx({"detect": {"detr": mock}})
        node = Detect(using="detr")
        result = node.run(ctx, image=sample_image)

        assert len(result["dets"].instances) == 0

    def test_classify_empty(self, sample_image):
        mock = MagicMock()
        mock.classify.return_value = ClassifyResult(predictions=[], meta={})

        ctx = _make_ctx({"classify": {"resnet": mock}})
        node = Classify(using="resnet")
        result = node.run(ctx, image=sample_image)

        assert len(result["classifications"]) == 0

    def test_segment_empty(self, sample_image):
        """Empty VisionResult raises ValueError in Masks.from_vision_result,
        so the node should propagate that error."""
        mock = MagicMock()
        mock.segment.return_value = VisionResult(instances=[], meta={})

        ctx = _make_ctx({"segment": {"sam": mock}})
        node = SegmentImage(using="sam")
        with pytest.raises(ValueError, match="no instances with masks"):
            node.run(ctx, image=sample_image)

    def test_segment_empty_masks_artifact(self, sample_image):
        """If provider returns an empty Masks artifact directly, passthrough works."""
        mock = MagicMock()
        mock.segment.return_value = Masks(instances=[], instance_ids=[], meta={})

        ctx = _make_ctx({"segment": {"sam": mock}})
        node = SegmentImage(using="sam")
        result = node.run(ctx, image=sample_image)
        assert len(result["masks"].instances) == 0

    def test_depth_zero_map(self, sample_image):
        mock = MagicMock()
        mock.estimate.return_value = DepthResult(depth=np.zeros((10, 10), dtype=np.float32), meta={})

        ctx = _make_ctx({"depth": {"midas": mock}})
        node = EstimateDepth(using="midas")
        result = node.run(ctx, image=sample_image)

        assert result["depth"].height == 10


# ===========================================================================
# Classifications artifact tests
# ===========================================================================


class TestClassificationsArtifact:
    """Tests for the Classifications wrapper artifact."""

    def test_from_classify_result(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        assert len(cls) == 3
        assert cls.top1.label_name == "cat"

    def test_to_classify_result(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        restored = cls.to_classify_result()
        assert isinstance(restored, ClassifyResult)
        assert len(restored.predictions) == 3

    def test_top5(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        top5 = cls.top5
        assert len(top5) == 3  # only 3 predictions
        assert top5[0].score >= top5[1].score

    def test_labels_and_scores(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        assert cls.labels == ["cat", "dog", "bird"]
        assert len(cls.scores) == 3

    def test_serialization_roundtrip(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        d = cls.to_dict()
        restored = Classifications.from_dict(d)
        assert len(restored) == len(cls)
        assert restored.top1.label_name == cls.top1.label_name

    def test_empty(self):
        cls = Classifications(predictions=())
        assert cls.top1 is None
        assert len(cls) == 0

    def test_validate_ok(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        cls.validate()  # should not raise

    def test_is_artifact(self, sample_classify_result):
        cls = Classifications.from_classify_result(sample_classify_result)
        assert isinstance(cls, Artifact)


# ===========================================================================
# DepthMap artifact tests
# ===========================================================================


class TestDepthMapArtifact:
    """Tests for the DepthMap wrapper artifact."""

    def test_from_depth_result(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        assert dm.height == 100
        assert dm.width == 200

    def test_to_depth_result(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        restored = dm.to_depth_result()
        assert isinstance(restored, DepthResult)
        np.testing.assert_array_equal(restored.depth, sample_depth_result.depth)

    def test_shape(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        assert dm.shape == (100, 200)

    def test_serialization_roundtrip(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        d = dm.to_dict()
        restored = DepthMap.from_dict(d)
        assert restored.shape == dm.shape

    def test_validate_ok(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        dm.validate()  # should not raise

    def test_validate_none_raises(self):
        dm = DepthMap(depth=None)
        with pytest.raises(ValueError, match="required"):
            dm.validate()

    def test_validate_3d_raises(self):
        dm = DepthMap(depth=np.zeros((10, 10, 3)))
        with pytest.raises(ValueError, match="2D"):
            dm.validate()

    def test_is_artifact(self, sample_depth_result):
        dm = DepthMap.from_depth_result(sample_depth_result)
        assert isinstance(dm, Artifact)


# ===========================================================================
# Converter integration tests
# ===========================================================================


class TestConverterIntegration:
    """Ensure the converters now work with the new artifact types."""

    def test_classify_result_to_artifact(self, sample_classify_result):
        from mata.core.artifacts.converters import classify_result_to_artifact

        artifact = classify_result_to_artifact(sample_classify_result)
        assert isinstance(artifact, Classifications)
        assert artifact.top1.label_name == "cat"

    def test_artifact_to_classify_result(self, sample_classify_result):
        from mata.core.artifacts.converters import (
            artifact_to_classify_result,
            classify_result_to_artifact,
        )

        artifact = classify_result_to_artifact(sample_classify_result)
        restored = artifact_to_classify_result(artifact)
        assert isinstance(restored, ClassifyResult)

    def test_depth_result_to_artifact(self, sample_depth_result):
        from mata.core.artifacts.converters import depth_result_to_artifact

        artifact = depth_result_to_artifact(sample_depth_result)
        assert isinstance(artifact, DepthMap)
        assert artifact.height == 100

    def test_artifact_to_depth_result(self, sample_depth_result):
        from mata.core.artifacts.converters import (
            artifact_to_depth_result,
            depth_result_to_artifact,
        )

        artifact = depth_result_to_artifact(sample_depth_result)
        restored = artifact_to_depth_result(artifact)
        assert isinstance(restored, DepthResult)
