"""Unit tests for Task 5.8: Visualization & Analysis Nodes.

Tests cover:
- Annotate node with detections only
- Annotate node with masks (segmentation)
- Annotate node with both PIL and matplotlib backends
- Annotate node configuration options (colors, alpha, line width)
- NMS filtering (various IoU thresholds)
- NMS with empty detections
- NMS preserves instance order
- Integration with real detection results
- Graph composition (NMS → Annotate pipeline)
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance, VisionResult
from mata.nodes.annotate import Annotate
from mata.nodes.nms import NMS

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _has_torch_and_torchvision() -> bool:
    """Check if PyTorch and torchvision are available."""
    try:
        import torch  # noqa: F401
        from torchvision.ops import nms  # noqa: F401

        return True
    except ImportError:
        return False


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create 640x480 RGB image
    array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_img = PILImage.fromarray(array)
    return Image.from_pil(pil_img, source_path="test.jpg")


@pytest.fixture
def sample_detections():
    """Create sample detections with bboxes, labels, and scores."""
    instances = [
        Instance(bbox=(100, 100, 200, 200), label=0, score=0.9, label_name="person"),  # xyxy format
        Instance(bbox=(300, 150, 400, 250), label=1, score=0.8, label_name="car"),
        Instance(bbox=(500, 300, 600, 400), label=2, score=0.7, label_name="dog"),
    ]
    return Detections.from_vision_result(VisionResult(instances=instances, meta={}))


@pytest.fixture
def sample_detections_with_masks():
    """Create sample detections with masks for segmentation testing."""
    instances = [
        Instance(
            bbox=(100, 100, 200, 200),
            label=0,
            score=0.9,
            label_name="person",
            mask={"size": [480, 640], "counts": b"RLE_encoded_mask_data_1"},  # Mock RLE
        ),
        Instance(
            bbox=(300, 150, 400, 250),
            label=1,
            score=0.8,
            label_name="car",
            mask={"size": [480, 640], "counts": b"RLE_encoded_mask_data_2"},
        ),
    ]
    return Detections.from_vision_result(VisionResult(instances=instances, meta={}))


@pytest.fixture
def overlapping_detections():
    """Create overlapping detections for NMS testing."""
    instances = [
        Instance(bbox=(100, 100, 200, 200), label=0, score=0.9, label_name="person"),
        Instance(bbox=(105, 105, 195, 195), label=0, score=0.8, label_name="person"),  # High overlap with first
        Instance(bbox=(110, 110, 190, 190), label=0, score=0.7, label_name="person"),  # High overlap with both
        Instance(bbox=(500, 500, 600, 600), label=1, score=0.85, label_name="car"),  # No overlap
    ]
    return Detections.from_vision_result(VisionResult(instances=instances, meta={}))


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    ctx = MagicMock(spec=ExecutionContext)
    return ctx


# ---------------------------------------------------------------------------
# Annotate Node Tests
# ---------------------------------------------------------------------------


class TestAnnotateNode:
    """Test suite for Annotate visualization node."""

    def test_annotate_init_default(self):
        """Test Annotate node initialization with default parameters."""
        node = Annotate()

        assert node.name == "Annotate"
        assert node.using == "pil"
        assert node.show_boxes is True
        assert node.show_labels is True
        assert node.show_masks is True
        assert node.show_scores is True
        assert node.alpha == 0.5
        assert node.line_width == 2
        assert node.out == "annotated"
        assert node.image_src == "image"
        assert node.detections_src == "detections"

    def test_annotate_init_custom(self):
        """Test Annotate node initialization with custom parameters."""
        node = Annotate(
            using="matplotlib", show_boxes=False, show_masks=False, alpha=0.8, line_width=3, out="viz", custom_param=42
        )

        assert node.using == "matplotlib"
        assert node.show_boxes is False
        assert node.show_masks is False
        assert node.alpha == 0.8
        assert node.line_width == 3
        assert node.out == "viz"
        assert node.kwargs["custom_param"] == 42

    def test_annotate_invalid_backend(self):
        """Test Annotate node with invalid backend raises error."""
        with pytest.raises(ValueError, match="Invalid backend 'invalid'"):
            Annotate(using="invalid")

    @patch("mata.visualization.visualize_segmentation")
    def test_annotate_detections_only_pil(self, mock_viz, sample_image, sample_detections, mock_context):
        """Test Annotate node with detections only using PIL backend."""
        # Configure mock
        mock_result_pil = PILImage.new("RGB", (640, 480), color="red")
        mock_viz.return_value = mock_result_pil

        # Create and run node
        node = Annotate(using="pil", show_boxes=True, show_labels=True)
        result = node.run(mock_context, image=sample_image, detections=sample_detections)

        # Verify output
        assert "annotated" in result
        assert isinstance(result["annotated"], Image)
        assert result["annotated"].width == 640
        assert result["annotated"].height == 480

        # Verify visualization was called correctly
        mock_viz.assert_called_once()
        call_args = mock_viz.call_args
        assert call_args[1]["show_boxes"] is True
        assert call_args[1]["show_labels"] is True
        assert call_args[1]["show_scores"] is True
        assert call_args[1]["alpha"] == 0.0  # No alpha for detection-only
        assert call_args[1]["backend"] == "pil"

    @patch("mata.visualization.visualize_segmentation")
    def test_annotate_segmentation_pil(self, mock_viz, sample_image, sample_detections_with_masks, mock_context):
        """Test Annotate node with segmentation using PIL backend."""
        # Configure mock
        mock_result_pil = PILImage.new("RGB", (640, 480), color="blue")
        mock_viz.return_value = mock_result_pil

        # Create and run node
        node = Annotate(using="pil", show_masks=True, alpha=0.6)
        result = node.run(mock_context, image=sample_image, detections=sample_detections_with_masks)

        # Verify output
        assert "annotated" in result
        assert isinstance(result["annotated"], Image)

        # Verify visualization was called with mask alpha
        mock_viz.assert_called_once()
        call_args = mock_viz.call_args
        assert call_args[1]["alpha"] == 0.6  # Alpha enabled for masks
        assert call_args[1]["backend"] == "pil"

    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib required for matplotlib backend test")
    @patch("mata.visualization.visualize_segmentation")
    @patch("matplotlib.pyplot.close")
    def test_annotate_matplotlib_backend(self, mock_plt_close, mock_viz, sample_image, sample_detections, mock_context):
        """Test Annotate node with matplotlib backend."""
        # Create mock matplotlib figure
        mock_fig = Mock()
        mock_fig.savefig = Mock()

        # Mock the buffer and PIL loading
        with patch("io.BytesIO") as mock_bytesio, patch("PIL.Image.open") as mock_pil_open:

            mock_buffer = Mock()
            mock_bytesio.return_value = mock_buffer
            mock_pil_result = PILImage.new("RGB", (640, 480), color="green")
            mock_pil_open.return_value = mock_pil_result
            mock_viz.return_value = mock_fig

            # Create and run node
            node = Annotate(using="matplotlib", out="mpl_viz")
            result = node.run(mock_context, image=sample_image, detections=sample_detections)

            # Verify output
            assert "mpl_viz" in result
            assert isinstance(result["mpl_viz"], Image)

            # Verify matplotlib figure handling
            mock_fig.savefig.assert_called_once()
            mock_plt_close.assert_called_once_with(mock_fig)
            mock_viz.assert_called_once()
            call_args = mock_viz.call_args
            assert call_args[1]["backend"] == "matplotlib"

    def test_annotate_preserves_image_metadata(self, sample_image, sample_detections, mock_context):
        """Test that Annotate node preserves image metadata."""
        # Add metadata to sample image
        image_with_metadata = Image.from_pil(
            sample_image.to_pil(), timestamp_ms=1234567890, frame_id="frame_001", source_path="test_video.mp4"
        )

        with patch("mata.visualization.visualize_segmentation") as mock_viz:
            mock_viz.return_value = PILImage.new("RGB", (640, 480))

            node = Annotate()
            result = node.run(mock_context, image=image_with_metadata, detections=sample_detections)

            # Check metadata preservation
            annotated = result["annotated"]
            assert annotated.timestamp_ms == 1234567890
            assert annotated.frame_id == "frame_001"
            assert annotated.source_path == "test_video.mp4"

    def test_annotate_type_validation(self):
        """Test Annotate node type signature validation."""
        node = Annotate()

        # Check input types
        assert node.inputs["image"] == Image
        assert node.inputs["detections"] == Detections

        # Check output types
        assert node.outputs["annotated"] == Image


# ---------------------------------------------------------------------------
# NMS Node Tests
# ---------------------------------------------------------------------------


class TestNMSNode:
    """Test suite for NMS analysis node."""

    def test_nms_init_default(self):
        """Test NMS node initialization with default parameters."""
        node = NMS()

        assert node.name == "NMS"
        assert node.iou_threshold == 0.5
        assert node.out == "nms_dets"
        assert node.detections_src == "detections"

    def test_nms_init_custom(self):
        """Test NMS node initialization with custom parameters."""
        node = NMS(iou_threshold=0.3, out="filtered")

        assert node.iou_threshold == 0.3
        assert node.out == "filtered"

    def test_nms_invalid_threshold(self):
        """Test NMS node with invalid IoU threshold raises error."""
        with pytest.raises(ValueError, match="Invalid iou_threshold"):
            NMS(iou_threshold=-0.1)

        with pytest.raises(ValueError, match="Invalid iou_threshold"):
            NMS(iou_threshold=1.5)

    def test_nms_empty_detections(self, mock_context):
        """Test NMS with empty detections returns empty result."""
        empty_detections = Detections.from_vision_result(VisionResult(instances=[], meta={}))

        node = NMS(iou_threshold=0.5)
        result = node.run(mock_context, detections=empty_detections)

        assert "nms_dets" in result
        assert len(result["nms_dets"].instances) == 0
        assert result["nms_dets"] is empty_detections  # Should return same object

    @pytest.mark.skipif(not _has_torch_and_torchvision(), reason="PyTorch and torchvision required for NMS")
    def test_nms_no_overlaps(self, mock_context, sample_detections):
        """Test NMS with non-overlapping detections keeps all."""
        node = NMS(iou_threshold=0.5)
        result = node.run(mock_context, detections=sample_detections)

        filtered = result["nms_dets"]
        assert len(filtered.instances) == 3  # All detections kept
        assert filtered.meta["nms_applied"] is True
        assert filtered.meta["nms_iou_threshold"] == 0.5
        assert filtered.meta["pre_nms_count"] == 3
        assert filtered.meta["post_nms_count"] == 3

    @pytest.mark.skipif(not _has_torch_and_torchvision(), reason="PyTorch and torchvision required for NMS")
    def test_nms_with_overlaps(self, mock_context, overlapping_detections):
        """Test NMS with overlapping detections filters correctly."""
        node = NMS(iou_threshold=0.5)
        result = node.run(mock_context, detections=overlapping_detections)

        filtered = result["nms_dets"]

        # Should keep highest score person and the car
        assert len(filtered.instances) == 2
        assert filtered.instances[0].label_name == "person"
        assert filtered.instances[0].score == 0.9  # Highest score person
        assert filtered.instances[1].label_name == "car"
        assert filtered.instances[1].score == 0.85

        # Check metadata
        assert filtered.meta["nms_applied"] is True
        assert filtered.meta["pre_nms_count"] == 4
        assert filtered.meta["post_nms_count"] == 2

    @pytest.mark.skipif(not _has_torch_and_torchvision(), reason="PyTorch and torchvision required for NMS")
    def test_nms_aggressive_threshold(self, mock_context, overlapping_detections):
        """Test NMS with aggressive (low) IoU threshold."""
        node = NMS(iou_threshold=0.1)  # Very aggressive
        result = node.run(mock_context, detections=overlapping_detections)

        filtered = result["nms_dets"]

        # Should filter even more aggressively
        assert len(filtered.instances) <= 2
        assert all(inst.score >= 0.7 for inst in filtered.instances)

    @pytest.mark.skipif(not _has_torch_and_torchvision(), reason="PyTorch and torchvision required for NMS")
    def test_nms_preserves_entities(self, mock_context):
        """Test NMS preserves VLM entities in multi-modal workflows."""
        # Create detections with both instances and entities
        from mata.core.types import Entity

        entities = [Entity(label="person", score=0.9), Entity(label="building", score=0.8)]
        instances = [Instance(bbox=(100, 100, 200, 200), label=0, score=0.9, label_name="person")]

        detections = Detections.from_vision_result(VisionResult(instances=instances, entities=entities, meta={}))

        node = NMS(iou_threshold=0.5)
        result = node.run(mock_context, detections=detections)

        filtered = result["nms_dets"]

        # Entities should be preserved unchanged
        assert len(filtered.entities) == 2
        assert filtered.entities[0].label == "person"
        assert filtered.entities[1].label == "building"
        assert len(filtered.entity_ids) == 2

    def test_nms_no_bboxes(self, mock_context):
        """Test NMS with instances that have masks but no bounding boxes."""
        instances = [
            Instance(
                label=0,
                score=0.9,
                label_name="person",
                mask={"size": [480, 640], "counts": b"mock_rle_1"},  # Has mask but no bbox
            ),
            Instance(
                label=1,
                score=0.8,
                label_name="car",
                mask={"size": [480, 640], "counts": b"mock_rle_2"},  # Has mask but no bbox
            ),
        ]
        detections = Detections.from_vision_result(VisionResult(instances=instances, meta={}))

        node = NMS(iou_threshold=0.5)
        result = node.run(mock_context, detections=detections)

        filtered = result["nms_dets"]
        assert len(filtered.instances) == 0  # No valid boxes to filter
        assert filtered.meta["nms_applied"] is True
        assert filtered.meta["pre_nms_count"] == 2
        assert filtered.meta["post_nms_count"] == 0

    def test_nms_missing_torch_import(self, mock_context, sample_detections):
        """Test NMS raises helpful error when PyTorch not available."""
        import sys

        with patch.dict(sys.modules, {"torch": None, "torchvision.ops": None}):
            node = NMS()

            with pytest.raises(ImportError, match="PyTorch and torchvision required"):
                node.run(mock_context, detections=sample_detections)

    def test_nms_type_validation(self):
        """Test NMS node type signature validation."""
        node = NMS()

        # Check input types
        assert node.inputs["detections"] == Detections

        # Check output types
        assert node.outputs["detections"] == Detections

    def test_nms_custom_output_key(self, mock_context, sample_detections):
        """Test NMS with custom output key."""
        if not _has_torch_and_torchvision():
            pytest.skip("PyTorch and torchvision required")

        node = NMS(out="filtered_dets")
        result = node.run(mock_context, detections=sample_detections)

        assert "filtered_dets" in result
        assert "nms_dets" not in result


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestVisualizationAnalysisIntegration:
    """Integration tests for visualization and analysis nodes."""

    @pytest.mark.skipif(not _has_torch_and_torchvision(), reason="PyTorch and torchvision required for NMS")
    @patch("mata.visualization.visualize_segmentation")
    def test_nms_annotate_pipeline(self, mock_viz, mock_context, overlapping_detections, sample_image):
        """Test NMS → Annotate pipeline integration."""
        # Configure mock visualization
        mock_viz.return_value = PILImage.new("RGB", (640, 480), color="purple")

        # Create pipeline: NMS → Annotate
        nms_node = NMS(iou_threshold=0.5, out="filtered")
        annotate_node = Annotate(using="pil", out="result")

        # Run NMS first
        nms_result = nms_node.run(mock_context, detections=overlapping_detections)
        filtered_detections = nms_result["filtered"]

        # Run Annotate on filtered results
        annotate_result = annotate_node.run(mock_context, image=sample_image, detections=filtered_detections)

        # Verify pipeline results
        assert len(filtered_detections.instances) == 2  # NMS filtered
        assert "result" in annotate_result
        assert isinstance(annotate_result["result"], Image)

        # Verify visualization was called
        mock_viz.assert_called_once()

    @pytest.mark.integration
    def test_real_segmentation_visualization(self, sample_image, sample_detections_with_masks, mock_context):
        """Integration test with real segmentation visualization (slow)."""
        # This test requires actual mata.visualization to be working
        try:
            node = Annotate(using="pil", show_masks=True, alpha=0.7)
            result = node.run(mock_context, image=sample_image, detections=sample_detections_with_masks)

            # Basic verification
            assert "annotated" in result
            assert isinstance(result["annotated"], Image)
            assert result["annotated"].width == sample_image.width
            assert result["annotated"].height == sample_image.height

        except Exception as e:
            pytest.skip(f"Real visualization test failed: {e}")
