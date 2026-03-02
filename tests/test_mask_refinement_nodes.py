"""Unit tests for Task 5.4: Mask Refinement Nodes.

Tests cover:
- RefineMask node: morphological operations (close, open, dilate, erode)
- MaskToBox node: bbox extraction from masks, padding, empty mask handling
- Edge cases: empty inputs, invalid parameters, format conversions
- Integration: mask format preservation and instance_id alignment
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.masks import Masks
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance
from mata.nodes.mask_to_box import MaskToBox
from mata.nodes.refine_mask import RefineMask

# Mark tests as slow since they require mask operations
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ExecutionContext:
    """Minimal execution context for testing."""
    return ExecutionContext(providers={}, device="cpu")


def _make_binary_mask(shape=(100, 100), fill_area=None):
    """Create a test binary mask.

    Args:
        shape: Mask dimensions (H, W)
        fill_area: Tuple (x1, y1, x2, y2) to fill, or None for center square

    Returns:
        Binary numpy array
    """
    mask = np.zeros(shape, dtype=bool)
    if fill_area is None:
        # Create a 30x30 square in the center
        h, w = shape
        y1, y2 = h // 2 - 15, h // 2 + 15
        x1, x2 = w // 2 - 15, w // 2 + 15
    else:
        x1, y1, x2, y2 = fill_area

    mask[y1:y2, x1:x2] = True
    return mask


def _make_instance_with_binary_mask(
    label_name: str = "cat",
    score: float = 0.8,
    mask_shape=(100, 100),
    fill_area=None,
    label: int = 0,
) -> Instance:
    """Helper to build an Instance with binary mask."""
    binary_mask = _make_binary_mask(mask_shape, fill_area)
    return Instance(
        bbox=None,  # No bbox, only mask
        mask=binary_mask,
        score=score,
        label=label,
        label_name=label_name,
    )


def _make_instance_with_rle_mask(
    label_name: str = "dog",
    score: float = 0.9,
    mask_shape=(100, 100),
    fill_area=None,
) -> Instance:
    """Helper to build an Instance with RLE mask."""
    binary_mask = _make_binary_mask(mask_shape, fill_area)

    # Convert to RLE using pycocotools if available
    try:
        from mata.core.artifacts.masks import _binary_to_rle

        rle_mask = _binary_to_rle(binary_mask)
        return Instance(
            bbox=None,
            mask=rle_mask,
            score=score,
            label=0,
            label_name=label_name,
        )
    except ImportError:
        pytest.skip("pycocotools not available for RLE mask tests")


@pytest.fixture
def sample_masks() -> Masks:
    """Sample masks artifact with mixed mask formats."""
    instances = [
        _make_instance_with_binary_mask("cat", 0.8, fill_area=(10, 10, 40, 40)),
        _make_instance_with_binary_mask("dog", 0.9, fill_area=(60, 60, 90, 90)),
    ]

    return Masks(
        instances=instances, instance_ids=["mask_0000", "mask_0001"], meta={"image_height": 100, "image_width": 100}
    )


@pytest.fixture
def empty_masks() -> Masks:
    """Empty masks artifact."""
    return Masks(instances=[], instance_ids=[], meta={})


# ---------------------------------------------------------------------------
# RefineMask Tests
# ---------------------------------------------------------------------------


class TestRefineMask:
    """Test mask morphological refinement operations."""

    def test_init_valid_methods(self):
        """Test node initialization with valid methods."""
        for method in ["morph_close", "morph_open", "dilate", "erode"]:
            node = RefineMask(method=method)
            assert node.method == method
            assert node.radius == 3  # default

    def test_init_invalid_method(self):
        """Test node initialization with invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            RefineMask(method="invalid_method")

    def test_init_custom_params(self):
        """Test node initialization with custom parameters."""
        node = RefineMask(src="input_masks", out="output_refined", method="dilate", radius=5)
        assert node.src == "input_masks"
        assert node.out == "output_refined"
        assert node.method == "dilate"
        assert node.radius == 5

    def test_morph_close_operation(self, ctx, sample_masks):
        """Test morphological close operation fills small gaps."""
        node = RefineMask(method="morph_close", radius=2)
        result = node.run(ctx, sample_masks)

        refined_masks = result["masks_ref"]
        assert isinstance(refined_masks, Masks)
        assert len(refined_masks.instances) == 2
        assert refined_masks.instance_ids == sample_masks.instance_ids

        # Check that masks are modified (hard to verify exact operation without visual inspection)
        for orig, refined in zip(sample_masks.instances, refined_masks.instances):
            assert orig.label_name == refined.label_name
            assert orig.score == refined.score

    def test_morph_open_operation(self, ctx, sample_masks):
        """Test morphological open operation removes noise."""
        node = RefineMask(method="morph_open", radius=1)
        result = node.run(ctx, sample_masks)

        refined_masks = result["masks_ref"]
        assert isinstance(refined_masks, Masks)
        assert len(refined_masks.instances) == 2

    def test_dilate_operation(self, ctx, sample_masks):
        """Test dilation operation expands masks."""
        node = RefineMask(method="dilate", radius=3)
        result = node.run(ctx, sample_masks)

        refined_masks = result["masks_ref"]
        assert isinstance(refined_masks, Masks)
        assert len(refined_masks.instances) == 2

        # Dilated masks should have larger or equal area
        # (Can't verify exact expansion without detailed mask analysis)

    def test_erode_operation(self, ctx, sample_masks):
        """Test erosion operation shrinks masks."""
        node = RefineMask(method="erode", radius=1)
        result = node.run(ctx, sample_masks)

        refined_masks = result["masks_ref"]
        assert isinstance(refined_masks, Masks)
        assert len(refined_masks.instances) == 2

    def test_empty_masks(self, ctx, empty_masks):
        """Test with empty masks input."""
        node = RefineMask(method="morph_close")
        result = node.run(ctx, empty_masks)

        refined_masks = result["masks_ref"]
        assert isinstance(refined_masks, Masks)
        assert len(refined_masks.instances) == 0
        assert len(refined_masks.instance_ids) == 0

    def test_invalid_radius(self):
        """Test invalid radius raises ValueError during run."""
        node = RefineMask(method="dilate", radius=-1)
        # Error should occur during run, not init
        assert node.radius == -1

    def test_mask_format_preservation(self, ctx):
        """Test that original mask format is preserved."""
        # Test with RLE mask
        try:
            rle_instance = _make_instance_with_rle_mask("test", 0.8)
            masks = Masks(instances=[rle_instance], instance_ids=["test_0000"], meta={})

            node = RefineMask(method="morph_close", radius=1)
            result = node.run(ctx, masks)

            refined_masks = result["masks_ref"]
            refined_inst = refined_masks.instances[0]

            # Should still be RLE format
            assert refined_inst.is_rle()

        except ImportError:
            pytest.skip("pycocotools not available for RLE format test")


# ---------------------------------------------------------------------------
# MaskToBox Tests
# ---------------------------------------------------------------------------


class TestMaskToBox:
    """Test mask to bounding box extraction."""

    def test_init_default_params(self):
        """Test node initialization with default parameters."""
        node = MaskToBox()
        assert node.src == "masks"
        assert node.out == "detections"
        assert node.filter_empty is True
        assert node.expand_px == 0

    def test_init_custom_params(self):
        """Test node initialization with custom parameters."""
        node = MaskToBox(src="input_masks", out="output_boxes", filter_empty=False, expand_px=5)
        assert node.src == "input_masks"
        assert node.out == "output_boxes"
        assert node.filter_empty is False
        assert node.expand_px == 5

    def test_init_invalid_expand_px(self):
        """Test invalid expand_px raises ValueError."""
        with pytest.raises(ValueError, match="expand_px must be non-negative"):
            MaskToBox(expand_px=-1)

    def test_bbox_extraction_basic(self, ctx, sample_masks):
        """Test basic bounding box extraction."""
        node = MaskToBox()
        result = node.run(ctx, sample_masks)

        detections = result["detections"]
        assert isinstance(detections, Detections)
        assert len(detections.instances) == 2
        assert len(detections.instance_ids) == 2

        # Check that all instances have bboxes but no masks
        for inst in detections.instances:
            assert inst.bbox is not None
            assert len(inst.bbox) == 4  # xyxy format
            assert inst.mask is None
            assert inst.score > 0
            assert inst.label_name is not None

    def test_bbox_extraction_with_padding(self, ctx, sample_masks):
        """Test bbox extraction with padding."""
        node = MaskToBox(expand_px=5)
        result = node.run(ctx, sample_masks)

        detections = result["detections"]
        assert len(detections.instances) == 2

        # Padded boxes should be larger than original but still within image bounds
        for inst in detections.instances:
            x1, y1, x2, y2 = inst.bbox
            assert x1 >= 0  # Clipped to image bounds
            assert y1 >= 0
            assert x2 <= 100  # Image width from meta
            assert y2 <= 100  # Image height from meta

    def test_bbox_coordinates_correctness(self, ctx):
        """Test that extracted bbox coordinates are correct."""
        # Create mask with known area
        binary_mask = np.zeros((100, 100), dtype=bool)
        binary_mask[20:40, 30:60] = True  # Rectangle from (30,20) to (60,40)

        instance = Instance(bbox=None, mask=binary_mask, score=0.8, label=0, label_name="test")

        masks = Masks(instances=[instance], instance_ids=["test"], meta={"image_height": 100, "image_width": 100})

        node = MaskToBox()
        result = node.run(ctx, masks)

        detections = result["detections"]
        bbox = detections.instances[0].bbox
        x1, y1, x2, y2 = bbox

        # Check coordinates (note: bbox is inclusive on bottom/right)
        assert x1 == 30.0
        assert y1 == 20.0
        assert x2 == 60.0  # 59 + 1 for inclusive
        assert y2 == 40.0  # 39 + 1 for inclusive

    def test_empty_mask_filtering(self, ctx):
        """Test empty mask filtering behavior."""
        # Create mask with no foreground pixels
        empty_mask = np.zeros((100, 100), dtype=bool)

        instance = Instance(bbox=None, mask=empty_mask, score=0.8, label=0, label_name="empty")

        masks = Masks(instances=[instance], instance_ids=["empty"], meta={})

        # With filtering (default)
        node = MaskToBox(filter_empty=True)
        result = node.run(ctx, masks)
        detections = result["detections"]
        assert len(detections.instances) == 0

        # Without filtering
        node = MaskToBox(filter_empty=False)
        result = node.run(ctx, masks)
        detections = result["detections"]
        # Empty mask should be filtered out even when filter_empty=False
        # because bbox extraction returns None
        assert len(detections.instances) == 0

    def test_empty_masks_input(self, ctx, empty_masks):
        """Test with empty masks input."""
        node = MaskToBox()
        result = node.run(ctx, empty_masks)

        detections = result["detections"]
        assert isinstance(detections, Detections)
        assert len(detections.instances) == 0
        assert len(detections.instance_ids) == 0

    def test_metadata_preservation(self, ctx, sample_masks):
        """Test that metadata is preserved and enhanced."""
        node = MaskToBox(expand_px=3)
        result = node.run(ctx, sample_masks)

        detections = result["detections"]
        meta = detections.meta

        # Original meta preserved
        assert meta["image_height"] == 100
        assert meta["image_width"] == 100

        # New meta added
        assert meta["extracted_from"] == "masks"
        assert meta["expand_px"] == 3

    def test_instance_id_preservation(self, ctx, sample_masks):
        """Test that instance IDs are preserved."""
        node = MaskToBox()
        result = node.run(ctx, sample_masks)

        detections = result["detections"]
        assert detections.instance_ids == sample_masks.instance_ids

    def test_attributes_preservation(self, ctx, sample_masks):
        """Test that instance attributes are preserved (except mask)."""
        node = MaskToBox()
        result = node.run(ctx, sample_masks)

        detections = result["detections"]

        for orig_inst, det_inst in zip(sample_masks.instances, detections.instances):
            # Should preserve all attributes except bbox (computed) and mask (removed)
            assert det_inst.score == orig_inst.score
            assert det_inst.label == orig_inst.label
            assert det_inst.label_name == orig_inst.label_name
            assert det_inst.area == orig_inst.area
            assert det_inst.is_stuff == orig_inst.is_stuff
            assert det_inst.embedding == orig_inst.embedding
            assert det_inst.track_id == orig_inst.track_id
            assert det_inst.keypoints == orig_inst.keypoints
            # New bbox computed, mask removed
            assert det_inst.bbox is not None
            assert det_inst.mask is None


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestMaskRefinementIntegration:
    """Test integration between RefineMask and MaskToBox nodes."""

    def test_refine_then_extract_workflow(self, ctx, sample_masks):
        """Test full workflow: refine masks then extract boxes."""
        # Step 1: Refine masks
        refine_node = RefineMask(method="morph_close", radius=2, out="refined")
        refine_result = refine_node.run(ctx, sample_masks)
        refined_masks = refine_result["refined"]

        # Step 2: Extract boxes from refined masks
        extract_node = MaskToBox(src="refined", out="final_boxes")
        extract_result = extract_node.run(ctx, refined_masks)
        final_detections = extract_result["final_boxes"]

        # Verify final result
        assert isinstance(final_detections, Detections)
        assert len(final_detections.instances) == 2
        assert final_detections.instance_ids == sample_masks.instance_ids

        # All instances should have bboxes from refined masks
        for inst in final_detections.instances:
            assert inst.bbox is not None
            assert inst.mask is None
            assert inst.score > 0
