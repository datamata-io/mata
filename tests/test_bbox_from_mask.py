"""Tests for bounding box computation from masks in visualization."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mata.core.types import Instance, VisionResult
from mata.visualization import _compute_bbox_from_mask, visualize_segmentation


class TestBboxFromMask:
    """Test automatic bbox computation from masks."""

    def test_compute_bbox_from_binary_mask(self):
        """Test computing bbox from a binary mask."""
        # Create a binary mask with a rectangle
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 30:80] = True  # Rectangle from (30,20) to (80,60)

        bbox = _compute_bbox_from_mask(mask)

        assert bbox is not None
        assert bbox == (30.0, 20.0, 79.0, 59.0)  # xyxy format

    def test_compute_bbox_from_empty_mask(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=bool)

        bbox = _compute_bbox_from_mask(mask)

        assert bbox is None

    def test_visualize_with_mask_no_bbox(self):
        """Test visualization computes bbox from mask when bbox is None."""
        # Create test image
        image = Image.new("RGB", (100, 100), color="white")

        # Create mask with known bounds
        binary_mask = np.zeros((100, 100), dtype=bool)
        binary_mask[20:60, 30:80] = True

        # Create instance with mask but no bbox
        instance = Instance(
            label=1,
            label_name="test",
            score=0.9,
            bbox=None,  # No bbox - should be computed
            mask=binary_mask,  # Just pass numpy array directly
        )

        result = VisionResult(instances=[instance])

        # Visualize with show_boxes=True
        output = visualize_segmentation(result, image=image, show_boxes=True, backend="pil")

        # Should not raise error and should return image
        assert isinstance(output, Image.Image)
        assert output.size == (100, 100)

    def test_visualize_with_bbox_and_mask(self):
        """Test that existing bbox is used when available."""
        image = Image.new("RGB", (100, 100), color="white")

        # Create mask
        binary_mask = np.zeros((100, 100), dtype=bool)
        binary_mask[20:60, 30:80] = True

        # Create instance with both bbox and mask
        custom_bbox = (10.0, 10.0, 90.0, 90.0)  # Different from mask bounds
        instance = Instance(label=1, label_name="test", score=0.9, bbox=custom_bbox, mask=binary_mask)

        result = VisionResult(instances=[instance])

        # Visualize - should use custom bbox, not compute from mask
        output = visualize_segmentation(result, image=image, show_boxes=True, backend="pil")

        assert isinstance(output, Image.Image)

    def test_visualize_with_show_boxes_false(self):
        """Test that bbox is not drawn when show_boxes=False."""
        image = Image.new("RGB", (100, 100), color="white")

        binary_mask = np.zeros((100, 100), dtype=bool)
        binary_mask[20:60, 30:80] = True

        instance = Instance(label=1, label_name="test", score=0.9, bbox=None, mask=binary_mask)

        result = VisionResult(instances=[instance])

        # Visualize without boxes
        output = visualize_segmentation(result, image=image, show_boxes=False, backend="pil")

        # Should still work but no bbox computation needed
        assert isinstance(output, Image.Image)

    def test_multiple_instances_mixed_bboxes(self):
        """Test with multiple instances, some with bbox, some without."""
        image = Image.new("RGB", (200, 200), color="white")

        # Instance 1: has mask, no bbox
        mask1 = np.zeros((200, 200), dtype=bool)
        mask1[10:50, 10:50] = True
        instance1 = Instance(label=1, label_name="obj1", score=0.9, bbox=None, mask=mask1)

        # Instance 2: has both mask and bbox
        mask2 = np.zeros((200, 200), dtype=bool)
        mask2[60:100, 60:100] = True
        instance2 = Instance(label=2, label_name="obj2", score=0.8, bbox=(60.0, 60.0, 100.0, 100.0), mask=mask2)

        # Instance 3: bbox only, no mask
        instance3 = Instance(label=3, label_name="obj3", score=0.7, bbox=(110.0, 110.0, 150.0, 150.0), mask=None)

        result = VisionResult(instances=[instance1, instance2, instance3])

        # Visualize all
        output = visualize_segmentation(result, image=image, show_boxes=True, backend="pil")

        assert isinstance(output, Image.Image)
        assert output.size == (200, 200)
