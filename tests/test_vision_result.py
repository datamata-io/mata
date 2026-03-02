"""Tests for VisionResult and Instance unified type system.

Tests the new unified VisionResult type that replaces DetectResult and SegmentResult
with backward compatibility.
"""

import numpy as np
import pytest

from mata.core.types import (
    Detection,
    Instance,
    SegmentMask,
    VisionResult,
)


class TestInstance:
    """Test the Instance dataclass."""

    def test_instance_with_bbox_only(self):
        """Test instance with only bbox (detection-only)."""
        inst = Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat")

        assert inst.bbox == (10.0, 20.0, 100.0, 200.0)
        assert inst.score == 0.95
        assert inst.label == 0
        assert inst.label_name == "cat"
        assert inst.mask is None
        assert inst.area is None

    def test_instance_with_mask_only(self):
        """Test instance with only mask (segmentation-only)."""
        rle_mask = {"size": [100, 100], "counts": "test_counts"}

        inst = Instance(mask=rle_mask, score=0.87, label=1, label_name="dog")

        assert inst.mask == rle_mask
        assert inst.score == 0.87
        assert inst.label == 1
        assert inst.label_name == "dog"
        assert inst.bbox is None

    def test_instance_with_bbox_and_mask(self):
        """Test instance with both bbox and mask (multi-modal)."""
        bbox = (50.0, 50.0, 300.0, 300.0)
        rle_mask = {"size": [640, 640], "counts": "rle_data"}

        inst = Instance(bbox=bbox, mask=rle_mask, score=0.92, label=2, label_name="person", area=12500)

        assert inst.bbox == bbox
        assert inst.mask == rle_mask
        assert inst.score == 0.92
        assert inst.label == 2
        assert inst.label_name == "person"
        assert inst.area == 12500

    def test_instance_requires_bbox_or_mask(self):
        """Test that instance requires at least bbox or mask."""
        with pytest.raises(ValueError, match="must have at least one"):
            Instance(score=0.5, label=0)

    def test_instance_mask_validation_binary(self):
        """Test binary mask validation."""
        # Valid 2D binary mask
        binary_mask = np.zeros((100, 100), dtype=bool)
        inst = Instance(mask=binary_mask, score=0.9, label=0)
        assert inst.is_binary()
        assert not inst.is_rle()
        assert not inst.is_polygon()

        # Invalid 3D mask
        with pytest.raises(ValueError, match="must be 2D"):
            Instance(mask=np.zeros((100, 100, 3), dtype=bool), score=0.9, label=0)

    def test_instance_mask_validation_rle(self):
        """Test RLE mask validation."""
        # Valid RLE
        rle_mask = {"size": [100, 100], "counts": "test"}
        inst = Instance(mask=rle_mask, score=0.9, label=0)
        assert inst.is_rle()
        assert not inst.is_binary()
        assert not inst.is_polygon()

        # Invalid RLE (missing keys)
        with pytest.raises(ValueError, match="must contain 'size' and 'counts'"):
            Instance(mask={"size": [100, 100]}, score=0.9, label=0)  # Missing 'counts'

    def test_instance_mask_validation_polygon(self):
        """Test polygon mask validation."""
        # Valid polygon (even number of coordinates)
        polygon = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        inst = Instance(mask=polygon, score=0.9, label=0)
        assert inst.is_polygon()
        assert not inst.is_binary()
        assert not inst.is_rle()

        # Invalid polygon (odd number of coordinates)
        with pytest.raises(ValueError, match="even number of coordinates"):
            Instance(mask=[10.0, 20.0, 30.0], score=0.9, label=0)  # Odd length

    def test_instance_to_dict(self):
        """Test Instance serialization to dict."""
        inst = Instance(
            bbox=(10.0, 20.0, 100.0, 200.0),
            mask={"size": [100, 100], "counts": "test"},
            score=0.95,
            label=0,
            label_name="cat",
            area=1000,
            track_id=123,
        )

        data = inst.to_dict()

        assert data["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert data["score"] == 0.95
        assert data["label"] == 0
        assert data["label_name"] == "cat"
        assert data["area"] == 1000
        assert data["track_id"] == 123
        assert data["mask"]["format"] == "rle"
        assert data["mask"]["data"]["counts"] == "test"


class TestVisionResult:
    """Test the VisionResult unified result type."""

    def test_vision_result_with_detections(self):
        """Test VisionResult for detection task."""
        instances = [
            Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(150.0, 50.0, 300.0, 250.0), score=0.87, label=1, label_name="dog"),
        ]

        result = VisionResult(instances=instances)

        assert len(result.instances) == 2
        assert len(result.detections) == 2  # Backward compat property
        assert len(result.masks) == 0  # No masks
        assert result.instances[0].label_name == "cat"

    def test_vision_result_with_masks(self):
        """Test VisionResult for segmentation task."""
        instances = [
            Instance(mask={"size": [100, 100], "counts": "rle1"}, score=0.92, label=0, label_name="person"),
            Instance(mask={"size": [100, 100], "counts": "rle2"}, score=0.88, label=0, label_name="person"),
        ]

        result = VisionResult(instances=instances)

        assert len(result.instances) == 2
        assert len(result.masks) == 2  # Backward compat property
        assert len(result.detections) == 0  # No bboxes

    def test_vision_result_with_bbox_and_mask(self):
        """Test VisionResult for multi-modal (pipeline) result."""
        instances = [
            Instance(
                bbox=(10.0, 20.0, 100.0, 200.0),
                mask={"size": [640, 640], "counts": "rle_data"},
                score=0.93,
                label=0,
                label_name="cat",
                area=12500,
            )
        ]

        result = VisionResult(instances=instances, meta={"pipeline": "grounding_sam"})

        assert len(result.instances) == 1
        assert len(result.detections) == 1  # Has bbox
        assert len(result.masks) == 1  # Has mask
        assert result.instances[0].bbox is not None
        assert result.instances[0].mask is not None
        assert result.meta["pipeline"] == "grounding_sam"

    def test_vision_result_filter_by_score(self):
        """Test filtering instances by score threshold."""
        instances = [
            Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0),
            Instance(bbox=(50.0, 60.0, 150.0, 250.0), score=0.4, label=1),
            Instance(bbox=(100.0, 110.0, 200.0, 300.0), score=0.7, label=2),
        ]

        result = VisionResult(instances=instances)
        filtered = result.filter_by_score(threshold=0.5)

        assert len(filtered.instances) == 2
        assert all(inst.score >= 0.5 for inst in filtered.instances)

    def test_vision_result_get_instances_and_stuff(self):
        """Test filtering by is_stuff field (panoptic segmentation)."""
        instances = [
            Instance(mask={"size": [100, 100], "counts": "rle1"}, score=0.9, label=0, is_stuff=False),
            Instance(mask={"size": [100, 100], "counts": "rle2"}, score=0.85, label=10, is_stuff=True),
            Instance(mask={"size": [100, 100], "counts": "rle3"}, score=0.92, label=0, is_stuff=False),
        ]

        result = VisionResult(instances=instances)

        instance_objects = result.get_instances()
        stuff_regions = result.get_stuff()

        assert len(instance_objects) == 2
        assert len(stuff_regions) == 1
        assert all(inst.is_stuff is False or inst.is_stuff is None for inst in instance_objects)
        assert all(inst.is_stuff is True for inst in stuff_regions)

    def test_vision_result_serialization(self):
        """Test JSON serialization and deserialization."""
        instances = [Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat")]

        result = VisionResult(instances=instances, meta={"model": "test"}, text="detected 1 cat", prompt="find cats")

        # Serialize
        json_str = result.to_json(indent=2)
        assert isinstance(json_str, str)

        # Deserialize
        restored = VisionResult.from_json(json_str)

        assert len(restored.instances) == 1
        assert restored.instances[0].bbox == (10.0, 20.0, 100.0, 200.0)
        assert restored.instances[0].label_name == "cat"
        assert restored.meta["model"] == "test"
        assert restored.text == "detected 1 cat"
        assert restored.prompt == "find cats"


class TestBackwardCompatibility:
    """Test backward compatibility with DetectResult and SegmentResult."""

    def test_detection_to_instance_conversion(self):
        """Test Detection.to_instance() conversion."""
        detection = Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat")

        instance = detection.to_instance()

        assert isinstance(instance, Instance)
        assert instance.bbox == detection.bbox
        assert instance.score == detection.score
        assert instance.label == detection.label
        assert instance.label_name == detection.label_name
        assert instance.mask is None

    def test_segment_mask_to_instance_conversion(self):
        """Test SegmentMask.to_instance() conversion."""
        seg_mask = SegmentMask(
            mask={"size": [100, 100], "counts": "test"},
            score=0.87,
            label=1,
            label_name="dog",
            bbox=(50.0, 50.0, 150.0, 150.0),
            area=10000,
            is_stuff=False,
        )

        instance = seg_mask.to_instance()

        assert isinstance(instance, Instance)
        assert instance.mask == seg_mask.mask
        assert instance.score == seg_mask.score
        assert instance.label == seg_mask.label
        assert instance.label_name == seg_mask.label_name
        assert instance.bbox == seg_mask.bbox
        assert instance.area == seg_mask.area
        assert instance.is_stuff == seg_mask.is_stuff

    def test_vision_result_detections_property(self):
        """Test VisionResult.detections property filters bbox instances."""
        instances = [
            Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0),  # Has bbox
            Instance(mask={"size": [100, 100], "counts": "test"}, score=0.87, label=1),  # No bbox
            Instance(
                bbox=(50.0, 60.0, 150.0, 250.0), mask={"size": [100, 100], "counts": "test2"}, score=0.92, label=2
            ),  # Has both
        ]

        result = VisionResult(instances=instances)

        detections = result.detections
        assert len(detections) == 2  # Two instances with bbox
        assert all(inst.bbox is not None for inst in detections)

    def test_vision_result_masks_property(self):
        """Test VisionResult.masks property filters mask instances."""
        instances = [
            Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0),  # No mask
            Instance(mask={"size": [100, 100], "counts": "test"}, score=0.87, label=1),  # Has mask
            Instance(
                bbox=(50.0, 60.0, 150.0, 250.0), mask={"size": [100, 100], "counts": "test2"}, score=0.92, label=2
            ),  # Has both
        ]

        result = VisionResult(instances=instances)

        masks = result.masks
        assert len(masks) == 2  # Two instances with mask
        assert all(inst.mask is not None for inst in masks)

    def test_vision_result_used_as_detect_result(self):
        """Test VisionResult can be used where DetectResult was expected."""
        # Create VisionResult with detection-like instances
        instances = [
            Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(150.0, 50.0, 300.0, 250.0), score=0.87, label=1, label_name="dog"),
        ]

        result = VisionResult(instances=instances, meta={"model": "test"})

        # Access as DetectResult-like interface
        assert len(result.detections) == 2
        assert result.detections[0].bbox == (10.0, 20.0, 100.0, 200.0)
        assert result.detections[0].label_name == "cat"

        # Serialize/deserialize
        json_str = result.to_json()
        restored = VisionResult.from_json(json_str)
        assert len(restored.detections) == 2

    def test_vision_result_used_as_segment_result(self):
        """Test VisionResult can be used where SegmentResult was expected."""
        # Create VisionResult with segmentation-like instances
        instances = [
            Instance(mask={"size": [100, 100], "counts": "rle1"}, score=0.92, label=0, label_name="person", area=5000),
            Instance(mask={"size": [100, 100], "counts": "rle2"}, score=0.88, label=0, label_name="person", area=4800),
        ]

        result = VisionResult(instances=instances, meta={"mode": "instance"})

        # Access as SegmentResult-like interface
        assert len(result.masks) == 2
        assert result.masks[0].mask == {"size": [100, 100], "counts": "rle1"}
        assert result.masks[0].label_name == "person"

        # Use SegmentResult methods
        filtered = result.filter_by_score(0.9)
        assert len(filtered.masks) == 1
        assert filtered.masks[0].score == 0.92


class TestVisionResultWithVQA:
    """Test VisionResult for future VQA/captioning tasks."""

    def test_vision_result_with_text_output(self):
        """Test VisionResult with text field (for VQA/captioning)."""
        result = VisionResult(
            instances=[], text="A cat sitting on a couch", prompt="What is in this image?", meta={"task": "vqa"}
        )

        assert result.text == "A cat sitting on a couch"
        assert result.prompt == "What is in this image?"
        assert len(result.instances) == 0

    def test_vision_result_multimodal_with_text(self):
        """Test VisionResult combining detections and text."""
        instances = [Instance(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=0, label_name="cat")]

        result = VisionResult(
            instances=instances, text="Found 1 cat in the image", meta={"task": "detection_with_caption"}
        )

        assert len(result.instances) == 1
        assert result.text == "Found 1 cat in the image"
