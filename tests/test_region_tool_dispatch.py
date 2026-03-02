"""Tests for Task C2: Region-Based Tool Dispatch.

Verifies that when VLM calls a tool with a region parameter, the tool registry:
1. Crops the image to the specified region
2. Runs the adapter on the cropped sub-image
3. Remaps bbox coordinates back to original image space
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from PIL import Image as PILImage

from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.tool_registry import ToolRegistry
from mata.core.tool_schema import ToolCall
from mata.core.types import Instance, VisionResult


@pytest.fixture
def sample_image():
    """Create a sample 640x480 image."""
    pil_img = PILImage.new("RGB", (640, 480), color="white")
    return Image.from_pil(pil_img)


@pytest.fixture
def mock_detect_provider():
    """Mock detection provider that returns fixed detections."""
    provider = Mock()

    # Return detections in cropped image space (as Detections artifact)
    # For a 200x200 crop, objects at [10, 20, 50, 80] and [100, 120, 180, 190]
    from mata.core.artifacts.converters import vision_result_to_detections

    vision_result = VisionResult(
        instances=[
            Instance(bbox=(10, 20, 50, 80), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(100, 120, 180, 190), score=0.87, label=1, label_name="dog"),
        ]
    )
    provider.predict.return_value = vision_result_to_detections(vision_result)
    return provider


@pytest.fixture
def mock_classify_provider():
    """Mock classification provider."""
    from mata.core.artifacts.classifications import Classification, Classifications

    provider = Mock()
    provider.predict.return_value = Classifications(
        predictions=[
            Classification(label=0, label_name="cat", score=0.92),
            Classification(label=1, label_name="dog", score=0.08),
        ]
    )
    return provider


@pytest.fixture
def execution_context(mock_detect_provider, mock_classify_provider):
    """Create ExecutionContext with mock providers."""
    return ExecutionContext(
        providers={
            "detect": {"detector": mock_detect_provider},
            "classify": {"classifier": mock_classify_provider},
        }
    )


class TestRegionCropAndRemap:
    """Test region-based cropping and coordinate remapping."""

    def test_detect_with_region_crops_image(self, sample_image, mock_detect_provider, execution_context):
        """Verify that detect(region=[...]) crops the image before running detection."""
        registry = ToolRegistry(execution_context, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [100, 50, 300, 250]},  # Crop region
            raw_text="detect objects in region",
        )

        registry.execute_tool(tool_call, sample_image)

        # Verify provider was called
        assert mock_detect_provider.predict.called

        # Verify provider received cropped image (200x200)
        called_image = mock_detect_provider.predict.call_args[0][0]
        assert called_image.width == 200  # 300 - 100
        assert called_image.height == 200  # 250 - 50

    def test_detect_with_region_remaps_coordinates(self, sample_image, mock_detect_provider, execution_context):
        """Verify bbox coordinates are remapped from crop space to original image space."""
        registry = ToolRegistry(execution_context, ["detector"])

        # Crop region: [100, 50, 300, 250]
        # Expected remapping: add [100, 50] to all bbox coordinates
        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [100, 50, 300, 250]},
            raw_text="detect objects in region",
        )

        result = registry.execute_tool(tool_call, sample_image)

        assert result.success
        detections = result.artifacts["detections"]

        # Original bbox in crop: [10, 20, 50, 80]
        # Expected remapped: [110, 70, 150, 130]
        inst1 = detections.instances[0]
        assert inst1.bbox == (110, 70, 150, 130)
        assert inst1.label_name == "cat"

        # Original bbox in crop: [100, 120, 180, 190]
        # Expected remapped: [200, 170, 280, 240]
        inst2 = detections.instances[1]
        assert inst2.bbox == (200, 170, 280, 240)
        assert inst2.label_name == "dog"

    def test_detect_without_region_uses_full_image(self, sample_image, mock_detect_provider, execution_context):
        """Verify detect() without region parameter uses full image."""
        registry = ToolRegistry(execution_context, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={},  # No region
            raw_text="detect all objects",
        )

        result = registry.execute_tool(tool_call, sample_image)

        # Verify provider received full image
        called_image = mock_detect_provider.predict.call_args[0][0]
        assert called_image.width == 640
        assert called_image.height == 480

        # Verify coordinates are NOT remapped (no offset)
        detections = result.artifacts["detections"]
        inst1 = detections.instances[0]
        assert inst1.bbox == (10, 20, 50, 80)  # Original coords

    def test_classify_with_region_crops_image(self, sample_image, mock_classify_provider, execution_context):
        """Verify classify(region=[...]) also crops before classification."""
        registry = ToolRegistry(execution_context, ["classifier"])

        tool_call = ToolCall(
            tool_name="classifier",
            arguments={"region": [200, 100, 500, 400]},
            raw_text="classify region",
        )

        result = registry.execute_tool(tool_call, sample_image)

        # Verify provider received cropped image (300x300)
        called_image = mock_classify_provider.predict.call_args[0][0]
        assert called_image.width == 300
        assert called_image.height == 300

        assert result.success
        assert "classifications" in result.artifacts

    def test_region_offset_in_metadata(self, sample_image, mock_detect_provider, execution_context):
        """Verify region offset is stored in metadata."""
        registry = ToolRegistry(execution_context, ["detector"])

        region = [100, 50, 300, 250]
        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": region},
            raw_text="detect objects",
        )

        result = registry.execute_tool(tool_call, sample_image)

        detections = result.artifacts["detections"]
        assert "region_offset" in detections.meta
        assert detections.meta["region_offset"] == region

    def test_mask_only_instance_preserved(self, sample_image, execution_context):
        """Verify instances without bbox (mask-only) are preserved during remapping."""
        # Mock provider returning mask-only instance
        from mata.core.artifacts.converters import vision_result_to_detections

        mock_provider = Mock()
        vision_result = VisionResult(
            instances=[
                Instance(
                    bbox=None,
                    mask={"size": [200, 200], "counts": "dummy_rle"},
                    score=0.88,
                    label=0,
                    label_name="cat",
                ),
            ]
        )
        mock_provider.predict.return_value = vision_result_to_detections(vision_result)

        ctx = ExecutionContext(providers={"detect": {"detector": mock_provider}})
        registry = ToolRegistry(ctx, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [100, 50, 300, 250]},
            raw_text="segment objects",
        )

        result = registry.execute_tool(tool_call, sample_image)

        # Verify instance is preserved with None bbox
        detections = result.artifacts["detections"]
        assert len(detections.instances) == 1
        assert detections.instances[0].bbox is None
        assert detections.instances[0].mask is not None

    def test_mixed_bbox_and_mask_instances(self, sample_image, execution_context):
        """Verify mixed instances (some with bbox, some mask-only) are handled correctly."""
        from mata.core.artifacts.converters import vision_result_to_detections

        mock_provider = Mock()
        vision_result = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 50, 80), score=0.95, label=0, label_name="cat"),
                Instance(
                    bbox=None,
                    mask={"size": [100, 100], "counts": "rle"},
                    score=0.85,
                    label=1,
                    label_name="dog",
                ),
                Instance(bbox=(100, 120, 180, 190), score=0.92, label=2, label_name="bird"),
            ]
        )
        mock_provider.predict.return_value = vision_result_to_detections(vision_result)

        ctx = ExecutionContext(providers={"detect": {"detector": mock_provider}})
        registry = ToolRegistry(ctx, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [50, 30, 250, 230]},  # Offset by [50, 30]
            raw_text="detect objects",
        )

        result = registry.execute_tool(tool_call, sample_image)

        detections = result.artifacts["detections"]
        assert len(detections.instances) == 3

        # First instance: bbox remapped
        assert detections.instances[0].bbox == (60, 50, 100, 110)

        # Second instance: mask-only, no bbox
        assert detections.instances[1].bbox is None
        assert detections.instances[1].mask is not None

        # Third instance: bbox remapped
        assert detections.instances[2].bbox == (150, 150, 230, 220)

    def test_invalid_crop_region_returns_error(self, sample_image, execution_context):
        """Verify invalid crop regions (empty, inverted) return error ToolResult."""
        registry = ToolRegistry(execution_context, ["detector"])

        # Empty region (x1 == x2)
        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [100, 50, 100, 250]},
            raw_text="detect",
        )

        result = registry.execute_tool(tool_call, sample_image)

        assert not result.success
        assert "Invalid region" in result.summary

    def test_out_of_bounds_region_clamped(self, sample_image, mock_detect_provider, execution_context):
        """Verify out-of-bounds regions are clamped to image bounds."""
        registry = ToolRegistry(execution_context, ["detector"])

        # Region exceeds image bounds (640x480)
        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [500, 400, 800, 600]},  # Exceeds bounds
            raw_text="detect objects",
        )

        registry.execute_tool(tool_call, sample_image)

        # Verify provider received clamped crop (640-500=140, 480-400=80)
        called_image = mock_detect_provider.predict.call_args[0][0]
        assert called_image.width == 140
        assert called_image.height == 80

    def test_region_with_other_arguments(self, sample_image, mock_detect_provider, execution_context):
        """Verify region parameter works alongside other arguments (threshold, etc.)."""
        registry = ToolRegistry(execution_context, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={
                "region": [100, 50, 300, 250],
                "threshold": 0.7,  # Other argument
            },
            raw_text="detect with threshold",
        )

        registry.execute_tool(tool_call, sample_image)

        # Verify provider received threshold argument (region removed)
        call_kwargs = mock_detect_provider.predict.call_args[1]
        assert "threshold" in call_kwargs
        assert call_kwargs["threshold"] == 0.7
        assert "region" not in call_kwargs  # Region removed after cropping

    def test_coordinate_remapping_with_float_region(self, sample_image, mock_detect_provider, execution_context):
        """Verify coordinate remapping works with float region coordinates."""
        registry = ToolRegistry(execution_context, ["detector"])

        tool_call = ToolCall(
            tool_name="detector",
            arguments={"region": [100.5, 50.3, 300.7, 250.9]},
            raw_text="detect objects",
        )

        result = registry.execute_tool(tool_call, sample_image)

        # Verify remapping uses clamped integer offsets [100, 50]
        detections = result.artifacts["detections"]
        inst1 = detections.instances[0]
        assert inst1.bbox == (110, 70, 150, 130)  # [10+100, 20+50, 50+100, 80+50]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
