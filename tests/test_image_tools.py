"""Unit tests for image_tools module (built-in crop/zoom tools).

Tests the crop_tool and zoom_tool implementations, including:
- Basic functionality (crop and zoom operations)
- Coordinate validation and clamping
- Error handling (invalid regions, invalid parameters)
- Edge cases (out-of-bounds regions, zero-area crops)
- ToolResult formatting

Coverage target: 100% of image_tools.py
"""

from __future__ import annotations

import pytest
from PIL import Image

from mata.core.image_tools import BUILTIN_TOOLS, _validate_and_clamp_region, crop_tool, zoom_tool


class TestValidateAndClampRegion:
    """Tests for _validate_and_clamp_region helper function."""

    def test_valid_region_no_clamping(self):
        """Region fully within bounds needs no clamping."""
        region = (10, 20, 100, 200)
        result = _validate_and_clamp_region(region, 800, 600)
        assert result == (10, 20, 100, 200)

    def test_clamp_negative_coords(self):
        """Negative coordinates clamped to 0."""
        region = (-10, -20, 100, 200)
        result = _validate_and_clamp_region(region, 800, 600)
        assert result == (0, 0, 100, 200)

    def test_clamp_exceeds_bounds(self):
        """Coordinates beyond image bounds clamped to edges."""
        region = (100, 200, 900, 700)
        result = _validate_and_clamp_region(region, 800, 600)
        assert result == (100, 200, 800, 600)

    def test_clamp_both_sides(self):
        """Clamp both negative and exceeding coordinates."""
        region = (-50, -30, 900, 700)
        result = _validate_and_clamp_region(region, 800, 600)
        assert result == (0, 0, 800, 600)

    def test_float_coords_to_int(self):
        """Float coordinates converted to integers."""
        region = (10.7, 20.3, 100.9, 200.1)
        result = _validate_and_clamp_region(region, 800, 600)
        assert result == (10, 20, 100, 200)

    def test_invalid_region_zero_width(self):
        """Region with zero width raises ValueError."""
        region = (100, 100, 100, 200)
        with pytest.raises(ValueError, match="Invalid crop region"):
            _validate_and_clamp_region(region, 800, 600)

    def test_invalid_region_zero_height(self):
        """Region with zero height raises ValueError."""
        region = (100, 100, 200, 100)
        with pytest.raises(ValueError, match="Invalid crop region"):
            _validate_and_clamp_region(region, 800, 600)

    def test_invalid_region_negative_width(self):
        """Region with x2 < x1 raises ValueError."""
        region = (200, 100, 100, 200)
        with pytest.raises(ValueError, match="Invalid crop region"):
            _validate_and_clamp_region(region, 800, 600)

    def test_invalid_region_negative_height(self):
        """Region with y2 < y1 raises ValueError."""
        region = (100, 200, 200, 100)
        with pytest.raises(ValueError, match="Invalid crop region"):
            _validate_and_clamp_region(region, 800, 600)

    def test_error_message_includes_context(self):
        """Error message includes original region and image size."""
        region = (100, 100, 100, 200)
        with pytest.raises(ValueError) as exc_info:
            _validate_and_clamp_region(region, 800, 600)
        assert "Original region: (100, 100, 100, 200)" in str(exc_info.value)
        assert "Image size: 800x600" in str(exc_info.value)


class TestCropTool:
    """Tests for crop_tool function."""

    def test_basic_crop(self):
        """Crop a valid region from image."""
        img = Image.new("RGB", (800, 600), color="red")
        result = crop_tool(img, (100, 100, 300, 400))

        assert result.success is True
        assert result.tool_name == "crop"
        assert "image" in result.artifacts
        assert result.artifacts["image"].size == (200, 300)  # 300-100, 400-100
        assert "region" in result.artifacts
        assert result.artifacts["region"] == (100, 100, 300, 400)
        assert "Cropped region" in result.summary
        assert "200x300" in result.summary

    def test_crop_with_clamping(self):
        """Crop region that needs clamping to image bounds."""
        img = Image.new("RGB", (800, 600), color="blue")
        result = crop_tool(img, (-10, -20, 100, 200))

        assert result.success is True
        assert result.artifacts["image"].size == (100, 200)  # Clamped from (-10,-20)
        assert result.artifacts["region"] == (0, 0, 100, 200)

    def test_crop_exceeds_bounds(self):
        """Crop region that exceeds image bounds."""
        img = Image.new("RGB", (800, 600), color="green")
        result = crop_tool(img, (700, 500, 900, 700))

        assert result.success is True
        assert result.artifacts["image"].size == (100, 100)  # Clamped to 800x600
        assert result.artifacts["region"] == (700, 500, 800, 600)

    def test_crop_full_image(self):
        """Crop entire image (no actual cropping)."""
        img = Image.new("RGB", (800, 600), color="white")
        result = crop_tool(img, (0, 0, 800, 600))

        assert result.success is True
        assert result.artifacts["image"].size == (800, 600)

    def test_crop_invalid_region_zero_area(self):
        """Crop with zero-area region returns failure ToolResult."""
        img = Image.new("RGB", (800, 600), color="black")
        result = crop_tool(img, (100, 100, 100, 200))

        assert result.success is False
        assert result.tool_name == "crop"
        assert "Failed to crop region" in result.summary
        assert "Invalid crop region" in result.summary

    def test_crop_invalid_region_inverted(self):
        """Crop with inverted coordinates returns failure ToolResult."""
        img = Image.new("RGB", (800, 600), color="yellow")
        result = crop_tool(img, (300, 400, 100, 100))

        assert result.success is False
        assert "Failed to crop region" in result.summary

    def test_crop_format_float_coords(self):
        """Crop handles float coordinates (converts to int)."""
        img = Image.new("RGB", (800, 600), color="cyan")
        result = crop_tool(img, (10.5, 20.7, 100.9, 200.1))

        assert result.success is True
        assert result.artifacts["region"] == (10, 20, 100, 200)
        assert result.artifacts["image"].size == (90, 180)


class TestZoomTool:
    """Tests for zoom_tool function."""

    def test_basic_zoom_2x(self):
        """Zoom a region with default 2x scaling."""
        img = Image.new("RGB", (800, 600), color="red")
        result = zoom_tool(img, (100, 100, 300, 400), scale=2.0)

        assert result.success is True
        assert result.tool_name == "zoom"
        assert "image" in result.artifacts
        # Original crop: 200x300, zoomed 2x = 400x600
        assert result.artifacts["image"].size == (400, 600)
        assert result.artifacts["region"] == (100, 100, 300, 400)
        assert result.artifacts["scale"] == 2.0
        assert result.artifacts["original_size"] == (200, 300)
        assert result.artifacts["zoomed_size"] == (400, 600)
        assert "Zoomed region" in result.summary
        assert "2.0x" in result.summary

    def test_zoom_3x(self):
        """Zoom with custom 3x scaling."""
        img = Image.new("RGB", (800, 600), color="blue")
        result = zoom_tool(img, (100, 100, 200, 200), scale=3.0)

        assert result.success is True
        # Original crop: 100x100, zoomed 3x = 300x300
        assert result.artifacts["image"].size == (300, 300)
        assert result.artifacts["scale"] == 3.0

    def test_zoom_half_scale(self):
        """Zoom with 0.5x scaling (downsampling)."""
        img = Image.new("RGB", (800, 600), color="green")
        result = zoom_tool(img, (0, 0, 400, 400), scale=0.5)

        assert result.success is True
        # Original crop: 400x400, zoomed 0.5x = 200x200
        assert result.artifacts["image"].size == (200, 200)

    def test_zoom_with_clamping(self):
        """Zoom region that needs clamping."""
        img = Image.new("RGB", (800, 600), color="white")
        result = zoom_tool(img, (-10, -20, 100, 200), scale=2.0)

        assert result.success is True
        # Clamped to (0, 0, 100, 200) = 100x200 crop, 2x = 200x400
        assert result.artifacts["region"] == (0, 0, 100, 200)
        assert result.artifacts["image"].size == (200, 400)

    def test_zoom_exceeds_bounds(self):
        """Zoom region that exceeds image bounds."""
        img = Image.new("RGB", (800, 600), color="black")
        result = zoom_tool(img, (700, 500, 900, 700), scale=1.5)

        assert result.success is True
        # Clamped to (700, 500, 800, 600) = 100x100 crop, 1.5x = 150x150
        assert result.artifacts["region"] == (700, 500, 800, 600)
        assert result.artifacts["image"].size == (150, 150)

    def test_zoom_default_scale(self):
        """Zoom uses default 2.0 scale when not specified."""
        img = Image.new("RGB", (800, 600), color="yellow")
        result = zoom_tool(img, (100, 100, 200, 200))  # No scale param

        assert result.success is True
        assert result.artifacts["scale"] == 2.0
        assert result.artifacts["image"].size == (200, 200)  # 100x100 * 2

    def test_zoom_invalid_scale_zero(self):
        """Zoom with scale=0 returns failure ToolResult."""
        img = Image.new("RGB", (800, 600), color="cyan")
        result = zoom_tool(img, (100, 100, 300, 400), scale=0.0)

        assert result.success is False
        assert "Failed to zoom region" in result.summary
        assert "Scale must be positive" in result.summary

    def test_zoom_invalid_scale_negative(self):
        """Zoom with negative scale returns failure ToolResult."""
        img = Image.new("RGB", (800, 600), color="magenta")
        result = zoom_tool(img, (100, 100, 300, 400), scale=-1.5)

        assert result.success is False
        assert "Failed to zoom region" in result.summary

    def test_zoom_invalid_region(self):
        """Zoom with invalid region returns failure ToolResult."""
        img = Image.new("RGB", (800, 600), color="purple")
        result = zoom_tool(img, (100, 100, 100, 200), scale=2.0)

        assert result.success is False
        assert "Failed to zoom region" in result.summary

    def test_zoom_float_coords(self):
        """Zoom handles float coordinates (converts to int)."""
        img = Image.new("RGB", (800, 600), color="orange")
        result = zoom_tool(img, (10.5, 20.7, 110.9, 120.1), scale=2.0)

        assert result.success is True
        assert result.artifacts["region"] == (10, 20, 110, 120)
        # 100x100 crop * 2 = 200x200
        assert result.artifacts["image"].size == (200, 200)

    def test_zoom_preserves_color_mode(self):
        """Zoom preserves image color mode (RGB)."""
        img = Image.new("RGB", (800, 600), color=(128, 64, 32))
        result = zoom_tool(img, (100, 100, 200, 200), scale=2.0)

        assert result.success is True
        assert result.artifacts["image"].mode == "RGB"


class TestBuiltinToolsRegistry:
    """Tests for BUILTIN_TOOLS dictionary."""

    def test_registry_contains_crop(self):
        """Registry contains crop_tool."""
        assert "crop" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["crop"] is crop_tool

    def test_registry_contains_zoom(self):
        """Registry contains zoom_tool."""
        assert "zoom" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["zoom"] is zoom_tool

    def test_registry_exactly_two_tools(self):
        """Registry contains exactly the expected tools."""
        assert len(BUILTIN_TOOLS) == 2
        assert set(BUILTIN_TOOLS.keys()) == {"crop", "zoom"}

    def test_registry_tools_callable(self):
        """All tools in registry are callable."""
        for tool_name, tool_func in BUILTIN_TOOLS.items():
            assert callable(tool_func), f"{tool_name} should be callable"


class TestToolResultFormat:
    """Tests for ToolResult formatting and structure."""

    def test_crop_result_has_all_fields(self):
        """Crop ToolResult has all required fields."""
        img = Image.new("RGB", (800, 600), color="red")
        result = crop_tool(img, (100, 100, 300, 400))

        assert hasattr(result, "tool_name")
        assert hasattr(result, "success")
        assert hasattr(result, "summary")
        assert hasattr(result, "artifacts")
        assert result.tool_name == "crop"
        assert isinstance(result.success, bool)
        assert isinstance(result.summary, str)
        assert isinstance(result.artifacts, dict)

    def test_zoom_result_has_all_fields(self):
        """Zoom ToolResult has all required fields."""
        img = Image.new("RGB", (800, 600), color="blue")
        result = zoom_tool(img, (100, 100, 300, 400), scale=2.0)

        assert hasattr(result, "tool_name")
        assert hasattr(result, "success")
        assert hasattr(result, "summary")
        assert hasattr(result, "artifacts")
        assert result.tool_name == "zoom"
        assert isinstance(result.success, bool)
        assert isinstance(result.summary, str)
        assert isinstance(result.artifacts, dict)

    def test_crop_summary_includes_dimensions(self):
        """Crop summary includes region coordinates and dimensions."""
        img = Image.new("RGB", (800, 600), color="green")
        result = crop_tool(img, (50, 75, 250, 375))

        assert "(50, 75, 250, 375)" in result.summary
        assert "200x300" in result.summary  # width x height

    def test_zoom_summary_includes_scale_and_sizes(self):
        """Zoom summary includes scale factor and before/after sizes."""
        img = Image.new("RGB", (800, 600), color="white")
        result = zoom_tool(img, (100, 100, 300, 400), scale=2.5)

        assert "2.5x" in result.summary
        assert "200x300" in result.summary  # original crop size
        assert "500x750" in result.summary  # zoomed size

    def test_failure_result_has_error_message(self):
        """Failure ToolResult includes error description in summary."""
        img = Image.new("RGB", (800, 600), color="black")
        result = crop_tool(img, (100, 100, 100, 200))  # Invalid region

        assert result.success is False
        assert len(result.summary) > 0
        assert "Failed" in result.summary or "Invalid" in result.summary
