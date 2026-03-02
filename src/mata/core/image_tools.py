"""Built-in image tools for VLM agent tool-calling system.

This module implements pure image operations (zoom, crop) that VLMs can call
to focus on regions of interest. These tools do not perform model inference —
they are preprocessing steps that narrow the VLM's visual attention.

Design principles:
- Zero external dependencies (PIL only)
- Coordinate validation with clamping (no crashes on out-of-bounds)
- xyxy format (absolute pixels) matching MATA convention
- ToolResult with both summary text and image artifact

Version: 1.7.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PIL import Image

from mata.core.logging import get_logger
from mata.core.tool_schema import ToolResult

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _validate_and_clamp_region(
    region: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Validate and clamp region to image bounds.

    Args:
        region: Bounding box in xyxy format (x1, y1, x2, y2)
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Clamped region as (x1, y1, x2, y2) in integer pixel coordinates

    Raises:
        ValueError: If region is invalid (x2 <= x1 or y2 <= y1 after clamping)

    Examples:
        >>> _validate_and_clamp_region((10, 20, 100, 200), 800, 600)
        (10, 20, 100, 200)
        >>> _validate_and_clamp_region((-10, -20, 100, 200), 800, 600)
        (0, 0, 100, 200)
        >>> _validate_and_clamp_region((100, 200, 900, 700), 800, 600)
        (100, 200, 800, 600)
    """
    x1, y1, x2, y2 = region

    # Clamp to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image_width, int(x2))
    y2 = min(image_height, int(y2))

    # Validate region is non-empty
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop region after clamping: ({x1}, {y1}, {x2}, {y2}). "
            f"Region must have positive width and height. "
            f"Original region: {region}, Image size: {image_width}x{image_height}"
        )

    return x1, y1, x2, y2


def crop_tool(image: Image.Image, region: tuple[float, float, float, float]) -> ToolResult:
    """Crop a region from the image without upscaling.

    Extracts a rectangular region from the image. The VLM can use this to focus
    on a specific area without zooming in. Coordinates are clamped to image bounds.

    Args:
        image: PIL Image to crop
        region: Bounding box in xyxy format (x1, y1, x2, y2) in absolute pixels

    Returns:
        ToolResult with cropped image and text summary

    Raises:
        ValueError: If region is invalid after clamping

    Examples:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (800, 600), color="white")
        >>> result = crop_tool(img, (100, 100, 300, 400))
        >>> result.success
        True
        >>> result.artifacts["image"].size
        (200, 300)
        >>> "Cropped region" in result.summary
        True
    """
    logger.debug(f"crop_tool called with region={region}, image size={image.size}")

    try:
        # Validate and clamp region
        x1, y1, x2, y2 = _validate_and_clamp_region(region, image.width, image.height)

        # Crop image (PIL.Image.crop uses xyxy format)
        cropped = image.crop((x1, y1, x2, y2))

        # Build summary
        width = x2 - x1
        height = y2 - y1
        summary = f"Cropped region ({x1}, {y1}, {x2}, {y2}) from image. " f"Crop size: {width}x{height} pixels."

        logger.info(f"crop_tool successful: {width}x{height} crop")

        return ToolResult(
            tool_name="crop",
            success=True,
            summary=summary,
            artifacts={"image": cropped, "region": (x1, y1, x2, y2)},
        )

    except Exception as e:
        logger.error(f"crop_tool failed: {e}")
        return ToolResult(
            tool_name="crop",
            success=False,
            summary=f"Failed to crop region: {e}",
            artifacts={},
        )


def zoom_tool(
    image: Image.Image,
    region: tuple[float, float, float, float],
    scale: float = 2.0,
) -> ToolResult:
    """Crop a region from the image and upscale by a scale factor.

    Extracts a rectangular region and upscales it so the VLM can see more detail.
    Uses LANCZOS resampling for high-quality upscaling. Coordinates are clamped
    to image bounds.

    Args:
        image: PIL Image to crop and zoom
        region: Bounding box in xyxy format (x1, y1, x2, y2) in absolute pixels
        scale: Upscaling factor (default: 2.0). Must be > 0. Values < 1 will downsample.

    Returns:
        ToolResult with zoomed image and text summary

    Raises:
        ValueError: If region is invalid after clamping or scale <= 0

    Examples:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (800, 600), color="white")
        >>> result = zoom_tool(img, (100, 100, 300, 400), scale=2.0)
        >>> result.success
        True
        >>> result.artifacts["image"].size
        (400, 600)
        >>> "Zoomed" in result.summary
        True
        >>> "2.0x" in result.summary
        True
    """
    logger.debug(f"zoom_tool called with region={region}, scale={scale}, image size={image.size}")

    try:
        # Validate scale parameter
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")

        # Validate and clamp region
        x1, y1, x2, y2 = _validate_and_clamp_region(region, image.width, image.height)

        # Crop image
        cropped = image.crop((x1, y1, x2, y2))

        # Calculate new dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
        new_width = int(crop_width * scale)
        new_height = int(crop_height * scale)

        # Upscale with LANCZOS resampling (high quality)
        zoomed = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Build summary
        summary = (
            f"Zoomed region ({x1}, {y1}, {x2}, {y2}) by {scale}x. "
            f"Original crop: {crop_width}x{crop_height} pixels. "
            f"Zoomed size: {new_width}x{new_height} pixels."
        )

        logger.info(f"zoom_tool successful: {crop_width}x{crop_height} → {new_width}x{new_height} ({scale}x)")

        return ToolResult(
            tool_name="zoom",
            success=True,
            summary=summary,
            artifacts={
                "image": zoomed,
                "region": (x1, y1, x2, y2),
                "scale": scale,
                "original_size": (crop_width, crop_height),
                "zoomed_size": (new_width, new_height),
            },
        )

    except Exception as e:
        logger.error(f"zoom_tool failed: {e}")
        return ToolResult(
            tool_name="zoom",
            success=False,
            summary=f"Failed to zoom region: {e}",
            artifacts={},
        )


# Registry of built-in tools
BUILTIN_TOOLS: dict[str, Callable] = {
    "zoom": zoom_tool,
    "crop": crop_tool,
}
