"""Crop exporter for MATA detection results.

Extracts individual detection crops from images and saves them as separate files.
Useful for dataset augmentation, object galleries, and detailed inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError
from mata.core.logging import get_logger

if TYPE_CHECKING:
    from mata.core.types import DetectResult, VisionResult

logger = get_logger(__name__)


def export_crops(
    result: VisionResult | DetectResult,
    output_path: str | Path,
    image: str | Path | Image.Image | np.ndarray | None = None,
    crop_dir: str | None = None,
    padding: int = 0,
    **kwargs,
) -> None:
    """Export individual detection crops.

    Extracts each detection as a separate cropped image file.
    Creates a subdirectory with numbered crops (e.g., output_000.png, output_001.png).

    Args:
        result: Detection result with bounding boxes
        output_path: Base path for crop directory/naming
        image: Original image (path, PIL Image, or numpy array).
            If None, uses result.meta['input_path'] if available.
        crop_dir: Custom crop directory name (default: '{output_path}_crops')
        padding: Pixels to add around each crop (default: 0)
        **kwargs: Reserved for future use

    Raises:
        InvalidInputError: If image is None and not in result.meta
        ValueError: If no detections with bboxes found
        IOError: If crop save fails

    Examples:
        >>> # Auto-create crops in 'output_crops/' directory
        >>> result = mata.run("detect", "image.jpg")
        >>> export_crops(result, "output.png")
        >>> # Creates: output_crops/output_000.png, output_001.png, ...
        >>>
        >>> # Custom crop directory
        >>> export_crops(result, "detections.png", crop_dir="my_crops")
        >>> # Creates: my_crops/detections_000.png, detections_001.png, ...
        >>>
        >>> # Add padding around crops
        >>> export_crops(result, "output.png", padding=10)
    """
    output_path = Path(output_path)

    # Resolve image path
    if image is None:
        if hasattr(result, "get_input_path"):
            image = result.get_input_path()
        elif hasattr(result, "meta") and result.meta is not None and "input_path" in result.meta:
            image = result.meta["input_path"]

        if image is None:
            raise InvalidInputError(
                "Crop export requires original image. "
                "Provide via: export_crops(result, output_path, image='path.jpg')"
            )

    # Load image
    pil_image = _load_image(image)
    img_width, img_height = pil_image.size

    # Get instances with bboxes
    if hasattr(result, "detections"):
        instances = result.detections
    elif hasattr(result, "instances"):
        instances = [inst for inst in result.instances if inst.bbox is not None]
    else:
        instances = []

    if len(instances) == 0:
        raise ValueError("No detections with bounding boxes found for crop extraction")

    # Determine crop directory
    if crop_dir is None:
        crop_dir = f"{output_path.stem}_crops"

    crop_path = output_path.parent / crop_dir
    crop_path.mkdir(parents=True, exist_ok=True)

    # Extract and save crops
    saved_count = 0
    for idx, inst in enumerate(instances):
        if inst.bbox is None:
            continue

        x1, y1, x2, y2 = inst.bbox

        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)

        # Ensure valid crop region
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Skipping invalid crop {idx}: bbox {inst.bbox}")
            continue

        # Extract crop
        crop = pil_image.crop((x1, y1, x2, y2))

        # Save crop
        crop_filename = f"{output_path.stem}_{idx:03d}{output_path.suffix}"
        crop_file_path = crop_path / crop_filename

        try:
            crop.save(crop_file_path)
            saved_count += 1
            logger.debug(f"Saved crop {idx} to {crop_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save crop {idx}: {e}")
            continue

    logger.info(f"Exported {saved_count}/{len(instances)} crops to {crop_path}/")


def _load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load and convert image to PIL Image."""
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        raise InvalidInputError(f"Unsupported image type: {type(image).__name__}")
    return pil_image
