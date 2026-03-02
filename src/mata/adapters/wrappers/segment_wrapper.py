"""Segmentation adapter wrapper for graph system.

Converts existing segmentation adapters (HuggingFace Mask2Former, MaskFormer) into
Segmenter capability providers that work with the graph artifact system.

The wrapper bridges:
- Input: Image artifact → PIL/path for adapter
- Output: VisionResult → Masks artifact
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.converters import vision_result_to_masks
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks

logger = logging.getLogger(__name__)


class SegmenterWrapper:
    """Wraps segment adapters as Segmenter capability providers.

    Bridges the graph system's artifact-based types and the existing adapter
    predict() interface, translating method names (segment → predict) and
    converting types (Image → PIL, VisionResult → Masks).

    Implements the Segmenter protocol:
        segment(image: Image, **kwargs) -> Masks

    Supported adapters:
    - HuggingFaceSegmentAdapter (Mask2Former, MaskFormer, panoptic)

    Note:
        For SAM-style prompt-based segmentation, use SAMWrapper instead.

    Example:
        >>> from mata.adapters.wrappers import wrap_segmenter
        >>> adapter = HuggingFaceSegmentAdapter("facebook/mask2former-swin-base-coco-instance")
        >>> segmenter = wrap_segmenter(adapter)
        >>> img = Image.from_path("photo.jpg")
        >>> masks = segmenter.segment(img, threshold=0.5)
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with an existing segment adapter.

        Args:
            adapter: Any segmentation adapter with a predict() method that
                accepts image input and returns VisionResult with mask data.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"SegmenterWrapper requires an adapter with predict(image, **kwargs) -> VisionResult."
            )
        self.adapter = adapter

    def segment(self, image: Image, **kwargs) -> Masks:
        """Segment an Image artifact into masks.

        Converts Image artifact to adapter-compatible format, calls adapter.predict(),
        and converts VisionResult to Masks artifact.

        Args:
            image: Image artifact from the graph system.
            **kwargs: Passed through to adapter.predict(). Common kwargs:
                - threshold (float): Confidence threshold

        Returns:
            Masks artifact with instance masks and instance_ids.

        Raises:
            TypeError: If image is not an Image artifact.
            RuntimeError: If adapter prediction fails.
        """
        if not isinstance(image, Image):
            raise TypeError(
                f"Expected Image artifact, got {type(image).__name__}. "
                f"Use Image.from_path() or Image.from_pil() to create an Image artifact."
            )

        img_input = self._convert_image(image)

        try:
            result = self.adapter.predict(img_input, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Segment adapter {type(self.adapter).__name__} failed: {e}") from e

        return vision_result_to_masks(result)

    def _convert_image(self, image: Image) -> Any:
        """Convert Image artifact to adapter-compatible format.

        Args:
            image: Image artifact to convert.

        Returns:
            str path or PIL.Image.Image suitable for adapter.predict()
        """
        if image.source_path:
            return image.source_path
        return image.to_pil()

    def __repr__(self) -> str:
        return f"SegmenterWrapper(adapter={type(self.adapter).__name__})"


def wrap_segmenter(adapter: Any) -> SegmenterWrapper:
    """Wrap any segment adapter as a Segmenter capability provider.

    Factory function for creating SegmenterWrapper instances.

    Args:
        adapter: Segmentation adapter with predict() method returning VisionResult.

    Returns:
        SegmenterWrapper implementing the Segmenter protocol.
    """
    return SegmenterWrapper(adapter)
