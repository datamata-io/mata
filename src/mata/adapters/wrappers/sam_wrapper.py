"""SAM adapter wrapper for graph system.

Converts the SAM adapter (SAM, SAM2, SAM3) into a Segmenter capability provider
that supports prompt-based segmentation with the graph artifact system.

The wrapper bridges:
- Input: Image artifact + prompt kwargs → PIL/path for adapter
- Output: VisionResult → Masks artifact

SAM supports multiple prompting modes:
- Box prompts: Bounding boxes from prior detection results
- Point prompts: Interactive point clicks (foreground/background)
- Text prompts: Zero-shot text descriptions (SAM3)
- Everything mode: Automatic mask generation
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.converters import vision_result_to_masks
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks

logger = logging.getLogger(__name__)


class SAMWrapper:
    """Wraps SAM adapters as Segmenter capability providers with prompt support.

    Provides the Segmenter protocol interface while supporting SAM-specific
    prompt types (boxes, points, text, everything mode).

    Implements the Segmenter protocol:
        segment(image: Image, **kwargs) -> Masks

    Supported adapters:
    - HuggingFaceSAMAdapter (SAM, SAM2, SAM3 with text prompts)

    Prompt modes via kwargs:
    - box_prompts: List of (x1, y1, x2, y2) bounding boxes
    - point_prompts: List of (x, y, label) point prompts
    - text_prompts: Text description for SAM3 zero-shot
    - mode: "everything" for automatic mask generation

    Example:
        >>> from mata.adapters.wrappers import wrap_sam
        >>> adapter = HuggingFaceSAMAdapter("facebook/sam-vit-base")
        >>> segmenter = wrap_sam(adapter)
        >>>
        >>> img = Image.from_path("photo.jpg")
        >>>
        >>> # Box-prompted segmentation
        >>> masks = segmenter.segment(img, box_prompts=[(50, 50, 300, 300)])
        >>>
        >>> # Point-prompted segmentation
        >>> masks = segmenter.segment(img, point_prompts=[(175, 175, 1)])
        >>>
        >>> # Text-prompted (SAM3)
        >>> masks = segmenter.segment(img, text_prompts="cat")
        >>>
        >>> # Everything mode
        >>> masks = segmenter.segment(img, mode="everything")
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with a SAM adapter.

        Args:
            adapter: A SAM adapter with a predict() method supporting
                box_prompts, point_prompts, text_prompts, and automatic mode.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"SAMWrapper requires a SAM adapter with predict(image, **kwargs) -> VisionResult."
            )
        self.adapter = adapter

    def segment(self, image: Image, **kwargs) -> Masks:
        """Segment an Image artifact using SAM with optional prompts.

        Converts Image artifact to adapter-compatible format, extracts
        SAM-specific prompt kwargs, calls adapter.predict(), and converts
        VisionResult to Masks artifact.

        Args:
            image: Image artifact from the graph system.
            **kwargs: SAM-specific parameters:
                - box_prompts (List[Tuple]): Bounding boxes [(x1,y1,x2,y2), ...]
                - box_labels (List[int]): Labels per box (1=foreground default)
                - point_prompts (List[Tuple]): Points [(x,y,label), ...]
                - text_prompts (str/List[str]): Text prompts (SAM3 zero-shot)
                - threshold (float): Confidence threshold
                - mode (str): "everything" for automatic mask generation

        Returns:
            Masks artifact with segmentation masks and instance_ids.

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

        # Extract SAM-specific kwargs and pass through
        sam_kwargs = {}
        sam_params = ["box_prompts", "box_labels", "point_prompts", "text_prompts", "threshold", "mode"]
        for param in sam_params:
            if param in kwargs:
                sam_kwargs[param] = kwargs.pop(param)

        # Pass remaining kwargs through as well
        sam_kwargs.update(kwargs)

        try:
            result = self.adapter.predict(img_input, **sam_kwargs)
        except Exception as e:
            raise RuntimeError(f"SAM adapter {type(self.adapter).__name__} failed: {e}") from e

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
        return f"SAMWrapper(adapter={type(self.adapter).__name__})"


def wrap_sam(adapter: Any) -> SAMWrapper:
    """Wrap a SAM adapter as a Segmenter capability provider.

    Factory function for creating SAMWrapper instances. Use this for SAM,
    SAM2, and SAM3 adapters that support prompt-based segmentation.

    Args:
        adapter: SAM adapter with predict() method returning VisionResult.

    Returns:
        SAMWrapper implementing the Segmenter protocol with prompt support.
    """
    return SAMWrapper(adapter)
