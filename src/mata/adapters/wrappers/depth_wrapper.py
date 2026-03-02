"""Depth estimation adapter wrapper for graph system.

Converts existing depth adapters (HuggingFace Depth Anything V1/V2) into
DepthEstimator capability providers that work with the graph artifact system.

The wrapper bridges:
- Input: Image artifact → PIL/path for adapter
- Output: DepthResult (passed through, no artifact conversion needed)

Note: The DepthEstimator protocol returns DepthResult directly since the
DepthMap artifact is not yet implemented. When it becomes available,
this wrapper will be updated to convert to the artifact type.
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.image import Image
from mata.core.types import DepthResult

logger = logging.getLogger(__name__)


class DepthWrapper:
    """Wraps depth adapters as DepthEstimator capability providers.

    Bridges the graph system's Image artifact input to the adapter's predict()
    interface, and maps the estimate() method to predict().

    Implements the DepthEstimator protocol:
        estimate(image: Image, **kwargs) -> DepthResult

    Supported adapters:
    - HuggingFaceDepthAdapter (Depth Anything V1/V2, MiDaS, DPT)

    Example:
        >>> from mata.adapters.wrappers import wrap_depth
        >>> adapter = HuggingFaceDepthAdapter("depth-anything/Depth-Anything-V2-Small-hf")
        >>> depth_estimator = wrap_depth(adapter)
        >>> img = Image.from_path("photo.jpg")
        >>> result = depth_estimator.estimate(img)
        >>> result.save("depth.png", colormap="magma")
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with an existing depth adapter.

        Args:
            adapter: Any depth adapter with a predict() method that
                accepts image input and returns DepthResult.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"DepthWrapper requires an adapter with predict(image, **kwargs) -> DepthResult."
            )
        self.adapter = adapter

    def estimate(self, image: Image, **kwargs) -> DepthResult:
        """Estimate depth from an Image artifact.

        Converts Image artifact to adapter-compatible format and calls
        adapter.predict(). Returns DepthResult directly (no artifact
        conversion, as DepthMap artifact is not yet implemented).

        Args:
            image: Image artifact from the graph system.
            **kwargs: Passed through to adapter.predict(). Common kwargs:
                - output_type (str): "normalized" or "raw"

        Returns:
            DepthResult with depth map array and metadata.

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
            raise RuntimeError(f"Depth adapter {type(self.adapter).__name__} failed: {e}") from e

        return result

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
        return f"DepthWrapper(adapter={type(self.adapter).__name__})"


def wrap_depth(adapter: Any) -> DepthWrapper:
    """Wrap any depth adapter as a DepthEstimator capability provider.

    Factory function for creating DepthWrapper instances.

    Args:
        adapter: Depth adapter with predict() returning DepthResult.

    Returns:
        DepthWrapper implementing the DepthEstimator protocol.
    """
    return DepthWrapper(adapter)
