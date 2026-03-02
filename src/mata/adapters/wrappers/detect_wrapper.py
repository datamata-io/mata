"""Detection adapter wrapper for graph system.

Converts existing detection adapters (HuggingFace, ONNX, TorchScript) into
Detector capability providers that work with the graph artifact system.

The wrapper bridges:
- Input: Image artifact → PIL/path for adapter
- Output: VisionResult → Detections artifact
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.converters import vision_result_to_detections
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image

logger = logging.getLogger(__name__)


class DetectorWrapper:
    """Wraps detect adapters as Detector capability providers.

    Bridges the gap between the graph system's artifact-based types and the
    existing adapter predict() interface which accepts paths/PIL/numpy and
    returns VisionResult.

    Implements the Detector protocol:
        predict(image: Image, **kwargs) -> Detections

    Supported adapters:
    - HuggingFaceDetectAdapter (DETR, RT-DETR, DINO)
    - HuggingFaceZeroShotDetectAdapter (GroundingDINO, OWL-ViT)
    - ONNXDetectAdapter
    - TorchScriptDetectAdapter
    - TorchVisionDetectAdapter

    Example:
        >>> from mata.adapters.wrappers import wrap_detector
        >>> adapter = HuggingFaceDetectAdapter("facebook/detr-resnet-50")
        >>> detector = wrap_detector(adapter)
        >>> img = Image.from_path("photo.jpg")
        >>> dets = detector.predict(img, threshold=0.5)
        >>> print(len(dets.instances))
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with an existing detect adapter.

        Args:
            adapter: Any detection adapter with a predict() method that
                accepts image input and returns VisionResult.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"DetectorWrapper requires an adapter with predict(image, **kwargs) -> VisionResult."
            )
        self.adapter = adapter

    def predict(self, image: Image, **kwargs) -> Detections:
        """Detect objects in an Image artifact.

        Converts the Image artifact to adapter-compatible format, calls the
        underlying adapter's predict(), and converts VisionResult to Detections.

        Args:
            image: Image artifact from the graph system.
            **kwargs: Passed through to adapter.predict(). Common kwargs:
                - threshold (float): Confidence threshold
                - text_prompts (str/List[str]): For zero-shot models
                - nms_iou (float): NMS IoU threshold

        Returns:
            Detections artifact with instances, instance_ids, and metadata.

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
            raise RuntimeError(f"Detection adapter {type(self.adapter).__name__} failed: {e}") from e

        return vision_result_to_detections(result)

    def _convert_image(self, image: Image) -> Any:
        """Convert Image artifact to adapter-compatible format.

        Prefers source_path (avoids unnecessary conversion) then falls back
        to PIL Image.

        Args:
            image: Image artifact to convert.

        Returns:
            str path or PIL.Image.Image suitable for adapter.predict()
        """
        if image.source_path:
            return image.source_path
        return image.to_pil()

    def __repr__(self) -> str:
        return f"DetectorWrapper(adapter={type(self.adapter).__name__})"


def wrap_detector(adapter: Any) -> DetectorWrapper:
    """Wrap any detect adapter as a Detector capability provider.

    Factory function for creating DetectorWrapper instances.

    Args:
        adapter: Detection adapter with predict() method returning VisionResult.

    Returns:
        DetectorWrapper implementing the Detector protocol.

    Example:
        >>> detector = wrap_detector(HuggingFaceDetectAdapter("facebook/detr-resnet-50"))
        >>> isinstance(detector, Detector)  # True (runtime protocol check)
        True
    """
    return DetectorWrapper(adapter)
