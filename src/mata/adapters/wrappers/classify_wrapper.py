"""Classification adapter wrapper for graph system.

Converts existing classification adapters (HuggingFace, ONNX, TorchScript, CLIP)
into Classifier capability providers that work with the graph artifact system.

The wrapper bridges:
- Input: Image artifact → PIL/path for adapter
- Output: ClassifyResult (passed through, no artifact conversion needed)

Note: The Classifier protocol returns ClassifyResult directly since the
Classifications artifact is not yet implemented. When it becomes available,
this wrapper will be updated to convert to the artifact type.
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.image import Image
from mata.core.types import ClassifyResult

logger = logging.getLogger(__name__)


class ClassifierWrapper:
    """Wraps classification adapters as Classifier capability providers.

    Bridges the graph system's Image artifact input to the adapter's predict()
    interface, and maps the classify() method to predict().

    Implements the Classifier protocol:
        classify(image: Image, **kwargs) -> ClassifyResult

    Supported adapters:
    - HuggingFaceClassifyAdapter (ResNet, ViT, EfficientNet, ConvNeXt)
    - HuggingFaceCLIPAdapter (CLIP zero-shot classification)
    - ONNXClassifyAdapter
    - TorchScriptClassifyAdapter
    - PyTorchClassifyAdapter

    Example:
        >>> from mata.adapters.wrappers import wrap_classifier
        >>> adapter = HuggingFaceClassifyAdapter("google/vit-base-patch16-224")
        >>> classifier = wrap_classifier(adapter)
        >>> img = Image.from_path("photo.jpg")
        >>> result = classifier.classify(img, top_k=5)
        >>> print(result.top1.label_name, result.top1.score)
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with an existing classification adapter.

        Args:
            adapter: Any classification adapter with a predict() method that
                accepts image input and returns ClassifyResult.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"ClassifierWrapper requires an adapter with predict(image, **kwargs) -> ClassifyResult."
            )
        self.adapter = adapter

    def classify(self, image: Image, **kwargs) -> ClassifyResult:
        """Classify an Image artifact.

        Converts Image artifact to adapter-compatible format and calls
        adapter.predict(). Returns ClassifyResult directly (no artifact
        conversion, as Classifications artifact is not yet implemented).

        Args:
            image: Image artifact from the graph system.
            **kwargs: Passed through to adapter.predict(). Common kwargs:
                - top_k (int): Number of top predictions
                - text_prompts (List[str]): For zero-shot models (CLIP)

        Returns:
            ClassifyResult with sorted classification predictions.

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
            raise RuntimeError(f"Classification adapter {type(self.adapter).__name__} failed: {e}") from e

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
        return f"ClassifierWrapper(adapter={type(self.adapter).__name__})"


def wrap_classifier(adapter: Any) -> ClassifierWrapper:
    """Wrap any classification adapter as a Classifier capability provider.

    Factory function for creating ClassifierWrapper instances.

    Args:
        adapter: Classification adapter with predict() returning ClassifyResult.

    Returns:
        ClassifierWrapper implementing the Classifier protocol.
    """
    return ClassifierWrapper(adapter)
