"""VLM adapter wrapper for graph system.

Converts the HuggingFace VLM adapter (Qwen2-VL, Qwen3-VL, etc.) into a
VisionLanguageModel capability provider that works with the graph artifact system.

The wrapper bridges:
- Input: Image artifact(s) + prompt → PIL/path for adapter
- Output: VisionResult (passed through with entities/instances preserved)

Supports:
- Single-image queries (standard VQA)
- Multi-image queries (comparison, multi-view reasoning)
- Structured output modes (json, detect, classify, describe)
- Entity auto-promotion (entities → instances when spatial data present)
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.image import Image
from mata.core.types import VisionResult

logger = logging.getLogger(__name__)


class VLMWrapper:
    """Wraps VLM adapters as VisionLanguageModel capability providers.

    Bridges the graph system's Image artifact types to the VLM adapter's predict()
    interface, and maps the query() method to predict().

    Implements the VisionLanguageModel protocol:
        query(image: Union[Image, List[Image]], prompt: str,
              output_mode: Optional[str], auto_promote: bool, **kwargs) -> VisionResult

    Supported adapters:
    - HuggingFaceVLMAdapter (Qwen2-VL, Qwen3-VL, LLaVA, InstructBLIP)

    Example:
        >>> from mata.adapters.wrappers import wrap_vlm
        >>> adapter = HuggingFaceVLMAdapter("Qwen/Qwen2-VL-7B-Instruct")
        >>> vlm = wrap_vlm(adapter)
        >>>
        >>> # Simple query
        >>> img = Image.from_path("photo.jpg")
        >>> result = vlm.query(img, "What objects are in this image?")
        >>> print(result.text)
        >>>
        >>> # Structured detection
        >>> result = vlm.query(img, "List all objects.", output_mode="detect")
        >>> for entity in result.entities:
        ...     print(f"{entity.label}: {entity.score:.2f}")
        >>>
        >>> # Multi-image reasoning
        >>> images = [Image.from_path(f"img{i}.jpg") for i in range(3)]
        >>> result = vlm.query(images, "What changed between these images?")
        >>>
        >>> # Auto-promotion (entities → instances with spatial data)
        >>> result = vlm.query(img, "Detect objects with bounding boxes.",
        ...                    output_mode="detect", auto_promote=True)
    """

    def __init__(self, adapter: Any):
        """Initialize wrapper with a VLM adapter.

        Args:
            adapter: A VLM adapter with a predict() method that accepts image
                input, prompt, and optional output_mode/auto_promote parameters,
                returning VisionResult.

        Raises:
            TypeError: If adapter does not have a predict method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(
                f"Adapter {type(adapter).__name__} does not have a predict() method. "
                f"VLMWrapper requires an adapter with predict(image, prompt=..., **kwargs) -> VisionResult."
            )
        self.adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped adapter.

        This allows transparent access to adapter methods like predict(),
        info(), etc., so VLMWrapper is a drop-in replacement for the raw
        adapter in all contexts.
        """
        return getattr(self.adapter, name)

    def query(
        self,
        image: Image | list[Image],
        prompt: str,
        output_mode: str | None = None,
        auto_promote: bool = False,
        **kwargs: Any,
    ) -> VisionResult:
        """Query VLM with image(s) and text prompt.

        Converts Image artifact(s) to adapter-compatible format, calls
        adapter.predict(), and returns VisionResult with entities/instances
        preserved from the VLM adapter's structured output parsing.

        Args:
            image: Single Image artifact or list of Image artifacts for
                multi-image queries.
            prompt: Text prompt/question for the VLM.
            output_mode: Structured output format:
                - None: Raw text response
                - "json": Generic JSON output
                - "detect": Detection format (label + confidence)
                - "classify": Classification format
                - "describe": Natural language description
            auto_promote: If True, promote entities with spatial data
                (bbox/mask) to Instance objects automatically.
            **kwargs: Additional VLM parameters:
                - system_prompt (str): System prompt override
                - max_new_tokens (int): Max response length
                - temperature (float): Sampling temperature
                - top_p (float): Nucleus sampling probability
                - top_k (int): Top-k sampling

        Returns:
            VisionResult with:
                - text: Raw VLM response
                - entities: Parsed entities (if output_mode set)
                - instances: Auto-promoted instances (if auto_promote=True)
                - meta: Model metadata

        Raises:
            TypeError: If image is not Image or List[Image].
            ValueError: If image list is empty or prompt is empty.
            RuntimeError: If VLM inference fails.
        """
        if isinstance(image, list):
            if not image:
                raise ValueError("Image list cannot be empty.")
            if not all(isinstance(img, Image) for img in image):
                raise TypeError(
                    "All items in image list must be Image artifacts. "
                    "Use Image.from_path() or Image.from_pil() to create Image artifacts."
                )
            # First image is primary, rest are additional
            primary = self._convert_image(image[0])
            additional = [self._convert_image(img) for img in image[1:]] if len(image) > 1 else None
        elif isinstance(image, Image):
            primary = self._convert_image(image)
            additional = None
        else:
            raise TypeError(
                f"Expected Image or List[Image], got {type(image).__name__}. "
                f"Use Image.from_path() or Image.from_pil() to create an Image artifact."
            )

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        # Build adapter kwargs
        adapter_kwargs = {
            "prompt": prompt,
            "output_mode": output_mode,
            "auto_promote": auto_promote,
        }
        if additional is not None:
            adapter_kwargs["images"] = additional
        adapter_kwargs.update(kwargs)

        try:
            result = self.adapter.predict(primary, **adapter_kwargs)
        except Exception as e:
            raise RuntimeError(f"VLM adapter {type(self.adapter).__name__} failed: {e}") from e

        return result

    def _convert_image(self, image: Image) -> Any:
        """Convert Image artifact to adapter-compatible format.

        Prefers source_path (adapter handles file loading internally)
        over PIL to avoid unnecessary PIL→PIL round-trips.

        Args:
            image: Image artifact to convert.

        Returns:
            str path or PIL.Image.Image suitable for adapter.predict()
        """
        if image.source_path:
            return image.source_path
        return image.to_pil()

    def __repr__(self) -> str:
        return f"VLMWrapper(adapter={type(self.adapter).__name__})"


def wrap_vlm(adapter: Any) -> VLMWrapper:
    """Wrap any VLM adapter as a VisionLanguageModel capability provider.

    Factory function for creating VLMWrapper instances.
    Idempotent: if adapter is already a VLMWrapper, returns it as-is.

    Args:
        adapter: VLM adapter with predict() method returning VisionResult,
            or an existing VLMWrapper.

    Returns:
        VLMWrapper implementing the VisionLanguageModel protocol.
    """
    if isinstance(adapter, VLMWrapper):
        return adapter
    return VLMWrapper(adapter)
