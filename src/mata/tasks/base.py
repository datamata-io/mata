"""Task adapter protocols defining stable contracts.

These protocols define the interface that all adapters must implement.
They ensure model-agnostic task execution with consistent input/output types.
"""

from typing import Any, ClassVar, Protocol

from ..core.types import ClassifyResult, DepthResult, DetectResult, OCRResult, SegmentResult, TrackResult, VisionResult


class TaskAdapter(Protocol):
    """Base protocol for all task adapters.

    All adapters must implement:
    - name: str - Unique adapter name
    - task: str - Task type ("detect", "segment", etc.)
    - info() -> Dict - Return adapter metadata
    """

    name: str
    task: str

    def info(self) -> dict[str, Any]:
        """Return adapter information and metadata.

        Returns:
            Dictionary with adapter details (name, task, model info, etc.)
        """
        ...


class DetectAdapter(TaskAdapter, Protocol):
    """Object detection adapter protocol.

    Adapters implementing this protocol must:
    - Accept single images (PIL, path, numpy)
    - Return DetectResult with bounding boxes in xyxy format
    - Handle preprocessing and postprocessing internally
    """

    task: str  # Must be "detect"

    def predict(self, image: Any, **kwargs: Any) -> DetectResult:
        """Perform object detection on an image.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            **kwargs: Additional inference parameters (threshold, etc.)

        Returns:
            DetectResult with detected objects
        """
        ...


class SegmentAdapter(TaskAdapter, Protocol):
    """Instance segmentation adapter protocol.

    Adapters implementing this protocol must:
    - Accept single images
    - Return SegmentResult with instance masks
    - Handle preprocessing and postprocessing internally
    """

    task: str  # Must be "segment"

    def predict(self, image: Any, **kwargs: Any) -> SegmentResult:
        """Perform instance segmentation on an image.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            **kwargs: Additional inference parameters

        Returns:
            SegmentResult with segmented instances
        """
        ...


class ClassifyAdapter(TaskAdapter, Protocol):
    """Image classification adapter protocol.

    Adapters implementing this protocol must:
    - Accept single images
    - Return ClassifyResult with top-k predictions
    - Handle preprocessing internally
    """

    task: str  # Must be "classify"

    def predict(self, image: Any, **kwargs: Any) -> ClassifyResult:
        """Perform image classification.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            **kwargs: Additional inference parameters (topk, etc.)

        Returns:
            ClassifyResult with classification predictions
        """
        ...


class DepthAdapter(TaskAdapter, Protocol):
    """Depth estimation adapter protocol.

    Adapters implementing this protocol must:
    - Accept single images
    - Return DepthResult with raw depth map
    - Handle preprocessing and postprocessing internally
    """

    task: str  # Must be "depth"

    def predict(self, image: Any, **kwargs: Any) -> DepthResult:
        """Perform depth estimation.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            **kwargs: Additional inference parameters

        Returns:
            DepthResult with depth map
        """
        ...


class OCRAdapter(TaskAdapter, Protocol):
    """Task adapter protocol for OCR text extraction.

    Adapters implementing this protocol must:
    - Accept single images (PIL, path, numpy)
    - Return OCRResult with text regions and bounding boxes
    - Handle preprocessing and postprocessing internally
    """

    task: ClassVar[str]  # Must be "ocr"
    name: ClassVar[str]

    def predict(self, image: Any, **kwargs: Any) -> OCRResult:
        """Extract text from image.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            **kwargs: Additional inference parameters (lang, conf_threshold, etc.)

        Returns:
            OCRResult with detected text regions
        """
        ...

    def info(self) -> dict[str, Any]:
        """Return adapter metadata.

        Returns:
            Dictionary with adapter details (name, task, engine, etc.)
        """
        ...


class VLMAdapter(TaskAdapter, Protocol):
    """Vision-language model adapter protocol.

    Adapters implementing this protocol must:
    - Accept single images and text prompts
    - Return VisionResult with generated text response
    - Handle preprocessing and postprocessing internally
    - Support optional output_mode for structured output parsing
    - Support optional images kwarg for multi-image input
    """

    task: str  # Must be "vlm"

    def predict(self, image: Any, prompt: str, **kwargs: Any) -> VisionResult:
        """Perform vision-language inference.

        Args:
            image: Input image (PIL.Image, str path, or numpy.ndarray)
            prompt: Required text prompt/question for the model
            **kwargs: Additional inference parameters (system_prompt, max_new_tokens, etc.)

        Returns:
            VisionResult with generated text in the text field
        """
        ...


class TrackAdapter(TaskAdapter, Protocol):
    """Object tracking adapter protocol.

    Tracking is stateful - it consumes DetectResult from detection
    and maintains track IDs across frames.

    Adapters implementing this protocol must:
    - Maintain internal state across update() calls
    - Associate detections with existing tracks
    - Create new tracks for new objects
    """

    task: str  # Must be "track"

    def update(self, detections: DetectResult, **kwargs: Any) -> TrackResult:
        """Update tracker with new detections.

        Args:
            detections: Detection results from current frame
            **kwargs: Additional tracking parameters

        Returns:
            TrackResult with updated tracks
        """
        ...
