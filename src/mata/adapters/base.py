"""Base adapter classes for MATA framework.

This module provides abstract base classes that eliminate code duplication
across model-type-specific adapters. All adapters should inherit from these
base classes rather than implementing common functionality independently.

Architecture:
    BaseAdapter (abstract)
        ├── PyTorchBaseAdapter (for HuggingFace, PyTorch, TorchScript)
        └── ONNXBaseAdapter (for ONNX Runtime)

Usage:
    class MyDetectAdapter(PyTorchBaseAdapter):
        def __init__(self, model_path, **kwargs):
            super().__init__(device=kwargs.get('device', 'auto'),
                        threshold=kwargs.get('threshold', 0.3))
            # Adapter-specific initialization

        def predict(self, image, **kwargs):
            pil_image = self._load_image(image)  # Use base class method
            # Adapter-specific inference
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError
from mata.core.logging import get_logger

logger = get_logger(__name__)


class BaseAdapter(ABC):
    """Base class for all MATA adapters.

    Provides common functionality shared across all model types:
    - Image loading and validation
    - COCO label mapping
    - Threshold validation
    - Info metadata structure

    All concrete adapters must inherit from this class or one of its
    subclasses (PyTorchBaseAdapter, ONNXBaseAdapter).

    Attributes:
        threshold: Detection/segmentation confidence threshold [0.0, 1.0]
        id2label: Optional custom label mapping (int -> str)
    """

    def __init__(
        self,
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
    ):
        """Initialize base adapter.

        Args:
            threshold: Confidence threshold [0.0, 1.0] (default: 0.3)
            id2label: Optional custom label mapping dict

        Raises:
            ValueError: If threshold out of valid range
        """
        self.threshold = self._validate_threshold(threshold)
        self.id2label = id2label or {}

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        """Validate and normalize threshold value.

        Args:
            threshold: Raw threshold value (will be cast to float)

        Returns:
            Validated threshold as float

        Raises:
            ValueError: If threshold not in [0.0, 1.0] range
        """
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Threshold must be numeric, got {type(threshold).__name__}: {threshold}") from e

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in range [0.0, 1.0], got {threshold}")

        return threshold

    def _load_image(self, image: str | Path | Image.Image | np.ndarray | Any) -> tuple[Image.Image, str | None]:
        """Load and validate image input.

        Supports multiple input formats:
        - HTTP/HTTPS URL (str) - downloaded and converted to RGB
        - File path (str or Path) - loaded and converted to RGB
        - PIL Image - converted to RGB
        - Numpy array - converted to PIL Image and RGB
        - MATA Image artifact - extracts PIL image via to_pil()

        Args:
            image: Image source in supported format

        Returns:
            Tuple of (PIL Image in RGB format, original path/URL if from file/URL)

        Raises:
            InvalidInputError: If image type unsupported or loading fails
        """
        input_path: str | None = None

        try:
            # Handle MATA Image artifact (check by class name to avoid circular imports)
            if type(image).__name__ == "Image" and hasattr(image, "to_pil"):
                pil_image = image.to_pil().convert("RGB")
                input_path = getattr(image, "source_path", None)
                logger.debug("Extracted PIL Image from MATA Image artifact")
                return pil_image, input_path
            # Handle HTTP/HTTPS URLs
            elif isinstance(image, str) and image.startswith(("http://", "https://")):
                import urllib.request
                from io import BytesIO

                try:
                    with urllib.request.urlopen(image, timeout=30) as response:
                        image_data = response.read()
                    pil_image = Image.open(BytesIO(image_data)).convert("RGB")
                    input_path = image  # Store URL as path
                    logger.debug(f"Loaded image from URL: {image}")
                except Exception as e:
                    raise InvalidInputError(f"Failed to load image from URL: {e}") from e
                return pil_image, input_path
            elif isinstance(image, (str, Path)):
                input_path = str(image)
                pil_image = Image.open(image).convert("RGB")
                logger.debug(f"Loaded image from path: {image}")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
                logger.debug(f"Converted PIL Image to RGB (mode was {image.mode})")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert("RGB")
                logger.debug(f"Converted numpy array to PIL Image (shape: {image.shape})")
            else:
                raise InvalidInputError(
                    f"Unsupported image type: {type(image).__name__}. "
                    f"Expected: str, Path, PIL.Image, np.ndarray, or MATA Image artifact"
                )
            return pil_image, input_path
        except InvalidInputError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Failed to load image: {e}") from e

    def _load_images(self, images: list[str | Path | Image.Image | np.ndarray]) -> list[tuple[Image.Image, str | None]]:
        """Load and validate multiple image inputs.

        Calls _load_image() for each input and returns a list of results.
        Stops on first failure with a clear error message including the index.

        Args:
            images: List of image sources in supported formats

        Returns:
            List of (PIL Image in RGB format, original path if from file) tuples

        Raises:
            InvalidInputError: If any image type is unsupported or loading fails
        """
        if not images:
            raise InvalidInputError("images list cannot be empty")

        results = []
        for i, img in enumerate(images):
            try:
                results.append(self._load_image(img))
            except InvalidInputError as e:
                raise InvalidInputError(f"Failed to load image at index {i}: {e}") from e
        return results

    @staticmethod
    def _get_coco_labels() -> dict[int, str]:
        """Get default COCO-80 label mapping.

        Returns standard COCO dataset class names indexed 0-79.

        Returns:
            Dictionary mapping class IDs (0-79) to class names
        """
        coco_classes = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        return {i: name for i, name in enumerate(coco_classes)}

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Get adapter information and metadata.

        Must be implemented by subclasses. Should return a dictionary
        containing at minimum:
        - name: str - Adapter identifier
        - task: str - Task type (detect, segment, classify, etc.)
        - model_path or model_id: str - Model source
        - device: str - Device information
        - threshold: float - Confidence threshold
        - backend: str - Backend name (transformers, onnxruntime, etc.)

        Returns:
            Dictionary with adapter metadata
        """
        ...

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Any:
        """Run prediction on input image.

        Must be implemented by subclasses with task-specific logic.

        Args:
            image: Input image (str path, Path, PIL.Image, or np.ndarray)
            **kwargs: Additional task-specific parameters

        Returns:
            Task-specific result object (DetectResult, SegmentResult, etc.)
        """
        ...
