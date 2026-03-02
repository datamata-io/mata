"""TorchScript model adapter for object detection.

Supports loading and inference with PyTorch TorchScript (.pt) models,
particularly RT-DETRv4 and similar architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torchvision.transforms as T  # noqa: N812
from PIL import Image

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Detection, DetectResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)


class TorchScriptDetectAdapter(PyTorchBaseAdapter):
    """TorchScript model detection adapter.

    Loads and runs inference with PyTorch TorchScript models (.pt files).
    Designed for RT-DETRv4 and similar DETR-based architectures that expect
    image tensor and original size as inputs.

    Examples:
        >>> # Load from local file
        >>> detector = TorchScriptDetectAdapter("models/rtv4_l.pt")
        >>> # With custom parameters
        >>> detector = TorchScriptDetectAdapter(
        ...     "models/rtv4_l.pt",
        ...     device="cuda",
        ...     threshold=0.5,
        ...     input_size=640
        ... )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        threshold: float = 0.3,
        input_size: int = 640,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize TorchScript detection adapter.

        Args:
            model_path: Path to TorchScript model (.pt file)
            device: Device ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            input_size: Model input size in pixels (default: 640 for square input)
            id2label: Optional custom label mapping (defaults to COCO-80)

        Raises:
            ImportError: If torch is not installed
            FileNotFoundError: If model file does not exist
            ModelLoadError: If model loading fails
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Validate model path
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        # Store TorchScript-specific parameters
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError(f"Input size must be positive, got {self.input_size}")

        # Use COCO labels if not provided
        if not self.id2label:
            self.id2label = self._get_coco_labels()

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load TorchScript model from file."""
        try:
            logger.info(f"Loading TorchScript model: {self.model_path}")
            logger.info(f"Device: {self.device}")

            # Load model - always to CPU first due to frozen TorchScript constants
            # TorchScript models have hardcoded device placement in frozen constants
            # which cannot be moved after tracing. Loading to CPU ensures compatibility.
            self.model = self.torch.jit.load(str(self.model_path), map_location="cpu")
            self.model.eval()

            # Override device to CPU for TorchScript models
            # This prevents device mismatch errors with frozen constants
            if self.device != self.torch.device("cpu"):
                logger.warning(
                    f"TorchScript models must run on CPU due to frozen constants. "
                    f"Overriding device from '{self.device}' to 'cpu'."
                )
                self.device = self.torch.device("cpu")

            logger.info("✓ TorchScript model loaded successfully")

        except Exception as e:
            raise ModelLoadError(str(self.model_path), f"Failed to load TorchScript model: {e}") from e

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "TorchScriptDetectAdapter",
            "task": "detect",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "threshold": self.threshold,
            "input_size": self.input_size,
            "backend": "torchscript",
        }

    def _preprocess_image(self, image: Image.Image) -> tuple[Any, Any]:
        """Preprocess image for TorchScript model input.

        Args:
            image: PIL Image in RGB format

        Returns:
            Tuple of (image_tensor, orig_size)
            - image_tensor: [1, 3, H, W] normalized tensor
            - orig_size: [1, 2] tensor with [height, width]
        """
        # Store original size (HW format)
        orig_width, orig_height = image.size
        orig_size = self.torch.tensor([[orig_height, orig_width]], dtype=self.torch.float32)

        # Resize image to model input size
        image_resized = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to tensor and normalize (ImageNet stats)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension [1, 3, H, W]

        return image_tensor, orig_size

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        threshold: float | None = None,
    ) -> DetectResult:
        """Run object detection on image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional confidence threshold override

        Returns:
            DetectResult containing list of detections

        Raises:
            InvalidInputError: If image is invalid
        """
        # Use instance threshold if not overridden
        threshold = threshold if threshold is not None else self.threshold

        # Load and preprocess image (capture original path if from file)
        pil_image, input_path = self._load_image(image)
        image_tensor, orig_size = self._preprocess_image(pil_image)

        # Move to device
        image_tensor = image_tensor.to(self.device)
        orig_size = orig_size.to(self.device)

        # Run inference
        with self.torch.no_grad():
            labels, boxes, scores = self.model(image_tensor, orig_size)

        # Build detections list
        detections: list[Detection] = []

        # Iterate through batch (assume batch_size=1)
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            score_val = float(score.item())

            # Filter by threshold
            if score_val >= threshold:
                label_id = int(label.item())
                bbox = [float(coord) for coord in box.tolist()]  # [x1, y1, x2, y2]

                # Get label name
                label_name = self.id2label.get(label_id, f"class_{label_id}")

                detections.append(
                    Detection(
                        bbox=tuple(bbox),  # type: ignore
                        score=score_val,
                        label=label_id,
                        label_name=label_name,
                    )
                )

        # Build result with metadata
        meta = {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "threshold": threshold,
            "input_size": self.input_size,
            "num_detections": len(detections),
            "input_path": input_path,
        }

        return DetectResult(detections=detections, meta=meta)
