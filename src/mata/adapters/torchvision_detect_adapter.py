"""Torchvision detection adapter for Apache 2.0 CNN models.

Supports multiple torchvision detection models with unified API:
- Faster R-CNN (ResNet-50 FPN, v1 and v2)
- RetinaNet (ResNet-50 FPN, v1 and v2)
- FCOS (ResNet-50 FPN)
- SSD (VGG16)
- SSDLite (MobileNetV3)

All models are Apache 2.0 licensed and use PyTorch's torchvision library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import ModelLoadError, UnsupportedModelError
from mata.core.logging import get_logger
from mata.core.types import Instance, VisionResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | np.ndarray

# Lazy imports for torchvision
_torchvision = None
_torchvision_transforms = None
TORCHVISION_AVAILABLE = None


def _ensure_torchvision():
    """Ensure torchvision is imported (lazy loading).

    Returns:
        Dictionary with torchvision modules

    Raises:
        ImportError: If torchvision is not installed
    """
    global _torchvision, _torchvision_transforms, TORCHVISION_AVAILABLE
    if _torchvision is None:
        try:
            import torchvision.models.detection as detection_models
            import torchvision.transforms as transforms

            _torchvision = detection_models
            _torchvision_transforms = transforms
            TORCHVISION_AVAILABLE = True
            logger.debug("Torchvision loaded successfully")
        except ImportError:
            TORCHVISION_AVAILABLE = False
            raise ImportError("torchvision is required for this adapter. " "Install with: pip install torchvision")
    return _torchvision, _torchvision_transforms


class TorchvisionDetectAdapter(PyTorchBaseAdapter):
    """Object detection adapter for torchvision CNN models.

    Automatically loads and runs inference with Apache 2.0-licensed CNN
    detection models from PyTorch's torchvision library. All models are
    pretrained on COCO dataset (80 classes).

    Supported Models:
    - **Faster R-CNN**: Two-stage detector, high accuracy (~42 mAP)
        - torchvision/fasterrcnn_resnet50_fpn (v1)
        - torchvision/fasterrcnn_resnet50_fpn_v2 (improved)
    - **RetinaNet**: Single-stage detector, balanced speed/accuracy (~40 mAP)
        - torchvision/retinanet_resnet50_fpn (v1)
        - torchvision/retinanet_resnet50_fpn_v2 (improved)
    - **FCOS**: Anchor-free single-stage detector (~40 mAP)
        - torchvision/fcos_resnet50_fpn
    - **SSD**: Very fast single-stage detector (~25 mAP, 60+ FPS)
        - torchvision/ssd300_vgg16
    - **SSDLite**: Mobile-optimized detector for edge devices
        - torchvision/ssdlite320_mobilenet_v3_large

    All models handle variable input sizes and return boxes in xyxy format
    (absolute pixel coordinates).

    Attributes:
        task: Always "detect"
        model_name: Torchvision model identifier
        weights: Pretrained weights specification ("DEFAULT" or checkpoint path)
        model: Loaded torchvision detection model
        transform: Preprocessing pipeline (ImageNet normalization)

    Examples:
        >>> # RetinaNet with default settings
        >>> detector = TorchvisionDetectAdapter("torchvision/retinanet_resnet50_fpn")
        >>> result = detector.predict("image.jpg", threshold=0.4)
        >>>
        >>> # Faster R-CNN V2 with custom threshold
        >>> detector = TorchvisionDetectAdapter(
        ...     "torchvision/fasterrcnn_resnet50_fpn_v2",
        ...     threshold=0.5,
        ...     device="cuda"
        ... )
        >>> result = detector.predict("image.jpg")
        >>>
        >>> # SSDLite for mobile/edge deployment
        >>> detector = TorchvisionDetectAdapter(
        ...     "torchvision/ssdlite320_mobilenet_v3_large",
        ...     device="cpu"
        ... )
        >>> result = detector.predict("image.jpg", threshold=0.35)
        >>>
        >>> # Custom fine-tuned model from checkpoint
        >>> detector = TorchvisionDetectAdapter(
        ...     "torchvision/retinanet_resnet50_fpn",
        ...     weights="/path/to/finetuned.pth"
        ... )
    """

    task = "detect"

    # Model builder mapping
    MODEL_BUILDERS = {
        "fasterrcnn_resnet50_fpn": "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
        "retinanet_resnet50_fpn": "retinanet_resnet50_fpn",
        "retinanet_resnet50_fpn_v2": "retinanet_resnet50_fpn_v2",
        "fcos_resnet50_fpn": "fcos_resnet50_fpn",
        "ssd300_vgg16": "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large": "ssdlite320_mobilenet_v3_large",
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
        weights: str | Any = "DEFAULT",
    ) -> None:
        """Initialize torchvision detection adapter.

        Args:
            model_name: Model identifier (e.g., "torchvision/retinanet_resnet50_fpn")
                Can include "torchvision/" prefix or just the model name
            device: Device specification ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0] (default: 0.3)
            id2label: Optional custom label mapping (uses COCO labels by default)
            weights: Pretrained weights specification
                - "DEFAULT": Use torchvision pretrained COCO weights
                - Path to .pth file: Load custom checkpoint

        Raises:
            ImportError: If torchvision is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model name is not recognized
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Lazy import torchvision
        self.detection_models, self.transforms = _ensure_torchvision()

        # Parse model name (strip "torchvision/" prefix if present)
        self.model_name = model_name
        parsed_name = model_name.replace("torchvision/", "")

        # Validate model name
        if parsed_name not in self.MODEL_BUILDERS:
            available = ", ".join(self.MODEL_BUILDERS.keys())
            raise UnsupportedModelError(
                f"Unknown torchvision model '{parsed_name}'. "
                f"Supported models: {available}. "
                f"Use format 'torchvision/<model_name>' or just '<model_name>'."
            )

        self.parsed_model_name = parsed_name
        self.weights = weights

        # Load model
        self._load_model()

        # Setup preprocessing pipeline
        self._setup_preprocessing()

        # Setup label mapping (use COCO labels if not provided)
        if not self.id2label:
            self.id2label = self._get_coco_labels()

    def _load_model(self) -> None:
        """Load torchvision detection model.

        Handles both pretrained weights and custom checkpoints with
        support for old and new torchvision API versions.

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading torchvision model: {self.parsed_model_name}")

            # Get model builder function
            builder_name = self.MODEL_BUILDERS[self.parsed_model_name]
            builder_fn = getattr(self.detection_models, builder_name)

            # Handle different torchvision API versions
            # New API (torchvision >= 0.13): weights parameter
            # Old API (torchvision < 0.13): pretrained parameter
            try:
                if self.weights == "DEFAULT":
                    # Try new API first (weights="DEFAULT")
                    try:
                        self.model = builder_fn(weights="DEFAULT")
                        logger.info("Loaded with new API (weights='DEFAULT')")
                    except TypeError:
                        # Fall back to old API (pretrained=True)
                        self.model = builder_fn(pretrained=True)
                        logger.info("Loaded with old API (pretrained=True)")
                else:
                    # Custom checkpoint path
                    checkpoint_path = Path(self.weights)
                    if not checkpoint_path.exists():
                        raise ModelLoadError(
                            self.model_name,
                            f"Checkpoint file not found: {self.weights}. "
                            f"Provide valid path or use weights='DEFAULT' for pretrained model.",
                        )

                    # Create model without pretrained weights
                    try:
                        self.model = builder_fn(weights=None)
                    except TypeError:
                        # Old API fallback
                        self.model = builder_fn(pretrained=False)

                    # Load checkpoint
                    logger.info(f"Loading checkpoint from: {self.weights}")
                    checkpoint = self.torch.load(
                        str(checkpoint_path),
                        map_location="cpu",
                        weights_only=True,  # Security: CVE-2025-32434 mitigation
                    )

                    # Handle different checkpoint formats
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint

                    self.model.load_state_dict(state_dict)
                    logger.info("Checkpoint loaded successfully")

            except Exception as e:
                raise ModelLoadError(
                    self.model_name,
                    f"Failed to load model weights: {type(e).__name__}: {str(e)}. "
                    f"Ensure torchvision version is compatible.",
                )

            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                self.model_name,
                f"Unexpected error loading torchvision model: {type(e).__name__}: {str(e)}",
            )

    def _setup_preprocessing(self) -> None:
        """Setup preprocessing pipeline.

        Torchvision models use ImageNet normalization:
        - mean=[0.485, 0.456, 0.406]
        - std=[0.229, 0.224, 0.225]

        No resizing is needed as torchvision models handle variable input sizes.
        """
        self.transform = self.transforms.Compose(
            [
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _preprocess(self, image: Image.Image) -> Any:
        """Preprocess PIL Image for torchvision model.

        Args:
            image: PIL Image in RGB format

        Returns:
            Preprocessed tensor with shape [3, H, W]
        """
        return self.transform(image)

    def predict(
        self,
        image: ImageInput,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Run object detection on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional threshold override [0.0, 1.0]
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            VisionResult with detection instances in xyxy format

        Raises:
            InvalidInputError: If image is invalid

        Examples:
            >>> detector = TorchvisionDetectAdapter("torchvision/retinanet_resnet50_fpn")
            >>> result = detector.predict("image.jpg", threshold=0.5)
            >>> for inst in result.instances:
            ...     print(f"{inst.label_name}: {inst.score:.2f} at {inst.bbox}")
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Preprocess image
        img_tensor = self._preprocess(pil_image)

        # Add batch dimension and move to device: [1, 3, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Inference (no gradient computation)
        with self.torch.no_grad():
            predictions = self.model(img_tensor)

        # Extract predictions from batch (batch size = 1)
        pred = predictions[0]

        # Convert tensors to numpy
        boxes = pred["boxes"].cpu().numpy()  # [N, 4] in xyxy format
        scores = pred["scores"].cpu().numpy()  # [N]
        labels = pred["labels"].cpu().numpy()  # [N]

        # Filter by threshold and create Instance objects
        instances: list[Instance] = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= conf_threshold:
                instance = Instance(
                    bbox=tuple(float(c) for c in box),  # xyxy format
                    score=float(score),
                    label=int(label),
                    label_name=self.id2label.get(int(label), f"class_{label}"),
                )
                instances.append(instance)

        # Create result with metadata
        return VisionResult(
            instances=instances,
            meta={
                "model_name": self.model_name,
                "threshold": conf_threshold,
                "device": str(self.device),
                "backend": "torchvision",
                "architecture": self.parsed_model_name,
                "input_path": input_path,
                "num_detections": len(instances),
            },
        )

    def info(self) -> dict[str, Any]:
        """Get adapter information and metadata.

        Returns:
            Dictionary with adapter metadata including:
                - name: Adapter class name
                - task: Task type ("detect")
                - model_id: Full model identifier
                - model_name: Parsed model name
                - device: Device string
                - threshold: Current confidence threshold
                - backend: Backend identifier ("torchvision")
                - weights: Weights specification

        Examples:
            >>> detector = TorchvisionDetectAdapter("torchvision/retinanet_resnet50_fpn")
            >>> info = detector.info()
            >>> print(info["model_name"])  # "retinanet_resnet50_fpn"
            >>> print(info["backend"])  # "torchvision"
        """
        return {
            "name": "TorchvisionDetectAdapter",
            "task": self.task,
            "model_id": self.model_name,
            "model_name": self.parsed_model_name,
            "device": str(self.device),
            "threshold": self.threshold,
            "backend": "torchvision",
            "weights": self.weights if isinstance(self.weights, str) else "<custom>",
            "num_classes": len(self.id2label),
        }
