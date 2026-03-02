"""TorchScript model adapter for image classification.

Supports loading and inference with PyTorch TorchScript (.pt) classification models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torchvision.transforms as T  # noqa: N812
from PIL import Image

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Classification, ClassifyResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)


class TorchScriptClassifyAdapter(PyTorchBaseAdapter):
    """TorchScript model classification adapter.

    Loads and runs inference with PyTorch TorchScript classification models (.pt files).
    Designed for ResNet, ViT, and other classification architectures exported to TorchScript.

    Examples:
        >>> # Load from local file
        >>> classifier = TorchScriptClassifyAdapter("models/resnet50.pt", top_k=5)
        >>> # With custom parameters
        >>> classifier = TorchScriptClassifyAdapter(
        ...     "models/vit.pt",
        ...     device="cuda",
        ...     top_k=10,
        ...     input_size=224
        ... )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        top_k: int = 5,
        threshold: float = 0.0,
        input_size: int = 224,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize TorchScript classification adapter.

        Args:
            model_path: Path to TorchScript model (.pt file)
            device: Device ("cuda", "cpu", or "auto")
            top_k: Number of top predictions to return (default: 5)
            threshold: Minimum confidence threshold for predictions (default: 0.0)
            input_size: Model input size in pixels (default: 224 for square input)
            id2label: Optional custom label mapping

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
        self.top_k = top_k
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError(f"Input size must be positive, got {self.input_size}")

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load TorchScript model from file."""
        try:
            logger.info(f"Loading TorchScript classification model: {self.model_path}")
            logger.info(f"Device: {self.device}")

            # Load model
            self.model = self.torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            self.model = self.model.to(self.device)

            logger.info(f"✓ TorchScript classification model loaded successfully on {self.device}")

        except Exception as e:
            raise ModelLoadError(
                str(self.model_path), f"Failed to load TorchScript classification model: {type(e).__name__}: {str(e)}"
            )

    def classify(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image (Classify node interface).

        This method provides compatibility with the Classify graph node,
        which expects a classify() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            top_k: Number of top predictions to return
            **kwargs: Additional arguments passed to predict()

        Returns:
            ClassifyResult with predictions
        """
        return self.predict(image=image, top_k=top_k, **kwargs)

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "TorchScriptClassifyAdapter",
            "task": "classify",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "top_k": self.top_k,
            "input_size": self.input_size,
            "num_classes": len(self.id2label) if self.id2label else "unknown",
            "backend": "torchscript",
        }

    def _preprocess_image(self, image: Image.Image) -> Any:
        """Preprocess image for TorchScript classification model.

        Args:
            image: PIL Image in RGB format

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize image to model input size
        image_resized = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to tensor and normalize (ImageNet stats)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_tensor = transform(image_resized)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, top_k: int | None = None, **kwargs: Any
    ) -> ClassifyResult:
        """Perform image classification on input image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Number of top predictions to return (overrides instance setting)
            **kwargs: Additional inference parameters

        Returns:
            ClassifyResult with top-k predictions sorted by confidence (descending)

        Raises:
            InvalidInputError: If image is invalid or cannot be loaded
            RuntimeError: If inference fails
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)

        # Determine top_k (runtime override or instance default)
        k = top_k if top_k is not None else self.top_k

        try:
            # Preprocess image
            input_tensor = self._preprocess_image(pil_image)
            input_tensor = input_tensor.to(self.device)

            # Run inference
            with self.torch.no_grad():
                outputs = self.model(input_tensor)

            # Get logits - handle both direct logits and dict outputs
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", list(outputs.values())[0]))
            else:
                logits = outputs

            logits = logits[0]  # Remove batch dimension, shape: (num_classes,)

            # Convert to probabilities using softmax
            probs = self.torch.nn.functional.softmax(logits, dim=-1)

            # Get top-k predictions
            top_k_probs, top_k_indices = self.torch.topk(probs, min(k, len(probs)))

            # Convert to Classification objects
            predictions = [
                Classification(
                    label=int(idx.item()),
                    score=float(prob.item()),
                    label_name=self.id2label.get(int(idx.item()), f"class_{int(idx.item())}"),
                )
                for prob, idx in zip(top_k_probs.cpu(), top_k_indices.cpu())
            ]

            # Build metadata
            meta = {
                "model_path": str(self.model_path),
                "device": str(self.device),
                "backend": "torchscript",
                "image_size": pil_image.size,
                "top_k": len(predictions),
                "input_path": input_path,
            }

            return ClassifyResult(predictions=predictions, meta=meta)

        except Exception as e:
            raise RuntimeError(f"TorchScript classification inference failed: {type(e).__name__}: {str(e)}")
