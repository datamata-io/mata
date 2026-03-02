"""Universal HuggingFace classification adapter with architecture auto-detection.

Supports automatic detection and loading of various transformer-based image
classification models from HuggingFace Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Classification, ClassifyResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

# Lazy imports for transformers-specific modules
_transformers = None
TRANSFORMERS_AVAILABLE = None


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading)."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import (
                AutoConfig,
                AutoImageProcessor,
                AutoModelForImageClassification,
            )

            _transformers = {
                "AutoImageProcessor": AutoImageProcessor,
                "AutoModelForImageClassification": AutoModelForImageClassification,
                "AutoConfig": AutoConfig,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


class HuggingFaceClassifyAdapter(PyTorchBaseAdapter):
    """Universal HuggingFace classification adapter with auto-detection.

    Automatically detects and loads the appropriate model architecture from
    HuggingFace Hub, including:
    - ResNet (all variants)
    - Vision Transformer (ViT)
    - ConvNeXt
    - EfficientNet
    - Swin Transformer
    - BEiT
    - DeiT

    Examples:
        >>> # ResNet-50
        >>> classifier = HuggingFaceClassifyAdapter("microsoft/resnet-50")
        >>> # Vision Transformer
        >>> classifier = HuggingFaceClassifyAdapter("google/vit-base-patch16-224")
        >>> # ConvNeXt
        >>> classifier = HuggingFaceClassifyAdapter("facebook/convnext-base-224")
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        top_k: int = 5,
        threshold: float = 0.0,
        id2label: dict[int, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HuggingFace classification adapter.

        Args:
            model_id: HuggingFace model ID (e.g., "microsoft/resnet-50")
            device: Device ("cuda", "cpu", or "auto")
            top_k: Number of top predictions to return (default: 5)
            threshold: Minimum confidence threshold for predictions (default: 0.0)
            id2label: Optional custom label mapping
            **kwargs: Additional parameters (e.g., template, use_softmax for CLIP)

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model architecture is not supported
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Store kwargs for CLIP delegation
        self._clip_kwargs = kwargs

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers is required for HuggingFace adapter. " "Install with: pip install transformers torch"
            )

        # Store transformers modules for later use
        self.transformers = transformers

        self.model_id = model_id
        self.top_k = top_k

        # Initialize CLIP delegation flag
        self._is_clip = False
        self._clip_delegate = None

        # Load model with architecture auto-detection
        self._load_model()

    def _load_model(self) -> None:
        """Load model with automatic architecture detection."""
        try:
            logger.info(f"Loading HuggingFace classification model: {self.model_id}")

            # Try to detect architecture from model_id or config
            architecture = self._detect_architecture()
            logger.info(f"Detected architecture: {architecture}")

            # Route CLIP models to specialized CLIP adapter
            if architecture == "clip":
                from .clip_adapter import HuggingFaceCLIPAdapter

                logger.info("Routing to CLIP zero-shot classification adapter")

                # Filter out kwargs that are already passed explicitly OR are inference-time params
                # Inference-time params (text_prompts, etc.) are passed to predict(), not __init__()
                clip_kwargs = {
                    k: v
                    for k, v in self._clip_kwargs.items()
                    if k not in {"model_id", "device", "top_k", "threshold", "text_prompts"}
                }

                # Create CLIP adapter instance with same parameters + kwargs
                self._clip_delegate = HuggingFaceCLIPAdapter(
                    model_id=self.model_id,
                    device=str(self.device),
                    top_k=self.top_k,
                    threshold=self.threshold,
                    **clip_kwargs,  # Pass through CLIP-specific params (template, use_softmax, etc.)
                )

                # Mark that we're delegating to CLIP
                self._is_clip = True
                return

            # Suppress noisy third-party output (progress bars, unexpected key warnings, etc.)
            from mata.core.logging import suppress_third_party_logs

            with suppress_third_party_logs():

                # Load processor and model using auto-detection
                self.processor = self.transformers["AutoImageProcessor"].from_pretrained(self.model_id)
                self.model = self.transformers["AutoModelForImageClassification"].from_pretrained(self.model_id)

            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Extract label mapping from model config if available
            if hasattr(self.model.config, "id2label") and not self.id2label:
                self.id2label = self.model.config.id2label
                logger.info(f"Loaded {len(self.id2label)} class labels from model config")
            elif self.id2label:
                logger.info(f"Using custom label mapping with {len(self.id2label)} classes")
            else:
                # Fallback: Use generic class names
                num_labels = self.model.config.num_labels
                self.id2label = {i: f"class_{i}" for i in range(num_labels)}
                logger.warning(
                    f"No label mapping found. Using generic labels (class_0, class_1, ...) " f"for {num_labels} classes"
                )

            logger.info(
                f"✓ Classification model loaded successfully on {self.device} " f"(architecture: {architecture})"
            )

        except Exception as e:
            raise ModelLoadError(
                self.model_id, f"Failed to load HuggingFace classification model: {type(e).__name__}: {str(e)}"
            )

    def _detect_architecture(self) -> str:
        """Detect model architecture from model ID or config.

        Returns:
            Architecture name ("clip", "resnet", "vit", "convnext", etc.)
        """
        model_id_lower = self.model_id.lower()

        # Check for CLIP models first (zero-shot classification)
        if "clip" in model_id_lower:
            return "clip"

        # Check model ID for known patterns
        if "resnet" in model_id_lower:
            return "resnet"
        elif "vit" in model_id_lower or "vision-transformer" in model_id_lower:
            return "vit"
        elif "convnext" in model_id_lower:
            return "convnext"
        elif "efficientnet" in model_id_lower:
            return "efficientnet"
        elif "swin" in model_id_lower:
            return "swin"
        elif "beit" in model_id_lower:
            return "beit"
        elif "deit" in model_id_lower:
            return "deit"

        # Try to get architecture from config
        try:
            config = self.transformers["AutoConfig"].from_pretrained(self.model_id)
            if hasattr(config, "model_type"):
                return config.model_type
        except Exception:
            pass

        # Default to generic auto-loading
        logger.warning(
            f"Could not detect specific architecture from '{self.model_id}'. " f"Using generic auto-loading."
        )
        return "auto"

    def classify(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int | None = None,
        text_prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image (Classify node interface).

        This method provides compatibility with the Classify graph node,
        which expects a classify() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            top_k: Number of top predictions to return
            text_prompts: Text prompts for zero-shot classification (CLIP only)
            **kwargs: Additional arguments passed to predict()

        Returns:
            ClassifyResult with predictions
        """
        return self.predict(
            image=image,
            top_k=top_k,
            text_prompts=text_prompts,
            **kwargs,
        )

    def info(self) -> dict[str, Any]:
        """Get adapter information and metadata.

        Returns:
            Dictionary with adapter metadata including model ID, task,
            device, architecture, top_k, and number of classes
        """
        # Delegate to CLIP adapter if CLIP model
        if self._is_clip:
            return self._clip_delegate.info()

        num_classes = len(self.id2label) if self.id2label else "unknown"

        return {
            "name": "HuggingFaceClassifyAdapter",
            "task": "classify",
            "model_id": self.model_id,
            "device": str(self.device),
            "architecture": self._detect_architecture(),
            "backend": "transformers",
            "top_k": self.top_k,
            "num_classes": num_classes,
        }

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, top_k: int | None = None, **kwargs: Any
    ) -> ClassifyResult:
        """Perform image classification on input image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Number of top predictions to return (overrides instance setting)
            **kwargs: Additional inference parameters (including text_prompts for CLIP)

        Returns:
            ClassifyResult with top-k predictions sorted by confidence (descending)

        Raises:
            InvalidInputError: If image is invalid or cannot be loaded
            RuntimeError: If inference fails
        """
        # Delegate to CLIP adapter if CLIP model
        if self._is_clip:
            return self._clip_delegate.predict(image, top_k=top_k, **kwargs)

        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)

        # Determine top_k (runtime override or instance default)
        k = top_k if top_k is not None else self.top_k

        try:
            # Preprocess image
            inputs = self.processor(images=pil_image, return_tensors="pt")

            # Move inputs to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Run inference
            with self.torch.no_grad():
                outputs = self.model(**inputs)

            # Get logits and convert to probabilities
            logits = outputs.logits[0]  # Shape: (num_classes,)
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
                "model_id": self.model_id,
                "device": str(self.device),
                "architecture": self._detect_architecture(),
                "image_size": pil_image.size,
                "top_k": len(predictions),
                "input_path": input_path,
            }

            return ClassifyResult(predictions=predictions, meta=meta)

        except Exception as e:
            logger.error(f"Classification inference failed: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Classification failed: {e}") from e
