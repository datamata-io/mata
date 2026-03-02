"""Universal HuggingFace adapter with architecture auto-detection.

Supports automatic detection and loading of various transformer-based object
detection models from HuggingFace Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Instance, VisionResult

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
                AutoModelForObjectDetection,
                RTDetrForObjectDetection,
                RTDetrImageProcessor,
                RTDetrV2ForObjectDetection,
            )

            _transformers = {
                "AutoImageProcessor": AutoImageProcessor,
                "AutoModelForObjectDetection": AutoModelForObjectDetection,
                "AutoConfig": AutoConfig,
                "RTDetrImageProcessor": RTDetrImageProcessor,
                "RTDetrForObjectDetection": RTDetrForObjectDetection,
                "RTDetrV2ForObjectDetection": RTDetrV2ForObjectDetection,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


class HuggingFaceDetectAdapter(PyTorchBaseAdapter):
    """Universal HuggingFace detection adapter with auto-detection.

    Automatically detects and loads the appropriate model architecture from
    HuggingFace Hub, including:
    - RT-DETR / RT-DETRv2
    - DINO
    - Conditional DETR
    - Standard DETR
    - YOLOS

    Examples:
        >>> # Auto-detect RT-DETR
        >>> detector = HuggingFaceDetectAdapter("facebook/detr-resnet-50")
        >>> # Auto-detect DINO
        >>> detector = HuggingFaceDetectAdapter("IDEA-Research/dino-resnet-50")
        >>> # Auto-detect Conditional DETR
        >>> detector = HuggingFaceDetectAdapter("microsoft/conditional-detr-resnet-50")
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize HuggingFace detection adapter.

        Args:
            model_id: HuggingFace model ID (e.g., "facebook/detr-resnet-50")
            device: Device ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model architecture is not supported
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers is required for HuggingFace adapter. " "Install with: pip install transformers torch"
            )

        # Store transformers modules for later use
        self.transformers = transformers

        self.model_id = model_id

        # Load model with architecture auto-detection
        self._load_model()

    def _load_model(self) -> None:
        """Load model with automatic architecture detection."""
        try:
            logger.info(f"Loading HuggingFace model: {self.model_id}")

            # Try to detect architecture from model_id or config
            architecture = self._detect_architecture()
            logger.info(f"Detected architecture: {architecture}")

            # Suppress noisy third-party output (progress bars, unexpected key warnings, etc.)
            from mata.core.logging import suppress_third_party_logs

            with suppress_third_party_logs():

                # Load appropriate processor and model
                if architecture == "rtdetr":
                    # Detect RT-DETR version (v1 or v2) from model config
                    rtdetr_version = self._detect_rtdetr_version()
                    logger.info(f"Detected RT-DETR version: {rtdetr_version}")

                    try:
                        self.processor = self.transformers["RTDetrImageProcessor"].from_pretrained(self.model_id)
                        if rtdetr_version == "v2":
                            self.model = self.transformers["RTDetrV2ForObjectDetection"].from_pretrained(self.model_id)
                        else:  # v1
                            self.model = self.transformers["RTDetrForObjectDetection"].from_pretrained(self.model_id)
                    except (KeyError, ValueError) as e:
                        # Fallback for community models with broken configs
                        error_msg = str(e)
                        logger.warning(
                            f"⚠️  Failed to load with specific RT-DETR class: {type(e).__name__}: {error_msg}. "
                            f"Attempting generic AutoModel fallback..."
                        )
                        try:
                            self.processor = self.transformers["AutoImageProcessor"].from_pretrained(self.model_id)
                            self.model = self.transformers["AutoModelForObjectDetection"].from_pretrained(self.model_id)
                        except Exception:
                            # Community model has fundamentally broken config
                            raise ModelLoadError(
                                self.model_id,
                                f"Community model has incompatible config. "
                                f"Config error: {error_msg}. "
                                f"This model may be incompatible with current transformers version. "
                                f"Try using official models from: facebook/, PekingU/, microsoft/. "
                                f"Original error: {type(e).__name__}: {str(e)}",
                            )
                else:
                    # Generic auto-loading for DETR-based models
                    self.processor = self.transformers["AutoImageProcessor"].from_pretrained(self.model_id)
                    self.model = self.transformers["AutoModelForObjectDetection"].from_pretrained(self.model_id)

            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()

            # Extract label mapping
            if not self.id2label and hasattr(self.model.config, "id2label"):
                self.id2label = self.model.config.id2label

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            raise ModelLoadError(self.model_id, f"Failed to load HuggingFace model: {type(e).__name__}: {str(e)}")

    def _detect_architecture(self) -> str:
        """Detect model architecture from model ID or config.

        Returns:
            Architecture name ("rtdetr", "dino", "detr", etc.)
        """
        model_id_lower = self.model_id.lower()

        # Check model ID for known patterns
        if "rtdetr" in model_id_lower or "rt-detr" in model_id_lower:
            return "rtdetr"
        elif "dino" in model_id_lower:
            return "dino"
        elif "conditional-detr" in model_id_lower or "conditional_detr" in model_id_lower:
            return "conditional_detr"
        elif "yolos" in model_id_lower:
            return "yolos"
        elif "detr" in model_id_lower:
            return "detr"

        # Default to generic DETR-based auto-loading
        logger.warning(
            f"Could not detect specific architecture from '{self.model_id}'. " f"Using generic auto-loading."
        )
        return "auto"

    def _detect_rtdetr_version(self) -> str:
        """Detect RT-DETR version (v1 or v2) from model config.

        Returns:
            "v1" or "v2"
        """
        # Check if this is a community model (not from official sources)
        official_orgs = ["facebook", "PekingU", "IDEA-Research", "microsoft"]
        is_community_model = not any(self.model_id.startswith(f"{org}/") for org in official_orgs)

        if is_community_model:
            logger.warning(
                f"⚠️  Using community model '{self.model_id}' (not from official organization). "
                f"Config variations may occur - using pattern matching for version detection."
            )

        try:
            # Load config to check model_type
            config = self.transformers["AutoConfig"].from_pretrained(self.model_id)

            # Check model_type in config using pattern matching (handles variations)
            if hasattr(config, "model_type"):
                model_type = str(config.model_type).lower()

                # Pattern matching for v2 (handles: rt_detr_v2, rt_detr_v2_resnet, rtdetr_v2, etc.)
                if "v2" in model_type or "rtdetr_v2" in model_type or "rt_detr_v2" in model_type:
                    if is_community_model and model_type not in ["rt_detr_v2", "rtdetr_v2"]:
                        logger.info(
                            f"📋 Non-standard model_type '{config.model_type}' detected. "
                            f"Using pattern matching → identified as RT-DETR v2."
                        )
                    return "v2"

                # Pattern matching for v1 (handles: rt_detr, rtdetr, rt_detr_resnet, etc.)
                elif "rt_detr" in model_type or "rtdetr" in model_type:
                    if is_community_model and model_type not in ["rt_detr", "rtdetr"]:
                        logger.info(
                            f"📋 Non-standard model_type '{config.model_type}' detected. "
                            f"Using pattern matching → identified as RT-DETR v1."
                        )
                    return "v1"

            # Fallback: check model_id for version indicators
            logger.warning("model_type not found in config. Falling back to model ID analysis.")
            model_id_lower = self.model_id.lower()
            if "v2" in model_id_lower or "rtdetrv2" in model_id_lower:
                return "v2"
            elif "v1" in model_id_lower or any(x in model_id_lower for x in ["r18vd", "r34vd", "r50vd", "r101vd"]):
                # v2 models typically use vd (visdrone) backbones
                return "v2"

            # Default to v1 for backward compatibility
            logger.warning(
                f"Could not determine RT-DETR version from config for '{self.model_id}'. " f"Defaulting to v1."
            )
            return "v1"

        except Exception as e:
            logger.warning(
                f"Error accessing config for version detection: {type(e).__name__}: {str(e)}. "
                f"Falling back to model ID analysis."
            )

            # Fallback to model_id analysis on error
            model_id_lower = self.model_id.lower()
            if "v2" in model_id_lower or "rtdetrv2" in model_id_lower:
                logger.info("Detected v2 from model ID pattern.")
                return "v2"
            elif "v1" in model_id_lower or any(x in model_id_lower for x in ["r18vd", "r34vd", "r50vd", "r101vd"]):
                logger.info("Detected v2 from model ID pattern (vd backbone).")
                return "v2"

            logger.warning("Defaulting to v1.")
            return "v1"

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "HuggingFaceDetectAdapter",
            "task": "detect",
            "model_id": self.model_id,
            "device": str(self.device),
            "threshold": self.threshold,
            "num_classes": len(self.id2label) if self.id2label else "unknown",
            "backend": "huggingface-transformers",
        }

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, threshold: float | None = None, **kwargs: Any
    ) -> VisionResult:
        """Run object detection on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional threshold override
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            VisionResult with detection instances

        Raises:
            InvalidInputError: If image is invalid
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference (no gradient computation)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess
        target_sizes = self.torch.tensor([[pil_image.height, pil_image.width]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        # Convert to Instance objects
        instances: list[Instance] = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(), results["labels"].cpu().numpy(), results["boxes"].cpu().numpy()
        ):
            # box is in xyxy format already
            x1, y1, x2, y2 = box

            instance = Instance(
                bbox=tuple(float(c) for c in [x1, y1, x2, y2]),
                score=float(score),
                label=int(label),
                label_name=self.id2label.get(int(label), f"class_{label}"),
            )
            instances.append(instance)

        # Create result
        return VisionResult(
            instances=instances,
            meta={
                "model_id": self.model_id,
                "threshold": conf_threshold,
                "device": str(self.device),
                "backend": "huggingface-transformers",
                "input_path": input_path,
            },
        )
