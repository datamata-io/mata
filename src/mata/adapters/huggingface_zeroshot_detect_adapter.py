"""HuggingFace zero-shot object detection adapter.

Supports text-prompt-based object detection using open-vocabulary models:
- GroundingDINO (IDEA-Research)
- OWL-ViT v1 (Google)
- OWL-ViT v2 (Google)

Enables detection of arbitrary objects specified via text prompts without
training on predefined class vocabularies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Instance, VisionResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | np.ndarray

# Lazy imports for transformers-specific modules
_transformers = None
TRANSFORMERS_AVAILABLE = None


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading)."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Owlv2ForObjectDetection,
                Owlv2Processor,
                OwlViTForObjectDetection,
                OwlViTProcessor,
            )

            _transformers = {
                "AutoProcessor": AutoProcessor,
                "AutoModelForZeroShotObjectDetection": AutoModelForZeroShotObjectDetection,
                "OwlViTProcessor": OwlViTProcessor,
                "OwlViTForObjectDetection": OwlViTForObjectDetection,
                "Owlv2Processor": Owlv2Processor,
                "Owlv2ForObjectDetection": Owlv2ForObjectDetection,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


class HuggingFaceZeroShotDetectAdapter(PyTorchBaseAdapter):
    """Zero-shot object detection with text prompts.

    Supports open-vocabulary detection using text descriptions of objects
    to detect, without requiring training on predefined classes. Models
    supported:

    - **GroundingDINO**: State-of-the-art zero-shot detection
        - Model IDs: "IDEA-Research/grounding-dino-*"
        - Prompt format: Space-dot separated: "cat . dog . person"
        - Best performance but slower inference

    - **OWL-ViT v1**: Fast zero-shot detection
        - Model IDs: "google/owlvit-*"
        - Prompt format: List or space-dot: ["cat", "dog"] or "cat . dog"
        - Faster inference, good accuracy

    - **OWL-ViT v2**: Improved OWL-ViT
        - Model IDs: "google/owlv2-*"
        - Prompt format: List or space-dot: ["cat", "dog"] or "cat . dog"
        - Best speed/accuracy tradeoff

    Attributes:
        task: Always "detect"
        model_id: HuggingFace model identifier
        architecture: Detected architecture (grounding_dino, owlvit_v1, owlv2)
        processor: HuggingFace processor for preprocessing
        model: HuggingFace model for inference

    Examples:
        >>> # GroundingDINO - space-dot format
        >>> detector = HuggingFaceZeroShotDetectAdapter(
        ...     "IDEA-Research/grounding-dino-tiny"
        >>> )
        >>> result = detector.predict("image.jpg", text_prompts="cat . dog . person")
        >>>
        >>> # OWL-ViT v2 - list format
        >>> detector = HuggingFaceZeroShotDetectAdapter("google/owlv2-base-patch16")
        >>> result = detector.predict("image.jpg", text_prompts=["cat", "dog"])
        >>>
        >>> # Batch processing
        >>> results = detector.predict(
        ...     ["img1.jpg", "img2.jpg"],
        ...     text_prompts="car . truck"
        ... )
    """

    task = "detect"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.3,
        max_batch_size: int = 8,
    ) -> None:
        """Initialize zero-shot detection adapter.

        Args:
            model_id: HuggingFace model ID (e.g., "IDEA-Research/grounding-dino-tiny")
            device: Device ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            max_batch_size: Maximum batch size for processing (default: 8)

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model architecture is not supported
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold)

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers library is required for zero-shot detection. " "Install with: pip install transformers"
            )

        self.model_id = model_id
        self.max_batch_size = max_batch_size

        # Detect architecture
        self.architecture = self._detect_architecture()
        logger.info(f"Detected zero-shot architecture: {self.architecture}")

        # Load model and processor
        self._load_model()

    def _detect_architecture(self) -> str:
        """Detect zero-shot model architecture from model ID.

        Returns:
            Architecture name: grounding_dino, owlvit_v1, owlvit_v2, or auto
        """
        model_id_lower = self.model_id.lower()

        # GroundingDINO variants
        if "grounding" in model_id_lower or "groundingdino" in model_id_lower:
            return "grounding_dino"

        # OWL-ViT v2 (check before v1)
        elif "owlv2" in model_id_lower or "owl-v2" in model_id_lower:
            return "owlvit_v2"

        # OWL-ViT v1
        elif "owlvit" in model_id_lower or "owl-vit" in model_id_lower:
            return "owlvit_v1"

        # Unknown - try generic auto-loading
        else:
            logger.warning(
                f"Unknown zero-shot architecture for '{self.model_id}'. " f"Attempting generic auto-loading."
            )
            return "auto"

    def _load_model(self) -> None:
        """Load zero-shot detection model and processor.

        Raises:
            ModelLoadError: If model loading fails
        """
        from mata.core.logging import suppress_third_party_logs

        try:
            transformers = _ensure_transformers()

            with suppress_third_party_logs():
                if self.architecture == "grounding_dino":
                    # GroundingDINO uses AutoProcessor and AutoModel
                    logger.info(f"Loading GroundingDINO model: {self.model_id}")
                    self.processor = transformers["AutoProcessor"].from_pretrained(self.model_id)
                    self.model = transformers["AutoModelForZeroShotObjectDetection"].from_pretrained(self.model_id)

                elif self.architecture == "owlvit_v1":
                    # OWL-ViT v1 uses specific classes
                    logger.info(f"Loading OWL-ViT v1 model: {self.model_id}")
                    self.processor = transformers["OwlViTProcessor"].from_pretrained(self.model_id)
                    self.model = transformers["OwlViTForObjectDetection"].from_pretrained(self.model_id)

                elif self.architecture == "owlvit_v2":
                    # OWL-ViT v2 uses specific classes
                    logger.info(f"Loading OWL-ViT v2 model: {self.model_id}")
                    self.processor = transformers["Owlv2Processor"].from_pretrained(self.model_id)
                    self.model = transformers["Owlv2ForObjectDetection"].from_pretrained(self.model_id)

                else:
                    # Generic auto-loading fallback
                    logger.info(f"Loading zero-shot model with auto-detection: {self.model_id}")
                    self.processor = transformers["AutoProcessor"].from_pretrained(self.model_id)
                    self.model = transformers["AutoModelForZeroShotObjectDetection"].from_pretrained(self.model_id)

            # Move model to device and set to eval mode
            self.model.to(self.device).eval()
            logger.info(f"Successfully loaded {self.architecture} on {self.device}")

        except Exception as e:
            raise ModelLoadError(
                self.model_id, f"Failed to load zero-shot detection model: {type(e).__name__}: {str(e)}"
            ) from e

    def _normalize_text_prompts(self, text_prompts: str | list[str]) -> str | list[str]:
        """Normalize text prompts to model-specific format.

        Args:
            text_prompts: Single string or list of strings

        Returns:
            Normalized prompts in model-expected format
        """
        # GroundingDINO expects space-dot-separated string
        if self.architecture == "grounding_dino":
            if isinstance(text_prompts, list):
                # Convert list to space-dot format
                return " . ".join(text_prompts)
            else:
                # Already string - ensure space-dot format
                return text_prompts

        # OWL-ViT models accept both list and space-dot string
        # But list format is preferred for clarity
        elif self.architecture in ("owlvit_v1", "owlvit_v2"):
            if isinstance(text_prompts, str):
                # Convert space-dot format to list
                if " . " in text_prompts:
                    return [p.strip() for p in text_prompts.split(".")]
                else:
                    # Single prompt
                    return [text_prompts]
            else:
                # Already list
                return text_prompts

        # Auto mode - keep as-is
        return text_prompts

    def _extract_label_names(self, text_prompts: str | list[str]) -> list[str]:
        """Extract label names from text prompts.

        Args:
            text_prompts: Text prompts in string or list format

        Returns:
            List of label names
        """
        if isinstance(text_prompts, list):
            return text_prompts
        else:
            # Split space-dot format
            return [p.strip() for p in text_prompts.split(".") if p.strip()]

    def predict(
        self,
        image: ImageInput | list[ImageInput],
        text_prompts: str | list[str],
        threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult | list[VisionResult]:
        """Run zero-shot detection with text prompts.

        Supports both single image and batch processing.

        Args:
            image: Input image(s)
                - Single: str path, Path, PIL.Image, or numpy array
                - Batch: List of any of the above
            text_prompts: Text description(s) of objects to detect
                - GroundingDINO: "cat . dog . person" (space-dot separated)
                - OWL-ViT: ["cat", "dog", "person"] (list) or space-dot string
            threshold: Detection confidence threshold override (optional)
            **kwargs: Additional inference parameters

        Returns:
            VisionResult for single image, or List[VisionResult] for batch

        Raises:
            InvalidInputError: If text_prompts is missing or empty

        Examples:
            >>> # Single image
            >>> result = detector.predict("image.jpg", text_prompts="cat . dog")
            >>>
            >>> # Batch images
            >>> results = detector.predict(
            ...     ["img1.jpg", "img2.jpg"],
            ...     text_prompts=["cat", "dog"]
            ... )
        """
        # Validate text prompts
        if not text_prompts:
            raise InvalidInputError(
                "text_prompts required for zero-shot detection. "
                "Example: text_prompts='cat . dog' or text_prompts=['cat', 'dog']"
            )

        # Determine if batch processing
        is_batch = isinstance(image, list)

        if is_batch:
            # Batch processing - process sequentially for v1.5
            logger.info(f"Processing batch of {len(image)} images")
            return self._predict_batch(image, text_prompts, threshold, **kwargs)
        else:
            # Single image processing
            return self._predict_single(image, text_prompts, threshold, **kwargs)

    def _predict_single(
        self, image: ImageInput, text_prompts: str | list[str], threshold: float | None = None, **kwargs: Any
    ) -> VisionResult:
        """Run zero-shot detection on single image.

        Args:
            image: Single input image
            text_prompts: Text prompts for detection
            threshold: Detection confidence threshold override
            **kwargs: Additional parameters

        Returns:
            VisionResult with detected instances
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)
        conf_threshold = threshold if threshold is not None else self.threshold

        # Normalize prompts to model-specific format
        normalized_prompts = self._normalize_text_prompts(text_prompts)
        label_names = self._extract_label_names(text_prompts)

        # Preprocess
        inputs = self.processor(images=pil_image, text=normalized_prompts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference (no gradients)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess with model-specific method
        target_sizes = self.torch.tensor([[pil_image.height, pil_image.width]])
        target_sizes = target_sizes.to(self.device)

        # Use appropriate postprocessing method
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            # GroundingDINO postprocessing
            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]
        elif hasattr(self.processor, "post_process_object_detection"):
            # OWL-ViT postprocessing
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]
        else:
            # Fallback - extract from outputs manually
            logger.warning("Using fallback postprocessing - results may be suboptimal")
            results = self._fallback_postprocess(outputs, pil_image, conf_threshold)

        # Convert to Instance objects
        instances: list[Instance] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Handle different label formats:
            # - GroundingDINO: labels are text strings
            # - OWL-ViT: labels are numeric indices/tensors
            if isinstance(label, str):
                # GroundingDINO - label is already text
                label_name = label
                # Find index in label_names if possible, otherwise use -1
                try:
                    label_idx = label_names.index(label)
                except (ValueError, AttributeError):
                    label_idx = -1
            else:
                # OWL-ViT - label is numeric index (may be tensor)
                label_idx = int(label)
                if label_idx < len(label_names):
                    label_name = label_names[label_idx]
                else:
                    label_name = f"object_{label_idx}"

            instances.append(
                Instance(
                    bbox=tuple(float(x) for x in box),  # xyxy format
                    score=float(score),
                    label=label_idx,
                    label_name=label_name,
                )
            )

        return VisionResult(
            instances=instances,
            meta={
                "model_id": self.model_id,
                "architecture": self.architecture,
                "text_prompts": text_prompts,
                "threshold": conf_threshold,
                "mode": "zeroshot",
                "image_size": (pil_image.width, pil_image.height),
                "input_path": input_path,
            },
        )

    def _predict_batch(
        self,
        images: list[ImageInput],
        text_prompts: str | list[str],
        threshold: float | None = None,
        **kwargs: Any,
    ) -> list[VisionResult]:
        """Run zero-shot detection on batch of images.

        Uses sequential processing per-image for v1.5 simplicity.
        Future versions may optimize with parallel processing.

        Args:
            images: List of input images
            text_prompts: Text prompts (same for all images)
            threshold: Detection confidence threshold override
            **kwargs: Additional parameters

        Returns:
            List of VisionResult objects
        """
        # Check batch size warning
        if len(images) > self.max_batch_size:
            logger.warning(
                f"Batch size {len(images)} exceeds recommended max {self.max_batch_size}. "
                f"This may cause memory issues. Consider processing in smaller batches."
            )

        # Process each image sequentially
        results = []
        for i, img in enumerate(images):
            logger.debug(f"Processing image {i+1}/{len(images)}")
            result = self._predict_single(img, text_prompts, threshold, **kwargs)
            results.append(result)

        return results

    def _fallback_postprocess(self, outputs: Any, image: Image.Image, threshold: float) -> dict[str, Any]:
        """Fallback postprocessing when processor method unavailable.

        Args:
            outputs: Model outputs
            image: Input PIL image
            threshold: Confidence threshold

        Returns:
            Dictionary with scores, labels, boxes
        """
        # Extract logits and boxes from outputs
        logits = outputs.logits[0]  # (num_queries, num_classes)
        boxes = outputs.pred_boxes[0]  # (num_queries, 4)

        # Apply sigmoid/softmax to logits
        if self.architecture == "grounding_dino":
            probs = logits.sigmoid()
        else:
            probs = logits.softmax(-1)

        # Get max scores and labels
        scores, labels = probs.max(-1)

        # Filter by threshold
        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # Convert boxes to xyxy format
        # Assuming cxcywh normalized format from model
        img_w, img_h = image.size
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes_xyxy = self.torch.stack([x1, y1, x2, y2], dim=-1)

        return {"scores": scores.cpu(), "labels": labels.cpu(), "boxes": boxes_xyxy.cpu()}

    def info(self) -> dict[str, Any]:
        """Get adapter metadata.

        Returns:
            Dictionary with adapter information
        """
        return {
            "name": "HuggingFaceZeroShotDetectAdapter",
            "task": self.task,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "device": str(self.device),
            "threshold": self.threshold,
            "max_batch_size": self.max_batch_size,
            "mode": "zeroshot",
            "backend": "huggingface-transformers",
        }
