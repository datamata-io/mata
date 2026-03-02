"""HuggingFace CLIP zero-shot image classification adapter.

Supports text-prompt-based image classification using CLIP models (Contrastive
Language-Image Pre-training). Enables classification with custom categories
defined at runtime via text prompts, without requiring training on predefined
class vocabularies.

Key Features:
- Zero-shot classification with dynamic text prompts
- Template customization with predefined ensemble sets
- Softmax-calibrated similarity scores (configurable)
- Combined threshold and top-k filtering
- Support for any HuggingFace CLIP-compatible model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Classification, ClassifyResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | np.ndarray

# Lazy imports for transformers-specific modules
_transformers = None
TRANSFORMERS_AVAILABLE = None

# Predefined template sets for common use cases
TEMPLATE_SETS = {
    "basic": ["a photo of a {}"],
    "ensemble": [
        "a photo of a {}",
        "a picture of a {}",
        "an image of a {}",
        "a rendering of a {}",
        "a cropped photo of a {}",
        "a good photo of a {}",
    ],
    "detailed": [
        "a photo of a {}",
        "a blurry photo of a {}",
        "a black and white photo of a {}",
        "a low contrast photo of a {}",
        "a high contrast photo of a {}",
        "a bad photo of a {}",
        "a good photo of a {}",
        "a photo of a small {}",
        "a photo of a big {}",
        "a photo of the {}",
        "a blurry photo of the {}",
        "a black and white photo of the {}",
        "a low contrast photo of the {}",
        "a high contrast photo of the {}",
        "a bad photo of the {}",
        "a good photo of the {}",
        "a photo of the small {}",
        "a photo of the big {}",
    ],
}


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading)."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import CLIPModel, CLIPProcessor

            _transformers = {
                "CLIPProcessor": CLIPProcessor,
                "CLIPModel": CLIPModel,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


class HuggingFaceCLIPAdapter(PyTorchBaseAdapter):
    """Zero-shot image classification with CLIP.

    Uses Contrastive Language-Image Pre-training (CLIP) to perform image
    classification with arbitrary text-defined categories. Unlike traditional
    classifiers, CLIP allows you to define categories at inference time rather
    than training time.

    Supports:
    - OpenAI CLIP models (clip-vit-base-patch32, clip-vit-large-patch14, etc.)
    - LAION OpenCLIP models
    - Any HuggingFace model compatible with CLIPProcessor/CLIPModel

    Template Options:
    - Single template: "a photo of a {}" (default)
    - Template list: ["a photo of a {}", "a picture of a {}"]
    - Template shortcuts: "basic", "ensemble", "detailed" (from TEMPLATE_SETS)

    Attributes:
        task: Always "classify"
        model_id: HuggingFace model identifier
        processor: CLIP processor for image/text preprocessing
        model: CLIP model for inference
        template: Text template(s) for formatting prompts
        top_k: Number of top predictions to return
        use_softmax: Whether to apply softmax to similarity scores

    Examples:
        >>> # Basic usage with default template
        >>> classifier = HuggingFaceCLIPAdapter("openai/clip-vit-base-patch32")
        >>> result = classifier.predict(
        ...     "image.jpg",
        ...     text_prompts=["cat", "dog", "bird"]
        >>> )
        >>> print(result.get_top1().label_name)  # "cat"

        >>> # Custom template
        >>> classifier = HuggingFaceCLIPAdapter(
        ...     "openai/clip-vit-base-patch32",
        ...     template="this is a photo of a {}"
        >>> )
        >>>
        >>> # Template ensemble (improved accuracy)
        >>> classifier = HuggingFaceCLIPAdapter(
        ...     "openai/clip-vit-base-patch32",
        ...     template="ensemble"  # Uses TEMPLATE_SETS["ensemble"]
        >>> )
        >>>
        >>> # Threshold + top-k filtering
        >>> result = classifier.predict(
        ...     "image.jpg",
        ...     text_prompts=["cat", "dog", "bird", "fish", "horse"],
        ...     threshold=0.1,  # Filter low-confidence predictions
        ...     top_k=3  # Return max 3 predictions
        >>> )
        >>>
        >>> # Raw similarities (no softmax)
        >>> result = classifier.predict(
        ...     "image.jpg",
        ...     text_prompts=["cat", "dog"],
        ...     use_softmax=False  # Return raw CLIP similarity scores
        >>> )
    """

    task = "classify"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        top_k: int = 5,
        threshold: float = 0.0,
        template: str | list[str] = "a photo of a {}",
        use_softmax: bool = True,
    ) -> None:
        """Initialize CLIP zero-shot classification adapter.

        Args:
            model_id: HuggingFace model ID (e.g., "openai/clip-vit-base-patch32")
            device: Device ("cuda", "cpu", or "auto")
            top_k: Number of top predictions to return (default: 5)
            threshold: Minimum confidence threshold (default: 0.0, disabled)
            template: Text template(s) for prompts. Can be:
                - String: Single template like "a photo of a {}"
                - List: Multiple templates for ensemble averaging
                - Shortcut: "basic", "ensemble", or "detailed" (from TEMPLATE_SETS)
            use_softmax: Apply softmax to similarities (default: True)

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            ValueError: If template shortcut is invalid
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold)

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers library is required for CLIP adapter. " "Install with: pip install transformers"
            )

        self.model_id = model_id
        self.top_k = top_k
        self.use_softmax = use_softmax

        # Resolve template (handle shortcuts)
        self.template = self._resolve_template(template)
        logger.info(
            f"Using template: {self.template if isinstance(self.template, str) else f'{len(self.template)} templates (ensemble)'}"
        )

        # Load model and processor
        self._load_model()

    def _resolve_template(self, template: str | list[str]) -> str | list[str]:
        """Resolve template from shortcuts or validate custom template.

        Args:
            template: Template string, list, or shortcut name

        Returns:
            Resolved template (string or list)

        Raises:
            ValueError: If template shortcut is invalid
        """
        # If it's a list, return as-is
        if isinstance(template, list):
            if not template:
                raise ValueError("Template list cannot be empty")
            # Validate all templates have {} placeholder
            for tmpl in template:
                if "{}" not in tmpl:
                    raise ValueError(f"Template '{tmpl}' must contain {{}} placeholder")
            return template

        # If it's a shortcut, resolve from TEMPLATE_SETS
        if template in TEMPLATE_SETS:
            logger.debug(f"Resolved template shortcut '{template}' to {len(TEMPLATE_SETS[template])} templates")
            return TEMPLATE_SETS[template]

        # Otherwise, treat as custom template string
        if "{}" not in template:
            raise ValueError(f"Template '{template}' must contain {{}} placeholder")

        return template

    def _load_model(self) -> None:
        """Load CLIP model and processor.

        Raises:
            ModelLoadError: If model loading fails
        """
        from mata.core.logging import suppress_third_party_logs

        try:
            transformers = _ensure_transformers()

            logger.info(f"Loading CLIP model: {self.model_id}")

            # Load processor and model (suppress noisy HF output)
            with suppress_third_party_logs():
                self.processor = transformers["CLIPProcessor"].from_pretrained(self.model_id)
                self.model = transformers["CLIPModel"].from_pretrained(self.model_id)

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded CLIP model on device: {self.device}")

        except Exception as e:
            raise ModelLoadError(self.model_id, f"Failed to load CLIP model: {str(e)}") from e

    def _format_text_prompts(self, text_prompts: str | list[str]) -> list[str]:
        """Format text prompts with template(s).

        Args:
            text_prompts: Raw class names (e.g., ["cat", "dog"] or "cat, dog")

        Returns:
            Formatted prompts using template(s)
        """
        # Normalize to list
        if isinstance(text_prompts, str):
            # Handle comma-separated string
            prompts = [p.strip() for p in text_prompts.split(",")]
        else:
            prompts = text_prompts

        # If template is a single string, format directly
        if isinstance(self.template, str):
            return [self.template.format(prompt) for prompt in prompts]

        # If template is a list, create ensemble (all templates × all prompts)
        # We'll average similarities across templates later
        formatted = []
        for prompt in prompts:
            for tmpl in self.template:
                formatted.append(tmpl.format(prompt))

        return formatted

    def _extract_label_names(self, text_prompts: str | list[str]) -> list[str]:
        """Extract original label names from text prompts.

        Args:
            text_prompts: Raw class names

        Returns:
            List of label names
        """
        if isinstance(text_prompts, str):
            return [p.strip() for p in text_prompts.split(",")]
        return text_prompts

    def predict(
        self,
        image: ImageInput,
        text_prompts: str | list[str],
        top_k: int | None = None,
        threshold: float | None = None,
        use_softmax: bool | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Perform zero-shot image classification with text prompts.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            text_prompts: Text prompts defining classes. Can be:
                - List: ["cat", "dog", "bird"]
                - Comma-separated string: "cat, dog, bird"
            top_k: Number of top predictions (overrides instance setting)
            threshold: Confidence threshold (overrides instance setting)
            use_softmax: Apply softmax to scores (overrides instance setting)
            **kwargs: Additional inference parameters

        Returns:
            ClassifyResult with predictions sorted by confidence (descending)

        Raises:
            InvalidInputError: If image or text_prompts are invalid
            RuntimeError: If inference fails

        Example:
            >>> result = classifier.predict(
            ...     "cat.jpg",
            ...     text_prompts=["cat", "dog", "bird"],
            ...     threshold=0.1,
            ...     top_k=2
            >>> )
            >>> for pred in result.predictions:
            ...     print(f"{pred.label_name}: {pred.score:.3f}")
            cat: 0.856
            dog: 0.124
        """
        # Load image
        pil_image, input_path = self._load_image(image)

        # Extract label names for id2label mapping
        label_names = self._extract_label_names(text_prompts)
        num_classes = len(label_names)

        if num_classes == 0:
            raise InvalidInputError("text_prompts must contain at least one class")

        # Format prompts with template(s)
        formatted_prompts = self._format_text_prompts(text_prompts)

        # Determine if we're using ensemble
        is_ensemble = isinstance(self.template, list)
        num_templates = len(self.template) if is_ensemble else 1

        logger.debug(
            f"Processing image with {num_classes} classes, "
            f"{num_templates} template(s), "
            f"total {len(formatted_prompts)} text inputs"
        )

        # Preprocess inputs
        try:
            inputs = self.processor(text=formatted_prompts, images=pil_image, return_tensors="pt", padding=True)

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        except Exception as e:
            raise InvalidInputError(f"Failed to preprocess inputs: {str(e)}") from e

        # Run inference
        try:
            with self.torch.no_grad():
                outputs = self.model(**inputs)

            # Get image-text similarities
            # Shape: (1, num_prompts) where num_prompts = num_classes * num_templates
            logits_per_image = outputs.logits_per_image

            # If ensemble, average across templates
            if is_ensemble:
                # Reshape to (1, num_classes, num_templates)
                logits_reshaped = logits_per_image.view(1, num_classes, num_templates)
                # Average across templates (dim=2)
                logits_per_image = logits_reshaped.mean(dim=2)

            # Squeeze to 1D: (num_classes,)
            similarities = logits_per_image.squeeze(0).cpu().numpy()

        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}") from e

        # Apply softmax if requested
        _use_softmax = use_softmax if use_softmax is not None else self.use_softmax
        if _use_softmax:
            # Convert logits to probabilities
            exp_logits = np.exp(similarities - np.max(similarities))  # Numerical stability
            scores = exp_logits / exp_logits.sum()
        else:
            # Use raw similarities
            scores = similarities

        # Build id2label mapping dynamically
        id2label = {i: name for i, name in enumerate(label_names)}

        # Create predictions list
        predictions = []
        for label_idx, score in enumerate(scores):
            predictions.append(Classification(label=label_idx, score=float(score), label_name=id2label[label_idx]))

        # Apply threshold filtering
        _threshold = threshold if threshold is not None else self.threshold
        if _threshold > 0.0:
            predictions = [p for p in predictions if p.score >= _threshold]
            logger.debug(f"Filtered to {len(predictions)} predictions with threshold {_threshold}")

        # Sort by score (descending)
        predictions.sort(key=lambda x: x.score, reverse=True)

        # Apply top-k selection
        _top_k = top_k if top_k is not None else self.top_k
        if _top_k is not None and _top_k > 0:
            predictions = predictions[:_top_k]

        # Build metadata
        meta = {
            "model_id": self.model_id,
            "device": str(self.device),
            "num_classes": num_classes,
            "template_type": "ensemble" if is_ensemble else "single",
            "num_templates": num_templates,
            "use_softmax": _use_softmax,
            "threshold": _threshold,
            "top_k": _top_k,
        }

        if input_path:
            meta["input_path"] = str(input_path)

        return ClassifyResult(predictions=predictions, meta=meta)

    def classify(
        self,
        image: str | Path | Image.Image | np.ndarray,
        text_prompts: list[str] | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image (Classify node interface).

        This method provides compatibility with the Classify graph node,
        which expects a classify() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            text_prompts: Text prompts for zero-shot classification
            top_k: Number of top predictions to return
            **kwargs: Additional arguments passed to predict()

        Returns:
            ClassifyResult with predictions
        """
        return self.predict(
            image=image,
            text_prompts=text_prompts,
            top_k=top_k,
            **kwargs,
        )

    def info(self) -> dict[str, Any]:
        """Get adapter information and metadata.

        Returns:
            Dictionary with adapter metadata
        """
        template_info = (
            f"{len(self.template)} templates (ensemble)"
            if isinstance(self.template, list)
            else f"single: {self.template}"
        )

        return {
            "name": "HuggingFaceCLIPAdapter",
            "task": "classify",
            "model_id": self.model_id,
            "device": str(self.device),
            "architecture": "clip",
            "backend": "transformers",
            "top_k": self.top_k,
            "threshold": self.threshold,
            "template": template_info,
            "use_softmax": self.use_softmax,
        }
