"""HuggingFace zero-shot image segmentation adapter.

Supports text-prompt-based image segmentation using lightweight Transformer
models that produce per-pixel class predictions from natural language queries.
Unlike SAM (prompt-based mask generator), these models are true zero-shot
semantic segmenters — they output class-specific segmentation maps directly
from text descriptions.

Supported models:
- **CLIPSeg** (CIDAS/clipseg-rd64-refined, CIDAS/clipseg-rd16)
    - ~150M params (vs SAM's 375M–641M)
    - Text-only input, no point/box prompts needed
    - Per-prompt heatmaps with sigmoid activation

Key Features:
- Zero-shot segmentation with arbitrary text prompts
- Lightweight compared to SAM-based approaches
- Per-class probability maps from text descriptions
- Configurable threshold and mask format (RLE/polygon/binary)
- Automatic resizing of output masks to original image dimensions
"""

from __future__ import annotations

import warnings
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

# Lazy imports for transformers and pycocotools
_transformers = None
_mask_utils = None
TRANSFORMERS_AVAILABLE = None
PYCOCOTOOLS_AVAILABLE = None


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading)."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import (
                AutoProcessor,
                CLIPSegForImageSegmentation,
                CLIPSegProcessor,
            )

            _transformers = {
                "CLIPSegForImageSegmentation": CLIPSegForImageSegmentation,
                "CLIPSegProcessor": CLIPSegProcessor,
                "AutoProcessor": AutoProcessor,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


def _ensure_pycocotools():
    """Ensure pycocotools is imported for RLE encoding (optional)."""
    global _mask_utils, PYCOCOTOOLS_AVAILABLE
    if _mask_utils is None:
        try:
            from pycocotools import mask as mask_utils

            _mask_utils = mask_utils
            PYCOCOTOOLS_AVAILABLE = True
        except ImportError:
            PYCOCOTOOLS_AVAILABLE = False
    return _mask_utils


class HuggingFaceZeroShotSegmentAdapter(PyTorchBaseAdapter):
    """Zero-shot image segmentation with text prompts.

    Uses CLIPSeg and similar Transformer models to produce per-pixel
    segmentation masks from natural language descriptions. Unlike SAM
    (which requires point/box/text prompts and generates class-agnostic
    masks), this adapter produces class-specific segmentation maps
    directly from text — a true zero-shot semantic segmenter.

    Model comparison:
        - **CLIPSeg** (~150M params): Lightweight, fast, text-only
        - **SAM/SAM3** (375M–641M params): Heavy, prompt-based, class-agnostic

    Supported Models:
        - ``CIDAS/clipseg-rd64-refined`` (recommended, 64px decoder)
        - ``CIDAS/clipseg-rd16`` (16px decoder, faster but less precise)

    Output:
        Each text prompt produces a binary mask instance with:
        - ``label_name``: The text prompt used
        - ``score``: Mean sigmoid probability within the mask region
        - ``mask``: Binary mask in configured format (RLE/polygon/binary)
        - ``bbox``: Bounding box computed from the mask (xyxy format)

    Examples:
        >>> # Basic zero-shot segmentation
        >>> segmenter = HuggingFaceZeroShotSegmentAdapter(
        ...     "CIDAS/clipseg-rd64-refined"
        ... )
        >>> result = segmenter.predict(
        ...     "image.jpg",
        ...     text_prompts=["cat", "dog", "background"]
        ... )
        >>> for inst in result.instances:
        ...     print(f"{inst.label_name}: area={inst.area}")
        >>>
        >>> # With higher threshold for precise masks
        >>> result = segmenter.predict(
        ...     "image.jpg",
        ...     text_prompts="car . person . road",
        ...     threshold=0.6
        ... )
        >>>
        >>> # RLE-encoded masks for storage
        >>> segmenter = HuggingFaceZeroShotSegmentAdapter(
        ...     "CIDAS/clipseg-rd64-refined",
        ...     use_rle=True
        ... )
    """

    task = "segment"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.5,
        use_rle: bool = True,
        use_polygon: bool = False,
        polygon_tolerance: float = 2.0,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize zero-shot segmentation adapter.

        Args:
            model_id: HuggingFace model ID
                Examples:
                - "CIDAS/clipseg-rd64-refined" (recommended)
                - "CIDAS/clipseg-rd16"
            device: Device ("cuda", "cpu", or "auto")
            threshold: Sigmoid probability threshold for mask binarization [0.0, 1.0]
                - Lower values: More inclusive masks (more pixels included)
                - Higher values: More precise masks (fewer pixels included)
                - Default: 0.5
            use_rle: Use RLE encoding for masks (requires pycocotools)
                - True: Compact RLE format (recommended for storage)
                - False: Binary numpy arrays or polygons
            use_polygon: Use polygon format for masks (requires opencv-python)
                - True: Polygon coordinates in COCO format
                - False: RLE or binary masks (default)
                - When True, overrides use_rle setting
            polygon_tolerance: Polygon approximation tolerance
                - Lower values: More precise polygons
                - Higher values: Simpler polygons
                - Default: 2.0
            id2label: Optional custom label mapping (overrides text prompts)

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers library is required for zero-shot segmentation. " "Install with: pip install transformers"
            )

        self.model_id = model_id

        # Detect architecture
        self.architecture = self._detect_architecture()
        logger.info(f"Detected zero-shot segmentation architecture: {self.architecture}")

        # Handle mask format options
        if use_polygon:
            try:
                import cv2  # noqa: F401

                self.use_polygon = True
                self.use_rle = False  # Polygon takes precedence
                self.polygon_tolerance = polygon_tolerance
                logger.info(f"Using polygon mask format (tolerance={polygon_tolerance})")
            except ImportError:
                raise ImportError(
                    "OpenCV is required for polygon mask format. " "Install with: pip install opencv-python"
                )
        else:
            self.use_polygon = False
            self.polygon_tolerance = polygon_tolerance

            if use_rle:
                mask_utils = _ensure_pycocotools()
                if not mask_utils:
                    warnings.warn(
                        "pycocotools not available. Falling back to binary masks. "
                        "Install with: pip install mata[segmentation] or pip install pycocotools",
                        UserWarning,
                        stacklevel=2,
                    )
                    use_rle = False
            self.use_rle = use_rle

        # Load model and processor
        self._load_model()

    def _detect_architecture(self) -> str:
        """Detect zero-shot segmentation model architecture from model ID.

        Returns:
            Architecture name: "clipseg" or "auto"
        """
        model_id_lower = self.model_id.lower()

        if "clipseg" in model_id_lower:
            return "clipseg"

        # Future architectures can be added here:
        # elif "groupvit" in model_id_lower:
        #     return "groupvit"

        # Unknown - try generic auto-loading
        logger.warning(
            f"Unknown zero-shot segmentation architecture for '{self.model_id}'. "
            f"Attempting CLIPSeg-compatible loading."
        )
        return "auto"

    def _load_model(self) -> None:
        """Load zero-shot segmentation model and processor.

        Raises:
            ModelLoadError: If model loading fails
        """
        from mata.core.logging import suppress_third_party_logs

        try:
            transformers = _ensure_transformers()

            with suppress_third_party_logs():
                if self.architecture == "clipseg":
                    logger.info(f"Loading CLIPSeg model: {self.model_id}")
                    self.processor = transformers["CLIPSegProcessor"].from_pretrained(self.model_id)
                    self.model = transformers["CLIPSegForImageSegmentation"].from_pretrained(self.model_id)
                else:
                    # Generic auto-loading fallback (CLIPSeg-compatible)
                    logger.info(f"Loading zero-shot segmentation model with auto-detection: {self.model_id}")
                    self.processor = transformers["AutoProcessor"].from_pretrained(self.model_id)
                    self.model = transformers["CLIPSegForImageSegmentation"].from_pretrained(self.model_id)

            # Move model to device and set to eval mode
            self.model.to(self.device).eval()
            logger.info(f"Successfully loaded {self.architecture} on {self.device}")

        except Exception as e:
            raise ModelLoadError(
                self.model_id,
                f"Failed to load zero-shot segmentation model: {type(e).__name__}: {str(e)}",
            ) from e

    def _normalize_text_prompts(self, text_prompts: str | list[str]) -> list[str]:
        """Normalize text prompts to a list of strings.

        Accepts multiple input formats for user convenience:
        - List: ["cat", "dog", "person"]
        - Comma-separated: "cat, dog, person"
        - Space-dot separated: "cat . dog . person" (GroundingDINO style)

        Args:
            text_prompts: Text prompts in any supported format

        Returns:
            List of individual prompt strings
        """
        if isinstance(text_prompts, list):
            return [p.strip() for p in text_prompts if p.strip()]

        # String input - detect format
        text = text_prompts.strip()

        # Space-dot separated (e.g., "cat . dog . person")
        if " . " in text:
            return [p.strip() for p in text.split(".") if p.strip()]

        # Comma-separated (e.g., "cat, dog, person")
        if "," in text:
            return [p.strip() for p in text.split(",") if p.strip()]

        # Single prompt
        return [text]

    def predict(
        self,
        image: ImageInput,
        text_prompts: str | list[str] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Run zero-shot segmentation with text prompts.

        Each text prompt produces a per-pixel probability map. Pixels above
        the threshold are included in the binary mask for that class. Empty
        masks (no pixels above threshold) are omitted from results.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            text_prompts: Text description(s) of objects to segment
                - List: ["cat", "dog", "background"]
                - Comma-separated: "cat, dog, background"
                - Space-dot: "cat . dog . background"
            threshold: Sigmoid threshold override for mask binarization
            **kwargs: Additional inference parameters

        Returns:
            VisionResult with segmentation mask instances

        Raises:
            InvalidInputError: If text_prompts is missing or empty

        Examples:
            >>> result = segmenter.predict(
            ...     "image.jpg",
            ...     text_prompts=["cat", "dog"]
            ... )
            >>> for inst in result.instances:
            ...     print(f"{inst.label_name}: score={inst.score:.3f}")
        """
        # Validate text prompts
        if not text_prompts:
            raise InvalidInputError(
                "text_prompts required for zero-shot segmentation. "
                "Example: text_prompts=['cat', 'dog'] or text_prompts='cat . dog'"
            )

        # Load and validate image
        pil_image, input_path = self._load_image(image)
        orig_width, orig_height = pil_image.size
        conf_threshold = threshold if threshold is not None else self.threshold

        # Normalize prompts
        prompts = self._normalize_text_prompts(text_prompts)
        if not prompts:
            raise InvalidInputError("text_prompts must contain at least one non-empty prompt")

        logger.info(
            f"Running zero-shot segmentation with {len(prompts)} prompts " f"on {orig_width}x{orig_height} image"
        )

        # CLIPSeg expects: images=[image]*N, text=prompts (one image per prompt)
        inputs = self.processor(
            text=prompts,
            images=[pil_image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits: (N, H_model, W_model) — one heatmap per prompt
        logits = outputs.logits

        # Handle single prompt case (logits may be (H, W) instead of (1, H, W))
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)

        # Apply sigmoid to convert logits to probabilities
        probs = self.torch.sigmoid(logits)

        # Resize probability maps to original image dimensions
        probs_resized = self.torch.nn.functional.interpolate(
            probs.unsqueeze(1),  # (N, 1, H_model, W_model)
            size=(orig_height, orig_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # (N, H_orig, W_orig)

        # Convert to numpy
        probs_np = probs_resized.cpu().numpy()

        # Generate instances from probability maps
        instances = []
        for idx, prompt in enumerate(prompts):
            prob_map = probs_np[idx]  # (H, W)

            # Binarize with threshold
            binary_mask = prob_map >= conf_threshold

            # Skip empty masks
            area = int(binary_mask.sum())
            if area == 0:
                logger.debug(f"Prompt '{prompt}' produced empty mask at threshold={conf_threshold}")
                continue

            # Compute mean probability within mask as confidence score
            score = float(prob_map[binary_mask].mean())

            # Compute bounding box from mask
            bbox = self._mask_to_bbox(binary_mask)

            # Convert mask to configured format
            mask_data = self._convert_mask_format(binary_mask)

            instances.append(
                Instance(
                    mask=mask_data,
                    score=score,
                    label=idx,
                    label_name=prompt,
                    bbox=bbox,
                    is_stuff=None,  # Zero-shot — no thing/stuff distinction
                    area=area,
                )
            )

        result = VisionResult(
            instances=instances,
            meta={
                "model_id": self.model_id,
                "architecture": self.architecture,
                "text_prompts": prompts,
                "threshold": conf_threshold,
                "mode": "zeroshot",
                "image_size": [orig_width, orig_height],
                "backend": "transformers",
                "mask_format": "polygon" if self.use_polygon else ("rle" if self.use_rle else "binary"),
                "input_path": input_path,
            },
        )

        logger.info(f"Found {len(instances)} segments from {len(prompts)} prompts " f"(threshold={conf_threshold})")

        return result

    def segment(
        self,
        image: ImageInput,
        text_prompts: str | list[str] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Segment image (graph node interface).

        Wraps predict() for compatibility with the Segment graph node,
        which expects a segment() method.

        Args:
            image: Input image
            text_prompts: Text prompts for zero-shot segmentation
            threshold: Sigmoid threshold override
            **kwargs: Additional arguments passed to predict()

        Returns:
            VisionResult with segmentation instances
        """
        return self.predict(
            image=image,
            text_prompts=text_prompts,
            threshold=threshold,
            **kwargs,
        )

    def _convert_mask_format(
        self, binary_mask: np.ndarray
    ) -> dict[str, Any] | np.ndarray | list[float] | list[list[float]]:
        """Convert binary mask to desired output format.

        Args:
            binary_mask: Binary numpy array (H, W) with dtype bool or uint8

        Returns:
            Mask in the configured format:
            - Polygon: List of polygon coordinates [[x1, y1, x2, y2, ...], ...]
            - RLE: Dict with 'size' and 'counts' keys
            - Binary: numpy array (H, W)
        """
        if self.use_polygon:
            from mata.core.mask_utils import binary_mask_to_polygon

            polygons = binary_mask_to_polygon(binary_mask, tolerance=self.polygon_tolerance, min_area=10)
            return polygons[0] if len(polygons) == 1 else polygons

        elif self.use_rle and PYCOCOTOOLS_AVAILABLE:
            binary_mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
            rle = _mask_utils.encode(binary_mask_fortran)
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        else:
            return binary_mask

    def _mask_to_bbox(self, binary_mask: np.ndarray) -> tuple[float, float, float, float] | None:
        """Compute bounding box from binary mask.

        Args:
            binary_mask: Binary mask array (H, W)

        Returns:
            Bounding box (x1, y1, x2, y2) in xyxy format, or None if empty
        """
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (float(x_min), float(y_min), float(x_max), float(y_max))

    def info(self) -> dict[str, Any]:
        """Get adapter metadata.

        Returns:
            Dictionary with adapter information
        """
        return {
            "name": "HuggingFaceZeroShotSegmentAdapter",
            "task": self.task,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "device": str(self.device),
            "threshold": self.threshold,
            "mode": "zeroshot",
            "mask_format": "polygon" if self.use_polygon else ("rle" if self.use_rle else "binary"),
            "backend": "huggingface-transformers",
        }
