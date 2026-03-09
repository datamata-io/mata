"""HuggingFace SAM (Segment Anything Model) adapter for zero-shot segmentation.

Supports automatic detection and loading of SAM models from HuggingFace Hub,
enabling prompt-based mask generation with:
- **Text prompts** (SAM3 only - zero-shot concept segmentation)
- **Point prompts** (foreground/background clicks)
- **Box prompts** (bounding boxes)
- **Combined prompting** (text + negative boxes for refinement)
- Multi-mask output (3 masks per prompt with IoU scores)

SAM3 adds Promptable Concept Segmentation (PCS) with text prompts:
- Text-only: "cat", "person", "laptop" (finds all instances)
- Text + negative boxes: Exclude regions from text-based segmentation
- Supports 270K+ open-vocabulary concepts

Examples:
    >>> # Text prompt (SAM3 only)
    >>> sam = HuggingFaceSAMAdapter("facebook/sam3")
    >>> result = sam.predict("image.jpg", text_prompts="cat")
    >>> print(f"Found {len(result.masks)} cats")
    >>>
    >>> # Text + negative box (exclude region)
    >>> result = sam.predict(
    ...     "image.jpg",
    ...     text_prompts="handle",
    ...     box_prompts=[(40, 183, 318, 204)],
    ...     box_labels=[0]  # Exclude oven handle
    ... )
    >>>
    >>> # Point prompts (original SAM)
    >>> sam = HuggingFaceSAMAdapter("facebook/sam-vit-base")
    >>> result = sam.predict("image.jpg", point_prompts=[(100, 150, 1)])  # x, y, fg
    >>> print(f"Generated {len(result.masks)} masks")  # 3 masks with different IoU
    >>>
    >>> # Box prompts
    >>> result = sam.predict("image.jpg", box_prompts=[(50, 50, 300, 300)])
    >>>
    >>> # Threshold filtering
    >>> result = sam.predict("image.jpg", text_prompts="dog", threshold=0.7)
    >>> # Returns only masks with score >= 0.7
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError, ModelLoadError, UnsupportedModelError
from mata.core.logging import get_logger
from mata.core.types import Instance, VisionResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

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
            # Import original SAM classes
            from transformers import SamModel, SamProcessor

            # Try importing SAM3 classes (may not exist in older transformers)
            try:
                from transformers import Sam3Model, Sam3Processor

                sam3_available = True
            except ImportError:
                sam3_available = False
                logger.warning(
                    "SAM3 classes not found in transformers. Text prompts require "
                    "transformers>=4.46.0. Install with: pip install -U transformers"
                )

            _transformers = {
                "SamModel": SamModel,
                "SamProcessor": SamProcessor,
            }

            if sam3_available:
                _transformers["Sam3Model"] = Sam3Model
                _transformers["Sam3Processor"] = Sam3Processor

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


class HuggingFaceSAMAdapter(PyTorchBaseAdapter):
    """HuggingFace SAM adapter for zero-shot segmentation.

    SAM (Segment Anything Model) generates class-agnostic masks based on
    user-provided prompts (points, boxes, or masks). Unlike traditional
    segmentation models, SAM doesn't require training on specific classes.

    Key Features:
    - Prompt-based mask generation (no class labels needed)
    - Multi-mask output (3 predictions per prompt with IoU scores)
    - Optional threshold filtering by IoU prediction scores
    - Supports point prompts (foreground/background clicks)
    - Supports box prompts (rectangular regions)
    - RLE mask encoding (compact JSON-serializable format)

    Mask Output:
    - By default, returns 3 masks per prompt (different granularities)
    - Each mask has an IoU prediction score (quality metric)
    - Use threshold parameter to filter low-quality masks
    - Without threshold: Returns all masks for user-side filtering

    Coordinate System:
    - Uses MATA standard: xyxy format (absolute pixel coordinates)
    - Point prompts: (x, y, label) where label=1 (foreground), 0 (background)
    - Box prompts: (x1, y1, x2, y2) in absolute pixels

    Class-Agnostic Design:
    - All masks have label=0, label_name="object"
    - is_stuff field is None (not applicable to zero-shot)
    - Use external classifier for class assignment (future feature)

    Examples:
        >>> # Single point prompt
        >>> sam = HuggingFaceSAMAdapter("facebook/sam-vit-base")
        >>> result = sam.predict("image.jpg", point_prompts=[(100, 150, 1)])
        >>> print(f"Generated {len(result.masks)} masks")  # Typically 3
        >>>
        >>> # Filter by IoU score
        >>> result = sam.predict(
        ...     "image.jpg",
        ...     point_prompts=[(100, 150, 1)],
        ...     threshold=0.8  # Only high-quality masks
        ... )
        >>>
        >>> # Box prompt (segment entire region)
        >>> result = sam.predict("image.jpg", box_prompts=[(50, 50, 300, 300)])
        >>>
        >>> # Combined prompts for refinement
        >>> result = sam.predict(
        ...     "image.jpg",
        ...     point_prompts=[(100, 150, 1), (200, 250, 0)],  # Include/exclude
        ...     box_prompts=[(50, 50, 300, 300)]
        ... )
        >>>
        >>> # Access mask details
        >>> for mask in result.masks:
        ...     print(f"IoU: {mask.score:.3f}, Area: {mask.area} pixels")
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.0,
        use_rle: bool = True,
        use_polygon: bool = False,
        polygon_tolerance: float = 2.0,
        id2label: dict[int, str] | None = None,
        token: str | bool | None = None,
    ) -> None:
        """Initialize HuggingFace SAM adapter.

        Args:
            model_id: HuggingFace model ID
                Examples:
                - "facebook/sam-vit-base"
                - "facebook/sam-vit-large"
                - "facebook/sam-vit-huge"
            device: Device ("cuda", "cpu", or "auto")
            threshold: IoU score threshold for mask filtering [0.0, 1.0]
                - 0.0: Return all masks (default)
                - Higher values: Filter low-quality predictions
                - Typical range: 0.5-0.9
            use_rle: Use RLE encoding for masks (requires pycocotools)
                - True: Compact RLE format (recommended for storage)
                - False: Binary numpy arrays or polygons
                - Note: Ignored if use_polygon=True
            use_polygon: Use polygon format for masks (requires opencv-python)
                - True: Polygon coordinates [x1, y1, x2, y2, ...] in COCO format
                - False: RLE or binary masks (default)
                - When True, overrides use_rle setting
            polygon_tolerance: Polygon approximation tolerance (epsilon for cv2.approxPolyDP)
                - Lower values: More precise polygons with more points (e.g., 1.0)
                - Higher values: Simpler polygons with fewer points (e.g., 3.0)
                - Default: 2.0 (balanced)
            id2label: Ignored for SAM (class-agnostic), kept for API compatibility
            token: HuggingFace authentication token for gated models (e.g., SAM3)
                - None: Use default authentication (cached token if available)
                - str: Explicit token string (e.g., "hf_...")
                - True: Force use of cached token
                - False: Don't use authentication

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model is not a SAM variant
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers is required for HuggingFace SAM adapter. " "Install with: pip install transformers"
            )

        # Check dependencies for mask formats
        if use_polygon:
            # Check for OpenCV (required for polygon conversion)
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

            # Check for pycocotools if RLE encoding requested
            if use_rle:
                mask_utils = _ensure_pycocotools()
                if not mask_utils:
                    warnings.warn(
                        "pycocotools not available. Falling back to binary masks. "
                        "Install with: pip install datamata[segmentation] or pip install pycocotools",
                        UserWarning,
                        stacklevel=2,
                    )
                    use_rle = False
            self.use_rle = use_rle

        self.model_id = model_id

        # Load model and processor
        try:
            logger.info(f"Loading SAM model: {model_id}")

            # Validate model ID contains 'sam'
            if "sam" not in model_id.lower():
                warnings.warn(
                    f"Model ID '{model_id}' does not contain 'sam'. " "Ensure this is a Segment Anything Model.",
                    UserWarning,
                    stacklevel=2,
                )

            # Detect SAM version and load appropriate classes
            is_sam3 = "sam3" in model_id.lower()

            # Suppress noisy third-party output during model loading
            from mata.core.logging import suppress_third_party_logs

            with suppress_third_party_logs():

                if is_sam3:
                    # SAM3 supports text prompts via Sam3Processor/Sam3Model (PCS mode)
                    logger.info("Detected SAM3 model - text prompts enabled")

                    # Check if SAM3 classes are available
                    if "Sam3Processor" not in transformers or "Sam3Model" not in transformers:
                        raise UnsupportedModelError(
                            "SAM3 model requested but not available in transformers. "
                            "Please upgrade: pip install -U transformers>=4.46.0"
                        )

                    Sam3Processor = transformers["Sam3Processor"]  # noqa: N806
                    Sam3Model = transformers["Sam3Model"]  # noqa: N806

                    self.processor = Sam3Processor.from_pretrained(model_id, token=token)
                    self.model = Sam3Model.from_pretrained(model_id, token=token)
                else:
                    # Original SAM - visual prompts only
                    SamProcessor = transformers["SamProcessor"]  # noqa: N806
                    SamModel = transformers["SamModel"]  # noqa: N806

                    self.processor = SamProcessor.from_pretrained(model_id, token=token)
                    self.model = SamModel.from_pretrained(model_id, token=token)

            # Move model to device and set eval mode
            self.model = self.model.to(self.device).eval()

            # SAM is class-agnostic, use default label
            self.id2label = {0: "object"}

            logger.info(f"Loaded SAM model on {self.device} " f"(zero-shot mode, RLE={self.use_rle})")

        except Exception as e:
            if isinstance(e, (ImportError, UnsupportedModelError)):
                raise
            raise ModelLoadError(model_id, f"Failed to load SAM model: {type(e).__name__}: {str(e)}")

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        text_prompts: str | list[str] | None = None,
        point_prompts: list[tuple[float, float, int]] | None = None,
        box_prompts: list[tuple[float, float, float, float]] | None = None,
        box_labels: list[int] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Generate masks from prompts using SAM.

        SAM generates multiple mask predictions per prompt, each with an IoU
        prediction score indicating mask quality. By default, returns all masks
        (typically 3 per prompt). Use threshold to filter by IoU score.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            text_prompts: Text description(s) of objects to segment (SAM3 only)
                - Single string: "cat", "person", "red car"
                - List of strings: ["cat", "dog"] for multiple concepts
                - Supports open-vocabulary concepts (270K+ unique concepts)
                Example: "ear", "handle", "laptop"
            point_prompts: List of point prompts as (x, y, label) tuples
                - x, y: Pixel coordinates (absolute)
                - label: 1 = foreground point, 0 = background point
                Example: [(100, 150, 1), (200, 250, 0)]
            box_prompts: List of box prompts as (x1, y1, x2, y2) tuples
                - x1, y1: Top-left corner (absolute pixels)
                - x2, y2: Bottom-right corner (absolute pixels)
                Example: [(50, 50, 300, 300)]
            box_labels: Labels for box prompts (1=positive/include, 0=negative/exclude)
                - Must match length of box_prompts if provided
                - Default: [1] * len(box_prompts) (all positive)
                Example: [1, 0] for one positive box, one negative box
            threshold: Optional IoU score threshold override [0.0, 1.0]
                - None: Return all masks (default)
                - >0.0: Filter masks with IoU < threshold
                - Typical values: 0.5-0.9
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            VisionResult with zero-shot mask instances
                - instances: List of Instance objects with masks
                - Each instance has score=IoU prediction score
                - Typically 3 masks per prompt if no threshold
                - label=0, label_name="object" (class-agnostic)

        Raises:
            InvalidInputError: If image is invalid or no prompts provided

        Examples:
            >>> # Text prompt (SAM3 only - zero-shot concept)
            >>> result = sam.predict("image.jpg", text_prompts="cat")
            >>> print(f"Found {len(result.instances)} cats")
            >>>
            >>> # Point prompt (foreground click)
            >>> result = sam.predict("image.jpg", point_prompts=[(100, 150, 1)])
            >>> print(f"Generated {len(result.instances)} masks")  # Typically 3
            >>>
            >>> # Text + negative box (exclude region)
            >>> result = sam.predict(
            ...     "image.jpg",
            ...     text_prompts="handle",
            ...     box_prompts=[(40, 183, 318, 204)],
            ...     box_labels=[0],  # Exclude oven handle
            ...     threshold=0.5
            ... )
            >>>
            >>> # Box prompt
            >>> result = sam.predict("image.jpg", box_prompts=[(50, 50, 300, 300)])
            >>>
            >>> # Refinement with foreground + background points
            >>> result = sam.predict(
            ...     "image.jpg",
            ...     point_prompts=[(100, 150, 1), (200, 250, 0)]  # Include + exclude
            ... )
        """
        # Validate at least one prompt provided
        if not text_prompts and not point_prompts and not box_prompts:
            raise InvalidInputError(
                "At least one prompt type (text_prompts, point_prompts, or box_prompts) must be provided. "
                "Examples: text_prompts='cat', point_prompts=[(100, 150, 1)], or box_prompts=[(50, 50, 300, 300)]"
            )

        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)
        orig_width, orig_height = pil_image.size

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Check if this is SAM3 (supports text prompts)
        is_sam3 = "sam3" in self.model_id.lower()
        if text_prompts and not is_sam3:
            raise InvalidInputError(
                f"Text prompts are only supported by SAM3 models. "
                f"Current model '{self.model_id}' does not support text prompts. "
                f"Use 'facebook/sam3' or switch to point/box prompts."
            )

        # Prepare prompts for processor
        # SAM3 processor expects:
        # - text: str or List[str] for text prompts
        # - input_points: List[List[List[float]]] - batch, num_points, 2 (x, y)
        # - input_labels: List[List[int]] - batch, num_points (1=fg, 0=bg)
        # - input_boxes: List[List[List[float]]] - batch, num_boxes, 4 (x1, y1, x2, y2)
        # - input_boxes_labels: List[List[int]] - batch, num_boxes (1=positive, 0=negative)

        input_points = None
        input_labels = None
        input_boxes = None
        input_boxes_labels = None

        if point_prompts:
            # Extract coordinates and labels
            points = [[p[0], p[1]] for p in point_prompts]
            labels = [p[2] for p in point_prompts]
            input_points = [points]  # Batch dimension
            input_labels = [labels]  # Batch dimension

        if box_prompts:
            # Validate box_labels length if provided
            if box_labels is not None and len(box_labels) != len(box_prompts):
                raise InvalidInputError(
                    f"box_labels length ({len(box_labels)}) must match box_prompts length ({len(box_prompts)})"
                )

            # Convert to nested list format
            if is_sam3:
                # SAM3 uses different format: List[List[List[float]]] for batching
                input_boxes = [[list(box) for box in box_prompts]]
                # SAM3 uses box labels (1=positive, 0=negative)
                # Format: List[List[int]] = [image_batch, [per_box_labels]]
                input_boxes_labels = [box_labels or ([1] * len(box_prompts))]
            else:
                # Original SAM: List[List[List[float]]] - batch, num_boxes, 4
                # All boxes in a single batch
                input_boxes = [[list(box) for box in box_prompts]]

        # Preprocess image with prompts
        log_prompts = []
        if text_prompts:
            log_prompts.append(f"text='{text_prompts}'")
        if point_prompts:
            log_prompts.append(f"points={len(point_prompts)}")
        if box_prompts:
            log_prompts.append(f"boxes={len(box_prompts)}")

        logger.info(f"Running SAM on {orig_width}x{orig_height} image ({', '.join(log_prompts)})")

        # Build processor kwargs (only include non-None values)
        # SAM3 with a list of text prompts requires one image copy per prompt
        # (pixel_values batch size must match input_ids batch size)
        if is_sam3 and isinstance(text_prompts, list) and len(text_prompts) > 1:
            processor_images = [pil_image] * len(text_prompts)
        else:
            processor_images = pil_image
        processor_kwargs = {"images": processor_images, "return_tensors": "pt"}

        if text_prompts is not None:
            processor_kwargs["text"] = text_prompts
        if input_points is not None:
            processor_kwargs["input_points"] = input_points
        if input_labels is not None:
            processor_kwargs["input_labels"] = input_labels
        if input_boxes is not None:
            processor_kwargs["input_boxes"] = input_boxes
        if input_boxes_labels is not None:
            processor_kwargs["input_boxes_labels"] = input_boxes_labels

        inputs = self.processor(**processor_kwargs)

        # Move tensors to device
        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, self.torch.Tensor):
                device_inputs[k] = v.to(self.device)
            else:
                device_inputs[k] = v

        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**device_inputs)

        # Post-process outputs differently for SAM3 (text-based) vs SAM (visual prompts)
        if is_sam3 and text_prompts:
            # SAM3 uses post_process_instance_segmentation for text prompts.
            # When batched (list of prompts), original_sizes has one entry per prompt;
            # post_process_instance_segmentation returns one result dict per batch item.
            original_sizes = device_inputs.get("original_sizes", None)
            if original_sizes is None:
                # Fallback: build target_sizes from the number of prompts
                n = len(text_prompts) if isinstance(text_prompts, list) else 1
                target_sizes = [[orig_height, orig_width]] * n
            elif isinstance(original_sizes, self.torch.Tensor):
                target_sizes = original_sizes.tolist()
            else:
                target_sizes = original_sizes

            processed_list = self.processor.post_process_instance_segmentation(
                outputs, threshold=conf_threshold, mask_threshold=0.5, target_sizes=target_sizes
            )

            # Merge results from all batch items (one per text prompt)
            masks = []
            for processed in processed_list:
                masks.extend(self._process_sam3_outputs(processed, conf_threshold, orig_width, orig_height))
        else:
            # Original SAM or SAM3 with visual prompts only
            # SAM outputs:
            # - pred_masks: (batch, num_multimask_outputs, H, W) - typically (1, 3, H, W)
            # - iou_scores: (batch, num_multimask_outputs) - typically (1, 3)
            masks = self._process_sam_outputs(outputs, conf_threshold, orig_width, orig_height)

        # Create result with metadata
        result = VisionResult(
            instances=masks,
            meta={
                "model_id": self.model_id,
                "mode": "zeroshot",
                "threshold": conf_threshold,
                "image_size": [orig_width, orig_height],
                "backend": "transformers",
                "mask_format": "polygon" if self.use_polygon else ("rle" if self.use_rle else "binary"),
                "num_prompts": (1 if text_prompts else 0)
                + (len(point_prompts) if point_prompts else 0)
                + (len(box_prompts) if box_prompts else 0),
                "masks_per_prompt": len(masks),
                "prompts": {
                    "text": text_prompts if text_prompts else None,
                    "points": point_prompts,
                    "boxes": box_prompts,
                },
                "input_path": input_path,
            },
        )

        logger.info(f"Generated {len(masks)} masks " f"(threshold={conf_threshold:.2f})")

        return result

    def _process_sam3_outputs(
        self, processed: dict[str, Any], threshold: float, orig_width: int, orig_height: int
    ) -> list[Instance]:
        """Process SAM3 post-processed outputs (from post_process_instance_segmentation).

        SAM3 with text prompts returns different format than visual-only SAM.

        Args:
            processed: Post-processed output dict with 'masks', 'boxes', 'scores'
            threshold: Score threshold (already applied by post-processor)
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            List of Instance objects
        """
        masks = []

        # SAM3 format: {'masks': Tensor(N, H, W), 'boxes': Tensor(N, 4), 'scores': Tensor(N)}
        pred_masks = processed.get("masks", [])
        pred_boxes = processed.get("boxes", [])
        pred_scores = processed.get("scores", [])

        if not hasattr(pred_masks, "__len__") or len(pred_masks) == 0:
            return masks

        # Convert to numpy
        if isinstance(pred_masks, self.torch.Tensor):
            pred_masks = pred_masks.cpu().numpy()
        if isinstance(pred_boxes, self.torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, self.torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()

        # Process each detected instance
        for idx in range(len(pred_masks)):
            score = float(pred_scores[idx]) if len(pred_scores) > idx else 1.0

            # Get binary mask
            binary_mask = pred_masks[idx] > 0.5 if pred_masks[idx].dtype == float else pred_masks[idx].astype(bool)

            # Skip empty masks
            if not binary_mask.any():
                continue

            # Convert to desired format (polygon/RLE/binary)
            mask_data = self._convert_mask_format(binary_mask)

            # Use provided bbox or compute from mask
            if len(pred_boxes) > idx:
                bbox = tuple(float(x) for x in pred_boxes[idx])
            else:
                bbox = self._mask_to_bbox(binary_mask)

            # Compute area
            area = int(binary_mask.sum())

            # SAM3 text prompts are class-agnostic
            masks.append(
                Instance(
                    mask=mask_data,
                    score=score,
                    label=0,  # Class-agnostic
                    label_name="object",
                    bbox=bbox,
                    is_stuff=None,  # Not applicable for text-based segmentation
                    area=area,
                )
            )

        # Sort by score (descending)
        masks.sort(key=lambda m: m.score, reverse=True)

        return masks

    def _process_sam_outputs(self, outputs: Any, threshold: float, orig_width: int, orig_height: int) -> list[Instance]:
        """Process SAM model outputs into Instance objects.

        SAM returns multiple mask predictions per prompt, each with an IoU score.
        This method converts them to MATA's Instance format.

        For multiple box prompts, SAM returns:
        - pred_masks: (batch, num_boxes, num_masks_per_box, H, W)
        - iou_scores: (batch, num_boxes, num_masks_per_box)

        Args:
            outputs: SAM model outputs with pred_masks and iou_scores
            threshold: IoU score threshold for filtering
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            List of Instance objects (typically 3 per prompt)
        """
        masks = []

        # Extract predictions - squeeze batch dimension
        pred_masks = outputs.pred_masks.squeeze(0).cpu().numpy()
        iou_scores = outputs.iou_scores.squeeze(0).cpu().numpy()

        # Handle different output shapes based on number of prompts
        # Single box: (3, H, W) masks and (3,) scores
        # Multiple boxes: (num_boxes, 3, H, W) masks and (num_boxes, 3) scores
        if pred_masks.ndim == 2:
            # Single mask result: (H, W) -> (1, H, W)
            pred_masks = pred_masks[np.newaxis, ...]
            iou_scores = iou_scores.reshape(1)
        elif pred_masks.ndim == 3:
            # Single box with 3 masks: (3, H, W)
            # Already correct format
            pass
        elif pred_masks.ndim == 4:
            # Multiple boxes: (num_boxes, 3, H, W) -> flatten to (num_boxes * 3, H, W)
            num_boxes, num_masks_per_box = pred_masks.shape[:2]
            pred_masks = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1])
            iou_scores = iou_scores.reshape(-1)

        # Process each mask
        for idx in range(len(pred_masks)):
            iou_score = float(iou_scores[idx])

            # Skip if below threshold
            if iou_score < threshold:
                continue

            # Get binary mask (threshold at 0.0 for SAM logits)
            binary_mask = pred_masks[idx] > 0.0

            # Skip empty masks
            if not binary_mask.any():
                continue

            # Resize mask to original image size if needed
            if binary_mask.shape != (orig_height, orig_width):
                from PIL import Image as PILImage

                mask_pil = PILImage.fromarray(binary_mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((orig_width, orig_height), PILImage.NEAREST)
                binary_mask = np.array(mask_pil) > 0

            # Convert to desired format (polygon/RLE/binary)
            mask_data = self._convert_mask_format(binary_mask)

            # Compute bounding box from mask
            bbox = self._mask_to_bbox(binary_mask)

            # Compute area
            area = int(binary_mask.sum())

            # SAM is class-agnostic: label=0, label_name="object", is_stuff=None
            masks.append(
                Instance(
                    mask=mask_data,
                    score=iou_score,  # IoU prediction score
                    label=0,  # Class-agnostic
                    label_name="object",  # Generic label
                    bbox=bbox,
                    is_stuff=None,  # Not applicable for zero-shot
                    area=area,
                )
            )

        # Sort by IoU score (descending)
        masks.sort(key=lambda m: m.score, reverse=True)

        return masks

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
            # Convert to polygon format
            from mata.core.mask_utils import binary_mask_to_polygon

            polygons = binary_mask_to_polygon(binary_mask, tolerance=self.polygon_tolerance, min_area=10)
            # Return first polygon if single object, or list if multiple contours
            return polygons[0] if len(polygons) == 1 else polygons

        elif self.use_rle and PYCOCOTOOLS_AVAILABLE:
            # Convert to RLE format
            binary_mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
            rle = _mask_utils.encode(binary_mask_fortran)
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        else:
            # Return binary mask as-is
            return binary_mask

    def _mask_to_bbox(self, binary_mask: np.ndarray) -> tuple[float, float, float, float] | None:
        """Compute bounding box from binary mask.

        Args:
            binary_mask: Binary mask array (H, W)

        Returns:
            Bounding box (x1, y1, x2, y2) or None if mask is empty
        """
        # Find non-zero pixels
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (float(x_min), float(y_min), float(x_max), float(y_max))

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "huggingface_sam",
            "task": "segment",
            "model_id": self.model_id,
            "mode": "zeroshot",
            "device": str(self.device),
            "threshold": self.threshold,
            "use_rle": self.use_rle,
            "use_polygon": self.use_polygon,
            "backend": "transformers",
            "class_agnostic": True,
        }

    def segment(
        self,
        image: str | Path | Image.Image | np.ndarray,
        box_prompts: list[tuple[float, float, float, float]] | None = None,
        point_prompts: list[tuple[float, float, int]] | None = None,
        text_prompts: str | list[str] | None = None,
        threshold: float | None = None,
        mode: str = "boxes",
        **kwargs: Any,
    ) -> VisionResult:
        """Segment using prompts (PromptBoxes node interface).

        This method provides compatibility with the PromptBoxes graph node,
        which expects a segment() method. It wraps predict() with the appropriate
        parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            box_prompts: List of box prompts as (x1, y1, x2, y2) tuples
            point_prompts: List of point prompts as (x, y, label) tuples
            text_prompts: Text description(s) for SAM3
            threshold: Optional IoU score threshold [0.0, 1.0]
            mode: Prompt mode ("boxes", "points", "text") - for compatibility
            **kwargs: Additional arguments passed to predict()

        Returns:
            VisionResult with segmentation masks
        """
        return self.predict(
            image=image,
            text_prompts=text_prompts,
            point_prompts=point_prompts,
            box_prompts=box_prompts,
            threshold=threshold,
            **kwargs,
        )
