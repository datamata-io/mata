"""HuggingFace adapter for instance and panoptic segmentation.

Supports automatic detection and loading of transformer-based segmentation
models from HuggingFace Hub, including:
- Mask2Former (instance and panoptic)
- MaskFormer (panoptic)
- OneFormer (universal segmentation)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import ModelLoadError, UnsupportedModelError
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
            from transformers import (
                AutoConfig,
                AutoImageProcessor,
                DetrForSegmentation,
                Mask2FormerForUniversalSegmentation,
                MaskFormerForInstanceSegmentation,
                OneFormerForUniversalSegmentation,
                OneFormerProcessor,
                SegformerForSemanticSegmentation,
            )

            _transformers = {
                "AutoImageProcessor": AutoImageProcessor,
                "OneFormerProcessor": OneFormerProcessor,
                "Mask2FormerForUniversalSegmentation": Mask2FormerForUniversalSegmentation,
                "MaskFormerForInstanceSegmentation": MaskFormerForInstanceSegmentation,
                "OneFormerForUniversalSegmentation": OneFormerForUniversalSegmentation,
                "DetrForSegmentation": DetrForSegmentation,
                "SegformerForSemanticSegmentation": SegformerForSemanticSegmentation,
                "AutoConfig": AutoConfig,
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


class HuggingFaceSegmentAdapter(PyTorchBaseAdapter):
    """HuggingFace segmentation adapter with auto-detection.

    Automatically detects and loads the appropriate segmentation model from
    HuggingFace Hub. Supports both instance and panoptic segmentation modes.

    For panoptic segmentation, the adapter automatically detects "stuff" vs "instance"
    classes using the model's is_thing_map configuration and sets the is_stuff field
    accordingly.

    Mask Format:
    - By default, returns RLE-encoded masks (compact, JSON-serializable)
    - Requires pycocotools: pip install datamata[segmentation]
    - Falls back to binary numpy arrays if pycocotools not available

    Segmentation Modes:
    - "auto": Automatically detect from model config (default)
    - "instance": Force instance segmentation mode
    - "panoptic": Force panoptic segmentation mode

    Examples:
        >>> # Instance segmentation (COCO instances)
        >>> segmenter = HuggingFaceSegmentAdapter(
        ...     "facebook/mask2former-swin-tiny-coco-instance"
        ... )
        >>> result = segmenter.predict("image.jpg", threshold=0.5)
        >>> print(f"Found {len(result.masks)} instances")
        >>>
        >>> # Panoptic segmentation (instances + stuff)
        >>> segmenter = HuggingFaceSegmentAdapter(
        ...     "facebook/mask2former-swin-tiny-coco-panoptic",
        ...     segment_mode="panoptic"
        ... )
        >>> result = segmenter.predict("image.jpg")
        >>> instances = result.get_instances()  # Countable objects
        >>> stuff = result.get_stuff()          # Uncountable regions
        >>>
        >>> # Semantic segmentation (Segformer)
        >>> segmenter = HuggingFaceSegmentAdapter(
        ...     "nvidia/segformer-b0-finetuned-ade-512-512"
        ... )
        >>> result = segmenter.predict("image.jpg")
        >>> print(f"Found {len(result.masks)} semantic regions")
        >>>
        >>> # DETR panoptic segmentation
        >>> segmenter = HuggingFaceSegmentAdapter(
        ...     "facebook/detr-resnet-101-panoptic"
        ... )
        >>>
        >>> # OneFormer (universal segmentation)
        >>> segmenter = HuggingFaceSegmentAdapter(
        ...     "shi-labs/oneformer_coco_swin_large"
        ... )
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.5,
        segment_mode: str = "auto",
        use_rle: bool = True,
        use_polygon: bool = False,
        polygon_tolerance: float = 2.0,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize HuggingFace segmentation adapter.

        Args:
            model_id: HuggingFace model ID
                Examples:
                - "facebook/mask2former-swin-tiny-coco-instance"
                - "facebook/mask2former-swin-tiny-coco-panoptic"
                - "facebook/maskformer-swin-base-ade"
                - "facebook/detr-resnet-101-panoptic"
                - "shi-labs/oneformer_coco_swin_large"
                - "nvidia/segformer-b0-finetuned-ade-512-512"
                - "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            device: Device ("cuda", "cpu", or "auto")
            threshold: Segmentation confidence threshold [0.0, 1.0]
            segment_mode: Segmentation mode:
                - "auto": Detect from model config (default)
                - "instance": Force instance segmentation
                - "panoptic": Force panoptic segmentation
                - "semantic": Force semantic segmentation
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
            id2label: Optional custom label mapping (overrides model config)

        Raises:
            ImportError: If transformers is not installed
            ModelLoadError: If model loading fails
            UnsupportedModelError: If model architecture is not supported
        """
        # Initialize base class (handles torch import and device setup)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Validate segment_mode
        if segment_mode not in ("auto", "instance", "panoptic", "semantic", "zeroshot"):
            raise ValueError(
                f"segment_mode must be 'auto', 'instance', 'panoptic', 'semantic', or 'zeroshot', got '{segment_mode}'"
            )

        # Lazy import transformers
        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers is required for HuggingFace adapter. " "Install with: pip install transformers"
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
        self.segment_mode = segment_mode

        # Load model and processor
        try:
            logger.info(f"Loading segmentation model: {model_id}")

            # Load config to detect architecture and capabilities
            AutoConfig = transformers["AutoConfig"]  # noqa: N806
            config = AutoConfig.from_pretrained(model_id)

            # Detect architecture and mode
            architecture, detected_mode = self._detect_architecture(config, model_id)
            logger.info(f"Detected architecture: {architecture}, mode: {detected_mode}")

            # Use detected mode if auto, otherwise use specified mode
            if segment_mode == "auto":
                self.active_mode = detected_mode
            else:
                # Validate mode compatibility with architecture
                # SAM only supports zeroshot mode
                if architecture == "sam" and segment_mode != "zeroshot":
                    raise ValueError(
                        f"SAM architecture only supports 'zeroshot' mode, "
                        f"but '{segment_mode}' was requested. "
                        f"Use HuggingFaceSAMAdapter directly or set segment_mode='zeroshot'."
                    )

                # Segformer only supports semantic segmentation
                if architecture == "segformer" and segment_mode != "semantic":
                    raise ValueError(
                        f"Segformer architecture only supports 'semantic' mode, "
                        f"but '{segment_mode}' was requested. "
                        f"Either use segment_mode='semantic' or segment_mode='auto' for Segformer models."
                    )

                self.active_mode = segment_mode
                if detected_mode != segment_mode:
                    warnings.warn(
                        f"Model was trained for {detected_mode} segmentation, "
                        f"but forcing {segment_mode} mode. Results may be suboptimal.",
                        UserWarning,
                        stacklevel=2,
                    )

            logger.info(f"Using segmentation mode: {self.active_mode}")

            # Suppress noisy third-party output during model loading
            from mata.core.logging import suppress_third_party_logs

            with suppress_third_party_logs():

                # Load processor (handles preprocessing)
                # Use OneFormerProcessor for OneFormer models, AutoImageProcessor for others
                if architecture == "oneformer":
                    OneFormerProcessor = transformers["OneFormerProcessor"]  # noqa: N806
                    self.processor = OneFormerProcessor.from_pretrained(model_id)
                else:
                    AutoImageProcessor = transformers["AutoImageProcessor"]  # noqa: N806
                    self.processor = AutoImageProcessor.from_pretrained(model_id)

                # Load model with appropriate class
                if architecture == "mask2former":
                    Mask2FormerForUniversalSegmentation = transformers[  # noqa: N806
                        "Mask2FormerForUniversalSegmentation"
                    ]
                    self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                        model_id, ignore_mismatched_sizes=False
                    )
                elif architecture == "maskformer":
                    MaskFormerForInstanceSegmentation = transformers["MaskFormerForInstanceSegmentation"]  # noqa: N806
                    self.model = MaskFormerForInstanceSegmentation.from_pretrained(model_id)
                elif architecture == "detr":
                    DetrForSegmentation = transformers["DetrForSegmentation"]  # noqa: N806
                    self.model = DetrForSegmentation.from_pretrained(model_id)
                elif architecture == "oneformer":
                    OneFormerForUniversalSegmentation = transformers["OneFormerForUniversalSegmentation"]  # noqa: N806
                    self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
                elif architecture == "segformer":
                    SegformerForSemanticSegmentation = transformers["SegformerForSemanticSegmentation"]  # noqa: N806
                    self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
                else:
                    raise UnsupportedModelError(
                        f"Architecture '{architecture}' not supported for segmentation. "
                        f"Supported: mask2former, maskformer, detr, oneformer, segformer"
                    )

            # Move model to device and set eval mode
            self.model = self.model.to(self.device).eval()

            # Extract label mapping and is_thing_map for panoptic
            if self.id2label is None:
                # Use model config labels
                if hasattr(config, "id2label") and config.id2label:
                    self.id2label = {int(k): v for k, v in config.id2label.items()}
                else:
                    logger.warning("Model config has no id2label mapping. Using numeric labels.")
                    self.id2label = {}

            # Get is_thing_map for panoptic segmentation
            self.is_thing_map = None
            if self.active_mode == "panoptic":
                if hasattr(config, "is_thing_map") and config.is_thing_map:
                    # is_thing_map: dict mapping label_id -> bool (True=instance, False=stuff)
                    self.is_thing_map = {int(k): v for k, v in config.is_thing_map.items()}
                    logger.info(f"Loaded is_thing_map with {len(self.is_thing_map)} classes")
                else:
                    warnings.warn(
                        "Model config missing 'is_thing_map' for panoptic segmentation. "
                        "All classes will be treated as instances (is_stuff=False).",
                        UserWarning,
                        stacklevel=2,
                    )

            logger.info(
                f"Loaded {architecture} model on {self.device} "
                f"(mode={self.active_mode}, labels={len(self.id2label)})"
            )

        except Exception as e:
            if isinstance(e, (ImportError, UnsupportedModelError)):
                raise
            raise ModelLoadError(model_id, f"Failed to load segmentation model: {type(e).__name__}: {str(e)}")

    def _detect_architecture(self, config: Any, model_id: str) -> tuple[str, str]:
        """Detect model architecture and segmentation mode from config.

        Args:
            config: Model configuration object
            model_id: HuggingFace model ID

        Returns:
            Tuple of (architecture, mode)
            - architecture: "mask2former", "maskformer", "detr", "oneformer", "segformer", or "sam"
            - mode: "instance", "panoptic", "semantic", or "zeroshot"
        """
        model_type = getattr(config, "model_type", "").lower()

        # Detect architecture from model_type
        if "sam" in model_type:
            architecture = "sam"
        elif "mask2former" in model_type:
            architecture = "mask2former"
        elif "maskformer" in model_type and "mask2former" not in model_type:
            architecture = "maskformer"
        elif "detr" in model_type:
            architecture = "detr"
        elif "oneformer" in model_type:
            architecture = "oneformer"
        elif "segformer" in model_type:
            architecture = "segformer"
        else:
            # Fallback: detect from model ID
            model_id_lower = model_id.lower()
            if "sam" in model_id_lower:
                architecture = "sam"
            elif "mask2former" in model_id_lower:
                architecture = "mask2former"
            elif "maskformer" in model_id_lower:
                architecture = "maskformer"
            elif "detr" in model_id_lower and "panoptic" in model_id_lower:
                architecture = "detr"
            elif "oneformer" in model_id_lower:
                architecture = "oneformer"
            elif "segformer" in model_id_lower:
                architecture = "segformer"
            else:
                raise UnsupportedModelError(
                    f"Could not detect segmentation architecture from model_type='{model_type}' "
                    f"or model_id='{model_id}'. Supported architectures: mask2former, maskformer, detr, oneformer, segformer, sam"
                )

        # Detect mode from model ID or config
        model_id_lower = model_id.lower()
        if architecture == "sam":
            # SAM is zero-shot segmentation
            mode = "zeroshot"
        elif architecture == "segformer":
            # Segformer is semantic segmentation
            mode = "semantic"
        elif "panoptic" in model_id_lower:
            mode = "panoptic"
        elif "instance" in model_id_lower:
            mode = "instance"
        elif "semantic" in model_id_lower:
            mode = "semantic"
        elif hasattr(config, "is_thing_map") and config.is_thing_map:
            # Has is_thing_map → likely panoptic
            mode = "panoptic"
        else:
            # Default to instance
            mode = "instance"

        return architecture, mode

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, threshold: float | None = None, **kwargs: Any
    ) -> VisionResult:
        """Run segmentation on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional confidence threshold override
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            VisionResult with segmentation mask instances

        Raises:
            InvalidInputError: If image is invalid

        Examples:
            >>> result = segmenter.predict("image.jpg", threshold=0.7)
            >>> print(f"Segments: {len(result.instances)}")
            >>>
            >>> # Filter by score
            >>> high_conf = result.filter_by_score(0.8)
            >>>
            >>> # Get instances only (panoptic mode)
            >>> instances = result.get_instances()
            >>> stuff = result.get_stuff()
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)
        orig_width, orig_height = pil_image.size

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Detect architecture to determine if task_inputs is needed
        architecture = self._detect_architecture(self.model.config, self.model_id)[0]

        # Preprocess image
        # OneFormer uses task_inputs parameter to specify the segmentation mode
        if architecture == "oneformer":
            task_input = self.active_mode  # "semantic", "instance", or "panoptic"
            inputs = self.processor(images=pil_image, task_inputs=[task_input], return_tensors="pt")
            log_task = f" (task={task_input})"
        else:
            # Other models don't use task_inputs
            inputs = self.processor(images=pil_image, return_tensors="pt")
            log_task = ""

        # Move tensors to device
        # Note: Only move tensor values, not metadata like task_inputs
        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, self.torch.Tensor):
                device_inputs[k] = v.to(self.device)
            else:
                # Keep non-tensor values (e.g., task_inputs list for OneFormer)
                device_inputs[k] = v

        # Run inference
        logger.info(f"Running {self.active_mode} segmentation on {orig_width}x{orig_height} image{log_task}")

        with self.torch.no_grad():
            outputs = self.model(**device_inputs)

        # Post-process outputs
        if self.active_mode == "semantic":
            # Semantic segmentation post-processing
            target_sizes = [(orig_height, orig_width)]
            results = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
        elif self.active_mode == "panoptic":
            # Panoptic segmentation post-processing
            target_sizes = [(orig_height, orig_width)]
            results = self.processor.post_process_panoptic_segmentation(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]
        else:
            # Instance segmentation post-processing
            target_sizes = [(orig_height, orig_width)]
            results = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]

        # Convert to Instance objects
        masks = self._process_results(results, conf_threshold, orig_width, orig_height)

        # Create result
        result = VisionResult(
            instances=masks,
            meta={
                "model_id": self.model_id,
                "mode": self.active_mode,
                "threshold": conf_threshold,
                "image_size": [orig_width, orig_height],
                "backend": "transformers",
                "mask_format": "rle" if self.use_rle else "binary",
                "input_path": input_path,
            },
        )

        logger.info(
            f"Found {len(masks)} segments "
            f"({len(result.get_instances())} instances, {len(result.get_stuff())} stuff)"
        )

        return result

    def _process_results(
        self, results: dict[str, Any], threshold: float, orig_width: int, orig_height: int
    ) -> list[Instance]:
        """Process model outputs into Instance objects.

        Args:
            results: Post-processed results from processor
            threshold: Confidence threshold
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            List of Instance objects
        """
        masks = []

        if self.active_mode == "semantic":
            # Semantic segmentation format:
            # results: Tensor of shape (H, W) with class labels per pixel

            if isinstance(results, self.torch.Tensor):
                segmentation_map = results.cpu().numpy()
            else:
                segmentation_map = results

            # Extract unique labels (excluding background if label 0)
            unique_labels = np.unique(segmentation_map)

            for label_id in unique_labels:
                # Skip background (typically label 0)
                if label_id == 0:
                    continue

                # Extract binary mask for this class
                binary_mask = segmentation_map == label_id

                # Convert to desired format (polygon/RLE/binary)
                mask_data = self._convert_mask_format(binary_mask)

                # Compute bounding box from mask
                bbox = self._mask_to_bbox(binary_mask)

                # Get label name
                label_name = self.id2label.get(int(label_id), f"class_{label_id}")

                # Compute area
                area = int(binary_mask.sum())

                # Semantic segmentation: all regions are "stuff" (not countable instances)
                masks.append(
                    Instance(
                        mask=mask_data,
                        score=1.0,  # Semantic segmentation doesn't have confidence scores
                        label=int(label_id),
                        label_name=label_name,
                        bbox=bbox,
                        is_stuff=True,  # Semantic regions are "stuff"
                        area=area,
                    )
                )

        elif self.active_mode == "panoptic":
            # Panoptic segmentation format:
            # results['segmentation']: Tensor of shape (H, W) with segment IDs
            # results['segments_info']: List of dicts with id, label_id, score

            segmentation_map = results["segmentation"].cpu().numpy()
            segments_info = results["segments_info"]

            for segment in segments_info:
                segment_id = segment["id"]
                label_id = segment["label_id"]
                score = segment.get("score", 1.0)  # Panoptic may not have scores

                # Skip if below threshold
                if score < threshold:
                    continue

                # Extract binary mask for this segment
                binary_mask = segmentation_map == segment_id

                # Convert to desired format (polygon/RLE/binary)
                mask_data = self._convert_mask_format(binary_mask)

                # Compute bounding box from mask
                bbox = self._mask_to_bbox(binary_mask)

                # Determine if this is a "thing" (instance) or "stuff" (semantic region)
                is_stuff = None
                if self.is_thing_map is not None:
                    # is_thing_map: True = instance, False = stuff
                    # Invert for is_stuff field
                    is_thing = self.is_thing_map.get(label_id, True)
                    is_stuff = not is_thing

                # Get label name
                label_name = self.id2label.get(label_id, f"class_{label_id}")

                # Compute area
                area = int(binary_mask.sum())

                masks.append(
                    Instance(
                        mask=mask_data,
                        score=score,
                        label=label_id,
                        label_name=label_name,
                        bbox=bbox,
                        is_stuff=is_stuff,
                        area=area,
                    )
                )

        else:
            # Instance segmentation format - two possible formats:
            #
            # Format 1 (Mask2Former/OneFormer style):
            # results["segmentation"]: Tensor of shape (H, W) with instance IDs
            # results["segments_info"]: List of dicts with id, label_id, score
            #
            # Format 2 (DETR style):
            # results['masks']: Tensor of shape (N, H, W) with binary masks
            # results['scores']: Tensor of shape (N,) with confidence scores
            # results['labels']: Tensor of shape (N,) with class labels

            # Check format and process accordingly
            if "segmentation" in results and "segments_info" in results:
                # Format 1: Mask2Former/OneFormer style
                segmentation_map = results["segmentation"].cpu().numpy()
                segments_info = results["segments_info"]

                for segment in segments_info:
                    segment_id = segment["id"]
                    label_id = segment["label_id"]
                    score = segment.get("score", 1.0)

                    # Skip if below threshold
                    if score < threshold:
                        continue

                    # Extract binary mask for this segment
                    binary_mask = segmentation_map == segment_id

                    # Convert to desired format (polygon/RLE/binary)
                    mask_data = self._convert_mask_format(binary_mask)

                    # Compute bounding box from mask
                    bbox = self._mask_to_bbox(binary_mask)

                    # Get label name
                    label_name = self.id2label.get(label_id, f"class_{label_id}")

                    # Compute area
                    area = int(binary_mask.sum())

                    masks.append(
                        Instance(
                            mask=mask_data,
                            score=score,
                            label=label_id,
                            label_name=label_name,
                            bbox=bbox,
                            is_stuff=False,  # All instances
                            area=area,
                        )
                    )

            elif "masks" in results and len(results.get("masks", [])) > 0:
                # Format 2: DETR style
                instance_masks = results["masks"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"].cpu().numpy()

                for idx in range(len(instance_masks)):
                    score = float(scores[idx])
                    label_id = int(labels[idx])

                    # Skip if below threshold
                    if score < threshold:
                        continue

                    # Get binary mask
                    binary_mask = instance_masks[idx]

                    # Convert to desired format (polygon/RLE/binary)
                    mask_data = self._convert_mask_format(binary_mask)

                    # Compute bounding box
                    bbox = self._mask_to_bbox(binary_mask)

                    # Get label name
                    label_name = self.id2label.get(label_id, f"class_{label_id}")

                    # Compute area
                    area = int(binary_mask.sum())

                    # Instance segmentation: is_stuff is always False (or None)
                    masks.append(
                        Instance(
                            mask=mask_data,
                            score=score,
                            label=label_id,
                            label_name=label_name,
                            bbox=bbox,
                            is_stuff=False,  # Instances are always "things"
                            area=area,
                        )
                    )

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
            # For segmentation, we usually return the first/main polygon
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
            "name": "huggingface_segment",
            "task": "segment",
            "model_id": self.model_id,
            "mode": self.active_mode,
            "device": str(self.device),
            "threshold": self.threshold,
            "use_rle": self.use_rle,
            "num_classes": len(self.id2label),
            "backend": "transformers",
        }
