"""Multi-modal pipeline adapter for chaining detection and segmentation.

Implements GroundingDINO→SAM3 pipeline for text→bbox→mask workflows,
enabling text-prompt-based instance segmentation.
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


class GroundingDINOSAMPipeline(PyTorchBaseAdapter):
    """GroundingDINO→SAM3 pipeline for text-prompted instance segmentation.

    Combines zero-shot object detection (GroundingDINO) with zero-shot
    instance segmentation (SAM3) to enable text→bbox→mask workflows.

    Pipeline steps:
    1. GroundingDINO detects objects from text prompts
    2. Extract bounding boxes from detections
    3. SAM3 generates precise masks using boxes as prompts
    4. Merge detection + segmentation into unified VisionResult

    Attributes:
        task: Always "pipeline"
        detector: HuggingFaceZeroShotDetectAdapter instance
        segmenter: HuggingFaceSAMAdapter instance
        detector_model_id: GroundingDINO model identifier
        sam_model_id: SAM3 model identifier

    Examples:
        >>> # Create pipeline
        >>> pipeline = GroundingDINOSAMPipeline(
        ...     detector_model_id="IDEA-Research/grounding-dino-tiny",
        ...     sam_model_id="facebook/sam-vit-base"
        ... )
        >>>
        >>> # Text → bbox → mask
        >>> result = pipeline.predict(
        ...     "image.jpg",
        ...     text_prompts="cat . dog"
        ... )
        >>> for inst in result.instances:
        ...     print(f"{inst.label_name}: bbox={inst.bbox}, has_mask={inst.mask is not None}")
        >>>
        >>> # Batch processing
        >>> results = pipeline.predict(
        ...     ["img1.jpg", "img2.jpg"],
        ...     text_prompts="person . car"
        ... )
    """

    task = "pipeline"

    def __init__(
        self,
        detector_model_id: str,
        sam_model_id: str,
        device: str = "auto",
        detection_threshold: float = 0.3,
        segmentation_threshold: float = 0.5,
        max_batch_size: int = 8,
    ) -> None:
        """Initialize GroundingDINO→SAM3 pipeline.

        Args:
            detector_model_id: HuggingFace ID for GroundingDINO model
                Example: "IDEA-Research/grounding-dino-tiny"
            sam_model_id: HuggingFace ID for SAM model
                Example: "facebook/sam-vit-base" or "facebook/sam3"
            device: Device ("cuda", "cpu", or "auto")
            detection_threshold: Detection confidence threshold [0.0, 1.0]
            segmentation_threshold: Segmentation confidence threshold [0.0, 1.0]
            max_batch_size: Maximum batch size (default: 8)

        Raises:
            ImportError: If required dependencies missing
            ModelLoadError: If model loading fails
        """
        # Initialize base class
        super().__init__(device=device, threshold=detection_threshold)

        self.detector_model_id = detector_model_id
        self.sam_model_id = sam_model_id
        self.detection_threshold = detection_threshold
        self.segmentation_threshold = segmentation_threshold
        self.max_batch_size = max_batch_size

        # Lazy-load adapters (imported when first predict() call)
        self.detector = None
        self.segmenter = None

        logger.info(f"Initialized GroundingDINO→SAM pipeline: " f"detector={detector_model_id}, sam={sam_model_id}")

    def _load_adapters(self) -> None:
        """Lazy-load detector and segmenter adapters.

        Raises:
            ModelLoadError: If adapter loading fails
        """
        if self.detector is not None and self.segmenter is not None:
            return  # Already loaded

        try:
            # Import adapters (lazy)
            from .huggingface_sam_adapter import HuggingFaceSAMAdapter
            from .huggingface_zeroshot_detect_adapter import HuggingFaceZeroShotDetectAdapter

            # Load detector
            if self.detector is None:
                logger.info(f"Loading detector: {self.detector_model_id}")
                self.detector = HuggingFaceZeroShotDetectAdapter(
                    model_id=self.detector_model_id,
                    device=self.device,
                    threshold=self.detection_threshold,
                    max_batch_size=self.max_batch_size,
                )

            # Load segmenter
            if self.segmenter is None:
                logger.info(f"Loading segmenter: {self.sam_model_id}")
                self.segmenter = HuggingFaceSAMAdapter(
                    model_id=self.sam_model_id,
                    device=self.device,
                    threshold=self.segmentation_threshold,
                )

            logger.info("Pipeline adapters loaded successfully")

        except Exception as e:
            raise ModelLoadError(
                f"{self.detector_model_id} + {self.sam_model_id}",
                f"Failed to load pipeline adapters: {type(e).__name__}: {str(e)}",
            ) from e

    def predict(
        self,
        image: ImageInput | list[ImageInput],
        text_prompts: str | list[str],
        detection_threshold: float | None = None,
        segmentation_threshold: float | None = None,
        **kwargs: Any,
    ) -> VisionResult | list[VisionResult]:
        """Run text→bbox→mask pipeline.

        Args:
            image: Input image(s)
                - Single: str path, Path, PIL.Image, or numpy array
                - Batch: List of any of the above
            text_prompts: Text description(s) of objects to detect/segment
                Format: "cat . dog . person" (space-dot separated)
            detection_threshold: Detection confidence override (optional)
            segmentation_threshold: Segmentation confidence override (optional)
            **kwargs: Additional pipeline parameters

        Returns:
            VisionResult with instances containing both bbox and mask,
            or List[VisionResult] for batch processing

        Raises:
            InvalidInputError: If text_prompts is missing or invalid

        Examples:
            >>> # Single image
            >>> result = pipeline.predict(
            ...     "image.jpg",
            ...     text_prompts="cat . dog"
            ... )
            >>>
            >>> # Batch images
            >>> results = pipeline.predict(
            ...     ["img1.jpg", "img2.jpg"],
            ...     text_prompts="person . car . bicycle"
            ... )
        """
        # Validate text prompts
        if not text_prompts:
            raise InvalidInputError(
                "text_prompts required for pipeline. "
                "Example: text_prompts='cat . dog' or text_prompts=['cat', 'dog']"
            )

        # Lazy-load adapters
        self._load_adapters()

        # Use threshold overrides if provided
        det_threshold = detection_threshold if detection_threshold is not None else self.detection_threshold
        seg_threshold = segmentation_threshold if segmentation_threshold is not None else self.segmentation_threshold

        # Determine if batch processing
        is_batch = isinstance(image, list)

        if is_batch:
            # Batch processing - process sequentially per-image
            logger.info(f"Processing pipeline batch of {len(image)} images")
            return self._predict_batch(image, text_prompts, det_threshold, seg_threshold, **kwargs)
        else:
            # Single image processing
            return self._predict_single(image, text_prompts, det_threshold, seg_threshold, **kwargs)

    def _predict_single(
        self,
        image: ImageInput,
        text_prompts: str | list[str],
        detection_threshold: float,
        segmentation_threshold: float,
        **kwargs: Any,
    ) -> VisionResult:
        """Run pipeline on single image.

        Args:
            image: Single input image
            text_prompts: Text prompts for detection
            detection_threshold: Detection confidence threshold
            segmentation_threshold: Segmentation confidence threshold
            **kwargs: Additional parameters

        Returns:
            VisionResult with instances containing bbox and mask
        """
        # Step 1: Detect objects with GroundingDINO
        logger.debug(f"Running detection with prompts: {text_prompts}")
        detect_result = self.detector.predict(image, text_prompts=text_prompts, threshold=detection_threshold)

        # Check if any detections found
        if len(detect_result.instances) == 0:
            logger.warning(
                f"No objects detected with threshold={detection_threshold}. "
                f"Skipping segmentation. Returning detection-only result."
            )
            return VisionResult(
                instances=[],
                meta={
                    "pipeline": "grounding_sam",
                    "detector_model": self.detector_model_id,
                    "sam_model": self.sam_model_id,
                    "detection_threshold": detection_threshold,
                    "segmentation_threshold": segmentation_threshold,
                    "text_prompts": text_prompts,
                    "detections_found": 0,
                    "note": "No detections found - segmentation skipped",
                    "input_path": detect_result.meta.get("input_path"),
                },
            )

        logger.debug(f"Found {len(detect_result.instances)} detections")

        # Step 2: Extract box prompts from detections
        box_prompts = [inst.bbox for inst in detect_result.instances if inst.bbox is not None]
        if len(box_prompts) == 0:
            logger.warning("Detections have no bboxes - cannot segment")
            return detect_result  # Return detection-only result

        # Step 3: Segment with SAM using box prompts
        logger.debug(f"Running SAM segmentation with {len(box_prompts)} box prompts")
        segment_result = self.segmenter.predict(
            image,
            box_prompts=box_prompts,
            box_labels=[1] * len(box_prompts),  # All positive prompts
            threshold=segmentation_threshold,
        )

        # Step 4: Merge detection and segmentation results
        merged_instances = self._merge_results(detect_result.instances, segment_result.instances, text_prompts)

        return VisionResult(
            instances=merged_instances,
            meta={
                "pipeline": "grounding_sam",
                "detector_model": self.detector_model_id,
                "sam_model": self.sam_model_id,
                "detection_threshold": detection_threshold,
                "segmentation_threshold": segmentation_threshold,
                "text_prompts": text_prompts,
                "detections_found": len(detect_result.instances),
                "masks_generated": len(segment_result.instances),
                "merged_instances": len(merged_instances),
                "input_path": detect_result.meta.get("input_path"),
            },
        )

    def _predict_batch(
        self,
        images: list[ImageInput],
        text_prompts: str | list[str],
        detection_threshold: float,
        segmentation_threshold: float,
        **kwargs: Any,
    ) -> list[VisionResult]:
        """Run pipeline on batch of images.

        Processes each image sequentially for v1.5 simplicity.

        Args:
            images: List of input images
            text_prompts: Text prompts (same for all images)
            detection_threshold: Detection confidence threshold
            segmentation_threshold: Segmentation confidence threshold
            **kwargs: Additional parameters

        Returns:
            List of VisionResult objects
        """
        # Check batch size warning
        if len(images) > self.max_batch_size:
            logger.warning(
                f"Batch size {len(images)} exceeds recommended max {self.max_batch_size}. "
                f"Processing may be slow or cause memory issues."
            )

        # Process each image sequentially
        results = []
        for i, img in enumerate(images):
            logger.debug(f"Processing pipeline for image {i+1}/{len(images)}")
            result = self._predict_single(img, text_prompts, detection_threshold, segmentation_threshold, **kwargs)
            results.append(result)

        return results

    def _merge_results(
        self, detections: list[Instance], masks: list[Instance], text_prompts: str | list[str]
    ) -> list[Instance]:
        """Merge detection and segmentation instances.

        Assumes SAM returns masks in same order as input box prompts (by index).
        Each detection gets matched with corresponding SAM mask.

        Args:
            detections: Detection instances (with bbox, label, score)
            masks: Segmentation instances (with mask, score)
            text_prompts: Original text prompts for label names

        Returns:
            List of merged Instance objects with both bbox and mask
        """
        merged: list[Instance] = []

        # SAM can return multiple masks per prompt (usually 3)
        # We'll take the first (highest IoU) mask for each detection
        mask_idx = 0

        for det_idx, detection in enumerate(detections):
            # Check if we have masks available
            if mask_idx >= len(masks):
                logger.warning(f"Not enough masks for detection {det_idx}. " f"Using detection-only instance.")
                # Keep detection without mask
                merged.append(detection)
                continue

            # Get corresponding mask (first of the 3 SAM returns per prompt)
            # SAM typically returns 3 masks per box prompt, we take the first (best)
            mask_instance = masks[mask_idx]

            # Create merged instance with both bbox and mask
            merged_inst = Instance(
                bbox=detection.bbox,
                mask=mask_instance.mask,
                score=(detection.score + mask_instance.score) / 2,  # Average scores
                label=detection.label,
                label_name=detection.label_name,
                area=mask_instance.area,
                is_stuff=False,  # Detections are always instances
            )
            merged.append(merged_inst)

            # Move to next mask (skip the other 2 masks SAM returns per prompt)
            mask_idx += 1

        logger.debug(f"Merged {len(detections)} detections with {min(len(detections), len(masks))} masks")

        return merged

    def info(self) -> dict[str, Any]:
        """Get pipeline metadata.

        Returns:
            Dictionary with pipeline information
        """
        return {
            "name": "GroundingDINOSAMPipeline",
            "task": self.task,
            "detector_model": self.detector_model_id,
            "sam_model": self.sam_model_id,
            "device": str(self.device),
            "detection_threshold": self.detection_threshold,
            "segmentation_threshold": self.segmentation_threshold,
            "max_batch_size": self.max_batch_size,
            "pipeline_type": "grounding_sam",
            "backend": "huggingface-transformers",
        }
