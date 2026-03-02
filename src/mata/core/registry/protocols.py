"""Capability protocols for MATA task types.

This module defines runtime-checkable protocols for all task capabilities in the
MATA graph system. Protocols enable duck-typing with static type checking and
runtime verification, allowing any class implementing the required interface to
serve as a provider for graph nodes.

All protocols are runtime checkable, meaning you can use isinstance() at runtime
to verify that an object implements a protocol:

    >>> if isinstance(my_model, Detector):
    ...     detections = my_model.predict(image)

Each protocol corresponds to a specific vision task capability and defines the
minimum interface required for that task.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Forward references for type hints (avoid circular imports)
# These will be resolved at runtime
if False:  # TYPE_CHECKING would be True, but we avoid import cycles
    import numpy as np

    from mata.core.artifacts.detections import Detections
    from mata.core.artifacts.image import Image
    from mata.core.artifacts.keypoints import Keypoints
    from mata.core.artifacts.masks import Masks
    from mata.core.artifacts.rois import ROIs
    from mata.core.artifacts.tracks import Tracks
    from mata.core.types import ClassifyResult, DepthResult, OCRResult, VisionResult


@runtime_checkable
class Detector(Protocol):
    """Detection capability protocol.

    Providers implementing this protocol can detect objects in images,
    returning bounding boxes with labels and confidence scores.

    Supported models:
    - Object detection: DETR, RT-DETR, YOLO, Faster R-CNN
    - Zero-shot detection: GroundingDINO, OWL-ViT

    Example:
        ```python
        from mata.core.artifacts import Image
        from mata.core.registry.protocols import Detector

        class MyDetector:
            def predict(self, image: Image, **kwargs) -> Detections:
                # Detection logic
                return detections

        detector = MyDetector()
        assert isinstance(detector, Detector)  # Runtime check

        img = Image.from_path("photo.jpg")
        dets = detector.predict(img, threshold=0.5)
        ```

    Args:
        image: Input image artifact
        **kwargs: Model-specific parameters (threshold, nms_iou, etc.)

    Returns:
        Detections artifact with instances and instance_ids
    """

    def predict(self, image: Image, **kwargs) -> Detections:
        """Detect objects in image.

        Args:
            image: Input image artifact
            **kwargs: Model-specific parameters
                - threshold (float): Confidence threshold (typical: 0.5)
                - nms_iou (float): NMS IoU threshold (typical: 0.45)
                - max_detections (int): Maximum number of detections
                - text_prompts (str or List[str]): For zero-shot models

        Returns:
            Detections artifact with bounding boxes, labels, and scores
        """
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Segmentation capability protocol.

    Providers implementing this protocol can segment images, generating pixel-level
    masks for objects or regions. Supports both instance and semantic segmentation,
    as well as prompt-based segmentation (e.g., SAM).

    Supported models:
    - Instance segmentation: Mask2Former, MaskFormer, Mask R-CNN
    - Panoptic segmentation: Mask2Former, Panoptic-DeepLab
    - Prompt-based: SAM, SAM2, SAM3

    Example:
        ```python
        from mata.core.artifacts import Image, Detections
        from mata.core.registry.protocols import Segmenter

        class MySAMSegmenter:
            def segment(self, image: Image, **kwargs) -> Masks:
                # Segmentation logic
                return masks

        segmenter = MySAMSegmenter()
        assert isinstance(segmenter, Segmenter)

        # Prompt-based segmentation
        img = Image.from_path("photo.jpg")
        masks = segmenter.segment(
            img,
            box_prompts=[(50, 50, 300, 300)],
            mode="boxes"
        )
        ```

    Args:
        image: Input image artifact
        **kwargs: Model-specific parameters (prompts, mode, etc.)

    Returns:
        Masks artifact with instance masks and metadata
    """

    def segment(self, image: Image, **kwargs) -> Masks:
        """Segment image into masks.

        Args:
            image: Input image artifact
            **kwargs: Model-specific parameters
                - box_prompts (List[Tuple]): Bounding box prompts for SAM
                - point_prompts (List[Tuple]): Point prompts (x, y, label)
                - mode (str): Segmentation mode ("everything", "boxes", "points")
                - threshold (float): Confidence threshold
                - text_prompts (str or List[str]): For zero-shot models (SAM3)

        Returns:
            Masks artifact with instance masks and instance_ids
        """
        ...


@runtime_checkable
class Classifier(Protocol):
    """Classification capability protocol.

    Providers implementing this protocol can classify images or image regions,
    returning class labels with confidence scores.

    Supported models:
    - Standard classification: ResNet, EfficientNet, ViT, ConvNeXt
    - Zero-shot classification: CLIP, ALIGN

    Example:
        ```python
        from mata.core.artifacts import Image
        from mata.core.registry.protocols import Classifier

        class MyClassifier:
            def classify(self, image: Image, **kwargs) -> Classifications:
                # Classification logic
                return classifications

        classifier = MyClassifier()
        assert isinstance(classifier, Classifier)

        img = Image.from_path("photo.jpg")
        result = classifier.classify(img, top_k=5)
        print(result.top1.label_name, result.top1.score)
        ```

    Args:
        image: Input image artifact
        **kwargs: Model-specific parameters (top_k, text_prompts, etc.)

    Returns:
        ClassifyResult with classifications
    """

    def classify(self, image: Image, **kwargs) -> ClassifyResult:
        """Classify image.

        Args:
            image: Input image artifact
            **kwargs: Model-specific parameters
                - top_k (int): Number of top predictions to return
                - text_prompts (List[str]): For zero-shot models (CLIP)

        Returns:
            ClassifyResult with sorted classifications
        """
        ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Pose estimation capability protocol.

    Providers implementing this protocol can estimate human or animal pose,
    returning keypoint locations with confidence scores.

    Supported models:
    - Human pose: HRNet, OpenPose, MediaPipe
    - Animal pose: DeepLabCut, SLEAP

    Example:
        ```python
        from mata.core.artifacts import Image, ROIs
        from mata.core.registry.protocols import PoseEstimator

        class MyPoseEstimator:
            def estimate(self, image: Image, rois: Optional[ROIs] = None, **kwargs) -> Keypoints:
                # Pose estimation logic
                return keypoints

        pose_estimator = MyPoseEstimator()
        assert isinstance(pose_estimator, PoseEstimator)

        img = Image.from_path("photo.jpg")
        keypoints = pose_estimator.estimate(img)
        ```

    Args:
        image: Input image artifact
        rois: Optional region proposals (for two-stage pose estimation)
        **kwargs: Model-specific parameters

    Returns:
        Keypoints artifact with keypoint locations and skeleton
    """

    def estimate(self, image: Image, rois: ROIs | None = None, **kwargs) -> Keypoints:
        """Estimate pose keypoints.

        Args:
            image: Input image artifact
            rois: Optional region proposals for two-stage estimation
            **kwargs: Model-specific parameters
                - threshold (float): Keypoint confidence threshold
                - skeleton (List[Tuple[int, int]]): Skeleton connections

        Returns:
            Keypoints artifact with (x, y, score) arrays
        """
        ...


@runtime_checkable
class DepthEstimator(Protocol):
    """Depth estimation capability protocol.

    Providers implementing this protocol can estimate depth from monocular images,
    returning depth maps with per-pixel depth values.

    Supported models:
    - Monocular depth: Depth Anything V1/V2, MiDaS, DPT
    - Relative depth: ZoeDepth, AdaBins

    Example:
        ```python
        from mata.core.artifacts import Image
        from mata.core.registry.protocols import DepthEstimator

        class MyDepthEstimator:
            def estimate(self, image: Image, **kwargs) -> DepthResult:
                # Depth estimation logic
                return depth_result

        depth_estimator = MyDepthEstimator()
        assert isinstance(depth_estimator, DepthEstimator)

        img = Image.from_path("photo.jpg")
        depth = depth_estimator.estimate(img)
        depth.save("depth.png", colormap="magma")
        ```

    Args:
        image: Input image artifact
        **kwargs: Model-specific parameters

    Returns:
        DepthResult with depth map and metadata
    """

    def estimate(self, image: Image, **kwargs) -> DepthResult:
        """Estimate depth map from image.

        Args:
            image: Input image artifact
            **kwargs: Model-specific parameters
                - output_type (str): "normalized" or "raw"
                - colormap (str): For visualization

        Returns:
            DepthResult with (H, W) depth map array
        """
        ...


@runtime_checkable
class OCRReader(Protocol):
    """OCR capability protocol for extracting text from images.

    Providers implementing this protocol can perform Optical Character Recognition
    on images, returning structured text results with bounding boxes and confidence
    scores.

    Supported models:
    - Tesseract OCR (via pytesseract)
    - PaddleOCR
    - HuggingFace TrOCR / DocTR

    Example:
        ```python
        from mata.core.artifacts import Image
        from mata.core.registry.protocols import OCRReader

        class MyOCRReader:
            def predict(self, image: Image, **kwargs) -> OCRResult:
                # OCR logic
                return ocr_result

        reader = MyOCRReader()
        assert isinstance(reader, OCRReader)  # Runtime check

        img = Image.from_path("document.jpg")
        result = reader.predict(img)
        print(result.full_text)
        ```

    Args:
        image: Input image artifact
        **kwargs: Model-specific parameters (lang, conf_threshold, etc.)

    Returns:
        OCRResult with detected text regions and metadata
    """

    def predict(self, image: Any, **kwargs) -> OCRResult:
        """Run OCR on an image, returning structured text results.

        Args:
            image: Input image artifact
            **kwargs: Model-specific parameters
                - lang (str): Language hint (e.g. "en", "chi_sim")
                - conf_threshold (float): Minimum confidence for regions

        Returns:
            OCRResult with text regions, bounding boxes, and scores
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Feature embedding capability protocol.

    Providers implementing this protocol can extract feature embeddings from images
    or image regions, useful for similarity search, clustering, and retrieval.

    Supported models:
    - Image embeddings: CLIP, DINOv2, ResNet (penultimate layer)
    - Region embeddings: ROI pooling + backbone

    Example:
        ```python
        from mata.core.artifacts import Image, ROIs
        from mata.core.registry.protocols import Embedder

        class MyEmbedder:
            def embed(self, input: Union[Image, ROIs], **kwargs) -> Embeddings:
                # Embedding extraction logic
                return embeddings

        embedder = MyEmbedder()
        assert isinstance(embedder, Embedder)

        # Image-level embedding
        img = Image.from_path("photo.jpg")
        emb = embedder.embed(img)  # Shape: (embedding_dim,)

        # Region-level embeddings
        rois = extract_rois(img, detections)
        embs = embedder.embed(rois)  # Shape: (num_rois, embedding_dim)
        ```

    Args:
        input: Image or ROIs artifact
        **kwargs: Model-specific parameters

    Returns:
        Embeddings artifact (np.ndarray or list of arrays)
    """

    def embed(self, input: Image | ROIs, **kwargs) -> np.ndarray:
        """Extract feature embeddings.

        Args:
            input: Image artifact or ROIs artifact
            **kwargs: Model-specific parameters
                - layer (str): Layer to extract from
                - normalize (bool): L2 normalization

        Returns:
            NumPy array of embeddings:
                - Image input: (embedding_dim,)
                - ROIs input: (num_rois, embedding_dim)
        """
        ...


@runtime_checkable
class Tracker(Protocol):
    """Object tracking capability protocol (stateful).

    Providers implementing this protocol can track objects across video frames,
    maintaining temporal identity and handling occlusions.

    Supported trackers:
    - Multi-object tracking: BYTETrack, DeepSORT, FairMOT
    - Single-object tracking: SiamRPN, TransTrack

    **Important**: Tracker is stateful - it maintains internal state across
    update() calls. Call reset() between video sequences.

    Example:
        ```python
        from mata.core.artifacts import Detections
        from mata.core.registry.protocols import Tracker

        class MyTracker:
            def __init__(self):
                self.state = {}

            def update(self, detections: Detections, frame_id: str, **kwargs) -> Tracks:
                # Tracking logic
                return tracks

            def reset(self) -> None:
                self.state = {}

        tracker = MyTracker()
        assert isinstance(tracker, Tracker)

        # Track across video frames
        for frame_idx, frame in enumerate(video):
            img = Image(data=frame, frame_id=f"frame_{frame_idx}")
            dets = detector.predict(img)
            tracks = tracker.update(dets, frame_id=f"frame_{frame_idx}")

        # Reset for next video
        tracker.reset()
        ```

    Args:
        detections: Detections from current frame
        frame_id: Unique frame identifier
        **kwargs: Tracker-specific parameters

    Returns:
        Tracks artifact with track IDs and history
    """

    def update(self, detections: Detections, frame_id: str, **kwargs) -> Tracks:
        """Update tracker with new frame detections.

        Args:
            detections: Detections from current frame
            frame_id: Unique identifier for this frame
            **kwargs: Tracker-specific parameters
                - track_thresh (float): Track activation threshold
                - track_buffer (int): Number of frames to keep lost tracks
                - match_thresh (float): IoU threshold for matching

        Returns:
            Tracks artifact with current tracks
        """
        ...

    def reset(self) -> None:
        """Reset tracker state (call between video sequences).

        Clears all internal state including active tracks, track history,
        and track ID counter.
        """
        ...


@runtime_checkable
class VisionLanguageModel(Protocol):
    """Vision-language model capability protocol (NEW in v1.6).

    Providers implementing this protocol can process images with text prompts,
    enabling multi-modal understanding tasks like visual question answering,
    image captioning, and semantic object detection.

    Supports:
    - Single or multiple images (multi-image reasoning)
    - Structured output modes (JSON, detection, classification)
    - Entity extraction with auto-promotion to instances
    - Zero-shot detection/classification via text prompts

    Supported models:
    - Qwen2-VL, Qwen3-VL (multi-modal understanding)
    - LLaVA, InstructBLIP (visual question answering)
    - GPT-4V, Gemini Vision (closed-source VLMs)

    Example:
        ```python
        from mata.core.artifacts import Image
        from mata.core.registry.protocols import VisionLanguageModel

        class MyVLM:
            def query(
                self,
                image: Union[Image, List[Image]],
                prompt: str,
                output_mode: Optional[str] = None,
                auto_promote: bool = False,
                **kwargs
            ) -> VisionResult:
                # VLM inference logic
                return vision_result

        vlm = MyVLM()
        assert isinstance(vlm, VisionLanguageModel)

        # Image captioning
        img = Image.from_path("photo.jpg")
        result = vlm.query(img, "Describe this image in detail.")
        print(result.text)

        # Structured detection (zero-shot)
        result = vlm.query(
            img,
            "List all objects in the image with their locations.",
            output_mode="detect",
            auto_promote=False
        )
        print(f"Found {len(result.entities)} entities")
        for entity in result.entities:
            print(f"  - {entity.label}: {entity.attributes}")

        # Multi-image reasoning
        images = [Image.from_path(f"img{i}.jpg") for i in range(3)]
        result = vlm.query(
            images,
            "What changed between these images?",
            output_mode="json"
        )
        print(result.text)

        # Auto-promotion workflow (VLM → GroundingDINO fusion)
        # 1. VLM extracts entity concepts
        vlm_result = vlm.query(img, "What objects are present?", output_mode="json")

        # 2. Use entities as text prompts for GroundingDINO
        text_prompts = " . ".join([e.label for e in vlm_result.entities])
        spatial_result = grounding_dino.predict(img, text_prompts=text_prompts)

        # 3. Promote entities to instances with spatial data
        detections = Detections.from_vision_result(vlm_result)
        promoted = detections.promote_entities(
            Detections.from_vision_result(spatial_result),
            match_strategy="label_fuzzy"
        )
        ```

    Args:
        image: Single image or list of images for multi-image reasoning
        prompt: Text prompt/question for the VLM
        output_mode: Optional structured output mode
            - None: Raw text response
            - "json": Structured JSON output (parsed to entities)
            - "detect": Detection format (entities with labels/attributes)
            - "classify": Classification format (single entity)
            - "describe": Natural language description
        auto_promote: If True, automatically promote entities to instances
            when spatial data (bbox/mask) is present in VLM output
        **kwargs: Model-specific parameters

    Returns:
        VisionResult with:
            - text: Raw VLM response text
            - entities: Parsed entities (if output_mode set)
            - instances: Auto-promoted instances (if auto_promote=True and spatial data available)
            - meta: Model metadata, image paths, parse stats
    """

    def query(
        self,
        image: Image | list[Image],
        prompt: str,
        output_mode: str | None = None,
        auto_promote: bool = False,
        **kwargs: Any,
    ) -> VisionResult:
        """Query VLM with image(s) and text prompt.

        Args:
            image: Single Image artifact or list of Image artifacts for multi-image queries
            prompt: Text prompt/question for the VLM
            output_mode: Structured output format
                - None: Return raw text in result.text
                - "json": Parse JSON response to entities
                - "detect": Parse as detection format (entities with labels)
                - "classify": Parse as classification (single entity)
                - "describe": Natural language description
            auto_promote: If True, promote entities to instances when spatial data present
            **kwargs: Model-specific parameters
                - max_tokens (int): Maximum response length
                - temperature (float): Sampling temperature
                - system_prompt (str): System prompt override

        Returns:
            VisionResult containing:
                - text (str): Raw VLM response
                - entities (List[Entity]): Parsed entities (if output_mode set)
                - instances (List[Instance]): Auto-promoted instances (if auto_promote=True)
                - meta (Dict): Metadata including:
                    - model_name: Model identifier
                    - image_paths: Source image paths
                    - num_images: Number of input images
                    - output_mode: Requested output mode
                    - parse_success: Whether parsing succeeded
                    - parse_errors: Any parsing errors

        Raises:
            ValueError: If image list is empty or prompt is empty
            RuntimeError: If VLM inference fails
        """
        ...
