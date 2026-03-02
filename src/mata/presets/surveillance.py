"""Security and surveillance graph presets.

Provides graph factories for surveillance workflows including crowd monitoring,
suspicious object detection, and situational awareness analysis.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.annotate import Annotate
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.refine_mask import RefineMask
from mata.nodes.track import BotSortWrapper, ByteTrackWrapper, Track  # noqa: F401
from mata.nodes.vlm_query import VLMQuery


def crowd_monitoring(
    detection_threshold: float = 0.4,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph:
    """Person detection + tracking for crowd monitoring.

    Detects persons, filters to person class only, assigns track IDs
    via BYTETrack for counting unique individuals, and annotates
    frames for visual monitoring.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — person detection adapter
        - ``"tracker"`` — tracker instance (e.g. ByteTrackWrapper)

    Args:
        detection_threshold: Minimum confidence for person detections
            (default ``0.4``).
        track_threshold: Minimum confidence for track association
            (default ``0.5``).
        match_threshold: IoU threshold for track matching
            (default ``0.8``).
        track_buffer: Frames to keep lost tracks (default ``30``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import crowd_monitoring
        >>> from mata.nodes.track import ByteTrackWrapper
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> tracker = ByteTrackWrapper()  # or BotSortWrapper() for camera-motion robustness
        >>>
        >>> # Process video frames
        >>> for frame in video_frames:
        ...     result = mata.infer(
        ...         frame,
        ...         crowd_monitoring(detection_threshold=0.5),
        ...         providers={"detector": detector, "tracker": tracker},
        ...     )
        ...     tracks = result["final"]
        ...     print(f"Active tracks: {len(tracks.instances)}")
    """
    return (
        Graph("crowd_monitoring")
        .then(Detect(using="detector", out="dets"))
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                label_in=["person"],
                fuzzy=True,
                out="filtered",
            )
        )
        .then(
            Track(
                using="tracker",
                dets="filtered",
                out="tracks",
                track_thresh=track_threshold,
                match_thresh=match_threshold,
                track_buffer=track_buffer,
            )
        )
        .then(
            Annotate(
                detections_src="tracks",
                out="annotated",
            )
        )
        .then(Fuse(out="final", tracks="tracks", annotated="annotated"))
    )


def crowd_monitoring_botsort(
    detection_threshold: float = 0.4,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph:
    """Person detection + BotSort tracking for crowd monitoring.

    Identical graph to :func:`crowd_monitoring` but pre-configured for
    BotSort, which adds Global Motion Compensation (GMC) via sparse
    optical flow. Use this variant when the camera may pan, tilt, or
    zoom — GMC corrects for camera motion so tracks are not broken by
    background movement.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — person detection adapter
        - ``"tracker"`` — :class:`~mata.nodes.track.BotSortWrapper` instance

    Args:
        detection_threshold: Minimum confidence for person detections
            (default ``0.4``).
        track_threshold: Minimum confidence for track association
            (default ``0.5``).
        match_threshold: IoU threshold for track matching
            (default ``0.8``).
        track_buffer: Frames to keep lost tracks (default ``30``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import crowd_monitoring_botsort
        >>> from mata.nodes.track import BotSortWrapper
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> tracker = BotSortWrapper()  # GMC-enabled for camera motion
        >>>
        >>> for frame in video_frames:
        ...     result = mata.infer(
        ...         frame,
        ...         crowd_monitoring_botsort(detection_threshold=0.5),
        ...         providers={"detector": detector, "tracker": tracker},
        ...     )
        ...     tracks = result["final"]
        ...     print(f"Active tracks: {len(tracks.instances)}")
    """
    return (
        Graph("crowd_monitoring_botsort")
        .then(Detect(using="detector", out="dets"))
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                label_in=["person"],
                fuzzy=True,
                out="filtered",
            )
        )
        .then(
            Track(
                using="tracker",
                dets="filtered",
                out="tracks",
                track_thresh=track_threshold,
                match_thresh=match_threshold,
                track_buffer=track_buffer,
            )
        )
        .then(
            Annotate(
                detections_src="tracks",
                out="annotated",
            )
        )
        .then(Fuse(out="final", tracks="tracks", annotated="annotated"))
    )


def suspicious_object_detection(
    object_prompts: str = "backpack . suitcase . bag . package . unattended object",
    detection_threshold: float = 0.25,
    vlm_prompt: str = "Analyze this object. Is it unattended, abandoned, or suspicious? Describe its state and surroundings.",
    refine_method: str = "morph_close",
    refine_radius: int = 3,
) -> Graph:
    """Zero-shot suspicious object detection + segmentation + VLM analysis.

    Detects potentially suspicious objects using text prompts,
    segments each with SAM for precise boundaries, then queries
    a VLM for contextual reasoning about whether the object is
    abandoned or suspicious.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — text-prompted detector (e.g. GroundingDINO)
        - ``"segmenter"`` — SAM-style segmentation adapter
        - ``"vlm"`` — VLM adapter for reasoning (e.g. Qwen3-VL)

    Args:
        object_prompts: Dot-separated text prompts for suspicious
            objects (default ``"backpack . suitcase . bag . package . unattended object"``).
        detection_threshold: Minimum confidence for object detections
            (default ``0.25``).
        vlm_prompt: Prompt for VLM contextual analysis (default asks
            about unattended/abandoned/suspicious status).
        refine_method: Morphological operation for mask refinement.
            One of ``"morph_close"``, ``"morph_open"``, ``"dilate"``,
            ``"erode"`` (default ``"morph_close"``).
        refine_radius: Kernel radius for mask refinement (default ``3``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import suspicious_object_detection
        >>>
        >>> detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        >>> segmenter = mata.load("segment", "facebook/sam-vit-base")
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        >>>
        >>> result = mata.infer(
        ...     "surveillance_frame.jpg",
        ...     suspicious_object_detection(
        ...         object_prompts="backpack . suitcase . bag",
        ...     ),
        ...     providers={
        ...         "detector": detector,
        ...         "segmenter": segmenter,
        ...         "vlm": vlm,
        ...     },
        ... )
        >>> for inst in result["final"].instances:
        ...     print(f"{inst.label_name}: {inst.vlm_response}")
    """
    return (
        Graph("suspicious_object_detection")
        .then(
            Detect(
                using="detector",
                out="dets",
                text_prompts=object_prompts,
            )
        )
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                out="filtered",
            )
        )
        .then(
            PromptBoxes(
                using="segmenter",
                dets_src="filtered",
                out="masks",
            )
        )
        .then(
            RefineMask(
                src="masks",
                method=refine_method,
                radius=refine_radius,
                out="refined",
            )
        )
        .then(
            VLMQuery(
                using="vlm",
                prompt=vlm_prompt,
                out="vlm_analysis",
            )
        )
        .then(
            Fuse(
                out="final",
                dets="filtered",
                masks="refined",
                vlm_analysis="vlm_analysis",
            )
        )
    )
