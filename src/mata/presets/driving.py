"""Autonomous driving and ADAS graph presets.

Provides graph factories for vehicle detection, road scene analysis,
distance estimation, and traffic tracking workflows.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.annotate import Annotate
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.segment import SegmentImage
from mata.nodes.track import BotSortWrapper, ByteTrackWrapper, Track  # noqa: F401


def vehicle_distance_estimation(
    detection_threshold: float = 0.4,
    vehicle_labels: list[str] | None = None,
) -> Graph:
    """Parallel detection + depth for distance estimation.

    Detects vehicles/pedestrians and simultaneously estimates depth,
    enabling distance-aware filtering and analysis. Objects are
    filtered to traffic-relevant classes.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (e.g. DETR)
        - ``"depth"`` — depth estimation adapter (e.g. Depth Anything)

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.4``).
        vehicle_labels: Classes to keep. Defaults to
            ``["car", "truck", "bus", "motorcycle", "bicycle", "person"]``.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import vehicle_distance_estimation
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        >>> result = mata.infer(
        ...     "street.jpg",
        ...     vehicle_distance_estimation(detection_threshold=0.5),
        ...     providers={"detector": detector, "depth": depth},
        ... )
        >>> # Analyze distance to detected vehicles
        >>> for inst in result["final"].instances:
        ...     print(f"{inst.label_name}: bbox={inst.bbox}")
    """
    if vehicle_labels is None:
        vehicle_labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]

    return (
        Graph("vehicle_distance_estimation")
        # Run detection and depth in parallel
        .parallel(
            [
                Detect(using="detector", out="dets"),
                EstimateDepth(using="depth", out="depth"),
            ]
        )
        # Filter to vehicle/pedestrian classes and confidence threshold
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                label_in=vehicle_labels,
                out="filtered",
            )
        )
        # Bundle detections with depth map
        .then(Fuse(out="final", dets="filtered", depth="depth"))
    )


def road_scene_analysis(
    detection_threshold: float = 0.3,
    scene_labels: list[str] | None = None,
) -> Graph:
    """Complete road scene analysis with 4 parallel tasks.

    Comprehensive scene understanding: object detection, panoptic
    segmentation (road/sidewalk/sky), depth estimation, and CLIP
    scene classification (urban/highway/rural).

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter
        - ``"segmenter"`` — panoptic segmentation adapter (e.g. Mask2Former)
        - ``"depth"`` — depth estimation adapter
        - ``"classifier"`` — zero-shot classifier (e.g. CLIP)

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.3``).
        scene_labels: Scene type labels for zero-shot classification.
            Defaults to ``["urban_road", "highway", "rural_road",
            "intersection", "parking_lot"]``.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import road_scene_analysis
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> segmenter = mata.load("segment", "facebook/mask2former-swin-tiny-coco-panoptic")
        >>> depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        >>> classifier = mata.load("classify", "openai/clip-vit-base-patch32")
        >>> result = mata.infer(
        ...     "road.jpg",
        ...     road_scene_analysis(),
        ...     providers={
        ...         "detector": detector,
        ...         "segmenter": segmenter,
        ...         "depth": depth,
        ...         "classifier": classifier,
        ...     },
        ... )
        >>> # Access comprehensive scene analysis
        >>> print(f"Scene type: {result['final'].classifications[0].label}")
        >>> print(f"Detected objects: {len(result['final'].instances)}")
    """
    if scene_labels is None:
        scene_labels = ["urban_road", "highway", "rural_road", "intersection", "parking_lot"]

    # Build classifier kwargs for zero-shot classification
    classify_kwargs = {"text_prompts": scene_labels}

    return (
        Graph("road_scene_analysis")
        # Run all 4 tasks in parallel
        .parallel(
            [
                Detect(using="detector", out="dets"),
                SegmentImage(using="segmenter", out="segments"),
                EstimateDepth(using="depth", out="depth"),
                Classify(using="classifier", out="class", **classify_kwargs),
            ]
        )
        # Filter low-confidence detections
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        # Bundle all results
        .then(
            Fuse(
                out="final",
                dets="filtered",
                masks="segments",
                depth="depth",
                classifications="class",
            )
        )
    )


def traffic_tracking(
    detection_threshold: float = 0.4,
    vehicle_labels: list[str] | None = None,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph:
    """Detection + tracking for traffic monitoring video.

    Detects traffic objects, filters to vehicle classes, assigns
    persistent track IDs via BYTETrack, and annotates frames.
    Designed for frame-by-frame video processing.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (fast model recommended)
        - ``"tracker"`` — tracker instance (e.g. ByteTrackWrapper)

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.4``).
        vehicle_labels: Classes to keep. Defaults to
            ``["car", "truck", "bus", "motorcycle", "bicycle", "person"]``.
        track_threshold: Minimum confidence for track association
            (default ``0.5``).
        match_threshold: IoU threshold for track matching
            (default ``0.8``).
        track_buffer: Number of frames to keep lost tracks
            (default ``30``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import traffic_tracking
        >>> from mata.nodes.track import ByteTrackWrapper  # or BotSortWrapper
        >>>
        >>> detector = mata.load("detect", "PekingU/rtdetr_v2_r18vd")
        >>> tracker = ByteTrackWrapper()  # swap for BotSortWrapper() for camera-motion robustness
        >>> graph = traffic_tracking(detection_threshold=0.5)
        >>> # Execute per frame in a video loop
        >>> for frame in video_frames:
        ...     result = mata.infer(
        ...         frame,
        ...         graph,
        ...         providers={"detector": detector, "tracker": tracker},
        ...     )
        ...     annotated_frame = result["final"].image
    """
    if vehicle_labels is None:
        vehicle_labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]

    return (
        Graph("traffic_tracking")
        # Detect objects
        .then(Detect(using="detector", out="dets"))
        # Filter to vehicle/pedestrian classes and confidence threshold
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                label_in=vehicle_labels,
                out="filtered",
            )
        )
        # Assign track IDs
        .then(
            Track(
                using="tracker",
                dets="filtered",
                out="tracks",
                track_thresh=track_threshold,
                match_thresh=match_threshold,
            )
        )
        # Annotate frame with tracks
        .then(
            Annotate(
                image_src="image",
                detections_src="tracks",
                show_boxes=True,
                show_labels=True,
                show_scores=True,
                out="annotated",
            )
        )
        # Bundle all results
        .then(Fuse(out="final", dets="filtered", tracks="tracks", image="annotated"))
    )


def traffic_tracking_botsort(
    detection_threshold: float = 0.4,
    vehicle_labels: list[str] | None = None,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph:
    """Detection + BotSort tracking for traffic monitoring video.

    Identical graph to :func:`traffic_tracking` but pre-configured for
    BotSort, which adds Global Motion Compensation (GMC) via sparse
    optical flow. Use this variant when the camera may pan or zoom
    — GMC prevents spurious track breaks caused by camera motion.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (fast model recommended)
        - ``"tracker"`` — :class:`~mata.nodes.track.BotSortWrapper` instance

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.4``).
        vehicle_labels: Classes to keep. Defaults to
            ``["car", "truck", "bus", "motorcycle", "bicycle", "person"]``.
        track_threshold: Minimum confidence for track association
            (default ``0.5``).
        match_threshold: IoU threshold for track matching
            (default ``0.8``).
        track_buffer: Number of frames to keep lost tracks
            (default ``30``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import traffic_tracking_botsort
        >>> from mata.nodes.track import BotSortWrapper
        >>>
        >>> detector = mata.load("detect", "PekingU/rtdetr_v2_r18vd")
        >>> tracker = BotSortWrapper()  # GMC-enabled for moving cameras
        >>> graph = traffic_tracking_botsort(detection_threshold=0.5)
        >>> for frame in video_frames:
        ...     result = mata.infer(
        ...         frame,
        ...         graph,
        ...         providers={"detector": detector, "tracker": tracker},
        ...     )
        ...     annotated_frame = result["final"].image
    """
    if vehicle_labels is None:
        vehicle_labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]

    return (
        Graph("traffic_tracking_botsort")
        .then(Detect(using="detector", out="dets"))
        .then(
            Filter(
                src="dets",
                score_gt=detection_threshold,
                label_in=vehicle_labels,
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
                image_src="image",
                detections_src="tracks",
                show_boxes=True,
                show_labels=True,
                show_scores=True,
                out="annotated",
            )
        )
        .then(Fuse(out="final", dets="filtered", tracks="tracks", image="annotated"))
    )
