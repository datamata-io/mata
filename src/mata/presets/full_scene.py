"""Full-scene analysis and tracking graph presets.

Provides multi-task parallel workflows for scene understanding and
temporal object tracking.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.track import Track


def full_scene_analysis(
    detection_threshold: float = 0.3,
    classification_labels: list[str] | None = None,
) -> Graph:
    """Parallel detection, classification, and depth estimation.

    Runs three independent tasks in parallel and fuses the results into a
    single :class:`~mata.core.artifacts.result.MultiResult`:

    - **Detection**: object detection with configurable threshold
    - **Classification**: image-level classification (optionally zero-shot
      with CLIP via ``text_prompts``)
    - **Depth**: monocular depth estimation

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (e.g. RT-DETR)
        - ``"classifier"`` — classification adapter (e.g. CLIP)
        - ``"depth"`` — depth estimation adapter (e.g. Depth-Anything)

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.3``).
        classification_labels: Optional list of text prompts for zero-shot
            classification (e.g. ``["indoor", "outdoor"]``).  If ``None``,
            standard classification is performed.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import full_scene_analysis
        >>>
        >>> detector = mata.load("detect", "PekingU/rtdetr_v2_r18vd")
        >>> classifier = mata.load("classify", "openai/clip-vit-base-patch32")
        >>> depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        >>> result = mata.infer(
        ...     "scene.jpg",
        ...     full_scene_analysis(
        ...         classification_labels=["indoor", "outdoor", "urban", "nature"],
        ...     ),
        ...     providers={
        ...         "detector": detector,
        ...         "classifier": classifier,
        ...         "depth": depth,
        ...     },
        ... )
    """
    # Build classifier kwargs
    classify_kwargs: dict = {}
    if classification_labels:
        classify_kwargs["text_prompts"] = classification_labels

    return (
        Graph("full_scene")
        # Run detection, classification, and depth in parallel
        .parallel(
            [
                Detect(using="detector", out="dets", threshold=detection_threshold),
                Classify(using="classifier", out="class", **classify_kwargs),
                EstimateDepth(using="depth", out="depth"),
            ]
        )
        # Filter low-confidence detections
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        # Bundle everything into a single MultiResult
        .then(
            Fuse(
                out="scene",
                dets="filtered",
                classifications="class",
                depth="depth",
            )
        )
    )


def detect_and_track(
    detection_threshold: float = 0.5,
    track_threshold: float = 0.5,
    match_threshold: float = 0.8,
    track_buffer: int = 30,
) -> Graph:
    """Detection + BYTETrack temporal tracking.

    Designed for frame-by-frame video processing: detects objects in each
    frame, filters by confidence, then updates a stateful tracker to assign
    consistent track IDs across frames.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter
        - ``"tracker"`` — tracker instance (e.g. ``ByteTrackWrapper``)

    Args:
        detection_threshold: Minimum confidence for detections
            (default ``0.5``).
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
        >>> from mata.presets import detect_and_track
        >>>
        >>> detector = mata.load("detect", "PekingU/rtdetr_v2_r18vd")
        >>> # tracker is typically provided via ByteTrackWrapper
        >>> graph = detect_and_track(detection_threshold=0.4)
        >>> # Execute per frame in a loop
        >>> for frame in frames:
        ...     result = mata.infer(frame, graph, providers={...})
    """
    return (
        Graph("detect_and_track")
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        .then(
            Track(
                using="tracker",
                dets="filtered",
                out="tracks",
                track_thresh=track_threshold,
                match_thresh=match_threshold,
            )
        )
        .then(Fuse(out="final", dets="filtered", tracks="tracks"))
    )
