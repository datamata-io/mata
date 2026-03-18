"""AnnotateRT node — real-time OpenCV annotation for graph pipelines.

Stateful node that maintains per-track trail history across frames.
Uses the low-level helpers in ``mata.visualization_cv2`` to render
bounding boxes, trajectories, camera labels, and cross-camera highlights
directly onto BGR NumPy arrays with minimal overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.tracks import Tracks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.artifacts.cross_matches import CrossMatches
    from mata.core.graph.context import ExecutionContext


class AnnotateRT(Node):
    """Real-time annotation node that draws bounding boxes, trails, and labels.

    Uses OpenCV (cv2) to overlay detections, track IDs, trajectory trails,
    a camera name bar, and cross-camera ReID highlights onto each frame.
    The node is **stateful** — it accumulates per-track centre-point history
    across successive ``run()`` calls so that ``draw_trails`` can render a
    smoothly fading polyline for each tracked object.

    No provider is required — ``AnnotateRT`` is a pure processing node
    (similar to ``Filter`` or ``Fuse``).

    Args:
        show_boxes: Draw bounding box rectangles. Default ``True``.
        show_labels: Include class label in the overlay text. Default ``True``.
        show_scores: Include confidence score in the overlay text. Default ``True``.
        show_track_ids: Prepend ``#<id>`` to the overlay text. Default ``True``.
        show_trails: Draw trajectory polylines from accumulated trail history.
            Default ``False``.
        trail_length: Maximum number of centre-points retained per track.
            Default ``30``.
        camera_label: If set, renders a coloured label bar in the top-left
            corner of every frame. Default ``None`` (disabled).
        camera_color: BGR colour for the camera label bar. Falls back to the
            default orange ``(255, 100, 60)`` when ``None``. Default ``None``.
        line_width: Bounding box border thickness in pixels. Default ``2``.
        out: Output artifact key. Default ``"annotated"``.
        image_src: Input image artifact key. Default ``"image"``.
        detections_src: Input detections (or tracks) artifact key. Accepts
            both :class:`~mata.core.artifacts.Detections` and
            :class:`~mata.core.artifacts.Tracks` via duck-typing.
            Default ``"detections"``.
        tracks_src: Optional key for an additional Tracks artifact used
            exclusively for trail history updates. When ``None``, trails fall
            back to track_id values present on the ``detections_src`` artifact.
            Default ``None``.
        cross_matches_src: Optional key for a CrossMatches artifact. When
            provided, matched tracks receive a yellow double-border highlight
            and a ``~camId#trackId`` label suffix. Default ``None``.
        name: Optional human-readable node name.

    Inputs:
        image (Image): Input frame.
        detections (Detections): Detection or tracking results to visualise.
        tracks (Tracks): *Optional.* Active tracks for trail accumulation.
        cross_matches (CrossMatches): *Optional.* ReID matches for highlights.

    Outputs:
        annotated (Image): Annotated frame in BGR colour space.

    Example:
        ```python
        from mata.nodes import AnnotateRT

        node = AnnotateRT(
            show_track_ids=True,
            show_trails=True,
            trail_length=40,
            camera_label="CAM-1",
        )
        result = node.run(ctx, image=img, detections=tracks_art)
        annotated = result["annotated"]

        # Reset trail state between video clips
        node.reset()
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image, "detections": Artifact}
    outputs: dict[str, type[Artifact]] = {"annotated": Image}

    def __init__(
        self,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_scores: bool = True,
        show_track_ids: bool = True,
        show_trails: bool = False,
        trail_length: int = 30,
        camera_label: str | None = None,
        camera_color: tuple[int, int, int] | None = None,
        line_width: int = 2,
        out: str = "annotated",
        image_src: str = "image",
        detections_src: str = "detections",
        tracks_src: str | None = None,
        cross_matches_src: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_scores = show_scores
        self.show_track_ids = show_track_ids
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.camera_label = camera_label
        self.camera_color = camera_color
        self.line_width = line_width
        self.out = out
        self.image_src = image_src
        self.detections_src = detections_src
        self.tracks_src = tracks_src
        self.cross_matches_src = cross_matches_src

        # Trail state — mutable, persists across run() calls
        self._trail_history: dict[int, list[tuple[int, int]]] = {}

        # Build dynamic artifact map based on which optional srcs are set
        # detections_src uses base Artifact as the type because AnnotateRT
        # accepts both Detections and Tracks via duck-typing at runtime.
        _inputs: dict[str, type[Artifact]] = {
            image_src: Image,
            detections_src: Artifact,
        }
        if tracks_src is not None:
            _inputs[tracks_src] = Tracks
        if cross_matches_src is not None:
            from mata.core.artifacts.cross_matches import CrossMatches

            _inputs[cross_matches_src] = CrossMatches  # type: ignore[assignment]
        self.inputs = _inputs
        self.outputs = {out: Image}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated trail history.

        Call between video clips or whenever track IDs are reset to prevent
        stale trails from a previous sequence.
        """
        self._trail_history.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Render annotations onto the input frame.

        Args:
            ctx: Execution context (metrics recording only — no provider needed).
            **inputs: Input artifacts keyed by their configured src names.

        Returns:
            Dict with key ``self.out`` mapping to an annotated Image artifact
            in BGR colour space.
        """
        from mata.visualization_cv2 import draw_boxes, draw_camera_label, draw_trails

        # ------------------------------------------------------------------
        # 1. Obtain BGR numpy array from input image
        # ------------------------------------------------------------------
        image: Image = inputs[self.image_src]  # type: ignore[assignment]
        frame = image.to_numpy().copy()

        if image.color_space == "RGB" and frame.ndim == 3:
            try:
                import cv2

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except ImportError:
                # Fallback: flip channel order without cv2
                frame = frame[..., ::-1].copy()

        # ------------------------------------------------------------------
        # 2. Resolve instances from detections artifact (duck-typed)
        # ------------------------------------------------------------------
        dets_artifact = inputs[self.detections_src]
        instances: list[Any]
        if hasattr(dets_artifact, "instances"):
            instances = dets_artifact.instances  # type: ignore[union-attr]
        elif hasattr(dets_artifact, "tracks"):
            instances = dets_artifact.tracks  # type: ignore[union-attr]
        else:
            instances = []

        # ------------------------------------------------------------------
        # 3. Resolve optional CrossMatches artifact
        # ------------------------------------------------------------------
        cross_matches: CrossMatches | None = None
        if self.cross_matches_src is not None:
            cross_matches = inputs.get(self.cross_matches_src)  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # 4a. Update trail history and draw trail polylines
        # ------------------------------------------------------------------
        if self.show_trails:
            # Prefer dedicated tracks_src for trail updates when provided
            if self.tracks_src is not None and self.tracks_src in inputs:
                tracks_art: Tracks = inputs[self.tracks_src]  # type: ignore[assignment]
                trail_candidates: list[Any] = tracks_art.get_active_tracks().tracks
            else:
                # Fall back to instances that carry a track_id
                trail_candidates = [t for t in instances if getattr(t, "track_id", None) is not None]

            for t in trail_candidates:
                tid: int | None = getattr(t, "track_id", None)
                if tid is None:
                    continue
                bbox = t.bbox
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                pts = self._trail_history.setdefault(tid, [])
                pts.append((cx, cy))
                if len(pts) > self.trail_length:
                    # Trim oldest points in-place
                    del pts[: len(pts) - self.trail_length]

            draw_trails(frame, self._trail_history, self.trail_length)

        # ------------------------------------------------------------------
        # 4b. Draw bounding boxes with labels/scores/track IDs
        # ------------------------------------------------------------------
        if self.show_boxes and instances:
            draw_boxes(
                frame,
                instances,
                show_labels=self.show_labels,
                show_scores=self.show_scores,
                show_track_ids=self.show_track_ids,
                line_width=self.line_width,
                cross_matches=cross_matches,
            )

        # ------------------------------------------------------------------
        # 4c. Draw camera label bar
        # ------------------------------------------------------------------
        if self.camera_label:
            cam_color = self.camera_color if self.camera_color is not None else (255, 100, 60)
            draw_camera_label(frame, self.camera_label, color=cam_color)

        # ------------------------------------------------------------------
        # 5. Wrap result as Image artifact, preserving frame-level metadata
        # ------------------------------------------------------------------
        h, w = frame.shape[:2]
        annotated = Image(
            data=frame,
            width=w,
            height=h,
            color_space="BGR",
            timestamp_ms=image.timestamp_ms,
            frame_id=image.frame_id,
            source_path=image.source_path,
        )

        ctx.record_metric(self.name, "num_instances", len(instances))

        return {self.out: annotated}

    def __repr__(self) -> str:
        return (
            f"AnnotateRT(show_boxes={self.show_boxes!r}, "
            f"show_trails={self.show_trails!r}, "
            f"camera_label={self.camera_label!r}, "
            f"out={self.out!r})"
        )
