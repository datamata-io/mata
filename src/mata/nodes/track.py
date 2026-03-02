"""Track node — temporal object tracking across frames.

Provides object tracking using BYTETrack or simple IoU-based tracking
to maintain temporal object identity across video frames.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.tracks import Track as TrackArtifact
from mata.core.artifacts.tracks import Tracks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Track(Node):
    """Temporal object tracking across frames.

    Tracks objects across video frames using either BYTETrack (if available)
    or a simple IoU-based tracker fallback. Maintains track IDs and handles
    track lifecycle (active/lost/terminated).

    Args:
        using: Name of the tracker provider registered in the context
            (e.g. ``"bytetrack"``, ``"iou_tracker"``).
        dets: Input detections artifact name (default ``"dets"``).
        out: Key under which the output tracks are stored (default ``"tracks"``).
        frame_id: Frame identifier - can be string or will be auto-generated.
        track_thresh: Track activation threshold for new tracks (default 0.5).
        track_buffer: Number of frames to keep lost tracks (default 30).
        match_thresh: IoU threshold for matching detections to tracks (default 0.8).
        name: Optional human-readable node name.

    Inputs:
        detections (Detections): Detection results from current frame.

    Outputs:
        tracks (Tracks): Tracking results with track IDs and state.

    Example:
        ```python
        from mata.nodes import Track

        # Using BYTETrack
        tracker = Track(
            using="bytetrack",
            out="tracks",
            track_thresh=0.5,
            match_thresh=0.8
        )

        # Process video frames
        for frame_idx, frame in enumerate(video_frames):
            result = tracker.run(
                ctx,
                detections=frame_detections,
                frame_id=f"frame_{frame_idx:04d}"
            )
            tracks = result["tracks"]
        ```
    """

    inputs: dict[str, type[Artifact]] = {"detections": Detections}
    outputs: dict[str, type[Artifact]] = {"tracks": Tracks}

    def __init__(
        self,
        using: str,
        dets: str = "dets",
        out: str = "tracks",
        frame_id: str | None = None,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.dets_src = dets
        self.output_name = out
        self.frame_id_template = frame_id
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        # Frame counter for auto-generating frame IDs
        self._frame_counter = 0

    def run(
        self, ctx: ExecutionContext, detections: Detections, frame_id: str | None = None, **kwargs: Any
    ) -> dict[str, Artifact]:
        """Execute tracking on detections from current frame.

        Args:
            ctx: Execution context with providers and metrics.
            detections: Detection results from current frame.
            frame_id: Optional frame identifier. If not provided, auto-generated.
            **kwargs: Additional tracking parameters.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Tracks artifact.

        Raises:
            KeyError: If the tracker provider is not found in context.
        """
        tracker = ctx.get_provider("track", self.provider_name)

        # Determine frame ID
        if frame_id is None:
            if self.frame_id_template:
                current_frame_id = self.frame_id_template.format(frame=self._frame_counter)
            else:
                current_frame_id = f"frame_{self._frame_counter:04d}"
        else:
            current_frame_id = frame_id

        self._frame_counter += 1

        start = time.time()

        # Update tracker with current detections
        tracks = tracker.update(
            detections,
            frame_id=current_frame_id,
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            **kwargs,
        )

        latency_ms = (time.time() - start) * 1000

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_tracks", len(tracks.tracks))
        ctx.record_metric(self.name, "num_active_tracks", len(tracks.get_active_tracks().tracks))
        ctx.record_metric(self.name, "num_lost_tracks", len(tracks.get_lost_tracks().tracks))

        return {self.output_name: tracks}

    def __repr__(self) -> str:
        return (
            f"Track(using='{self.provider_name}', out='{self.output_name}', "
            f"track_thresh={self.track_thresh}, match_thresh={self.match_thresh})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers shared by ByteTrackWrapper and BotSortWrapper
# ---------------------------------------------------------------------------


def _detections_to_detection_results(detections: Detections) -> Any:
    """Convert a graph ``Detections`` artifact to a vendored ``DetectionResults``.

    Only instances that carry a bounding box are forwarded to the tracker.
    Instances without bboxes (segmentation-only, VLM entities) are skipped.

    Args:
        detections: Graph system ``Detections`` artifact.

    Returns:
        ``mata.trackers.byte_tracker.DetectionResults`` compatible with
        ``BYTETracker.update()`` and ``BOTSORT.update()``.
    """
    from mata.trackers.byte_tracker import DetectionResults

    instances = [inst for inst in detections.instances if inst.bbox is not None]
    if not instances:
        return DetectionResults(
            conf=np.empty(0, dtype=np.float32),
            xyxy=np.empty((0, 4), dtype=np.float32),
            xywh=np.empty((0, 4), dtype=np.float32),
            cls=np.empty(0, dtype=np.float32),
        )

    conf = np.array(
        [inst.score if inst.score is not None else 1.0 for inst in instances],
        dtype=np.float32,
    )
    xyxy = np.array([inst.bbox for inst in instances], dtype=np.float32)

    # xyxy → xywh (cx, cy, w, h)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    cls = np.array(
        [inst.label if isinstance(inst.label, (int, float)) else 0 for inst in instances],
        dtype=np.float32,
    )

    return DetectionResults(conf=conf, xyxy=xyxy, xywh=xywh, cls=cls)


def _tracker_array_to_tracks(tracked: np.ndarray, frame_id: str) -> Tracks:
    """Convert a vendored tracker output array to a ``Tracks`` artifact.

    Args:
        tracked: ``(N, 8)`` float array with columns
            ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
        frame_id: Frame identifier string to embed in the returned artifact.

    Returns:
        ``Tracks`` artifact with one ``Track`` per active confirmed track.
    """
    track_list: list[TrackArtifact] = []
    for row in tracked:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        track_id = int(row[4])
        score = float(np.clip(row[5], 0.0, 1.0))
        cls_id = int(row[6])

        # Guard against degenerate boxes that would fail Track validation
        if x2 <= x1 or y2 <= y1:
            continue

        track_list.append(
            TrackArtifact(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                score=score,
                label=str(cls_id),
                age=1,
                state="active",
                label_id=cls_id,
            )
        )

    return Tracks(tracks=track_list, frame_id=frame_id)


# ---------------------------------------------------------------------------
# Tracker wrappers
# ---------------------------------------------------------------------------


class ByteTrackWrapper:
    """Wrapper for vendored ``mata.trackers.BYTETracker`` as a graph provider.

    Uses the vendored ByteTrack implementation (always available, no external
    dependency required).  For a zero-dependency fallback, use
    :class:`SimpleIOUTracker` instead.

    Args:
        track_buffer: Frames to keep lost tracks alive (default 30).
        frame_rate: Video frame rate for track-lifetime calculation (default 30).
        track_thresh: High-confidence threshold for first association (default 0.5).
        low_thresh: Low-confidence threshold for second association (default 0.1).
        new_track_thresh: Minimum score to initialise a new track (default 0.6).
        match_thresh: Maximum IoU cost for Hungarian matching (default 0.8).
        fuse_score: Fuse detection scores into the IoU cost (default True).

    Example:
        ```python
        tracker = ByteTrackWrapper(track_buffer=30, frame_rate=30)
        ctx.providers["track"]["bytetrack"] = tracker
        tracks = tracker.update(detections, frame_id="frame_0001")
        ```
    """

    def __init__(
        self,
        track_buffer: int = 30,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ):
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score

        from mata.trackers.byte_tracker import BYTETracker

        self.tracker = BYTETracker(
            args={
                "track_high_thresh": track_thresh,
                "track_low_thresh": low_thresh,
                "new_track_thresh": new_track_thresh,
                "track_buffer": track_buffer,
                "match_thresh": match_thresh,
                "fuse_score": fuse_score,
            },
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: Detections,
        frame_id: str,
        track_thresh: float | None = None,
        track_buffer: int | None = None,
        match_thresh: float | None = None,
        **kwargs: Any,
    ) -> Tracks:
        """Update tracker with detections from the current frame.

        Args:
            detections: Graph ``Detections`` artifact for this frame.
            frame_id: Unique frame identifier (used as label in the output).
            track_thresh: Ignored — threshold is fixed at construction time.
            track_buffer: Ignored — buffer is fixed at construction time.
            match_thresh: Ignored — threshold is fixed at construction time.
            **kwargs: Accepted for API compatibility; not forwarded.

        Returns:
            ``Tracks`` artifact with active confirmed tracks.
        """
        det_results = _detections_to_detection_results(detections)
        tracked_array = self.tracker.update(det_results)
        return _tracker_array_to_tracks(tracked_array, frame_id)

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.tracker.reset()


class BotSortWrapper:
    """Wrapper for vendored ``mata.trackers.BOTSORT`` as a graph provider.

    BotSort extends BYTETracker with:

    * **KalmanFilterXYWH** — width-height state instead of aspect-height.
    * **Global Motion Compensation (GMC)** — sparse optical flow adjusts
      predictions for camera movement.
    * **ReID stubs** — appearance fields are reserved for v1.9+.

    Args:
        track_buffer: Frames to keep lost tracks alive (default 30).
        frame_rate: Video frame rate for track-lifetime calculation (default 30).
        track_thresh: High-confidence threshold for first association (default 0.5).
        low_thresh: Low-confidence threshold for second association (default 0.1).
        new_track_thresh: Minimum score to initialise a new track (default 0.6).
        match_thresh: Maximum IoU cost for Hungarian matching (default 0.8).
        fuse_score: Fuse detection scores into the IoU cost (default True).
        gmc_method: GMC algorithm — ``'sparseOptFlow'`` or ``None`` to disable
            (default ``'sparseOptFlow'``).
        proximity_thresh: Minimum IoU required before ReID distance is computed
            (default 0.5).
        appearance_thresh: Maximum appearance distance for ReID match (default 0.25).
        with_reid: Enable ReID (requires encoder model — disabled in v1.8,
            default False).

    Example:
        ```python
        tracker = BotSortWrapper(track_buffer=30, frame_rate=30)
        ctx.providers["track"]["botsort"] = tracker
        tracks = tracker.update(detections, frame_id="frame_0001")
        ```
    """

    def __init__(
        self,
        track_buffer: int = 30,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        gmc_method: str | None = "sparseOptFlow",
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        with_reid: bool = False,
    ):
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.gmc_method = gmc_method
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid

        from mata.trackers.bot_sort import BOTSORT

        self.tracker = BOTSORT(
            args={
                "track_high_thresh": track_thresh,
                "track_low_thresh": low_thresh,
                "new_track_thresh": new_track_thresh,
                "track_buffer": track_buffer,
                "match_thresh": match_thresh,
                "fuse_score": fuse_score,
                "gmc_method": gmc_method,
                "proximity_thresh": proximity_thresh,
                "appearance_thresh": appearance_thresh,
                "with_reid": with_reid,
            },
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: Detections,
        frame_id: str,
        track_thresh: float | None = None,
        track_buffer: int | None = None,
        match_thresh: float | None = None,
        **kwargs: Any,
    ) -> Tracks:
        """Update tracker with detections from the current frame.

        Args:
            detections: Graph ``Detections`` artifact for this frame.
            frame_id: Unique frame identifier (used as label in the output).
            track_thresh: Ignored — threshold is fixed at construction time.
            track_buffer: Ignored — buffer is fixed at construction time.
            match_thresh: Ignored — threshold is fixed at construction time.
            **kwargs: Accepted for API compatibility; not forwarded.

        Returns:
            ``Tracks`` artifact with active confirmed tracks.
        """
        det_results = _detections_to_detection_results(detections)
        tracked_array = self.tracker.update(det_results)
        return _tracker_array_to_tracks(tracked_array, frame_id)

    def reset(self) -> None:
        """Reset tracker state (including GMC) for a new video sequence."""
        self.tracker.reset()


class SimpleIOUTracker:
    """Simple IoU-based tracker as a standalone provider.

    Provides a lightweight tracking solution without external dependencies.
    Suitable for basic tracking scenarios where BYTETrack is not available.

    Args:
        track_buffer: Number of frames to keep lost tracks (default 30).
        track_thresh: Minimum detection score for track initialization (default 0.5).
        match_thresh: IoU threshold for matching (default 0.8).
    """

    def __init__(self, track_buffer: int = 30, track_thresh: float = 0.5, match_thresh: float = 0.8):
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh

        self._tracks: dict[int, dict[str, Any]] = {}
        self._next_track_id = 1
        self._frame_count = 0

    def update(
        self,
        detections: Detections,
        frame_id: str,
        track_thresh: float | None = None,
        track_buffer: int | None = None,
        match_thresh: float | None = None,
        **kwargs: Any,
    ) -> Tracks:
        """Update tracker with new frame detections."""
        # Use provided parameters or defaults
        current_track_thresh = track_thresh or self.track_thresh
        current_track_buffer = track_buffer or self.track_buffer
        current_match_thresh = match_thresh or self.match_thresh

        self._frame_count += 1

        # Filter detections by score threshold
        valid_detections = [
            inst for inst in detections.instances if (inst.score is not None and inst.score >= current_track_thresh)
        ]

        # Calculate IoU matrix between detections and existing tracks
        matches, unmatched_dets, unmatched_trks = self._associate(valid_detections, current_match_thresh)

        # Update matched tracks
        for det_idx, trk_id in matches:
            detection = valid_detections[det_idx]
            track_info = self._tracks[trk_id]

            # Update track with new detection
            history = track_info.get("history", [])
            if len(history) >= 10:  # Keep only last 10 positions
                history = history[-9:]

            track_info.update(
                {
                    "bbox": detection.bbox,
                    "score": detection.score or 1.0,
                    "label": detection.label_name if detection.label_name else str(detection.label),
                    "age": track_info["age"] + 1,
                    "state": "active",
                    "last_frame": self._frame_count,
                    "history": history + [track_info["bbox"]],
                }
            )

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = valid_detections[det_idx]
            self._tracks[self._next_track_id] = {
                "track_id": self._next_track_id,
                "bbox": detection.bbox,
                "score": detection.score or 1.0,
                "label": detection.label_name if detection.label_name else str(detection.label),
                "age": 1,
                "state": "active",
                "last_frame": self._frame_count,
                "history": [],
            }
            self._next_track_id += 1

        # Mark unmatched tracks as lost
        for trk_id in unmatched_trks:
            if trk_id in self._tracks:
                track_info = self._tracks[trk_id]
                if track_info["state"] == "active":  # Only mark active tracks as lost
                    track_info["state"] = "lost"
                track_info["age"] += 1

        # Remove old lost tracks
        tracks_to_remove = []
        for trk_id, track_info in self._tracks.items():
            if track_info["state"] == "lost" and self._frame_count - track_info["last_frame"] > current_track_buffer:
                tracks_to_remove.append(trk_id)

        for trk_id in tracks_to_remove:
            del self._tracks[trk_id]

        # Convert to Track artifacts
        track_list = []
        for track_info in self._tracks.values():
            track = TrackArtifact(
                track_id=track_info["track_id"],
                bbox=track_info["bbox"],
                score=track_info["score"],
                label=track_info["label"],
                age=track_info["age"],
                state=track_info["state"],
                history=track_info.get("history"),
            )
            track_list.append(track)

        return Tracks(tracks=track_list, frame_id=frame_id)

    def _associate(
        self, detections: list[Any], iou_threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections to tracks using IoU."""
        if not detections or not self._tracks:
            return [], list(range(len(detections))), list(self._tracks.keys())

        # Only consider active tracks for matching
        active_tracks = {
            track_id: track_info for track_id, track_info in self._tracks.items() if track_info["state"] == "active"
        }

        if not active_tracks:
            return [], list(range(len(detections))), list(self._tracks.keys())

        # Calculate IoU matrix
        track_ids = list(active_tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)

        for d, detection in enumerate(detections):
            det_box = detection.bbox
            for t, track_id in enumerate(track_ids):
                track_box = active_tracks[track_id]["bbox"]
                iou_matrix[d, t] = self._calculate_iou(det_box, track_box)

        # Simple greedy matching using highest IoU first
        matched_pairs = []
        used_detections = set()
        used_tracks = set()

        # Find matches above threshold
        while True:
            # Find maximum IoU
            max_iou = 0
            max_det, max_track = -1, -1

            for d in range(len(detections)):
                if d in used_detections:
                    continue
                for t in range(len(track_ids)):
                    if t in used_tracks:
                        continue
                    if iou_matrix[d, t] > max_iou and iou_matrix[d, t] >= iou_threshold:
                        max_iou = iou_matrix[d, t]
                        max_det, max_track = d, t

            if max_det == -1:  # No more matches found
                break

            matched_pairs.append((max_det, track_ids[max_track]))
            used_detections.add(max_det)
            used_tracks.add(max_track)

        # Find unmatched detections and tracks
        unmatched_dets = [d for d in range(len(detections)) if d not in used_detections]
        unmatched_trks = [track_ids[t] for t in range(len(track_ids)) if t not in used_tracks]

        # Add all inactive tracks as unmatched (they will be marked as lost)
        inactive_tracks = [track_id for track_id, track_info in self._tracks.items() if track_info["state"] != "active"]
        unmatched_trks.extend(inactive_tracks)

        return matched_pairs, unmatched_dets, unmatched_trks

    def _calculate_iou(self, box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0
