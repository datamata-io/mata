"""ByteTrack multi-object tracker.

Contains:
    STrack  — Single tracked object with Kalman filter state and lifecycle.
    BYTETracker — (Task A4) Full two-stage ByteTrack algorithm.

STrack is ported from Ultralytics' ``ultralytics/trackers/byte_tracker.py``
(MIT-compatible).  The ``BYTETracker`` class will be appended in Task A4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.trackers.basetrack import BaseTrack, TrackState

# ---------------------------------------------------------------------------
# STrack
# ---------------------------------------------------------------------------


class STrack(BaseTrack):
    """Single object track with Kalman filter state management.

    Internally the bounding box is stored as ``tlwh`` (top-left-x,
    top-left-y, width, height).  The Kalman filter operates in ``xyah``
    space (center_x, center_y, aspect_ratio, height).  Property accessors
    expose ``tlwh``, ``xyxy``, and ``xywh`` views derived from the live
    filter state (or from the frozen ``_tlwh`` before activation).

    Class attributes:
        shared_kalman (KalmanFilterXYAH): Single KF instance shared across
            **all** ``STrack`` objects for batch prediction in
            :meth:`multi_predict`.  Initialised lazily on first use to
            respect the MATA lazy-import convention.

    Attributes:
        _tlwh (np.ndarray): Frozen bbox ``[x1, y1, w, h]`` set at creation.
            Used verbatim until the track is activated and the Kalman mean
            takes over.
        kalman_filter (KalmanFilterXYAH | None): Per-track KF instance
            (same object as ``shared_kalman`` — set by :meth:`activate`).
        mean (np.ndarray | None): Kalman state mean ``(8,)`` in xyah space.
        covariance (np.ndarray | None): Kalman state covariance ``(8, 8)``.
        is_activated (bool): True once the track has been confirmed.
        score (float): Detection confidence for the latest observation.
        tracklet_len (int): Consecutive frames the track has been matched.
        cls (Any): Class label (int index or string).
        idx (int): Index of the detection that spawned / last updated this
            track within its detection frame.
        angle (float | None): Optional rotation angle (OBB support).
    """

    # Shared KF — lazy init to avoid circular import at module load time.
    shared_kalman: Any = None  # KalmanFilterXYAH; set on first use

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, xywh: list[float] | np.ndarray, score: float, cls: Any) -> None:
        """Initialise an *unactivated* STrack from a raw detection.

        Args:
            xywh: Detection bounding box as ``[cx, cy, w, h]``, optionally
                extended to ``[cx, cy, w, h, angle]`` (5 elements) or
                ``[cx, cy, w, h, angle, idx]`` (6 elements).  Extra
                elements beyond 4 are treated as angle and idx respectively.
            score: Detection confidence score ``[0, 1]``.
            cls: Class label (integer index or string).
        """
        super().__init__()

        xywh = np.asarray(xywh, dtype=np.float64)

        # Convert center-format to top-left format and store frozen copy.
        cx, cy, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
        self._tlwh: np.ndarray = np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float64)

        # Optional extended fields.
        self.angle: float | None = float(xywh[4]) if len(xywh) > 4 else None
        self.idx: int = int(xywh[5]) if len(xywh) > 5 else 0

        # Kalman state (None until activate()).
        self.kalman_filter: Any = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        # Lifecycle.
        self.is_activated: bool = False
        self.score: float = float(score)
        self.tracklet_len: int = 0
        self.cls: Any = cls

    # ------------------------------------------------------------------
    # Class-level shared Kalman filter
    # ------------------------------------------------------------------

    @classmethod
    def _get_shared_kalman(cls) -> Any:
        """Return (and lazily initialise) the shared KalmanFilterXYAH."""
        if cls.shared_kalman is None:
            from mata.trackers.utils.kalman_filter import KalmanFilterXYAH

            cls.shared_kalman = KalmanFilterXYAH()
        return cls.shared_kalman

    # ------------------------------------------------------------------
    # Single-track prediction
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Advance this track's Kalman state by one time step (in-place).

        Zeroes the velocity component of the mean when the track has been
        lost for one or more frames (avoids drift from stale velocity).
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # zero height velocity for lost tracks
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    @staticmethod
    def multi_predict(stracks: list[STrack]) -> None:
        """Batch-predict all active tracks using the shared Kalman filter.

        Modifies each track's ``mean`` and ``covariance`` in-place.
        Tracks that have not been activated (``mean is None``) are skipped.

        Each track is predicted independently via the shared KF instance.
        Using the shared instance (rather than per-track instances) keeps
        the method self-contained and mirrors the Ultralytics implementation's
        intent: one stateless KF object does all arithmetic.

        Args:
            stracks: Tracks to predict forward by one time step.
        """
        if not stracks:
            return

        shared_kf = STrack._get_shared_kalman()

        for st in stracks:
            if st.mean is None:
                continue
            mean_state = st.mean.copy()
            # Zero height velocity for lost/non-tracked states to suppress drift.
            if st.state != TrackState.Tracked:
                mean_state[7] = 0
            st.mean, st.covariance = shared_kf.predict(mean_state, st.covariance)

    @staticmethod
    def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)) -> None:  # noqa: N803
        """Apply a 2×3 affine (homography) matrix to all track states.

        Used for Global Motion Compensation (GMC) — adjusts Kalman means
        for camera movement so track predictions stay aligned with the scene.

        Only position and velocity in x/y are transformed; aspect ratio,
        height and their velocities remain unchanged.

        Args:
            stracks: Tracks to transform (modified in-place).
            H: ``(2, 3)`` affine transformation matrix representing the
               camera motion from the previous frame to the current frame.
        """
        if not stracks:
            return

        R = H[:2, :2]  # rotation / scale part  # noqa: N806
        t = H[:2, 2]  # translation part

        for st in stracks:
            if st.mean is None:
                continue
            # Transform position (cx, cy).
            st.mean[:2] = R @ st.mean[:2] + t
            # Transform velocity (vcx, vcy) — no translation component.
            st.mean[4:6] = R @ st.mean[4:6]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, kalman_filter: Any, frame_id: int) -> None:
        """Activate the track for the first time.

        Assigns a new unique track ID, initialises the Kalman filter, and
        transitions to the *Tracked* state.

        Args:
            kalman_filter: A ``KalmanFilterXYAH`` instance (typically the
                shared instance stored on the ``BYTETracker``).
            frame_id: Current frame number.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # On frame 1 every track is immediately confirmed; on later frames
        # new tracks start unconfirmed and are confirmed on their second hit.
        self.is_activated = frame_id == 1
        self.start_frame = frame_id
        self.frame_id = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a previously lost track with a new detection.

        Runs a Kalman update step with the new detection, resets tracklet
        length, and restores *Tracked* state.

        Args:
            new_track: Freshly detected track carrying the new observation.
            frame_id: Current frame number.
            new_id: If ``True``, assign a fresh track ID (used for
                identity switches caused by long occlusion).
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.angle = getattr(new_track, "angle", None)

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update the track with a new matched detection (active track).

        Runs a Kalman update step, increments the tracklet length, and
        copies detection metadata from ``new_track``.

        Args:
            new_track: Freshly detected track carrying the new observation.
            frame_id: Current frame number.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.angle = getattr(new_track, "angle", None)

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a tlwh box to xyah format for Kalman filter input.

        This is the measurement function ``z = h(x)`` for STrack — it maps
        the top-left bbox representation to the Kalman measurement space.

        Args:
            tlwh: Bounding box ``[x1, y1, w, h]``.

        Returns:
            np.ndarray: ``[cx, cy, a, h]`` where ``a = w / h``.
        """
        return self.tlwh_to_xyah(tlwh)

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert ``[x1, y1, w, h]`` to ``[cx, cy, a, h]``.

        Args:
            tlwh: Array ``[top_left_x, top_left_y, width, height]``.

        Returns:
            np.ndarray: ``[center_x, center_y, aspect_ratio, height]``
                where ``aspect_ratio = width / height``.
        """
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[0] += ret[2] / 2.0  # cx = x1 + w/2
        ret[1] += ret[3] / 2.0  # cy = y1 + h/2
        ret[2] /= ret[3]  # a  = w / h
        return ret

    # ------------------------------------------------------------------
    # Properties — live bbox views
    # ------------------------------------------------------------------

    @property
    def tlwh(self) -> np.ndarray:
        """Bounding box in ``[x1, y1, w, h]`` (top-left + size) format.

        Before activation: returns the frozen ``_tlwh`` set at construction.
        After activation: recovers ``[x1, y1, w, h]`` from the Kalman mean
        ``[cx, cy, a, h]``:
            - ``w = a * h``
            - ``x1 = cx - w/2``,  ``y1 = cy - h/2``
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()  # [cx, cy, a, h]
        ret[2] *= ret[3]  # w = a * h
        ret[:2] -= ret[2:] / 2.0  # top-left: x1 = cx - w/2, y1 = cy - h/2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Bounding box in ``[x1, y1, x2, y2]`` (corner) format.

        Derived from :attr:`tlwh` as ``[x1, y1, x1+w, y1+h]``.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]  # x2 = x1 + w,  y2 = y1 + h
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Bounding box in ``[cx, cy, w, h]`` (center + size) format.

        Derived from :attr:`tlwh` as ``[x1+w/2, y1+h/2, w, h]``.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0  # shift to center
        return ret

    @property
    def result(self) -> list[float]:
        """Output format for downstream consumers.

        Returns:
            list: ``[x1, y1, x2, y2, track_id, score, cls, idx]``
                — an 8-element list compatible with the MATA ``VisionResult``
                conversion in :class:`mata.adapters.tracking_adapter.TrackingAdapter`.
        """
        coords = self.xyxy.tolist()
        return coords + [self.track_id, self.score, self.cls, self.idx]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"STrack(id={self.track_id}, state={self.state}, "
            f"frame={self.frame_id}, score={self.score:.3f}, "
            f"cls={self.cls}, tlwh={self.tlwh.tolist()})"
        )


# ---------------------------------------------------------------------------
# DetectionResults — adapter wrapping MATA VisionResult → tracker input
# ---------------------------------------------------------------------------


class DetectionResults:
    """Lightweight adapter that converts MATA detection outputs to tracker input.

    Wraps a :class:`~mata.core.types.VisionResult` — or raw arrays — into an
    object exposing ``.conf``, ``.xyxy``, ``.xywh``, and ``.cls`` attributes
    that :class:`BYTETracker` and :class:`~mata.trackers.bot_sort.BOTSORT`
    expect.

    The class also supports boolean-mask and integer-index slicing so that
    :meth:`BYTETracker.update` can split high/low confidence detection sets
    without copying full arrays.

    Args:
        conf:    1-D array of confidence scores ``(N,)``.
        xyxy:    2-D bbox array ``(N, 4)`` in xyxy format.
        xywh:    2-D bbox array ``(N, 4)`` in center format (cx, cy, w, h).
        cls:     1-D array of class IDs ``(N,)``.
        indices: Optional 1-D array of original detection indices ``(N,)``.
            When *None* indices are assigned as ``0, 1, …, N-1``.

    Example::

        from mata.core.types import VisionResult
        dr = DetectionResults.from_vision_result(result)
        high = dr[dr.conf >= 0.5]
    """

    def __init__(
        self,
        conf: np.ndarray,
        xyxy: np.ndarray,
        xywh: np.ndarray,
        cls: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        self._conf = np.asarray(conf, dtype=np.float32).ravel()
        n = len(self._conf)
        self._xyxy = np.asarray(xyxy, dtype=np.float32).reshape(n, 4) if n > 0 else np.empty((0, 4), dtype=np.float32)
        self._xywh = np.asarray(xywh, dtype=np.float32).reshape(n, 4) if n > 0 else np.empty((0, 4), dtype=np.float32)
        self._cls = np.asarray(cls, dtype=np.float32).ravel()
        self._indices: np.ndarray = (
            np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64).ravel()
        )
        # Optional per-detection appearance feature vectors (for ReID).
        # Each entry is either None or a 1-D float32 array.
        self.features: list = [None] * n

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def conf(self) -> np.ndarray:
        """Confidence scores ``(N,)`` float32."""
        return self._conf

    @property
    def xyxy(self) -> np.ndarray:
        """Bounding boxes ``(N, 4)`` float32 in [x1, y1, x2, y2] format."""
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes ``(N, 4)`` float32 in [cx, cy, w, h] format."""
        return self._xywh

    @property
    def cls(self) -> np.ndarray:
        """Class IDs ``(N,)`` float32."""
        return self._cls

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, idx: Any) -> DetectionResults:
        """Slice by boolean mask, integer index, or slice object.

        Returns a new :class:`DetectionResults` with a subset of detections.
        Original indices (within the *parent* object) are preserved so that
        :meth:`BYTETracker.init_track` can embed them in :class:`STrack` as
        ``idx``.
        """
        # Slice features using numpy object-array indexing so that both
        # boolean masks and integer-index arrays work uniformly.
        feats_arr = np.empty(len(self.features), dtype=object)
        for _i, _f in enumerate(self.features):
            feats_arr[_i] = _f
        sliced_feats = list(feats_arr[idx])
        sliced = DetectionResults(
            conf=self._conf[idx],
            xyxy=self._xyxy[idx],
            xywh=self._xywh[idx],
            cls=self._cls[idx],
            indices=self._indices[idx],
        )
        sliced.features = sliced_feats
        return sliced

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vision_result(cls, vision_result: Any) -> DetectionResults:
        """Construct from a MATA :class:`~mata.core.types.VisionResult`.

        Only instances that carry a bounding box are included.  Instances
        without bboxes (e.g., segmentation-only) are silently skipped.

        Args:
            vision_result: A :class:`~mata.core.types.VisionResult` with
                a populated ``.instances`` list.

        Returns:
            :class:`DetectionResults` wrapping the detection data.
        """
        instances = [inst for inst in vision_result.instances if inst.bbox is not None]
        if not instances:
            return cls(
                conf=np.empty(0, dtype=np.float32),
                xyxy=np.empty((0, 4), dtype=np.float32),
                xywh=np.empty((0, 4), dtype=np.float32),
                cls=np.empty(0, dtype=np.float32),
            )

        conf = np.array([inst.score for inst in instances], dtype=np.float32)
        xyxy = np.array([inst.bbox for inst in instances], dtype=np.float32)  # (N,4)

        # xyxy → xywh (center format)
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # cx
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # cy
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # w
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # h

        cls_ids = np.array(
            [inst.label if inst.label is not None else 0 for inst in instances],
            dtype=np.float32,
        )
        return cls(conf=conf, xyxy=xyxy, xywh=xywh, cls=cls_ids)

    @classmethod
    def empty(cls) -> DetectionResults:
        """Return an empty :class:`DetectionResults` (zero detections)."""
        return cls(
            conf=np.empty(0, dtype=np.float32),
            xyxy=np.empty((0, 4), dtype=np.float32),
            xywh=np.empty((0, 4), dtype=np.float32),
            cls=np.empty(0, dtype=np.float32),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"DetectionResults(n={len(self)})"


# ---------------------------------------------------------------------------
# BYTETracker — two-stage IoU + Kalman multi-object tracker
# ---------------------------------------------------------------------------


class _TrackerArgs:
    """Lightweight configuration holder for :class:`BYTETracker`.

    Accepts keyword arguments matching the ByteTrack YAML schema.  Default
    values mirror the Ultralytics ``bytetrack.yaml`` configuration.

    Args:
        track_high_thresh: Minimum score for first-stage association (default 0.5).
        track_low_thresh:  Minimum score for second-stage association (default 0.1).
        new_track_thresh:  Minimum score to initialise a new track (default 0.6).
        track_buffer:      Frames to keep a lost track before removal (default 30).
        match_thresh:      Maximum IoU cost for first-stage assignment (default 0.8).
        fuse_score:        If ``True``, fuse detection scores with IoU cost (default True).
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        **_extra: Any,  # Silently accept BotSort-only fields for compatibility
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score

    @classmethod
    def from_any(cls, args: Any) -> Any:
        """Normalise *args* to an object that has the required attributes.

        * If *args* is already a :class:`_TrackerArgs` (or any object with
          the needed fields), return it unchanged.
        * If *args* is a :class:`dict`, construct a :class:`_TrackerArgs`
          from it.

        Args:
            args: A :class:`_TrackerArgs`, an object with tracker attributes,
                or a ``dict``.

        Returns:
            Object with ``.track_high_thresh``, ``.track_low_thresh``, etc.
        """
        if isinstance(args, dict):
            return cls(**args)
        return args  # Trust that caller provides correct attributes


class BYTETracker:
    """BYTETracker multi-object tracker.

    Implements the ByteTrack two-stage association algorithm:

    1. **First association** — high-confidence detections ↔ all active+lost
       tracks via IoU distance (optionally fused with detection scores) and
       the Hungarian algorithm.
    2. **Second association** — low-confidence detections ↔ remaining *tracked*
       (not lost) tracks via IoU-only matching.  This recovers objects that are
       temporarily occluded or partially visible.
    3. **Unconfirmed handling** — tracks seen for exactly one frame are matched
       against remaining high-conf detections; unmatched ones are removed.
    4. **New track init** — high-conf detections still unmatched become new
       (unconfirmed) tracks if their score exceeds ``new_track_thresh``.
    5. **Track lifecycle** — tracks not updated for ``max_time_lost`` frames
       are permanently removed.

    Args:
        args: Tracker configuration.  May be a :class:`_TrackerArgs` instance,
            any object with the required attributes, or a ``dict``.
        frame_rate: Video frame rate used to compute ``max_time_lost``
            ``= frame_rate / 30 * track_buffer`` (default 30).

    Attributes:
        tracked_stracks:  List of currently active :class:`STrack` objects.
        lost_stracks:     Recently lost tracks that may still be re-found.
        removed_stracks:  Permanently removed tracks (capped at 1000).
        frame_id:         Most recently processed frame number.
        max_time_lost:    Frame budget before a lost track is removed.
        kalman_filter:    Shared :class:`~mata.trackers.utils.kalman_filter.KalmanFilterXYAH`.
    """

    def __init__(self, args: Any, frame_rate: int = 30) -> None:
        self.args = _TrackerArgs.from_any(args)
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id: int = 0
        self.max_time_lost: int = int(frame_rate / 30.0 * self.args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()

    # ------------------------------------------------------------------
    # Overridable hooks (used by BOTSORT subclass)
    # ------------------------------------------------------------------

    def get_kalmanfilter(self) -> Any:
        """Return a new :class:`~mata.trackers.utils.kalman_filter.KalmanFilterXYAH`.

        Subclasses (e.g. :class:`~mata.trackers.bot_sort.BOTSORT`) override
        this to return a :class:`~mata.trackers.utils.kalman_filter.KalmanFilterXYWH`
        instance.
        """
        from mata.trackers.utils.kalman_filter import KalmanFilterXYAH

        return KalmanFilterXYAH()

    def init_track(self, results: Any, img: np.ndarray | None = None) -> list[STrack]:
        """Convert detection results to a list of *unactivated* :class:`STrack` objects.

        Each detection becomes one :class:`STrack`.  The original detection
        index (within the parent :class:`DetectionResults` object) is encoded
        into the extended ``xywh`` array so that downstream consumers can
        trace each track back to its originating detection.

        Subclasses (e.g. :class:`~mata.trackers.bot_sort.BOTSORT`) override
        this to produce :class:`~mata.trackers.bot_sort.BOTrack` instances.

        Args:
            results: A :class:`DetectionResults` (or compatible object) with
                ``.conf``, ``.xywh``, ``.cls``, and optionally ``._indices``.
            img: Unused in base ByteTrack; reserved for subclass hooks.

        Returns:
            List of :class:`STrack` objects, one per detection.
        """
        if len(results) == 0:
            return []

        stracks: list[STrack] = []
        for i in range(len(results)):
            xywh = results.xywh[i]  # [cx, cy, w, h]
            score = float(results.conf[i])
            cls = results.cls[i]
            # Encode original detection index as the 6th element so that
            # STrack.result can expose [x1,y1,x2,y2,track_id,score,cls,idx].
            orig_idx = float(results._indices[i] if hasattr(results, "_indices") else i)
            xywh_ext = np.array(
                [xywh[0], xywh[1], xywh[2], xywh[3], 0.0, orig_idx],
                dtype=np.float64,
            )
            stracks.append(STrack(xywh_ext, score, cls))
        return stracks

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Compute association cost matrix between *tracks* and *detections*.

        By default uses IoU distance, optionally fused with detection
        confidence scores when ``args.fuse_score`` is ``True``.

        Subclasses may override to blend additional cues (e.g. appearance).

        Args:
            tracks:     Active/lost :class:`STrack` objects (rows).
            detections: Candidate :class:`STrack` objects from current frame
                (columns).

        Returns:
            ``(len(tracks), len(detections))`` float32 cost matrix.
        """
        from mata.trackers.utils.matching import fuse_score, iou_distance

        dists = iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]) -> None:
        """Batch-predict all tracks one time step forward via Kalman filter.

        Delegates to :meth:`STrack.multi_predict` which uses the shared KF
        instance.  Subclasses that use a different KF variant
        (e.g. :class:`~mata.trackers.bot_sort.BOTrack`) override this.

        Args:
            tracks: Tracks to advance in-place.
        """
        STrack.multi_predict(tracks)

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------

    def update(self, results: Any, img: np.ndarray | None = None) -> np.ndarray:
        """Update tracker state with detections from the current frame.

        Implements the full ByteTrack 6-step algorithm:

        1. Split detections into high / low confidence tiers.
        2. **First association** — high-conf dets ↔ all active+lost tracks
           (Kalman-predicted) via :meth:`get_dists` + Hungarian matching.
        3. **Second association** — low-conf dets ↔ remaining *tracked* (not
           lost) tracks via IoU-only matching at a fixed 0.5 threshold.
        4. **Unconfirmed matching** — remaining high-conf dets ↔ unconfirmed
           (1-frame-old) tracks at a fixed 0.7 threshold.
        5. **New track init** — unmatched high-conf dets above
           ``new_track_thresh`` become new unconfirmed tracks.
        6. **Cleanup** — lost tracks older than ``max_time_lost`` are removed;
           duplicate tracks (IoU > 0.85) are pruned.

        Args:
            results: Detection results for the current frame.  Must expose
                ``.conf``, ``.xyxy``, ``.xywh``, and ``.cls`` array attributes
                and support ``len()``.  Pass a :class:`DetectionResults`
                wrapping a :class:`~mata.core.types.VisionResult`,
                or any compatible object.
            img: Optional current frame image array.  The base ByteTrack does
                not use it; it is reserved for GMC in the
                :class:`~mata.trackers.bot_sort.BOTSORT` subclass.

        Returns:
            ``np.ndarray`` of shape ``(N, 8)`` — one row per active confirmed
            track — with columns ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
            Returns an empty ``(0, 8)`` array when no tracks are confirmed.
        """
        from mata.trackers.utils.matching import iou_distance, linear_assignment

        self.frame_id += 1

        activated_stracks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks_new: list[STrack] = []
        removed_stracks_new: list[STrack] = []

        # ------------------------------------------------------------------ #
        # Step 1: Split detections by confidence threshold                     #
        # ------------------------------------------------------------------ #
        if len(results) > 0:
            scores = results.conf  # (N,)
            high_mask: np.ndarray = scores >= self.args.track_high_thresh
            low_mask: np.ndarray = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)
            dets_high = results[high_mask]
            dets_low = results[low_mask]
        else:
            dets_high = DetectionResults.empty()
            dets_low = DetectionResults.empty()

        detections: list[STrack] = self.init_track(dets_high, img)
        detections_second: list[STrack] = self.init_track(dets_low, img)

        # ------------------------------------------------------------------ #
        # Step 2: First association — high-conf dets ↔ active+lost pool       #
        # ------------------------------------------------------------------ #
        # Partition tracked_stracks into confirmed (is_activated) and
        # unconfirmed (seen for exactly 1 frame, not yet confirmed).
        unconfirmed: list[STrack] = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_confirmed: list[STrack] = [t for t in self.tracked_stracks if t.is_activated]
        # Pool for first association: confirmed tracked + recently lost
        strack_pool = self.joint_stracks(tracked_confirmed, self.lost_stracks)

        # Advance all pool tracks by one time step (Kalman prediction / prior)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, unmatched_a, unmatched_b = linear_assignment(dists, self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # Re-associate a previously lost track
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ------------------------------------------------------------------ #
        # Step 3: Second association — low-conf dets ↔ remaining tracked      #
        # ------------------------------------------------------------------ #
        # Only *actively tracked* (not lost) tracks participate in round 2.
        r_tracked: list[STrack] = [strack_pool[i] for i in unmatched_a if strack_pool[i].state == TrackState.Tracked]

        dists2 = iou_distance(r_tracked, detections_second)
        matches2, unmatched_a2, _ = linear_assignment(dists2, thresh=0.5)

        for itracked, idet in matches2:
            r_tracked[itracked].update(detections_second[idet], self.frame_id)
            activated_stracks.append(r_tracked[itracked])

        for it in unmatched_a2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks_new.append(track)

        # ------------------------------------------------------------------ #
        # Step 4: Unconfirmed track matching                                   #
        # ------------------------------------------------------------------ #
        # Match remaining high-conf detections against unconfirmed tracks.
        # Unconfirmed tracks that still don't match are removed immediately.
        detections_remain: list[STrack] = [detections[i] for i in unmatched_b]

        dists3 = self.get_dists(unconfirmed, detections_remain)
        matches3, unmatched_a3, unmatched_b3 = linear_assignment(dists3, thresh=0.7)

        for itracked, idet in matches3:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in unmatched_a3:
            unconfirmed[it].mark_removed()
            removed_stracks_new.append(unconfirmed[it])

        # ------------------------------------------------------------------ #
        # Step 5: Initialise new tracks from unmatched high-conf detections    #
        # ------------------------------------------------------------------ #
        for idet in unmatched_b3:
            track = detections_remain[idet]
            if track.score >= self.args.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # ------------------------------------------------------------------ #
        # Step 6: Clean up — mark stale lost tracks as removed                 #
        # ------------------------------------------------------------------ #
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks_new.append(track)

        # ------------------------------------------------------------------ #
        # Merge and book-keep track lists                                       #
        # ------------------------------------------------------------------ #
        # Keep only still-tracked objects in tracked list.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        # Add newly-lost tracks; remove any that have been re-found or removed.
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = self.lost_stracks + lost_stracks_new
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)

        # Accumulate removed tracks, capped at 1000 to prevent memory growth.
        self.removed_stracks = self.removed_stracks + removed_stracks_new
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        # Prune duplicate tracks (same object, different ID due to ID switch).
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # ------------------------------------------------------------------ #
        # Build output array                                                    #
        # ------------------------------------------------------------------ #
        active = [t for t in self.tracked_stracks if t.is_activated]
        if not active:
            return np.empty((0, 8), dtype=np.float64)
        return np.array([t.result for t in active], dtype=np.float64)

    # ------------------------------------------------------------------
    # Static track-list utilities
    # ------------------------------------------------------------------

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Merge two track lists without duplicating any track ID.

        ``tlista`` entries are preferred; entries from ``tlistb`` whose
        ``track_id`` already exists in ``tlista`` are skipped.

        Args:
            tlista: Primary track list.
            tlistb: Secondary track list.

        Returns:
            Merged list with unique ``track_id`` values.
        """
        exists = {t.track_id for t in tlista}
        result = list(tlista)
        for t in tlistb:
            if t.track_id not in exists:
                exists.add(t.track_id)
                result.append(t)
        return result

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Remove every track in *tlistb* from *tlista* (by track ID).

        Args:
            tlista: Source list.
            tlistb: Tracks to remove.

        Returns:
            Filtered copy of *tlista*.
        """
        remove_ids = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in remove_ids]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate tracks between two track lists.

        Two tracks are considered duplicates when their IoU exceeds 0.85
        (i.e. IoU distance < 0.15).  The track with the shorter lifetime
        (``frame_id − start_frame``) is removed.

        Args:
            stracksa: First track list (e.g. ``tracked_stracks``).
            stracksb: Second track list (e.g. ``lost_stracks``).

        Returns:
            Tuple ``(filtered_a, filtered_b)`` with duplicates removed.
        """
        if not stracksa or not stracksb:
            return stracksa, stracksb

        from mata.trackers.utils.matching import iou_distance

        pdist = iou_distance(stracksa, stracksb)  # (N, M) cost = 1 − IoU
        # Pairs with IoU > 0.85 ↔ pdist < 0.15
        pairs = np.argwhere(pdist < 0.15)

        dupa: set[int] = set()
        dupb: set[int] = set()
        for p, q in pairs:
            age_a = stracksa[p].frame_id - stracksa[p].start_frame
            age_b = stracksb[q].frame_id - stracksb[q].start_frame
            if age_a > age_b:
                dupb.add(q)
            else:
                dupa.add(p)

        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter.

        Delegates to :meth:`STrack.reset_id` (which in turn calls
        :meth:`~mata.trackers.basetrack.BaseTrack.reset_id`).  Must be
        called between independent video sequences so that IDs start
        from 1.
        """
        STrack.reset_id()

    def reset(self) -> None:
        """Reset all tracker state for a new video sequence.

        Clears all track lists, resets the frame counter and global track
        ID counter, and instantiates a fresh Kalman filter.
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BYTETracker("
            f"tracked={len(self.tracked_stracks)}, "
            f"lost={len(self.lost_stracks)}, "
            f"frame={self.frame_id})"
        )
