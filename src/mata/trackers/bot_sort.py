"""BotSort multi-object tracker.

Contains:
    BOTrack  — STrack extended with KalmanFilterXYWH and appearance features.
    BOTSORT  — BYTETracker extended with GMC and optional ReID.

Ported from Ultralytics tracker (MIT-compatible).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from mata.trackers.basetrack import TrackState
from mata.trackers.byte_tracker import BYTETracker, STrack

# ---------------------------------------------------------------------------
# BOTrack — STrack with KalmanFilterXYWH and appearance features
# ---------------------------------------------------------------------------


class BOTrack(STrack):
    """Single tracked object using the XYWH Kalman filter with ReID stubs.

    Extends :class:`~mata.trackers.byte_tracker.STrack` with two enhancements
    over the base ByteTrack track representation:

    1. **KalmanFilterXYWH** — The Kalman state is ``(cx, cy, w, h, vcx, vcy,
       vw, vh)``, using absolute width/height directly rather than the
       aspect-ratio parameterisation of :class:`~mata.trackers.byte_tracker.STrack`.

    2. **Appearance feature history** — Stores a rolling window of feature
       vectors with exponential moving-average (EMA) smoothing for future
       ReID integration.

    Class attributes:
        shared_kalman: Lazily initialised :class:`KalmanFilterXYWH` instance
            shared across all ``BOTrack`` objects (class-level, analogous to
            :attr:`STrack.shared_kalman`).

    Attributes:
        smooth_feat (np.ndarray | None): EMA-smoothed feature vector.
        curr_feat (np.ndarray | None): Most recent (L2-normalised) feature.
        features (deque): Rolling history of features, capped at
            ``feat_history`` entries.
        alpha (float): EMA blending coefficient (default 0.9).
    """

    # Lazily initialised KalmanFilterXYWH shared across all BOTrack instances.
    shared_kalman: Any = None  # set on first use via _get_shared_kalman()

    def __init__(
        self,
        xywh: list[float] | np.ndarray,
        score: float,
        cls: Any,
        feat: np.ndarray | None = None,
        feat_history: int = 50,
    ) -> None:
        """Initialise an *unactivated* BOTrack from a raw detection.

        Args:
            xywh: Detection bounding box ``[cx, cy, w, h]``, optionally
                extended to ``[cx, cy, w, h, angle]`` (5 elements) or
                ``[cx, cy, w, h, angle, idx]`` (6 elements).
            score: Detection confidence score ``[0, 1]``.
            cls: Class label (integer or string).
            feat: Optional appearance feature vector for ReID.
            feat_history: Maximum number of past feature vectors to retain.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat: np.ndarray | None = None
        self.curr_feat: np.ndarray | None = None
        self.features: deque = deque(maxlen=feat_history)
        self.alpha: float = 0.9

        if feat is not None:
            self.update_features(feat)

    # ------------------------------------------------------------------
    # Class-level shared Kalman filter
    # ------------------------------------------------------------------

    @classmethod
    def _get_shared_kalman(cls) -> Any:
        """Return (and lazily initialise) the shared :class:`KalmanFilterXYWH`."""
        if cls.shared_kalman is None:
            from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

            cls.shared_kalman = KalmanFilterXYWH()
        return cls.shared_kalman

    # ------------------------------------------------------------------
    # Appearance features
    # ------------------------------------------------------------------

    def update_features(self, feat: np.ndarray) -> None:
        """Update appearance features with EMA smoothing and L2 normalisation.

        Algorithm:
        1. L2-normalise the incoming feature.
        2. First call: store directly as ``smooth_feat``.
        3. Subsequent calls: ``smooth = alpha * smooth + (1-alpha) * feat``,
           then re-normalise ``smooth_feat``.
        4. Append to ``features`` history deque.

        Args:
            feat: Raw feature vector (any shape; flattened internally).
        """
        feat = np.asarray(feat, dtype=np.float64).ravel()
        norm = np.linalg.norm(feat)
        if norm > 1e-9:
            feat = feat / norm

        self.curr_feat = feat.copy()

        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
            # Re-normalise to keep smooth_feat on the unit hypersphere.
            s_norm = np.linalg.norm(self.smooth_feat)
            if s_norm > 1e-9:
                self.smooth_feat /= s_norm

        self.features.append(feat.copy())

    # ------------------------------------------------------------------
    # Single-track prediction
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Advance this track's Kalman state by one time step (in-place).

        Uses ``KalmanFilterXYWH`` via the per-track ``kalman_filter`` set
        during :meth:`activate`.  Zeroes the width and height velocity
        components when the track is not in *Tracked* state to suppress
        drift during lost periods.
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0.0  # zero vw
            mean_state[7] = 0.0  # zero vh
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        """Batch-predict all BOTrack objects using the shared KalmanFilterXYWH.

        Modifies each track's ``mean`` and ``covariance`` in-place.  Tracks
        that have not been activated (``mean is None``) are skipped.

        Args:
            stracks: BOTrack objects to predict forward by one time step.
        """
        if not stracks:
            return

        shared_kf = BOTrack._get_shared_kalman()

        for st in stracks:
            if st.mean is None:
                continue
            mean_state = st.mean.copy()
            if st.state != TrackState.Tracked:
                mean_state[6] = 0.0  # zero vw
                mean_state[7] = 0.0  # zero vh
            st.mean, st.covariance = shared_kf.predict(mean_state, st.covariance)

    # ------------------------------------------------------------------
    # Lifecycle — extend parent to propagate appearance features
    # ------------------------------------------------------------------

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track, updating appearance features if present.

        Args:
            new_track: New detection carrying updated appearance features.
            frame_id: Current frame number.
            new_id: If ``True``, assign a fresh track ID.
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        """Update the track with a new matched detection.

        Propagates appearance features from the detection before running
        the parent Kalman update.

        Args:
            new_track: Freshly detected track carrying the new observation.
            frame_id: Current frame number.
        """
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    # ------------------------------------------------------------------
    # Coordinate conversions — XYWH measurement space
    # ------------------------------------------------------------------

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a tlwh bbox to xywh (center format) for KalmanFilterXYWH.

        Overrides :meth:`STrack.convert_coords` which converts to xyah.

        Args:
            tlwh: Bounding box ``[x1, y1, w, h]``.

        Returns:
            np.ndarray: ``[cx, cy, w, h]``.
        """
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert ``[x1, y1, w, h]`` to ``[cx, cy, w, h]``.

        Args:
            tlwh: Array ``[top_left_x, top_left_y, width, height]``.

        Returns:
            np.ndarray: ``[center_x, center_y, width, height]``.
        """
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[0] += ret[2] / 2.0  # cx = x1 + w/2
        ret[1] += ret[3] / 2.0  # cy = y1 + h/2
        return ret

    # ------------------------------------------------------------------
    # Properties — XYWH Kalman mean → tlwh
    # ------------------------------------------------------------------

    @property
    def tlwh(self) -> np.ndarray:
        """Bounding box in ``[x1, y1, w, h]`` format.

        Before activation: returns the frozen ``_tlwh`` set at construction.
        After activation: recovers ``[x1, y1, w, h]`` from the XYWH Kalman
        mean ``[cx, cy, w, h, ...]``:
            - ``x1 = cx - w/2``,  ``y1 = cy - h/2``
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()  # [cx, cy, w, h]
        ret[:2] -= ret[2:] / 2.0  # top-left: x1 = cx - w/2, y1 = cy - h/2
        return ret

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BOTrack(id={self.track_id}, state={self.state}, "
            f"frame={self.frame_id}, score={self.score:.3f}, "
            f"cls={self.cls}, tlwh={self.tlwh.tolist()})"
        )


# ---------------------------------------------------------------------------
# BOTSORT — BYTETracker with GMC and optional ReID
# ---------------------------------------------------------------------------


class BOTSORT(BYTETracker):
    """BotSort multi-object tracker.

    Extends :class:`~mata.trackers.byte_tracker.BYTETracker` with three
    additional capabilities:

    1. **Global Motion Compensation (GMC)** — adjusts track states for
       camera movement before each predict–match cycle, using sparse
       Lucas-Kanade optical flow (or a fallback identity if OpenCV is
       unavailable).

    2. **KalmanFilterXYWH** — Uses absolute width/height state instead of
       the aspect-ratio parameterisation of the base ByteTrack.

    3. **ReID appearance distance stub** — Fields ``with_reid``,
       ``proximity_thresh``, and ``appearance_thresh`` are preserved for
       future integration.  The encoder is ``None`` in v1.8.

    Args:
        args: Tracker configuration.  May be a :class:`_TrackerArgs`
            instance, any attribute-carrying object, or a ``dict``.
            BotSort-specific keys consumed here:

            * ``gmc_method`` (str | None) — default ``'sparseOptFlow'``
            * ``proximity_thresh`` (float) — default ``0.5``
            * ``appearance_thresh`` (float) — default ``0.25``
            * ``with_reid`` (bool) — default ``False``

        frame_rate: Video frame rate for ``max_time_lost`` calculation.

    Attributes:
        gmc (GMC): Global Motion Compensation instance.
        proximity_thresh (float): IoU threshold above which detections are
            considered too far from the track to be associated by ReID.
        appearance_thresh (float): Minimum cosine similarity for ReID match.
        with_reid (bool): Whether ReID matching is active (False in v1.8).
        encoder: ReID feature encoder — ``None``, reserved for v1.9+.
    """

    def __init__(self, args: Any, frame_rate: int = 30) -> None:
        # Read BotSort-specific fields directly from the original args dict (or
        # object) *before* super().__init__() passes them through _TrackerArgs,
        # which only stores ByteTrack fields and silently drops the rest.
        if isinstance(args, dict):
            _gmc_method = args.get("gmc_method", "sparseOptFlow") or "sparseOptFlow"
            _proximity = float(args.get("proximity_thresh", 0.5))
            _appearance = float(args.get("appearance_thresh", 0.25))
            _with_reid = bool(args.get("with_reid", False))
        else:
            _gmc_method = getattr(args, "gmc_method", "sparseOptFlow") or "sparseOptFlow"
            _proximity = float(getattr(args, "proximity_thresh", 0.5))
            _appearance = float(getattr(args, "appearance_thresh", 0.25))
            _with_reid = bool(getattr(args, "with_reid", False))

        super().__init__(args, frame_rate)

        from mata.trackers.utils.gmc import GMC

        self.gmc: GMC = GMC(method=_gmc_method)
        self.proximity_thresh: float = _proximity
        self.appearance_thresh: float = _appearance
        self.with_reid: bool = _with_reid
        self.encoder: Any = None  # ReID encoder — disabled in v1.8

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def get_kalmanfilter(self) -> Any:
        """Return a new :class:`KalmanFilterXYWH` instance.

        Overrides :meth:`BYTETracker.get_kalmanfilter` which returns
        :class:`KalmanFilterXYAH`.
        """
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        return KalmanFilterXYWH()

    def init_track(self, results: Any, img: np.ndarray | None = None) -> list[BOTrack]:
        """Convert detection results to :class:`BOTrack` instances.

        Overrides :meth:`BYTETracker.init_track` to produce ``BOTrack``
        objects (which use :class:`KalmanFilterXYWH`) instead of
        :class:`~mata.trackers.byte_tracker.STrack`.

        Args:
            results: Detection results (``DetectionResults`` or compatible).
            img: Optional frame image (not used directly here).

        Returns:
            List of uninitialised :class:`BOTrack` objects, one per detection.
        """
        if len(results) == 0:
            return []

        tracks: list[BOTrack] = []
        for i in range(len(results)):
            xywh = results.xywh[i]  # [cx, cy, w, h]
            score = float(results.conf[i])
            cls = results.cls[i]
            orig_idx = float(results._indices[i] if hasattr(results, "_indices") else i)
            xywh_ext = np.array(
                [xywh[0], xywh[1], xywh[2], xywh[3], 0.0, orig_idx],
                dtype=np.float64,
            )
            tracks.append(BOTrack(xywh_ext, score, cls))
        return tracks

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Compute association cost matrix with proximity threshold gating.

        Extends :meth:`BYTETracker.get_dists` by:

        1. Computing IoU distance (optionally fused with confidence scores).
        2. Setting entries above ``proximity_thresh`` to ``1.0``
           (infeasible), preventing mis-associations across large spatial
           gaps.
        3. When ``with_reid=True`` and ``encoder`` is available: blending
           appearance distance for pairs inside the proximity gate.  This
           path is stubbed in v1.8 — only the IoU + gate is active.

        Args:
            tracks: Active/lost :class:`BOTrack` objects (rows).
            detections: Candidate detections from the current frame (cols).

        Returns:
            ``(len(tracks), len(detections))`` float32 cost matrix.
        """
        from mata.trackers.utils.matching import (
            embedding_distance,
            fuse_score,
            iou_distance,
        )

        dists = iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = fuse_score(dists, detections)

        # Gate: pairs with IoU distance above proximity_thresh are infeasible.
        if self.with_reid and self.encoder is not None:
            # ReID path (future v1.9): use appearance similarity inside the gate.
            emb_dists = embedding_distance(tracks, detections)
            dists = np.where(dists > self.proximity_thresh, 1.0, emb_dists)
        else:
            # Stub: apply proximity gate on IoU distances.
            dists[dists > self.proximity_thresh] = 1.0

        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Batch-predict :class:`BOTrack` objects with :class:`KalmanFilterXYWH`.

        Delegates to :meth:`BOTrack.multi_predict`, overriding
        :meth:`BYTETracker.multi_predict` which uses
        :meth:`STrack.multi_predict` (XYAH).

        Args:
            tracks: Tracks to advance in-place.
        """
        BOTrack.multi_predict(tracks)

    # ------------------------------------------------------------------
    # Core update loop — adds GMC step
    # ------------------------------------------------------------------

    def update(self, results: Any, img: np.ndarray | None = None) -> np.ndarray:
        """Update tracker with detections from the current frame.

        Applies Global Motion Compensation to all existing track states
        before delegating to :meth:`BYTETracker.update`, which handles the
        two-stage association algorithm.

        GMC is applied to **both** ``tracked_stracks`` and ``lost_stracks``
        before prediction so that Kalman predictions start from
        camera-corrected positions.

        Args:
            results: Detection results for the current frame.
            img: BGR frame as ``np.ndarray`` for GMC.  If ``None``, GMC is
                skipped and the tracker behaves identically to
                :class:`BYTETracker`.

        Returns:
            ``np.ndarray`` of shape ``(N, 8)`` with columns
            ``[x1, y1, x2, y2, track_id, score, cls, idx]``.
        """
        if img is not None:
            # Compute camera-motion warp from previous → current frame.
            warp = self.gmc.apply(img)
            # Adjust all stored track states to compensate for camera motion.
            # multi_gmc modifies mean[:2] (cx, cy) and mean[4:6] (vcx, vcy).
            STrack.multi_gmc(self.tracked_stracks, warp)
            STrack.multi_gmc(self.lost_stracks, warp)

        return super().update(results, img)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracker state and GMC for a new video sequence.

        Calls :meth:`BYTETracker.reset` then clears GMC inter-frame state
        so that optical flow estimation does not bleed across sequences.
        """
        super().reset()
        self.gmc.reset_params()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BOTSORT("
            f"tracked={len(self.tracked_stracks)}, "
            f"lost={len(self.lost_stracks)}, "
            f"frame={self.frame_id}, "
            f"gmc={self.gmc.method!r})"
        )
