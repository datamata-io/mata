"""Global Motion Compensation (GMC) for multi-object tracking.

Estimates camera motion between consecutive frames using sparse optical flow
and returns a 2×3 affine transformation matrix.

Ported from Ultralytics tracker utils (MIT-compatible).
"""

from __future__ import annotations

import numpy as np


class GMC:
    """Global Motion Compensation using sparse optical flow.

    Estimates camera motion between consecutive frames and returns a 2×3
    affine transformation matrix that can be applied to track states via
    :meth:`~mata.trackers.byte_tracker.STrack.multi_gmc`.

    Supported methods:
    - ``'sparseOptFlow'``: Lucas-Kanade sparse optical flow (default).
    - ``None`` / ``''``: No compensation — always returns identity matrix.

    The ``'orb'`` and ``'ecc'`` methods are reserved for future use and
    currently fall back to the identity matrix.

    Args:
        method: GMC method name (case-insensitive). Default ``'sparseOptFlow'``.

    Attributes:
        method (str): Normalised method name.

    Example::

        gmc = GMC(method="sparseOptFlow")
        H = gmc.apply(frame_bgr)   # np.ndarray (2, 3)
    """

    def __init__(self, method: str = "sparseOptFlow") -> None:
        self.method: str = (method or "").strip().lower()

        # Internal state for inter-frame matching.
        self._prev_frame_gray: np.ndarray | None = None
        self._prev_keypoints: np.ndarray | None = None  # (N, 1, 2) float32

        # Lucas-Kanade parameters.
        self._lk_win_size: tuple[int, int] = (15, 15)
        self._lk_max_level: int = 3
        self._lk_criteria: tuple[int, int, float] = (3, 20, 0.001)

        # Shi-Tomasi feature detection parameters.
        self._max_corners: int = 200
        self._quality_level: float = 0.01
        self._min_distance: float = 1.0
        self._block_size: int = 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        raw_frame: np.ndarray,
        detections: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the camera-motion warp matrix from the previous frame.

        Args:
            raw_frame: Current BGR (or grayscale) frame as ``np.ndarray``
                with shape ``(H, W, 3)`` or ``(H, W)``.
            detections: Optional ``(N, 4)`` bounding-box array
                (currently unused; reserved for masking keypoints inside
                detection regions in a future implementation).

        Returns:
            ``np.ndarray`` of shape ``(2, 3)`` representing the 2-D affine
            transformation from the previous frame's coordinate system to
            the current frame's.  Returns ``np.eye(2, 3)`` (identity) when:

            - this is the first frame (no previous frame available),
            - OpenCV is not installed,
            - fewer than 4 matching keypoints were found, or
            - the method is ``None`` / unrecognised.
        """
        if self.method == "sparseoptflow":
            return self._apply_sparse_optical_flow(raw_frame)
        # 'orb', 'ecc' and unknown methods → identity (future expansion).
        return np.eye(2, 3, dtype=np.float64)

    def reset_params(self) -> None:
        """Clear all inter-frame state.

        Must be called between independent video sequences so that motion
        estimation does not bleed across sequence boundaries.
        """
        self._prev_frame_gray = None
        self._prev_keypoints = None

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _apply_sparse_optical_flow(self, raw_frame: np.ndarray) -> np.ndarray:
        """Lucas-Kanade sparse optical flow GMC implementation.

        Algorithm:
        1. Convert frame to grayscale.
        2. On the first frame: detect Shi-Tomasi corners and cache them;
           return identity.
        3. On subsequent frames: track cached keypoints with LK optical
           flow, estimate a partial-affine transform via RANSAC, and
           re-detect keypoints for the next call.

        Args:
            raw_frame: Current frame (BGR or grayscale).

        Returns:
            ``(2, 3)`` affine matrix, or ``np.eye(2, 3)`` on failure.
        """
        identity = np.eye(2, 3, dtype=np.float64)

        try:
            import cv2
        except ImportError:
            # OpenCV not installed — silently return identity.
            return identity

        # ------------------------------------------------------------------ #
        # Step 1: Convert to grayscale                                         #
        # ------------------------------------------------------------------ #
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        elif raw_frame.ndim == 3 and raw_frame.shape[2] == 4:
            frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2GRAY)
        else:
            frame_gray = np.asarray(raw_frame, dtype=np.uint8)

        # ------------------------------------------------------------------ #
        # Step 2: Initialise on first call                                     #
        # ------------------------------------------------------------------ #
        if self._prev_frame_gray is None:
            self._prev_keypoints = cv2.goodFeaturesToTrack(
                frame_gray,
                maxCorners=self._max_corners,
                qualityLevel=self._quality_level,
                minDistance=self._min_distance,
                blockSize=self._block_size,
                mask=None,
            )
            self._prev_frame_gray = frame_gray.copy()
            return identity

        # ------------------------------------------------------------------ #
        # Step 3: Need previous keypoints to track                             #
        # ------------------------------------------------------------------ #
        if self._prev_keypoints is None or len(self._prev_keypoints) == 0:
            # No keypoints from previous frame — re-detect and return identity.
            self._prev_frame_gray = frame_gray.copy()
            self._prev_keypoints = cv2.goodFeaturesToTrack(
                frame_gray,
                maxCorners=self._max_corners,
                qualityLevel=self._quality_level,
                minDistance=self._min_distance,
                blockSize=self._block_size,
            )
            return identity

        # ------------------------------------------------------------------ #
        # Step 4: Sparse optical flow tracking                                 #
        # ------------------------------------------------------------------ #
        try:
            curr_keypoints, status, _err = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                self._prev_frame_gray,
                frame_gray,
                self._prev_keypoints,
                None,
                winSize=self._lk_win_size,
                maxLevel=self._lk_max_level,
                criteria=self._lk_criteria,
            )
        except cv2.error:
            # Optical flow failed (e.g. frame size mismatch) — reset and return.
            self._reset_and_redetect(cv2, frame_gray)
            return identity

        if status is None or curr_keypoints is None:
            self._reset_and_redetect(cv2, frame_gray)
            return identity

        # ------------------------------------------------------------------ #
        # Step 5: Filter to well-tracked keypoints                             #
        # ------------------------------------------------------------------ #
        mask = status.ravel().astype(bool)
        good_prev = self._prev_keypoints[mask]  # (K, 1, 2)
        good_curr = curr_keypoints[mask]  # (K, 1, 2)

        if len(good_prev) < 4:
            # Too few matches — cannot reliably estimate affine transform.
            self._reset_and_redetect(cv2, frame_gray)
            return identity

        # ------------------------------------------------------------------ #
        # Step 6: Estimate partial affine (rotation + scale + translation)     #
        # ------------------------------------------------------------------ #
        try:
            H, _inlier_mask = cv2.estimateAffinePartial2D(  # noqa: N806
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
        except cv2.error:
            H = None  # noqa: N806

        if H is None:
            H = identity  # noqa: N806

        # ------------------------------------------------------------------ #
        # Step 7: Update state for next call                                   #
        # ------------------------------------------------------------------ #
        self._prev_frame_gray = frame_gray.copy()
        self._prev_keypoints = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            blockSize=self._block_size,
        )

        return H.astype(np.float64)

    def _reset_and_redetect(self, cv2: object, frame_gray: np.ndarray) -> None:
        """Reset inter-frame state and re-detect keypoints.

        Called when optical flow tracking fails so the next call can
        attempt a fresh track from detected corners.

        Args:
            cv2: The imported cv2 module.
            frame_gray: Current grayscale frame.
        """
        self._prev_frame_gray = frame_gray.copy()
        try:
            self._prev_keypoints = cv2.goodFeaturesToTrack(  # type: ignore[attr-defined]
                frame_gray,
                maxCorners=self._max_corners,
                qualityLevel=self._quality_level,
                minDistance=self._min_distance,
                blockSize=self._block_size,
            )
        except Exception:  # noqa: BLE001
            self._prev_keypoints = None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"GMC(method={self.method!r})"
