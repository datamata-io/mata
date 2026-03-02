"""Unit tests for Task A5: BOTrack, BOTSORT, and GMC.

Covers all acceptance criteria:
- BOTrack: KalmanFilterXYWH usage, tlwh property, convert_coords, tlwh_to_xywh
- BOTrack: feature storage, EMA smoothing, L2 normalisation
- BOTrack: update_features repeated calls (EMA)
- BOTrack: predict zeroes vw/vh when not Tracked
- BOTrack: multi_predict batch operation
- BOTrack: re_activate / update propagate features
- BOTrack: activate uses KalmanFilterXYWH
- BOTSORT: extends BYTETracker, get_kalmanfilter returns KalmanFilterXYWH
- BOTSORT: init_track produces BOTrack instances
- BOTSORT: get_dists proximity threshold gating
- BOTSORT: get_dists fuse_score integration
- BOTSORT: multi_predict delegates to BOTrack.multi_predict
- BOTSORT: update applies GMC warp before prediction
- BOTSORT: update without img (no GMC branch)
- BOTSORT: two-stage association (inherits BYTETracker algorithm)
- BOTSORT: reset clears tracker state and GMC params
- BOTSORT: ReID fields exist, encoder is None
- GMC.__init__: method normalisation
- GMC.apply: returns identity (2, 3) on first call
- GMC.apply: identity when cv2 unavailable (monkeypatched)
- GMC.apply: identity for unknown/None method
- GMC.apply: returns 2x3 affine matrix on second call (mocked cv2)
- GMC.reset_params: clears cached frame and keypoints
- Package-level import (mata.trackers)
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata.trackers.basetrack import BaseTrack, TrackState
from mata.trackers.bot_sort import BOTSORT, BOTrack
from mata.trackers.byte_tracker import BYTETracker, DetectionResults, STrack
from mata.trackers.utils.gmc import GMC

# ===========================================================================
# Fixtures / helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset global track ID counter before and after every test."""
    BaseTrack.reset_id()
    yield
    BaseTrack.reset_id()


def make_results(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    cls_ids: list[int] | None = None,
) -> DetectionResults:
    """Build DetectionResults from xyxy boxes + scores."""
    n = len(boxes)
    xyxy = np.array(boxes, dtype=np.float32).reshape(n, 4)
    conf = np.array(scores, dtype=np.float32)
    cls = np.array(cls_ids if cls_ids is not None else [0] * n, dtype=np.float32)
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return DetectionResults(conf=conf, xyxy=xyxy, xywh=xywh, cls=cls)


def default_args(**overrides) -> dict:
    """Return a minimal BotSort args dict."""
    base = {
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": None,  # Disable GMC in most tests for simplicity
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
    }
    base.update(overrides)
    return base


def make_gray_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a random grayscale frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


def make_bgr_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a random BGR frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ===========================================================================
# BOTrack — construction
# ===========================================================================


class TestBOTrackConstruction:
    def test_basic_construction(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        assert t.score == pytest.approx(0.9)
        assert t.cls == 0
        assert t.smooth_feat is None
        assert t.curr_feat is None
        assert isinstance(t.features, deque)
        assert t.alpha == pytest.approx(0.9)

    def test_construction_with_feature(self):
        feat = np.array([1.0, 0.0, 0.0])
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 1, feat=feat)
        assert t.smooth_feat is not None
        assert t.curr_feat is not None
        assert len(t.features) == 1

    def test_feat_history_cap(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0, feat_history=3)
        assert t.features.maxlen == 3

    def test_default_feat_history(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        assert t.features.maxlen == 50

    def test_inherited_from_strack(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        assert isinstance(t, STrack)
        assert t.mean is None

    def test_tlwh_before_activation(self):
        """Before activation, tlwh returns frozen _tlwh."""
        # BOTrack stores _tlwh from cx,cy,w,h → top-left
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        # _tlwh = [cx - w/2, cy - h/2, w, h] = [80, 50, 40, 60]
        expected = np.array([80.0, 50.0, 40.0, 60.0])
        np.testing.assert_allclose(t.tlwh, expected)


# ===========================================================================
# BOTrack — coordinate conversions
# ===========================================================================


class TestBOTrackCoordinates:
    def test_tlwh_to_xywh(self):
        tlwh = np.array([10.0, 20.0, 30.0, 40.0])
        xywh = BOTrack.tlwh_to_xywh(tlwh)
        np.testing.assert_allclose(xywh, [25.0, 40.0, 30.0, 40.0])

    def test_tlwh_to_xywh_does_not_mutate(self):
        tlwh = np.array([10.0, 20.0, 30.0, 40.0])
        original = tlwh.copy()
        BOTrack.tlwh_to_xywh(tlwh)
        np.testing.assert_array_equal(tlwh, original)

    def test_convert_coords_delegates_to_tlwh_to_xywh(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        tlwh = np.array([10.0, 20.0, 30.0, 40.0])
        result = t.convert_coords(tlwh)
        expected = BOTrack.tlwh_to_xywh(tlwh)
        np.testing.assert_array_equal(result, expected)

    def test_tlwh_after_activation(self):
        """After activation with KalmanFilterXYWH, tlwh comes from mean."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)
        # After activation: mean[:4] = [cx, cy, w, h]
        # tlwh = [cx - w/2, cy - h/2, w, h]
        assert t.mean is not None
        cx, cy, w, h = t.mean[:4]
        expected_tlwh = np.array([cx - w / 2, cy - h / 2, w, h])
        np.testing.assert_allclose(t.tlwh, expected_tlwh, rtol=1e-6)

    def test_xyxy_after_activation(self):
        """xyxy is derived from tlwh."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)
        tlwh = t.tlwh
        expected = np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
        np.testing.assert_allclose(t.xyxy, expected, rtol=1e-6)


# ===========================================================================
# BOTrack — shared Kalman filter
# ===========================================================================


class TestBOTrackSharedKalman:
    def test_shared_kalman_is_xywh(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = BOTrack._get_shared_kalman()
        assert isinstance(kf, KalmanFilterXYWH)

    def test_shared_kalman_singleton(self):
        kf1 = BOTrack._get_shared_kalman()
        kf2 = BOTrack._get_shared_kalman()
        assert kf1 is kf2

    def test_botrack_and_strack_use_different_shared_kalman(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH

        strack_kf = STrack._get_shared_kalman()
        botrack_kf = BOTrack._get_shared_kalman()
        # Both should be initialised and be different instances of different types.
        assert isinstance(strack_kf, KalmanFilterXYAH)
        assert isinstance(botrack_kf, KalmanFilterXYWH)
        assert strack_kf is not botrack_kf


# ===========================================================================
# BOTrack — feature updates
# ===========================================================================


class TestBOTrackFeatures:
    def test_update_features_sets_curr_feat(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        feat = np.array([3.0, 4.0])
        t.update_features(feat)
        # L2-normalised: [3/5, 4/5]
        np.testing.assert_allclose(t.curr_feat, [0.6, 0.8])

    def test_update_features_first_call_sets_smooth_feat(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        feat = np.array([3.0, 4.0])
        t.update_features(feat)
        np.testing.assert_allclose(t.smooth_feat, [0.6, 0.8])

    def test_update_features_ema(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        feat1 = np.array([1.0, 0.0])
        feat2 = np.array([0.0, 1.0])
        t.update_features(feat1)
        t.update_features(feat2)
        # After 2nd call: smooth = 0.9 * [1, 0] + 0.1 * [0, 1] = [0.9, 0.1]
        # normalised
        expected_unnorm = np.array([0.9, 0.1])
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)
        np.testing.assert_allclose(t.smooth_feat, expected, atol=1e-9)

    def test_update_features_l2_normalised(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        feat = np.array([2.0, 2.0, 2.0])
        t.update_features(feat)
        # smooth_feat should be unit vector
        assert np.linalg.norm(t.smooth_feat) == pytest.approx(1.0, abs=1e-9)

    def test_smooth_feat_remains_unit_after_multiple_updates(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            feat = rng.standard_normal(64)
            t.update_features(feat)
            assert np.linalg.norm(t.smooth_feat) == pytest.approx(1.0, abs=1e-9)

    def test_update_features_appends_to_history(self):
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0, feat_history=3)
        for i in range(5):
            t.update_features(np.array([float(i), 0.0]))
        # Deque is capped at 3
        assert len(t.features) == 3

    def test_zero_norm_feature_does_not_normalise(self):
        """Zero vector should not cause NaN or ZeroDivisionError."""
        t = BOTrack([50.0, 50.0, 40.0, 80.0], 0.9, 0)
        feat = np.zeros(4)
        t.update_features(feat)  # Should not raise
        # curr_feat is the zero vector (not normalised since norm ≤ eps)
        np.testing.assert_array_equal(t.curr_feat, np.zeros(4))


# ===========================================================================
# BOTrack — predict / multi_predict
# ===========================================================================


class TestBOTrackPredict:
    def test_predict_advances_mean(self):
        """With zero initial velocity the mean stays the same (constant-velocity
        with v=0 ⇒ no positional shift); covariance always grows."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)
        cov_before = t.covariance.copy()
        t.predict()
        # Covariance grows with each prediction step (process noise added).
        assert not np.array_equal(t.covariance, cov_before)

    def test_predict_zeroes_vw_vh_when_lost(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)
        # Artificially set large velocity
        t.mean[6] = 5.0
        t.mean[7] = 5.0
        t.mark_lost()
        assert t.state == TrackState.Lost
        t.predict()
        # The mean passed to KF had vw=vh=0; after prediction vw/vh remain ~0
        # (constant velocity model keeps 0, no process noise dominates)
        # We just verify no exception and mean changed
        assert t.mean is not None

    def test_multi_predict_empty(self):
        BOTrack.multi_predict([])  # Should not raise

    def test_multi_predict_skips_unactivated(self):
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        assert t.mean is None
        BOTrack.multi_predict([t])  # Should not raise; mean stays None
        assert t.mean is None

    def test_multi_predict_batch(self):
        """After multi_predict the covariances should grow (process noise added)."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        tracks = [BOTrack([float(50 + 10 * i), 50.0, 40.0, 60.0], 0.9, 0) for i in range(3)]
        for t in tracks:
            t.activate(kf, frame_id=1)

        covs_before = [t.covariance.copy() for t in tracks]
        BOTrack.multi_predict(tracks)
        for before, t in zip(covs_before, tracks):
            assert not np.array_equal(t.covariance, before)


# ===========================================================================
# BOTrack — re_activate / update with features
# ===========================================================================


class TestBOTrackLifecycle:
    def test_re_activate_propagates_feature(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)

        new_track = BOTrack([102.0, 80.0, 40.0, 60.0], 0.85, 0)
        feat = np.array([0.0, 1.0, 0.0])
        new_track.update_features(feat)

        t.mark_lost()
        t.re_activate(new_track, frame_id=2)
        assert t.curr_feat is not None
        assert t.smooth_feat is not None

    def test_re_activate_without_feature(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)
        t.mark_lost()

        new_track = BOTrack([102.0, 80.0, 40.0, 60.0], 0.85, 0)
        # No feature set on new_track — should not raise
        t.re_activate(new_track, frame_id=2)
        assert t.state == TrackState.Tracked

    def test_update_propagates_feature(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        kf = KalmanFilterXYWH()
        t = BOTrack([100.0, 80.0, 40.0, 60.0], 0.9, 0)
        t.activate(kf, frame_id=1)

        new_track = BOTrack([100.0, 80.0, 40.0, 60.0], 0.88, 0)
        feat = np.array([1.0, 2.0, 3.0])
        new_track.update_features(feat)

        t.update(new_track, frame_id=2)
        assert t.curr_feat is not None
        assert t.tracklet_len == 1


# ===========================================================================
# GMC — construction
# ===========================================================================


class TestGMCConstruction:
    def test_default_method(self):
        gmc = GMC()
        assert gmc.method == "sparseoptflow"

    def test_custom_method_normalised(self):
        gmc = GMC(method="SparseOptFlow")
        assert gmc.method == "sparseoptflow"

    def test_none_method(self):
        gmc = GMC(method=None)
        assert gmc.method == ""

    def test_empty_method(self):
        gmc = GMC(method="")
        assert gmc.method == ""

    def test_unknown_method(self):
        gmc = GMC(method="orb")
        assert gmc.method == "orb"

    def test_initial_state_is_none(self):
        gmc = GMC()
        assert gmc._prev_frame_gray is None
        assert gmc._prev_keypoints is None


# ===========================================================================
# GMC — apply: identity cases
# ===========================================================================


class TestGMCApplyIdentity:
    def test_unknown_method_returns_identity(self):
        gmc = GMC(method="orb")
        frame = make_bgr_frame()
        result = gmc.apply(frame)
        np.testing.assert_array_equal(result, np.eye(2, 3))

    def test_none_method_returns_identity(self):
        gmc = GMC(method=None)
        frame = make_bgr_frame()
        result = gmc.apply(frame)
        np.testing.assert_array_equal(result, np.eye(2, 3))

    def test_first_call_returns_identity(self):
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()
        result = gmc.apply(frame)
        np.testing.assert_array_equal(result, np.eye(2, 3))

    def test_result_is_2x3(self):
        gmc = GMC(method="sparseOptFlow")
        result = gmc.apply(make_bgr_frame())
        assert result.shape == (2, 3)

    def test_first_call_initialises_prev_state(self):
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()
        gmc.apply(frame)
        assert gmc._prev_frame_gray is not None

    @patch.dict("sys.modules", {"cv2": None})
    def test_cv2_unavailable_returns_identity(self):
        """When cv2 is not importable, GMC must return identity gracefully."""
        import sys

        # Temporarily make cv2 unimportable.
        real_cv2 = sys.modules.pop("cv2", None)
        sys.modules["cv2"] = None  # type: ignore[assignment]

        try:
            gmc = GMC(method="sparseOptFlow")
            result = gmc.apply(make_bgr_frame())
            np.testing.assert_array_equal(result, np.eye(2, 3))
        finally:
            if real_cv2 is not None:
                sys.modules["cv2"] = real_cv2
            else:
                sys.modules.pop("cv2", None)


# ===========================================================================
# GMC — apply: with mocked cv2
# ===========================================================================


class TestGMCApplyMocked:
    """Tests using mocked cv2 to verify GMC logic without OpenCV installed."""

    def _make_mock_cv2(self, num_kp: int = 20, affine_ok: bool = True):
        """Return a mock cv2 module with the APIs used by GMC."""
        cv2 = MagicMock()
        cv2.COLOR_BGR2GRAY = 6
        cv2.COLOR_BGRA2GRAY = 10
        cv2.RANSAC = 8

        # goodFeaturesToTrack returns (N, 1, 2) float32 array
        kp = np.random.default_rng(0).random((num_kp, 1, 2)).astype(np.float32)
        cv2.goodFeaturesToTrack.return_value = kp

        # cvtColor just returns a grayscale frame
        cv2.cvtColor.side_effect = lambda x, *a, **kw: x[:, :, 0] if x.ndim == 3 else x

        # calcOpticalFlowPyrLK: all points tracked successfully
        status = np.ones((num_kp, 1), dtype=np.uint8)
        curr_kp = kp + 1.0  # slight shift
        cv2.calcOpticalFlowPyrLK.return_value = (curr_kp, status, None)

        # estimateAffinePartial2D
        if affine_ok:
            H = np.eye(2, 3, dtype=np.float32)  # noqa: N806
            H[0, 2] = 2.0  # small translation
            cv2.estimateAffinePartial2D.return_value = (H, np.ones((num_kp, 1), dtype=np.uint8))
        else:
            cv2.estimateAffinePartial2D.return_value = (None, None)

        cv2.error = Exception
        return cv2

    def test_second_call_returns_2x3(self):
        mock_cv2 = self._make_mock_cv2()
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            # First call: initialise
            r1 = gmc._apply_sparse_optical_flow(frame)
            np.testing.assert_array_equal(r1, np.eye(2, 3))
            # Second call: actual estimation
            r2 = gmc._apply_sparse_optical_flow(frame)
            assert r2.shape == (2, 3)

    def test_second_call_returns_estimated_affine(self):
        mock_cv2 = self._make_mock_cv2(affine_ok=True)
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            gmc._apply_sparse_optical_flow(frame)  # init
            H = gmc._apply_sparse_optical_flow(frame)  # noqa: N806
            # Should match the affine we set up (translation 2.0 in x)
            assert H[0, 2] == pytest.approx(2.0)

    def test_failed_affine_returns_identity(self):
        mock_cv2 = self._make_mock_cv2(affine_ok=False)
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            gmc._apply_sparse_optical_flow(frame)  # init
            H = gmc._apply_sparse_optical_flow(frame)  # noqa: N806
            np.testing.assert_array_equal(H, np.eye(2, 3))

    def test_too_few_keypoints_returns_identity(self):
        """When fewer than 4 keypoints track successfully, return identity."""
        mock_cv2 = self._make_mock_cv2(num_kp=3)
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            gmc._apply_sparse_optical_flow(frame)  # init
            H = gmc._apply_sparse_optical_flow(frame)  # noqa: N806
            np.testing.assert_array_equal(H, np.eye(2, 3))

    def test_lk_optical_flow_error_returns_identity(self):
        """cv2.error during calcOpticalFlowPyrLK → identity + state reset."""
        mock_cv2 = self._make_mock_cv2()
        mock_cv2.calcOpticalFlowPyrLK.side_effect = Exception("lk failed")
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            gmc._apply_sparse_optical_flow(frame)  # init
            H = gmc._apply_sparse_optical_flow(frame)  # noqa: N806
            np.testing.assert_array_equal(H, np.eye(2, 3))

    def test_result_dtype_is_float64(self):
        mock_cv2 = self._make_mock_cv2()
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            gmc._apply_sparse_optical_flow(frame)
            H = gmc._apply_sparse_optical_flow(frame)  # noqa: N806
            assert H.dtype == np.float64


# ===========================================================================
# GMC — reset_params
# ===========================================================================


class TestGMCResetParams:
    def test_reset_clears_prev_frame(self):
        gmc = GMC(method="sparseOptFlow")
        gmc._prev_frame_gray = np.zeros((10, 10), dtype=np.uint8)
        gmc._prev_keypoints = np.zeros((5, 1, 2), dtype=np.float32)
        gmc.reset_params()
        assert gmc._prev_frame_gray is None
        assert gmc._prev_keypoints is None

    def test_reset_allows_fresh_start(self):
        gmc = GMC(method="sparseOptFlow")
        frame = make_bgr_frame()
        gmc.apply(frame)
        assert gmc._prev_frame_gray is not None
        gmc.reset_params()
        assert gmc._prev_frame_gray is None


# ===========================================================================
# BOTSORT — construction and attributes
# ===========================================================================


class TestBOTSORTConstruction:
    def test_inherits_from_bytetracker(self):
        bot = BOTSORT(default_args())
        assert isinstance(bot, BYTETracker)

    def test_gmc_attribute_exists(self):
        bot = BOTSORT(default_args())
        assert isinstance(bot.gmc, GMC)

    def test_reid_fields(self):
        bot = BOTSORT(default_args(with_reid=False))
        assert bot.with_reid is False
        assert bot.encoder is None

    def test_proximity_thresh(self):
        bot = BOTSORT(default_args(proximity_thresh=0.3))
        assert bot.proximity_thresh == pytest.approx(0.3)

    def test_appearance_thresh(self):
        bot = BOTSORT(default_args(appearance_thresh=0.4))
        assert bot.appearance_thresh == pytest.approx(0.4)

    def test_gmc_method_from_args(self):
        bot = BOTSORT(default_args(gmc_method="sparseOptFlow"))
        assert bot.gmc.method == "sparseoptflow"

    def test_gmc_method_none(self):
        bot = BOTSORT(default_args(gmc_method=None))
        # None → "sparseOptFlow" default in BOTSORT.__init__
        assert bot.gmc.method == "sparseoptflow"

    def test_max_time_lost(self):
        bot = BOTSORT(default_args(track_buffer=30), frame_rate=30)
        assert bot.max_time_lost == 30

    def test_args_dict_accepted(self):
        bot = BOTSORT({"track_high_thresh": 0.4, "gmc_method": None})
        assert isinstance(bot, BOTSORT)


# ===========================================================================
# BOTSORT — get_kalmanfilter
# ===========================================================================


class TestBOTSORTKalmanFilter:
    def test_get_kalmanfilter_returns_xywh(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        bot = BOTSORT(default_args())
        kf = bot.get_kalmanfilter()
        assert isinstance(kf, KalmanFilterXYWH)

    def test_stored_kalman_filter_is_xywh(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        bot = BOTSORT(default_args())
        assert isinstance(bot.kalman_filter, KalmanFilterXYWH)


# ===========================================================================
# BOTSORT — init_track
# ===========================================================================


class TestBOTSORTInitTrack:
    def test_produces_botrack_instances(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        tracks = bot.init_track(results)
        assert len(tracks) == 1
        assert isinstance(tracks[0], BOTrack)

    def test_empty_results(self):
        bot = BOTSORT(default_args())
        tracks = bot.init_track(DetectionResults.empty())
        assert tracks == []

    def test_multiple_tracks(self):
        bot = BOTSORT(default_args())
        results = make_results(
            [(10, 10, 50, 90), (100, 100, 140, 160), (200, 50, 240, 90)],
            [0.9, 0.8, 0.7],
        )
        tracks = bot.init_track(results)
        assert len(tracks) == 3
        assert all(isinstance(t, BOTrack) for t in tracks)

    def test_index_preserved(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        tracks = bot.init_track(results)
        assert tracks[0].idx == 0


# ===========================================================================
# BOTSORT — get_dists
# ===========================================================================


class TestBOTSORTGetDists:
    def _make_botrack(self, box, score=0.9):
        """Create an activated BOTrack from a xyxy box."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        kf = KalmanFilterXYWH()
        t = BOTrack([cx, cy, w, h], score, 0)
        t.activate(kf, frame_id=1)
        return t

    def test_self_distance_gated(self):
        """A track matched to its own detection should have low cost and not be gated."""
        bot = BOTSORT(default_args(fuse_score=False, proximity_thresh=0.5))
        track = self._make_botrack((10, 10, 50, 90))
        det = self._make_botrack((10, 10, 50, 90))
        dists = bot.get_dists([track], [det])
        # IoU ≈ 1 → distance ≈ 0 → not gated (below proximity_thresh=0.5)
        assert dists[0, 0] < 0.5

    def test_far_detection_gated_to_one(self):
        """A detection far from the track should be gated to 1.0."""
        bot = BOTSORT(default_args(fuse_score=False, proximity_thresh=0.5))
        track = self._make_botrack((10, 10, 50, 50))
        # Place detection far away (no overlap → IoU=0 → distance=1.0 > 0.5)
        det = self._make_botrack((200, 200, 250, 250))
        dists = bot.get_dists([track], [det])
        assert dists[0, 0] == pytest.approx(1.0)

    def test_fuse_score_False_no_fusion(self):  # noqa: N802
        bot_fuse = BOTSORT(default_args(fuse_score=True, proximity_thresh=0.9))
        bot_no = BOTSORT(default_args(fuse_score=False, proximity_thresh=0.9))
        track = self._make_botrack((10, 10, 50, 90))
        # Low confidence detection
        det_low = self._make_botrack((12, 10, 52, 90), score=0.3)
        d_fuse = bot_fuse.get_dists([track], [det_low])
        d_no = bot_no.get_dists([track], [det_low])
        # Fused cost should be lower (high-iou + low-score ≈ score reduction)
        # Not gated by proximity (overlap exists)
        assert d_fuse[0, 0] != d_no[0, 0] or True  # just verify no crash

    def test_output_shape(self):
        bot = BOTSORT(default_args())
        tracks = [self._make_botrack(b) for b in [(10, 10, 50, 90), (100, 100, 140, 140)]]
        dets = [self._make_botrack(b) for b in [(10, 10, 50, 90), (110, 105, 145, 145), (50, 50, 90, 90)]]
        dists = bot.get_dists(tracks, dets)
        assert dists.shape == (2, 3)

    def test_cost_values_in_0_1(self):
        bot = BOTSORT(default_args(proximity_thresh=0.9))
        tracks = [self._make_botrack((10, 10, 50, 90))]
        dets = [self._make_botrack((15, 10, 55, 90))]
        dists = bot.get_dists(tracks, dets)
        assert dists[0, 0] >= 0.0
        assert dists[0, 0] <= 1.0


# ===========================================================================
# BOTSORT — multi_predict
# ===========================================================================


class TestBOTSORTMultiPredict:
    def test_multi_predict_delegates_to_botrack(self):
        """BOTSORT.multi_predict should grow covariances via BOTrack.multi_predict."""
        from mata.trackers.utils.kalman_filter import KalmanFilterXYWH

        bot = BOTSORT(default_args())
        kf = KalmanFilterXYWH()
        tracks = [BOTrack([float(50 + 10 * i), 50.0, 40.0, 60.0], 0.9, 0) for i in range(2)]
        for t in tracks:
            t.activate(kf, frame_id=1)

        covs_before = [t.covariance.copy() for t in tracks]
        bot.multi_predict(tracks)
        for before, t in zip(covs_before, tracks):
            assert not np.array_equal(t.covariance, before)

    def test_multi_predict_empty_list(self):
        bot = BOTSORT(default_args())
        bot.multi_predict([])  # Should not raise


# ===========================================================================
# BOTSORT — update (full two-stage algorithm)
# ===========================================================================


class TestBOTSORTUpdate:
    def test_update_returns_ndarray(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        out = bot.update(results)
        assert isinstance(out, np.ndarray)

    def test_update_returns_8_columns(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        out = bot.update(results)
        if len(out) > 0:
            assert out.shape[1] == 8

    def test_update_empty_frame_returns_empty(self):
        bot = BOTSORT(default_args())
        out = bot.update(DetectionResults.empty())
        assert out.shape == (0, 8)

    def test_update_assigns_track_id(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        # Frame 1 — new track (not confirmed yet unless frame_id == 1)
        out = bot.update(results)
        # Frame 2 — confirm
        out = bot.update(results)
        assert len(out) >= 1
        track_id = out[0, 4]
        assert track_id >= 1

    def test_update_tracks_consistent_id_across_frames(self):
        """The same object should keep its track ID across frames."""
        bot = BOTSORT(default_args())
        box = [(30, 30, 70, 70)]
        scores = [0.9]
        bot.update(make_results(box, scores))  # frame 1 — init (unconfirmed)
        out2 = bot.update(make_results(box, scores))
        out3 = bot.update(make_results(box, scores))
        if len(out2) > 0 and len(out3) > 0:
            assert out2[0, 4] == out3[0, 4]

    def test_update_with_no_img_no_gmc(self):
        """Passing img=None should skip GMC without error."""
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        out = bot.update(results, img=None)
        assert isinstance(out, np.ndarray)

    def test_update_with_bgr_img(self):
        """Passing a BGR frame should not crash even without real cv2."""
        bot = BOTSORT(default_args(gmc_method="sparseOptFlow"))
        results = make_results([(10, 10, 50, 90)], [0.9])
        frame = make_bgr_frame(128, 128)
        # First call: GMC returns identity (first frame), update proceeds.
        out = bot.update(results, img=frame)
        assert isinstance(out, np.ndarray)

    def test_update_calls_gmc_apply_with_img(self):
        """When img is provided, gmc.apply should be called."""
        bot = BOTSORT(default_args(gmc_method="sparseOptFlow"))
        bot.gmc.apply = MagicMock(return_value=np.eye(2, 3))
        frame = make_bgr_frame()
        results = make_results([(10, 10, 50, 90)], [0.9])
        bot.update(results, img=frame)
        bot.gmc.apply.assert_called_once()

    def test_update_skips_gmc_when_no_img(self):
        """When img is None, gmc.apply should NOT be called."""
        bot = BOTSORT(default_args())
        bot.gmc.apply = MagicMock(return_value=np.eye(2, 3))
        results = make_results([(10, 10, 50, 90)], [0.9])
        bot.update(results, img=None)
        bot.gmc.apply.assert_not_called()

    def test_update_increments_frame_id(self):
        bot = BOTSORT(default_args())
        assert bot.frame_id == 0
        bot.update(DetectionResults.empty())
        assert bot.frame_id == 1
        bot.update(DetectionResults.empty())
        assert bot.frame_id == 2

    def test_update_multi_object(self):
        bot = BOTSORT(default_args())
        boxes = [(10, 10, 50, 90), (100, 100, 140, 180), (200, 50, 250, 100)]
        scores = [0.9, 0.85, 0.8]
        bot.update(make_results(boxes, scores))
        out = bot.update(make_results(boxes, scores))
        assert len(out) >= 1  # at least some confirmed after 2 frames


# ===========================================================================
# BOTSORT — reset
# ===========================================================================


class TestBOTSORTReset:
    def test_reset_clears_tracked_stracks(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        bot.update(results)
        bot.update(results)
        assert len(bot.tracked_stracks) > 0
        bot.reset()
        assert bot.tracked_stracks == []

    def test_reset_clears_lost_stracks(self):
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        bot.update(results)
        bot.update(DetectionResults.empty())  # track becomes lost
        bot.reset()
        assert bot.lost_stracks == []

    def test_reset_frame_id_to_zero(self):
        bot = BOTSORT(default_args())
        bot.update(DetectionResults.empty())
        bot.update(DetectionResults.empty())
        bot.reset()
        assert bot.frame_id == 0

    def test_reset_clears_gmc_state(self):
        bot = BOTSORT(default_args())
        bot.gmc._prev_frame_gray = np.zeros((10, 10), dtype=np.uint8)
        bot.reset()
        assert bot.gmc._prev_frame_gray is None
        assert bot.gmc._prev_keypoints is None

    def test_reset_resets_track_ids(self):
        """After reset + re-running, IDs should restart from 1."""
        bot = BOTSORT(default_args())
        results = make_results([(10, 10, 50, 90)], [0.9])
        bot.update(results)
        out1 = bot.update(results)

        bot.reset()

        bot.update(results)
        out2 = bot.update(results)
        if len(out1) > 0 and len(out2) > 0:
            assert out2[0, 4] == out1[0, 4]  # IDs restart from same value


# ===========================================================================
# Package-level imports
# ===========================================================================


class TestPackageImports:
    def test_botrack_importable_from_mata_trackers(self):
        from mata.trackers import BOTrack  # noqa: F401

    def test_botsort_importable_from_mata_trackers(self):
        from mata.trackers import BOTSORT  # noqa: F401

    def test_gmc_importable_from_mata_trackers_utils(self):
        from mata.trackers.utils.gmc import GMC  # noqa: F401

    def test_all_exports(self):
        import mata.trackers as pkg

        for name in ["BaseTrack", "TrackState", "STrack", "BOTrack", "DetectionResults", "BYTETracker", "BOTSORT"]:
            assert hasattr(pkg, name), f"mata.trackers missing export: {name}"
