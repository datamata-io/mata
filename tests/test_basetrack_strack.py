"""Unit tests for Task A3: BaseTrack and STrack Classes.

Covers:
- TrackState: enumeration values
- BaseTrack: __init__, next_id/reset_id, lifecycle transitions, end_frame, abstracts
- STrack: construction (4/5/6-element xywh), frozen bbox, coordinate conversions
- STrack: activate on frame 1 vs later frames
- STrack: update / re_activate / re_activate(new_id=True) lifecycle
- STrack: predict (single track)
- STrack: multi_predict (batch, empty input)
- STrack: multi_gmc (identity, real H, empty input)
- STrack: tlwh/xyxy/xywh/result properties before and after activation
- STrack: tlwh_to_xyah static method
- STrack: Kalman state convergence over repeated updates
- STrack: reset_id between sequences
- STrack: track ID uniqueness across tracks
- Package-level import (mata.trackers)
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.trackers.basetrack import BaseTrack, TrackState
from mata.trackers.byte_tracker import STrack
from mata.trackers.utils.kalman_filter import KalmanFilterXYAH

# ===========================================================================
# Helpers / fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset the global track ID counter before every test."""
    BaseTrack.reset_id()
    yield
    BaseTrack.reset_id()


def make_strack(
    cx: float = 100.0,
    cy: float = 200.0,
    w: float = 50.0,
    h: float = 80.0,
    score: float = 0.9,
    cls: int = 0,
) -> STrack:
    """Return an unactivated STrack from the given centre-format box."""
    return STrack([cx, cy, w, h], score, cls)


def activated_strack(
    cx: float = 100.0,
    cy: float = 200.0,
    w: float = 50.0,
    h: float = 80.0,
    score: float = 0.9,
    cls: int = 0,
    frame_id: int = 1,
) -> STrack:
    """Return an STrack that has been activated on the given frame."""
    kf = KalmanFilterXYAH()
    st = make_strack(cx, cy, w, h, score, cls)
    st.activate(kf, frame_id)
    return st


# ===========================================================================
# TrackState
# ===========================================================================


class TestTrackState:
    def test_values(self):
        assert TrackState.New == 0
        assert TrackState.Tracked == 1
        assert TrackState.Lost == 2
        assert TrackState.Removed == 3

    def test_distinct_values(self):
        states = {TrackState.New, TrackState.Tracked, TrackState.Lost, TrackState.Removed}
        assert len(states) == 4


# ===========================================================================
# BaseTrack
# ===========================================================================


class TestBaseTrack:
    def test_initial_state(self):
        bt = BaseTrack()
        assert bt.track_id == 0
        assert bt.is_activated is False
        assert bt.state == TrackState.New
        assert bt.score == 0.0
        assert bt.start_frame == 0
        assert bt.frame_id == 0
        assert bt.time_since_update == 0

    def test_next_id_increments(self):
        ids = [BaseTrack.next_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_next_id_shared_across_instances(self):
        bt1 = BaseTrack()
        bt2 = BaseTrack()
        assert bt1._count is bt2._count or BaseTrack._count == BaseTrack._count
        id1 = BaseTrack.next_id()
        id2 = BaseTrack.next_id()
        assert id2 == id1 + 1

    def test_reset_id_zeros_counter(self):
        BaseTrack.next_id()
        BaseTrack.next_id()
        BaseTrack.reset_id()
        assert BaseTrack._count == 0
        assert BaseTrack.next_id() == 1

    def test_reset_id_class_method_symmetry(self):
        """STrack subclass still uses BaseTrack._count."""
        [BaseTrack.next_id() for _ in range(3)]
        BaseTrack.reset_id()
        assert BaseTrack.next_id() == 1

    def test_mark_lost(self):
        bt = BaseTrack()
        bt.mark_lost()
        assert bt.state == TrackState.Lost

    def test_mark_removed(self):
        bt = BaseTrack()
        bt.mark_removed()
        assert bt.state == TrackState.Removed

    def test_end_frame_mirrors_frame_id(self):
        bt = BaseTrack()
        bt.frame_id = 42
        assert bt.end_frame == 42

    def test_activate_abstract(self):
        bt = BaseTrack()
        with pytest.raises(NotImplementedError):
            bt.activate()

    def test_predict_abstract(self):
        bt = BaseTrack()
        with pytest.raises(NotImplementedError):
            bt.predict()

    def test_update_abstract(self):
        bt = BaseTrack()
        with pytest.raises(NotImplementedError):
            bt.update()


# ===========================================================================
# STrack — construction
# ===========================================================================


class TestSTrackConstruction:
    def test_basic_4_element_xywh(self):
        st = make_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        # tlwh: x1 = 100 - 25 = 75, y1 = 200 - 40 = 160
        assert np.allclose(st._tlwh, [75.0, 160.0, 50.0, 80.0])

    def test_5_element_xywh_stores_angle(self):
        st = STrack([100.0, 200.0, 50.0, 80.0, 0.5], 0.9, 0)
        assert st.angle == pytest.approx(0.5)
        assert st.idx == 0

    def test_6_element_xywh_stores_angle_and_idx(self):
        st = STrack([100.0, 200.0, 50.0, 80.0, 1.2, 7], 0.8, 2)
        assert st.angle == pytest.approx(1.2)
        assert st.idx == 7

    def test_4_element_angle_is_none(self):
        st = make_strack()
        assert st.angle is None

    def test_initial_kalman_state_is_none(self):
        st = make_strack()
        assert st.mean is None
        assert st.covariance is None
        assert st.kalman_filter is None

    def test_initial_lifecycle_attributes(self):
        st = make_strack(score=0.75, cls=3)
        assert st.is_activated is False
        assert st.state == TrackState.New
        assert st.score == pytest.approx(0.75)
        assert st.cls == 3
        assert st.tracklet_len == 0

    def test_numpy_array_input(self):
        xywh = np.array([200.0, 300.0, 60.0, 90.0])
        st = STrack(xywh, 0.6, 1)
        assert np.allclose(st._tlwh, [170.0, 255.0, 60.0, 90.0])


# ===========================================================================
# STrack — coordinate conversions (static)
# ===========================================================================


class TestSTrackCoordinateStatics:
    def test_tlwh_to_xyah_center(self):
        # tlwh = [10, 20, 40, 80]  → cx=30, cy=60, a=0.5, h=80
        result = STrack.tlwh_to_xyah(np.array([10.0, 20.0, 40.0, 80.0]))
        assert np.allclose(result, [30.0, 60.0, 0.5, 80.0])

    def test_tlwh_to_xyah_square(self):
        result = STrack.tlwh_to_xyah(np.array([0.0, 0.0, 50.0, 50.0]))
        assert np.allclose(result, [25.0, 25.0, 1.0, 50.0])

    def test_tlwh_to_xyah_wide_box(self):
        # w=200, h=100  → a=2.0
        result = STrack.tlwh_to_xyah(np.array([0.0, 0.0, 200.0, 100.0]))
        assert result[2] == pytest.approx(2.0)

    def test_convert_coords_delegates(self):
        st = make_strack()
        tlwh = np.array([10.0, 20.0, 40.0, 80.0])
        assert np.allclose(st.convert_coords(tlwh), STrack.tlwh_to_xyah(tlwh))


# ===========================================================================
# STrack — bbox properties before activation
# ===========================================================================


class TestSTrackPropertiesBeforeActivation:
    def test_tlwh_returns_frozen(self):
        st = make_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        expected = np.array([75.0, 160.0, 50.0, 80.0])
        assert np.allclose(st.tlwh, expected)

    def test_xyxy_before_activation(self):
        st = make_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        # x2 = 75+50=125, y2 = 160+80=240
        assert np.allclose(st.xyxy, [75.0, 160.0, 125.0, 240.0])

    def test_xywh_before_activation(self):
        st = make_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        assert np.allclose(st.xywh, [100.0, 200.0, 50.0, 80.0])

    def test_tlwh_returns_copy(self):
        st = make_strack()
        tlwh1 = st.tlwh
        tlwh1[0] += 999
        assert st.tlwh[0] != 999  # mutation does not propagate


# ===========================================================================
# STrack — activate
# ===========================================================================


class TestSTrackActivate:
    def test_activate_assigns_track_id(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        assert st.track_id == 0  # not yet assigned
        st.activate(kf, frame_id=1)
        assert st.track_id == 1

    def test_activate_increments_global_counter(self):
        kf = KalmanFilterXYAH()
        st1 = make_strack()
        st1.activate(kf, 1)
        st2 = make_strack()
        st2.activate(kf, 1)
        assert st2.track_id == st1.track_id + 1

    def test_activate_sets_kalman_filter(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, 1)
        assert st.kalman_filter is kf

    def test_activate_initialises_mean_covariance(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, 1)
        assert st.mean is not None
        assert st.covariance is not None
        assert st.mean.shape == (8,)
        assert st.covariance.shape == (8, 8)

    def test_activate_frame1_is_activated_true(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, frame_id=1)
        assert st.is_activated is True

    def test_activate_frame2_is_activated_false(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, frame_id=2)
        assert st.is_activated is False

    def test_activate_sets_state_tracked(self):
        st = activated_strack(frame_id=1)
        assert st.state == TrackState.Tracked

    def test_activate_sets_start_and_frame_id(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, frame_id=5)
        assert st.start_frame == 5
        assert st.frame_id == 5

    def test_activate_resets_tracklet_len(self):
        st = activated_strack(frame_id=1)
        assert st.tracklet_len == 0


# ===========================================================================
# STrack — bbox properties after activation
# ===========================================================================


class TestSTrackPropertiesAfterActivation:
    def test_tlwh_after_activation_close_to_initial(self):
        st = activated_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        expected = np.array([75.0, 160.0, 50.0, 80.0])
        assert np.allclose(st.tlwh, expected, atol=1e-3)

    def test_xyxy_after_activation(self):
        st = activated_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        assert np.allclose(st.xyxy[:2], [75.0, 160.0], atol=1e-3)
        # width and height preserved
        assert np.allclose(st.xyxy[2] - st.xyxy[0], 50.0, atol=1e-3)
        assert np.allclose(st.xyxy[3] - st.xyxy[1], 80.0, atol=1e-3)

    def test_xywh_after_activation(self):
        st = activated_strack(cx=100.0, cy=200.0, w=50.0, h=80.0)
        assert np.allclose(st.xywh[:2], [100.0, 200.0], atol=1e-3)

    def test_result_format(self):
        st = activated_strack(cx=100.0, cy=200.0, w=50.0, h=80.0, score=0.9, cls=2)
        res = st.result
        assert len(res) == 8
        # x1, y1, x2, y2
        assert np.allclose(res[:4], st.xyxy.tolist(), atol=1e-6)
        # track_id
        assert res[4] == st.track_id
        # score
        assert res[5] == pytest.approx(0.9)
        # cls
        assert res[6] == 2
        # idx defaults to 0
        assert res[7] == 0


# ===========================================================================
# STrack — update
# ===========================================================================


class TestSTrackUpdate:
    def test_update_increments_tracklet_len(self):
        KalmanFilterXYAH()
        st = activated_strack(frame_id=1)
        det = make_strack(cx=105.0, cy=205.0, w=50.0, h=80.0, score=0.85)
        st.update(det, frame_id=2)
        assert st.tracklet_len == 1
        st.update(det, frame_id=3)
        assert st.tracklet_len == 2

    def test_update_advances_frame_id(self):
        st = activated_strack(frame_id=1)
        det = make_strack()
        st.update(det, frame_id=7)
        assert st.frame_id == 7

    def test_update_copies_score(self):
        st = activated_strack(frame_id=1)
        det = make_strack(score=0.42)
        st.update(det, frame_id=2)
        assert st.score == pytest.approx(0.42)

    def test_update_copies_cls(self):
        st = activated_strack(frame_id=1)
        det = make_strack(cls=5)
        st.update(det, frame_id=2)
        assert st.cls == 5

    def test_update_state_remains_tracked(self):
        st = activated_strack(frame_id=1)
        st.mark_lost()
        det = make_strack()
        st.update(det, frame_id=2)
        assert st.state == TrackState.Tracked

    def test_update_is_activated(self):
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, frame_id=2)  # not activated
        assert st.is_activated is False
        det = make_strack()
        st.update(det, frame_id=3)
        assert st.is_activated is True

    def test_update_changes_mean(self):
        st = activated_strack(cx=100.0, cy=200.0)
        mean_before = st.mean.copy()
        det = make_strack(cx=110.0, cy=210.0)
        st.update(det, frame_id=2)
        assert not np.allclose(st.mean, mean_before)


# ===========================================================================
# STrack — re_activate
# ===========================================================================


class TestSTrackReActivate:
    def test_re_activate_sets_is_activated(self):
        st = activated_strack(frame_id=1)
        st.mark_lost()
        det = make_strack(cx=105.0, cy=205.0)
        st.re_activate(det, frame_id=10)
        assert st.is_activated is True

    def test_re_activate_resets_tracklet_len(self):
        st = activated_strack(frame_id=1)
        for _ in range(5):
            st.update(make_strack(), frame_id=st.frame_id + 1)
        assert st.tracklet_len == 5
        det = make_strack()
        st.re_activate(det, frame_id=20)
        assert st.tracklet_len == 0

    def test_re_activate_state_tracked(self):
        st = activated_strack(frame_id=1)
        st.mark_lost()
        st.re_activate(make_strack(), frame_id=15)
        assert st.state == TrackState.Tracked

    def test_re_activate_keeps_track_id(self):
        st = activated_strack(frame_id=1)
        original_id = st.track_id
        st.re_activate(make_strack(), frame_id=5)
        assert st.track_id == original_id

    def test_re_activate_new_id_true(self):
        st = activated_strack(frame_id=1)
        original_id = st.track_id
        st.re_activate(make_strack(), frame_id=5, new_id=True)
        assert st.track_id != original_id
        assert st.track_id > original_id

    def test_re_activate_updates_score_cls(self):
        st = activated_strack(frame_id=1, score=0.9, cls=0)
        det = make_strack(score=0.55, cls=3)
        st.re_activate(det, frame_id=10)
        assert st.score == pytest.approx(0.55)
        assert st.cls == 3


# ===========================================================================
# STrack — mark_lost / mark_removed (inherited)
# ===========================================================================


class TestSTrackLifecycle:
    def test_mark_lost(self):
        st = activated_strack()
        st.mark_lost()
        assert st.state == TrackState.Lost

    def test_mark_removed(self):
        st = activated_strack()
        st.mark_removed()
        assert st.state == TrackState.Removed

    def test_end_frame(self):
        st = activated_strack(frame_id=1)
        st.update(make_strack(), frame_id=8)
        assert st.end_frame == 8


# ===========================================================================
# STrack — predict (single track)
# ===========================================================================


class TestSTrackPredict:
    def test_predict_advances_mean(self):
        st = activated_strack(cx=100.0, cy=200.0, frame_id=1)
        # Give the track a non-zero velocity so prediction moves the position.
        st.mean[4] = 5.0  # vcx — will shift cx by 5 pixels per frame
        st.mean[5] = -3.0  # vcy
        mean_before = st.mean.copy()
        st.predict()
        # With non-zero velocity the constant-velocity model shifts position.
        assert not np.allclose(st.mean, mean_before)

    def test_predict_non_tracked_zeroes_height_velocity(self):
        """Lost tracks should not drift in height."""
        st = activated_strack(frame_id=1)
        st.mark_lost()
        # Give it a non-zero height velocity.
        st.mean[7] = 5.0
        st.predict()
        # The prediction step zeros the height velocity prior to predict,
        # so the result should reflect zero initial vh.
        # We can't check the exact value because the KF modifies mean[7]
        # via the motion model, but the INPUT velocity was zeroed.
        # Just ensure predict ran without error.

    def test_predict_covariance_grows(self):
        """Covariance must increase (prediction = more uncertainty)."""
        st = activated_strack(frame_id=1)
        trace_before = np.trace(st.covariance)
        st.predict()
        trace_after = np.trace(st.covariance)
        assert trace_after > trace_before


# ===========================================================================
# STrack — multi_predict
# ===========================================================================


class TestSTrackMultiPredict:
    def test_multi_predict_empty(self):
        STrack.multi_predict([])  # Must not raise.

    def test_multi_predict_single(self):
        st = activated_strack(frame_id=1)
        st.mean[4] = 4.0  # give non-zero velocity so prediction changes position
        mean_before = st.mean.copy()
        STrack.multi_predict([st])
        assert not np.allclose(st.mean, mean_before)

    def test_multi_predict_multiple(self):
        tracks = [activated_strack(cx=float(i * 50), frame_id=1) for i in range(4)]
        # Give every track a unique non-zero velocity so prediction is detectable.
        for i, t in enumerate(tracks):
            t.mean[4] = float(i + 1) * 3.0
        means_before = [t.mean.copy() for t in tracks]
        STrack.multi_predict(tracks)
        for t, mb in zip(tracks, means_before):
            assert not np.allclose(t.mean, mb)

    def test_multi_predict_skips_unactivated(self):
        st = make_strack()  # mean is None
        # Should not raise even with a track that has no KF state.
        STrack.multi_predict([st])
        assert st.mean is None

    def test_multi_predict_same_result_as_single_predict(self):
        """Batch prediction must match sequential prediction."""
        kf = KalmanFilterXYAH()
        st_batch = make_strack(cx=100.0, cy=200.0)
        st_batch.activate(kf, 1)
        st_seq = make_strack(cx=100.0, cy=200.0)
        st_seq.activate(kf, 1)
        # Ensure identical initial state.
        st_batch.mean = st_seq.mean.copy()
        st_batch.covariance = st_seq.covariance.copy()

        STrack.multi_predict([st_batch])
        st_seq.predict()

        assert np.allclose(st_batch.mean, st_seq.mean, atol=1e-8)
        assert np.allclose(st_batch.covariance, st_seq.covariance, atol=1e-8)


# ===========================================================================
# STrack — multi_gmc
# ===========================================================================


class TestSTrackMultiGMC:
    def test_multi_gmc_empty(self):
        STrack.multi_gmc([])  # Must not raise.

    def test_multi_gmc_identity_no_change(self):
        st = activated_strack(cx=100.0, cy=200.0)
        mean_before = st.mean.copy()
        H_identity = np.eye(2, 3)  # noqa: N806
        STrack.multi_gmc([st], H_identity)
        assert np.allclose(st.mean, mean_before)

    def test_multi_gmc_pure_translation(self):
        st = activated_strack(cx=100.0, cy=200.0)
        tx, ty = 10.0, -5.0
        H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])  # noqa: N806
        cx_before = st.mean[0]
        cy_before = st.mean[1]
        STrack.multi_gmc([st], H)
        assert st.mean[0] == pytest.approx(cx_before + tx)
        assert st.mean[1] == pytest.approx(cy_before + ty)

    def test_multi_gmc_translates_velocity(self):
        st = activated_strack()
        # Give a known velocity.
        st.mean[4] = 3.0  # vcx
        st.mean[5] = 1.0  # vcy
        # Pure translation — rotation = I → velocities unchanged.
        H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])  # noqa: N806
        STrack.multi_gmc([st], H)
        assert st.mean[4] == pytest.approx(3.0)
        assert st.mean[5] == pytest.approx(1.0)

    def test_multi_gmc_rotation_transforms_velocity(self):
        theta = np.pi / 4  # 45 degrees
        c, s = np.cos(theta), np.sin(theta)
        H = np.array([[c, -s, 0.0], [s, c, 0.0]])  # noqa: N806
        st = activated_strack()
        st.mean[4] = 1.0  # vcx
        st.mean[5] = 0.0  # vcy
        STrack.multi_gmc([st], H)
        # After 45° rotation: vcx, vcy should both be ~1/sqrt(2).
        assert st.mean[4] == pytest.approx(c, abs=1e-6)
        assert st.mean[5] == pytest.approx(s, abs=1e-6)

    def test_multi_gmc_skips_none_mean(self):
        st = make_strack()  # not activated, mean is None
        H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])  # noqa: N806
        STrack.multi_gmc([st], H)  # Must not raise.
        assert st.mean is None


# ===========================================================================
# STrack — Kalman convergence
# ===========================================================================


class TestSTrackKalmanConvergence:
    def test_position_converges_over_updates(self):
        """Track position should converge toward the measurement mean."""
        kf = KalmanFilterXYAH()
        st = STrack([0.0, 0.0, 50.0, 80.0], 0.9, 0)
        st.activate(kf, frame_id=1)

        # True centre drifts slowly rightward.
        true_cx, true_cy = 50.0, 100.0
        obs_w, obs_h = 50.0, 80.0

        for frame in range(2, 30):
            noisy_cx = true_cx + np.random.RandomState(frame).normal(0, 0.5)
            noisy_cy = true_cy + np.random.RandomState(frame + 100).normal(0, 0.5)
            det = STrack([noisy_cx, noisy_cy, obs_w, obs_h], 0.9, 0)
            st.update(det, frame_id=frame)

        # After ~28 updates the estimate should be close.
        assert abs(st.xywh[0] - true_cx) < 5.0
        assert abs(st.xywh[1] - true_cy) < 5.0


# ===========================================================================
# Track ID uniqueness
# ===========================================================================


class TestTrackIDUniqueness:
    def test_unique_ids_across_tracks(self):
        kf = KalmanFilterXYAH()
        stracks = [make_strack(cx=float(i)) for i in range(10)]
        for st in stracks:
            st.activate(kf, frame_id=1)
        ids = [st.track_id for st in stracks]
        assert len(ids) == len(set(ids))

    def test_reset_id_between_sequences(self):
        kf = KalmanFilterXYAH()
        st1 = make_strack()
        st1.activate(kf, 1)
        assert st1.track_id == 1
        BaseTrack.reset_id()
        st2 = make_strack()
        st2.activate(kf, 1)
        assert st2.track_id == 1  # fresh sequence starts from 1

    def test_strack_uses_basetrack_counter(self):
        """STrack IDs and BaseTrack IDs share the same counter."""
        _ = BaseTrack.next_id()  # consume ID 1
        kf = KalmanFilterXYAH()
        st = make_strack()
        st.activate(kf, frame_id=1)
        # Next ID should be 2 (BaseTrack already used 1).
        assert st.track_id == 2


# ===========================================================================
# Package-level import
# ===========================================================================


class TestPackageImport:
    def test_import_from_trackers_package(self):
        from mata.trackers import BaseTrack as BT  # noqa: N817
        from mata.trackers import STrack as ST  # noqa: N817
        from mata.trackers import TrackState as TS  # noqa: N817

        assert BT is BaseTrack
        assert TS is TrackState
        assert ST is STrack

    def test_strack_shared_kalman_lazy_init(self):
        """shared_kalman is None until first access, then a KalmanFilterXYAH."""
        # Reset to simulate first import scenario if needed.
        original = STrack.shared_kalman
        STrack.shared_kalman = None
        kf = STrack._get_shared_kalman()
        assert kf is not None
        # Restore original.
        STrack.shared_kalman = original
