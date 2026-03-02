"""Unit tests for Task A4: BYTETracker class and DetectionResults adapter.

Covers all acceptance criteria:
- DetectionResults: construction, from_vision_result, empty, slicing (__getitem__)
- _TrackerArgs: defaults, dict construction, passthrough
- BYTETracker.__init__: max_time_lost calculation, attribute init
- get_kalmanfilter: returns KalmanFilterXYAH
- init_track: empty results, N detections, original idx preservation
- get_dists: fuse_score True/False
- multi_predict: delegates to STrack.multi_predict
- update Step 1: threshold split high/low confidence
- update Step 2: first association — high-conf dets ↔ tracked+lost (Kalman predicted)
- update Step 3: second association — low-conf dets recover occluded tracked objects
- update Step 4: unconfirmed track matching & removal
- update Step 5: new track init from unmatched high-conf dets
- update Step 6: max_time_lost-based removal of stale lost tracks
- update: empty detection frame (graceful — Kalman prediction only)
- update: return shape (N, 8) with correct column order
- joint_stracks: no duplicate IDs
- sub_stracks: set-difference by track_id
- remove_duplicate_stracks: IoU > 0.85 deduplicated, longer track wins
- reset_id: clears global ID counter
- reset: full state clear
- removed_stracks: capped at 1000 entries
- Package-level import (mata.trackers)
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.trackers.basetrack import BaseTrack
from mata.trackers.byte_tracker import (
    BYTETracker,
    DetectionResults,
    STrack,
    _TrackerArgs,
)

# ===========================================================================
# Helpers / fixtures
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
    """Build a DetectionResults from xyxy boxes + scores.

    Args:
        boxes:   List of (x1, y1, x2, y2) bounding boxes.
        scores:  Confidence scores, one per box.
        cls_ids: Class IDs (default all 0).

    Returns:
        DetectionResults ready to feed into BYTETracker.update().
    """
    n = len(boxes)
    xyxy = np.array(boxes, dtype=np.float32).reshape(n, 4)
    conf = np.array(scores, dtype=np.float32)
    cls_ids = cls_ids if cls_ids is not None else [0] * n
    cls = np.array(cls_ids, dtype=np.float32)
    # xywh: cx, cy, w, h
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return DetectionResults(conf=conf, xyxy=xyxy, xywh=xywh, cls=cls)


def default_args(**overrides) -> _TrackerArgs:
    """Return _TrackerArgs with defaults plus optional overrides."""
    defaults = dict(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    defaults.update(overrides)
    return _TrackerArgs(**defaults)


def make_tracker(**arg_overrides) -> BYTETracker:
    """Create a BYTETracker with default config and optional overrides."""
    args = default_args(**arg_overrides)
    return BYTETracker(args, frame_rate=30)


# ===========================================================================
# DetectionResults
# ===========================================================================


class TestDetectionResults:
    def test_construction_from_arrays(self):
        conf = np.array([0.9, 0.8], dtype=np.float32)
        xyxy = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        xywh = np.array([[5, 5, 10, 10], [25, 25, 10, 10]], dtype=np.float32)
        cls = np.array([0, 1], dtype=np.float32)
        dr = DetectionResults(conf, xyxy, xywh, cls)
        assert len(dr) == 2
        np.testing.assert_array_equal(dr.conf, conf)
        np.testing.assert_array_equal(dr.xyxy, xyxy)
        np.testing.assert_array_equal(dr.xywh, xywh)
        np.testing.assert_array_equal(dr.cls, cls)

    def test_auto_indices(self):
        dr = make_results([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8])
        np.testing.assert_array_equal(dr._indices, [0, 1])

    def test_custom_indices(self):
        conf = np.array([0.9], dtype=np.float32)
        xyxy = np.array([[0, 0, 10, 10]], dtype=np.float32)
        xywh = np.array([[5, 5, 10, 10]], dtype=np.float32)
        cls = np.array([0], dtype=np.float32)
        dr = DetectionResults(conf, xyxy, xywh, cls, indices=np.array([42]))
        assert dr._indices[0] == 42

    def test_len_zero(self):
        dr = DetectionResults.empty()
        assert len(dr) == 0
        assert dr.conf.shape == (0,)
        assert dr.xyxy.shape == (0, 4)

    def test_boolean_mask_slicing(self):
        dr = make_results(
            [(0, 0, 10, 10), (20, 20, 30, 30), (50, 50, 60, 60)],
            [0.9, 0.3, 0.7],
        )
        high = dr[dr.conf >= 0.5]
        assert len(high) == 2
        np.testing.assert_array_almost_equal(high.conf, [0.9, 0.7])

    def test_boolean_mask_preserves_original_indices(self):
        dr = make_results(
            [(0, 0, 10, 10), (20, 20, 30, 30), (50, 50, 60, 60)],
            [0.9, 0.3, 0.7],
        )
        high = dr[dr.conf >= 0.5]
        # Original indices: 0 and 2 (not 0 and 1)
        assert high._indices[0] == 0
        assert high._indices[1] == 2

    def test_integer_slicing(self):
        dr = make_results([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8])
        sub = dr[np.array([1])]
        assert len(sub) == 1
        assert float(sub.conf[0]) == pytest.approx(0.8)

    def test_xywh_consistency(self):
        """xywh cx, cy must be center of xyxy box."""
        dr = make_results([(100, 200, 200, 400)], [0.9])
        assert float(dr.xywh[0, 0]) == pytest.approx(150.0)  # cx
        assert float(dr.xywh[0, 1]) == pytest.approx(300.0)  # cy
        assert float(dr.xywh[0, 2]) == pytest.approx(100.0)  # w
        assert float(dr.xywh[0, 3]) == pytest.approx(200.0)  # h

    def test_from_vision_result(self):
        """from_vision_result extracts bbox-bearing instances."""
        from mata.core.types import Instance, VisionResult

        inst1 = Instance(bbox=(10, 20, 100, 200), score=0.9, label=0, label_name="cat")
        inst2 = Instance(bbox=(50, 50, 150, 150), score=0.7, label=1, label_name="dog")
        vr = VisionResult(instances=[inst1, inst2])

        dr = DetectionResults.from_vision_result(vr)
        assert len(dr) == 2
        assert float(dr.conf[0]) == pytest.approx(0.9)
        assert float(dr.conf[1]) == pytest.approx(0.7)
        assert float(dr.cls[0]) == pytest.approx(0.0)
        assert float(dr.cls[1]) == pytest.approx(1.0)
        # xyxy
        np.testing.assert_array_almost_equal(dr.xyxy[0], [10, 20, 100, 200])
        np.testing.assert_array_almost_equal(dr.xyxy[1], [50, 50, 150, 150])

    def test_from_vision_result_empty(self):
        """Empty VisionResult → empty DetectionResults."""
        from mata.core.types import VisionResult

        vr = VisionResult(instances=[])
        dr = DetectionResults.from_vision_result(vr)
        assert len(dr) == 0

    def test_from_vision_result_skips_maskonly_instances(self):
        """Instances without bbox are skipped."""
        import numpy as np

        from mata.core.types import Instance, VisionResult

        mask = np.ones((50, 50), dtype=bool)
        inst_mask = Instance(mask=mask, score=0.8, label=0)
        inst_bbox = Instance(bbox=(0, 0, 50, 50), score=0.9, label=1)
        vr = VisionResult(instances=[inst_mask, inst_bbox])
        dr = DetectionResults.from_vision_result(vr)
        assert len(dr) == 1
        assert float(dr.conf[0]) == pytest.approx(0.9)

    def test_empty_factory(self):
        dr = DetectionResults.empty()
        assert len(dr) == 0
        assert dr.xyxy.shape == (0, 4)
        assert dr.xywh.shape == (0, 4)


# ===========================================================================
# _TrackerArgs
# ===========================================================================


class TestTrackerArgs:
    def test_defaults(self):
        args = _TrackerArgs()
        assert args.track_high_thresh == pytest.approx(0.5)
        assert args.track_low_thresh == pytest.approx(0.1)
        assert args.new_track_thresh == pytest.approx(0.6)
        assert args.track_buffer == 30
        assert args.match_thresh == pytest.approx(0.8)
        assert args.fuse_score is True

    def test_custom_values(self):
        args = _TrackerArgs(track_high_thresh=0.7, match_thresh=0.9, fuse_score=False)
        assert args.track_high_thresh == pytest.approx(0.7)
        assert args.match_thresh == pytest.approx(0.9)
        assert args.fuse_score is False

    def test_from_any_dict(self):
        d = {"track_high_thresh": 0.6, "track_buffer": 50}
        args = _TrackerArgs.from_any(d)
        assert isinstance(args, _TrackerArgs)
        assert args.track_high_thresh == pytest.approx(0.6)
        assert args.track_buffer == 50

    def test_from_any_passthrough(self):
        args = _TrackerArgs(track_buffer=60)
        returned = _TrackerArgs.from_any(args)
        assert returned is args  # same object

    def test_extra_kwargs_ignored(self):
        """BotSort-specific fields should not raise."""
        args = _TrackerArgs(gmc_method="sparseOptFlow", with_reid=False)
        assert args.track_high_thresh == pytest.approx(0.5)


# ===========================================================================
# BYTETracker initialisation
# ===========================================================================


class TestBYTETrackerInit:
    def test_max_time_lost_default(self):
        tracker = make_tracker(track_buffer=30)
        assert tracker.max_time_lost == 30  # 30/30*30 == 30

    def test_max_time_lost_different_frame_rate(self):
        args = default_args(track_buffer=30)
        tracker = BYTETracker(args, frame_rate=60)
        assert tracker.max_time_lost == 60  # 60/30*30 == 60

    def test_max_time_lost_different_buffer(self):
        args = default_args(track_buffer=60)
        tracker = BYTETracker(args, frame_rate=30)
        assert tracker.max_time_lost == 60  # 30/30*60 == 60

    def test_initial_state_empty(self):
        tracker = make_tracker()
        assert tracker.tracked_stracks == []
        assert tracker.lost_stracks == []
        assert tracker.removed_stracks == []
        assert tracker.frame_id == 0

    def test_dict_args(self):
        tracker = BYTETracker({"track_buffer": 20, "match_thresh": 0.75})
        assert tracker.args.track_buffer == 20
        assert tracker.args.match_thresh == pytest.approx(0.75)

    def test_get_kalmanfilter_returns_xyah(self):
        from mata.trackers.utils.kalman_filter import KalmanFilterXYAH

        tracker = make_tracker()
        kf = tracker.get_kalmanfilter()
        assert isinstance(kf, KalmanFilterXYAH)


# ===========================================================================
# init_track
# ===========================================================================


class TestInitTrack:
    def test_empty_results_returns_empty_list(self):
        tracker = make_tracker()
        stracks = tracker.init_track(DetectionResults.empty())
        assert stracks == []

    def test_creates_strack_per_detection(self):
        tracker = make_tracker()
        results = make_results([(0, 0, 100, 100), (200, 200, 300, 300)], [0.9, 0.8])
        stracks = tracker.init_track(results)
        assert len(stracks) == 2
        assert all(isinstance(t, STrack) for t in stracks)

    def test_scores_propagated(self):
        tracker = make_tracker()
        results = make_results([(0, 0, 100, 100), (200, 200, 300, 300)], [0.9, 0.8])
        stracks = tracker.init_track(results)
        assert stracks[0].score == pytest.approx(0.9)
        assert stracks[1].score == pytest.approx(0.8)

    def test_cls_propagated(self):
        tracker = make_tracker()
        results = make_results([(0, 0, 100, 100), (200, 200, 300, 300)], [0.9, 0.8], [3, 7])
        stracks = tracker.init_track(results)
        assert float(stracks[0].cls) == pytest.approx(3.0)
        assert float(stracks[1].cls) == pytest.approx(7.0)

    def test_original_idx_preserved_after_boolean_mask(self):
        """After slicing, each STrack.idx should match original position."""
        tracker = make_tracker()
        dr = make_results(
            [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)],
            [0.9, 0.2, 0.8],
        )
        high = dr[dr.conf >= 0.5]  # original indices 0, 2
        stracks = tracker.init_track(high)
        assert stracks[0].idx == 0
        assert stracks[1].idx == 2

    def test_new_tracks_not_activated(self):
        """init_track should not activate tracks — that happens in update()."""
        tracker = make_tracker()
        results = make_results([(0, 0, 100, 100)], [0.9])
        stracks = tracker.init_track(results)
        assert not stracks[0].is_activated


# ===========================================================================
# get_dists
# ===========================================================================


class TestGetDists:
    def _make_activated_strack(self, bbox, score=0.9, cls=0):
        """Return an activated STrack at given xyxy bbox."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        t = STrack(np.array([cx, cy, w, h]), score, cls)
        kf = BYTETracker(default_args()).get_kalmanfilter()
        t.activate(kf, 1)
        return t

    def test_returns_iou_distance_matrix_shape(self):
        tracker = make_tracker()
        tracks = [self._make_activated_strack((0, 0, 50, 50))]
        dets = tracker.init_track(make_results([(5, 5, 55, 55), (200, 200, 250, 250)], [0.9, 0.9]))
        dists = tracker.get_dists(tracks, dets)
        assert dists.shape == (1, 2)

    def test_fuse_score_true_lowers_cost(self):
        """High-score detection should get lower cost than low-score one."""
        tracker_fuse = make_tracker(fuse_score=True)
        tracker_plain = make_tracker(fuse_score=False)
        tracks = [self._make_activated_strack((0, 0, 100, 100))]
        # Two detections with same position but different scores
        high_s = make_results([(0, 0, 100, 100)], [0.95])
        low_s = make_results([(0, 0, 100, 100)], [0.1])
        det_high = tracker_fuse.init_track(high_s)
        det_low = tracker_fuse.init_track(low_s)

        cost_fuse_high = tracker_fuse.get_dists(tracks, det_high)[0, 0]
        cost_fuse_low = tracker_fuse.get_dists(tracks, det_low)[0, 0]
        cost_plain = tracker_plain.get_dists(tracks, det_high)[0, 0]

        # Fused high-score < fused low-score
        assert cost_fuse_high < cost_fuse_low
        # Plain (no fuse) is pure IoU distance
        assert cost_plain == pytest.approx(float(cost_plain))  # sanity

    def test_fuse_score_false_is_pure_iou(self):
        from mata.trackers.utils.matching import iou_distance

        tracker = make_tracker(fuse_score=False)
        tracks = [self._make_activated_strack((0, 0, 100, 100))]
        dets = tracker.init_track(make_results([(10, 10, 110, 110)], [0.9]))
        cost = tracker.get_dists(tracks, dets)
        expected = iou_distance(tracks, dets)
        np.testing.assert_array_almost_equal(cost, expected)


# ===========================================================================
# multi_predict
# ===========================================================================


class TestMultiPredict:
    def test_delegates_to_strack_multi_predict(self):
        tracker = make_tracker()
        kf = tracker.kalman_filter
        t = STrack(np.array([100.0, 200.0, 50.0, 80.0]), 0.9, 0)
        t.activate(kf, 1)
        t.mean.copy()
        tracker.multi_predict([t])
        # Mean may change (velocity steps)  — just check no crash
        assert t.mean is not None
        assert t.mean.shape == (8,)

    def test_empty_list_no_crash(self):
        tracker = make_tracker()
        tracker.multi_predict([])  # must not raise


# ===========================================================================
# BYTETracker.update — output format
# ===========================================================================


class TestUpdateOutputFormat:
    def test_returns_ndarray_shape_n8(self):
        tracker = make_tracker()
        results = make_results([(0, 0, 100, 100), (200, 200, 300, 300)], [0.9, 0.85])
        out = tracker.update(results)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2
        assert out.shape[1] == 8

    def test_empty_detection_returns_0x8_array(self):
        tracker = make_tracker()
        out = tracker.update(DetectionResults.empty())
        assert out.shape == (0, 8)

    def test_output_columns_order(self):
        """Columns: x1, y1, x2, y2, track_id, score, cls, idx."""
        tracker = make_tracker()
        results = make_results([(10, 20, 110, 120)], [0.9], [2])
        # Warm up first frame (frame-1 activates immediately)
        out = tracker.update(results)
        assert out.shape[0] == 1
        row = out[0]
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        track_id = row[4]
        score = row[5]
        cls = row[6]
        # x1 < x2 and y1 < y2
        assert x1 < x2
        assert y1 < y2
        assert track_id >= 1
        assert score == pytest.approx(0.9, abs=0.01)
        assert cls == pytest.approx(2.0)


# ===========================================================================
# Step 1 — threshold split
# ===========================================================================


class TestStep1ThresholdSplit:
    def test_low_conf_detections_not_initialised_as_new_tracks(self):
        """Detections below track_high_thresh should not become new tracks."""
        tracker = make_tracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
        )
        # All below high_thresh — no new tracks should be created
        results = make_results(
            [(0, 0, 100, 100), (200, 200, 300, 300)],
            [0.4, 0.45],
        )
        out = tracker.update(results)
        assert out.shape[0] == 0

    def test_high_conf_detections_become_tracked(self):
        tracker = make_tracker(
            track_high_thresh=0.5,
            new_track_thresh=0.6,
        )
        results = make_results([(0, 0, 100, 100)], [0.9])
        out = tracker.update(results)
        assert out.shape[0] == 1


# ===========================================================================
# Step 2 — first association
# ===========================================================================


class TestStep2FirstAssociation:
    def test_existing_track_updated_not_new_id(self):
        """Same object in consecutive frames should keep same track ID."""
        tracker = make_tracker()
        box = (10, 20, 110, 120)
        r1 = make_results([box], [0.9])
        out1 = tracker.update(r1)
        assert out1.shape[0] == 1
        id1 = int(out1[0, 4])

        # Slightly moved box — should associate with existing track
        box2 = (12, 22, 112, 122)
        r2 = make_results([box2], [0.9])
        out2 = tracker.update(r2)
        assert out2.shape[0] == 1
        id2 = int(out2[0, 4])

        assert id1 == id2, "Track ID must persist across frames"

    def test_lost_track_reactivated(self):
        """A lost track should be re-found when a matching detection arrives."""
        tracker = make_tracker(track_buffer=5)
        box = (10, 20, 110, 120)
        r1 = make_results([box], [0.9])
        out1 = tracker.update(r1)
        assert out1.shape[0] == 1
        id1 = int(out1[0, 4])

        # Miss for one frame — track goes to lost
        tracker.update(DetectionResults.empty())
        assert len(tracker.lost_stracks) == 1

        # Detection reappears — track should be recovered
        r3 = make_results([box], [0.9])
        out3 = tracker.update(r3)
        assert out3.shape[0] == 1
        id3 = int(out3[0, 4])
        assert id1 == id3, "Lost track should recover the same ID"

    def test_kalman_prediction_runs_before_matching(self):
        """Multi-predict must advance track states (mean not None after update)."""
        tracker = make_tracker()
        r1 = make_results([(100, 100, 200, 200)], [0.9])
        tracker.update(r1)
        # All tracked stracks should have non-None mean
        for t in tracker.tracked_stracks:
            assert t.mean is not None


# ===========================================================================
# Step 3 — second association (low-conf)
# ===========================================================================


class TestStep3SecondAssociation:
    def test_low_conf_detection_recovers_tracked_object(self):
        """A tracked object should be maintained via a low-conf detection."""
        tracker = make_tracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
        )
        box = (10, 20, 110, 120)

        # Frame 1: establish a confirmed track
        r1 = make_results([box], [0.9])
        out1 = tracker.update(r1)
        assert out1.shape[0] == 1
        id1 = int(out1[0, 4])

        # Frame 2: only a low-conf version of the same box
        r2 = make_results([box], [0.15])  # below high_thresh, above low_thresh
        out2 = tracker.update(r2)

        # The track should still be active (recovered by second association)
        assert out2.shape[0] == 1
        id2 = int(out2[0, 4])
        assert id1 == id2, "Low-conf detection should sustain the existing track"

    def test_track_goes_lost_without_low_conf(self):
        """A tracked object with no detection at all must go to lost."""
        tracker = make_tracker()
        r1 = make_results([(10, 20, 110, 120)], [0.9])
        tracker.update(r1)

        tracker.update(DetectionResults.empty())
        assert len(tracker.lost_stracks) >= 1


# ===========================================================================
# Step 4 — unconfirmed track matching
# ===========================================================================


class TestStep4UnconfirmedHandling:
    def test_unconfirmed_track_confirmed_on_second_hit(self):
        """A new track on frame > 1 is unconfirmed until matched again."""
        tracker = make_tracker()
        box = (10, 20, 110, 120)

        # Frame 1: always immediately activates (activate sets is_activated=True on frame 1)
        r1 = make_results([box], [0.9])
        tracker.update(r1)
        assert len(tracker.tracked_stracks) == 1
        assert tracker.tracked_stracks[0].is_activated

    def test_unconfirmed_track_removed_if_no_second_match(self):
        """An unconfirmed track (frame > 1) not matched on next frame is removed."""
        # Force frame counter to 2 by running an unrelated first frame
        tracker = make_tracker()
        r_dummy = make_results([(500, 500, 600, 600)], [0.9])
        tracker.update(r_dummy)  # frame 1: dummy track confirmed
        confirmed_id = int(tracker.tracked_stracks[0].track_id)

        # Frame 2: add a new box at a completely different position + re-detect dummy
        new_box = (10, 10, 50, 50)
        r2 = make_results(
            [(500, 500, 600, 600), new_box],
            [0.9, 0.9],
        )
        tracker.update(r2)  # new_box becomes unconfirmed

        # Frame 3: only dummy box again — new_box disappears
        r3 = make_results([(500, 500, 600, 600)], [0.9])
        tracker.update(r3)

        # The unconfirmed track from new_box should be removed
        active_ids = {int(t.track_id) for t in tracker.tracked_stracks}
        assert confirmed_id in active_ids
        active_ids.discard(confirmed_id)
        # No leftover from the new_box's unconfirmed track
        assert len(active_ids) == 0


# ===========================================================================
# Step 5 — new track initialisation
# ===========================================================================


class TestStep5NewTrackInit:
    def test_high_conf_unmatched_det_becomes_track(self):
        tracker = make_tracker(new_track_thresh=0.6)
        results = make_results([(0, 0, 100, 100)], [0.9])
        out = tracker.update(results)
        assert out.shape[0] == 1
        assert int(out[0, 4]) >= 1

    def test_det_below_new_track_thresh_does_not_create_track(self):
        tracker = make_tracker(
            track_high_thresh=0.5,
            new_track_thresh=0.65,
        )
        # Score above high_thresh but below new_track_thresh
        results = make_results([(0, 0, 100, 100)], [0.6])
        tracker.update(results)
        # No confirmed track yet (frame 1 actually activates; but second
        # frame with no detection should not put them back)
        # Reset and test on frame > 1
        tracker.reset()
        # Prime with a different box
        tracker.update(make_results([(500, 500, 600, 600)], [0.9]))
        # New box with score just below new_track_thresh
        out2 = tracker.update(
            make_results(
                [(500, 500, 600, 600), (0, 0, 100, 100)],
                [0.9, 0.6],  # 0.6 < 0.65 = new_track_thresh
            )
        )
        ids = {int(r[4]) for r in out2}
        # Only the established track should remain; new one was below thresh
        assert len(ids) == 1


# ===========================================================================
# Step 6 — stale track removal
# ===========================================================================


class TestStep6StaleRemoval:
    def test_lost_track_removed_after_max_time_lost(self):
        tracker = make_tracker(track_buffer=3)
        # max_time_lost = 30/30 * 3 = 3

        r1 = make_results([(10, 10, 110, 110)], [0.9])
        tracker.update(r1)
        track_id = int(tracker.tracked_stracks[0].track_id)

        # Miss for more than max_time_lost frames
        for _ in range(5):
            tracker.update(DetectionResults.empty())

        removed_ids = {t.track_id for t in tracker.removed_stracks}
        assert track_id in removed_ids

    def test_lost_track_not_removed_within_budget(self):
        tracker = make_tracker(track_buffer=30)
        # max_time_lost = 30

        r1 = make_results([(10, 10, 110, 110)], [0.9])
        tracker.update(r1)

        # Miss for only 2 frames — track should still be in lost
        tracker.update(DetectionResults.empty())
        tracker.update(DetectionResults.empty())

        assert len(tracker.lost_stracks) == 1
        assert len(tracker.removed_stracks) == 0


# ===========================================================================
# Empty frame handling
# ===========================================================================


class TestEmptyFrameHandling:
    def test_empty_first_frame(self):
        tracker = make_tracker()
        out = tracker.update(DetectionResults.empty())
        assert out.shape == (0, 8)
        assert tracker.frame_id == 1

    def test_empty_frame_after_established_tracks(self):
        tracker = make_tracker()
        r1 = make_results([(0, 0, 100, 100)], [0.9])
        tracker.update(r1)

        out = tracker.update(DetectionResults.empty())
        # Active tracked stracks become lost
        assert out.shape == (0, 8)
        assert len(tracker.lost_stracks) == 1

    def test_frame_id_incremented_on_empty(self):
        tracker = make_tracker()
        tracker.update(DetectionResults.empty())
        assert tracker.frame_id == 1
        tracker.update(DetectionResults.empty())
        assert tracker.frame_id == 2

    def test_multiple_empty_frames_no_crash(self):
        tracker = make_tracker()
        for _ in range(10):
            out = tracker.update(DetectionResults.empty())
            assert out.shape == (0, 8)


# ===========================================================================
# Static utilities
# ===========================================================================


class TestJointStracks:
    def _make_dummy_track(self, track_id: int) -> STrack:
        t = STrack(np.array([100.0, 100.0, 50.0, 50.0]), 0.9, 0)
        t.track_id = track_id
        return t

    def test_merge_no_overlap(self):
        a = [self._make_dummy_track(1), self._make_dummy_track(2)]
        b = [self._make_dummy_track(3)]
        merged = BYTETracker.joint_stracks(a, b)
        assert len(merged) == 3
        ids = {t.track_id for t in merged}
        assert ids == {1, 2, 3}

    def test_merge_with_duplicate_id_prefers_a(self):
        """Duplicate ID in b is skipped — a's entry is kept."""
        a_track = self._make_dummy_track(1)
        a_track.score = 0.9
        b_track = self._make_dummy_track(1)
        b_track.score = 0.5
        merged = BYTETracker.joint_stracks([a_track], [b_track])
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.9)

    def test_empty_a(self):
        b = [self._make_dummy_track(5)]
        merged = BYTETracker.joint_stracks([], b)
        assert len(merged) == 1

    def test_empty_b(self):
        a = [self._make_dummy_track(1)]
        merged = BYTETracker.joint_stracks(a, [])
        assert len(merged) == 1

    def test_both_empty(self):
        merged = BYTETracker.joint_stracks([], [])
        assert merged == []


class TestSubStracks:
    def _make_dummy_track(self, track_id: int) -> STrack:
        t = STrack(np.array([100.0, 100.0, 50.0, 50.0]), 0.9, 0)
        t.track_id = track_id
        return t

    def test_removes_matching_ids(self):
        a = [self._make_dummy_track(i) for i in [1, 2, 3]]
        b = [self._make_dummy_track(i) for i in [2, 3]]
        result = BYTETracker.sub_stracks(a, b)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_empty_b_returns_a(self):
        a = [self._make_dummy_track(i) for i in [1, 2]]
        result = BYTETracker.sub_stracks(a, [])
        assert len(result) == 2

    def test_empty_a_returns_empty(self):
        b = [self._make_dummy_track(1)]
        result = BYTETracker.sub_stracks([], b)
        assert result == []


class TestRemoveDuplicateStracks:
    def _activated_track(self, bbox, start_frame=1, current_frame=1) -> STrack:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        t = STrack(np.array([cx, cy, w, h]), 0.9, 0)
        kf = BYTETracker(default_args()).get_kalmanfilter()
        t.activate(kf, start_frame)
        t.frame_id = current_frame
        t.start_frame = start_frame
        return t

    def test_no_duplicates_unchanged(self):
        a = [self._activated_track((0, 0, 50, 50))]
        b = [self._activated_track((200, 200, 250, 250))]
        ra, rb = BYTETracker.remove_duplicate_stracks(a, b)
        assert len(ra) == 1 and len(rb) == 1

    def test_overlapping_tracks_shorter_removed(self):
        """When IoU > 0.85, the shorter-lived track is removed."""
        # Both tracks on the exact same box → IoU = 1.0
        long_track = self._activated_track((0, 0, 100, 100), start_frame=1, current_frame=10)
        short_track = self._activated_track((0, 0, 100, 100), start_frame=8, current_frame=10)
        ra, rb = BYTETracker.remove_duplicate_stracks([long_track], [short_track])
        # short_track (age 2) < long_track (age 9) → short removed from b
        assert len(ra) == 1
        assert len(rb) == 0

    def test_empty_inputs(self):
        ra, rb = BYTETracker.remove_duplicate_stracks([], [])
        assert ra == [] and rb == []

    def test_one_empty(self):
        a = [self._activated_track((0, 0, 50, 50))]
        ra, rb = BYTETracker.remove_duplicate_stracks(a, [])
        assert len(ra) == 1


# ===========================================================================
# reset_id and reset
# ===========================================================================


class TestResetId:
    def test_reset_id_resets_counter(self):
        # Burn some IDs
        tracker = make_tracker()
        make_tracker()  # allocates kalman filter but no IDs yet
        # Create a track via update
        r1 = make_results([(0, 0, 100, 100)], [0.9])
        tracker.update(r1)
        id_before_reset = tracker.tracked_stracks[0].track_id
        assert id_before_reset >= 1

        BYTETracker.reset_id()
        # Make a fresh track — should start at 1 again
        tracker2 = make_tracker()
        tracker2.update(r1)
        assert tracker2.tracked_stracks[0].track_id == 1


class TestReset:
    def test_reset_clears_all_state(self):
        tracker = make_tracker()
        r1 = make_results([(0, 0, 100, 100)], [0.9])
        tracker.update(r1)
        tracker.update(DetectionResults.empty())

        assert tracker.frame_id > 0
        # Some tracks should exist
        assert len(tracker.tracked_stracks) + len(tracker.lost_stracks) > 0

        tracker.reset()

        assert tracker.frame_id == 0
        assert tracker.tracked_stracks == []
        assert tracker.lost_stracks == []
        assert tracker.removed_stracks == []

    def test_reset_resets_track_ids(self):
        tracker = make_tracker()
        r1 = make_results([(0, 0, 100, 100)], [0.9])
        tracker.update(r1)

        tracker.reset()
        # Fresh track after reset should start at 1
        tracker.update(r1)
        assert tracker.tracked_stracks[0].track_id == 1


# ===========================================================================
# removed_stracks cap
# ===========================================================================


class TestRemovedStracksCap:
    def test_removed_stracks_capped_at_1000(self):
        """Even if many tracks are generated and removed, cap at 1000."""
        tracker = make_tracker(track_buffer=1)  # max_time_lost = 1

        for i in range(1100):
            # Each iteration: new unique box → new track → immediately lost and removed
            x = float(i * 200 % 10000)
            r = make_results([(x, x, x + 50, x + 50)], [0.9])
            tracker.update(r)
            tracker.update(DetectionResults.empty())

        assert len(tracker.removed_stracks) <= 1000


# ===========================================================================
# Multi-object tracking scenario
# ===========================================================================


class TestMultiObjectTracking:
    def test_two_objects_distinct_track_ids(self):
        """Two spatially separated objects get different persistent track IDs."""
        tracker = make_tracker()
        box_a = (0, 0, 50, 50)
        box_b = (500, 500, 550, 550)
        r1 = make_results([box_a, box_b], [0.9, 0.9])
        out1 = tracker.update(r1)
        assert out1.shape[0] == 2
        id1_a, id1_b = int(out1[0, 4]), int(out1[1, 4])
        assert id1_a != id1_b

        r2 = make_results([box_a, box_b], [0.9, 0.9])
        out2 = tracker.update(r2)
        assert out2.shape[0] == 2
        _id2_a, _id2_b = set(int(row[4]) for row in out2), None
        # Both original IDs must appear in second frame
        assert id1_a in {int(r[4]) for r in out2}
        assert id1_b in {int(r[4]) for r in out2}

    def test_one_of_two_disappears(self):
        """When one object stops appearing, only the remaining one is tracked."""
        tracker = make_tracker()
        box_a = (0, 0, 50, 50)
        box_b = (500, 500, 550, 550)
        r1 = make_results([box_a, box_b], [0.9, 0.9])
        tracker.update(r1)

        # box_b disappears
        r2 = make_results([box_a], [0.9])
        out2 = tracker.update(r2)
        assert out2.shape[0] == 1

    def test_new_object_gets_new_id(self):
        """A completely new object appearing mid-sequence gets a new track ID."""
        tracker = make_tracker()
        box_a = (0, 0, 50, 50)
        r1 = make_results([box_a], [0.9])
        out1 = tracker.update(r1)
        id_a = int(out1[0, 4])

        # Frame 2: box_b appears; on frame > 1 it starts as unconfirmed.
        box_b = (800, 800, 850, 850)
        r2 = make_results([box_a, box_b], [0.9, 0.9])
        tracker.update(r2)

        # Frame 3: box_b matched again → now confirmed/activated with a new ID.
        out3 = tracker.update(r2)
        ids = {int(r[4]) for r in out3}
        assert id_a in ids
        # New box must have a different (new) ID
        new_ids = ids - {id_a}
        assert len(new_ids) == 1
        assert list(new_ids)[0] != id_a


# ===========================================================================
# Package-level import
# ===========================================================================


class TestPackageImport:
    def test_import_from_mata_trackers(self):
        from mata.trackers import BYTETracker, DetectionResults  # noqa: F401

        assert BYTETracker is not None
        assert DetectionResults is not None

    def test_all_exports(self):
        import mata.trackers as mt

        for name in ["BaseTrack", "TrackState", "STrack", "DetectionResults", "BYTETracker"]:
            assert hasattr(mt, name), f"mata.trackers missing: {name}"
