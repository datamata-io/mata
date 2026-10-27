"""Cache-invalidation tests for track bounding-box views."""

import numpy as np

from mata.trackers.bot_sort import BOTrack
from mata.trackers.byte_tracker import STrack
from mata.trackers.utils.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH


def _active_strack(x=100.0, y=100.0, w=50.0, h=80.0) -> STrack:
    track = STrack([x, y, w, h, 0.0, 0.0], 0.9, 0)
    track.activate(KalmanFilterXYAH(), frame_id=1)
    return track


def test_tlwh_is_cached_until_state_changes():
    track = _active_strack()
    first = track.tlwh
    cached = track._tlwh_cache
    assert cached is not None
    second = track.tlwh
    assert track._tlwh_cache is cached
    assert np.array_equal(first, second)
    assert first is not second  # callers get an independent copy


def test_predict_invalidates_the_box_cache():
    track = _active_strack()
    track.mean[4] = 10.0  # vcx
    before = track.tlwh
    track.predict()
    after = track.tlwh
    assert after[0] > before[0]
    assert np.allclose(after, track._tlwh_cache)


def test_update_invalidates_and_keeps_views_consistent():
    track = _active_strack()
    before = track.xyxy
    new_track = STrack([120.0, 120.0, 40.0, 40.0, 0.0, 0.0], 0.9, 0)
    track.update(new_track, frame_id=2)
    after = track.xyxy
    assert not np.array_equal(before, after)
    assert np.allclose(track.xywh[:2], (after[:2] + after[2:]) / 2.0)
    assert np.allclose(track.xywh[2:], after[2:] - after[:2])


def test_multi_gmc_invalidates_the_box_cache():
    tracks = [_active_strack() for _ in range(3)]
    before = [track.tlwh for track in tracks]
    STrack.multi_gmc(tracks, np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]]))
    for old, track in zip(before, tracks):
        assert np.allclose(track.tlwh, old + [10.0, 5.0, 0.0, 0.0], atol=1e-3)


def test_botrack_predict_invalidates_the_box_cache():
    track = BOTrack([100.0, 100.0, 50.0, 80.0, 0.0, 0.0], 0.9, 0)
    track.activate(KalmanFilterXYWH(), frame_id=1)
    track.mean[4] = 10.0  # vcx
    before = track.tlwh
    track.predict()
    assert track.tlwh[0] > before[0]
    assert np.allclose(track.xyxy[:2], track.tlwh[:2])


def test_box_views_are_independent_copies():
    track = _active_strack()
    view = track.tlwh
    view[0] = 9999.0
    assert track.tlwh[0] != 9999.0
