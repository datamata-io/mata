"""Unit tests for mata.trackers.utils.matching — Task A2.

Covers:
- iou_batch(): vectorized IoU between box sets
- iou_distance(): 1-IoU cost matrix from track objects
- linear_assignment(): Hungarian + greedy fallback with threshold gating
- fuse_score(): confidence-score weighting of IoU costs
- embedding_distance(): stub returning np.inf + real cosine path
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from mata.trackers.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_batch,
    iou_distance,
    linear_assignment,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTrack:
    """Minimal duck-type track object used by iou_distance tests."""

    xyxy: np.ndarray
    score: float = 0.9
    smooth_feat: np.ndarray | None = None


@dataclass
class FakeTrackTLWH:
    """Track object with only tlwh (no xyxy) to test fallback path."""

    tlwh: np.ndarray
    score: float = 0.9


@dataclass
class FakeDet:
    """Detection stub with score + optional feature."""

    xyxy: np.ndarray
    score: float = 0.9
    curr_feat: np.ndarray | None = None


def _box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ===========================================================================
# iou_batch
# ===========================================================================


class TestIouBatch:
    def test_identical_boxes_iou_one(self):
        b = _box(0, 0, 10, 10)
        result = iou_batch(b[np.newaxis], b[np.newaxis])
        assert result.shape == (1, 1)
        assert float(result[0, 0]) == pytest.approx(1.0)

    def test_non_overlapping_iou_zero(self):
        b1 = _box(0, 0, 10, 10)
        b2 = _box(20, 20, 30, 30)
        result = iou_batch(b1[np.newaxis], b2[np.newaxis])
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = _box(0, 0, 10, 10)  # area=100
        b2 = _box(5, 5, 15, 15)  # area=100, intersection 5x5=25
        result = iou_batch(b1[np.newaxis], b2[np.newaxis])
        # union = 100 + 100 - 25 = 175
        expected = 25.0 / 175.0
        assert float(result[0, 0]) == pytest.approx(expected, abs=1e-5)

    def test_one_contained_in_other(self):
        outer = _box(0, 0, 10, 10)  # area=100
        inner = _box(2, 2, 8, 8)  # area=36, fully inside outer
        result = iou_batch(outer[np.newaxis], inner[np.newaxis])
        # intersection=36, union=100
        expected = 36.0 / 100.0
        assert float(result[0, 0]) == pytest.approx(expected, abs=1e-5)

    def test_multi_box_shape(self):
        boxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)
        result = iou_batch(boxes1, boxes2)
        assert result.shape == (2, 3)
        # box0 vs box0 → 1.0
        assert result[0, 0] == pytest.approx(1.0)
        # box1 vs box2 → 1.0
        assert result[1, 2] == pytest.approx(1.0)
        # box0 vs box2 → 0.0 (no overlap)
        assert result[0, 2] == pytest.approx(0.0)

    def test_empty_first_set(self):
        b = _box(0, 0, 10, 10)
        result = iou_batch(np.empty((0, 4), dtype=np.float32), b[np.newaxis])
        assert result.shape == (0, 1)

    def test_empty_second_set(self):
        b = _box(0, 0, 10, 10)
        result = iou_batch(b[np.newaxis], np.empty((0, 4), dtype=np.float32))
        assert result.shape == (1, 0)

    def test_both_empty(self):
        result = iou_batch(np.empty((0, 4)), np.empty((0, 4)))
        assert result.shape == (0, 0)

    def test_zero_area_box(self):
        degenerate = _box(5, 5, 5, 5)  # zero area
        normal = _box(0, 0, 10, 10)
        result = iou_batch(degenerate[np.newaxis], normal[np.newaxis])
        # union = 100, inter = 0
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_output_dtype(self):
        b = _box(0, 0, 10, 10)
        result = iou_batch(b[np.newaxis], b[np.newaxis])
        assert result.dtype == np.float32

    def test_vectorised_no_loops(self):
        """Smoke test: large batch must complete quickly (no O(N²) Python loops)."""
        rng = np.random.default_rng(42)
        boxes1 = rng.uniform(0, 100, size=(200, 4)).astype(np.float32)
        # Ensure x2>x1, y2>y1
        boxes1[:, 2] = boxes1[:, 0] + rng.uniform(1, 20, 200).astype(np.float32)
        boxes1[:, 3] = boxes1[:, 1] + rng.uniform(1, 20, 200).astype(np.float32)
        boxes2 = boxes1.copy()
        result = iou_batch(boxes1, boxes2)
        assert result.shape == (200, 200)
        # Diagonal should be all 1.0 (identical boxes)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-5)


# ===========================================================================
# iou_distance
# ===========================================================================


class TestIouDistance:
    def test_identical_tracks_cost_zero(self):
        t1 = FakeTrack(xyxy=_box(0, 0, 10, 10))
        result = iou_distance([t1], [t1])
        assert result.shape == (1, 1)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_no_overlap_cost_one(self):
        t1 = FakeTrack(xyxy=_box(0, 0, 10, 10))
        t2 = FakeTrack(xyxy=_box(50, 50, 60, 60))
        result = iou_distance([t1], [t2])
        assert float(result[0, 0]) == pytest.approx(1.0)

    def test_partial_overlap_cost(self):
        t1 = FakeTrack(xyxy=_box(0, 0, 10, 10))
        t2 = FakeTrack(xyxy=_box(5, 5, 15, 15))
        result = iou_distance([t1], [t2])
        expected_iou = 25.0 / 175.0
        assert float(result[0, 0]) == pytest.approx(1.0 - expected_iou, abs=1e-5)

    def test_shape_nxm(self):
        tracks = [FakeTrack(xyxy=_box(0, 0, 10, 10)), FakeTrack(xyxy=_box(20, 20, 30, 30))]
        dets = [FakeTrack(xyxy=_box(0, 0, 10, 10))]
        result = iou_distance(tracks, dets)
        assert result.shape == (2, 1)

    def test_tlwh_fallback(self):
        """Objects with only .tlwh (no .xyxy) must be handled correctly."""
        # tlwh = [x, y, w, h] → xyxy = [x, y, x+w, y+h]
        t_tlwh = FakeTrackTLWH(tlwh=np.array([0.0, 0.0, 10.0, 10.0]))
        t_xyxy = FakeTrack(xyxy=_box(0, 0, 10, 10))
        result = iou_distance([t_tlwh], [t_xyxy])
        # Same box → cost ≈ 0
        assert float(result[0, 0]) == pytest.approx(0.0, abs=1e-5)

    def test_empty_atracks(self):
        dets = [FakeTrack(xyxy=_box(0, 0, 10, 10))]
        result = iou_distance([], dets)
        assert result.shape == (0, 1)

    def test_empty_btracks(self):
        tracks = [FakeTrack(xyxy=_box(0, 0, 10, 10))]
        result = iou_distance(tracks, [])
        assert result.shape == (1, 0)

    def test_both_empty(self):
        result = iou_distance([], [])
        assert result.shape == (0, 0)

    def test_cost_bounds(self):
        """All costs must be in [0, 1]."""
        rng = np.random.default_rng(7)
        tracks = [FakeTrack(xyxy=rng.uniform(0, 100, 4).astype(np.float32)) for _ in range(5)]
        dets = [FakeTrack(xyxy=rng.uniform(0, 100, 4).astype(np.float32)) for _ in range(5)]
        # Ensure valid boxes
        for obj in tracks + dets:
            obj.xyxy[2] = obj.xyxy[0] + abs(obj.xyxy[2]) + 1
            obj.xyxy[3] = obj.xyxy[1] + abs(obj.xyxy[3]) + 1
        result = iou_distance(tracks, dets)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


# ===========================================================================
# linear_assignment
# ===========================================================================


class TestLinearAssignment:
    def _check_valid(self, matches, unmatched_a, unmatched_b, total_a, total_b):
        """Helper: verify partition covers all indices exactly once."""
        matched_a = set(matches[:, 0].tolist()) if len(matches) else set()
        matched_b = set(matches[:, 1].tolist()) if len(matches) else set()
        all_a = matched_a | set(unmatched_a.tolist())
        all_b = matched_b | set(unmatched_b.tolist())
        assert all_a == set(range(total_a))
        assert all_b == set(range(total_b))
        # No duplicates
        assert len(matched_a) == len(matches)
        assert len(matched_b) == len(matches)

    def test_perfect_matching_identity(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.shape[1] == 2
        # Both should match
        self._check_valid(matches, ua, ub, 2, 2)
        assert len(matches) == 2
        assert len(ua) == 0
        assert len(ub) == 0

    def test_threshold_filters_high_cost(self):
        cost = np.array([[0.9]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert list(ua) == [0]
        assert list(ub) == [0]

    def test_all_above_threshold_all_unmatched(self):
        cost = np.ones((3, 3), dtype=np.float32) * 0.9
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert set(ua.tolist()) == {0, 1, 2}
        assert set(ub.tolist()) == {0, 1, 2}

    def test_empty_cost_matrix(self):
        cost = np.empty((0, 0), dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert matches.shape == (0, 2)
        assert len(ua) == 0
        assert len(ub) == 0

    def test_empty_rows(self):
        cost = np.empty((0, 3), dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert len(ua) == 0
        assert set(ub.tolist()) == {0, 1, 2}

    def test_empty_cols(self):
        cost = np.empty((3, 0), dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert set(ua.tolist()) == {0, 1, 2}
        assert len(ub) == 0

    def test_n_greater_m(self):
        """More tracks than detections — some tracks unmatched."""
        cost = np.array([[0.1], [0.2], [0.8]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        self._check_valid(matches, ua, ub, 3, 1)
        assert len(matches) == 1

    def test_m_greater_n(self):
        """More detections than tracks — some detections unmatched."""
        cost = np.array([[0.1, 0.2, 0.8]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        self._check_valid(matches, ua, ub, 1, 3)
        assert len(matches) == 1

    def test_single_cell_match(self):
        cost = np.array([[0.3]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 1
        assert matches[0, 0] == 0 and matches[0, 1] == 0

    def test_partition_completeness(self):
        """All indices must appear in exactly one of matches / unmatched."""
        rng = np.random.default_rng(99)
        cost = rng.uniform(0, 1, size=(5, 4)).astype(np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.6)
        self._check_valid(matches, ua, ub, 5, 4)

    def test_greedy_fallback(self):
        """Greedy path must produce valid results when scipy is unavailable."""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        with patch.dict("sys.modules", {"scipy": None, "scipy.optimize": None}):
            # Re-import matching module so scipy import fails inside function
            # We patch at the function level via importlib trick
            matches, ua, ub = _linear_assignment_greedy_only(cost, thresh=0.5)
        assert len(matches) == 2
        self._check_valid(matches, ua, ub, 2, 2)

    def test_output_dtype_int(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.2]], dtype=np.float32)
        matches, ua, ub = linear_assignment(cost, thresh=0.5)
        if len(matches):
            assert matches.dtype in (np.int32, np.int64, int, np.intp)


def _linear_assignment_greedy_only(cost_matrix: np.ndarray, thresh: float):
    """Execute only the greedy path (scipy bypassed)."""
    matches_list: list[list[int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    flat_idx = np.argsort(cost_matrix, axis=None)
    rows, cols = np.unravel_index(flat_idx, cost_matrix.shape)

    for r, c in zip(rows.tolist(), cols.tolist()):
        if cost_matrix[r, c] > thresh:
            break
        if r in matched_rows or c in matched_cols:
            continue
        matches_list.append([r, c])
        matched_rows.add(r)
        matched_cols.add(c)

    matches = np.array(matches_list, dtype=int).reshape(-1, 2)
    unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_rows], dtype=int)
    unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_cols], dtype=int)
    return matches, unmatched_a, unmatched_b


# ===========================================================================
# fuse_score
# ===========================================================================


class TestFuseScore:
    def _det(self, score: float) -> FakeDet:
        return FakeDet(xyxy=_box(0, 0, 10, 10), score=score)

    def test_formula_single_cell(self):
        """fuse_cost = 1 - (1 - cost) * score."""
        cost = np.array([[0.4]], dtype=np.float32)
        dets = [self._det(0.8)]
        result = fuse_score(cost, dets)
        expected = 1.0 - (1.0 - 0.4) * 0.8
        assert float(result[0, 0]) == pytest.approx(expected, abs=1e-5)

    def test_score_one_leaves_cost_unchanged_for_perfect_iou(self):
        """When score=1 and cost=0: fuse = 1-(1-0)*1 = 0."""
        cost = np.array([[0.0]], dtype=np.float32)
        dets = [self._det(1.0)]
        result = fuse_score(cost, dets)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_score_zero_raises_cost_to_one(self):
        """When score=0: fuse = 1-(1-cost)*0 = 1 regardless of cost."""
        cost = np.array([[0.2]], dtype=np.float32)
        dets = [self._det(0.0)]
        result = fuse_score(cost, dets)
        assert float(result[0, 0]) == pytest.approx(1.0, abs=1e-5)

    def test_multiple_detections_broadcast(self):
        """Score vector must broadcast correctly over N track rows."""
        cost = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32)
        dets = [self._det(0.9), self._det(0.5)]
        result = fuse_score(cost, dets)
        assert result.shape == (2, 2)
        # col 0: score=0.9
        assert result[0, 0] == pytest.approx(1 - (1 - 0.2) * 0.9, abs=1e-5)
        assert result[1, 0] == pytest.approx(1 - (1 - 0.4) * 0.9, abs=1e-5)
        # col 1: score=0.5
        assert result[0, 1] == pytest.approx(1 - (1 - 0.6) * 0.5, abs=1e-5)

    def test_empty_cost_matrix_passthrough(self):
        cost = np.empty((0, 0), dtype=np.float32)
        result = fuse_score(cost, [])
        assert result.shape == (0, 0)

    def test_output_dtype(self):
        cost = np.array([[0.5]], dtype=np.float32)
        dets = [self._det(0.7)]
        result = fuse_score(cost, dets)
        assert result.dtype == np.float32


# ===========================================================================
# embedding_distance
# ===========================================================================


class TestEmbeddingDistance:
    def _track(self, feat=None):
        t = FakeTrack(xyxy=_box(0, 0, 10, 10))
        t.smooth_feat = feat
        return t

    def _det(self, feat=None):
        d = FakeDet(xyxy=_box(0, 0, 10, 10))
        d.curr_feat = feat
        return d

    def test_stub_returns_inf_no_features(self):
        tracks = [self._track(None), self._track(None)]
        dets = [self._det(None), self._det(None), self._det(None)]
        result = embedding_distance(tracks, dets)
        assert result.shape == (2, 3)
        assert np.all(np.isinf(result))

    def test_empty_tracks(self):
        dets = [self._det()]
        result = embedding_distance([], dets)
        assert result.shape == (0, 1)

    def test_empty_detections(self):
        tracks = [self._track()]
        result = embedding_distance(tracks, [])
        assert result.shape == (1, 0)

    def test_both_empty(self):
        result = embedding_distance([], [])
        assert result.shape == (0, 0)

    def test_real_cosine_identical_features(self):
        """When features are present and identical, cosine distance = 0."""
        feat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tracks = [self._track(feat)]
        dets = [self._det(feat)]
        result = embedding_distance(tracks, dets)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) == pytest.approx(0.0, abs=1e-5)

    def test_real_cosine_orthogonal_features(self):
        """Orthogonal feature vectors → cosine distance = 1."""
        feat_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feat_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tracks = [self._track(feat_a)]
        dets = [self._det(feat_b)]
        result = embedding_distance(tracks, dets)
        assert float(result[0, 0]) == pytest.approx(1.0, abs=1e-5)

    def test_real_cosine_opposite_features(self):
        """Opposite unit vectors → cosine distance = 2."""
        feat_a = np.array([1.0, 0.0], dtype=np.float32)
        feat_b = np.array([-1.0, 0.0], dtype=np.float32)
        tracks = [self._track(feat_a)]
        dets = [self._det(feat_b)]
        result = embedding_distance(tracks, dets)
        assert float(result[0, 0]) == pytest.approx(2.0, abs=1e-5)

    def test_mixed_partial_features_returns_inf(self):
        """If any feature is None, fall back to all-inf stub."""
        tracks = [self._track(np.array([1.0, 0.0])), self._track(None)]
        dets = [self._det(np.array([1.0, 0.0]))]
        result = embedding_distance(tracks, dets)
        # Mixed → stub path
        assert np.all(np.isinf(result))

    def test_output_dtype(self):
        result = embedding_distance([], [])
        assert result.dtype == np.float32
