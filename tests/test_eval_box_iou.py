"""Unit tests for box IoU computation (Task A2).

Tests cover:
- box_iou: identical, non-overlapping, nested, partial-overlap, edge cases
- box_iou_batch: shape, dtype, threshold semantics
- COCO_IOU_THRESHOLDS: constant integrity
"""

from __future__ import annotations

import numpy as np

from mata.eval.metrics.iou import (
    COCO_IOU_THRESHOLDS,
    box_iou,
    box_iou_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


# ---------------------------------------------------------------------------
# COCO_IOU_THRESHOLDS
# ---------------------------------------------------------------------------


class TestCocoIouThresholds:
    def test_length(self) -> None:
        assert len(COCO_IOU_THRESHOLDS) == 10

    def test_values(self) -> None:
        expected = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        assert COCO_IOU_THRESHOLDS == expected

    def test_first_and_last(self) -> None:
        assert COCO_IOU_THRESHOLDS[0] == 0.50
        assert COCO_IOU_THRESHOLDS[-1] == 0.95

    def test_step(self) -> None:
        for i in range(1, len(COCO_IOU_THRESHOLDS)):
            diff = round(COCO_IOU_THRESHOLDS[i] - COCO_IOU_THRESHOLDS[i - 1], 10)
            assert abs(diff - 0.05) < 1e-9, f"step at index {i}: {diff}"


# ---------------------------------------------------------------------------
# box_iou — output properties
# ---------------------------------------------------------------------------


class TestBoxIouOutputProperties:
    def test_output_dtype_float32(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        result = box_iou(boxes, boxes)
        assert result.dtype == np.float32

    def test_output_shape_n_m(self) -> None:
        a = np.random.rand(5, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        result = box_iou(a, b)
        assert result.shape == (5, 3)

    def test_output_shape_square(self) -> None:
        a = np.random.rand(4, 4).astype(np.float32)
        result = box_iou(a, a)
        assert result.shape == (4, 4)

    def test_values_in_range_zero_one(self) -> None:
        a = np.array([[0, 0, 50, 50], [10, 10, 40, 40]], dtype=np.float32)
        b = np.array([[5, 5, 45, 45], [20, 20, 60, 60]], dtype=np.float32)
        result = box_iou(a, b)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# box_iou — correctness
# ---------------------------------------------------------------------------


class TestBoxIouCorrectness:
    def test_identical_boxes_iou_one(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32)
        iou = box_iou(boxes, boxes)
        np.testing.assert_allclose(np.diag(iou), 1.0, atol=1e-6)

    def test_identical_single_box(self) -> None:
        box = _box(10, 20, 30, 40)
        iou = box_iou(box, box)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_non_overlapping_iou_zero(self) -> None:
        a = _box(0, 0, 5, 5)
        b = _box(10, 10, 20, 20)
        iou = box_iou(a, b)
        assert iou[0, 0] == 0.0

    def test_adjacent_boxes_no_overlap(self) -> None:
        # Touching at edge — no area overlap
        a = _box(0, 0, 5, 5)
        b = _box(5, 0, 10, 5)
        iou = box_iou(a, b)
        assert iou[0, 0] == 0.0

    def test_fully_nested_iou_equals_ratio(self) -> None:
        outer = _box(0, 0, 10, 10)  # area = 100
        inner = _box(2, 2, 6, 6)  # area = 16
        iou = box_iou(inner, outer)
        expected = 16.0 / 100.0
        assert abs(iou[0, 0] - expected) < 1e-5

    def test_outer_nested_in_inner(self) -> None:
        # Swap order — result must be the same
        outer = _box(0, 0, 10, 10)
        inner = _box(2, 2, 6, 6)
        iou_io = box_iou(inner, outer)
        iou_oi = box_iou(outer, inner)
        np.testing.assert_allclose(iou_io, iou_oi.T, atol=1e-6)

    def test_half_overlap(self) -> None:
        a = _box(0, 0, 10, 10)  # area=100
        b = _box(5, 0, 15, 10)  # area=100, inter=50
        iou = box_iou(a, b)
        expected = 50.0 / 150.0
        assert abs(iou[0, 0] - expected) < 1e-5

    def test_quarter_overlap(self) -> None:
        a = _box(0, 0, 10, 10)  # area=100
        b = _box(5, 5, 15, 15)  # area=100, inter=25
        iou = box_iou(a, b)
        expected = 25.0 / 175.0
        assert abs(iou[0, 0] - expected) < 1e-5

    def test_symmetry(self) -> None:
        a = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32)
        b = np.array([[2, 2, 8, 8], [15, 15, 25, 25]], dtype=np.float32)
        iou_ab = box_iou(a, b)
        iou_ba = box_iou(b, a)
        np.testing.assert_allclose(iou_ab, iou_ba.T, atol=1e-6)

    def test_single_box_vs_multiple(self) -> None:
        single = _box(0, 0, 10, 10)
        multi = np.array([[0, 0, 10, 10], [0, 0, 5, 5], [20, 20, 30, 30]], dtype=np.float32)
        iou = box_iou(single, multi)
        assert iou.shape == (1, 3)
        assert abs(iou[0, 0] - 1.0) < 1e-6  # identical
        assert abs(iou[0, 1] - 0.25) < 1e-5  # 25/100
        assert iou[0, 2] == 0.0  # non-overlapping

    def test_float64_input_coerced(self) -> None:
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float64)
        result = box_iou(boxes, boxes)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# box_iou — edge cases
# ---------------------------------------------------------------------------


class TestBoxIouEdgeCases:
    def test_empty_boxes1_zero_rows(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        other = np.array([[0, 0, 5, 5], [6, 6, 10, 10]], dtype=np.float32)
        result = box_iou(empty, other)
        assert result.shape == (0, 2)

    def test_empty_boxes2_zero_rows(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        other = np.array([[0, 0, 5, 5]], dtype=np.float32)
        result = box_iou(other, empty)
        assert result.shape == (1, 0)

    def test_both_empty(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        result = box_iou(empty, empty)
        assert result.shape == (0, 0)

    def test_zero_area_box_iou_zero(self) -> None:
        # Degenerate box (x1==x2, y1==y2)
        degenerate = _box(5, 5, 5, 5)
        normal = _box(4, 4, 6, 6)
        iou = box_iou(degenerate, normal)
        assert iou[0, 0] == 0.0

    def test_zero_area_both_same_point(self) -> None:
        pt = _box(5, 5, 5, 5)
        iou = box_iou(pt, pt)
        assert iou[0, 0] == 0.0

    def test_inverted_coords_treated_as_zero_area(self) -> None:
        # x2 < x1 → negative width → area = 0
        inv = _box(10, 0, 5, 10)  # "width" = 5-10 = -5 → clamped to 0 in intersection
        normal = _box(0, 0, 8, 8)
        iou = box_iou(inv, normal)
        assert iou[0, 0] == 0.0

    def test_integer_input(self) -> None:
        boxes = np.array([[0, 0, 10, 10]], dtype=np.int32)
        result = box_iou(boxes, boxes)
        assert result.dtype == np.float32
        assert abs(result[0, 0] - 1.0) < 1e-6

    def test_1d_input_broadcast(self) -> None:
        # Single box as 1-D array
        box1d = np.array([0, 0, 10, 10], dtype=np.float32)
        boxes2 = _box(0, 0, 10, 10)
        result = box_iou(box1d, boxes2)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# box_iou_batch
# ---------------------------------------------------------------------------


class TestBoxIouBatch:
    def test_output_shape(self) -> None:
        pred = np.random.rand(4, 4).astype(np.float32)
        gt = np.random.rand(3, 4).astype(np.float32)
        thresholds = [0.25, 0.50, 0.75]
        result = box_iou_batch(pred, gt, thresholds)
        assert result.shape == (3, 4, 3)

    def test_output_dtype_bool(self) -> None:
        pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        result = box_iou_batch(pred, gt, [0.5])
        assert result.dtype == bool

    def test_identical_boxes_true_at_all_thresholds(self) -> None:
        pred = _box(0, 0, 10, 10)
        gt = _box(0, 0, 10, 10)
        result = box_iou_batch(pred, gt, COCO_IOU_THRESHOLDS)
        # IoU = 1.0 → True at every threshold
        assert result.all()

    def test_non_overlapping_false_at_all_thresholds(self) -> None:
        pred = _box(0, 0, 5, 5)
        gt = _box(10, 10, 20, 20)
        result = box_iou_batch(pred, gt, COCO_IOU_THRESHOLDS)
        assert not result.any()

    def test_threshold_boundary(self) -> None:
        # IoU = 0.50 exactly
        a = _box(0, 0, 10, 10)  # area=100
        b = _box(5, 0, 15, 10)  # area=100, inter=50, union=150 → IoU≈0.333
        result = box_iou_batch(a, b, [0.3, 0.34, 0.5])
        assert result[0, 0, 0]  # 0.333 ≥ 0.3
        assert not result[1, 0, 0]  # 0.333 < 0.34
        assert not result[2, 0, 0]  # 0.333 < 0.5

    def test_single_threshold(self) -> None:
        pred = _box(0, 0, 10, 10)
        gt = _box(0, 0, 10, 10)
        result = box_iou_batch(pred, gt, [0.5])
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0]

    def test_empty_predictions(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        gt = _box(0, 0, 10, 10)
        result = box_iou_batch(empty, gt, [0.5])
        assert result.shape == (1, 0, 1)

    def test_coco_thresholds_length(self) -> None:
        pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        result = box_iou_batch(pred, gt, COCO_IOU_THRESHOLDS)
        assert result.shape == (10, 2, 1)

    def test_stricter_threshold_subset_of_lenient(self) -> None:
        """Anything True at 0.9 must also be True at 0.5."""
        pred = np.random.rand(6, 4).astype(np.float32) * 20
        gt = np.random.rand(4, 4).astype(np.float32) * 20
        result = box_iou_batch(pred, gt, [0.5, 0.9])
        # Wherever strict (0.9) is True, lenient (0.5) must also be True
        assert np.all(result[0] | ~result[1])
