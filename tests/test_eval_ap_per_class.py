"""Unit tests for ap_per_class() and Metric container (Tasks B1 & B2).

Tests cover:
- ap_per_class: perfect predictions, all-FP, degenerate inputs, curve shapes
- Metric: pre-update defaults, post-update properties, fitness formula
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.iou import COCO_IOU_THRESHOLDS

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_scenario(
    n_cls: int = 2,
    n_per_cls: int = 20,
    tp_value: bool = True,
    n_thr: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (tp, conf, pred_cls, target_cls) for a uniform scenario."""
    pred_cls = np.repeat(np.arange(n_cls), n_per_cls).astype(np.int32)
    target_cls = pred_cls.copy()
    conf = np.linspace(0.99, 0.51, n_cls * n_per_cls, dtype=np.float32)
    if n_thr == 1:
        tp = np.full(len(pred_cls), tp_value, dtype=bool)
    else:
        tp = np.full((len(pred_cls), n_thr), tp_value, dtype=bool)
    return tp, conf, pred_cls, target_cls


@pytest.fixture()
def perfect_results():
    tp, conf, pc, tc = _make_scenario(n_cls=2, n_per_cls=20, tp_value=True)
    return ap_per_class(tp, conf, pc, tc)


@pytest.fixture()
def populated_metric(perfect_results):
    m = Metric()
    m.update(perfect_results)
    return m


# ---------------------------------------------------------------------------
# ap_per_class — return-tuple structure
# ---------------------------------------------------------------------------


class TestApPerClassReturnStructure:
    def test_returns_12_element_tuple(self):
        tp, conf, pc, tc = _make_scenario()
        result = ap_per_class(tp, conf, pc, tc)
        assert isinstance(result, tuple)
        assert len(result) == 12

    def test_tp_at_max_f1_shape(self, perfect_results):
        tp_mf1 = perfect_results[0]
        assert tp_mf1.shape == (2,)
        assert tp_mf1.dtype in (np.int32, np.int64)

    def test_fp_at_max_f1_shape(self, perfect_results):
        fp_mf1 = perfect_results[1]
        assert fp_mf1.shape == (2,)

    def test_p_r_f1_shape(self, perfect_results):
        _, _, p, r, f1, *_ = perfect_results
        for arr in (p, r, f1):
            assert arr.shape == (2,)
            assert arr.dtype == np.float32

    def test_all_ap_shape_single_threshold(self):
        tp, conf, pc, tc = _make_scenario(n_cls=3)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=[0.5])
        all_ap = result[5]
        assert all_ap.shape == (3, 1)
        assert all_ap.dtype == np.float32

    def test_all_ap_shape_10_thresholds(self, perfect_results):
        all_ap = perfect_results[5]
        assert all_ap.shape == (2, 10)

    def test_unique_classes(self, perfect_results):
        unique_cls = perfect_results[6]
        np.testing.assert_array_equal(unique_cls, [0, 1])

    def test_curve_shapes(self, perfect_results):
        _, _, _, _, _, _, _, p_curve, r_curve, f1_curve, x, prec_values = perfect_results
        assert p_curve.shape == (2, 1000)
        assert r_curve.shape == (2, 1000)
        assert f1_curve.shape == (2, 1000)
        assert prec_values.shape == (2, 1000)

    def test_x_axis(self, perfect_results):
        x = perfect_results[10]
        assert len(x) == 1000
        assert x.dtype == np.float32
        assert abs(float(x[0])) < 1e-6
        assert abs(float(x[-1]) - 1.0) < 2e-3


# ---------------------------------------------------------------------------
# ap_per_class — correctness
# ---------------------------------------------------------------------------


class TestApPerClassCorrectness:
    def test_perfect_predictions_ap_near_one(self, perfect_results):
        all_ap = perfect_results[5]
        assert np.all(all_ap > 0.95), f"min ap={all_ap.min()}"

    def test_all_fp_ap_zero(self):
        tp, conf, pc, tc = _make_scenario(tp_value=False)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=[0.5])
        all_ap = result[5]
        assert np.all(all_ap == 0.0)

    def test_perfect_precision_recall_one(self, perfect_results):
        _, _, p, r, *_ = perfect_results
        assert np.all(p > 0.9)
        assert np.all(r > 0.9)

    def test_all_fp_tp_count_zero(self):
        tp, conf, pc, tc = _make_scenario(tp_value=False)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=[0.5])
        tp_mf1 = result[0]
        assert np.all(tp_mf1 == 0)

    def test_single_class(self):
        pred_cls = np.zeros(15, dtype=np.int32)
        target_cls = np.zeros(15, dtype=np.int32)
        conf = np.linspace(0.9, 0.5, 15, dtype=np.float32)
        tp = np.ones(15, dtype=bool)
        result = ap_per_class(tp, conf, pred_cls, target_cls, iou_thresholds=[0.5])
        all_ap = result[5]
        assert all_ap.shape == (1, 1)
        assert all_ap[0, 0] > 0.95

    def test_three_classes_independent(self):
        tp, conf, pc, tc = _make_scenario(n_cls=3, n_per_cls=15)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=[0.5])
        all_ap = result[5]
        assert all_ap.shape == (3, 1)
        assert np.all(all_ap > 0.9)

    def test_mixed_tp_fp_ap_between_zero_one(self):
        rng = np.random.default_rng(42)
        n = 30
        pred_cls = np.zeros(n, dtype=np.int32)
        target_cls = np.zeros(n, dtype=np.int32)
        conf = rng.random(n).astype(np.float32)
        tp = rng.random(n) > 0.4  # ~60 % TP
        result = ap_per_class(tp, conf, pred_cls, target_cls, iou_thresholds=[0.5])
        all_ap = result[5]
        assert 0.0 <= float(all_ap[0, 0]) <= 1.0

    def test_precision_in_range_zero_one(self, perfect_results):
        _, _, p, r, f1, *_ = perfect_results
        assert np.all((p >= 0) & (p <= 1))
        assert np.all((r >= 0) & (r <= 1))
        assert np.all((f1 >= 0) & (f1 <= 1))

    def test_f1_consistent_with_p_r(self, perfect_results):
        _, _, p, r, f1, *_ = perfect_results
        eps = 1e-16
        expected_f1 = 2 * p * r / (p + r + eps)
        # f1 at max-F1 operating point should be consistent
        np.testing.assert_allclose(f1, expected_f1, atol=1e-4)

    def test_coco_10_thresholds(self):
        tp, conf, pc, tc = _make_scenario(n_cls=2)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=COCO_IOU_THRESHOLDS)
        all_ap = result[5]
        assert all_ap.shape == (2, 10)
        # Perfect predictions → AP should be high at all thresholds
        assert np.all(all_ap > 0.9)

    def test_ap_decreases_with_stricter_threshold(self):
        """AP at IoU=0.95 ≤ AP at IoU=0.50 for imperfect predictions."""
        rng = np.random.default_rng(0)
        n = 40
        pred_cls = np.zeros(n, dtype=np.int32)
        target_cls = np.zeros(n, dtype=np.int32)
        conf = rng.random(n).astype(np.float32)
        tp_50 = rng.random(n) > 0.3  # 70% TP at 0.50
        tp_95 = tp_50 & (rng.random(n) > 0.6)  # stricter: fewer TP
        tp_2d = np.stack([tp_50, tp_95], axis=1)
        result = ap_per_class(tp_2d, conf, pred_cls, target_cls, iou_thresholds=[0.50, 0.95])
        all_ap = result[5]
        assert float(all_ap[0, 0]) >= float(all_ap[0, 1])


# ---------------------------------------------------------------------------
# ap_per_class — degenerate / edge cases
# ---------------------------------------------------------------------------


class TestApPerClassDegenerate:
    def test_zero_gt_no_error(self):
        result = ap_per_class(
            np.array([], dtype=bool),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        all_ap = result[5]
        assert all_ap.size == 0

    def test_zero_predictions_no_error(self):
        result = ap_per_class(
            np.array([], dtype=bool),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([0, 0, 1, 1], dtype=np.int32),
            iou_thresholds=[0.5],
        )
        _, _, p, r, f1, all_ap, unique_cls, *_ = result
        assert all_ap.shape == (2, 1)
        assert np.all(all_ap == 0.0)

    def test_no_overlap_between_pred_and_gt_classes(self):
        """Predicted class 99 never in GT → AP = 0, unique_classes from GT."""
        tp = np.zeros(10, dtype=bool)
        conf = np.ones(10, dtype=np.float32) * 0.9
        pred_cls = np.full(10, 99, dtype=np.int32)
        target_cls = np.zeros(10, dtype=np.int32)
        result = ap_per_class(tp, conf, pred_cls, target_cls, iou_thresholds=[0.5])
        unique_cls = result[6]
        all_ap = result[5]
        assert 0 in unique_cls
        # Class 0 has no TP preds → AP = 0
        assert all_ap[0, 0] == 0.0

    def test_2d_tp_shape_mismatch_raises(self):
        tp, conf, pc, tc = _make_scenario(n_cls=2, n_per_cls=10, n_thr=3)
        with pytest.raises(ValueError):
            ap_per_class(tp, conf, pc, tc, iou_thresholds=[0.5, 0.75])  # 2 thr, 3 cols

    def test_2d_tp_correct_shape(self):
        tp, conf, pc, tc = _make_scenario(n_cls=2, n_per_cls=10, n_thr=10)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=COCO_IOU_THRESHOLDS)
        assert result[5].shape == (2, 10)

    def test_single_detection_single_gt(self):
        result = ap_per_class(
            np.array([True]),
            np.array([0.9], dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            iou_thresholds=[0.5],
        )
        all_ap = result[5]
        assert all_ap.shape == (1, 1)
        assert all_ap[0, 0] > 0.0


# ---------------------------------------------------------------------------
# Metric — pre-update defaults
# ---------------------------------------------------------------------------


class TestMetricDefaults:
    def test_map50_zero(self):
        assert Metric().map50 == 0.0

    def test_map75_zero(self):
        assert Metric().map75 == 0.0

    def test_map_zero(self):
        assert Metric().map == 0.0

    def test_mp_zero(self):
        assert Metric().mp == 0.0

    def test_mr_zero(self):
        assert Metric().mr == 0.0

    def test_maps_empty(self):
        assert Metric().maps.size == 0

    def test_ap50_empty(self):
        assert Metric().ap50.size == 0

    def test_ap_empty(self):
        assert Metric().ap.size == 0

    def test_fitness_zero(self):
        assert Metric().fitness() == 0.0

    def test_curves_names(self):
        assert Metric().curves == ["F1_curve", "PR_curve", "P_curve", "R_curve"]


# ---------------------------------------------------------------------------
# Metric — post-update correctness
# ---------------------------------------------------------------------------


class TestMetricPostUpdate:
    def test_map_near_one_perfect(self, populated_metric):
        assert populated_metric.map > 0.99

    def test_map50_near_one_perfect(self, populated_metric):
        assert populated_metric.map50 > 0.99

    def test_map75_near_one_perfect(self, populated_metric):
        assert populated_metric.map75 > 0.99

    def test_mp_near_one_perfect(self, populated_metric):
        assert populated_metric.mp > 0.9

    def test_mr_near_one_perfect(self, populated_metric):
        assert populated_metric.mr > 0.9

    def test_maps_shape(self, populated_metric):
        assert populated_metric.maps.shape == (2,)

    def test_ap50_shape(self, populated_metric):
        assert populated_metric.ap50.shape == (2,)

    def test_ap_class_index_populated(self, populated_metric):
        assert populated_metric.ap_class_index == [0, 1]

    def test_class_result_returns_4_floats(self, populated_metric):
        cr = populated_metric.class_result(0)
        assert len(cr) == 4
        assert all(isinstance(v, float) for v in cr)

    def test_class_result_values_near_one(self, populated_metric):
        p, r, ap50, ap5095 = populated_metric.class_result(0)
        assert p > 0.9 and r > 0.9 and ap50 > 0.9 and ap5095 > 0.9

    def test_mean_results_length(self, populated_metric):
        assert len(populated_metric.mean_results()) == 4

    def test_mean_results_floats(self, populated_metric):
        assert all(isinstance(v, float) for v in populated_metric.mean_results())

    def test_mean_results_values(self, populated_metric):
        mp, mr, map50, mapv = populated_metric.mean_results()
        assert mp > 0.9 and mr > 0.9 and map50 > 0.9 and mapv > 0.9

    def test_fitness_formula(self, populated_metric):
        m = populated_metric
        expected = 0.1 * m.map50 + 0.9 * m.map
        assert abs(m.fitness() - expected) < 1e-6

    def test_fitness_perfect_near_one(self, populated_metric):
        assert populated_metric.fitness() > 0.99

    def test_curves_results_length(self, populated_metric):
        assert len(populated_metric.curves_results) == 4

    def test_curves_results_x_1d(self, populated_metric):
        for i, (cx, cy, _) in enumerate(populated_metric.curves_results):
            assert cx.ndim == 1, f"curve {i} x.ndim={cx.ndim}"

    def test_curves_results_y_shape(self, populated_metric):
        for i, (_, cy, _) in enumerate(populated_metric.curves_results):
            assert cy.shape[1] == 1000, f"curve {i} y.shape={cy.shape}"

    def test_p_r_curve_shapes(self, populated_metric):
        assert populated_metric.p_curve.shape == (2, 1000)
        assert populated_metric.r_curve.shape == (2, 1000)
        assert populated_metric.f1_curve.shape == (2, 1000)

    def test_update_idempotent(self, perfect_results):
        """Calling update() twice replaces (not accumulates) state."""
        m = Metric()
        m.update(perfect_results)
        map1 = m.map
        m.update(perfect_results)
        assert abs(m.map - map1) < 1e-6

    def test_all_fp_metric(self):
        tp, conf, pc, tc = _make_scenario(tp_value=False)
        m = Metric()
        m.update(ap_per_class(tp, conf, pc, tc))
        assert m.map == 0.0
        assert m.map50 == 0.0

    def test_map75_uses_index_5(self, populated_metric):
        """map75 is all_ap[:, 5].mean() — IoU=0.75 is COCO index 5."""
        expected = float(populated_metric.all_ap[:, 5].mean())
        assert abs(populated_metric.map75 - expected) < 1e-6
