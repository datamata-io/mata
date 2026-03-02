"""Unit tests for ap_per_class() and Metric container — Task G1.

Covers all named test cases from the G1 acceptance spec:
  - ap_per_class: perfect preds, all-FP, zero GT, single class, shapes, curves
  - Metric: pre-update defaults, post-update properties, fitness, class_result,
    mean_results, map75 index, curves_results shapes

All tests use only numpy + synthetic data. No mocks, no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.iou import COCO_IOU_THRESHOLDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(
    n_cls: int = 2,
    n_per_cls: int = 20,
    tp_value: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (tp, conf, pred_cls, target_cls) for a uniform scenario."""
    pred_cls = np.repeat(np.arange(n_cls, dtype=np.int32), n_per_cls)
    target_cls = pred_cls.copy()
    conf = np.linspace(0.99, 0.51, n_cls * n_per_cls, dtype=np.float32)
    tp = np.full(len(pred_cls), tp_value, dtype=bool)
    return tp, conf, pred_cls, target_cls


@pytest.fixture()
def perfect_results() -> tuple:
    tp, conf, pc, tc = _scenario(n_cls=2, n_per_cls=20, tp_value=True)
    return ap_per_class(tp, conf, pc, tc, iou_thresholds=COCO_IOU_THRESHOLDS)


@pytest.fixture()
def all_fp_results() -> tuple:
    tp, conf, pc, tc = _scenario(n_cls=2, n_per_cls=20, tp_value=False)
    return ap_per_class(tp, conf, pc, tc, iou_thresholds=COCO_IOU_THRESHOLDS)


@pytest.fixture()
def populated_metric(perfect_results: tuple) -> Metric:
    m = Metric()
    m.update(perfect_results)
    return m


# ===========================================================================
# ap_per_class tests
# ===========================================================================


class TestApPerClass:
    # ---- G1-named tests ---------------------------------------------------

    def test_perfect_predictions_ap_equals_one(self, perfect_results: tuple):
        """Perfect TPs → AP close to 1.0 at every threshold."""
        all_ap = perfect_results[5]
        assert np.all(all_ap > 0.99), f"Expected near-1.0, got min={all_ap.min()}"

    def test_all_fp_predictions_ap_equals_zero(self, all_fp_results: tuple):
        """All FP → AP = 0.0 at every threshold."""
        all_ap = all_fp_results[5]
        assert np.all(all_ap == 0.0), f"Expected 0.0, got max={all_ap.max()}"

    def test_zero_gt_instances_handled(self):
        """Empty target_cls must not raise and returns zero arrays."""
        result = ap_per_class(
            tp=np.array([], dtype=bool),
            conf=np.array([], dtype=np.float32),
            pred_cls=np.array([], dtype=np.int32),
            target_cls=np.array([], dtype=np.int32),
            iou_thresholds=[0.50],
        )
        all_ap = result[5]
        assert all_ap.size == 0  # nc=0

    def test_single_class_case(self):
        """nc=1 works without error and returns correct shapes."""
        tp = np.ones(10, dtype=bool)
        conf = np.linspace(0.9, 0.5, 10, dtype=np.float32)
        pred_cls = np.zeros(10, dtype=np.int32)
        target_cls = np.zeros(10, dtype=np.int32)
        result = ap_per_class(tp, conf, pred_cls, target_cls, iou_thresholds=[0.5])
        all_ap = result[5]
        assert all_ap.shape == (1, 1)
        assert all_ap[0, 0] > 0.9

    def test_output_shapes_correct(self, perfect_results: tuple):
        tp_mf1, fp_mf1, p, r, f1, all_ap, unique_cls, p_curve, r_curve, f1_curve, x, prec_values = perfect_results
        nc = 2
        assert tp_mf1.shape == (nc,)
        assert fp_mf1.shape == (nc,)
        assert p.shape == (nc,)
        assert r.shape == (nc,)
        assert f1.shape == (nc,)
        assert all_ap.shape == (nc, 10)
        assert p_curve.shape == (nc, 1000)
        assert r_curve.shape == (nc, 1000)
        assert f1_curve.shape == (nc, 1000)
        assert prec_values.shape == (nc, 1000)
        assert unique_cls.tolist() == [0, 1]

    def test_confidence_curves_shape_1000(self, perfect_results: tuple):
        """x axis and all 1000-width curves have shape (1000,) / (nc, 1000)."""
        *_, p_curve, r_curve, f1_curve, x, prec_values = perfect_results
        assert x.shape == (1000,)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0, abs=2e-3)
        for arr in (p_curve, r_curve, f1_curve, prec_values):
            assert arr.shape[1] == 1000

    # ---- additional correctness tests ------------------------------------

    def test_returns_12_element_tuple(self, perfect_results: tuple):
        assert isinstance(perfect_results, tuple)
        assert len(perfect_results) == 12

    def test_precision_recall_in_range(self, perfect_results: tuple):
        _, _, p, r, f1, *_ = perfect_results
        assert np.all((p >= 0) & (p <= 1))
        assert np.all((r >= 0) & (r <= 1))
        assert np.all((f1 >= 0) & (f1 <= 1))

    def test_all_fp_tp_count_zero(self, all_fp_results: tuple):
        tp_mf1 = all_fp_results[0]
        assert np.all(tp_mf1 == 0)

    def test_zero_predictions_no_error(self):
        result = ap_per_class(
            tp=np.array([], dtype=bool),
            conf=np.array([], dtype=np.float32),
            pred_cls=np.array([], dtype=np.int32),
            target_cls=np.array([0, 0, 1, 1], dtype=np.int32),
            iou_thresholds=[0.5],
        )
        all_ap = result[5]
        assert all_ap.shape == (2, 1)
        assert np.all(all_ap == 0.0)

    def test_all_fp_precision_recall_zero(self, all_fp_results: tuple):
        _, _, p, r, *_ = all_fp_results
        assert np.all(p == pytest.approx(0.0, abs=0.01))

    def test_f1_consistent_with_p_r(self, perfect_results: tuple):
        _, _, p, r, f1, *_ = perfect_results
        eps = 1e-16
        expected = 2 * p * r / (p + r + eps)
        np.testing.assert_allclose(f1, expected, atol=1e-4)

    def test_all_ap_dtype_float32(self, perfect_results: tuple):
        all_ap = perfect_results[5]
        assert all_ap.dtype == np.float32

    def test_x_axis_dtype_float32(self, perfect_results: tuple):
        x = perfect_results[10]
        assert x.dtype == np.float32

    def test_unique_classes_type_int(self, perfect_results: tuple):
        unique_cls = perfect_results[6]
        assert unique_cls.dtype in (np.int32, np.int64)

    def test_coco_10_thresholds_shape(self):
        tp, conf, pc, tc = _scenario(n_cls=3)
        result = ap_per_class(tp, conf, pc, tc, iou_thresholds=COCO_IOU_THRESHOLDS)
        assert result[5].shape == (3, 10)

    def test_mixed_tp_fp_ap_between_zero_one(self):
        rng = np.random.default_rng(42)
        n = 50
        pred_cls = np.zeros(n, dtype=np.int32)
        target_cls = np.zeros(n, dtype=np.int32)
        conf = rng.random(n).astype(np.float32)
        tp = rng.random(n) > 0.5
        result = ap_per_class(tp, conf, pred_cls, target_cls, iou_thresholds=[0.5])
        ap = float(result[5][0, 0])
        assert 0.0 <= ap <= 1.0


# ===========================================================================
# Metric container tests
# ===========================================================================


class TestMetricDefaults:
    """All properties return 0.0 / empty arrays before update()."""

    def test_metric_properties_before_update_return_zeros(self):
        m = Metric()
        assert m.map50 == 0.0
        assert m.map75 == 0.0
        assert m.map == 0.0
        assert m.mp == 0.0
        assert m.mr == 0.0
        assert m.fitness() == 0.0

    def test_maps_empty_before_update(self):
        assert Metric().maps.size == 0

    def test_ap50_empty_before_update(self):
        assert Metric().ap50.size == 0

    def test_ap_empty_before_update(self):
        assert Metric().ap.size == 0

    def test_curves_names(self):
        assert Metric().curves == ["F1_curve", "PR_curve", "P_curve", "R_curve"]

    def test_ap_class_index_empty(self):
        assert Metric().ap_class_index == []

    def test_p_r_f1_lists_empty(self):
        m = Metric()
        assert m.p == []
        assert m.r == []
        assert m.f1 == []

    def test_no_attribute_error_on_all_properties(self):
        """Accessing every property on an un-populated Metric must never raise."""
        m = Metric()
        _ = m.ap50
        _ = m.ap
        _ = m.map50
        _ = m.map75
        _ = m.map
        _ = m.maps
        _ = m.mp
        _ = m.mr
        _ = m.curves
        _ = m.curves_results
        _ = m.fitness()
        _ = m.mean_results()


class TestMetricPostUpdate:
    """Correctness tests after Metric.update() with perfect predictions."""

    def test_metric_map50_after_update(self, populated_metric: Metric):
        assert populated_metric.map50 > 0.99

    def test_metric_map_after_update(self, populated_metric: Metric):
        assert populated_metric.map > 0.99

    def test_metric_map75_uses_iou_index_5(self, populated_metric: Metric):
        """map75 must equal all_ap[:, 5].mean() (index 5 = IoU 0.75)."""
        expected = float(populated_metric.all_ap[:, 5].mean())
        assert abs(populated_metric.map75 - expected) < 1e-6

    def test_metric_maps_shape_nc(self, populated_metric: Metric):
        """maps is per-class mAP50-95, shape (nc,)."""
        assert populated_metric.maps.shape == (2,)
        assert np.all(populated_metric.maps > 0.99)

    def test_metric_fitness_formula(self, populated_metric: Metric):
        """fitness() == 0.1 * map50 + 0.9 * map."""
        m = populated_metric
        expected = 0.1 * m.map50 + 0.9 * m.map
        assert abs(m.fitness() - expected) < 1e-6

    def test_metric_fitness_near_one_perfect(self, populated_metric: Metric):
        assert populated_metric.fitness() > 0.99

    def test_metric_class_result_returns_4_tuple(self, populated_metric: Metric):
        cr = populated_metric.class_result(0)
        assert isinstance(cr, tuple)
        assert len(cr) == 4
        assert all(isinstance(v, float) for v in cr)

    def test_metric_class_result_values_valid(self, populated_metric: Metric):
        p, r, ap50, ap5095 = populated_metric.class_result(0)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert 0.0 <= ap50 <= 1.0
        assert 0.0 <= ap5095 <= 1.0
        assert p > 0.9 and r > 0.9

    def test_metric_mean_results_returns_4_floats(self, populated_metric: Metric):
        result = populated_metric.mean_results()
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_metric_mean_results_values_near_one(self, populated_metric: Metric):
        mp, mr, map50, mapv = populated_metric.mean_results()
        assert mp > 0.9 and mr > 0.9 and map50 > 0.99 and mapv > 0.99

    def test_metric_ap_class_index_populated(self, populated_metric: Metric):
        assert populated_metric.ap_class_index == [0, 1]

    def test_metric_p_r_f1_lists_populated(self, populated_metric: Metric):
        assert len(populated_metric.p) == 2
        assert len(populated_metric.r) == 2
        assert len(populated_metric.f1) == 2

    def test_all_fp_metric_map_zero(self):
        tp, conf, pc, tc = _scenario(tp_value=False)
        m = Metric()
        m.update(ap_per_class(tp, conf, pc, tc))
        assert m.map == 0.0
        assert m.map50 == 0.0

    def test_update_replaces_state(self, perfect_results: tuple):
        """Calling update() twice replaces (not accumulates) state."""
        m = Metric()
        m.update(perfect_results)
        map1 = m.map
        m.update(perfect_results)
        assert abs(m.map - map1) < 1e-6

    def test_curves_results_length(self, populated_metric: Metric):
        assert len(populated_metric.curves_results) == 4

    def test_curves_results_x_is_1d(self, populated_metric: Metric):
        for i, (cx, cy, _) in enumerate(populated_metric.curves_results):
            assert cx.ndim == 1, f"Curve {i}: x.ndim={cx.ndim}, expected 1"

    def test_curves_results_y_width_1000(self, populated_metric: Metric):
        for i, (_, cy, _) in enumerate(populated_metric.curves_results):
            assert cy.shape[1] == 1000, f"Curve {i}: y.shape={cy.shape}"

    def test_map75_greater_than_zero_perfect(self, populated_metric: Metric):
        assert populated_metric.map75 > 0.98

    def test_mp_mr_near_one_perfect(self, populated_metric: Metric):
        assert populated_metric.mp > 0.9
        assert populated_metric.mr > 0.9
