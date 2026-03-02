"""Tests for DepthMetrics — Task G3.

Covers:
- Default construction
- process_batch() single-image accumulation
- finalize() averaging over multiple images
- align_scale=True bringing scale-misaligned predictor to correct abs_rel
- valid_mask exclusion
- Perfect predictor: abs_rel = 0.0, delta_1 = 1.0
- All seven metric fields (abs_rel, sq_rel, rmse, log_rmse, delta_1, delta_2, delta_3)
- Fitness property
- mean_results() / results_dict / keys
- summary() format
- to_json() / to_csv() serialisation
- Error handling: shape mismatch, empty valid region
- update() alias
- Zero-image finalize()
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from mata.eval.metrics.depth import DepthMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_gt(size: int = 100, value: float = 5.0) -> np.ndarray:
    """Return a flat (size,) ground-truth depth map."""
    return np.full(size, value, dtype=np.float64).reshape(10, 10)


def _perfect_pred(gt: np.ndarray) -> np.ndarray:
    """Return a perfect prediction identical to GT."""
    return gt.copy()


def _scaled_pred(gt: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Return a prediction scaled by a constant factor (simulates relative depth)."""
    return gt * scale


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDepthMetricsConstruction:
    def test_default_construction(self):
        m = DepthMetrics()
        assert isinstance(m, DepthMetrics)

    def test_default_align_scale_true(self):
        m = DepthMetrics()
        assert m.align_scale is True

    def test_align_scale_false(self):
        m = DepthMetrics(align_scale=False)
        assert m.align_scale is False

    def test_default_abs_rel_zero(self):
        assert DepthMetrics().abs_rel == 0.0

    def test_default_sq_rel_zero(self):
        assert DepthMetrics().sq_rel == 0.0

    def test_default_rmse_zero(self):
        assert DepthMetrics().rmse == 0.0

    def test_default_log_rmse_zero(self):
        assert DepthMetrics().log_rmse == 0.0

    def test_default_delta_1_zero(self):
        assert DepthMetrics().delta_1 == 0.0

    def test_default_delta_2_zero(self):
        assert DepthMetrics().delta_2 == 0.0

    def test_default_delta_3_zero(self):
        assert DepthMetrics().delta_3 == 0.0

    def test_default_speed_keys(self):
        m = DepthMetrics()
        assert set(m.speed.keys()) == {"preprocess", "inference", "postprocess"}

    def test_default_speed_values_zero(self):
        m = DepthMetrics()
        assert all(v == 0.0 for v in m.speed.values())


# ---------------------------------------------------------------------------
# Perfect predictor
# ---------------------------------------------------------------------------


class TestPerfectPredictor:
    def setup_method(self):
        self.gt = _flat_gt()
        self.pred = _perfect_pred(self.gt)
        self.m = DepthMetrics(align_scale=False)
        self.m.process_batch(self.pred, self.gt)
        self.m.finalize()

    def test_abs_rel_is_zero(self):
        assert self.m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_sq_rel_is_zero(self):
        assert self.m.sq_rel == pytest.approx(0.0, abs=1e-6)

    def test_rmse_is_zero(self):
        assert self.m.rmse == pytest.approx(0.0, abs=1e-6)

    def test_delta_1_is_one(self):
        assert self.m.delta_1 == pytest.approx(1.0, abs=1e-6)

    def test_delta_2_is_one(self):
        assert self.m.delta_2 == pytest.approx(1.0, abs=1e-6)

    def test_delta_3_is_one(self):
        assert self.m.delta_3 == pytest.approx(1.0, abs=1e-6)

    def test_fitness_equals_delta1_minus_abs_rel(self):
        assert self.m.fitness == pytest.approx(1.0 - 0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scale-misaligned predictor with align_scale=True
# ---------------------------------------------------------------------------


class TestAlignScale:
    def test_align_scale_true_corrects_abs_rel(self):
        """A predictor that is 2× too large should yield abs_rel ≈ 0 after alignment."""
        gt = _flat_gt(value=5.0)
        pred = _scaled_pred(gt, scale=2.0)  # all values are 10.0, GT is 5.0
        m = DepthMetrics(align_scale=True)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_align_scale_false_high_abs_rel(self):
        """Without alignment, a 2× scaled predictor should have higher abs_rel."""
        gt = _flat_gt(value=5.0)
        pred = _scaled_pred(gt, scale=2.0)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        # abs_rel = |10 - 5| / 5 = 1.0
        assert m.abs_rel == pytest.approx(1.0, abs=1e-6)

    def test_align_scale_half_scale_corrected(self):
        """A predictor that is 0.5× too small should yield abs_rel ≈ 0 after alignment."""
        gt = _flat_gt(value=4.0)
        pred = _scaled_pred(gt, scale=0.5)  # pred = 2.0 everywhere
        m = DepthMetrics(align_scale=True)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_align_scale_delta1_one_after_correction(self):
        gt = _flat_gt(value=3.0)
        pred = _scaled_pred(gt, scale=5.0)
        m = DepthMetrics(align_scale=True)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.delta_1 == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# valid_mask exclusion
# ---------------------------------------------------------------------------


class TestValidMask:
    def test_valid_mask_excludes_bad_pixels(self):
        """Only the first 5 pixels are valid; they are perfect predictions."""
        gt = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 900.0, 900.0, 900.0, 900.0, 900.0]])
        pred = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        mask = np.array([[True, True, True, True, True, False, False, False, False, False]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt, valid_mask=mask)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_valid_mask_negates_zero_gt(self):
        """GT pixels with value 0 are automatically excluded."""
        gt = np.array([[0.0, 5.0, 5.0]])
        pred = np.array([[0.0, 5.0, 5.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_valid_mask_negates_inf_gt(self):
        """GT pixels with inf are automatically excluded."""
        gt = np.array([[np.inf, 5.0]])
        pred = np.array([[99.0, 5.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)

    def test_empty_valid_region_raises(self):
        gt = np.array([[0.0, 0.0]])
        pred = np.array([[1.0, 1.0]])
        m = DepthMetrics()
        with pytest.raises(ValueError):
            m.process_batch(pred, gt)

    def test_explicit_all_false_mask_raises(self):
        gt = np.ones((2, 2), dtype=np.float64) * 5.0
        pred = gt.copy()
        mask = np.zeros((2, 2), dtype=bool)
        m = DepthMetrics()
        with pytest.raises(ValueError):
            m.process_batch(pred, gt, valid_mask=mask)


# ---------------------------------------------------------------------------
# finalize() — averaging over multiple images
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_averages_abs_rel(self):
        """abs_rel from two images should be averaged."""
        # Image 1: perfect (abs_rel = 0.0)
        gt1 = _flat_gt(value=5.0)
        pred1 = _perfect_pred(gt1)
        # Image 2: off by factor 2 without alignment (abs_rel = 1.0)
        gt2 = _flat_gt(value=5.0)
        pred2 = _scaled_pred(gt2, scale=2.0)

        m = DepthMetrics(align_scale=False)
        m.process_batch(pred1, gt1)
        m.process_batch(pred2, gt2)
        m.finalize()

        # Average: (0.0 + 1.0) / 2 = 0.5
        assert m.abs_rel == pytest.approx(0.5, abs=1e-6)

    def test_finalize_single_image_equals_that_image(self):
        gt = _flat_gt(value=10.0)
        pred = _scaled_pred(gt, scale=3.0)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        # abs_rel = |30-10|/10 = 2.0
        assert m.abs_rel == pytest.approx(2.0, abs=1e-6)

    def test_finalize_zero_images_all_zeros(self):
        m = DepthMetrics()
        m.finalize()
        assert m.abs_rel == 0.0
        assert m.delta_1 == 0.0

    def test_finalize_three_images_averaged(self):
        """Three perfect images → all metrics remain 0 / 1."""
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        m = DepthMetrics(align_scale=False)
        for _ in range(3):
            m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-6)
        assert m.delta_1 == pytest.approx(1.0, abs=1e-6)

    def test_finalize_overwrites_on_second_call(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        first_abs_rel = m.abs_rel
        # Call finalize again — should be idempotent on accumulated data
        m.finalize()
        assert m.abs_rel == pytest.approx(first_abs_rel, abs=1e-9)


# ---------------------------------------------------------------------------
# Mean results
# ---------------------------------------------------------------------------


class TestMeanResults:
    def test_mean_results_length_seven(self):
        m = DepthMetrics()
        m.process_batch(_flat_gt(), _perfect_pred(_flat_gt()))
        m.finalize()
        assert len(m.mean_results()) == 7

    def test_mean_results_perfect_values(self):
        gt = _flat_gt()
        m = DepthMetrics(align_scale=False)
        m.process_batch(_perfect_pred(gt), gt)
        m.finalize()
        results = m.mean_results()
        # First 4 are errors (should be 0), last 3 are deltas (should be 1)
        assert results[0] == pytest.approx(0.0, abs=1e-6)  # abs_rel
        assert results[4] == pytest.approx(1.0, abs=1e-6)  # delta_1


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


class TestFitness:
    def test_fitness_formula_delta1_minus_abs_rel(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.fitness == pytest.approx(m.delta_1 - m.abs_rel, abs=1e-9)

    def test_fitness_perfect_is_one(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.fitness == pytest.approx(1.0, abs=1e-6)

    def test_fitness_is_float(self):
        m = DepthMetrics()
        assert isinstance(m.fitness, float)


# ---------------------------------------------------------------------------
# results_dict
# ---------------------------------------------------------------------------


class TestResultsDict:
    def setup_method(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        self.m = DepthMetrics(align_scale=False)
        self.m.process_batch(pred, gt)
        self.m.finalize()

    def test_has_eight_keys(self):
        assert len(self.m.results_dict) == 8

    def test_has_all_metric_keys(self):
        rd = self.m.results_dict
        for k in self.m.keys:
            assert k in rd

    def test_has_fitness_key(self):
        assert "fitness" in self.m.results_dict

    def test_abs_rel_value(self):
        assert self.m.results_dict["metrics/abs_rel"] == pytest.approx(0.0, abs=1e-6)

    def test_delta_1_value(self):
        assert self.m.results_dict["metrics/delta_1"] == pytest.approx(1.0, abs=1e-6)

    def test_fitness_value(self):
        assert self.m.results_dict["fitness"] == pytest.approx(1.0, abs=1e-6)

    def test_all_values_floats(self):
        for v in self.m.results_dict.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# keys property
# ---------------------------------------------------------------------------


class TestKeysProperty:
    def test_keys_count_seven(self):
        m = DepthMetrics()
        assert len(m.keys) == 7

    def test_keys_contains_abs_rel(self):
        assert "metrics/abs_rel" in DepthMetrics().keys

    def test_keys_contains_delta_1(self):
        assert "metrics/delta_1" in DepthMetrics().keys

    def test_keys_no_fitness(self):
        assert "fitness" not in DepthMetrics().keys


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        self.m = DepthMetrics(align_scale=False)
        self.m.process_batch(pred, gt)
        self.m.finalize()

    def test_summary_single_row(self):
        assert len(self.m.summary()) == 1

    def test_summary_has_abs_rel(self):
        assert "abs_rel" in self.m.summary()[0]

    def test_summary_has_delta_1(self):
        assert "delta_1" in self.m.summary()[0]

    def test_summary_has_fitness(self):
        assert "fitness" in self.m.summary()[0]

    def test_summary_perfect_abs_rel_zero(self):
        assert self.m.summary()[0]["abs_rel"] == pytest.approx(0.0, abs=1e-6)

    def test_summary_perfect_delta_1_one(self):
        assert self.m.summary()[0]["delta_1"] == pytest.approx(1.0, abs=1e-6)

    def test_summary_all_keys_present(self):
        expected = {"abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3", "fitness"}
        assert expected.issubset(self.m.summary()[0].keys())


# ---------------------------------------------------------------------------
# to_json()
# ---------------------------------------------------------------------------


class TestToJson:
    def setup_method(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        self.m = DepthMetrics(align_scale=False)
        self.m.process_batch(pred, gt)
        self.m.finalize()

    def test_returns_string(self):
        assert isinstance(self.m.to_json(), str)

    def test_valid_json(self):
        data = json.loads(self.m.to_json())
        assert isinstance(data, dict)

    def test_has_results_key(self):
        assert "results" in json.loads(self.m.to_json())

    def test_has_speed_key(self):
        assert "speed" in json.loads(self.m.to_json())

    def test_has_summary_key(self):
        assert "summary" in json.loads(self.m.to_json())

    def test_abs_rel_in_results(self):
        data = json.loads(self.m.to_json())
        assert "metrics/abs_rel" in data["results"]

    def test_abs_rel_value_correct(self):
        data = json.loads(self.m.to_json())
        assert data["results"]["metrics/abs_rel"] == pytest.approx(0.0, abs=1e-6)

    def test_speed_serialised(self):
        self.m.speed["inference"] = 5.5
        data = json.loads(self.m.to_json())
        assert data["speed"]["inference"] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# to_csv()
# ---------------------------------------------------------------------------


class TestToCsv:
    def setup_method(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)
        self.m = DepthMetrics(align_scale=False)
        self.m.process_batch(pred, gt)
        self.m.finalize()

    def test_returns_string(self):
        assert isinstance(self.m.to_csv(), str)

    def test_has_header(self):
        lines = self.m.to_csv().strip().splitlines()
        assert len(lines) >= 1
        assert "abs_rel" in lines[0]

    def test_header_has_expected_columns(self):
        header = self.m.to_csv().strip().splitlines()[0].split(",")
        for col in ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3", "fitness"]:
            assert col in header

    def test_one_data_row(self):
        lines = [line for line in self.m.to_csv().strip().splitlines() if line]
        assert len(lines) == 2  # header + 1 data row

    def test_csv_parseable(self):
        buf = io.StringIO(self.m.to_csv())
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 1

    def test_csv_abs_rel_value(self):
        buf = io.StringIO(self.m.to_csv())
        rows = list(csv.DictReader(buf))
        assert float(rows[0]["abs_rel"]) == pytest.approx(0.0, abs=1e-6)

    def test_csv_delta_1_value(self):
        buf = io.StringIO(self.m.to_csv())
        rows = list(csv.DictReader(buf))
        assert float(rows[0]["delta_1"]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_shape_mismatch_raises(self):
        m = DepthMetrics()
        pred = np.ones((4, 4))
        gt = np.ones((5, 5))
        with pytest.raises(ValueError):
            m.process_batch(pred, gt)

    def test_shape_mismatch_error_contains_shapes(self):
        m = DepthMetrics()
        pred = np.ones((3, 3))
        gt = np.ones((4, 4))
        with pytest.raises(ValueError, match="3"):
            m.process_batch(pred, gt)


# ---------------------------------------------------------------------------
# update() alias
# ---------------------------------------------------------------------------


class TestUpdateAlias:
    def test_update_is_alias_for_process_batch(self):
        gt = _flat_gt()
        pred = _perfect_pred(gt)

        m1 = DepthMetrics(align_scale=False)
        m1.process_batch(pred, gt)
        m1.finalize()

        m2 = DepthMetrics(align_scale=False)
        m2.update(pred, gt)
        m2.finalize()

        assert m1.abs_rel == pytest.approx(m2.abs_rel, abs=1e-9)
        assert m1.delta_1 == pytest.approx(m2.delta_1, abs=1e-9)


# ---------------------------------------------------------------------------
# Varied depth values (non-uniform GT)
# ---------------------------------------------------------------------------


class TestNonUniformDepth:
    def test_abs_rel_known_value(self):
        """Manually verify abs_rel computation for a simple case."""
        # GT: [1.0, 2.0, 4.0], Pred: [2.0, 2.0, 4.0]
        # abs_rel = mean(|2-1|/1, |2-2|/2, |4-4|/4) = mean(1.0, 0.0, 0.0) = 1/3
        gt = np.array([[1.0, 2.0, 4.0]])
        pred = np.array([[2.0, 2.0, 4.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_rmse_known_value(self):
        """Verify RMSE for a simple case."""
        # GT: [1.0, 3.0], Pred: [3.0, 1.0]
        # rmse = sqrt(mean((3-1)^2, (1-3)^2)) = sqrt(mean(4, 4)) = 2.0
        gt = np.array([[1.0, 3.0]])
        pred = np.array([[3.0, 1.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.rmse == pytest.approx(2.0, abs=1e-6)

    def test_delta_1_partial(self):
        """Half the pixels within threshold, half outside."""
        # δ₁: max(p/g, g/p) < 1.25
        # pixel 0: pred=1.0, gt=1.0 → ratio=1.0 < 1.25 ✓
        # pixel 1: pred=2.0, gt=1.0 → ratio=2.0 ≥ 1.25 ✗
        gt = np.array([[1.0, 1.0]])
        pred = np.array([[1.0, 2.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.delta_1 == pytest.approx(0.5, abs=1e-6)
