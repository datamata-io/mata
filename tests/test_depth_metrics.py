"""Tests for DepthMetrics — Task C4."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mata.eval.metrics.depth import DepthMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_maps(h: int = 4, w: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, gt) depth maps with matching shapes."""
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1.0, 10.0, (h, w)).astype(np.float64)
    # small perturbation so metrics are non-trivial
    pred = gt + rng.uniform(-0.5, 0.5, (h, w))
    pred = np.maximum(pred, 0.01)
    return pred, gt


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestDepthMetricsInit:
    def test_defaults(self):
        m = DepthMetrics()
        assert m.abs_rel == 0.0
        assert m.sq_rel == 0.0
        assert m.rmse == 0.0
        assert m.log_rmse == 0.0
        assert m.delta_1 == 0.0
        assert m.delta_2 == 0.0
        assert m.delta_3 == 0.0
        assert m.align_scale is True
        assert m._count == 0

    def test_speed_defaults(self):
        m = DepthMetrics()
        assert set(m.speed.keys()) == {"preprocess", "inference", "postprocess"}
        assert all(v == 0.0 for v in m.speed.values())


# ---------------------------------------------------------------------------
# Perfect prediction
# ---------------------------------------------------------------------------


class TestPerfectPrediction:
    def test_perfect_pred_no_align(self):
        """abs_rel == 0, delta_1 == 1 when pred == gt."""
        gt = np.ones((5, 5), dtype=np.float64) * 3.0
        pred = gt.copy()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)
        assert m.sq_rel == pytest.approx(0.0, abs=1e-9)
        assert m.rmse == pytest.approx(0.0, abs=1e-9)
        assert m.log_rmse == pytest.approx(0.0, abs=1e-9)
        assert m.delta_1 == pytest.approx(1.0, abs=1e-9)
        assert m.delta_2 == pytest.approx(1.0, abs=1e-9)
        assert m.delta_3 == pytest.approx(1.0, abs=1e-9)

    def test_perfect_pred_with_align(self):
        """Median scaling on identical maps is identity — same result."""
        gt = np.ones((5, 5), dtype=np.float64) * 3.0
        pred = gt.copy()
        m = DepthMetrics(align_scale=True)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)
        assert m.delta_1 == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# process_batch — valid_mask
# ---------------------------------------------------------------------------


class TestValidMask:
    def test_none_mask_excludes_non_positive(self):
        gt = np.array([[1.0, 2.0, 0.0, -1.0]])  # last two invalid
        pred = gt.copy()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        # Only pixels 0 & 1 are evaluated — perfect prediction
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)

    def test_none_mask_excludes_inf(self):
        gt = np.array([[1.0, 2.0, np.inf]])
        pred = np.array([[1.0, 2.0, 5.0]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)

    def test_explicit_valid_mask(self):
        gt = np.array([[1.0, 2.0, 3.0]])
        pred = np.array([[2.0, 2.0, 2.0]])  # only middle pixel is perfect
        mask = np.array([[False, True, False]])
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt, valid_mask=mask)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)

    def test_explicit_mask_combined_with_auto(self):
        """User mask is AND-ed with gt > 0 filter."""
        gt = np.array([[1.0, 0.0, 3.0]])
        pred = np.array([[1.0, 5.0, 3.0]])
        mask = np.array([[True, True, True]])  # user allows all
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt, valid_mask=mask)
        m.finalize()
        # pixel 1 is excluded (gt <= 0) even though mask=True → perfect pred
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)

    def test_empty_valid_region_raises(self):
        gt = np.zeros((3, 3))
        pred = np.ones((3, 3))
        m = DepthMetrics()
        with pytest.raises(ValueError, match="No valid pixels"):
            m.process_batch(pred, gt)


# ---------------------------------------------------------------------------
# process_batch — shape mismatch
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    def test_raises_on_shape_mismatch(self):
        pred = np.ones((3, 4))
        gt = np.ones((4, 3))
        m = DepthMetrics()
        with pytest.raises(ValueError, match="shape"):
            m.process_batch(pred, gt)


# ---------------------------------------------------------------------------
# align_scale
# ---------------------------------------------------------------------------


class TestAlignScale:
    def test_scale_invariant(self):
        """pred = 2*gt; with align_scale, metrics should be perfect."""
        gt = np.ones((5, 5), dtype=np.float64) * 4.0
        pred = gt * 2.0  # 2x scale
        m = DepthMetrics(align_scale=True)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(0.0, abs=1e-9)
        assert m.delta_1 == pytest.approx(1.0, abs=1e-9)

    def test_no_align_scale_detects_error(self):
        """pred = 2*gt; without align_scale, abs_rel should be 1.0."""
        gt = np.ones((5, 5), dtype=np.float64) * 4.0
        pred = gt * 2.0
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel == pytest.approx(1.0, abs=1e-9)

    def test_zero_median_pred_no_crash(self):
        """If all predictions are 0, scale stays at 1 (no division by zero)."""
        gt = np.ones((3, 3), dtype=np.float64)
        pred = np.zeros((3, 3), dtype=np.float64)
        m = DepthMetrics(align_scale=True)
        # Should not raise — clamped to 1e-9 before log/ratio
        m.process_batch(pred, gt)
        m.finalize()
        assert np.isfinite(m.abs_rel)


# ---------------------------------------------------------------------------
# Running accumulation
# ---------------------------------------------------------------------------


class TestRunningAccumulation:
    def test_multiple_batches(self):
        """finalize() over N batches == average of individual results."""
        rng = np.random.default_rng(42)
        gt1 = rng.uniform(1.0, 5.0, (4, 4))
        pred1 = gt1 + rng.uniform(-0.2, 0.2, (4, 4))
        gt2 = rng.uniform(2.0, 8.0, (6, 6))
        pred2 = gt2 + rng.uniform(-0.3, 0.3, (6, 6))

        # Combined accumulator
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred1, gt1)
        m.process_batch(pred2, gt2)
        m.finalize()

        # Individual accumulators
        m1 = DepthMetrics(align_scale=False)
        m1.process_batch(pred1, gt1)
        m1.finalize()

        m2 = DepthMetrics(align_scale=False)
        m2.process_batch(pred2, gt2)
        m2.finalize()

        expected_abs_rel = (m1.abs_rel + m2.abs_rel) / 2.0
        assert m.abs_rel == pytest.approx(expected_abs_rel, rel=1e-6)

    def test_single_image_finalize(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert 0.0 < m.abs_rel < 5.0
        assert 0.0 <= m.delta_1 <= 1.0

    def test_finalize_with_no_batches_is_zero(self):
        m = DepthMetrics()
        m.finalize()
        assert m.abs_rel == 0.0
        assert m.delta_1 == 0.0


# ---------------------------------------------------------------------------
# update() alias
# ---------------------------------------------------------------------------


class TestUpdateAlias:
    def test_update_is_alias_for_process_batch(self):
        pred, gt = _make_maps()
        m1 = DepthMetrics(align_scale=False)
        m1.process_batch(pred, gt)
        m1.finalize()

        m2 = DepthMetrics(align_scale=False)
        m2.update(pred, gt)
        m2.finalize()

        assert m1.abs_rel == pytest.approx(m2.abs_rel, rel=1e-12)


# ---------------------------------------------------------------------------
# fitness
# ---------------------------------------------------------------------------


class TestFitness:
    def test_fitness_perfect(self):
        gt = np.ones((4, 4), dtype=np.float64) * 2.0
        pred = gt.copy()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        # delta_1=1.0, abs_rel=0.0 → fitness=1.0
        assert m.fitness == pytest.approx(1.0, abs=1e-9)

    def test_fitness_increases_with_accuracy(self):
        gt = np.ones((4, 4), dtype=np.float64) * 3.0

        m_bad = DepthMetrics(align_scale=False)
        m_bad.process_batch(gt * 3.0, gt)  # 200% error
        m_bad.finalize()

        m_good = DepthMetrics(align_scale=False)
        m_good.process_batch(gt * 1.1, gt)  # 10% error
        m_good.finalize()

        assert m_good.fitness > m_bad.fitness


# ---------------------------------------------------------------------------
# results_dict
# ---------------------------------------------------------------------------


class TestResultsDict:
    def test_results_dict_has_8_keys(self):
        m = DepthMetrics()
        m.finalize()
        d = m.results_dict
        assert len(d) == 8

    def test_results_dict_keys(self):
        m = DepthMetrics()
        m.finalize()
        expected = {
            "metrics/abs_rel",
            "metrics/sq_rel",
            "metrics/rmse",
            "metrics/log_rmse",
            "metrics/delta_1",
            "metrics/delta_2",
            "metrics/delta_3",
            "fitness",
        }
        assert set(m.results_dict.keys()) == expected

    def test_results_dict_values_are_floats(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        for k, v in m.results_dict.items():
            assert isinstance(v, float), f"Key {k!r} is not float: {type(v)}"


# ---------------------------------------------------------------------------
# mean_results
# ---------------------------------------------------------------------------


class TestMeanResults:
    def test_returns_7_elements(self):
        m = DepthMetrics()
        m.finalize()
        r = m.mean_results()
        assert len(r) == 7

    def test_order(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        r = m.mean_results()
        assert r[0] == m.abs_rel
        assert r[1] == m.sq_rel
        assert r[2] == m.rmse
        assert r[3] == m.log_rmse
        assert r[4] == m.delta_1
        assert r[5] == m.delta_2
        assert r[6] == m.delta_3


# ---------------------------------------------------------------------------
# keys property
# ---------------------------------------------------------------------------


class TestKeys:
    def test_keys_length(self):
        assert len(DepthMetrics().keys) == 7

    def test_keys_content(self):
        keys = DepthMetrics().keys
        assert "metrics/abs_rel" in keys
        assert "metrics/delta_1" in keys
        assert "metrics/log_rmse" in keys


# ---------------------------------------------------------------------------
# summary / to_json / to_csv
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_summary_returns_one_row(self):
        m = DepthMetrics()
        m.finalize()
        s = m.summary()
        assert isinstance(s, list)
        assert len(s) == 1

    def test_summary_row_keys(self):
        m = DepthMetrics()
        m.finalize()
        row = m.summary()[0]
        for k in ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3", "fitness"]:
            assert k in row

    def test_to_json_parseable(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        obj = json.loads(m.to_json())
        assert "results" in obj
        assert "speed" in obj
        assert "summary" in obj

    def test_to_csv_has_header_and_row(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        csv_str = m.to_csv()
        lines = [line for line in csv_str.strip().splitlines() if line]
        assert len(lines) == 2  # header + 1 data row
        assert "abs_rel" in lines[0]

    def test_to_csv_values_match(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        lines = m.to_csv().strip().splitlines()
        header = lines[0].split(",")
        values = lines[1].split(",")
        row = dict(zip(header, values))
        assert float(row["abs_rel"]) == pytest.approx(m.abs_rel, rel=1e-4)
        assert float(row["delta_1"]) == pytest.approx(m.delta_1, rel=1e-4)


# ---------------------------------------------------------------------------
# Metric monotonicity
# ---------------------------------------------------------------------------


class TestMetricMonotonicity:
    def test_delta_ordering(self):
        """δ₁ ≤ δ₂ ≤ δ₃ (larger threshold → more pixels pass)."""
        pred, gt = _make_maps(8, 8, seed=7)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.delta_1 <= m.delta_2 + 1e-9
        assert m.delta_2 <= m.delta_3 + 1e-9

    def test_delta_in_unit_interval(self):
        pred, gt = _make_maps(8, 8, seed=13)
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        for attr in ["delta_1", "delta_2", "delta_3"]:
            val = getattr(m, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_abs_rel_non_negative(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.abs_rel >= 0.0

    def test_rmse_non_negative(self):
        pred, gt = _make_maps()
        m = DepthMetrics(align_scale=False)
        m.process_batch(pred, gt)
        m.finalize()
        assert m.rmse >= 0.0
