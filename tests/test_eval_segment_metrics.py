"""Tests for SegmentMetrics — Task C2.

Covers:
- Inherits DetMetrics (isinstance checks, box AP properties)
- Default zero-state before update
- Populated state for both .box and .seg
- maps: element-wise mean of box.maps and seg.maps
- fitness: 0.5 * box.fitness() + 0.5 * seg.fitness()
- mean_results(): 8-element list
- class_result(i): 8-tuple
- results_dict: 9 keys (8 metrics + fitness)
- summary(): per-class dicts with box_ap50 AND mask_ap50
- to_json() / to_csv() serialisation
- Edge cases: only box populated, only seg populated, mixed class sets
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.detect import DetMetrics
from mata.eval.metrics.segment import SegmentMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_ap_results(nc: int = 2, n_thr: int = 10) -> tuple:
    ones = np.ones(nc, dtype=np.float32)
    all_ap = np.ones((nc, n_thr), dtype=np.float32)
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    return (
        np.ones(nc, dtype=np.int32),
        np.zeros(nc, dtype=np.int32),
        ones,
        ones,
        ones,
        all_ap,
        np.arange(nc, dtype=np.int32),
        np.ones((nc, 1000), dtype=np.float32),
        np.ones((nc, 1000), dtype=np.float32),
        np.ones((nc, 1000), dtype=np.float32),
        x,
        np.ones((nc, 1000), dtype=np.float32),
    )


def _zero_ap_results(nc: int = 2, n_thr: int = 10) -> tuple:
    zeros = np.zeros(nc, dtype=np.float32)
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    return (
        np.zeros(nc, dtype=np.int32),
        np.zeros(nc, dtype=np.int32),
        zeros,
        zeros,
        zeros,
        np.zeros((nc, n_thr), dtype=np.float32),
        np.arange(nc, dtype=np.int32),
        np.zeros((nc, 1000), dtype=np.float32),
        np.zeros((nc, 1000), dtype=np.float32),
        np.zeros((nc, 1000), dtype=np.float32),
        x,
        np.zeros((nc, 1000), dtype=np.float32),
    )


def _make_both_perfect(nc: int = 2, names: dict | None = None) -> SegmentMetrics:
    names = names or {i: f"cls{i}" for i in range(nc)}
    sm = SegmentMetrics(names=names)
    sm.box.update(_perfect_ap_results(nc))
    sm.seg.update(_perfect_ap_results(nc))
    return sm


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_instance_of_detmetrics(self):
        assert isinstance(SegmentMetrics(), DetMetrics)

    def test_is_instance_of_segmentmetrics(self):
        assert isinstance(SegmentMetrics(), SegmentMetrics)

    def test_has_box_field(self):
        sm = SegmentMetrics()
        assert isinstance(sm.box, Metric)

    def test_has_seg_field(self):
        sm = SegmentMetrics()
        assert isinstance(sm.seg, Metric)

    def test_default_speed_keys(self):
        sm = SegmentMetrics()
        assert set(sm.speed.keys()) == {"preprocess", "inference", "postprocess"}

    def test_default_names_empty(self):
        sm = SegmentMetrics()
        assert sm.names == {}

    def test_confusion_matrix_default_none(self):
        sm = SegmentMetrics()
        assert sm.confusion_matrix is None

    def test_save_dir_default_empty(self):
        sm = SegmentMetrics()
        assert sm.save_dir == ""


# ---------------------------------------------------------------------------
# Zero-state (before any update)
# ---------------------------------------------------------------------------


class TestZeroState:
    def setup_method(self):
        self.sm = SegmentMetrics(names={0: "cat", 1: "dog"})

    def test_box_map_zero(self):
        assert self.sm.box.map == 0.0

    def test_seg_map_zero(self):
        assert self.sm.seg.map == 0.0

    def test_seg_map50_zero(self):
        assert self.sm.seg.map50 == 0.0

    def test_seg_map75_zero(self):
        assert self.sm.seg.map75 == 0.0

    def test_seg_maps_empty(self):
        arr = self.sm.seg.maps
        assert isinstance(arr, np.ndarray)
        assert arr.size == 0

    def test_maps_property_empty_when_both_zero(self):
        arr = self.sm.maps
        assert isinstance(arr, np.ndarray)

    def test_fitness_zero(self):
        assert self.sm.fitness == 0.0

    def test_mean_results_length_eight(self):
        assert len(self.sm.mean_results()) == 8

    def test_mean_results_all_zeros(self):
        assert all(v == 0.0 for v in self.sm.mean_results())

    def test_results_dict_nine_keys(self):
        assert len(self.sm.results_dict) == 9

    def test_results_dict_has_fitness(self):
        assert "fitness" in self.sm.results_dict

    def test_results_dict_all_zeros(self):
        for k, v in self.sm.results_dict.items():
            assert v == 0.0, f"{k} should be 0.0 before update"

    def test_keys_eight_entries(self):
        assert len(self.sm.keys) == 8

    def test_summary_empty(self):
        assert self.sm.summary() == []


# ---------------------------------------------------------------------------
# Populated state
# ---------------------------------------------------------------------------


class TestPopulated:
    def setup_method(self):
        self.nc = 3
        self.names = {0: "cat", 1: "dog", 2: "bird"}
        self.sm = _make_both_perfect(nc=self.nc, names=self.names)

    # box metrics (inherited)
    def test_box_map_one(self):
        assert self.sm.box.map == pytest.approx(1.0, abs=1e-4)

    def test_box_map50_one(self):
        assert self.sm.box.map50 == pytest.approx(1.0, abs=1e-4)

    def test_box_map75_one(self):
        assert self.sm.box.map75 == pytest.approx(1.0, abs=1e-4)

    def test_box_mp_one(self):
        assert self.sm.box.mp == pytest.approx(1.0, abs=1e-4)

    def test_box_mr_one(self):
        assert self.sm.box.mr == pytest.approx(1.0, abs=1e-4)

    # seg metrics
    def test_seg_map_one(self):
        assert self.sm.seg.map == pytest.approx(1.0, abs=1e-4)

    def test_seg_map50_one(self):
        assert self.sm.seg.map50 == pytest.approx(1.0, abs=1e-4)

    def test_seg_map75_one(self):
        assert self.sm.seg.map75 == pytest.approx(1.0, abs=1e-4)

    def test_seg_maps_shape(self):
        assert self.sm.seg.maps.shape == (self.nc,)

    def test_seg_maps_all_one(self):
        assert np.allclose(self.sm.seg.maps, 1.0)

    # maps property
    def test_maps_shape(self):
        assert self.sm.maps.shape == (self.nc,)

    def test_maps_mean_of_box_and_seg(self):
        expected = (self.sm.box.maps + self.sm.seg.maps) / 2.0
        assert np.allclose(self.sm.maps, expected)

    def test_maps_all_one_when_both_perfect(self):
        assert np.allclose(self.sm.maps, 1.0)

    # fitness
    def test_fitness_formula(self):
        expected = 0.5 * self.sm.box.fitness() + 0.5 * self.sm.seg.fitness()
        assert self.sm.fitness == pytest.approx(expected, abs=1e-6)

    def test_fitness_one_when_both_perfect(self):
        assert self.sm.fitness == pytest.approx(1.0, abs=1e-4)

    def test_fitness_is_property_not_method(self):
        # fitness must be accessible as a property (not called as method)
        val = self.sm.fitness
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# mean_results()
# ---------------------------------------------------------------------------


class TestMeanResults:
    def setup_method(self):
        self.sm = _make_both_perfect(nc=2)

    def test_length_eight(self):
        assert len(self.sm.mean_results()) == 8

    def test_first_four_are_box(self):
        res = self.sm.mean_results()
        assert res[0] == pytest.approx(self.sm.box.mp, abs=1e-6)
        assert res[1] == pytest.approx(self.sm.box.mr, abs=1e-6)
        assert res[2] == pytest.approx(self.sm.box.map50, abs=1e-6)
        assert res[3] == pytest.approx(self.sm.box.map, abs=1e-6)

    def test_last_four_are_seg(self):
        res = self.sm.mean_results()
        assert res[4] == pytest.approx(self.sm.seg.mp, abs=1e-6)
        assert res[5] == pytest.approx(self.sm.seg.mr, abs=1e-6)
        assert res[6] == pytest.approx(self.sm.seg.map50, abs=1e-6)
        assert res[7] == pytest.approx(self.sm.seg.map, abs=1e-6)

    def test_all_one_when_perfect(self):
        assert all(v == pytest.approx(1.0, abs=1e-4) for v in self.sm.mean_results())


# ---------------------------------------------------------------------------
# class_result()
# ---------------------------------------------------------------------------


class TestClassResult:
    def setup_method(self):
        self.sm = _make_both_perfect(nc=3)

    def test_returns_eight_tuple(self):
        result = self.sm.class_result(0)
        assert len(result) == 8

    def test_all_floats(self):
        for v in self.sm.class_result(1):
            assert isinstance(v, float)

    def test_first_four_match_box(self):
        bp, br, bap50, bap, sp, sr, sap50, sap = self.sm.class_result(0)
        pb, rb, ab50, ab = self.sm.box.class_result(0)
        assert bp == pytest.approx(pb, abs=1e-6)
        assert br == pytest.approx(rb, abs=1e-6)
        assert bap50 == pytest.approx(ab50, abs=1e-6)
        assert bap == pytest.approx(ab, abs=1e-6)

    def test_last_four_match_seg(self):
        bp, br, bap50, bap, sp, sr, sap50, sap = self.sm.class_result(0)
        ps, rs, as50, asm = self.sm.seg.class_result(0)
        assert sp == pytest.approx(ps, abs=1e-6)
        assert sr == pytest.approx(rs, abs=1e-6)
        assert sap50 == pytest.approx(as50, abs=1e-6)
        assert sap == pytest.approx(asm, abs=1e-6)

    def test_out_of_range_raises(self):
        with pytest.raises((IndexError, Exception)):
            self.sm.class_result(100)


# ---------------------------------------------------------------------------
# results_dict
# ---------------------------------------------------------------------------


class TestResultsDict:
    def setup_method(self):
        self.sm = _make_both_perfect(nc=2, names={0: "a", 1: "b"})

    def test_nine_keys(self):
        assert len(self.sm.results_dict) == 9

    def test_all_keys_keys_present(self):
        rd = self.sm.results_dict
        for k in self.sm.keys:
            assert k in rd

    def test_fitness_present(self):
        assert "fitness" in self.sm.results_dict

    def test_box_precision_key(self):
        assert self.sm.results_dict["metrics/precision(B)"] == pytest.approx(1.0, abs=1e-4)

    def test_mask_precision_key(self):
        assert self.sm.results_dict["metrics/precision(M)"] == pytest.approx(1.0, abs=1e-4)

    def test_mask_map50_key(self):
        assert self.sm.results_dict["metrics/mAP50(M)"] == pytest.approx(1.0, abs=1e-4)

    def test_mask_map5095_key(self):
        assert self.sm.results_dict["metrics/mAP50-95(M)"] == pytest.approx(1.0, abs=1e-4)

    def test_fitness_value(self):
        sm = self.sm
        expected = 0.5 * sm.box.fitness() + 0.5 * sm.seg.fitness()
        assert sm.results_dict["fitness"] == pytest.approx(expected, abs=1e-6)

    def test_all_values_floats(self):
        for v in self.sm.results_dict.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.names = {0: "cat", 1: "dog"}
        self.sm = _make_both_perfect(nc=2, names=self.names)

    def test_length_equals_nc(self):
        assert len(self.sm.summary()) == 2

    def test_row_has_box_and_mask_ap50(self):
        for row in self.sm.summary():
            assert "ap50" in row
            assert "mask_ap50" in row

    def test_row_has_all_required_keys(self):
        required = {
            "class_id",
            "class_name",
            "num_targets",
            "precision",
            "recall",
            "f1_score",
            "ap50",
            "ap50_95",
            "mask_ap50",
            "mask_ap50_95",
        }
        for row in self.sm.summary():
            assert required.issubset(row.keys())

    def test_class_names_resolved(self):
        rows = self.sm.summary()
        by_id = {r["class_id"]: r for r in rows}
        assert by_id[0]["class_name"] == "cat"
        assert by_id[1]["class_name"] == "dog"

    def test_box_ap50_one(self):
        for row in self.sm.summary():
            assert row["ap50"] == pytest.approx(1.0, abs=1e-4)

    def test_mask_ap50_one(self):
        for row in self.sm.summary():
            assert row["mask_ap50"] == pytest.approx(1.0, abs=1e-4)

    def test_mask_ap50_95_one(self):
        for row in self.sm.summary():
            assert row["mask_ap50_95"] == pytest.approx(1.0, abs=1e-4)

    def test_empty_before_update(self):
        sm = SegmentMetrics(names={0: "x"})
        assert sm.summary() == []

    def test_unknown_class_name_fallback(self):
        sm = SegmentMetrics(names={})
        sm.box.update(_perfect_ap_results(nc=1))
        sm.seg.update(_perfect_ap_results(nc=1))
        row = sm.summary()[0]
        assert isinstance(row["class_name"], str)

    def test_only_box_populated(self):
        """Classes only in box show 0.0 for mask AP."""
        sm = SegmentMetrics(names={0: "cat", 1: "dog"})
        sm.box.update(_perfect_ap_results(nc=2))
        # seg stays empty
        rows = sm.summary()
        assert len(rows) == 2
        for row in rows:
            assert row["ap50"] == pytest.approx(1.0, abs=1e-4)
            assert row["mask_ap50"] == 0.0

    def test_only_seg_populated(self):
        """Classes only in seg show 0.0 for box AP."""
        sm = SegmentMetrics(names={0: "cat", 1: "dog"})
        sm.seg.update(_perfect_ap_results(nc=2))
        rows = sm.summary()
        assert len(rows) == 2
        for row in rows:
            assert row["mask_ap50"] == pytest.approx(1.0, abs=1e-4)
            assert row["ap50"] == 0.0


# ---------------------------------------------------------------------------
# to_json() / to_csv()
# ---------------------------------------------------------------------------


class TestSerialisation:
    def setup_method(self):
        self.sm = _make_both_perfect(nc=2, names={0: "cat", 1: "dog"})

    def test_to_json_valid(self):
        data = json.loads(self.sm.to_json())
        assert isinstance(data, dict)

    def test_to_json_has_results_speed_per_class(self):
        data = json.loads(self.sm.to_json())
        assert "results" in data
        assert "speed" in data
        assert "per_class" in data

    def test_to_json_per_class_length(self):
        data = json.loads(self.sm.to_json())
        assert len(data["per_class"]) == 2

    def test_to_json_per_class_has_mask_ap50(self):
        data = json.loads(self.sm.to_json())
        for row in data["per_class"]:
            assert "mask_ap50" in row

    def test_to_csv_returns_string(self):
        assert isinstance(self.sm.to_csv(), str)

    def test_to_csv_has_mask_columns(self):
        header = self.sm.to_csv().splitlines()[0]
        assert "mask_ap50" in header

    def test_to_csv_row_count(self):
        lines = [line for line in self.sm.to_csv().strip().splitlines() if line]
        assert len(lines) == 3  # header + 2 rows

    def test_to_csv_parseable(self):
        buf = io.StringIO(self.sm.to_csv())
        rows = list(csv.DictReader(buf))
        assert len(rows) == 2
        assert "mask_ap50" in rows[0]

    def test_to_json_empty_state(self):
        sm = SegmentMetrics(names={})
        data = json.loads(sm.to_json())
        assert data["per_class"] == []

    def test_to_csv_empty_state_header_only(self):
        sm = SegmentMetrics(names={})
        lines = [line for line in sm.to_csv().strip().splitlines() if line]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# maps property edge-cases
# ---------------------------------------------------------------------------


class TestMapsProperty:
    def test_maps_mean_of_box_and_seg(self):
        sm = SegmentMetrics()
        sm.box.update(_perfect_ap_results(nc=4))
        sm.seg.update(_zero_ap_results(nc=4))
        expected = (np.ones(4) + np.zeros(4)) / 2.0
        assert np.allclose(sm.maps, expected, atol=1e-4)

    def test_maps_only_box_populated_returns_box(self):
        sm = SegmentMetrics()
        sm.box.update(_perfect_ap_results(nc=2))
        # seg is empty (size 0) → maps falls back to box
        arr = sm.maps
        assert arr.size > 0

    def test_maps_only_seg_populated_returns_seg(self):
        sm = SegmentMetrics()
        sm.seg.update(_perfect_ap_results(nc=2))
        arr = sm.maps
        assert arr.size > 0

    def test_maps_both_empty_returns_empty(self):
        sm = SegmentMetrics()
        arr = sm.maps
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# fitness property vs method distinction
# ---------------------------------------------------------------------------


class TestFitnessProperty:
    def test_fitness_is_not_callable(self):
        """fitness must be a @property, not a plain method."""
        sm = _make_both_perfect(nc=2)
        # Accessing sm.fitness should give a float, not a bound method
        val = sm.fitness
        assert isinstance(val, float)

    def test_fitness_half_half(self):
        sm = SegmentMetrics(names={0: "x"})
        sm.box.update(_perfect_ap_results(nc=1))  # box fitness = 1.0
        sm.seg.update(_zero_ap_results(nc=1))  # seg fitness = 0.0
        assert sm.fitness == pytest.approx(0.5, abs=1e-4)

    def test_fitness_both_zero(self):
        sm = SegmentMetrics()
        assert sm.fitness == 0.0

    def test_fitness_both_perfect(self):
        sm = _make_both_perfect(nc=2)
        assert sm.fitness == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Integration with real ap_per_class()
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_real_ap_per_class_both_metrics(self):
        n = 60
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.4, 1.0, n).astype(np.float32)
        pred_cls = np.tile([0, 1, 2], n // 3).astype(np.int32)
        target_cls = np.tile([0, 1, 2], n // 3).astype(np.int32)
        tp = np.ones(n, dtype=bool)

        box_res = ap_per_class(tp, conf, pred_cls, target_cls)
        seg_res = ap_per_class(tp, conf, pred_cls, target_cls)

        sm = SegmentMetrics(names={0: "a", 1: "b", 2: "c"})
        sm.box.update(box_res)
        sm.seg.update(seg_res)

        assert sm.box.map > 0.0
        assert sm.seg.map > 0.0
        assert len(sm.summary()) == 3
        rd = sm.results_dict
        assert len(rd) == 9
        assert rd["fitness"] == pytest.approx(sm.fitness, abs=1e-6)

    def test_seg_independent_of_box(self):
        """box and seg can hold different AP values."""
        sm = SegmentMetrics(names={0: "x", 1: "y"})
        sm.box.update(_perfect_ap_results(nc=2))
        sm.seg.update(_zero_ap_results(nc=2))

        assert sm.box.map == pytest.approx(1.0, abs=1e-4)
        assert sm.seg.map == pytest.approx(0.0, abs=1e-4)
        assert sm.fitness == pytest.approx(0.5 * 1.0 + 0.5 * 0.0, abs=1e-4)
