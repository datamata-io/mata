"""Tests for DetMetrics — Task C1.

Covers:
- Default construction with all-zero / empty state
- All property accessors
- results_dict exact keys and fitness formula
- summary() per-class rows
- to_json() / to_csv() serialisation
- Populated state after box.update(ap_per_class(...))
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.detect import DetMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_ap_results(nc: int = 2, n_thr: int = 10) -> tuple:
    """Return an ap_per_class result tuple that simulates perfect detection.

    All nc classes have AP = 1.0 at every IoU threshold.
    """
    ones = np.ones(nc, dtype=np.float32)
    all_ap = np.ones((nc, n_thr), dtype=np.float32)
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    return (
        np.ones(nc, dtype=np.int32),  # tp_at_max_f1
        np.zeros(nc, dtype=np.int32),  # fp_at_max_f1
        ones,  # p
        ones,  # r
        ones,  # f1
        all_ap,  # all_ap (nc, 10)
        np.arange(nc, dtype=np.int32),  # unique_classes
        np.ones((nc, 1000), dtype=np.float32),  # p_curve
        np.ones((nc, 1000), dtype=np.float32),  # r_curve
        np.ones((nc, 1000), dtype=np.float32),  # f1_curve
        x,  # x
        np.ones((nc, 1000), dtype=np.float32),  # prec_values
    )


def _make_populated(nc: int = 2, names: dict | None = None) -> DetMetrics:
    names = names or {i: f"cls{i}" for i in range(nc)}
    dm = DetMetrics(names=names)
    dm.box.update(_perfect_ap_results(nc))
    return dm


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestDetMetricsConstruction:
    def test_default_construction_no_args(self):
        dm = DetMetrics()
        assert isinstance(dm, DetMetrics)

    def test_names_kwarg(self):
        dm = DetMetrics(names={0: "cat", 1: "dog"})
        assert dm.names == {0: "cat", 1: "dog"}

    def test_default_names_empty_dict(self):
        dm = DetMetrics()
        assert dm.names == {}

    def test_default_speed_keys(self):
        dm = DetMetrics()
        assert set(dm.speed.keys()) == {"preprocess", "inference", "postprocess"}
        assert all(v == 0.0 for v in dm.speed.values())

    def test_box_is_metric_instance(self):
        dm = DetMetrics()
        assert isinstance(dm.box, Metric)

    def test_confusion_matrix_default_none(self):
        dm = DetMetrics()
        assert dm.confusion_matrix is None

    def test_save_dir_default_empty(self):
        dm = DetMetrics()
        assert dm.save_dir == ""


# ---------------------------------------------------------------------------
# Zero-state property tests (before box.update)
# ---------------------------------------------------------------------------


class TestDetMetricsZeroState:
    def setup_method(self):
        self.dm = DetMetrics(names={0: "cat"})

    def test_maps_empty_before_update(self):
        arr = self.dm.maps
        assert isinstance(arr, np.ndarray)
        assert arr.size == 0

    def test_ap_class_index_empty_before_update(self):
        assert self.dm.ap_class_index == []

    def test_mean_results_zeros_before_update(self):
        res = self.dm.mean_results()
        assert len(res) == 4
        assert all(v == 0.0 for v in res)

    def test_fitness_zero_before_update(self):
        assert self.dm.fitness() == 0.0

    def test_keys_correct(self):
        assert self.dm.keys == [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    def test_results_dict_has_five_keys(self):
        rd = self.dm.results_dict
        assert len(rd) == 5

    def test_results_dict_has_fitness_key(self):
        rd = self.dm.results_dict
        assert "fitness" in rd

    def test_results_dict_all_zeros_before_update(self):
        rd = self.dm.results_dict
        for k, v in rd.items():
            assert v == 0.0, f"{k} should be 0.0 before update"

    def test_summary_empty_before_update(self):
        assert self.dm.summary() == []

    def test_class_result_raises_index_error_on_empty(self):
        with pytest.raises((IndexError, Exception)):
            self.dm.class_result(0)


# ---------------------------------------------------------------------------
# Populated-state tests
# ---------------------------------------------------------------------------


class TestDetMetricsPopulated:
    def setup_method(self):
        self.nc = 3
        self.names = {0: "cat", 1: "dog", 2: "bird"}
        self.dm = _make_populated(nc=self.nc, names=self.names)

    def test_box_map_is_one(self):
        assert self.dm.box.map == pytest.approx(1.0, abs=1e-4)

    def test_box_map50_is_one(self):
        assert self.dm.box.map50 == pytest.approx(1.0, abs=1e-4)

    def test_box_map75_is_one(self):
        assert self.dm.box.map75 == pytest.approx(1.0, abs=1e-4)

    def test_box_mp_is_one(self):
        assert self.dm.box.mp == pytest.approx(1.0, abs=1e-4)

    def test_box_mr_is_one(self):
        assert self.dm.box.mr == pytest.approx(1.0, abs=1e-4)

    def test_maps_shape(self):
        assert self.dm.maps.shape == (self.nc,)

    def test_maps_values_one(self):
        assert np.allclose(self.dm.maps, 1.0)

    def test_ap_class_index_length(self):
        assert len(self.dm.ap_class_index) == self.nc

    def test_fitness_formula(self):
        expected = 0.1 * self.dm.box.map50 + 0.9 * self.dm.box.map
        assert self.dm.fitness() == pytest.approx(expected, abs=1e-6)

    def test_fitness_perfect_is_one(self):
        assert self.dm.fitness() == pytest.approx(1.0, abs=1e-4)

    def test_mean_results_length(self):
        assert len(self.dm.mean_results()) == 4

    def test_mean_results_all_ones(self):
        res = self.dm.mean_results()
        assert all(v == pytest.approx(1.0, abs=1e-4) for v in res)

    def test_class_result_returns_four_tuple(self):
        p, r, ap50, ap5095 = self.dm.class_result(0)
        assert all(isinstance(v, float) for v in (p, r, ap50, ap5095))

    def test_class_result_perfect_values(self):
        p, r, ap50, ap5095 = self.dm.class_result(1)
        assert p == pytest.approx(1.0, abs=1e-4)
        assert r == pytest.approx(1.0, abs=1e-4)
        assert ap50 == pytest.approx(1.0, abs=1e-4)
        assert ap5095 == pytest.approx(1.0, abs=1e-4)

    def test_class_result_out_of_range_raises(self):
        with pytest.raises((IndexError, Exception)):
            self.dm.class_result(self.nc + 5)


# ---------------------------------------------------------------------------
# results_dict tests
# ---------------------------------------------------------------------------


class TestResultsDict:
    def setup_method(self):
        self.dm = _make_populated(nc=2, names={0: "a", 1: "b"})

    def test_has_exactly_five_keys(self):
        assert len(self.dm.results_dict) == 5

    def test_all_metric_keys_present(self):
        rd = self.dm.results_dict
        for k in self.dm.keys:
            assert k in rd

    def test_fitness_key_present(self):
        assert "fitness" in self.dm.results_dict

    def test_precision_key_value(self):
        assert self.dm.results_dict["metrics/precision(B)"] == pytest.approx(1.0, abs=1e-4)

    def test_recall_key_value(self):
        assert self.dm.results_dict["metrics/recall(B)"] == pytest.approx(1.0, abs=1e-4)

    def test_map50_key_value(self):
        assert self.dm.results_dict["metrics/mAP50(B)"] == pytest.approx(1.0, abs=1e-4)

    def test_map5095_key_value(self):
        assert self.dm.results_dict["metrics/mAP50-95(B)"] == pytest.approx(1.0, abs=1e-4)

    def test_fitness_value_formula(self):
        dm = self.dm
        expected = 0.1 * dm.box.map50 + 0.9 * dm.box.map
        assert dm.results_dict["fitness"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# summary() tests
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.names = {0: "cat", 1: "dog"}
        self.dm = _make_populated(nc=2, names=self.names)

    def test_summary_length_equals_nc(self):
        assert len(self.dm.summary()) == 2

    def test_summary_row_keys(self):
        required_keys = {
            "class_id",
            "class_name",
            "num_targets",
            "precision",
            "recall",
            "f1_score",
            "ap50",
            "ap50_95",
        }
        for row in self.dm.summary():
            assert required_keys.issubset(row.keys())

    def test_summary_class_names_resolved(self):
        rows = self.dm.summary()
        assert rows[0]["class_name"] == "cat"
        assert rows[1]["class_name"] == "dog"

    def test_summary_unknown_class_fallback_to_str(self):
        dm = _make_populated(nc=1, names={})  # no name mapping
        row = dm.summary()[0]
        assert isinstance(row["class_name"], str)

    def test_summary_precision_recall_one(self):
        for row in self.dm.summary():
            assert row["precision"] == pytest.approx(1.0, abs=1e-4)
            assert row["recall"] == pytest.approx(1.0, abs=1e-4)

    def test_summary_ap50_one(self):
        for row in self.dm.summary():
            assert row["ap50"] == pytest.approx(1.0, abs=1e-4)

    def test_summary_ap50_95_one(self):
        for row in self.dm.summary():
            assert row["ap50_95"] == pytest.approx(1.0, abs=1e-4)

    def test_summary_f1_score_correct(self):
        for row in self.dm.summary():
            p, r = row["precision"], row["recall"]
            expected_f1 = 2 * p * r / (p + r + 1e-16)
            assert row["f1_score"] == pytest.approx(expected_f1, abs=1e-4)

    def test_summary_empty_before_update(self):
        dm = DetMetrics(names={0: "cat"})
        assert dm.summary() == []


# ---------------------------------------------------------------------------
# to_json() tests
# ---------------------------------------------------------------------------


class TestToJson:
    def setup_method(self):
        self.dm = _make_populated(nc=2, names={0: "a", 1: "b"})

    def test_to_json_returns_string(self):
        assert isinstance(self.dm.to_json(), str)

    def test_to_json_valid_json(self):
        data = json.loads(self.dm.to_json())
        assert isinstance(data, dict)

    def test_to_json_has_results_key(self):
        data = json.loads(self.dm.to_json())
        assert "results" in data

    def test_to_json_has_per_class_key(self):
        data = json.loads(self.dm.to_json())
        assert "per_class" in data

    def test_to_json_has_speed_key(self):
        data = json.loads(self.dm.to_json())
        assert "speed" in data

    def test_to_json_per_class_length(self):
        data = json.loads(self.dm.to_json())
        assert len(data["per_class"]) == 2

    def test_to_json_empty_state(self):
        dm = DetMetrics(names={})
        out = json.loads(dm.to_json())
        assert out["per_class"] == []


# ---------------------------------------------------------------------------
# to_csv() tests
# ---------------------------------------------------------------------------


class TestToCsv:
    def setup_method(self):
        self.dm = _make_populated(nc=2, names={0: "a", 1: "b"})

    def test_to_csv_returns_string(self):
        assert isinstance(self.dm.to_csv(), str)

    def test_to_csv_has_header(self):
        lines = self.dm.to_csv().strip().splitlines()
        assert len(lines) >= 1
        header = lines[0].split(",")
        assert "class_id" in header
        assert "precision" in header
        assert "ap50" in header

    def test_to_csv_row_count(self):
        lines = [line for line in self.dm.to_csv().strip().splitlines() if line]
        # header + 2 data rows
        assert len(lines) == 3

    def test_to_csv_parseable_by_csv_reader(self):
        buf = io.StringIO(self.dm.to_csv())
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 2

    def test_to_csv_empty_state_header_only(self):
        dm = DetMetrics(names={})
        lines = [line for line in dm.to_csv().strip().splitlines() if line]
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# Edge-case / integration tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_class(self):
        dm = DetMetrics(names={0: "only"})
        dm.box.update(_perfect_ap_results(nc=1))
        assert dm.box.map == pytest.approx(1.0, abs=1e-4)
        assert len(dm.summary()) == 1

    def test_many_classes(self):
        nc = 80
        dm = DetMetrics(names={i: f"cls{i}" for i in range(nc)})
        dm.box.update(_perfect_ap_results(nc=nc))
        assert dm.maps.shape == (nc,)
        assert len(dm.summary()) == nc

    def test_speed_custom_values(self):
        dm = DetMetrics()
        dm.speed = {"preprocess": 1.5, "inference": 10.0, "postprocess": 2.3}
        j = json.loads(dm.to_json())
        assert j["speed"]["inference"] == 10.0

    def test_custom_confusion_matrix_attached(self):
        # ConfusionMatrix is a stub in E1, so just confirm field assignment works
        dm = DetMetrics(names={0: "x"})
        dm.confusion_matrix = object()  # type: ignore[assignment]
        assert dm.confusion_matrix is not None

    def test_results_dict_values_are_floats(self):
        dm = _make_populated(nc=2)
        for v in dm.results_dict.values():
            assert isinstance(v, float)

    def test_real_ap_per_class_integration(self):
        """Build TP/FP arrays from scratch and run the full pipeline."""
        n = 100
        rng = np.random.default_rng(42)
        conf = rng.uniform(0.3, 1.0, n).astype(np.float32)
        # 2 classes, alternating
        pred_cls = np.tile([0, 1], n // 2).astype(np.int32)
        target_cls = np.tile([0, 1], n // 2).astype(np.int32)
        # Perfect TP: every prediction matches
        tp = np.ones(n, dtype=bool)

        results = ap_per_class(tp, conf, pred_cls, target_cls)
        dm = DetMetrics(names={0: "cat", 1: "dog"})
        dm.box.update(results)

        assert dm.box.map > 0.0
        assert dm.box.map50 > 0.0
        assert len(dm.summary()) == 2
        rd = dm.results_dict
        assert len(rd) == 5
        assert rd["fitness"] == pytest.approx(0.1 * dm.box.map50 + 0.9 * dm.box.map, abs=1e-6)
