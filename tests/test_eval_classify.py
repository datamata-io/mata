"""Tests for ClassifyMetrics — Task G3.

Covers:
- Default construction
- process_predictions() accumulation (top-1, top-5)
- Running average correctness across multiple batches
- Fitness property
- mean_results() / class_result()
- results_dict keys and values
- summary() format
- to_json() / to_csv() serialisation
- Edge cases: empty batch, mismatched lengths, single class, large nc
- keys property
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from mata.eval.metrics.classify import ClassifyMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_metrics(nc: int = 5) -> ClassifyMetrics:
    """Return a ClassifyMetrics with only perfect top-1 predictions."""
    names = {i: f"cls{i}" for i in range(nc)}
    m = ClassifyMetrics(names=names, nc=nc)
    labels = list(range(nc))
    m.process_predictions(labels, labels)
    return m


def _top5_predictions(nc: int = 10) -> tuple[list[list[int]], list[int]]:
    """Return (pred_top5, target_labels) where each target is the 2nd item."""
    targets = list(range(nc))
    # top5[i] = [wrong, target, wrong, wrong, wrong]
    pred_top5 = [[((t + 1) % nc), t, ((t + 2) % nc), ((t + 3) % nc), ((t + 4) % nc)] for t in targets]
    return pred_top5, targets


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestClassifyMetricsConstruction:
    def test_default_construction(self):
        m = ClassifyMetrics()
        assert isinstance(m, ClassifyMetrics)

    def test_names_kwarg(self):
        m = ClassifyMetrics(names={0: "cat", 1: "dog"})
        assert m.names == {0: "cat", 1: "dog"}

    def test_nc_inferred_from_names(self):
        m = ClassifyMetrics(names={0: "a", 1: "b", 2: "c"})
        assert m.nc == 3

    def test_nc_explicit_overrides(self):
        m = ClassifyMetrics(names={0: "a"}, nc=10)
        assert m.nc == 10

    def test_default_top1_zero(self):
        m = ClassifyMetrics()
        assert m.top1 == 0.0

    def test_default_top5_zero(self):
        m = ClassifyMetrics()
        assert m.top5 == 0.0

    def test_default_speed_keys(self):
        m = ClassifyMetrics()
        assert set(m.speed.keys()) == {"preprocess", "inference", "postprocess"}

    def test_default_speed_zero(self):
        m = ClassifyMetrics()
        assert all(v == 0.0 for v in m.speed.values())

    def test_confusion_matrix_default_none(self):
        m = ClassifyMetrics()
        assert m.confusion_matrix is None

    def test_summary_empty_before_process(self):
        m = ClassifyMetrics()
        assert m.summary() == []


# ---------------------------------------------------------------------------
# process_predictions — top-1 accuracy
# ---------------------------------------------------------------------------


class TestProcessPredictionsTop1:
    def test_perfect_top1(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 2], [0, 1, 2])
        assert m.top1 == pytest.approx(1.0)

    def test_all_wrong_top1(self):
        m = ClassifyMetrics()
        m.process_predictions([1, 2, 0], [0, 1, 2])
        assert m.top1 == pytest.approx(0.0)

    def test_half_correct_top1(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 2, 2], [0, 1, 2])
        assert m.top1 == pytest.approx(2 / 3)

    def test_single_correct_prediction(self):
        m = ClassifyMetrics()
        m.process_predictions([0], [0])
        assert m.top1 == pytest.approx(1.0)

    def test_single_wrong_prediction(self):
        m = ClassifyMetrics()
        m.process_predictions([1], [0])
        assert m.top1 == pytest.approx(0.0)

    def test_top1_is_float(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1], [0, 1])
        assert isinstance(m.top1, float)


# ---------------------------------------------------------------------------
# process_predictions — top-5 accuracy
# ---------------------------------------------------------------------------


class TestProcessPredictionsTop5:
    def test_top5_perfect_when_target_in_top5(self):
        m = ClassifyMetrics()
        pred_top5 = [[1, 2, 0, 3, 4], [0, 1, 2, 3, 4]]  # target is in list
        m.process_predictions([1, 0], [0, 1], pred_top5=pred_top5)
        assert m.top5 == pytest.approx(1.0)

    def test_top5_zero_when_target_not_in_top5(self):
        m = ClassifyMetrics()
        pred_top5 = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]  # targets [0,1] absent
        m.process_predictions([1, 2], [0, 1], pred_top5=pred_top5)
        assert m.top5 == pytest.approx(0.0)

    def test_top5_respects_5_item_limit(self):
        m = ClassifyMetrics()
        # Target is at index 5 (6th element) — should NOT count
        pred_top5 = [[1, 2, 3, 4, 5, 0]]  # 0 is at index 5, out of top-5
        m.process_predictions([1], [0], pred_top5=pred_top5)
        assert m.top5 == pytest.approx(0.0)

    def test_top5_fallback_equals_top1_when_no_top5(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 2], [0, 1, 2], pred_top5=None)
        assert m.top5 == pytest.approx(m.top1)


# ---------------------------------------------------------------------------
# Running accumulation across multiple batches
# ---------------------------------------------------------------------------


class TestRunningAccumulation:
    def test_two_batches_averaged(self):
        m = ClassifyMetrics()
        # Batch 1: 2/2 correct
        m.process_predictions([0, 1], [0, 1])
        # Batch 2: 0/2 correct
        m.process_predictions([1, 0], [0, 1])
        # Overall: 2/4 correct = 0.50
        assert m.top1 == pytest.approx(0.5)

    def test_three_batches_running_average(self):
        m = ClassifyMetrics()
        m.process_predictions([0], [0])  # 1/1
        m.process_predictions([0], [1])  # 0/1
        m.process_predictions([0, 1], [0, 1])  # 2/2
        # Total: 3/4 correct = 0.75
        assert m.top1 == pytest.approx(0.75)

    def test_empty_batch_does_not_change_metrics(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1], [0, 1])  # top1 = 1.0
        m.process_predictions([], [])  # empty — should not change
        assert m.top1 == pytest.approx(1.0)

    def test_accumulate_n_total(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1], [0, 1])
        m.process_predictions([0, 1, 2], [0, 1, 2])
        assert m._n_total == 5


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


class TestFitness:
    def test_fitness_perfect(self):
        m = _perfect_metrics()
        assert m.fitness == pytest.approx(1.0)

    def test_fitness_formula(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 0, 1], [0, 1, 1, 0])  # top1 = 0.5
        expected = (m.top1 + m.top5) / 2.0
        assert m.fitness == pytest.approx(expected)

    def test_fitness_zero_before_process(self):
        m = ClassifyMetrics()
        assert m.fitness == pytest.approx(0.0)

    def test_fitness_is_float(self):
        m = _perfect_metrics()
        assert isinstance(m.fitness, float)


# ---------------------------------------------------------------------------
# mean_results / class_result
# ---------------------------------------------------------------------------


class TestMeanResultsAndClassResult:
    def test_mean_results_returns_two_items(self):
        m = _perfect_metrics()
        assert len(m.mean_results()) == 2

    def test_mean_results_perfect(self):
        m = _perfect_metrics()
        t1, t5 = m.mean_results()
        assert t1 == pytest.approx(1.0)
        assert t5 == pytest.approx(1.0)

    def test_mean_results_before_process_zeros(self):
        m = ClassifyMetrics()
        assert m.mean_results() == [0.0, 0.0]

    def test_class_result_returns_top1_top5(self):
        m = _perfect_metrics()
        t1, t5 = m.class_result(0)
        assert t1 == pytest.approx(1.0)
        assert t5 == pytest.approx(1.0)

    def test_class_result_same_for_all_indices(self):
        m = _perfect_metrics(nc=5)
        for i in range(5):
            assert m.class_result(i) == m.class_result(0)


# ---------------------------------------------------------------------------
# results_dict
# ---------------------------------------------------------------------------


class TestResultsDict:
    def test_results_dict_has_three_keys(self):
        m = _perfect_metrics()
        assert len(m.results_dict) == 3

    def test_results_dict_has_top1_key(self):
        m = _perfect_metrics()
        assert "metrics/accuracy_top1" in m.results_dict

    def test_results_dict_has_top5_key(self):
        m = _perfect_metrics()
        assert "metrics/accuracy_top5" in m.results_dict

    def test_results_dict_has_fitness_key(self):
        m = _perfect_metrics()
        assert "fitness" in m.results_dict

    def test_results_dict_top1_value(self):
        m = _perfect_metrics()
        assert m.results_dict["metrics/accuracy_top1"] == pytest.approx(1.0)

    def test_results_dict_top5_value(self):
        m = _perfect_metrics()
        assert m.results_dict["metrics/accuracy_top5"] == pytest.approx(1.0)

    def test_results_dict_fitness_matches_property(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 1], [0, 0, 1])
        assert m.results_dict["fitness"] == pytest.approx(m.fitness)

    def test_results_dict_values_are_floats(self):
        m = _perfect_metrics()
        for v in m.results_dict.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# keys property
# ---------------------------------------------------------------------------


class TestKeysProperty:
    def test_keys_returns_list(self):
        m = ClassifyMetrics()
        assert isinstance(m.keys, list)

    def test_keys_has_two_entries(self):
        m = ClassifyMetrics()
        assert len(m.keys) == 2

    def test_keys_values(self):
        m = ClassifyMetrics()
        assert m.keys == ["metrics/accuracy_top1", "metrics/accuracy_top5"]


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty_before_process(self):
        m = ClassifyMetrics()
        assert m.summary() == []

    def test_summary_single_element_list(self):
        m = _perfect_metrics()
        s = m.summary()
        assert len(s) == 1

    def test_summary_has_required_keys(self):
        m = _perfect_metrics()
        row = m.summary()[0]
        assert "top1_acc" in row
        assert "top5_acc" in row
        assert "n_samples" in row

    def test_summary_perfect_accuracy(self):
        m = _perfect_metrics(nc=3)
        row = m.summary()[0]
        assert row["top1_acc"] == pytest.approx(1.0)
        assert row["top5_acc"] == pytest.approx(1.0)

    def test_summary_n_samples_correct(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 2], [0, 1, 2])
        row = m.summary()[0]
        assert row["n_samples"] == 3

    def test_summary_n_samples_accumulates_across_batches(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1], [0, 1])
        m.process_predictions([2, 3], [2, 3])
        assert m.summary()[0]["n_samples"] == 4

    def test_summary_values_rounded_to_6dp(self):
        m = ClassifyMetrics()
        m.process_predictions([0, 1, 0], [0, 1, 2])
        row = m.summary()[0]
        # Verify rounding does not explode precision
        assert len(str(row["top1_acc"]).rstrip("0").split(".")[-1]) <= 7


# ---------------------------------------------------------------------------
# to_json()
# ---------------------------------------------------------------------------


class TestToJson:
    def test_returns_string(self):
        m = _perfect_metrics()
        assert isinstance(m.to_json(), str)

    def test_valid_json(self):
        m = _perfect_metrics()
        data = json.loads(m.to_json())
        assert isinstance(data, dict)

    def test_has_results_key(self):
        data = json.loads(_perfect_metrics().to_json())
        assert "results" in data

    def test_has_speed_key(self):
        data = json.loads(_perfect_metrics().to_json())
        assert "speed" in data

    def test_has_summary_key(self):
        data = json.loads(_perfect_metrics().to_json())
        assert "summary" in data

    def test_results_contains_top1(self):
        data = json.loads(_perfect_metrics().to_json())
        assert "metrics/accuracy_top1" in data["results"]

    def test_results_top1_value_correct(self):
        data = json.loads(_perfect_metrics().to_json())
        assert data["results"]["metrics/accuracy_top1"] == pytest.approx(1.0)

    def test_speed_values_serialised(self):
        m = _perfect_metrics()
        m.speed["inference"] = 7.3
        data = json.loads(m.to_json())
        assert data["speed"]["inference"] == pytest.approx(7.3)

    def test_empty_state_serialises(self):
        m = ClassifyMetrics()
        data = json.loads(m.to_json())
        assert "results" in data


# ---------------------------------------------------------------------------
# to_csv()
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_returns_string(self):
        m = _perfect_metrics()
        assert isinstance(m.to_csv(), str)

    def test_has_header_row(self):
        m = _perfect_metrics()
        lines = m.to_csv().strip().splitlines()
        assert len(lines) >= 1
        assert "top1_acc" in lines[0]

    def test_header_has_expected_columns(self):
        m = _perfect_metrics()
        header = m.to_csv().strip().splitlines()[0].split(",")
        assert "top1_acc" in header
        assert "top5_acc" in header
        assert "n_samples" in header

    def test_data_row_present_after_process(self):
        m = _perfect_metrics()
        lines = [line for line in m.to_csv().strip().splitlines() if line]
        assert len(lines) == 2  # header + 1 data row

    def test_empty_state_header_only(self):
        m = ClassifyMetrics()
        lines = [line for line in m.to_csv().strip().splitlines() if line]
        assert len(lines) == 1  # header only

    def test_csv_parseable(self):
        m = _perfect_metrics()
        buf = io.StringIO(m.to_csv())
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["top1_acc"]) == pytest.approx(1.0)

    def test_csv_round_trip_n_samples(self):
        m = ClassifyMetrics()
        m.process_predictions(list(range(7)), list(range(7)))
        buf = io.StringIO(m.to_csv())
        rows = list(csv.DictReader(buf))
        assert int(rows[0]["n_samples"]) == 7


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_mismatched_lengths_raises(self):
        m = ClassifyMetrics()
        with pytest.raises(ValueError):
            m.process_predictions([0, 1, 2], [0, 1])

    def test_mismatched_lengths_error_message_contains_lengths(self):
        m = ClassifyMetrics()
        with pytest.raises(ValueError, match="3"):
            m.process_predictions([0, 1, 2], [0, 1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_class_perfect(self):
        m = ClassifyMetrics(names={0: "only"}, nc=1)
        m.process_predictions([0, 0, 0], [0, 0, 0])
        assert m.top1 == pytest.approx(1.0)

    def test_many_classes(self):
        nc = 1000
        m = ClassifyMetrics(nc=nc)
        labels = list(range(10))
        m.process_predictions(labels, labels)
        assert m.top1 == pytest.approx(1.0)

    def test_nc_zero_and_no_names(self):
        # Should not crash on construction
        m = ClassifyMetrics(nc=0, names={})
        assert m.nc == 0

    def test_confusion_matrix_can_be_set(self):
        m = ClassifyMetrics()
        m.confusion_matrix = object()  # type: ignore[assignment]
        assert m.confusion_matrix is not None

    def test_speed_custom_values_preserved(self):
        m = _perfect_metrics()
        m.speed["preprocess"] = 2.5
        assert m.speed["preprocess"] == pytest.approx(2.5)

    def test_top5_when_nc_less_than_5(self):
        """For nc < 5, top5 with pred_top5=None should fallback to top1."""
        m = ClassifyMetrics(names={0: "a", 1: "b"}, nc=2)
        m.process_predictions([0, 1], [0, 1])
        assert m.top5 == pytest.approx(m.top1)

    def test_top5_partial_hit(self):
        m = ClassifyMetrics(nc=10)
        pred_top5 = [[1, 0, 2, 3, 4], [5, 6, 7, 8, 9]]  # only first is a hit
        m.process_predictions([1, 5], [0, 1], pred_top5=pred_top5)
        assert m.top5 == pytest.approx(0.5)
