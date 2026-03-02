"""Tests for Printer — YOLO-style console table output — Task G6.

All tests exercise the Printer class in isolation, capturing stdout with
pytest's capsys fixture.  No real models or datasets needed.
"""

from __future__ import annotations

import numpy as np

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.classify import ClassifyMetrics
from mata.eval.metrics.depth import DepthMetrics
from mata.eval.metrics.detect import DetMetrics
from mata.eval.metrics.segment import SegmentMetrics
from mata.eval.printer import _HEADER_CLASSIFY, _HEADER_DEPTH, _HEADER_DETECT, _HEADER_SEGMENT, Printer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric_from_ap_per_class(tp, conf, pred_cls, target_cls) -> Metric:
    """Build a populated Metric via ap_per_class."""
    result = ap_per_class(
        np.asarray(tp, dtype=bool),
        np.asarray(conf, dtype=np.float32),
        np.asarray(pred_cls, dtype=np.int32),
        np.asarray(target_cls, dtype=np.int32),
    )
    m = Metric()
    m.update(result)
    return m


def _simple_det_metrics(nc: int = 2) -> DetMetrics:
    """Return a DetMetrics with nc classes, all 0.5 AP."""
    n = 20
    tp = np.ones((n, 10), dtype=bool)
    conf = np.full(n, 0.9, dtype=np.float32)
    pred_cls = np.tile(np.arange(nc), n // nc + 1)[:n].astype(np.int32)
    target_cls = np.tile(np.arange(nc), n // nc + 1)[:n].astype(np.int32)

    m = DetMetrics(names={i: f"cls{i}" for i in range(nc)})
    m.box = _metric_from_ap_per_class(tp, conf, pred_cls, target_cls)
    return m


def _simple_seg_metrics(nc: int = 2) -> SegmentMetrics:
    n = 20
    tp = np.ones((n, 10), dtype=bool)
    conf = np.full(n, 0.9, dtype=np.float32)
    pred_cls = np.tile(np.arange(nc), n // nc + 1)[:n].astype(np.int32)
    target_cls = np.tile(np.arange(nc), n // nc + 1)[:n].astype(np.int32)

    m = SegmentMetrics(names={i: f"cls{i}" for i in range(nc)})
    m.box = _metric_from_ap_per_class(tp, conf, pred_cls, target_cls)
    m.seg = _metric_from_ap_per_class(tp, conf, pred_cls, target_cls)
    return m


def _classify_metrics(top1: float = 0.75, top5: float = 0.95) -> ClassifyMetrics:
    m = ClassifyMetrics(names={0: "cat", 1: "dog"})
    m.top1 = top1
    m.top5 = top5
    return m


def _depth_metrics() -> DepthMetrics:
    m = DepthMetrics()
    m.abs_rel = 0.123
    m.sq_rel = 0.456
    m.rmse = 1.234
    m.log_rmse = 0.089
    m.delta_1 = 0.876
    m.delta_2 = 0.950
    m.delta_3 = 0.980
    return m


# ---------------------------------------------------------------------------
# Header rows
# ---------------------------------------------------------------------------


class TestPrinterHeaders:
    def test_detect_header_correct(self, capsys):
        p = Printer(task="detect")
        p.print_header()
        out = capsys.readouterr().out
        for col in _HEADER_DETECT:
            assert col in out

    def test_segment_header_correct(self, capsys):
        p = Printer(task="segment")
        p.print_header()
        out = capsys.readouterr().out
        for col in _HEADER_SEGMENT:
            assert col in out

    def test_classify_header_correct(self, capsys):
        p = Printer(task="classify")
        p.print_header()
        out = capsys.readouterr().out
        for col in _HEADER_CLASSIFY:
            assert col in out

    def test_depth_header_correct(self, capsys):
        p = Printer(task="depth")
        p.print_header()
        out = capsys.readouterr().out
        for col in _HEADER_DEPTH:
            assert col in out

    def test_header_ends_with_newline(self, capsys):
        p = Printer(task="detect")
        p.print_header()
        out = capsys.readouterr().out
        assert out.endswith("\n")


# ---------------------------------------------------------------------------
# Numeric formatting (3 decimal places)
# ---------------------------------------------------------------------------


class TestPrinterNumericFormatting:
    def test_detect_all_row_three_decimal_places(self, capsys):
        p = Printer(task="detect", names={0: "cat", 1: "dog"})
        m = _simple_det_metrics(nc=2)
        p.print_table(m)
        out = capsys.readouterr().out
        # At least one value formatted to 3dp should appear
        import re

        matches = re.findall(r"\d+\.\d{3}", out)
        assert len(matches) >= 4, f"Expected 3dp numbers in:\n{out}"

    def test_classify_row_three_decimal_places(self, capsys):
        p = Printer(task="classify")
        m = _classify_metrics(top1=0.756789, top5=0.951234)
        p.print_table(m)
        out = capsys.readouterr().out
        # 0.757 and 0.951 should appear (3dp truncation/rounding)
        import re

        matches = re.findall(r"\d+\.\d{3}", out)
        assert len(matches) >= 2

    def test_depth_row_three_decimal_places(self, capsys):
        p = Printer(task="depth")
        m = _depth_metrics()
        p.print_table(m)
        out = capsys.readouterr().out
        assert "0.123" in out  # abs_rel
        assert "0.876" in out  # delta_1


# ---------------------------------------------------------------------------
# Speed line format
# ---------------------------------------------------------------------------


class TestPrinterSpeedLine:
    def test_speed_line_format(self, capsys):
        p = Printer(task="detect")
        p.print_speed({"preprocess": 0.5, "inference": 3.2, "postprocess": 1.1})
        out = capsys.readouterr().out
        assert out.startswith("Speed:")
        assert "per image" in out
        assert "0.5ms preprocess" in out
        assert "3.2ms inference" in out
        assert "1.1ms postprocess" in out

    def test_speed_line_with_shape(self, capsys):
        p = Printer(task="detect")
        p.print_speed({"inference": 2.0}, image_shape=(640, 640))
        out = capsys.readouterr().out
        assert "640" in out
        assert "shape" in out

    def test_speed_line_ends_with_newline(self, capsys):
        p = Printer(task="detect")
        p.print_speed({"inference": 1.0})
        out = capsys.readouterr().out
        assert out.endswith("\n")


# ---------------------------------------------------------------------------
# Zero-GT class suppression
# ---------------------------------------------------------------------------


class TestPrinterZeroGTSuppression:
    def test_zero_gt_class_suppressed(self, capsys):
        p = Printer(task="detect", names={0: "cat", 1: "dog"})
        m = _simple_det_metrics(nc=2)
        # class 1 ("dog") has 0 GT instances → should be suppressed
        nc = {0: 10, 1: 0}
        p.print_table(m, nc=nc)
        out = capsys.readouterr().out
        assert "dog" not in out

    def test_nonzero_gt_class_present(self, capsys):
        p = Printer(task="detect", names={0: "cat", 1: "dog"})
        m = _simple_det_metrics(nc=2)
        nc = {0: 10, 1: 5}
        p.print_table(m, nc=nc)
        out = capsys.readouterr().out
        assert "cat" in out
        assert "dog" in out

    def test_all_row_always_present(self, capsys):
        p = Printer(task="detect", names={0: "cat"})
        m = _simple_det_metrics(nc=1)
        nc = {0: 0}
        p.print_table(m, nc=nc)
        out = capsys.readouterr().out
        assert "all" in out


# ---------------------------------------------------------------------------
# Single-class case
# ---------------------------------------------------------------------------


class TestPrinterSingleClass:
    def test_single_class_only_all_row(self, capsys):
        """With a single class, only the 'all' summary row should appear."""
        p = Printer(task="detect", names={0: "cat"})
        m = _simple_det_metrics(nc=1)
        p.print_table(m)
        out = capsys.readouterr().out
        lines = [line for line in out.strip().splitlines() if line.strip()]
        # header + all row only; no per-class row for single-class
        class_rows = [line for line in lines if "cat" in line]
        assert len(class_rows) == 0  # per-class row suppressed

    def test_multi_class_has_per_class_rows(self, capsys):
        p = Printer(task="detect", names={0: "cat", 1: "dog"})
        m = _simple_det_metrics(nc=2)
        p.print_table(m, nc={0: 5, 1: 5})
        out = capsys.readouterr().out
        assert "cat" in out
        assert "dog" in out


# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------


class TestPrinterTaskRouting:
    def test_classify_no_class_instances_columns(self, capsys):
        p = Printer(task="classify")
        m = _classify_metrics()
        p.print_table(m)
        out = capsys.readouterr().out
        # classify table should NOT have "Images" or "Instances" columns
        assert "Images" not in out
        assert "Instances" not in out

    def test_depth_no_class_instances_columns(self, capsys):
        p = Printer(task="depth")
        m = _depth_metrics()
        p.print_table(m)
        out = capsys.readouterr().out
        assert "Images" not in out
        assert "Instances" not in out

    def test_detect_has_box_columns(self, capsys):
        p = Printer(task="detect", names={0: "cat"})
        _simple_det_metrics(nc=1)
        p.print_header()
        out = capsys.readouterr().out
        assert "Box(P" in out

    def test_segment_has_both_box_and_mask_columns(self, capsys):
        p = Printer(task="segment", names={0: "cat"})
        p.print_header()
        out = capsys.readouterr().out
        assert "Box(P" in out
        assert "Mask(P" in out


# ---------------------------------------------------------------------------
# print_results convenience method
# ---------------------------------------------------------------------------


class TestPrinterPrintResults:
    def test_print_results_includes_speed_when_present(self, capsys):
        p = Printer(task="classify")
        m = _classify_metrics()
        m.speed = {"preprocess": 0.1, "inference": 2.5, "postprocess": 0.3}
        p.print_results(m)
        out = capsys.readouterr().out
        assert "Speed:" in out

    def test_print_results_no_speed_when_empty(self, capsys):
        p = Printer(task="classify")
        m = _classify_metrics()
        m.speed = {}
        p.print_results(m)
        out = capsys.readouterr().out
        assert "Speed:" not in out
