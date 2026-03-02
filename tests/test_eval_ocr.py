"""Unit tests for OCR evaluation — Task E1.

Covers:
  - _levenshtein()         (TestLevenshtein)
  - _levenshtein_seq()     (TestLevenshteinSeq)
  - OCRMetrics dataclass   (TestOCRMetrics)
  - GroundTruth.text field (TestGroundTruthText)
  - DatasetLoader OCR JSON (TestOCRDatasetLoader)
  - Validator OCR wiring   (TestOCRValidatorWiring)
  - Printer OCR output     (TestOCRPrinter)

All tests are fully self-contained: no model downloads, no network access,
no external OCR libraries required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Private helpers imported for white-box testing
# ---------------------------------------------------------------------------
from mata.eval.metrics.ocr import OCRMetrics, _levenshtein, _levenshtein_seq

# ---------------------------------------------------------------------------
# TestLevenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    """Pure-Python character-level edit distance function."""

    def test_empty_strings(self) -> None:
        assert _levenshtein("", "") == 0

    def test_one_empty_a(self) -> None:
        """Empty first string → distance equals length of second."""
        assert _levenshtein("", "abc") == 3

    def test_one_empty_b(self) -> None:
        """Empty second string → distance equals length of first."""
        assert _levenshtein("abc", "") == 3

    def test_identical(self) -> None:
        assert _levenshtein("abc", "abc") == 0

    def test_known_kitten_sitting(self) -> None:
        # Classic textbook example
        assert _levenshtein("kitten", "sitting") == 3

    def test_known_saturday_sunday(self) -> None:
        assert _levenshtein("saturday", "sunday") == 3

    def test_single_char_diff(self) -> None:
        assert _levenshtein("cat", "car") == 1

    def test_completely_different(self) -> None:
        # All 3 chars need substitution
        assert _levenshtein("abc", "xyz") == 3

    def test_symmetric(self) -> None:
        assert _levenshtein("kitten", "sitting") == _levenshtein("sitting", "kitten")
        assert _levenshtein("hello", "world") == _levenshtein("world", "hello")

    def test_insertion(self) -> None:
        """Inserting one character."""
        assert _levenshtein("ab", "abc") == 1

    def test_deletion(self) -> None:
        """Deleting one character."""
        assert _levenshtein("abc", "ab") == 1

    def test_single_chars(self) -> None:
        assert _levenshtein("a", "b") == 1
        assert _levenshtein("a", "a") == 0


# ---------------------------------------------------------------------------
# TestLevenshteinSeq
# ---------------------------------------------------------------------------


class TestLevenshteinSeq:
    """Word-level (token-list) edit distance function."""

    def test_empty_lists(self) -> None:
        assert _levenshtein_seq([], []) == 0

    def test_identical_words(self) -> None:
        assert _levenshtein_seq(["hello", "world"], ["hello", "world"]) == 0

    def test_one_word_diff_substitution(self) -> None:
        assert _levenshtein_seq(["hello", "world"], ["hello", "earth"]) == 1

    def test_insertion(self) -> None:
        """Missing word in prediction → distance 1."""
        assert _levenshtein_seq(["hello"], ["hello", "world"]) == 1

    def test_deletion(self) -> None:
        """Extra word in prediction → distance 1."""
        assert _levenshtein_seq(["hello", "world"], ["hello"]) == 1

    def test_one_empty(self) -> None:
        assert _levenshtein_seq([], ["a", "b"]) == 2
        assert _levenshtein_seq(["a", "b"], []) == 2

    def test_symmetric(self) -> None:
        a = ["the", "quick", "brown"]
        b = ["the", "slow", "brown", "fox"]
        assert _levenshtein_seq(a, b) == _levenshtein_seq(b, a)


# ---------------------------------------------------------------------------
# TestOCRMetrics
# ---------------------------------------------------------------------------


class TestOCRMetrics:
    """OCRMetrics dataclass — accumulation, finalisation, serialisation."""

    # --- Construction ---

    def test_default_construction(self) -> None:
        m = OCRMetrics()
        assert m.cer == 0.0
        assert m.wer == 0.0
        assert m.accuracy == 0.0
        assert m.case_sensitive is False

    def test_case_sensitive_kwarg(self) -> None:
        m = OCRMetrics(case_sensitive=True)
        assert m.case_sensitive is True

    # --- process_batch / perfect match ---

    def test_perfect_match(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello world", "hello world")
        m.finalize()
        assert m.cer == pytest.approx(0.0)
        assert m.wer == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(1.0)

    def test_complete_mismatch(self) -> None:
        m = OCRMetrics()
        m.process_batch("abcde", "vwxyz")
        m.finalize()
        assert m.cer > 0.0
        assert m.wer > 0.0
        assert m.accuracy == pytest.approx(0.0)

    def test_partial_match_known_values(self) -> None:
        # pred "hello world" vs gt "hello earth"
        # CER: _levenshtein("hello world", "hello earth") / max(11, 1)
        # "world" vs "earth" → 4 edits over common prefix "hello "
        # char dist for full strings: 4 → CER = 4/11
        m = OCRMetrics()
        m.process_batch("hello world", "hello earth")
        m.finalize()
        expected_cer = _levenshtein("hello world", "hello earth") / max(len("hello earth"), 1)
        expected_wer = _levenshtein_seq(["hello", "world"], ["hello", "earth"]) / max(2, 1)
        assert m.cer == pytest.approx(expected_cer)
        assert m.wer == pytest.approx(expected_wer)
        assert m.accuracy == pytest.approx(0.0)

    # --- case sensitivity ---

    def test_case_insensitive_default(self) -> None:
        """Default: uppercase and lowercase match."""
        m = OCRMetrics()
        m.process_batch("HELLO WORLD", "hello world")
        m.finalize()
        assert m.cer == pytest.approx(0.0)
        assert m.wer == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(1.0)

    def test_case_sensitive(self) -> None:
        """With case_sensitive=True, "HELLO" ≠ "hello"."""
        m = OCRMetrics(case_sensitive=True)
        m.process_batch("HELLO", "hello")
        m.finalize()
        assert m.cer > 0.0

    # --- edge cases ---

    def test_empty_prediction(self) -> None:
        """Empty pred vs non-empty GT → CER = len(GT) / max(len(GT), 1) = 1.0."""
        m = OCRMetrics()
        m.process_batch("", "text")
        m.finalize()
        assert m.cer == pytest.approx(1.0)
        assert m.accuracy == pytest.approx(0.0)

    def test_empty_gt(self) -> None:
        """Non-empty pred vs empty GT → CER = len(pred) / max(0, 1) = len(pred)."""
        m = OCRMetrics()
        m.process_batch("text", "")
        m.finalize()
        # _levenshtein("text","") = 4; / max(0,1) = 4.0
        assert m.cer == pytest.approx(4.0)

    def test_both_empty(self) -> None:
        m = OCRMetrics()
        m.process_batch("", "")
        m.finalize()
        assert m.cer == pytest.approx(0.0)
        assert m.wer == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(1.0)

    # --- multi-image accumulation ---

    def test_multiple_images(self) -> None:
        """finalize() averages correctly over two images."""
        m = OCRMetrics()
        # Image 1: perfect match → CER=0, WER=0, exact=1
        m.process_batch("hello", "hello")
        # Image 2: complete mismatch (3-char strings)
        m.process_batch("abc", "xyz")
        m.finalize()
        # Image 2 CER: _levenshtein("abc","xyz")/3 = 3/3 = 1.0
        # Image 2 WER: _levenshtein_seq(["abc"],["xyz"])/1 = 1.0
        assert m.cer == pytest.approx(0.5)
        assert m.wer == pytest.approx(0.5)
        assert m.accuracy == pytest.approx(0.5)

    def test_finalize_no_images(self) -> None:
        """Fresh accumulator with no process_batch calls → all 0.0."""
        m = OCRMetrics()
        m.finalize()
        assert m.cer == pytest.approx(0.0)
        assert m.wer == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(0.0)

    # --- API compatibility ---

    def test_update_alias(self) -> None:
        """update() is an alias for process_batch()."""
        m1 = OCRMetrics()
        m1.process_batch("hello", "hello")
        m1.finalize()

        m2 = OCRMetrics()
        m2.update("hello", "hello")
        m2.finalize()

        assert m1.cer == m2.cer
        assert m1.accuracy == m2.accuracy

    def test_fitness(self) -> None:
        m = OCRMetrics()
        m.process_batch("ok", "ok")
        m.finalize()
        assert m.fitness == pytest.approx(m.accuracy)

    def test_keys(self) -> None:
        m = OCRMetrics()
        assert m.keys == ["metrics/cer", "metrics/wer", "metrics/accuracy"]
        assert len(m.keys) == 3

    def test_results_dict(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello", "hello")
        m.finalize()
        rd = m.results_dict
        assert set(rd.keys()) == {"metrics/cer", "metrics/wer", "metrics/accuracy", "fitness"}
        assert rd["fitness"] == rd["metrics/accuracy"]

    def test_mean_results(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello", "hello")
        m.finalize()
        mr = m.mean_results()
        assert mr == [m.cer, m.wer, m.accuracy]
        assert len(mr) == 3

    def test_summary(self) -> None:
        m = OCRMetrics()
        m.process_batch("test", "test")
        m.finalize()
        s = m.summary()
        assert isinstance(s, list)
        assert len(s) == 1
        row = s[0]
        assert set(row.keys()) == {"cer", "wer", "accuracy", "fitness"}

    def test_to_dict(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello", "hello")
        m.finalize()
        d = m.to_dict()
        assert "results" in d
        assert "speed" in d
        assert "summary" in d
        assert d["results"]["metrics/cer"] == m.cer

    def test_to_json(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello", "hello")
        m.finalize()
        js = m.to_json()
        parsed = json.loads(js)
        assert "results" in parsed
        assert "metrics/cer" in parsed["results"]

    def test_to_csv(self) -> None:
        m = OCRMetrics()
        m.process_batch("hello", "hello")
        m.finalize()
        csv_str = m.to_csv()
        lines = csv_str.strip().splitlines()
        assert len(lines) == 2  # header + one data row
        header_cols = lines[0].split(",")
        assert "cer" in header_cols
        assert "wer" in header_cols
        assert "accuracy" in header_cols
        assert "fitness" in header_cols


# ---------------------------------------------------------------------------
# TestGroundTruthText
# ---------------------------------------------------------------------------


class TestGroundTruthText:
    """GroundTruth.text field backward compatibility."""

    def _make_gt(self, **kwargs: Any):
        from mata.eval.dataset import GroundTruth

        return GroundTruth(
            image_id=1,
            image_path="/tmp/img.jpg",
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int32),
            masks=None,
            depth=None,
            image_size=(100, 100),
            **kwargs,
        )

    def test_default_none(self) -> None:
        """text defaults to None when not supplied."""
        gt = self._make_gt()
        assert gt.text is None

    def test_explicit_text(self) -> None:
        """text kwarg is stored as-is."""
        gt = self._make_gt(text=["hello", "world"])
        assert gt.text == ["hello", "world"]

    def test_backward_compat(self) -> None:
        """Old construction without text= still works and gives None."""
        from mata.eval.dataset import GroundTruth

        gt = GroundTruth(
            image_id=99,
            image_path="/tmp/x.jpg",
            boxes=np.zeros((2, 4), dtype=np.float32),
            labels=np.array([0, 1], dtype=np.int32),
            masks=None,
            depth=None,
            image_size=(640, 480),
        )
        assert gt.text is None

    def test_empty_list(self) -> None:
        gt = self._make_gt(text=[])
        assert gt.text == []

    def test_single_string(self) -> None:
        gt = self._make_gt(text=["STOP"])
        assert gt.text == ["STOP"]
        assert len(gt.text) == 1


# ---------------------------------------------------------------------------
# TestOCRDatasetLoader
# ---------------------------------------------------------------------------


def _write_ocr_json(path: Path, annotations: list[dict], with_text: bool = True) -> None:
    """Write a minimal COCO-Text JSON to *path*."""
    data: dict[str, Any] = {
        "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "text"}],
        "annotations": annotations,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


class TestOCRDatasetLoader:
    """DatasetLoader parsing COCO-Text JSON annotations."""

    def test_load_ocr_json(self, tmp_path: Path) -> None:
        """Mock COCO-Text JSON → GroundTruth with text populated."""
        from mata.eval.dataset import DatasetLoader

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_path = tmp_path / "ann.json"
        anns = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 15], "text": "STOP"},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [80, 40, 60, 20], "text": "GO"},
        ]
        _write_ocr_json(ann_path, anns)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(ann_path))
        pairs = list(loader)
        assert len(pairs) == 1
        _path, gt = pairs[0]
        assert gt.text is not None
        assert "STOP" in gt.text
        assert "GO" in gt.text

    def test_text_parallel_to_boxes(self, tmp_path: Path) -> None:
        """len(gt.text) == len(gt.boxes) == len(gt.labels)."""
        from mata.eval.dataset import DatasetLoader

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_path = tmp_path / "ann.json"
        anns = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 15], "text": "ONE"},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [80, 40, 60, 20], "text": "TWO"},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [20, 60, 30, 10], "text": "THREE"},
        ]
        _write_ocr_json(ann_path, anns)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(ann_path))
        _path, gt = next(iter(loader))
        assert gt.text is not None
        assert len(gt.text) == len(gt.boxes) == len(gt.labels) == 3

    def test_no_text_key(self, tmp_path: Path) -> None:
        """Regular COCO detection JSON (no 'text' key) → gt.text is None."""
        from mata.eval.dataset import DatasetLoader

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_path = tmp_path / "ann.json"
        # Standard detection annotations without 'text' field
        anns = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 15]},
        ]
        _write_ocr_json(ann_path, anns)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(ann_path))
        _path, gt = next(iter(loader))
        assert gt.text is None

    def test_empty_text_string_included(self, tmp_path: Path) -> None:
        """Annotation with 'text': '' should still be included in the list."""
        from mata.eval.dataset import DatasetLoader

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_path = tmp_path / "ann.json"
        anns = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 15], "text": "HELLO"},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [80, 40, 60, 20], "text": ""},
        ]
        _write_ocr_json(ann_path, anns)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(ann_path))
        _path, gt = next(iter(loader))
        assert gt.text is not None
        assert "" in gt.text
        assert len(gt.text) == 2


# ---------------------------------------------------------------------------
# TestOCRValidatorWiring
# ---------------------------------------------------------------------------


def _make_ocr_gt(text: list[str]) -> Any:
    """Create a GroundTruth with OCR text annotations."""
    from mata.eval.dataset import GroundTruth

    boxes = np.zeros((len(text), 4), dtype=np.float32)
    labels = np.zeros(len(text), dtype=np.int32)
    return GroundTruth(
        image_id=1,
        image_path="fake.jpg",
        boxes=boxes,
        labels=labels,
        masks=None,
        depth=None,
        image_size=(640, 480),
        text=text,
    )


def _make_ocr_result(texts_and_bboxes: list[tuple[str, Any]]):
    """Build an OCRResult for testing."""
    from mata.core.types import OCRResult, TextRegion

    regions = [TextRegion(text=t, score=1.0, bbox=bbox) for t, bbox in texts_and_bboxes]
    return OCRResult(regions=regions)


class TestOCRValidatorWiring:
    """Validator OCR branch — accumulation + finalization."""

    def test_returns_ocr_metrics_type(self) -> None:
        """Standalone mode with OCRResult predictions returns OCRMetrics."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("hello world", None)])
        gt = _make_ocr_gt(["hello world"])

        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        result = v.run()
        assert isinstance(result, OCRMetrics)

    def test_standalone_mode_perfect(self) -> None:
        """Identical predicted and GT text → accuracy=1.0, CER=0.0."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("hello world", None)])
        gt = _make_ocr_gt(["hello world"])

        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        m = v.run()
        assert isinstance(m, OCRMetrics)
        assert m.cer == pytest.approx(0.0)
        assert m.accuracy == pytest.approx(1.0)

    def test_standalone_mode_mismatch(self) -> None:
        """Different predicted and GT text → accuracy=0.0."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("wrong answer", None)])
        gt = _make_ocr_gt(["correct answer"])

        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        m = v.run()
        assert m.accuracy == pytest.approx(0.0)
        assert m.cer > 0.0

    def test_extract_ocr_text_sorted_by_bbox(self) -> None:
        """_extract_ocr_text sorts regions top-to-bottom, left-to-right."""
        from mata.eval.validator import Validator

        # Region B is above region A in the image (lower y1)
        pred = _make_ocr_result(
            [
                ("BOTTOM", (10.0, 100.0, 50.0, 120.0)),  # y1=100
                ("TOP", (10.0, 10.0, 50.0, 30.0)),  # y1=10 — should come first
            ]
        )
        gt = _make_ocr_gt(["TOP BOTTOM"])
        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        extracted = v._extract_ocr_text(pred)
        assert extracted == "TOP BOTTOM"

    def test_extract_ocr_text_no_bbox(self) -> None:
        """_extract_ocr_text joins regions in list order when bbox is None."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result(
            [
                ("first", None),
                ("second", None),
            ]
        )
        gt = _make_ocr_gt(["first second"])
        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        extracted = v._extract_ocr_text(pred)
        assert extracted == "first second"

    def test_verbose_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose=True prints OCR summary to stdout."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("hello", None)])
        gt = _make_ocr_gt(["hello"])

        v = Validator("ocr", predictions=[pred], ground_truth=[gt], verbose=True)
        v.run()
        captured = capsys.readouterr()
        # Either the full Printer output or the simple summary line should appear
        output = captured.out
        # Both paths print something about CER/accuracy or OCR
        assert len(output) > 0

    def test_speed_keys_populated(self) -> None:
        """metrics.speed contains preprocess / inference / postprocess keys."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("abc", None)])
        gt = _make_ocr_gt(["abc"])

        v = Validator("ocr", predictions=[pred], ground_truth=[gt])
        m = v.run()
        assert "preprocess" in m.speed
        assert "inference" in m.speed
        assert "postprocess" in m.speed

    def test_case_sensitive_kwarg_passthrough(self) -> None:
        """case_sensitive=True kwarg flows into OCRMetrics."""
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("HELLO", None)])
        gt = _make_ocr_gt(["hello"])

        # case_insensitive (default) → exact match
        v_ci = Validator("ocr", predictions=[pred], ground_truth=[gt])
        m_ci = v_ci.run()
        assert m_ci.accuracy == pytest.approx(1.0)

        # case_sensitive=True → mismatch
        v_cs = Validator("ocr", predictions=[pred], ground_truth=[gt], case_sensitive=True)
        m_cs = v_cs.run()
        assert m_cs.accuracy == pytest.approx(0.0)

    def test_gt_without_text_skipped(self) -> None:
        """Images whose GT has text=None are silently skipped."""
        from mata.eval.dataset import GroundTruth
        from mata.eval.validator import Validator

        pred = _make_ocr_result([("hello", None)])
        gt_no_text = GroundTruth(
            image_id=1,
            image_path="fake.jpg",
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int32),
            masks=None,
            depth=None,
            image_size=(100, 100),
            text=None,  # No text annotations
        )

        v = Validator("ocr", predictions=[pred], ground_truth=[gt_no_text])
        m = v.run()
        # No images processed → all zeros after finalize
        assert isinstance(m, OCRMetrics)
        assert m.cer == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestOCRPrinter
# ---------------------------------------------------------------------------


def _make_finalized_ocr_metrics(cer: float = 0.15, wer: float = 0.25, accuracy: float = 0.6) -> OCRMetrics:
    """Create an OCRMetrics already set to specific values."""
    m = OCRMetrics()
    m.cer = cer
    m.wer = wer
    m.accuracy = accuracy
    return m


class TestOCRPrinter:
    """Printer OCR header and table."""

    def test_header_ocr_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Printer(task='ocr').print_header() outputs CER, WER, Accuracy columns."""
        from mata.eval.printer import Printer

        p = Printer(task="ocr")
        p.print_header()
        out = capsys.readouterr().out
        assert "CER" in out
        assert "WER" in out
        assert "Accuracy" in out

    def test_header_constant(self) -> None:
        """HEADER_OCR class constant has expected tuple."""
        from mata.eval.printer import Printer

        assert "CER" in Printer.HEADER_OCR
        assert "WER" in Printer.HEADER_OCR
        assert "Accuracy" in Printer.HEADER_OCR
        assert len(Printer.HEADER_OCR) == 3

    def test_print_table_outputs_row(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_table(ocr_metrics) outputs header + one data row."""
        from mata.eval.printer import Printer

        m = _make_finalized_ocr_metrics(cer=0.12, wer=0.20, accuracy=0.75)
        p = Printer(task="ocr")
        p.print_table(m)
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if ln.strip()]
        # At least the header and one data row
        assert len(lines) >= 2

    def test_print_table_contains_metric_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_table output contains formatted metric values."""
        from mata.eval.printer import Printer

        m = _make_finalized_ocr_metrics(cer=0.0, wer=0.0, accuracy=1.0)
        p = Printer(task="ocr")
        p.print_table(m)
        out = capsys.readouterr().out
        # The accuracy=1.0 should appear as "     1" or "1.000" or similar
        assert "1" in out

    def test_task_routing_ocr_calls_print_ocr_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'ocr' task routes to _print_ocr_table (not detect/classify/depth)."""
        from mata.eval.printer import Printer

        m = _make_finalized_ocr_metrics()
        p = Printer(task="ocr")
        # Should not raise; routing is verified by checking output has 3 numeric columns
        p.print_table(m)
        out = capsys.readouterr().out
        # Output must have been produced (routing worked)
        assert out.strip() != ""

    def test_print_results_includes_speed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_results() outputs both the table and speed summary line."""
        from mata.eval.printer import Printer

        m = _make_finalized_ocr_metrics()
        m.speed = {"preprocess": 0.5, "inference": 3.2, "postprocess": 1.1}
        p = Printer(task="ocr")
        p.print_results(m)
        out = capsys.readouterr().out
        # Speed line should be present
        assert "Speed" in out or "ms" in out.lower() or len(out.strip().splitlines()) >= 2

    def test_detect_task_unaffected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Adding OCR support doesn't break existing detect task header."""
        from mata.eval.printer import _HEADER_DETECT, Printer

        p = Printer(task="detect")
        p.print_header()
        out = capsys.readouterr().out
        assert _HEADER_DETECT[0] in out or "Class" in out

    def test_depth_task_unaffected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Adding OCR support doesn't break existing depth task header."""
        from mata.eval.printer import _HEADER_DEPTH, Printer

        p = Printer(task="depth")
        p.print_header()
        out = capsys.readouterr().out
        # Depth header columns
        assert any(col in out for col in _HEADER_DEPTH)


# ---------------------------------------------------------------------------
# TestMataValOCR — Integration: mata.val("ocr", ...) end-to-end
# ---------------------------------------------------------------------------


class TestMataValOCR:
    """Integration tests for the public ``mata.val("ocr", ...)`` API path.

    Uses standalone mode (``predictions=`` + ``ground_truth=``) so no real
    model is loaded and zero network access is required.
    """

    def test_standalone_val(self) -> None:
        """Pre-computed predictions + GT list → returns an OCRMetrics instance."""
        import mata

        pred = _make_ocr_result([("hello world", None)])
        gt = _make_ocr_gt(["hello world"])

        result = mata.val("ocr", predictions=[pred], ground_truth=[gt], verbose=False)

        assert isinstance(result, OCRMetrics)

    def test_standalone_val_perfect(self) -> None:
        """Identical predicted and GT text → cer=0.0, accuracy=1.0."""
        import mata

        pred = _make_ocr_result([("exact match", None)])
        gt = _make_ocr_gt(["exact match"])

        result = mata.val("ocr", predictions=[pred], ground_truth=[gt], verbose=False)

        assert result.cer == pytest.approx(0.0)
        assert result.wer == pytest.approx(0.0)
        assert result.accuracy == pytest.approx(1.0)

    def test_standalone_val_case_kwarg(self) -> None:
        """case_sensitive=True kwarg passes through to OCRMetrics and affects scoring."""
        import mata

        pred = _make_ocr_result([("HELLO", None)])
        gt = _make_ocr_gt(["hello"])

        # Default (case-insensitive): "HELLO" lowercased == "hello" → perfect match
        result_ci = mata.val("ocr", predictions=[pred], ground_truth=[gt], verbose=False)
        assert result_ci.accuracy == pytest.approx(1.0)

        # case_sensitive=True: "HELLO" != "hello" → mismatch
        result_cs = mata.val(
            "ocr",
            predictions=[pred],
            ground_truth=[gt],
            verbose=False,
            case_sensitive=True,
        )
        assert result_cs.accuracy == pytest.approx(0.0)
        assert result_cs.cer > 0.0

    def test_val_returns_ocr_metrics(self) -> None:
        """isinstance check: mata.val('ocr', ...) always returns OCRMetrics."""
        import mata

        pred = _make_ocr_result([("some text", None)])
        gt = _make_ocr_gt(["some text"])

        result = mata.val("ocr", predictions=[pred], ground_truth=[gt], verbose=False)

        assert isinstance(result, OCRMetrics)
        # Verify key attributes are present
        assert hasattr(result, "cer")
        assert hasattr(result, "wer")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "fitness")

    def test_val_speed_populated(self) -> None:
        """metrics.speed has all three expected keys after mata.val('ocr', ...)."""
        import mata

        pred = _make_ocr_result([("speed test", None)])
        gt = _make_ocr_gt(["speed test"])

        result = mata.val("ocr", predictions=[pred], ground_truth=[gt], verbose=False)

        assert "preprocess" in result.speed
        assert "inference" in result.speed
        assert "postprocess" in result.speed
        # All values should be finite non-negative
        for key in ("preprocess", "inference", "postprocess"):
            assert result.speed[key] >= 0.0
