"""Tests for mata.val() public API and metric type exports — Task G6.

All tests use mocked Validator/adapters; no real models or datasets are
required.  Goals:
 - mata.val is accessible and callable
 - keyword-only enforcement
 - correct kwarg forwarding to Validator
 - standalone mode accepted
 - missing required args raise ValueError
 - metric types importable directly from mata
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import mata
from mata import ClassifyMetrics, DepthMetrics, DetMetrics, SegmentMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_det_metrics() -> DetMetrics:
    return DetMetrics(names={0: "cat"})


def _make_classify_metrics() -> ClassifyMetrics:
    m = ClassifyMetrics(names={0: "cat", 1: "dog"})
    m.top1 = 0.75
    m.top5 = 0.95
    return m


def _make_depth_metrics() -> DepthMetrics:
    from mata.eval.metrics.depth import DepthMetrics as DM  # noqa: N817

    m = DM()
    m.abs_rel = 0.12
    m.delta_1 = 0.88
    return m


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------


class TestMataValAccessible:
    def test_val_attribute_exists(self):
        assert hasattr(mata, "val")

    def test_val_is_callable(self):
        assert callable(mata.val)

    def test_val_in_all(self):
        assert "val" in mata.__all__

    def test_val_docstring_present(self):
        assert mata.val.__doc__ is not None
        assert len(mata.val.__doc__) > 20


# ---------------------------------------------------------------------------
# Signature constraints
# ---------------------------------------------------------------------------


class TestValSignature:
    def test_task_is_positional(self):
        sig = inspect.signature(mata.val)
        p = sig.parameters["task"]
        assert p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )

    def test_remaining_params_are_keyword_only(self):
        sig = inspect.signature(mata.val)
        kw_only = {
            "model",
            "data",
            "predictions",
            "ground_truth",
            "conf",
            "iou",
            "device",
            "verbose",
            "plots",
            "save_dir",
            "split",
        }
        for name in kw_only:
            assert name in sig.parameters, f"Missing parameter: {name}"
            assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, f"{name} should be keyword-only"

    def test_positional_call_beyond_task_raises(self):
        with pytest.raises(TypeError):
            mata.val("detect", "facebook/detr-resnet-50")  # type: ignore[call-arg]

    def test_defaults(self):
        sig = inspect.signature(mata.val)
        assert sig.parameters["conf"].default == pytest.approx(0.001)
        assert sig.parameters["iou"].default == pytest.approx(0.50)
        assert sig.parameters["verbose"].default is True
        assert sig.parameters["plots"].default is False
        assert sig.parameters["save_dir"].default == ""
        assert sig.parameters["split"].default == "val"


# ---------------------------------------------------------------------------
# Metric type exports
# ---------------------------------------------------------------------------


class TestMetricTypeExports:
    def test_det_metrics_importable_from_mata(self):
        from mata import DetMetrics as DM  # noqa: N817

        assert DM is DetMetrics

    def test_segment_metrics_importable_from_mata(self):
        from mata import SegmentMetrics as SM  # noqa: N817

        assert SM is SegmentMetrics

    def test_classify_metrics_importable_from_mata(self):
        from mata import ClassifyMetrics as CM  # noqa: N817

        assert CM is ClassifyMetrics

    def test_depth_metrics_importable_from_mata(self):
        from mata import DepthMetrics as DPM  # noqa: N814

        assert DPM is DepthMetrics

    def test_det_metrics_in_all(self):
        assert "DetMetrics" in mata.__all__

    def test_segment_metrics_in_all(self):
        assert "SegmentMetrics" in mata.__all__

    def test_classify_metrics_in_all(self):
        assert "ClassifyMetrics" in mata.__all__

    def test_depth_metrics_in_all(self):
        assert "DepthMetrics" in mata.__all__


# ---------------------------------------------------------------------------
# Kwarg forwarding
# ---------------------------------------------------------------------------


class TestValKwargForwarding:
    @patch("mata.eval.validator.Validator")
    def test_task_forwarded(self, MockValidator):  # noqa: N803
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_det_metrics()
        MockValidator.return_value = mock_instance

        mata.val("detect")

        call_kwargs = MockValidator.call_args.kwargs
        assert call_kwargs["task"] == "detect"

    @patch("mata.eval.validator.Validator")
    def test_all_kwargs_forwarded(self, MockValidator):  # noqa: N803
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_det_metrics()
        MockValidator.return_value = mock_instance

        mata.val(
            "segment",
            model="my-model",
            data="coco.yaml",
            conf=0.25,
            iou=0.6,
            device="cuda",
            verbose=False,
            plots=True,
            save_dir="runs/val",
            split="test",
        )

        kw = MockValidator.call_args.kwargs
        assert kw["task"] == "segment"
        assert kw["model"] == "my-model"
        assert kw["data"] == "coco.yaml"
        assert kw["conf"] == pytest.approx(0.25)
        assert kw["iou"] == pytest.approx(0.6)
        assert kw["device"] == "cuda"
        assert kw["verbose"] is False
        assert kw["plots"] is True
        assert kw["save_dir"] == "runs/val"
        assert kw["split"] == "test"

    @patch("mata.eval.validator.Validator")
    def test_extra_kwargs_forwarded(self, MockValidator):  # noqa: N803
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_det_metrics()
        MockValidator.return_value = mock_instance

        mata.val("classify", extra_flag=True, batch_size=32)

        kw = MockValidator.call_args.kwargs
        assert kw["extra_flag"] is True
        assert kw["batch_size"] == 32

    @patch("mata.eval.validator.Validator")
    def test_run_called_on_validator(self, MockValidator):  # noqa: N803
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_classify_metrics()
        MockValidator.return_value = mock_instance

        result = mata.val("classify")

        mock_instance.run.assert_called_once()
        assert isinstance(result, ClassifyMetrics)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------


class TestValStandaloneMode:
    @patch("mata.eval.validator.Validator")
    def test_predictions_and_ground_truth_forwarded(self, MockValidator):  # noqa: N803
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_classify_metrics()
        MockValidator.return_value = mock_instance

        preds = [MagicMock(), MagicMock()]
        gt = [MagicMock(), MagicMock()]

        mata.val("classify", predictions=preds, ground_truth=gt)

        kw = MockValidator.call_args.kwargs
        assert kw["predictions"] is preds
        assert kw["ground_truth"] is gt
        assert kw["data"] is None


# ---------------------------------------------------------------------------
# Unsupported task
# ---------------------------------------------------------------------------


class TestValUnsupportedTask:
    def test_unsupported_task_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            mata.val("magic_task")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestValReturnType:
    @patch("mata.eval.validator.Validator")
    def test_returns_det_metrics_for_detect(self, MockValidator):  # noqa: N803
        m = _make_det_metrics()
        MockValidator.return_value.run.return_value = m
        result = mata.val("detect")
        assert isinstance(result, DetMetrics)

    @patch("mata.eval.validator.Validator")
    def test_returns_classify_metrics_for_classify(self, MockValidator):  # noqa: N803
        m = _make_classify_metrics()
        MockValidator.return_value.run.return_value = m
        result = mata.val("classify")
        assert isinstance(result, ClassifyMetrics)

    @patch("mata.eval.validator.Validator")
    def test_returns_depth_metrics_for_depth(self, MockValidator):  # noqa: N803
        m = _make_depth_metrics()
        MockValidator.return_value.run.return_value = m
        result = mata.val("depth")
        assert isinstance(result, DepthMetrics)
