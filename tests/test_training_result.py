"""Tests for mata.training.result.TrainingResult.

Covers the previously uncovered lines from the A3 coverage report:
  - summary(): config info (lines 52-54), checkpoint paths (lines 57-59)
  - plot_loss(): matplotlib usage (lines 92-108)
  - plot_metrics(): matplotlib usage (lines 122-147)
  - _append_metrics(): object-style metrics (lines 156-164)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mata.training.result import TrainingResult, _append_metrics

# ---------------------------------------------------------------------------
# TrainingResult.summary() — coverage gaps
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for TrainingResult.summary() focusing on previously uncovered branches."""

    def test_summary_basic_returns_string(self):
        result = TrainingResult(epochs_completed=3)
        s = result.summary()
        assert isinstance(s, str)
        assert "3" in s

    def test_summary_includes_config_task_and_model(self):
        # Covers lines 52-54: config is not None branch
        cfg = SimpleNamespace(task="detect", model="facebook/detr-resnet-50")
        result = TrainingResult(epochs_completed=2, config=cfg)
        s = result.summary()
        assert "detect" in s
        assert "facebook/detr-resnet-50" in s

    def test_summary_includes_best_checkpoint(self):
        # Covers lines 57-58: best_checkpoint branch
        result = TrainingResult(
            epochs_completed=2,
            best_checkpoint="/runs/train/detect/best",
        )
        s = result.summary()
        assert "/runs/train/detect/best" in s

    def test_summary_includes_last_checkpoint(self):
        # Covers lines 58-59: last_checkpoint branch
        result = TrainingResult(
            epochs_completed=2,
            last_checkpoint="/runs/train/detect/last",
        )
        s = result.summary()
        assert "/runs/train/detect/last" in s

    def test_summary_includes_both_checkpoints(self):
        # Covers both checkpoint branches together
        result = TrainingResult(
            epochs_completed=5,
            best_checkpoint="/runs/best",
            last_checkpoint="/runs/last",
        )
        s = result.summary()
        assert "/runs/best" in s
        assert "/runs/last" in s

    def test_summary_with_best_metrics_dict(self):
        # Covers _append_metrics dict path (lines 156-161)
        result = TrainingResult(
            epochs_completed=3,
            best_metrics={"map50": 0.52, "map": 0.31},
        )
        s = result.summary()
        assert "Best validation metrics" in s
        assert "map50" in s

    def test_summary_with_final_metrics_dict(self):
        # Covers final_metrics path via _append_metrics dict
        result = TrainingResult(
            epochs_completed=3,
            final_metrics={"map50": 0.48},
        )
        s = result.summary()
        assert "Final epoch metrics" in s
        assert "map50" in s

    def test_summary_with_metrics_object(self):
        # Covers _append_metrics object branch (lines 162-164)
        metrics = SimpleNamespace(map50=0.66, map=0.41, fitness=0.70)
        result = TrainingResult(epochs_completed=4, best_metrics=metrics)
        s = result.summary()
        assert "map50" in s
        assert "0.6600" in s

    def test_summary_with_history_keys(self):
        result = TrainingResult(
            epochs_completed=2,
            history={
                "train_loss": [0.8, 0.5],
                "val_map50": [0.2, 0.4],
                "lr": [1e-4, 1e-4],
            },
        )
        s = result.summary()
        assert "train_loss" in s
        # first/last shown for known keys
        assert "first=" in s

    def test_summary_contains_separator_lines(self):
        result = TrainingResult(epochs_completed=1)
        s = result.summary()
        assert "=" * 50 in s

    def test_summary_config_with_getattr_fallback(self):
        # config without task/model attrs → uses "unknown"
        cfg = SimpleNamespace()  # no task, no model attrs
        result = TrainingResult(epochs_completed=1, config=cfg)
        s = result.summary()
        assert "unknown" in s


# ---------------------------------------------------------------------------
# TrainingResult.plot_loss() — coverage gaps (lines 92-108)
# ---------------------------------------------------------------------------


class TestPlotLoss:
    """Tests for TrainingResult.plot_loss()."""

    def test_plot_loss_missing_matplotlib_raises_import_error(self):
        # Covers the ImportError branch at the start of plot_loss
        import sys

        result = TrainingResult(epochs_completed=1, history={"train_loss": [0.5]})
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                result.plot_loss()

    def test_plot_loss_calls_show_when_no_save_path(self):
        # Covers the body of plot_loss (lines 92-108) with plt.show() path
        result = TrainingResult(
            epochs_completed=2,
            history={"train_loss": [0.8, 0.5]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show") as mock_show:
                result.plot_loss()
        mock_show.assert_called_once()

    def test_plot_loss_with_save_path(self, tmp_path):
        # Covers the fig.savefig() + plt.close() branch
        save_file = str(tmp_path / "loss.png")
        result = TrainingResult(
            epochs_completed=2,
            history={"train_loss": [0.8, 0.5], "val_loss": [0.9, 0.6]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.close") as mock_close:
                result.plot_loss(save_path=save_file)
        mock_fig.savefig.assert_called_once_with(save_file)
        mock_close.assert_called_once_with(mock_fig)

    def test_plot_loss_with_both_train_and_val_loss(self):
        # Covers the val_loss branch in plot_loss
        result = TrainingResult(
            epochs_completed=2,
            history={"train_loss": [0.8, 0.5], "val_loss": [0.9, 0.6]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                result.plot_loss()
        # ax.plot called twice: train_loss and val_loss
        assert mock_ax.plot.call_count == 2

    def test_plot_loss_with_only_train_loss(self):
        result = TrainingResult(
            epochs_completed=2,
            history={"train_loss": [0.8, 0.5]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                result.plot_loss()
        # ax.plot called once: only train_loss
        assert mock_ax.plot.call_count == 1

    def test_plot_loss_empty_history_does_not_raise(self):
        result = TrainingResult(epochs_completed=0, history={})
        with patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())):
            with patch("matplotlib.pyplot.show"):
                result.plot_loss()  # Should not raise


# ---------------------------------------------------------------------------
# TrainingResult.plot_metrics() — coverage gaps (lines 122-147)
# ---------------------------------------------------------------------------


class TestPlotMetrics:
    """Tests for TrainingResult.plot_metrics()."""

    def test_plot_metrics_missing_matplotlib_raises_import_error(self):
        import sys

        result = TrainingResult(epochs_completed=1)
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                result.plot_metrics()

    def test_plot_metrics_with_save_path(self, tmp_path):
        save_file = str(tmp_path / "metrics.png")
        result = TrainingResult(
            epochs_completed=3,
            history={"train_loss": [0.8, 0.6, 0.4], "val_map50": [0.3, 0.4, 0.5]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.close") as mock_close:
                result.plot_metrics(save_path=save_file)
        mock_fig.savefig.assert_called_once_with(save_file)
        mock_close.assert_called_once_with(mock_fig)

    def test_plot_metrics_calls_show_no_save_path(self):
        result = TrainingResult(
            epochs_completed=2,
            history={"val_map50": [0.3, 0.5], "val_top1": [0.7, 0.8]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show") as mock_show:
                result.plot_metrics()
        mock_show.assert_called_once()

    def test_plot_metrics_plots_all_val_keys(self):
        result = TrainingResult(
            epochs_completed=2,
            history={
                "train_loss": [0.8, 0.5],
                "val_map50": [0.3, 0.5],
                "val_top1": [0.6, 0.7],
                "val_loss": [0.9, 0.7],  # excluded — val_loss is filtered out
            },
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                result.plot_metrics()
        # Only val_ keys that are NOT val_loss are plotted: val_map50, val_top1
        assert mock_ax.plot.call_count == 2

    def test_plot_metrics_no_val_keys_plots_nothing(self):
        result = TrainingResult(
            epochs_completed=2,
            history={"train_loss": [0.8, 0.5]},
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                result.plot_metrics()
        assert mock_ax.plot.call_count == 0

    def test_plot_metrics_skips_empty_history_values(self):
        result = TrainingResult(
            epochs_completed=2,
            history={"val_map50": []},  # empty list — should not plot
        )
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)):
            with patch("matplotlib.pyplot.show"):
                result.plot_metrics()
        assert mock_ax.plot.call_count == 0


# ---------------------------------------------------------------------------
# _append_metrics helper — coverage gap (lines 156-164)
# ---------------------------------------------------------------------------


class TestAppendMetrics:
    """Tests for the _append_metrics module-level helper."""

    def test_dict_metrics_appended(self):
        # Covers the dict branch (lines 156-161)
        lines: list[str] = []
        _append_metrics(lines, {"map50": 0.52, "map": 0.31})
        assert any("map50" in line for line in lines)
        assert any("map" in line for line in lines)

    def test_object_with_map50_appended(self):
        # Covers the object/else branch (lines 162-164)
        lines: list[str] = []
        obj = SimpleNamespace(map50=0.55, map=None, top1=None, top5=None, fitness=None)
        _append_metrics(lines, obj)
        assert any("map50" in line and "0.5500" in line for line in lines)

    def test_object_with_top1_appended(self):
        lines: list[str] = []
        obj = SimpleNamespace(map50=None, map=None, top1=0.88, top5=0.96, fitness=None)
        _append_metrics(lines, obj)
        assert any("top1" in line for line in lines)
        assert any("top5" in line for line in lines)

    def test_object_with_fitness_appended(self):
        lines: list[str] = []
        obj = SimpleNamespace(map50=None, map=None, top1=None, top5=None, fitness=0.73)
        _append_metrics(lines, obj)
        assert any("fitness" in line for line in lines)

    def test_object_with_no_known_attrs_appends_nothing(self):
        lines: list[str] = []
        obj = SimpleNamespace()  # no known attrs
        _append_metrics(lines, obj)
        assert lines == []

    def test_empty_dict_appends_nothing(self):
        lines: list[str] = []
        _append_metrics(lines, {})
        assert lines == []

    def test_dict_with_multiple_keys_all_appended(self):
        lines: list[str] = []
        _append_metrics(lines, {"a": 1, "b": 2, "c": 3})
        assert len(lines) == 3
