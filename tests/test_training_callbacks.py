"""Tests for mata.training.callbacks — ValidationCallback, LoggingCallback, EarlyStoppingCallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mata.training.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
    ValidationCallback,
)

# =============================================================================
# ValidationCallback
# =============================================================================


class TestValidationCallback:
    def _make_mock_result(self, **kwargs):
        result = MagicMock()
        defaults = {"map50": 0.5, "map": 0.3, "top1": None, "top5": None, "fitness": None}
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(result, k, v)
        return result

    def test_calls_mata_val_with_correct_task(self):
        cb = ValidationCallback(task="detect", val_data="data.yaml")
        mock_result = self._make_mock_result()
        with patch("mata.val", return_value=mock_result) as mock_val:
            cb.on_epoch_end(0, model=None)
        mock_val.assert_called_once()
        call_kwargs = mock_val.call_args[1]
        assert call_kwargs["task"] == "detect"
        assert call_kwargs["data"] == "data.yaml"

    def test_returns_metrics_dict(self):
        cb = ValidationCallback(task="detect", val_data="data.yaml")
        mock_result = self._make_mock_result(map50=0.72, map=0.45)
        with patch("mata.val", return_value=mock_result):
            result = cb.on_epoch_end(0, model=None)
        assert isinstance(result, dict)
        assert pytest.approx(result["map50"]) == 0.72
        assert pytest.approx(result["map"]) == 0.45

    def test_model_restored_to_train_mode_after_callback(self):
        cb = ValidationCallback(task="classify", val_data="data.yaml")
        model = MagicMock()
        model.training = True
        mock_result = self._make_mock_result(top1=0.9)
        with patch("mata.val", return_value=mock_result):
            cb.on_epoch_end(0, model=model)
        model.train.assert_called()

    def test_model_set_to_eval_mode_before_validation(self):
        eval_called_before_val = []

        cb = ValidationCallback(task="classify", val_data="data.yaml")
        model = MagicMock()
        model.training = True

        def side_effect(**kw):
            eval_called_before_val.append(model.eval.called)
            return self._make_mock_result()

        with patch("mata.val", side_effect=side_effect):
            cb.on_epoch_end(0, model=model)

        assert eval_called_before_val[0], "model.eval() should be called before mata.val()"

    def test_fires_only_at_val_every_intervals(self):
        # val_every=2: fire at epoch #2, #4 (0-based: 1, 3)
        cb = ValidationCallback(task="detect", val_data="data.yaml", val_every=2)
        mock_result = self._make_mock_result()
        with patch("mata.val", return_value=mock_result) as mock_val:
            cb.on_epoch_end(0, model=None)  # epoch 1 — skip
            cb.on_epoch_end(1, model=None)  # epoch 2 — fire
            cb.on_epoch_end(2, model=None)  # epoch 3 — skip
            cb.on_epoch_end(3, model=None)  # epoch 4 — fire
        assert mock_val.call_count == 2

    def test_skips_epochs_not_at_interval(self):
        cb = ValidationCallback(task="detect", val_data="data.yaml", val_every=5)
        with patch("mata.val") as mock_val:
            result = cb.on_epoch_end(1, model=None)  # epoch 2 — skip
        assert result is None
        mock_val.assert_not_called()

    def test_handles_validation_failure_gracefully(self):
        cb = ValidationCallback(task="detect", val_data="bad_data.yaml")
        with patch("mata.val", side_effect=RuntimeError("val failed")):
            result = cb.on_epoch_end(0, model=None)
        # Should return None, not raise
        assert result is None

    def test_model_still_in_train_mode_after_failure(self):
        cb = ValidationCallback(task="detect", val_data="data.yaml")
        model = MagicMock()
        model.training = True
        with patch("mata.val", side_effect=RuntimeError("boom")):
            cb.on_epoch_end(0, model=model)
        model.train.assert_called()

    def test_verbose_passed_to_val(self):
        cb = ValidationCallback(task="detect", val_data="data.yaml", verbose=True)
        mock_result = self._make_mock_result()
        with patch("mata.val", return_value=mock_result) as mock_val:
            cb.on_epoch_end(0, model=None)
        call_kwargs = mock_val.call_args[1]
        assert call_kwargs["verbose"] is True


# =============================================================================
# LoggingCallback
# =============================================================================


class TestLoggingCallback:
    _METRICS = {
        "train_loss": 0.8542,
        "val_map50": 0.312,
        "val_map": 0.187,
    }

    def test_prints_header_on_first_epoch(self, capsys):
        cb = LoggingCallback(verbose=True)
        cb.on_epoch_end(0, 10, self._METRICS, lr=1e-4)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "train_loss" in captured.out

    def test_prints_data_row(self, capsys):
        cb = LoggingCallback(verbose=True)
        cb.on_epoch_end(0, 10, self._METRICS, lr=1e-4)
        captured = capsys.readouterr()
        assert "0.8542" in captured.out
        assert "0.312" in captured.out

    def test_epoch_number_formatting(self, capsys):
        cb = LoggingCallback(verbose=True)
        cb.on_epoch_end(2, 10, self._METRICS, lr=1e-4)
        captured = capsys.readouterr()
        assert "3/10" in captured.out

    def test_no_output_when_not_verbose(self, capsys):
        cb = LoggingCallback(verbose=False)
        cb.on_epoch_end(0, 10, self._METRICS, lr=1e-4)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_writes_to_log_file_when_save_dir_set(self, tmp_path):
        cb = LoggingCallback(save_dir=str(tmp_path), verbose=False)
        cb.on_epoch_end(0, 10, self._METRICS, lr=1e-4)
        log_file = tmp_path / "training.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "0.8542" in content

    def test_creates_save_dir_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "nested" / "logs"
        cb = LoggingCallback(save_dir=str(new_dir), verbose=False)
        cb.on_epoch_end(0, 10, self._METRICS, lr=1e-4)
        assert (new_dir / "training.log").exists()

    def test_handles_missing_metrics_gracefully(self, capsys):
        cb = LoggingCallback(verbose=True)
        # Metrics dict missing val_map50 and val_map
        cb.on_epoch_end(0, 5, {"train_loss": 1.23}, lr=0.001)
        captured = capsys.readouterr()
        assert "1.23" in captured.out
        # Missing columns should show '-'
        assert "-" in captured.out

    def test_header_printed_only_once(self, capsys):
        cb = LoggingCallback(verbose=True)
        for epoch in range(3):
            cb.on_epoch_end(epoch, 3, self._METRICS, lr=1e-4)
        captured = capsys.readouterr()
        # Header should appear exactly once
        assert captured.out.count("train_loss") == 1

    def test_lr_none_shows_dash(self, capsys):
        cb = LoggingCallback(verbose=True)
        cb.on_epoch_end(0, 5, self._METRICS, lr=None)
        captured = capsys.readouterr()
        assert "-" in captured.out

    def test_appends_multiple_epochs_to_same_log_file(self, tmp_path):
        cb = LoggingCallback(save_dir=str(tmp_path), verbose=False)
        cb.on_epoch_end(0, 3, {"train_loss": 1.0}, lr=0.01)
        cb.on_epoch_end(1, 3, {"train_loss": 0.8}, lr=0.01)
        content = (tmp_path / "training.log").read_text(encoding="utf-8")
        assert "1.0000" in content
        assert "0.8000" in content


# =============================================================================
# EarlyStoppingCallback
# =============================================================================


class TestEarlyStoppingCallback:
    def test_returns_false_while_metric_improving_max(self):
        cb = EarlyStoppingCallback(patience=3, metric_key="val_map50", mode="max")
        for val in [0.5, 0.6, 0.7, 0.8]:
            result = cb.on_epoch_end(0, {"val_map50": val})
            assert result is False

    def test_returns_false_while_metric_improving_min(self):
        cb = EarlyStoppingCallback(patience=3, metric_key="val_loss", mode="min")
        for val in [1.0, 0.8, 0.6, 0.4]:
            result = cb.on_epoch_end(0, {"val_loss": val})
            assert result is False

    def test_returns_true_after_patience_exceeded_max(self):
        cb = EarlyStoppingCallback(patience=3, metric_key="val_map50", mode="max")
        cb.on_epoch_end(0, {"val_map50": 0.8})  # sets best
        for epoch in range(3):
            stopped = cb.on_epoch_end(epoch + 1, {"val_map50": 0.5})  # no improvement
        assert stopped is True

    def test_returns_true_after_patience_exceeded_min(self):
        cb = EarlyStoppingCallback(patience=2, metric_key="val_loss", mode="min")
        cb.on_epoch_end(0, {"val_loss": 0.5})  # sets best
        for epoch in range(2):
            stopped = cb.on_epoch_end(epoch + 1, {"val_loss": 0.9})  # worse
        assert stopped is True

    def test_counter_resets_on_improvement(self):
        cb = EarlyStoppingCallback(patience=3, metric_key="val_map50", mode="max")
        cb.on_epoch_end(0, {"val_map50": 0.5})
        cb.on_epoch_end(1, {"val_map50": 0.4})  # no improvement
        cb.on_epoch_end(2, {"val_map50": 0.4})  # no improvement
        assert cb.epochs_without_improvement == 2
        cb.on_epoch_end(3, {"val_map50": 0.6})  # improvement — reset
        assert cb.epochs_without_improvement == 0

    def test_zero_patience_disables_early_stopping(self):
        cb = EarlyStoppingCallback(patience=0, metric_key="val_map50", mode="max")
        for epoch in range(100):
            result = cb.on_epoch_end(epoch, {"val_map50": 0.0})
            assert result is False

    def test_first_epoch_sets_best_and_not_stop(self):
        cb = EarlyStoppingCallback(patience=5, metric_key="val_map50", mode="max")
        assert cb.best is None
        result = cb.on_epoch_end(0, {"val_map50": 0.75})
        assert result is False
        assert pytest.approx(cb.best) == 0.75

    def test_missing_metric_key_returns_false(self):
        cb = EarlyStoppingCallback(patience=2, metric_key="val_map50", mode="max")
        result = cb.on_epoch_end(0, {"some_other_key": 0.5})
        assert result is False
        assert cb.epochs_without_improvement == 0

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStoppingCallback(patience=3, mode="invalid")

    def test_patience_exactly_at_boundary(self):
        cb = EarlyStoppingCallback(patience=2, metric_key="val_map50", mode="max")
        cb.on_epoch_end(0, {"val_map50": 0.9})
        assert cb.on_epoch_end(1, {"val_map50": 0.5}) is False  # 1st no-improve
        assert cb.on_epoch_end(2, {"val_map50": 0.5}) is True  # 2nd — stop

    def test_reset_clears_state(self):
        cb = EarlyStoppingCallback(patience=3, metric_key="val_map50", mode="max")
        cb.on_epoch_end(0, {"val_map50": 0.9})
        cb.on_epoch_end(1, {"val_map50": 0.5})
        cb.reset()
        assert cb.best is None
        assert cb.epochs_without_improvement == 0

    def test_strictly_equal_metric_does_not_count_as_improvement(self):
        cb = EarlyStoppingCallback(patience=2, metric_key="val_map50", mode="max")
        cb.on_epoch_end(0, {"val_map50": 0.7})
        # Same value — not strictly better
        assert cb.on_epoch_end(1, {"val_map50": 0.7}) is False
        assert cb.epochs_without_improvement == 1
