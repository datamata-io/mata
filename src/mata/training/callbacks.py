"""Training callbacks for validation, logging, and early stopping.

Used by training engines (HFTrainingEngine, TorchTrainingEngine) and
the TrainingOrchestrator to hook into the training loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ValidationCallback
# ---------------------------------------------------------------------------


class ValidationCallback:
    """Run mata.val() at configured intervals during training.

    Temporarily sets model to eval mode, runs validation, then restores
    training mode.  Returns task-specific metrics as a plain dict.

    Args:
        task: Task type — ``"detect"``, ``"classify"``, or ``"segment"``.
        val_data: Validation data path/config (same format as ``mata.val()``
            ``data`` argument).
        val_every: Fire every N epochs (1 = every epoch).
        verbose: Pass ``verbose=True`` to ``mata.val()``.
    """

    def __init__(
        self,
        task: str,
        val_data: Any,
        val_every: int = 1,
        verbose: bool = False,
    ) -> None:
        self.task = task
        self.val_data = val_data
        self.val_every = max(1, val_every)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_epoch_end(
        self,
        epoch: int,
        model: Any,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Run validation at the end of an epoch.

        Args:
            epoch: 0-based epoch index.
            model: The model being trained (must support ``.train()`` /
                ``.eval()`` via the PyTorch API, or may be ``None`` when
                using HuggingFace Trainer which manages its own eval).
            **kwargs: Forwarded keyword arguments for extensibility (e.g.
                ``adapter``, ``device``).

        Returns:
            Dict with metric values (e.g. ``{"map50": 0.312, "map": 0.187}``)
            when validation runs, or ``None`` if skipped / failed.
        """
        # Epoch is 0-based internally; fire at val_every intervals.
        if (epoch + 1) % self.val_every != 0:
            return None

        # Snapshot training mode so we can restore it afterwards.
        was_training = False
        if model is not None and hasattr(model, "training"):
            was_training = bool(model.training)
            model.eval()

        try:
            return self._run_val(model=model, epoch=epoch, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ValidationCallback: mata.val() failed at epoch %d — %s",
                epoch + 1,
                exc,
            )
            return None
        finally:
            # Always restore training mode.
            if model is not None and hasattr(model, "train") and was_training:
                model.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_val(
        self,
        model: Any,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Delegate to ``mata.val()`` and normalise result to a plain dict."""
        import mata

        adapter = kwargs.get("adapter")
        device = kwargs.get("device")

        val_kwargs: dict[str, Any] = {
            "task": self.task,
            "data": self.val_data,
            "verbose": self.verbose,
        }
        if adapter is not None:
            val_kwargs["model"] = adapter
        if device is not None:
            val_kwargs["device"] = str(device)

        result = mata.val(**val_kwargs)
        if result is None:
            return None

        metrics: dict[str, float] = {}
        # Extract common metric attributes into a plain dict.
        for attr in ("map50", "map", "top1", "top5", "fitness"):
            val = getattr(result, attr, None)
            if val is None and isinstance(result, dict):
                val = result.get(attr)
            if val is not None:
                try:
                    metrics[attr] = float(val)
                except (TypeError, ValueError):
                    pass

        logger.info(
            "ValidationCallback epoch %d — %s",
            epoch + 1,
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) or "no metrics",
        )
        return metrics if metrics else None


# ---------------------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------------------


class LoggingCallback:
    """Log training progress to console and optionally to a log file.

    Output format (YOLO-style table)::

        Epoch  GPU_mem  train_loss  val/mAP50  val/mAP50-95  lr
        1/10   3.2G     0.8542      0.312      0.187         1e-4

    Args:
        save_dir: If set, writes log to ``{save_dir}/training.log``.
        verbose: Print table to console when ``True``.
    """

    # Column spec: (header_label, metrics_key, width, format_spec)
    _COLUMNS: list[tuple[str, str, int, str]] = [
        ("Epoch", "__epoch__", 10, "s"),
        ("GPU_mem", "__gpu__", 8, "s"),
        ("train_loss", "train_loss", 12, ".4f"),
        ("val/mAP50", "val_map50", 10, ".3f"),
        ("val/mAP50-95", "val_map", 12, ".3f"),
        ("lr", "__lr__", 8, "s"),
    ]

    def __init__(
        self,
        save_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.save_dir = save_dir
        self.verbose = verbose
        self._file_handler: logging.FileHandler | None = None

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            log_path = os.path.join(save_dir, "training.log")
            self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
            self._file_handler.setLevel(logging.INFO)
            self._file_handler.setFormatter(logging.Formatter("%(message)s"))
            # Attach to the module-level logger so log file receives all
            # training messages as well as the formatted table rows.
            logger.logger.addHandler(  # type: ignore[attr-defined]
                self._file_handler
            ) if hasattr(logger, "logger") else None

        self._header_printed = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
        lr: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log one epoch row.

        Args:
            epoch: 0-based epoch index.
            total_epochs: Total number of epochs in the run (for formatting).
            metrics: Dict with keys like ``"train_loss"``, ``"val_map50"``,
                ``"val_map"``, ``"val_top1"``, etc.
            lr: Current learning rate.
            **kwargs: Ignored — for forward-compat.
        """
        row = self._build_row(epoch, total_epochs, metrics, lr)
        header = self._build_header()

        lines: list[str] = []
        if not self._header_printed:
            lines.append(header)
            self._header_printed = True
        lines.append(row)

        for line in lines:
            if self.verbose:
                print(line)
            self._write_to_file(line)

    def close(self) -> None:
        """Flush and close the log file handler (call when training ends)."""
        if self._file_handler is not None:
            self._file_handler.flush()
            self._file_handler.close()
            self._file_handler = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_header(self) -> str:
        parts = []
        for label, _, width, _ in self._COLUMNS:
            parts.append(f"{label:<{width}}")
        return "  ".join(parts)

    def _build_row(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, Any],
        lr: float | None,
    ) -> str:
        parts: list[str] = []
        for _, key, width, fmt in self._COLUMNS:
            if key == "__epoch__":
                cell = f"{epoch + 1}/{total_epochs}"
            elif key == "__gpu__":
                cell = self._gpu_mem_str()
            elif key == "__lr__":
                cell = f"{lr:.2e}" if lr is not None else "-"
            else:
                raw = metrics.get(key)
                if raw is None:
                    cell = "-"
                else:
                    try:
                        cell = format(float(raw), fmt)
                    except (TypeError, ValueError):
                        cell = str(raw)
            parts.append(f"{cell:<{width}}")
        return "  ".join(parts)

    @staticmethod
    def _gpu_mem_str() -> str:
        """Return a human-readable GPU memory string, or '-' if unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes = torch.cuda.memory_reserved()
                mem_gb = mem_bytes / (1024 ** 3)
                return f"{mem_gb:.1f}G"
        except Exception:  # noqa: BLE001
            pass
        return "-"

    def _write_to_file(self, line: str) -> None:
        """Write a line directly to the log file, if configured."""
        if self.save_dir is not None:
            log_path = os.path.join(self.save_dir, "training.log")
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------


class EarlyStoppingCallback:
    """Stop training if validation metric doesn't improve for ``patience`` epochs.

    Args:
        patience: Number of epochs to wait without improvement before stopping.
            ``0`` disables early stopping entirely.
        metric_key: Key to monitor in the metrics dict passed to
            ``on_epoch_end()``.  Typical values:

            - ``"val_map50"`` (detect / segment)
            - ``"top1"`` or ``"val_top1"`` (classify)
            - ``"val_loss"`` (when using ``mode="min"``)
        mode: ``"max"`` if higher is better (accuracy, mAP), ``"min"`` if
            lower is better (loss).

    Example::

        cb = EarlyStoppingCallback(patience=5, metric_key="val_map50")
        for epoch in range(100):
            metrics = ... # {"val_map50": 0.72, ...}
            if cb.on_epoch_end(epoch, metrics):
                print("Early stopping triggered")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        metric_key: str = "val_map50",
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
        self.patience = patience
        self.metric_key = metric_key
        self.mode = mode
        self._best: float | None = None
        self._epochs_no_improvement: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def best(self) -> float | None:
        """Best metric value seen so far (``None`` before first call)."""
        return self._best

    @property
    def epochs_without_improvement(self) -> int:
        """Counter of consecutive epochs without metric improvement."""
        return self._epochs_no_improvement

    def reset(self) -> None:
        """Reset internal state (use when restarting training)."""
        self._best = None
        self._epochs_no_improvement = 0

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, Any],
    ) -> bool:
        """Check whether training should stop.

        Args:
            epoch: 0-based epoch index (used for logging only).
            metrics: Dict of metric values.  Must contain ``self.metric_key``
                for early stopping to be active.

        Returns:
            ``True`` if training should stop (patience exceeded), ``False``
            otherwise.
        """
        # Patience=0 means disabled.
        if self.patience == 0:
            return False

        value = self._extract_metric(metrics)
        if value is None:
            # Metric not available this epoch — don't count it.
            logger.debug(
                "EarlyStoppingCallback: metric '%s' not found in metrics at "
                "epoch %d; skipping.",
                self.metric_key,
                epoch + 1,
            )
            return False

        improved = self._is_improvement(value)
        if improved:
            self._best = value
            self._epochs_no_improvement = 0
            logger.debug(
                "EarlyStoppingCallback: new best %s=%.6f at epoch %d.",
                self.metric_key,
                value,
                epoch + 1,
            )
        else:
            self._epochs_no_improvement += 1
            logger.info(
                "EarlyStoppingCallback: no improvement in '%s' for %d/%d epochs.",
                self.metric_key,
                self._epochs_no_improvement,
                self.patience,
            )
            if self._epochs_no_improvement >= self.patience:
                logger.info(
                    "EarlyStoppingCallback: patience %d reached — stopping early.",
                    self.patience,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metric(self, metrics: dict[str, Any]) -> float | None:
        """Extract the monitored metric as a float, or ``None`` if missing."""
        raw = metrics.get(self.metric_key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _is_improvement(self, value: float) -> bool:
        """Return ``True`` if *value* is strictly better than the current best."""
        if self._best is None:
            return True
        if self.mode == "max":
            return value > self._best
        return value < self._best  # mode == "min"
