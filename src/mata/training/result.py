"""TrainingResult dataclass returned by mata.train() and mata.finetune()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    """Result of a training or fine-tuning run.

    Attributes:
        best_metrics: Best validation metrics achieved during training.
            Actual type is DetMetrics | ClassifyMetrics | SegmentMetrics.
        final_metrics: Metrics from the final epoch.
        best_checkpoint: Path to the best model checkpoint directory.
        last_checkpoint: Path to the last checkpoint directory.
        history: Per-epoch metrics dict, e.g.
            {"train_loss": [...], "val_map50": [...], "lr": [...]}.
        config: The TrainingConfig used for this run.
        epochs_completed: Number of epochs actually completed (may differ from
            config.epochs if early stopping was triggered).
    """

    best_metrics: Any = None   # DetMetrics | ClassifyMetrics | SegmentMetrics
    final_metrics: Any = None
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    history: dict[str, list[float]] = field(default_factory=dict)
    config: Any = None         # TrainingConfig
    epochs_completed: int = 0

    def summary(self) -> str:
        """Return a human-readable training summary string."""
        lines = ["=" * 50, "Training Summary", "=" * 50]

        lines.append(f"Epochs completed : {self.epochs_completed}")

        if self.config is not None:
            task = getattr(self.config, "task", "unknown")
            model = getattr(self.config, "model", "unknown")
            lines.append(f"Task             : {task}")
            lines.append(f"Model            : {model}")

        if self.best_checkpoint:
            lines.append(f"Best checkpoint  : {self.best_checkpoint}")
        if self.last_checkpoint:
            lines.append(f"Last checkpoint  : {self.last_checkpoint}")

        if self.best_metrics is not None:
            lines.append("")
            lines.append("Best validation metrics:")
            _append_metrics(lines, self.best_metrics)

        if self.final_metrics is not None:
            lines.append("")
            lines.append("Final epoch metrics:")
            _append_metrics(lines, self.final_metrics)

        if self.history:
            lines.append("")
            lines.append("Training history keys: " + ", ".join(sorted(self.history)))
            for key in ("train_loss", "val_map50", "val_top1"):
                if key in self.history and self.history[key]:
                    vals = self.history[key]
                    lines.append(f"  {key}: first={vals[0]:.4f}, last={vals[-1]:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def plot_loss(self, save_path: str | None = None) -> None:
        """Plot the training (and optional validation) loss curve.

        Requires matplotlib to be installed.

        Args:
            save_path: If given, save the figure to this path instead of
                displaying it interactively.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_loss(). "
                "Install it with: pip install matplotlib"
            ) from exc

        fig, ax = plt.subplots()
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        if "train_loss" in self.history and self.history["train_loss"]:
            ax.plot(self.history["train_loss"], label="train_loss")
        if "val_loss" in self.history and self.history["val_loss"]:
            ax.plot(self.history["val_loss"], label="val_loss")

        ax.legend()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def plot_metrics(self, save_path: str | None = None) -> None:
        """Plot validation metrics over epochs.

        Requires matplotlib to be installed.

        Args:
            save_path: If given, save the figure to this path instead of
                displaying it interactively.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_metrics(). "
                "Install it with: pip install matplotlib"
            ) from exc

        metric_keys = [k for k in self.history if k.startswith("val_") and k != "val_loss"]

        fig, ax = plt.subplots()
        ax.set_title("Validation Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")

        for key in metric_keys:
            if self.history[key]:
                ax.plot(self.history[key], label=key)

        ax.legend()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_metrics(lines: list[str], metrics: Any) -> None:
    """Append formatted metric lines from a metrics object or dict."""
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")
    else:
        # Try common attribute names used by DetMetrics / ClassifyMetrics
        for attr in ("map50", "map", "top1", "top5", "fitness"):
            val = getattr(metrics, attr, None)
            if val is not None:
                lines.append(f"  {attr}: {val:.4f}")
