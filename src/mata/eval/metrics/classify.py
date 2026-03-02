"""Classification metrics (ClassifyMetrics) — Task C3 implementation."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.eval.confusion_matrix import ConfusionMatrix


@dataclass
class ClassifyMetrics:
    """Top-1 / Top-5 classification accuracy metrics.

    Tracks accumulated top-1 and top-5 accuracy over multiple
    :meth:`process_predictions` calls, then exposes the running averages
    via :attr:`top1` and :attr:`top5`.

    Args:
        names: Mapping from integer class ID to human-readable label,
               e.g. ``{0: "cat", 1: "dog"}``.  Used by :meth:`summary`.
        nc:    Number of classes.  Set automatically from ``names`` when
               left at the default of ``0``.

    Example::

        metrics = ClassifyMetrics(names={0: "cat", 1: "dog"}, nc=2)
        metrics.process_predictions([0, 1, 0], [0, 0, 0])
        print(metrics.top1)   # 0.6667
        print(metrics.fitness)  # mean(top1, top5)
    """

    #: Class-ID → class-name mapping for display / summary.
    names: dict[int, str] = field(default_factory=dict)

    #: Number of classes.  Inferred from ``names`` when left as 0.
    nc: int = 0

    #: Top-1 accuracy in [0, 1].  Updated by :meth:`process_predictions`.
    top1: float = 0.0

    #: Top-5 accuracy in [0, 1].  Updated by :meth:`process_predictions`.
    top5: float = 0.0

    #: Timing breakdown in ms/image.
    speed: dict[str, float] = field(
        default_factory=lambda: {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
    )

    #: Optional confusion matrix attached by the Validator (Task E1).
    confusion_matrix: ConfusionMatrix | None = None

    # ------------------------------------------------------------------
    # Private accumulators — not part of the public API.
    # ------------------------------------------------------------------
    _n_top1_correct: int = field(default=0, init=False, repr=False)
    _n_top5_correct: int = field(default=0, init=False, repr=False)
    _n_total: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Infer nc from names when caller omits it.
        if self.nc == 0 and self.names:
            self.nc = len(self.names)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def process_predictions(
        self,
        pred_labels: list[int],
        target_labels: list[int],
        pred_top5: list[list[int]] | None = None,
    ) -> None:
        """Accumulate top-1 and top-5 accuracy from one batch.

        Args:
            pred_labels:   Predicted top-1 class IDs, length N.
            target_labels: Ground-truth class IDs, length N.
            pred_top5:     Optional list of N lists, each containing up to
                           5 predicted class IDs in descending score order.
                           When ``None``, top-5 accuracy is treated the same
                           as top-1 (appropriate for ``nc < 5`` classifiers).

        Raises:
            ValueError: If ``pred_labels`` and ``target_labels`` have
                        different lengths.
        """
        if len(pred_labels) != len(target_labels):
            raise ValueError(
                f"pred_labels length ({len(pred_labels)}) must equal " f"target_labels length ({len(target_labels)})."
            )

        n = len(target_labels)
        if n == 0:
            return

        # Top-1 accuracy
        top1_correct = sum(int(p == t) for p, t in zip(pred_labels, target_labels))

        # Top-5 accuracy
        if pred_top5 is not None:
            top5_correct = sum(int(t in top5[:5]) for t, top5 in zip(target_labels, pred_top5))
        else:
            # When no top-5 list is provided, fall back to top-1 hits.
            # This is correct for binary / small nc classifiers where
            # all classes would be in the top-5 anyway.
            top5_correct = top1_correct

        self._n_top1_correct += top1_correct
        self._n_top5_correct += top5_correct
        self._n_total += n

        # Recompute running averages.
        self.top1 = self._n_top1_correct / self._n_total
        self.top5 = self._n_top5_correct / self._n_total

    # ------------------------------------------------------------------
    # Metric interface (mirrors Metric / DetMetrics API)
    # ------------------------------------------------------------------

    def mean_results(self) -> list[float]:
        """Return ``[top1, top5]`` for use by the console printer."""
        return [self.top1, self.top5]

    def class_result(self, i: int) -> tuple[float, float]:
        """Return ``(top1, top5)`` — same for every rank (no per-class breakdown)."""
        return (self.top1, self.top5)

    @property
    def fitness(self) -> float:
        """Scalar fitness: ``mean(top1, top5)``."""
        return (self.top1 + self.top5) / 2.0

    # ------------------------------------------------------------------
    # Logging / export helpers
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Metric keys (without ``"fitness"``) for use by loggers."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def results_dict(self) -> dict[str, float]:
        """Flat metrics dict for logging and external tools.

        Returns a dict with exactly 3 keys::

            {
                "metrics/accuracy_top1": <float>,
                "metrics/accuracy_top5": <float>,
                "fitness":               <float>,
            }
        """
        return {
            "metrics/accuracy_top1": self.top1,
            "metrics/accuracy_top5": self.top5,
            "fitness": self.fitness,
        }

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return overall accuracy summary as a single-element list.

        The list contains one ``dict`` with keys:

        * ``"top1_acc"`` — top-1 accuracy in [0, 1]
        * ``"top5_acc"`` — top-5 accuracy in [0, 1]
        * ``"n_samples"`` — total number of samples processed

        Returns an empty list when :meth:`process_predictions` has never
        been called (no samples accumulated yet).
        """
        if self._n_total == 0:
            return []
        return [
            {
                "top1_acc": round(self.top1, 6),
                "top5_acc": round(self.top5, 6),
                "n_samples": self._n_total,
            }
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all metrics."""
        return {
            "results": self.results_dict,
            "speed": self.speed,
            "summary": self.summary(),
        }

    def to_json(self) -> str:
        """Serialise metrics to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        """Serialise metrics to a CSV string.

        Returns a header row plus one data row with top-1 and top-5
        accuracy and total sample count.
        """
        fieldnames = ["top1_acc", "top5_acc", "n_samples"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.summary():
            writer.writerow(row)
        return buf.getvalue()
