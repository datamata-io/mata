"""Detection metrics (DetMetrics) — Task C1 implementation."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mata.eval.metrics.base import Metric

if TYPE_CHECKING:
    from mata.eval.confusion_matrix import ConfusionMatrix


@dataclass
class DetMetrics:
    """YOLO-style detection metrics container for bounding-box AP.

    The :class:`~mata.eval.metrics.base.Metric` instance at ``.box``
    is populated by the `Validator` (or directly via
    ``box.update(ap_per_class(...))``) before any properties are read.

    Args:
        names:    Mapping from integer class ID to human-readable name,
                  e.g. ``{0: "cat", 1: "dog"}``.

    Example::

        metrics = DetMetrics(names={0: "cat", 1: "dog"})
        metrics.box.update(ap_per_class(...))
        print(metrics.box.map)   # mAP @ 0.50:0.95
    """

    #: Mapping class ID → class name used by the printer and summary.
    names: dict[int, str] = field(default_factory=dict)

    #: Inner metric container for bounding-box AP.
    box: Metric = field(default_factory=Metric)

    #: Timing breakdown in ms/image.
    speed: dict[str, float] = field(
        default_factory=lambda: {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
    )

    #: Optional confusion matrix attached by the Validator.
    confusion_matrix: ConfusionMatrix | None = None

    #: Directory where plots are saved.  Empty string = no plots.
    save_dir: str = ""

    # ------------------------------------------------------------------
    # Delegated AP properties
    # ------------------------------------------------------------------

    @property
    def maps(self) -> np.ndarray:
        """Per-class mAP50-95 — same as ``self.box.maps``, shape ``(nc,)``."""
        return self.box.maps

    @property
    def ap_class_index(self) -> list[int]:
        """Class indices present in the evaluation (have ≥1 GT instance)."""
        return self.box.ap_class_index

    # ------------------------------------------------------------------
    # Metric interface (mirrors Metric API at the DetMetrics level)
    # ------------------------------------------------------------------

    def mean_results(self) -> list[float]:
        """Return ``[mp, mr, map50, map]`` — used by the console printer."""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return per-class ``(precision, recall, ap50, ap50-95)`` for index *i*."""
        return self.box.class_result(i)

    def fitness(self) -> float:
        """Scalar fitness score: ``0.1 * map50 + 0.9 * map`` (YOLO formula)."""
        return self.box.fitness()

    # ------------------------------------------------------------------
    # Logging / export helpers
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Metric keys (without ``"fitness"``) for use by loggers."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    @property
    def results_dict(self) -> dict[str, float]:
        """Flat metrics dict for logging and external tools.

        Returns a dict with exactly 5 keys::

            {
                "metrics/precision(B)":   <float>,
                "metrics/recall(B)":      <float>,
                "metrics/mAP50(B)":       <float>,
                "metrics/mAP50-95(B)":    <float>,
                "fitness":                <float>,
            }
        """
        return {
            "metrics/precision(B)": self.box.mp,
            "metrics/recall(B)": self.box.mr,
            "metrics/mAP50(B)": self.box.map50,
            "metrics/mAP50-95(B)": self.box.map,
            "fitness": self.fitness(),
        }

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return a per-class summary list for human consumption.

        Each entry is a dict with keys:

        * ``class_id``     — integer class ID
        * ``class_name``   — name string (or ``str(class_id)`` if unknown)
        * ``num_targets``  — placeholder (``-1``; GT counts are in the
          ``Validator``; kept for API symmetry)
        * ``precision``    — float
        * ``recall``       — float
        * ``f1_score``     — float
        * ``ap50``         — AP at IoU 0.50
        * ``ap50_95``      — mAP @ IoU 0.50:0.95

        Returns an empty list when no classes have been evaluated yet.
        """
        rows: list[dict[str, Any]] = []
        for rank, cls_id in enumerate(self.box.ap_class_index):
            p, r, ap50, ap5095 = self.box.class_result(rank)
            f1 = 2.0 * p * r / (p + r + 1e-16)
            rows.append(
                {
                    "class_id": int(cls_id),
                    "class_name": self.names.get(int(cls_id), str(cls_id)),
                    "num_targets": -1,
                    "precision": round(p, 6),
                    "recall": round(r, 6),
                    "f1_score": round(f1, 6),
                    "ap50": round(ap50, 6),
                    "ap50_95": round(ap5095, 6),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all metrics."""
        return {
            "results": self.results_dict,
            "speed": self.speed,
            "per_class": self.summary(),
        }

    def to_json(self) -> str:
        """Serialise metrics to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        """Serialise *per-class* metrics to a CSV string.

        The CSV has one header row followed by one row per class.
        Returns a header-only string when no classes have been evaluated.
        """
        fieldnames = [
            "class_id",
            "class_name",
            "num_targets",
            "precision",
            "recall",
            "f1_score",
            "ap50",
            "ap50_95",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.summary():
            writer.writerow(row)
        return buf.getvalue()
