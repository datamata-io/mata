"""Segmentation metrics (SegmentMetrics) — Task C2 implementation."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mata.eval.metrics.base import Metric
from mata.eval.metrics.detect import DetMetrics

if TYPE_CHECKING:
    pass


@dataclass
class SegmentMetrics(DetMetrics):
    """YOLO-style segmentation metrics container.

    Extends :class:`~mata.eval.metrics.detect.DetMetrics` with a second
    :class:`~mata.eval.metrics.base.Metric` instance (``.seg``) for
    mask-level AP.  The ``Validator`` populates ``.box`` via box IoU and
    ``.seg`` via mask IoU independently.

    Args:
        names: Mapping from integer class ID to human-readable name.

    Example::

        metrics = SegmentMetrics(names={0: "cat", 1: "dog"})
        metrics.box.update(ap_per_class(...))   # box AP
        metrics.seg.update(ap_per_class(...))   # mask AP
        print(metrics.box.map50, metrics.seg.map50)
    """

    #: Mask-level AP container (populated independently of ``.box``).
    seg: Metric = field(default_factory=Metric)

    # ------------------------------------------------------------------
    # Overridden properties
    # ------------------------------------------------------------------

    @property
    def maps(self) -> np.ndarray:
        """Element-wise mean of ``box.maps`` and ``seg.maps``, shape ``(nc,)``.

        When the two metrics cover different class sets (different sizes),
        returns the mean over the shorter length to avoid index errors.
        """
        bm = self.box.maps
        sm = self.seg.maps
        n = min(len(bm), len(sm))
        if n == 0:
            # Fall back to whichever is non-empty, else empty array
            if bm.size > 0:
                return bm
            return sm
        return (bm[:n] + sm[:n]) / 2.0

    @property
    def fitness(self) -> float:  # type: ignore[override]
        """Combined fitness: ``0.5 * box.fitness() + 0.5 * seg.fitness()``."""
        return 0.5 * self.box.fitness() + 0.5 * self.seg.fitness()

    # ------------------------------------------------------------------
    # Overridden keys / results_dict
    # ------------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        """Eight metric keys (box + mask) for loggers."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    @property
    def results_dict(self) -> dict[str, float]:
        """Flat metrics dict with 8 metric keys + ``"fitness"``.

        Returns::

            {
                "metrics/precision(B)":   <float>,
                "metrics/recall(B)":      <float>,
                "metrics/mAP50(B)":       <float>,
                "metrics/mAP50-95(B)":    <float>,
                "metrics/precision(M)":   <float>,
                "metrics/recall(M)":      <float>,
                "metrics/mAP50(M)":       <float>,
                "metrics/mAP50-95(M)":    <float>,
                "fitness":                <float>,
            }
        """
        return {
            "metrics/precision(B)": self.box.mp,
            "metrics/recall(B)": self.box.mr,
            "metrics/mAP50(B)": self.box.map50,
            "metrics/mAP50-95(B)": self.box.map,
            "metrics/precision(M)": self.seg.mp,
            "metrics/recall(M)": self.seg.mr,
            "metrics/mAP50(M)": self.seg.map50,
            "metrics/mAP50-95(M)": self.seg.map,
            "fitness": self.fitness,
        }

    # ------------------------------------------------------------------
    # Overridden metric interface
    # ------------------------------------------------------------------

    def mean_results(self) -> list[float]:
        """Return ``[box_mp, box_mr, box_map50, box_map, seg_mp, seg_mr, seg_map50, seg_map]``."""
        return [
            self.box.mp,
            self.box.mr,
            self.box.map50,
            self.box.map,
            self.seg.mp,
            self.seg.mr,
            self.seg.map50,
            self.seg.map,
        ]

    def class_result(self, i: int) -> tuple[float, float, float, float, float, float, float, float]:
        """Return per-class ``(box_p, box_r, box_ap50, box_ap, seg_p, seg_r, seg_ap50, seg_ap)``."""
        bp, br, bap50, bap = self.box.class_result(i)
        sp, sr, sap50, sap = self.seg.class_result(i)
        return (bp, br, bap50, bap, sp, sr, sap50, sap)

    # ------------------------------------------------------------------
    # Overridden summary / serialisation
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Per-class summary including both box and mask AP columns.

        Each dict has keys:

        * ``class_id``, ``class_name``, ``num_targets``
        * ``precision``, ``recall``, ``f1_score``
        * ``ap50``  (box AP @ 0.50)
        * ``ap50_95`` (box mAP @ 0.50:0.95)
        * ``mask_ap50``  (mask AP @ 0.50)
        * ``mask_ap50_95`` (mask mAP @ 0.50:0.95)
        """
        # Collect box rows first (reuse parent logic via Metric directly)
        box_index = self.box.ap_class_index
        seg_index = self.seg.ap_class_index

        # Build a union of class indices seen in either metric
        all_indices = sorted(set(box_index) | set(seg_index))

        rows: list[dict[str, Any]] = []
        for cls_id in all_indices:
            # box values
            if cls_id in box_index:
                rank = box_index.index(cls_id)
                bp, br, bap50, bap5095 = self.box.class_result(rank)
            else:
                bp = br = bap50 = bap5095 = 0.0

            # seg values
            if cls_id in seg_index:
                srank = seg_index.index(cls_id)
                sp, sr, sap50, sap5095 = self.seg.class_result(srank)
            else:
                sap50 = sap5095 = 0.0

            f1 = 2.0 * bp * br / (bp + br + 1e-16)
            rows.append(
                {
                    "class_id": int(cls_id),
                    "class_name": self.names.get(int(cls_id), str(cls_id)),
                    "num_targets": -1,
                    "precision": round(bp, 6),
                    "recall": round(br, 6),
                    "f1_score": round(f1, 6),
                    "ap50": round(bap50, 6),
                    "ap50_95": round(bap5095, 6),
                    "mask_ap50": round(sap50, 6),
                    "mask_ap50_95": round(sap5095, 6),
                }
            )
        return rows

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
        """Serialise per-class metrics to a CSV string (box + mask columns)."""
        fieldnames = [
            "class_id",
            "class_name",
            "num_targets",
            "precision",
            "recall",
            "f1_score",
            "ap50",
            "ap50_95",
            "mask_ap50",
            "mask_ap50_95",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.summary():
            writer.writerow(row)
        return buf.getvalue()
