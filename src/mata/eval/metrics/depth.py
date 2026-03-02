"""Depth estimation metrics (DepthMetrics) — Task C4 implementation.

Implements the standard Eigen et al. benchmark metrics used by
Depth Anything, MiDaS, DPT and related monocular depth models:

  * AbsRel, SqRel, RMSE, log-RMSE  (error — lower is better)
  * δ₁, δ₂, δ₃ threshold accuracy  (higher is better)

Running-accumulation pattern: call ``process_batch()`` for each image,
then call ``finalize()`` once to average over all images.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DepthMetrics:
    """Standard depth estimation accuracy metrics (Eigen et al.).

    Usage::

        metrics = DepthMetrics(align_scale=True)
        for pred, gt in dataset:
            metrics.process_batch(pred, gt)
        metrics.finalize()

        print(metrics.abs_rel)   # mean absolute relative error
        print(metrics.delta_1)   # % pixels with δ < 1.25

    Args:
        align_scale: When ``True`` (default), apply median scaling to
            align the predicted depth map to the ground-truth scale
            before computing metrics.  Required when the model outputs
            relative (scale-less) depth (e.g. Depth Anything, MiDaS).
        align_affine: When ``True``, use **least-squares scale+shift**
            alignment instead of median scaling.  This is the correct
            alignment for *affine-invariant* models such as Depth Anything
            V2, MiDaS, and DPT that output uncalibrated disparity.
            Supersedes ``align_scale`` when both are ``True``.

    Attributes:
        abs_rel:  Mean absolute relative error: mean |d̂ − d| / d
        sq_rel:   Mean squared relative error: mean (d̂ − d)² / d
        rmse:     Root mean squared error: √mean((d̂ − d)²)
        log_rmse: Log-space RMSE: √mean((log d̂ − log d)²)
        delta_1:  % pixels where max(d̂/d, d/d̂) < 1.25
        delta_2:  % pixels where max(d̂/d, d/d̂) < 1.25²
        delta_3:  % pixels where max(d̂/d, d/d̂) < 1.25³
        speed:    Timing breakdown in ms/image.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    #: Apply median scaling to align relative depth to GT scale.
    align_scale: bool = True
    #: Apply least-squares scale+shift alignment (affine-invariant models).
    #: Supersedes align_scale when True.
    align_affine: bool = False

    # ------------------------------------------------------------------
    # Accumulated metric fields (set by finalize())
    # ------------------------------------------------------------------

    #: Mean absolute relative error
    abs_rel: float = 0.0
    #: Mean squared relative error
    sq_rel: float = 0.0
    #: Root mean squared error
    rmse: float = 0.0
    #: Log-space RMSE
    log_rmse: float = 0.0
    #: δ₁ — % pixels with max(d̂/d, d/d̂) < 1.25
    delta_1: float = 0.0
    #: δ₂ — % pixels with max(d̂/d, d/d̂) < 1.25²
    delta_2: float = 0.0
    #: δ₃ — % pixels with max(d̂/d, d/d̂) < 1.25³
    delta_3: float = 0.0

    #: Timing breakdown in ms/image.
    speed: dict[str, float] = field(
        default_factory=lambda: {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # Private accumulators (not part of the public API)
    # ------------------------------------------------------------------

    #: Running sums for each metric (one entry per image batch).
    _sums: dict[str, float] = field(
        default_factory=lambda: {
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "rmse": 0.0,
            "log_rmse": 0.0,
            "delta_1": 0.0,
            "delta_2": 0.0,
            "delta_3": 0.0,
        },
        repr=False,
    )

    #: Number of images processed so far.
    _count: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def process_batch(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        """Accumulate metrics for one image.

        Args:
            pred_depth: ``(H, W)`` predicted depth map (metres or relative
                units — scale is aligned to GT when ``align_scale=True``).
            gt_depth:   ``(H, W)`` ground-truth depth map, same spatial
                resolution as *pred_depth*.
            valid_mask: Optional ``(H, W)`` boolean mask.  Only pixels
                where *valid_mask* is ``True`` are evaluated.  When
                ``None`` the mask is derived automatically by excluding
                pixels where ``gt_depth <= 0`` or ``gt_depth == inf``.

        Raises:
            ValueError: If *pred_depth* and *gt_depth* have different shapes
                or if the valid region is empty after masking.
        """
        pred_depth = np.asarray(pred_depth, dtype=np.float64)
        gt_depth = np.asarray(gt_depth, dtype=np.float64)

        if pred_depth.shape != gt_depth.shape:
            raise ValueError(f"pred_depth shape {pred_depth.shape} != " f"gt_depth shape {gt_depth.shape}")

        # Build validity mask: user-supplied OR auto-derived
        auto_mask = np.isfinite(gt_depth) & (gt_depth > 0.0)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool) & auto_mask
        else:
            valid_mask = auto_mask

        if not np.any(valid_mask):
            raise ValueError(
                "No valid pixels found after masking.  " "Ensure gt_depth contains positive, finite values."
            )

        pred = pred_depth[valid_mask]
        gt = gt_depth[valid_mask]

        # Alignment for relative-depth / affine-invariant models
        if self.align_affine:
            # Least-squares scale+shift: minimise ||s*pred + t - gt||²
            # Normal equations: [sum(p²)  sum(p) ] [s]   [sum(p*g)]
            #                   [sum(p)   n      ] [t] = [sum(g)  ]
            p = pred
            g = gt
            sum_pp = float(np.dot(p, p))
            sum_p = float(p.sum())
            sum_pg = float(np.dot(p, g))
            sum_g = float(g.sum())
            n_valid = float(p.size)
            denom = sum_pp * n_valid - sum_p * sum_p
            if abs(denom) > 1e-12:
                s = (sum_pg * n_valid - sum_p * sum_g) / denom
                t = (sum_g - s * sum_p) / n_valid
                pred = s * pred + t
        elif self.align_scale:
            # Median-scale-only alignment (legacy)
            median_gt = np.median(gt)
            median_pred = np.median(pred)
            if median_pred > 0.0:
                scale = median_gt / median_pred
                pred = pred * scale

        # Clamp predictions to be strictly positive (log safety)
        pred = np.maximum(pred, 1e-9)

        float(pred.size)

        # --- Error metrics ---
        diff = pred - gt
        abs_rel = float(np.mean(np.abs(diff) / gt))
        sq_rel = float(np.mean((diff**2) / gt))
        rmse = float(np.sqrt(np.mean(diff**2)))
        log_rmse = float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))

        # --- Threshold accuracy metrics ---
        ratio = np.maximum(pred / gt, gt / pred)
        delta_1 = float(np.mean(ratio < 1.25))
        delta_2 = float(np.mean(ratio < 1.25**2))
        delta_3 = float(np.mean(ratio < 1.25**3))

        # Accumulate
        self._sums["abs_rel"] += abs_rel
        self._sums["sq_rel"] += sq_rel
        self._sums["rmse"] += rmse
        self._sums["log_rmse"] += log_rmse
        self._sums["delta_1"] += delta_1
        self._sums["delta_2"] += delta_2
        self._sums["delta_3"] += delta_3
        self._count += 1

    def finalize(self) -> None:
        """Average accumulated metrics over all processed images.

        Must be called once after all ``process_batch()`` calls.
        Calling ``finalize()`` on a fresh (zero-image) accumulator sets
        all metrics to 0.0.  Calling it a second time overwrites previous
        values — always call after completing a full evaluation loop.
        """
        n = max(self._count, 1)
        self.abs_rel = self._sums["abs_rel"] / n
        self.sq_rel = self._sums["sq_rel"] / n
        self.rmse = self._sums["rmse"] / n
        self.log_rmse = self._sums["log_rmse"] / n
        self.delta_1 = self._sums["delta_1"] / n
        self.delta_2 = self._sums["delta_2"] / n
        self.delta_3 = self._sums["delta_3"] / n

    # ------------------------------------------------------------------
    # Legacy / Validator interface
    # ------------------------------------------------------------------

    def update(self, pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        """Alias for :meth:`process_batch` (Validator compatibility)."""
        self.process_batch(pred_depth, gt_depth, valid_mask)

    def mean_results(self) -> list[float]:
        """Return ``[abs_rel, sq_rel, rmse, log_rmse, delta_1, delta_2, delta_3]``.

        Used by the console printer to populate the summary row.
        """
        return [
            self.abs_rel,
            self.sq_rel,
            self.rmse,
            self.log_rmse,
            self.delta_1,
            self.delta_2,
            self.delta_3,
        ]

    # ------------------------------------------------------------------
    # Fitness & diagnostics
    # ------------------------------------------------------------------

    @property
    def fitness(self) -> float:  # type: ignore[override]
        """Scalar quality score: ``delta_1 - abs_rel``.

        Higher is better.  Increasing δ₁ pushes the score up; increasing
        AbsRel pushes it down, so both accuracy and error direction
        are captured in a single number.

        Returns:
            Float in approximately ``[-1, 1]``.
        """
        return self.delta_1 - self.abs_rel

    @property
    def keys(self) -> list[str]:
        """Metric keys (without ``"fitness"``) for use by loggers."""
        return [
            "metrics/abs_rel",
            "metrics/sq_rel",
            "metrics/rmse",
            "metrics/log_rmse",
            "metrics/delta_1",
            "metrics/delta_2",
            "metrics/delta_3",
        ]

    @property
    def results_dict(self) -> dict[str, float]:
        """Flat metrics dict for logging and external tools.

        Returns a dict with exactly 8 keys (7 metrics + fitness)::

            {
                "metrics/abs_rel":  <float>,
                "metrics/sq_rel":   <float>,
                "metrics/rmse":     <float>,
                "metrics/log_rmse": <float>,
                "metrics/delta_1":  <float>,
                "metrics/delta_2":  <float>,
                "metrics/delta_3":  <float>,
                "fitness":          <float>,
            }
        """
        return {
            "metrics/abs_rel": self.abs_rel,
            "metrics/sq_rel": self.sq_rel,
            "metrics/rmse": self.rmse,
            "metrics/log_rmse": self.log_rmse,
            "metrics/delta_1": self.delta_1,
            "metrics/delta_2": self.delta_2,
            "metrics/delta_3": self.delta_3,
            "fitness": self.fitness,
        }

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Return a single-row summary list for human consumption.

        The list always has exactly one entry — depth estimation produces
        a global scalar summary rather than per-class rows.

        Returns:
            A list with one dict containing all seven metrics and fitness.
        """
        return [
            {
                "abs_rel": round(self.abs_rel, 6),
                "sq_rel": round(self.sq_rel, 6),
                "rmse": round(self.rmse, 6),
                "log_rmse": round(self.log_rmse, 6),
                "delta_1": round(self.delta_1, 6),
                "delta_2": round(self.delta_2, 6),
                "delta_3": round(self.delta_3, 6),
                "fitness": round(self.fitness, 6),
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

        Returns a header row followed by one data row with all metrics.
        """
        fieldnames = [
            "abs_rel",
            "sq_rel",
            "rmse",
            "log_rmse",
            "delta_1",
            "delta_2",
            "delta_3",
            "fitness",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.summary():
            writer.writerow(row)
        return buf.getvalue()
