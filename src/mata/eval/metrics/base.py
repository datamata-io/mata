"""Base metric container and ap_per_class().

Implements COCO-style 101-point Average Precision computation and the
``Metric`` accumulator class used by all task-specific metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mata.eval.metrics.iou import COCO_IOU_THRESHOLDS

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point COCO AP via trapezoidal interpolation.

    Appends sentinel values (0, 0) and (1, 0) so the area under the
    PR curve starts and ends at zero, matching pycocotools behaviour.

    Args:
        recall:    Monotonically increasing recall values, shape (K,).
        precision: Corresponding precision values, shape (K,).

    Returns:
        Scalar AP (float).
    """
    # Append boundary sentinels
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing (envelope)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101 equally-spaced recall points
    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    return float(ap)


# ---------------------------------------------------------------------------
# ap_per_class — Task B1
# ---------------------------------------------------------------------------


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_thresholds: list[float] = COCO_IOU_THRESHOLDS,
    eps: float = 1e-16,
) -> tuple:
    """Compute per-class Average Precision at each IoU threshold.

    This is the statistical core of the MATA evaluation system.  It
    mirrors the Ultralytics YOLO ``ap_per_class`` signature so that
    ``Metric.update()`` can consume its output directly.

    The function operates in **single-threshold mode**: ``tp`` encodes
    TP status at *one* IoU threshold.  The ``Validator`` calls this
    once per COCO threshold and aggregates ``all_ap`` externally.
    Passing the full ``COCO_IOU_THRESHOLDS`` list here is only needed
    when ``tp`` already encodes multi-threshold matches (shape (N, T)).

    Args:
        tp:             (N,) bool *or* (N, T) bool — TP flag per detection,
                        optionally at T IoU thresholds simultaneously.
        conf:           (N,) float32 — confidence scores in [0, 1].
        pred_cls:       (N,) int — predicted class IDs.
        target_cls:     (M,) int — ground-truth class IDs (all images).
        iou_thresholds: T IoU thresholds (default: 10-point COCO sweep).
        eps:            Small constant to prevent division by zero.

    Returns:
        12-element tuple:

        * ``tp_at_max_f1``  — (nc,) int, TP count at max-F1 confidence
        * ``fp_at_max_f1``  — (nc,) int, FP count at max-F1 confidence
        * ``p``             — (nc,) float32, precision at max-F1
        * ``r``             — (nc,) float32, recall at max-F1
        * ``f1``            — (nc,) float32, F1 at max-F1
        * ``all_ap``        — (nc, T) float32, AP at each IoU threshold
        * ``unique_classes``— (nc,) int, class IDs that have ≥1 GT instance
        * ``p_curve``       — (nc, 1000) float32, precision vs confidence
        * ``r_curve``       — (nc, 1000) float32, recall vs confidence
        * ``f1_curve``      — (nc, 1000) float32, F1 vs confidence
        * ``x``             — (1000,) float32, confidence axis [0→1]
        * ``prec_values``   — (nc, 1000) float32, precision at AP50 recall
    """
    tp = np.asarray(tp)
    conf = np.asarray(conf, dtype=np.float32)
    pred_cls = np.asarray(pred_cls, dtype=np.int32)
    target_cls = np.asarray(target_cls, dtype=np.int32)

    n_pred = len(conf)
    n_gt = len(target_cls)
    n_thr = len(iou_thresholds)

    # Shared confidence axis for curves
    x = np.linspace(0, 1, 1000, dtype=np.float32)

    # Classes that have at least one GT instance
    unique_classes, nt_per_class = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)

    # ------------------------------------------------------------------
    # Degenerate: no predictions or no GT
    # ------------------------------------------------------------------
    if n_pred == 0 or n_gt == 0 or nc == 0:
        return (
            np.zeros(nc, dtype=np.int32),  # tp_at_max_f1
            np.zeros(nc, dtype=np.int32),  # fp_at_max_f1
            np.zeros(nc, dtype=np.float32),  # p
            np.zeros(nc, dtype=np.float32),  # r
            np.zeros(nc, dtype=np.float32),  # f1
            np.zeros((nc, n_thr), dtype=np.float32),  # all_ap
            unique_classes,  # unique_classes
            np.zeros((nc, 1000), dtype=np.float32),  # p_curve
            np.zeros((nc, 1000), dtype=np.float32),  # r_curve
            np.zeros((nc, 1000), dtype=np.float32),  # f1_curve
            x,  # x
            np.zeros((nc, 1000), dtype=np.float32),  # prec_values
        )

    # ------------------------------------------------------------------
    # Normalise tp to 2-D: (N, T)
    # ------------------------------------------------------------------
    if tp.ndim == 1:
        # Single threshold — broadcast to all T thresholds
        tp2d = np.tile(tp[:, np.newaxis].astype(bool), (1, n_thr))  # (N, T)
    else:
        tp2d = tp.astype(bool)  # already (N, T)
        if tp2d.shape[1] != n_thr:
            raise ValueError(f"tp has {tp2d.shape[1]} threshold columns but " f"{n_thr} thresholds were provided.")

    # Sort all detections by descending confidence
    sort_idx = np.argsort(-conf)
    tp2d = tp2d[sort_idx]  # (N, T)
    conf_s = conf[sort_idx]  # (N,)
    pred_cls_s = pred_cls[sort_idx]  # (N,)

    # ------------------------------------------------------------------
    # Per-class computation
    # ------------------------------------------------------------------
    all_ap = np.zeros((nc, n_thr), dtype=np.float32)
    p_out = np.zeros(nc, dtype=np.float32)
    r_out = np.zeros(nc, dtype=np.float32)
    f1_out = np.zeros(nc, dtype=np.float32)
    tp_out = np.zeros(nc, dtype=np.int32)
    fp_out = np.zeros(nc, dtype=np.int32)
    p_curve = np.zeros((nc, 1000), dtype=np.float32)
    r_curve = np.zeros((nc, 1000), dtype=np.float32)
    f1_curve = np.zeros((nc, 1000), dtype=np.float32)
    prec_values = np.zeros((nc, 1000), dtype=np.float32)

    for ci, cls in enumerate(unique_classes):
        n_gt_cls = int(nt_per_class[ci])

        # Detections for this class
        mask = pred_cls_s == cls
        tp_cls = tp2d[mask]  # (n_det_cls, T)
        conf_cls = conf_s[mask]  # (n_det_cls,)
        n_det = len(conf_cls)

        if n_det == 0:
            # No predictions for this class — AP stays 0
            continue

        # ------ AP at each IoU threshold ---------------------------------
        for ti in range(n_thr):
            tp_t = tp_cls[:, ti].astype(np.float32)  # (n_det_cls,)
            fp_t = 1.0 - tp_t

            cum_tp = np.cumsum(tp_t)
            cum_fp = np.cumsum(fp_t)

            recall = cum_tp / (n_gt_cls + eps)
            precision = cum_tp / (cum_tp + cum_fp + eps)

            all_ap[ci, ti] = _compute_ap(recall, precision)

        # ------ Precision / Recall / F1 curves at IoU thr[0] (AP50) -----
        tp_t0 = tp_cls[:, 0].astype(np.float32)
        fp_t0 = 1.0 - tp_t0

        # Confidence-sorted curves over 1000 confidence levels
        # For each threshold in x, count detections with conf >= x[i]
        # then compute precision and recall at that operating point
        p_at_x = np.zeros(1000, dtype=np.float32)
        r_at_x = np.zeros(1000, dtype=np.float32)

        for xi, thr in enumerate(x):
            above = conf_cls >= thr
            tp_above = tp_t0[above].sum()
            fp_above = fp_t0[above].sum()
            p_at_x[xi] = tp_above / (tp_above + fp_above + eps)
            r_at_x[xi] = tp_above / (n_gt_cls + eps)

        p_curve[ci] = p_at_x
        r_curve[ci] = r_at_x
        f1_c = 2 * p_at_x * r_at_x / (p_at_x + r_at_x + eps)
        f1_curve[ci] = f1_c

        # Precision values used for PR-curve plotting (same as p_curve at AP50)
        prec_values[ci] = p_at_x

        # Operating point: confidence that maximises F1
        best_xi = int(np.argmax(f1_c))
        p_out[ci] = float(p_at_x[best_xi])
        r_out[ci] = float(r_at_x[best_xi])
        f1_out[ci] = float(f1_c[best_xi])

        # TP / FP counts at the max-F1 threshold
        best_thr = float(x[best_xi])
        above_best = conf_cls >= best_thr
        tp_out[ci] = int(tp_t0[above_best].sum())
        fp_out[ci] = int(fp_t0[above_best].sum())

    return (
        tp_out,  # (nc,) int
        fp_out,  # (nc,) int
        p_out,  # (nc,) float32
        r_out,  # (nc,) float32
        f1_out,  # (nc,) float32
        all_ap,  # (nc, T) float32
        unique_classes,  # (nc,) int
        p_curve,  # (nc, 1000) float32
        r_curve,  # (nc, 1000) float32
        f1_curve,  # (nc, 1000) float32
        x,  # (1000,) float32
        prec_values,  # (nc, 1000) float32
    )


# ---------------------------------------------------------------------------
# Metric container — Task B2
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    """Per-task accuracy metrics container (YOLO ``Metric`` equivalent).

    Stores the output of :func:`ap_per_class` and exposes YOLO-style
    scalar properties (``map``, ``map50``, ``map75``, ``maps``, ``mp``,
    ``mr``) as well as per-class helpers used by the printer.

    All properties return ``0.0`` / empty arrays before :meth:`update`
    is called — no ``AttributeError`` on an un-populated metric object.
    """

    p: list[float] = field(default_factory=list)
    r: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    all_ap: np.ndarray = field(default_factory=lambda: np.zeros((0, 10), dtype=np.float32))
    ap_class_index: list[int] = field(default_factory=list)
    p_curve: np.ndarray = field(default_factory=lambda: np.zeros((0, 1000), dtype=np.float32))
    r_curve: np.ndarray = field(default_factory=lambda: np.zeros((0, 1000), dtype=np.float32))
    f1_curve: np.ndarray = field(default_factory=lambda: np.zeros((0, 1000), dtype=np.float32))
    prec_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 1000), dtype=np.float32))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ap50(self) -> np.ndarray:
        """AP at IoU = 0.50, shape (nc,)."""
        return self.all_ap[:, 0] if self.all_ap.shape[1] > 0 else self.all_ap[:, 0:0].ravel()

    @property
    def ap(self) -> np.ndarray:
        """Mean AP over all IoU thresholds, shape (nc,)."""
        return self.all_ap.mean(axis=1) if self.all_ap.size > 0 else np.zeros(0, dtype=np.float32)

    @property
    def map50(self) -> float:
        """Mean AP at IoU = 0.50 across all classes."""
        v = self.ap50
        return float(v.mean()) if v.size > 0 else 0.0

    @property
    def map75(self) -> float:
        """Mean AP at IoU = 0.75 across all classes.

        Index 5 in ``COCO_IOU_THRESHOLDS`` corresponds to 0.75.
        """
        if self.all_ap.shape[1] > 5:
            return float(self.all_ap[:, 5].mean()) if self.all_ap.shape[0] > 0 else 0.0
        return 0.0

    @property
    def map(self) -> float:
        """Mean AP @ IoU 0.50:0.95 across all classes."""
        v = self.ap
        return float(v.mean()) if v.size > 0 else 0.0

    @property
    def maps(self) -> np.ndarray:
        """Per-class mAP50-95, shape (nc,)."""
        return self.ap

    @property
    def mp(self) -> float:
        """Mean precision across all classes."""
        return float(np.mean(self.p)) if self.p else 0.0

    @property
    def mr(self) -> float:
        """Mean recall across all classes."""
        return float(np.mean(self.r)) if self.r else 0.0

    @property
    def curves(self) -> list[str]:
        """Curve display names."""
        return ["F1_curve", "PR_curve", "P_curve", "R_curve"]

    @property
    def curves_results(self) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
        """Raw curve data consumed by :mod:`mata.eval.plots`.

        Returns a list of 4 ``(x, y, names)`` tuples, one per curve type:
        F1, PR, P, R.  For the PR curve, ``x`` is the mean recall across
        classes (1-D), and ``y`` is ``prec_values`` (nc, 1000).  For the
        other three, ``x`` is the confidence axis (1-D, 1000 values).
        """
        x = np.linspace(0, 1, 1000, dtype=np.float32)
        names = [str(i) for i in self.ap_class_index]
        # Mean recall across classes for the PR-curve x-axis
        mean_recall = self.r_curve.mean(axis=0) if self.r_curve.shape[0] > 0 else x
        return [
            (x, self.f1_curve, names),  # F1_curve
            (mean_recall, self.prec_values, names),  # PR_curve (x=recall)
            (x, self.p_curve, names),  # P_curve
            (x, self.r_curve, names),  # R_curve
        ]

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def update(self, results: tuple) -> None:
        """Populate internal state from the output of :func:`ap_per_class`.

        Args:
            results: The 12-element tuple returned by :func:`ap_per_class`.
        """
        (
            _tp_mf1,
            _fp_mf1,
            p,
            r,
            f1,
            all_ap,
            unique_classes,
            p_curve,
            r_curve,
            f1_curve,
            _x,
            prec_values,
        ) = results

        self.p = list(p.astype(float))
        self.r = list(r.astype(float))
        self.f1 = list(f1.astype(float))
        self.all_ap = np.asarray(all_ap, dtype=np.float32)
        self.ap_class_index = list(unique_classes.astype(int))
        self.p_curve = np.asarray(p_curve, dtype=np.float32)
        self.r_curve = np.asarray(r_curve, dtype=np.float32)
        self.f1_curve = np.asarray(f1_curve, dtype=np.float32)
        self.prec_values = np.asarray(prec_values, dtype=np.float32)

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return ``(precision, recall, ap50, ap50-95)`` for class index *i*.

        Args:
            i: Index into the arrays (not the class ID itself).
        """
        return (
            float(self.p[i]),
            float(self.r[i]),
            float(self.ap50[i]),
            float(self.ap[i]),
        )

    def mean_results(self) -> list[float]:
        """Return ``[mp, mr, map50, map]`` — used by the console printer."""
        return [self.mp, self.mr, self.map50, self.map]

    def fitness(self) -> float:
        """Scalar fitness score: ``0.1 * map50 + 0.9 * map``."""
        return 0.1 * self.map50 + 0.9 * self.map
