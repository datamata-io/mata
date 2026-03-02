"""ConfusionMatrix — per-batch confusion matrix accumulator.

Supports both detection (nc+1 × nc+1 with background row/col) and
classification (nc × nc) accumulation modes.

Task E1 implementation.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


class ConfusionMatrix:
    """Accumulates a confusion matrix over validation batches.

    Operates at a **single** IoU threshold (default 0.45) and confidence
    threshold (default 0.25), unlike AP which sweeps thresholds.

    Matrix layout (detect mode):
        - Rows   = predicted class (0..nc-1) + background row (index nc)
        - Columns = GT class (0..nc-1) + background column (index nc)
        - Background row catches false-positives (no matching GT).
        - Background column catches false-negatives (no matching pred).

    Matrix layout (classify mode):
        - (nc × nc): ``matrix[pred, true]``

    Args:
        nc:             Number of foreground classes.
        names:          Mapping ``{class_id: name}`` — used for display only.
        task:           ``"detect"`` or ``"classify"``.
        conf_threshold: Minimum confidence for a detection to be considered.
        iou_threshold:  IoU threshold for a positive greedy match (detect only).
    """

    def __init__(
        self,
        nc: int,
        names: dict[int, str] | None = None,
        task: str = "detect",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if task not in {"detect", "classify"}:
            raise ValueError(f"task must be 'detect' or 'classify', got {task!r}")
        self.nc = nc
        self.names: dict[int, str] = names or {}
        self.task = task
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if task == "detect":
            # (nc+1) × (nc+1); last row/col index = background
            self._matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
        else:
            self._matrix = np.zeros((nc, nc), dtype=np.int64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """Accumulated count matrix.

        Shape ``(nc+1, nc+1)`` for detect; ``(nc, nc)`` for classify.
        """
        return self._matrix

    # ------------------------------------------------------------------
    # Accumulation — Detection
    # ------------------------------------------------------------------

    def process_batch(
        self,
        detections: list[tuple],
        labels: list[tuple],
    ) -> None:
        """Update confusion matrix for one image (detection mode).

        Args:
            detections: List of ``(bbox_xyxy, score, class_id)`` tuples for
                        predicted boxes.  Only entries with
                        ``score >= conf_threshold`` are considered.
            labels:     List of ``(bbox_xyxy, class_id)`` tuples for ground
                        truth boxes.

        Matrix update rules (greedy, highest-score-first):
        - Matched pred→GT pair: increment ``matrix[pred_cls, gt_cls]``.
        - Unmatched prediction (FP): increment ``matrix[pred_cls, nc]``
          (background column).
        - Unmatched GT (FN): increment ``matrix[nc, gt_cls]``
          (background row).
        """
        if self.task != "detect":
            raise RuntimeError("process_batch() is only valid in detect mode.")

        from mata.eval.metrics.iou import box_iou  # lazy – avoid circular import

        nc = self.nc
        bg = nc  # background index

        # --- filter by confidence -------------------------------------------
        dets = [d for d in detections if d[1] >= self.conf_threshold]
        # sort by confidence descending (greedy highest-first)
        dets = sorted(dets, key=lambda d: d[1], reverse=True)

        if len(dets) == 0 and len(labels) == 0:
            return

        pred_boxes = (
            np.array([d[0] for d in dets], dtype=np.float32).reshape(-1, 4)
            if dets
            else np.zeros((0, 4), dtype=np.float32)
        )
        pred_cls = np.array([d[2] for d in dets], dtype=np.int64) if dets else np.zeros(0, dtype=np.int64)

        gt_boxes = (
            np.array([lbl[0] for lbl in labels], dtype=np.float32).reshape(-1, 4)
            if labels
            else np.zeros((0, 4), dtype=np.float32)
        )
        gt_cls = np.array([lbl[1] for lbl in labels], dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)

        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        matched_gt: set[int] = set()

        if n_pred > 0 and n_gt > 0:
            iou_mat = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)

            for pi in range(n_pred):
                best_iou = -1.0
                best_gj = -1
                for gj in range(n_gt):
                    if gj in matched_gt:
                        continue
                    if iou_mat[pi, gj] > best_iou:
                        best_iou = iou_mat[pi, gj]
                        best_gj = gj

                if best_gj >= 0 and best_iou >= self.iou_threshold:
                    self._matrix[pred_cls[pi], gt_cls[best_gj]] += 1
                    matched_gt.add(best_gj)
                else:
                    # FP — no valid GT match
                    self._matrix[pred_cls[pi], bg] += 1
        elif n_pred > 0:
            # All predictions are FP
            for pi in range(n_pred):
                self._matrix[pred_cls[pi], bg] += 1

        # Unmatched GTs → FN (background row)
        for gj in range(n_gt):
            if gj not in matched_gt:
                self._matrix[bg, gt_cls[gj]] += 1

    # ------------------------------------------------------------------
    # Accumulation — Classification
    # ------------------------------------------------------------------

    def process_cls_preds(
        self,
        preds: list[int],
        targets: list[int],
    ) -> None:
        """Update confusion matrix for one batch (classification mode).

        Args:
            preds:   Predicted class IDs per sample.
            targets: Ground-truth class IDs per sample.
        """
        if self.task != "classify":
            raise RuntimeError("process_cls_preds() is only valid in classify mode.")

        for p, t in zip(preds, targets):
            if 0 <= p < self.nc and 0 <= t < self.nc:
                self._matrix[p, t] += 1

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-class true-positive and false-positive counts.

        For detect mode:
            - TP for class *c* = diagonal ``matrix[c, c]``.
            - FP for class *c* = sum of row *c* (all columns) minus TP,
              i.e. predictions of class *c* that were matched to a
              *different* GT class or to background.

        Returns:
            ``(tp, fp)`` each of shape ``(nc,)`` (background excluded).
        """
        m = self._matrix
        nc = self.nc
        diag = np.diag(m[:nc, :nc])
        row_sum = m[:nc].sum(axis=1)
        tp = diag.astype(np.float64)
        fp = (row_sum - diag).astype(np.float64)
        return tp, fp

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def plot(
        self,
        normalize: bool = True,
        save_dir: str | Path = "",
        names: dict[int, str] | None = None,
    ) -> None:
        """Save a confusion matrix heatmap as ``confusion_matrix.png``.

        Args:
            normalize: If ``True``, each column is divided by its sum so
                       values represent proportions (columns sum to 1).
                       Columns with zero sum remain zero.
            save_dir:  Directory to write the PNG.  If empty string ``""``
                       this method is a **no-op** (nothing is written).
            names:     Optional label override; defaults to ``self.names``.
        """
        if save_dir == "" or save_dir is None:
            return

        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        display_names = names or self.names
        nc = self.nc

        m = self._matrix.astype(np.float64)

        if normalize:
            col_sums = m.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(col_sums > 0, m / col_sums, 0.0)

        # Build tick labels
        if self.task == "detect":
            labels = [display_names.get(i, str(i)) for i in range(nc)] + ["background"]
        else:
            labels = [display_names.get(i, str(i)) for i in range(nc)]

        fig_size = max(8, len(labels) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(m, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
        plt.colorbar(im, ax=ax)

        tick_positions = np.arange(len(labels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("True label")
        ax.set_ylabel("Predicted label")
        ax.set_title("Confusion Matrix (normalised)" if normalize else "Confusion Matrix")

        # Annotate cells
        thresh = m.max() / 2.0 if m.max() > 0 else 0.5
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                val = m[i, j]
                text = f"{val:.2f}" if normalize else str(int(self._matrix[i, j]))
                ax.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black", fontsize=6)

        fig.tight_layout()
        out_path = save_dir / "confusion_matrix.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def print(self) -> None:
        """Print the raw confusion matrix to stdout."""
        nc = self.nc
        display_names: dict[int, str] = self.names or {}

        if self.task == "detect":
            labels = [display_names.get(i, str(i)) for i in range(nc)] + ["background"]
        else:
            labels = [display_names.get(i, str(i)) for i in range(nc)]

        col_w = max(len(lbl) for lbl in labels) if labels else 6
        col_w = max(col_w, 6)
        header = " " * (col_w + 2) + "  ".join(f"{lbl:>{col_w}}" for lbl in labels)
        print(header)
        for i, row_label in enumerate(labels):
            row = "  ".join(f"{int(self._matrix[i, j]):>{col_w}}" for j in range(len(labels)))
            print(f"{row_label:>{col_w}}  {row}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def summary(self, normalize: bool = False) -> list[dict]:
        """Return a list of dicts, one per class (foreground only).

        Each dict has keys: ``class_id``, ``class_name``, and one entry
        per class label for both rows (predicted as) and columns (GT is).

        When *normalize* is ``True``, values are column-normalised
        proportions.
        """
        nc = self.nc
        display_names = self.names or {}
        m = self._matrix.astype(np.float64)

        if normalize:
            col_sums = m.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(col_sums > 0, m / col_sums, 0.0)

        result = []
        for i in range(nc):
            row: dict[str, Any] = {
                "class_id": i,
                "class_name": display_names.get(i, str(i)),
            }
            size = self._matrix.shape[0]
            for j in range(size):
                key = display_names.get(j, str(j)) if j < nc else "background"
                row[key] = float(m[i, j])
            result.append(row)
        return result

    def to_json(self) -> str:
        """Serialise the raw matrix and metadata to a JSON string."""
        payload: dict[str, Any] = {
            "task": self.task,
            "nc": self.nc,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "names": {str(k): v for k, v in self.names.items()},
            "matrix": self._matrix.tolist(),
        }
        return json.dumps(payload, indent=2)

    def to_csv(self) -> str:
        """Serialise the raw matrix to a CSV string.

        First column is the row label; remaining columns are the GT labels.
        """
        nc = self.nc
        display_names = self.names or {}
        if self.task == "detect":
            labels = [display_names.get(i, str(i)) for i in range(nc)] + ["background"]
        else:
            labels = [display_names.get(i, str(i)) for i in range(nc)]

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([""] + labels)
        for i, row_label in enumerate(labels):
            writer.writerow([row_label] + [int(self._matrix[i, j]) for j in range(len(labels))])
        return buf.getvalue()
