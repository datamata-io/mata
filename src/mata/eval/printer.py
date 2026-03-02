"""Console table printer — YOLO-style formatted validation output.

Implements Task F2: per-class metrics table mirroring YOLO's validation output.

Example detect output::

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        100       1500      0.621      0.833      0.888      0.631
                person         80        500      0.721      0.700      0.853      0.543
                   dog         20        150      0.568      1.000      0.995      0.697

    Speed: 0.5ms preprocess, 3.2ms inference, 1.1ms postprocess per image
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.eval.metrics.classify import ClassifyMetrics
    from mata.eval.metrics.depth import DepthMetrics
    from mata.eval.metrics.detect import DetMetrics
    from mata.eval.metrics.segment import SegmentMetrics

# Column widths
_LEFT_WIDTH = 20  # class-name column (left-aligned)
_NUM_WIDTH = 12  # numeric columns (right-aligned)

# ---------------------------------------------------------------------------
# Header definitions — one tuple per task type
# ---------------------------------------------------------------------------
_HEADER_DETECT = (
    "Class",
    "Images",
    "Instances",
    "Box(P",
    "R",
    "mAP50",
    "mAP50-95)",
)
_HEADER_SEGMENT = (
    "Class",
    "Images",
    "Instances",
    "Box(P",
    "R",
    "mAP50",
    "mAP50-95)",
    "Mask(P",
    "R",
    "mAP50",
    "mAP50-95)",
)
_HEADER_CLASSIFY = ("top1_acc", "top5_acc")
_HEADER_DEPTH = (
    "abs_rel",
    "sq_rel",
    "RMSE",
    "log_RMSE",
    "δ<1.25",
    "δ<1.25²",
    "δ<1.25³",
)
_HEADER_OCR = ("CER", "WER", "Accuracy")


def _fmt(v: float) -> str:
    """Format a single float to 3 decimal places in a right-aligned field."""
    return f"{v:{_NUM_WIDTH}.3f}"


def _fmt_int(v: int) -> str:
    """Format an integer in a right-aligned field."""
    return f"{v:{_NUM_WIDTH}d}"


class Printer:
    """Formats and prints a YOLO-style validation results table.

    Example detect output::

                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                       all        100       1500      0.621      0.833      0.888      0.631
                    person         80        500      0.721      0.700      0.853      0.543
                       dog         20        150      0.568      1.000      0.995      0.697

        Speed: 0.5ms preprocess, 3.2ms inference, 1.1ms postprocess per image

    Args:
        names:    Mapping from class index to class name, e.g. ``{0: "cat", 1: "dog"}``.
        task:     One of ``"detect"``, ``"segment"``, ``"classify"``, ``"depth"``.
        save_dir: Directory for saving results alongside console output (unused by printer,
                  kept for API compatibility).
    """

    # Expose header tuples as class-level constants (matches spec)
    HEADER_DETECT = _HEADER_DETECT
    HEADER_SEGMENT = _HEADER_SEGMENT
    HEADER_CLASSIFY = _HEADER_CLASSIFY
    HEADER_DEPTH = _HEADER_DEPTH
    HEADER_OCR = _HEADER_OCR

    def __init__(
        self,
        names: dict[int, str] | None = None,
        task: str = "detect",
        save_dir: str = ".",
    ) -> None:
        self.names = names or {}
        self.task = task
        self.save_dir = save_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _header_for_task(self) -> tuple[str, ...]:
        """Return the header tuple for the current task."""
        return {
            "detect": _HEADER_DETECT,
            "segment": _HEADER_SEGMENT,
            "classify": _HEADER_CLASSIFY,
            "depth": _HEADER_DEPTH,
            "ocr": _HEADER_OCR,
        }.get(self.task, _HEADER_DETECT)

    def _format_header_line(self, header: tuple[str, ...]) -> str:
        """Build the full header string."""
        if self.task in ("classify", "depth", "ocr"):
            # No class/images/instances columns
            return "".join(f"{col:{_NUM_WIDTH}}" for col in header)
        # First column is class name (left-aligned), rest are numeric
        first = f"{header[0]:>{_LEFT_WIDTH}}"
        rest = "".join(f"{col:{_NUM_WIDTH}}" for col in header[1:])
        return first + rest

    def _format_row(self, label: str, counts: tuple[int, int] | None, values: list[float]) -> str:
        """Build a single results row string.

        Args:
            label:  Row label (class name or "all").
            counts: ``(n_images, n_instances)`` integers, or ``None`` for
                    classify / depth tasks that have no per-class counts.
            values: Numeric metric values.
        """
        if self.task in ("classify", "depth", "ocr"):
            return "".join(_fmt(v) for v in values)
        n_img, n_inst = counts if counts is not None else (0, 0)
        return f"{label:>{_LEFT_WIDTH}}" + _fmt_int(n_img) + _fmt_int(n_inst) + "".join(_fmt(v) for v in values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_header(self) -> None:
        """Print the table header row to stdout."""
        header = self._header_for_task()
        sys.stdout.write(self._format_header_line(header) + "\n")

    def print_row(self, label: str, results: list[float], counts: tuple[int, int] | None = None) -> None:
        """Print a single results row to stdout.

        Args:
            label:   Row label, e.g. ``"all"`` or ``"person"``.
            results: Numeric metric values corresponding to the non-label
                     header columns.
            counts:  ``(n_images, n_instances)`` for detect / segment tasks.
                     Pass ``None`` for classify / depth.
        """
        sys.stdout.write(self._format_row(label, counts, results) + "\n")

    def print_table(
        self,
        metrics: DetMetrics | SegmentMetrics | ClassifyMetrics | DepthMetrics,
        nc: dict[int, int] | None = None,
    ) -> None:
        """Print per-class metrics table to stdout.

        Prints the header, the "all" summary row first, then one row per
        class (suppressing classes with zero GT instances when *nc* is
        provided).

        Args:
            metrics: A populated metrics object (DetMetrics, SegmentMetrics,
                     ClassifyMetrics, or DepthMetrics).
            nc:      Optional mapping ``{class_id: gt_instance_count}``.
                     Classes with ``nc[class_id] == 0`` are suppressed.
        """
        # Lazy import to avoid circular dependencies at module load time.
        from mata.eval.metrics.classify import ClassifyMetrics
        from mata.eval.metrics.depth import DepthMetrics
        from mata.eval.metrics.ocr import OCRMetrics
        from mata.eval.metrics.segment import SegmentMetrics

        self.print_header()

        if isinstance(metrics, ClassifyMetrics):
            self._print_classify_table(metrics)
        elif isinstance(metrics, DepthMetrics):
            self._print_depth_table(metrics)
        elif isinstance(metrics, OCRMetrics):
            self._print_ocr_table(metrics)
        elif isinstance(metrics, SegmentMetrics):
            self._print_segment_table(metrics, nc)
        else:
            # DetMetrics (base case)
            self._print_detect_table(metrics, nc)

    def print_speed(
        self,
        speed: dict[str, float],
        image_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Print the speed summary line to stdout.

        Args:
            speed:       Mapping of stage name → ms/image, e.g.
                         ``{"preprocess": 0.5, "inference": 3.2, "postprocess": 1.1}``.
            image_shape: Optional ``(H, W)`` or ``(N, C, H, W)`` tuple printed
                         after the speed figures.
        """
        parts = [f"{v:.1f}ms {k}" for k, v in speed.items()]
        line = "Speed: " + ", ".join(parts) + " per image"
        if image_shape is not None:
            line += f" at shape {image_shape}"
        sys.stdout.write(line + "\n")

    # Convenience alias matching spec
    def print_results(self, metrics: Any, nc: dict[int, int] | None = None) -> None:
        """Print the full results table (header + rows + speed line).

        Args:
            metrics: Populated metrics object.
            nc:      Optional ``{class_id: gt_instance_count}`` filter.
        """
        self.print_table(metrics, nc=nc)
        speed = getattr(metrics, "speed", {})
        if speed:
            self.print_speed(speed)

    # ------------------------------------------------------------------
    # Task-specific table printers
    # ------------------------------------------------------------------

    def _print_detect_table(self, metrics: Any, nc: dict[int, int] | None) -> None:
        """Print 'all' row + per-class rows for detection metrics."""
        # Total image count is not stored directly in DetMetrics; use 0 as placeholder
        # when the caller has not attached it.
        n_images_total = getattr(metrics, "n_images", 0)
        n_instances_total = int(sum(nc.values())) if nc else 0

        all_results = metrics.mean_results()  # [mp, mr, map50, map]
        self._write_detect_row("all", n_images_total, n_instances_total, all_results)

        ap_index = list(metrics.ap_class_index)
        # Single-class case: only print the "all" summary row
        if len(ap_index) <= 1:
            return
        for i, cls_id in enumerate(ap_index):
            if nc is not None and nc.get(cls_id, 0) == 0:
                continue
            label = self.names.get(cls_id, str(cls_id))
            n_inst = nc.get(cls_id, 0) if nc else 0
            n_img = getattr(metrics, "n_images", 0)
            row_vals = list(metrics.class_result(i))  # (p, r, ap50, ap)
            self._write_detect_row(label, n_img, n_inst, row_vals)

    def _write_detect_row(self, label: str, n_img: int, n_inst: int, vals: list[float]) -> None:
        sys.stdout.write(
            f"{label:>{_LEFT_WIDTH}}" + _fmt_int(n_img) + _fmt_int(n_inst) + "".join(_fmt(v) for v in vals) + "\n"
        )

    def _print_segment_table(self, metrics: Any, nc: dict[int, int] | None) -> None:
        """Print 'all' row + per-class rows for segmentation metrics."""
        n_images_total = getattr(metrics, "n_images", 0)
        n_instances_total = int(sum(nc.values())) if nc else 0

        all_results = metrics.mean_results()  # [bp, br, bap50, bap, sp, sr, sap50, sap]
        self._write_segment_row("all", n_images_total, n_instances_total, all_results)

        ap_index = list(metrics.ap_class_index)
        # Single-class case: only print the "all" summary row
        if len(ap_index) <= 1:
            return
        for i, cls_id in enumerate(ap_index):
            if nc is not None and nc.get(cls_id, 0) == 0:
                continue
            label = self.names.get(cls_id, str(cls_id))
            n_inst = nc.get(cls_id, 0) if nc else 0
            n_img = getattr(metrics, "n_images", 0)
            row_vals = list(metrics.class_result(i))  # 8-tuple
            self._write_segment_row(label, n_img, n_inst, row_vals)

    def _write_segment_row(self, label: str, n_img: int, n_inst: int, vals: list[float]) -> None:
        sys.stdout.write(
            f"{label:>{_LEFT_WIDTH}}" + _fmt_int(n_img) + _fmt_int(n_inst) + "".join(_fmt(v) for v in vals) + "\n"
        )

    def _print_classify_table(self, metrics: Any) -> None:
        """Print summary row for classification metrics (no per-class rows)."""
        vals = [metrics.top1, metrics.top5]
        sys.stdout.write("".join(_fmt(v) for v in vals) + "\n")

    def _print_depth_table(self, metrics: Any) -> None:
        """Print summary row for depth metrics (no per-class rows)."""
        vals = [
            metrics.abs_rel,
            metrics.sq_rel,
            metrics.rmse,
            metrics.log_rmse,
            metrics.delta_1,
            metrics.delta_2,
            metrics.delta_3,
        ]
        sys.stdout.write("".join(_fmt(v) for v in vals) + "\n")

    def _print_ocr_table(self, metrics: Any) -> None:
        """Print summary row for OCR metrics (no per-class rows)."""
        vals = [metrics.cer, metrics.wer, metrics.accuracy]
        sys.stdout.write("".join(_fmt(v) for v in vals) + "\n")
