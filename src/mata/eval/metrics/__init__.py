"""MATA evaluation metrics sub-package."""

from __future__ import annotations

from mata.eval.metrics.base import Metric, ap_per_class
from mata.eval.metrics.classify import ClassifyMetrics
from mata.eval.metrics.depth import DepthMetrics
from mata.eval.metrics.detect import DetMetrics
from mata.eval.metrics.iou import COCO_IOU_THRESHOLDS, box_iou, box_iou_batch, mask_iou
from mata.eval.metrics.ocr import OCRMetrics
from mata.eval.metrics.segment import SegmentMetrics

__all__ = [
    "Metric",
    "ap_per_class",
    "box_iou",
    "box_iou_batch",
    "mask_iou",
    "COCO_IOU_THRESHOLDS",
    "DetMetrics",
    "SegmentMetrics",
    "ClassifyMetrics",
    "DepthMetrics",
    "OCRMetrics",
]
