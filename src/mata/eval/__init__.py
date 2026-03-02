"""MATA evaluation and validation metrics.

Public API
----------
.. code-block:: python

    from mata.eval import Validator, DetMetrics, SegmentMetrics, ClassifyMetrics, DepthMetrics
"""

from __future__ import annotations

from mata.eval.dataset import DatasetLoader, GroundTruth
from mata.eval.metrics.classify import ClassifyMetrics
from mata.eval.metrics.depth import DepthMetrics
from mata.eval.metrics.detect import DetMetrics
from mata.eval.metrics.ocr import OCRMetrics
from mata.eval.metrics.segment import SegmentMetrics
from mata.eval.validator import Validator

__all__ = [
    "Validator",
    "DatasetLoader",
    "GroundTruth",
    "DetMetrics",
    "SegmentMetrics",
    "ClassifyMetrics",
    "DepthMetrics",
    "OCRMetrics",
]
