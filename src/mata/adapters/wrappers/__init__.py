"""Adapter wrappers for the MATA graph system.

This package provides wrapper classes that convert existing MATA adapters
(detection, segmentation, classification, depth, SAM, VLM) into capability
providers that conform to the graph system's protocol interfaces.

Each wrapper:
- Converts Image artifacts to adapter-compatible format (PIL/path/numpy)
- Maps protocol method names to adapter.predict()
- Converts adapter results to graph artifact types where applicable
- Preserves adapter kwargs passthrough
- Provides clear error handling and messages

Factory functions (recommended usage):
    >>> from mata.adapters.wrappers import wrap_detector, wrap_segmenter
    >>> detector = wrap_detector(my_detect_adapter)
    >>> segmenter = wrap_segmenter(my_segment_adapter)

Direct class usage:
    >>> from mata.adapters.wrappers import DetectorWrapper
    >>> detector = DetectorWrapper(my_detect_adapter)
"""

from __future__ import annotations

from mata.adapters.wrappers.classify_wrapper import ClassifierWrapper, wrap_classifier
from mata.adapters.wrappers.depth_wrapper import DepthWrapper, wrap_depth
from mata.adapters.wrappers.detect_wrapper import DetectorWrapper, wrap_detector
from mata.adapters.wrappers.ocr_wrapper import OCRWrapper, wrap_ocr
from mata.adapters.wrappers.sam_wrapper import SAMWrapper, wrap_sam
from mata.adapters.wrappers.segment_wrapper import SegmenterWrapper, wrap_segmenter
from mata.adapters.wrappers.vlm_wrapper import VLMWrapper, wrap_vlm

__all__ = [
    # Wrapper classes
    "DetectorWrapper",
    "SegmenterWrapper",
    "ClassifierWrapper",
    "DepthWrapper",
    "OCRWrapper",
    "SAMWrapper",
    "VLMWrapper",
    # Factory functions
    "wrap_detector",
    "wrap_segmenter",
    "wrap_classifier",
    "wrap_depth",
    "wrap_ocr",
    "wrap_sam",
    "wrap_vlm",
]
