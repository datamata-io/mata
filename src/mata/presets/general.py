"""General-purpose graph presets.

Cross-industry patterns that apply to multiple domains.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS


def ensemble_detection(
    detection_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
) -> Graph:
    """Ensemble two detectors with NMS merging.

    Runs two independent detectors in parallel, concatenates their
    outputs, applies NMS to remove duplicate detections, and filters
    by confidence. Useful for boosting recall or comparing models.

    Provider keys expected in ``providers`` dict:
        - ``"detector_a"`` — first detection adapter
        - ``"detector_b"`` — second detection adapter

    Args:
        detection_threshold: Minimum confidence for final results
            (default ``0.3``).
        nms_iou_threshold: IoU threshold for NMS deduplication
            (default ``0.5``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import ensemble_detection
        >>>
        >>> det_a = mata.load("detect", "facebook/detr-resnet-50")
        >>> det_b = mata.load("detect", "torchvision/fasterrcnn_resnet50_fpn_v2")
        >>> result = mata.infer(
        ...     "image.jpg",
        ...     ensemble_detection(),
        ...     providers={"detector_a": det_a, "detector_b": det_b},
        ... )
    """
    return (
        Graph("ensemble_detection")
        .parallel(
            [
                Detect(using="detector_a", out="dets_a", name="detect_a"),
                Detect(using="detector_b", out="dets_b", name="detect_b"),
            ]
        )
        .then(NMS(iou_threshold=nms_iou_threshold, out="merged"))
        .then(Filter(src="merged", score_gt=detection_threshold, out="filtered"))
        .then(Fuse(out="final", dets="filtered"))
    )
