"""Detection + segmentation graph presets.

Provides ready-to-use graphs for detection followed by prompt-based
segmentation (e.g. GroundingDINO + SAM) and segmentation with mask
refinement workflows.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS
from mata.nodes.prompt_boxes import PromptBoxes
from mata.nodes.refine_mask import RefineMask
from mata.nodes.segment_everything import SegmentEverything


def grounding_dino_sam(
    detection_threshold: float = 0.3,
    nms_iou_threshold: float | None = None,
    refine_method: str = "morph_close",
    refine_radius: int = 3,
) -> Graph:
    """Pre-built GroundingDINO + SAM pipeline.

    Detects objects with a detection provider, filters by confidence,
    optionally applies NMS, segments each detection with a SAM-style
    segmenter, applies morphological mask refinement, and bundles results.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (e.g. GroundingDINO, DETR)
        - ``"segmenter"`` — SAM-style segmentation adapter

    Args:
        detection_threshold: Minimum confidence score for detections
            (default ``0.3``).
        nms_iou_threshold: If set, apply NMS with this IoU threshold
            after filtering. Useful for removing duplicate detections.
        refine_method: Morphological operation for mask refinement.
            One of ``"morph_close"``, ``"morph_open"``, ``"dilate"``,
            ``"erode"`` (default ``"morph_close"``).
        refine_radius: Kernel radius for mask refinement (default ``3``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import grounding_dino_sam
        >>>
        >>> detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        >>> segmenter = mata.load("segment", "facebook/sam-vit-base")
        >>> result = mata.infer(
        ...     "image.jpg",
        ...     grounding_dino_sam(detection_threshold=0.5),
        ...     providers={"detector": detector, "segmenter": segmenter},
        ... )
        >>> print(len(result["final"].instances))
    """
    graph = Graph("grounding_dino_sam")

    # Step 1: Detect objects
    graph = graph.then(Detect(using="detector", out="dets"))

    # Step 2: Filter by confidence
    graph = graph.then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))

    # Step 3: Optional NMS
    if nms_iou_threshold is not None:
        graph = graph.then(NMS(iou_threshold=nms_iou_threshold, out="filtered"))

    # Step 4: Prompt-based segmentation with detection boxes
    graph = graph.then(PromptBoxes(using="segmenter", dets_src="filtered", out="masks"))

    # Step 5: Refine masks
    graph = graph.then(
        RefineMask(
            src="masks",
            method=refine_method,
            radius=refine_radius,
            out="masks_ref",
        )
    )

    # Step 6: Bundle results
    graph = graph.then(Fuse(out="final", dets="filtered", masks="masks_ref"))

    return graph


def segment_and_refine(
    segmentation_method: str = "everything",
    refine_method: str = "morph_close",
    refine_radius: int = 3,
) -> Graph:
    """Segment entire image and refine masks.

    Runs a segmentation provider in automatic/everything mode to generate
    all masks, then applies morphological refinement.

    Provider keys expected in ``providers`` dict:
        - ``"segmenter"`` — segmentation adapter (SAM or Mask2Former)

    Args:
        segmentation_method: Segmentation mode — ``"everything"`` for SAM
            automatic mode (default ``"everything"``).
        refine_method: Morphological operation for mask refinement
            (default ``"morph_close"``).
        refine_radius: Kernel radius for refinement (default ``3``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import segment_and_refine
        >>>
        >>> segmenter = mata.load("segment", "facebook/sam-vit-base")
        >>> result = mata.infer(
        ...     "image.jpg",
        ...     segment_and_refine(),
        ...     providers={"segmenter": segmenter},
        ... )
    """
    return (
        Graph("segment_and_refine")
        .then(SegmentEverything(using="segmenter", out="raw_masks"))
        .then(
            RefineMask(
                src="raw_masks",
                method=refine_method,
                radius=refine_radius,
                out="masks",
            )
        )
        .then(Fuse(out="final", masks="masks"))
    )
