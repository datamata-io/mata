"""Retail graph presets.

Provides graph factories for retail inventory management, shelf analysis,
product detection and classification, and stock level assessment workflows.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.classify import Classify
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.nms import NMS
from mata.nodes.roi import ExtractROIs
from mata.nodes.vlm_describe import VLMDescribe


def shelf_product_analysis(
    detection_threshold: float = 0.5,
    nms_iou_threshold: float | None = 0.5,
    classification_labels: list[str] | None = None,
    roi_padding: int = 5,
) -> Graph:
    """Shelf product detection + brand/category classification.

    Detects products on retail shelves, applies NMS to remove
    overlapping detections, crops each product, and classifies
    with zero-shot CLIP against brand/category labels.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — product detection (e.g. Faster R-CNN)
        - ``"classifier"`` — zero-shot classifier (e.g. CLIP)

    Args:
        detection_threshold: Minimum confidence for product detections
            (default ``0.5``).
        nms_iou_threshold: IoU threshold for NMS. Set to ``None`` to skip
            NMS (default ``0.5``).
        classification_labels: Brand/category labels for classification.
            Defaults to generic product categories:
            ``["beverage", "snack", "dairy", "produce", "canned_goods",
            "household", "other"]``.
        roi_padding: Padding around detection crops in pixels
            (default ``5``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import shelf_product_analysis
        >>>
        >>> detector = mata.load("detect", "torchvision/fasterrcnn_resnet50_fpn_v2")
        >>> classifier = mata.load("classify", "openai/clip-vit-base-patch32")
        >>> result = mata.infer(
        ...     "shelf.jpg",
        ...     shelf_product_analysis(
        ...         classification_labels=["coca_cola", "pepsi", "sprite", "other"],
        ...     ),
        ...     providers={"detector": detector, "classifier": classifier},
        ... )
        >>> for inst in result["final"].instances:
        ...     print(f"{inst.label_name}: {inst.score:.2f}")
    """
    if classification_labels is None:
        classification_labels = [
            "beverage",
            "snack",
            "dairy",
            "produce",
            "canned_goods",
            "household",
            "other",
        ]

    graph = Graph("shelf_product_analysis")

    # Step 1: Detect products
    graph = graph.then(Detect(using="detector", out="dets"))

    # Step 2: Filter by confidence
    graph = graph.then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))

    # Step 3: Optional NMS to remove overlapping detections
    if nms_iou_threshold is not None:
        graph = graph.then(NMS(iou_threshold=nms_iou_threshold, out="filtered"))

    # Step 4: Extract ROI crops for each detection
    graph = graph.then(ExtractROIs(src_dets="filtered", padding=roi_padding, out="rois"))

    # Step 5: Classify each product crop
    graph = graph.then(
        Classify(
            using="classifier",
            out="classes",
            text_prompts=classification_labels,
        )
    )

    # Step 6: Fuse all results
    graph = graph.then(
        Fuse(
            out="final",
            dets="filtered",
            rois="rois",
            classifications="classes",
        )
    )

    return graph


def stock_level_analysis(
    vlm_prompt: str = "Describe the stock levels on each shelf. Note any empty spaces or low stock areas.",
    detection_threshold: float = 0.4,
    classification_labels: list[str] | None = None,
) -> Graph:
    """Parallel VLM + detection + classification for stock assessment.

    Three-pronged analysis: VLM provides semantic stock assessment,
    detector counts products, CLIP classifies stock level.

    Provider keys expected in ``providers`` dict:
        - ``"vlm"`` — VLM adapter (e.g. Qwen3-VL)
        - ``"detector"`` — product detector (e.g. DETR)
        - ``"classifier"`` — zero-shot classifier (e.g. CLIP)

    Args:
        vlm_prompt: Prompt for VLM stock assessment (default describes
            stock level analysis task).
        detection_threshold: Minimum confidence for product detections
            (default ``0.4``).
        classification_labels: Labels for classifying overall stock level.
            Defaults to ``["fully_stocked", "partially_stocked",
            "low_stock", "empty_shelf"]``.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import stock_level_analysis
        >>>
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> classifier = mata.load("classify", "openai/clip-vit-base-patch32")
        >>> result = mata.infer(
        ...     "shelf.jpg",
        ...     stock_level_analysis(),
        ...     providers={
        ...         "vlm": vlm,
        ...         "detector": detector,
        ...         "classifier": classifier,
        ...     },
        ... )
        >>> print(result["final"].vlm_description)
        >>> print(f"Products detected: {len(result['final'].instances)}")
        >>> print(f"Stock level: {result['final'].classifications[0].label}")
    """
    if classification_labels is None:
        classification_labels = [
            "fully_stocked",
            "partially_stocked",
            "low_stock",
            "empty_shelf",
        ]

    return (
        Graph("stock_level_analysis")
        # Run VLM description, detection, and classification in parallel
        .parallel(
            [
                VLMDescribe(using="vlm", prompt=vlm_prompt, out="vlm_description"),
                Detect(using="detector", out="dets"),
                Classify(
                    using="classifier",
                    out="stock_classes",
                    text_prompts=classification_labels,
                ),
            ]
        )
        # Filter detections by confidence
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        # Fuse all results
        .then(
            Fuse(
                out="final",
                dets="filtered",
                vlm_description="vlm_description",
                classifications="stock_classes",
            )
        )
    )
