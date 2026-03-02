"""Manufacturing graph presets.

Provides graph factories for quality inspection, defect detection,
assembly verification, and component inspection workflows.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.classify import Classify
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.roi import ExtractROIs
from mata.nodes.vlm_query import VLMQuery


def defect_detect_classify(
    detection_threshold: float = 0.3,
    defect_prompts: str = "scratch . crack . dent . stain . damage",
    classification_labels: list[str] | None = None,
) -> Graph:
    """Zero-shot defect detection + ROI classification.

    Detects defects using text-prompted detection (GroundingDINO),
    crops each detection as an ROI, then classifies each crop using
    zero-shot classification (CLIP) against defect type labels.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — text-prompted detector (e.g. GroundingDINO)
        - ``"classifier"`` — zero-shot classifier (e.g. CLIP)

    Args:
        detection_threshold: Minimum confidence for defect detections
            (default ``0.3``).
        defect_prompts: Dot-separated text prompts for the detector
            (default ``"scratch . crack . dent . stain . damage"``).
        classification_labels: Labels for zero-shot classification.
            Defaults to ``["scratch", "crack", "dent", "corrosion",
            "deformation", "stain", "normal"]``.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import defect_detect_classify
        >>>
        >>> detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        >>> classifier = mata.load("classify", "openai/clip-vit-base-patch32")
        >>> result = mata.infer(
        ...     "metal_surface.jpg",
        ...     defect_detect_classify(
        ...         defect_prompts="scratch . crack . corrosion",
        ...         classification_labels=["scratch", "crack", "corrosion", "normal"],
        ...     ),
        ...     providers={"detector": detector, "classifier": classifier},
        ... )
    """
    if classification_labels is None:
        classification_labels = [
            "scratch",
            "crack",
            "dent",
            "corrosion",
            "deformation",
            "stain",
            "normal",
        ]

    return (
        Graph("defect_detect_classify")
        .then(Detect(using="detector", out="dets", text_prompts=defect_prompts))
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        .then(ExtractROIs(src_dets="filtered", padding=10, out="rois"))
        .then(Classify(using="classifier", out="classes", text_prompts=classification_labels))
        .then(Fuse(out="final", dets="filtered", rois="rois", classifications="classes"))
    )


def assembly_verification(
    vlm_prompt: str = "Inspect this assembly. Are all components present and correctly positioned? List any issues.",
    detection_threshold: float = 0.4,
) -> Graph:
    """Parallel VLM inspection + detection for assembly verification.

    Runs a VLM to holistically inspect the assembly for correctness,
    while simultaneously detecting individual components. Results are
    fused for combined semantic + spatial analysis.

    Provider keys expected in ``providers`` dict:
        - ``"vlm"`` — VLM adapter (e.g. Qwen3-VL)
        - ``"detector"`` — detection adapter (e.g. DETR)

    Args:
        vlm_prompt: Prompt for VLM inspection (default describes
            assembly verification task).
        detection_threshold: Minimum confidence for component
            detections (default ``0.4``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import assembly_verification
        >>>
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> result = mata.infer(
        ...     "assembly.jpg",
        ...     assembly_verification(),
        ...     providers={"vlm": vlm, "detector": detector},
        ... )
    """
    return (
        Graph("assembly_verification")
        .parallel(
            [
                VLMQuery(using="vlm", prompt=vlm_prompt, out="vlm_assessment"),
                Detect(using="detector", out="dets"),
            ]
        )
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
        .then(Fuse(out="final", dets="filtered", vlm_assessment="vlm_assessment"))
    )


def component_inspection(
    detection_threshold: float = 0.5,
    vlm_prompt: str = "Inspect this component closely. Describe any visible defects, wear, or anomalies.",
    top_k: int | None = None,
) -> Graph:
    """Detect → crop → VLM per-component inspection.

    Detects all components in the image, extracts ROI crops for each,
    then queries a VLM for detailed per-component analysis.

    Provider keys expected in ``providers`` dict:
        - ``"detector"`` — detection adapter (e.g. DETR, GroundingDINO)
        - ``"vlm"`` — VLM adapter (e.g. Qwen3-VL)

    Args:
        detection_threshold: Minimum confidence for component
            detections (default ``0.5``).
        vlm_prompt: Prompt for per-component VLM inspection.
        top_k: If set, only inspect the top-K most confident
            detections (default ``None`` — inspect all).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import component_inspection
        >>>
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        >>> result = mata.infer(
        ...     "components.jpg",
        ...     component_inspection(top_k=5),
        ...     providers={"detector": detector, "vlm": vlm},
        ... )
    """
    graph = (
        Graph("component_inspection")
        .then(Detect(using="detector", out="dets"))
        .then(Filter(src="dets", score_gt=detection_threshold, out="filtered"))
    )

    if top_k is not None:
        from mata.nodes.topk import TopK

        graph = graph.then(TopK(k=top_k, src="filtered", out="filtered"))

    graph = (
        graph.then(ExtractROIs(src_dets="filtered", padding=10, out="rois"))
        .then(VLMQuery(using="vlm", prompt=vlm_prompt, out="inspection"))
        .then(Fuse(out="final", dets="filtered", rois="rois", inspection="inspection"))
    )

    return graph
