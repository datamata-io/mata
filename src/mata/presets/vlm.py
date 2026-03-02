"""VLM-based graph presets.

Provides Vision-Language Model presets that demonstrate Entity→Instance
promotion workflows, parallel VLM + spatial analysis, and multi-image
comparison pipelines.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.classify import Classify
from mata.nodes.depth import EstimateDepth
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.promote_entities import PromoteEntities
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_detect import VLMDetect
from mata.nodes.vlm_query import VLMQuery


def vlm_grounded_detection(
    vlm_prompt: str = "List all objects you can identify.",
    detection_threshold: float = 0.3,
    match_strategy: str = "label_fuzzy",
    auto_promote: bool = False,
) -> Graph:
    """VLM semantic detection + spatial grounding (Entity→Instance).

    Uses a VLM to semantically identify objects, then a spatial detector
    (GroundingDINO, DETR) to localize them.  The ``PromoteEntities`` node
    fuses VLM semantic labels with spatial bounding boxes by matching
    labels, producing spatially-grounded VisionResult instances.

    This demonstrates the **Entity→Instance workflow**: the VLM produces
    entities (labels without bboxes), the detector produces instances
    (bboxes without semantic context), and promotion merges both.

    Provider keys expected in ``providers`` dict:
        - ``"vlm"`` — VLM adapter (e.g. Qwen3-VL)
        - ``"detector"`` — spatial detection adapter (e.g. GroundingDINO)

    Args:
        vlm_prompt: Text prompt for the VLM to identify objects
            (default ``"List all objects you can identify."``).
        detection_threshold: Minimum confidence for spatial detections
            (default ``0.3``).
        match_strategy: Label matching strategy for entity→instance
            promotion — ``"label_exact"`` or ``"label_fuzzy"``
            (default ``"label_fuzzy"``).
        auto_promote: Whether the VLM adapter should auto-promote
            entities to instances where spatial data is available
            (default ``False``; promotion is handled explicitly by
            the ``PromoteEntities`` node).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import vlm_grounded_detection
        >>>
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B")
        >>> detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
        >>> result = mata.infer(
        ...     "image.jpg",
        ...     vlm_grounded_detection(
        ...         vlm_prompt="Find all animals and furniture.",
        ...     ),
        ...     providers={"vlm": vlm, "detector": detector},
        ... )
        >>> promoted = result["final"]
    """
    return (
        Graph("vlm_grounded_detection")
        # Run VLM detection and spatial detection in parallel
        .parallel(
            [
                VLMDetect(
                    using="vlm",
                    prompt=vlm_prompt,
                    auto_promote=auto_promote,
                    out="vlm_dets",
                ),
                Detect(using="detector", out="spatial_dets"),
            ]
        )
        # Filter spatial detections by confidence
        .then(
            Filter(
                src="spatial_dets",
                score_gt=detection_threshold,
                out="filtered_spatial",
            )
        )
        # Promote VLM entities to spatial instances
        .then(
            PromoteEntities(
                entities_src="vlm_dets",
                spatial_src="filtered_spatial",
                match_strategy=match_strategy,
                out="promoted",
            )
        )
        # Bundle
        .then(
            Fuse(
                out="final",
                promoted="promoted",
                vlm_dets="vlm_dets",
                spatial_dets="filtered_spatial",
            )
        )
    )


def vlm_scene_understanding(
    describe_prompt: str = "Describe this image in detail.",
    detection_threshold: float = 0.3,
    classification_labels: list[str] | None = None,
) -> Graph:
    """Parallel VLM description + detection + depth estimation.

    Runs three independent analysis tasks in parallel for comprehensive
    scene understanding:

    - **VLM description**: natural language image captioning
    - **Detection**: object detection with configurable threshold
    - **Depth estimation**: monocular depth maps

    Provider keys expected in ``providers`` dict:
        - ``"vlm"`` — VLM adapter (e.g. Qwen3-VL)
        - ``"detector"`` — detection adapter (e.g. DETR, RT-DETR)
        - ``"depth"`` — depth estimation adapter (e.g. Depth-Anything)

    Args:
        describe_prompt: Text prompt for VLM image description
            (default ``"Describe this image in detail."``).
        detection_threshold: Minimum confidence for detections
            (default ``0.3``).
        classification_labels: If provided, also run zero-shot classification
            as a fourth parallel task with the ``"classifier"`` provider.

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import vlm_scene_understanding
        >>>
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B")
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>> depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        >>> result = mata.infer(
        ...     "scene.jpg",
        ...     vlm_scene_understanding(),
        ...     providers={"vlm": vlm, "detector": detector, "depth": depth},
        ... )
        >>> description = result["scene"]  # contains all channels
    """
    # Core parallel tasks: VLM + Detection + Depth
    parallel_nodes = [
        VLMDescribe(using="vlm", prompt=describe_prompt, out="description"),
        Detect(using="detector", out="dets", threshold=detection_threshold),
        EstimateDepth(using="depth", out="depth"),
    ]

    # Optionally add classification
    classify_kwargs: dict = {}
    if classification_labels:
        classify_kwargs["text_prompts"] = classification_labels
        parallel_nodes.append(Classify(using="classifier", out="class", **classify_kwargs))

    # Build fuse kwargs — always include the three core channels
    fuse_kwargs: dict = {
        "out": "scene",
        "description": "description",
        "dets": "dets",
        "depth": "depth",
    }
    if classification_labels:
        fuse_kwargs["classifications"] = "class"

    return Graph("vlm_scene_understanding").parallel(parallel_nodes).then(Fuse(**fuse_kwargs))


def vlm_multi_image_comparison(
    prompt: str = "Compare these images and describe the differences.",
    output_mode: str | None = None,
) -> Graph:
    """Multi-image VLM comparison.

    Queries a VLM with two or more images for side-by-side analysis.
    The primary image is passed via ``mata.infer()``'s ``image`` argument;
    additional images are provided through the execution context.

    Provider keys expected in ``providers`` dict:
        - ``"vlm"`` — VLM adapter with multi-image support (e.g. Qwen3-VL)

    Args:
        prompt: Comparison prompt sent to the VLM
            (default ``"Compare these images and describe the differences."``).
        output_mode: Optional output mode for structured responses
            (``None``, ``"json"``, ``"detect"``, ``"classify"``,
            ``"describe"``).  Defaults to free-form text (``None``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import vlm_multi_image_comparison
        >>>
        >>> vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B")
        >>> result = mata.infer(
        ...     "image1.jpg",
        ...     vlm_multi_image_comparison(
        ...         prompt="What changed between these two images?",
        ...     ),
        ...     providers={"vlm": vlm},
        ... )
    """
    return (
        Graph("vlm_multi_image_comparison")
        .then(
            VLMQuery(
                using="vlm",
                prompt=prompt,
                output_mode=output_mode,
                out="comparison",
            )
        )
        .then(Fuse(out="final", comparison="comparison"))
    )
