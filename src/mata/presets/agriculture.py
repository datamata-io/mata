"""Agriculture graph presets.

Provides graph factories for crop analysis, disease detection,
and pest monitoring workflows.
"""

from __future__ import annotations

from mata.core.graph.graph import Graph
from mata.nodes.depth import EstimateDepth
from mata.nodes.fuse import Fuse
from mata.nodes.segment import SegmentImage


def aerial_crop_analysis(
    segment_mode: str = "panoptic",
) -> Graph:
    """Parallel segmentation + depth for aerial crop analysis.

    Segments the aerial image into crop regions (panoptic segmentation)
    and estimates terrain depth, enabling crop coverage analysis and
    terrain-aware spraying/planning.

    Provider keys expected in ``providers`` dict:
        - ``"segmenter"`` — segmentation adapter (e.g. Mask2Former panoptic)
        - ``"depth"`` — depth estimation adapter (e.g. Depth Anything)

    Args:
        segment_mode: Segmentation mode, typically ``"panoptic"`` for
            crop field analysis (default ``"panoptic"``).

    Returns:
        A :class:`Graph` ready for ``mata.infer()``.

    Example:
        >>> import mata
        >>> from mata.presets import aerial_crop_analysis
        >>>
        >>> segmenter = mata.load("segment", "facebook/mask2former-swin-large-coco-panoptic")
        >>> depth = mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        >>> result = mata.infer(
        ...     "aerial_field.jpg",
        ...     aerial_crop_analysis(),
        ...     providers={"segmenter": segmenter, "depth": depth},
        ... )
    """
    return (
        Graph("aerial_crop_analysis")
        .parallel(
            [
                SegmentImage(using="segmenter", out="segments"),
                EstimateDepth(using="depth", out="depth"),
            ]
        )
        .then(Fuse(out="final", masks="segments", depth="depth"))
    )
