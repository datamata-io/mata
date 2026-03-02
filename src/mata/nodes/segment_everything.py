"""SegmentEverything node for SAM automatic mask generation.

Generates all possible masks in an image using SAM's automatic segmentation
mode (no prompts required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class SegmentEverything(Node):
    """Generate all masks in an image (SAM automatic mode).

    Runs a segmentation provider in ``"everything"`` mode to produce masks
    for every detectable region in the image. No prompts are needed — the
    model automatically identifies all segments.

    Typical use cases:
    - Unsupervised region proposals
    - Panoptic-style full-image segmentation
    - Mask generation for downstream filtering and analysis

    Args:
        using: Name of the Segmenter provider in the context
            (e.g. ``"sam"``, ``"sam3"``).
        image_src: Artifact name for the input image (default: ``"image"``).
        out: Key used for the output Masks artifact (default: ``"masks"``).
        name: Optional human-readable node name.
        **kwargs: Additional keyword arguments forwarded to the segmenter's
            ``segment()`` call (e.g. ``points_per_side``,
            ``pred_iou_thresh``, ``stability_score_thresh``).

    Inputs:
        image (Image): Source image to segment.

    Outputs:
        masks (Masks): All segmentation masks found in the image.

    Example:
        ```python
        from mata.nodes.segment_everything import SegmentEverything

        node = SegmentEverything(using="sam", out="all_masks")
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"masks": Masks}

    def __init__(
        self,
        using: str,
        image_src: str = "image",
        out: str = "masks",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.image_src = image_src
        self.output_name = out
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Generate all masks in the image.

        Steps:
        1. Retrieve the Segmenter provider from context.
        2. Call the segmenter with ``mode="everything"``.
        3. Annotate output with metadata.
        4. Record metrics.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict mapping ``self.output_name`` to the resulting Masks artifact.

        Raises:
            KeyError: If the segmentation provider is not found in context.
        """
        segmenter = ctx.get_provider("segment", self.provider_name)

        # Segment everything — no prompts
        masks = segmenter.segment(
            image,
            mode="everything",
            **self.kwargs,
        )

        # Add prompt metadata
        masks = Masks(
            instances=masks.instances,
            instance_ids=masks.instance_ids,
            meta={**masks.meta, "prompt_type": "everything"},
        )

        # Record metrics
        ctx.record_metric(self.name, "num_masks", float(len(masks.instances)))

        return {self.output_name: masks}

    def __repr__(self) -> str:
        return f"SegmentEverything(name='{self.name}', " f"using='{self.provider_name}', " f"out='{self.output_name}')"
