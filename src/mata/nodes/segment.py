"""Segment node — image segmentation task node.

Runs a segmentation provider on an input image and returns
a Masks artifact with per-instance masks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class SegmentImage(Node):
    """Image segmentation node.

    Looks up a ``Segmenter`` provider from the execution context,
    runs inference on the input image, and returns a
    :class:`~mata.core.artifacts.masks.Masks` artifact.

    Args:
        using: Name of the segmentation provider registered in the context
            (e.g. ``"mask2former"``, ``"sam"``).
        out: Key under which the output artifact is stored
            (default ``"masks"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``segment()`` call (e.g. ``threshold``, ``mode``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        masks (Masks): Segmentation masks.

    Example:
        ```python
        from mata.nodes import SegmentImage

        node = SegmentImage(using="mask2former", out="masks")
        result = node.run(ctx, image=img)
        masks = result["masks"]
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"masks": Masks}

    def __init__(
        self,
        using: str,
        out: str = "masks",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.output_name = out
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute segmentation on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Masks artifact.

        Raises:
            KeyError: If the segmentation provider is not found in context.
        """
        segmenter = ctx.get_provider("segment", self.provider_name)

        start = time.time()
        result = segmenter.segment(image, **self.kwargs)
        latency_ms = (time.time() - start) * 1000

        # Convert to Masks artifact if needed
        if isinstance(result, Masks):
            masks = result
        else:
            # Assume VisionResult from adapter
            masks = Masks.from_vision_result(result)

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_masks", len(masks.instances))

        return {self.output_name: masks}

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = f"SegmentImage(using='{self.provider_name}', out='{self.output_name}'"
        if extra:
            parts += f", {extra}"
        return parts + ")"
