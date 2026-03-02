"""Depth node — monocular depth estimation task node.

Runs a depth estimation provider on an input image and returns
a DepthMap artifact with per-pixel depth values.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.image import Image
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class EstimateDepth(Node):
    """Monocular depth estimation node.

    Looks up a ``DepthEstimator`` provider from the execution context,
    runs inference on the input image, and returns a
    :class:`~mata.core.artifacts.depth_map.DepthMap` artifact.

    Args:
        using: Name of the depth estimation provider registered in the
            context (e.g. ``"depth-anything"``, ``"midas"``).
        out: Key under which the output artifact is stored
            (default ``"depth"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``estimate()`` call (e.g. ``output_type``, ``colormap``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        depth (DepthMap): Depth estimation result.

    Example:
        ```python
        from mata.nodes import EstimateDepth

        node = EstimateDepth(using="depth-anything", out="depth")
        result = node.run(ctx, image=img)
        depth = result["depth"]
        print(depth.shape)
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"depth": DepthMap}

    def __init__(
        self,
        using: str,
        out: str = "depth",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.output_name = out
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute depth estimation on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a DepthMap artifact.

        Raises:
            KeyError: If the depth estimation provider is not found
                in context.
        """
        estimator = ctx.get_provider("depth", self.provider_name)

        start = time.time()
        result = estimator.estimate(image, **self.kwargs)
        latency_ms = (time.time() - start) * 1000

        # Convert to DepthMap artifact if needed
        if isinstance(result, DepthMap):
            depth_map = result
        else:
            # Assume DepthResult from adapter
            depth_map = DepthMap.from_depth_result(result)

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "height", depth_map.height)
        ctx.record_metric(self.name, "width", depth_map.width)

        return {self.output_name: depth_map}

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = f"EstimateDepth(using='{self.provider_name}', out='{self.output_name}'"
        if extra:
            parts += f", {extra}"
        return parts + ")"
