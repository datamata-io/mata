"""Detect node — object detection task node.

Runs an object detection provider on an input image and returns
a Detections artifact with bounding boxes, labels, and scores.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Detect(Node):
    """Object detection node.

    Looks up a ``Detector`` provider from the execution context,
    runs inference on the input image, and returns a
    :class:`~mata.core.artifacts.detections.Detections` artifact.

    Args:
        using: Name of the detection provider registered in the context
            (e.g. ``"detr"``, ``"yolo"``).
        out: Key under which the output artifact is stored (default ``"dets"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``predict()`` call (e.g. ``threshold``, ``nms_iou``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        detections (Detections): Detection results.

    Example:
        ```python
        from mata.nodes import Detect

        node = Detect(using="detr", out="dets", threshold=0.5)
        result = node.run(ctx, image=img)
        dets = result["dets"]
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        using: str,
        out: str = "dets",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.output_name = out
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute detection on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Detections artifact.

        Raises:
            KeyError: If the detection provider is not found in context.
        """
        detector = ctx.get_provider("detect", self.provider_name)

        start = time.time()
        result = detector.predict(image, **self.kwargs)
        latency_ms = (time.time() - start) * 1000

        # Convert to Detections artifact if needed
        if isinstance(result, Detections):
            detections = result
        else:
            # Assume VisionResult from adapter
            detections = Detections.from_vision_result(result)

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_detections", len(detections.instances))

        return {self.output_name: detections}

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = f"Detect(using='{self.provider_name}', out='{self.output_name}'"
        if extra:
            parts += f", {extra}"
        return parts + ")"
