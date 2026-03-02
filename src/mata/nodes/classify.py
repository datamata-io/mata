"""Classify node — image classification task node.

Runs a classification provider on an input image and returns
a Classifications artifact with sorted predictions.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.image import Image
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Classify(Node):
    """Image classification node.

    Looks up a ``Classifier`` provider from the execution context,
    runs inference on the input image, and returns a
    :class:`~mata.core.artifacts.classifications.Classifications` artifact.

    Args:
        using: Name of the classification provider registered in the context
            (e.g. ``"resnet"``, ``"clip"``).
        out: Key under which the output artifact is stored
            (default ``"classifications"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``classify()`` call (e.g. ``top_k``, ``text_prompts``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        classifications (Classifications): Classification results.

    Example:
        ```python
        from mata.nodes import Classify

        node = Classify(using="resnet", out="cls", top_k=5)
        result = node.run(ctx, image=img)
        cls = result["cls"]
        print(cls.top1)
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"classifications": Classifications}

    def __init__(
        self,
        using: str,
        out: str = "classifications",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.output_name = out
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute classification on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Classifications artifact.

        Raises:
            KeyError: If the classification provider is not found in context.
        """
        classifier = ctx.get_provider("classify", self.provider_name)

        start = time.time()
        result = classifier.classify(image, **self.kwargs)
        latency_ms = (time.time() - start) * 1000

        # Convert to Classifications artifact if needed
        if isinstance(result, Classifications):
            classifications = result
        else:
            # Assume ClassifyResult from adapter
            classifications = Classifications.from_classify_result(result)

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_predictions", len(classifications))

        return {self.output_name: classifications}

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = f"Classify(using='{self.provider_name}', out='{self.output_name}'"
        if extra:
            parts += f", {extra}"
        return parts + ")"
