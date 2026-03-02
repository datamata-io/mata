"""TopK node for data transformation.

Keeps the top K detections by confidence score, returning a new
immutable Detections artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class TopK(Node):
    """Keep the top *k* detections ranked by confidence score.

    Both instances and entities are independently sorted by score and
    truncated to at most *k* entries each.  Instance IDs are preserved.

    Args:
        k: Maximum number of detections to keep.
        src: Name of the input artifact to read from context (default: "dets").
        out: Name of the output artifact key (default: "topk").
        name: Optional human-readable node name.

    Inputs:
        detections (Detections): Detection results to rank.

    Outputs:
        detections (Detections): Top-K detection results.

    Example:
        ```python
        from mata.nodes import TopK

        node = TopK(k=5, src="dets", out="top5")
        ```
    """

    inputs: dict[str, type[Artifact]] = {"detections": Detections}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        k: int,
        src: str = "dets",
        out: str = "topk",
        name: str | None = None,
    ):
        super().__init__(name=name)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.k = k
        self.src = src
        self.out = out

    def run(self, ctx: ExecutionContext, detections: Detections) -> dict[str, Artifact]:
        """Select top-K detections by score.

        Args:
            ctx: Execution context.
            detections: Input detections to rank and truncate.

        Returns:
            Dict with a single key (``self.out``) mapping to the truncated
            Detections artifact.
        """
        result = detections.top_k(self.k)

        ctx.record_metric(self.name, "k", self.k)
        ctx.record_metric(
            self.name,
            "input_count",
            len(detections.instances) + len(detections.entities),
        )
        ctx.record_metric(
            self.name,
            "output_count",
            len(result.instances) + len(result.entities),
        )

        return {self.out: result}

    def __repr__(self) -> str:
        return f"TopK(name='{self.name}', k={self.k})"
