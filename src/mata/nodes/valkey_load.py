"""Valkey/Redis source node for graph pipelines.

Loads previously stored artifacts from Valkey and injects them
into the graph execution context as typed artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class ValkeyLoad(Node):
    """Source node that loads artifacts from Valkey/Redis.

    Loads a previously stored result from a Valkey key and converts
    it into the appropriate graph artifact type. Typically used as
    an entry node in a graph that consumes results from another pipeline.

    Args:
        url: Valkey connection URL
        key: Key name to load from
        result_type: "auto", "vision", "classify", "depth", "ocr"
        out: Output artifact name (default: "loaded")

    Examples:
        >>> from mata.nodes import ValkeyLoad, Filter, Fuse
        >>> graph = (Graph()
        ...     .then(ValkeyLoad(
        ...         url="valkey://localhost:6379",
        ...         key="upstream:detections:latest",
        ...         result_type="vision",
        ...         out="dets",
        ...     ))
        ...     .then(Filter(src="dets", score_gt=0.7, out="filtered"))
        ...     .then(Fuse(detections="filtered"))
        ... )
    """

    inputs: dict[str, type[Artifact]] = {}  # Source node: no inputs
    outputs: dict[str, type[Artifact]] = {"artifact": Artifact}

    def __init__(
        self,
        url: str,
        key: str,
        result_type: str = "auto",
        out: str = "loaded",
    ):
        super().__init__(name="ValkeyLoad")
        self.url = url
        self.key = key
        self.result_type = result_type
        self.output_name = out

    def run(self, ctx: ExecutionContext) -> dict[str, Artifact]:
        """Load result from Valkey and convert to artifact.

        Args:
            ctx: Execution context.

        Returns:
            Dict mapping output name to the loaded artifact.

        Raises:
            KeyError: If the Valkey key does not exist.
        """
        from mata.core.exporters.valkey_exporter import load_valkey

        result = load_valkey(
            url=self.url,
            key=self.key,
            result_type=self.result_type,
        )

        artifact = self._result_to_artifact(result)
        return {self.output_name: artifact}

    @staticmethod
    def _result_to_artifact(result) -> Artifact:
        """Convert a result type back to a graph artifact."""
        from mata.core.artifacts.classifications import Classifications
        from mata.core.artifacts.depth_map import DepthMap
        from mata.core.artifacts.detections import Detections
        from mata.core.types import ClassifyResult, DepthResult, VisionResult

        if isinstance(result, VisionResult):
            return Detections.from_vision_result(result)
        elif isinstance(result, ClassifyResult):
            return Classifications.from_classify_result(result)
        elif isinstance(result, DepthResult):
            return DepthMap.from_depth_result(result)
        else:
            raise TypeError(f"Cannot convert {type(result).__name__} to graph artifact")
