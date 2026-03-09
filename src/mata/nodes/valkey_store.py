"""Valkey/Redis sink node for graph pipelines.

Writes artifacts to Valkey during graph execution, enabling
distributed result sharing and pipeline persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class ValkeyStore(Node):
    """Sink node that writes artifacts to Valkey/Redis.

    Writes the specified source artifact to a Valkey key during graph
    execution. Supports key templates with variable substitution and
    TTL-based expiration.

    This node is a terminal/sink — it stores data externally and
    passes the input artifact through unchanged as its output.

    Args:
        src: Name of the input artifact to store
        url: Valkey connection URL (e.g., "valkey://localhost:6379")
        key: Key name or template (supports {node}, {timestamp} placeholders)
        ttl: Time-to-live in seconds (None = no expiry)
        serializer: "json" (default) or "msgpack"
        out: Output artifact name (passes input through, default: same as src)

    Examples:
        >>> from mata.nodes import Detect, Filter, ValkeyStore
        >>> graph = (Graph()
        ...     .then(Detect(using="detr", out="dets"))
        ...     .then(Filter(src="dets", score_gt=0.5, out="filtered"))
        ...     .then(ValkeyStore(
        ...         src="filtered",
        ...         url="valkey://localhost:6379",
        ...         key="pipeline:detections:{timestamp}",
        ...         ttl=3600,
        ...     ))
        ... )
    """

    # inputs/outputs are set dynamically in __init__ based on src so that
    # the graph auto-wiring can match the actual artifact name in context.
    inputs: dict[str, type[Artifact]]
    outputs: dict[str, type[Artifact]]

    def __init__(
        self,
        src: str,
        url: str,
        key: str,
        ttl: int | None = None,
        serializer: str = "json",
        out: str | None = None,
    ):
        super().__init__(name="ValkeyStore")
        self.src_name = src
        self.url = url
        self.key_template = key
        self.ttl = ttl
        self.serializer = serializer
        self.output_name = out or src
        # Use src as the input key so graph auto-wiring matches by artifact name
        self.inputs = {src: Artifact}
        self.outputs = {self.output_name: Artifact}

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Store artifact to Valkey and pass through.

        Args:
            ctx: Execution context.
            **inputs: The single input artifact, keyed by src name.

        Returns:
            Dict mapping output name to the unchanged input artifact.
        """
        import time

        from mata.core.exporters.valkey_exporter import export_valkey

        # Accept the artifact under whatever key the scheduler passes it as
        artifact = next(iter(inputs.values()))

        resolved_key = self.key_template.format(
            node=self.name,
            timestamp=int(time.time()),
        )

        result = self._artifact_to_serializable(artifact)

        export_valkey(
            result=result,
            url=self.url,
            key=resolved_key,
            ttl=self.ttl,
            serializer=self.serializer,
        )

        return {self.output_name: artifact}

    @staticmethod
    def _artifact_to_serializable(artifact: Artifact):
        """Convert graph artifact to a serializable result type."""
        from mata.core.artifacts.classifications import Classifications
        from mata.core.artifacts.converters import (
            artifact_to_classify_result,
            artifact_to_depth_result,
            detections_to_vision_result,
            masks_to_vision_result,
        )
        from mata.core.artifacts.depth_map import DepthMap
        from mata.core.artifacts.detections import Detections
        from mata.core.artifacts.masks import Masks

        if isinstance(artifact, Detections):
            return detections_to_vision_result(artifact)
        elif isinstance(artifact, Masks):
            return masks_to_vision_result(artifact)
        elif isinstance(artifact, Classifications):
            return artifact_to_classify_result(artifact)
        elif isinstance(artifact, DepthMap):
            return artifact_to_depth_result(artifact)
        else:
            # Fallback: use artifact's own to_dict()
            return artifact
