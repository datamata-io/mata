"""Fuse node — bundle all artifacts into MultiResult.

Creates a unified MultiResult container with channel-based artifact storage,
provenance tracking, and metrics aggregation from graph execution.
"""

from __future__ import annotations

import importlib.metadata
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.result import MultiResult
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Fuse(Node):
    """Bundle all artifacts into MultiResult.

    The Fuse node collects artifacts from graph execution and packages them
    into a unified MultiResult container with:
    - Channel-based artifact storage (detections, masks, keypoints, etc.)
    - Provenance tracking (model hashes, graph config, timestamps)
    - Metrics aggregation (per-node timing, memory usage)
    - Instance cross-referencing capabilities

    This is typically used as the final node in a graph to bundle all
    intermediate results into a complete, traceable result.

    Args:
        out: Output artifact name in execution context (default: "final")
        **channel_sources: Mapping of channel names to artifact names in context.
                          Example: detections="dets", masks="masks_ref"

    Examples:
        >>> # Bundle detection and segmentation results
        >>> fuse_node = Fuse(
        ...     out="complete_result",
        ...     detections="filtered_dets",
        ...     masks="sam_masks",
        ...     image="input_image"
        ... )

        >>> # Simple bundling with default output name
        >>> fuse_node = Fuse(detections="dets", masks="masks")

        >>> # Use in graph
        >>> from mata.core.graph.graph import Graph
        >>> graph = (Graph("detect_and_segment")
        ...     .then(Load(path="image.jpg"))
        ...     .then(Detect(using="detr", out="dets"))
        ...     .then(PromptBoxes(using="sam", dets="dets", out="masks"))
        ...     .then(Fuse(detections="dets", masks="masks", image="image")))
    """

    # Inputs are dynamically declared in __init__ based on channel_sources
    outputs = {"result": MultiResult}

    def __init__(self, out: str = "final", **channel_sources: str):
        """Initialize Fuse node.

        Args:
            out: Name for the output MultiResult in execution context
            **channel_sources: Mapping of channel names to artifact names.
                              Keys become channel names in MultiResult,
                              values are artifact names to fetch from context.

        Examples:
            >>> Fuse(detections="dets", masks="refined_masks")            >>> Fuse(out="bundle", image="img", results="final_dets")
        """
        super().__init__(name="Fuse")
        self.output_name = out
        self.channel_sources = channel_sources

        # Dynamically declare inputs based on channel_sources
        # This allows the wiring system to understand dependencies
        # We map source artifact names to Artifact base type (accept any)
        self.inputs = {artifact_name: Artifact for artifact_name in channel_sources.values()}

    def run(self, ctx: ExecutionContext, **artifacts: Artifact) -> dict[str, Artifact]:
        """Bundle artifacts into MultiResult with provenance and metrics.

        Args:
            ctx: Graph execution context for provenance and metrics
            **artifacts: Named artifacts to bundle (matched against channel_sources)

        Returns:
            Dictionary mapping output_name to MultiResult containing:
            - channels: Artifacts organized by channel name
            - provenance: Metadata about models, graph, execution environment
            - metrics: Per-node timing and resource usage metrics

        Example:
            >>> # Assuming context has artifacts: "dets", "masks", "image"
            >>> result = fuse_node.run(ctx,
            ...                       dets=detections_artifact,
            ...                       masks=masks_artifact,
            ...                       image=image_artifact)
            >>> multi_result = result["final"]
            >>> multi_result.detections  # Access via dynamic attribute
            >>> multi_result.channels["masks"]  # Access via channel dict
        """
        start_time = time.time()

        # Build channels from available artifacts
        # Fuse declares inputs dynamically, so artifacts are passed as kwargs
        # We also support fetching from context as a fallback
        channels = {}
        for channel_name, artifact_name in self.channel_sources.items():
            artifact = None

            # Try to get from kwargs first (preferred, since we now declare inputs)
            if artifact_name in artifacts:
                artifact = artifacts[artifact_name]
            else:
                # Fallback: try to get from context
                try:
                    artifact = ctx.retrieve(artifact_name)
                except KeyError:
                    # Artifact not available — continue without error
                    # This allows for optional artifacts in fusion
                    continue

            if artifact is not None:
                channels[channel_name] = artifact

        # Collect provenance metadata
        provenance = self._build_provenance(ctx)

        # Collect execution metrics
        metrics = ctx.get_metrics()

        # Create MultiResult bundle
        result = MultiResult(
            channels=channels,
            provenance=provenance,
            metrics=metrics,
            meta={"fused_at": datetime.now(timezone.utc).isoformat()},
        )

        # Record fusion timing
        fusion_time = (time.time() - start_time) * 1000  # Convert to ms
        ctx.record_metric(self.name, "fusion_latency_ms", fusion_time)
        ctx.record_metric(self.name, "num_channels", len(channels))

        return {self.output_name: result}

    def _build_provenance(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Build provenance metadata from execution context.

        Args:
            ctx: Execution context with provider and metric information

        Returns:
            Provenance dictionary with model hashes, graph config, timestamps, etc.
        """
        provenance = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_type": "Fuse",
            "node_name": self.name,
            "framework_version": f"mata-{importlib.metadata.version('datamata')}",
        }

        # Add model information if available from context
        # Note: This would need enhancement once provider registry supports
        # model hash/version tracking
        try:
            # Get provider details (placeholder for future enhancement)
            provenance["providers"] = {
                "count": len(getattr(ctx, "providers", {})),
                "capabilities": list(getattr(ctx, "providers", {}).keys()),
            }
        except Exception:
            # Gracefully handle missing provider info
            provenance["providers"] = {"count": 0, "capabilities": []}

        # Add device information
        provenance["device"] = getattr(ctx, "device", "unknown")

        # Add channel mapping for traceability
        provenance["channel_mapping"] = self.channel_sources.copy()

        return provenance

    def __repr__(self) -> str:
        channel_list = ", ".join(f"{k}={v}" for k, v in self.channel_sources.items())
        return f"Fuse(out='{self.output_name}', {channel_list})"
