"""VLMDetect node — detect objects using VLM with structured output.

Queries a Vision-Language Model provider in ``"detect"`` output mode to
produce structured object detections (entities and/or instances), returned
as a Detections artifact.
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


class VLMDetect(Node):
    """Detect objects using VLM with structured output.

    Looks up a ``VisionLanguageModel`` provider from the execution context,
    queries it with ``output_mode="detect"`` to produce structured object
    detections containing entities and optionally auto-promoted instances.

    Args:
        using: Name of the VLM provider registered in the context
            (e.g. ``"qwen3-vl"``, ``"gemini"``).
        prompt: Detection prompt sent to the VLM
            (default ``"List all objects you can identify."``).
        out: Key under which the output artifact is stored
            (default ``"vlm_dets"``).
        auto_promote: Whether to auto-promote entities to instances
            when spatial data is available (default ``True``).
        name: Optional human-readable node name.
        **vlm_kwargs: Extra keyword arguments forwarded to the provider's
            ``query()`` call (e.g. ``max_tokens``, ``temperature``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        detections (Detections): VLM detection results containing entities,
            optionally auto-promoted instances, and metadata.

    Example:
        ```python
        from mata.nodes import VLMDetect

        # Standard mode - single VLM call
        node = VLMDetect(
            using="qwen3-vl",
            prompt="Detect all animals and vehicles.",
            auto_promote=True,
        )
        result = node.run(ctx, image=img)
        dets = result["vlm_dets"]

        # Agent mode - VLM can use tools iteratively
        node = VLMDetect(
            using="qwen3-vl",
            prompt="Analyze this medical image.",
            tools=["detect", "classify", "zoom"],
            max_iterations=5,
        )
        result = node.run(ctx, image=img)
        dets = result["vlm_dets"]  # Contains merged results from all tool calls
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        using: str,
        prompt: str = "List all objects you can identify.",
        out: str = "vlm_dets",
        auto_promote: bool = True,
        tools: list[str] | None = None,
        max_iterations: int = 5,
        on_error: str = "retry",
        name: str | None = None,
        **vlm_kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.prompt = prompt
        self.output_name = out
        self.auto_promote = auto_promote
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_error = on_error
        self.vlm_kwargs = vlm_kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute VLM detection on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Detections artifact containing entities/instances.

        Raises:
            KeyError: If the VLM provider is not found in context.
        """
        vlm = ctx.get_provider("vlm", self.provider_name)

        if self.tools:
            # Agent mode — delegate to AgentLoop
            from mata.core.agent_loop import AgentLoop
            from mata.core.tool_registry import ToolRegistry

            start = time.time()
            registry = ToolRegistry(ctx, self.tools)
            loop = AgentLoop(vlm, registry, self.max_iterations, self.on_error, ctx=ctx, node_name=self.name)
            agent_result = loop.run(image, self.prompt, **self.vlm_kwargs)
            latency_ms = (time.time() - start) * 1000

            # Convert AgentResult → Detections artifact
            detections = self._agent_result_to_detections(agent_result)

            # Record metrics
            ctx.record_metric(self.name, "latency_ms", latency_ms)
            ctx.record_metric(self.name, "agent_iterations", float(agent_result.iterations))
            ctx.record_metric(self.name, "tool_calls_count", float(len(agent_result.tool_calls)))
        else:
            # Standard mode — single VLM call (unchanged)
            start = time.time()
            result = vlm.query(
                image,
                prompt=self.prompt,
                output_mode="detect",
                auto_promote=self.auto_promote,
                **self.vlm_kwargs,
            )
            latency_ms = (time.time() - start) * 1000

            # Convert VisionResult to Detections artifact
            if isinstance(result, Detections):
                detections = result
            else:
                from mata.core.artifacts.converters import vision_result_to_detections

                detections = vision_result_to_detections(result)

            # Record metrics
            ctx.record_metric(self.name, "latency_ms", latency_ms)

        ctx.record_metric(self.name, "num_entities", float(len(detections.entities)))
        ctx.record_metric(self.name, "num_instances", float(len(detections.instances)))

        return {self.output_name: detections}

    def _agent_result_to_detections(self, agent_result) -> Detections:
        """Convert AgentResult to Detections artifact.

        Merges instances and entities from all tool calls into a single
        Detections artifact, preserving the VLM's final text synthesis
        and tool call history in metadata.

        Args:
            agent_result: AgentResult from AgentLoop.run()

        Returns:
            Detections artifact with merged instances, entities, and metadata
        """
        from mata.core.types import VisionResult

        # Create VisionResult from agent data
        vision_result = VisionResult(
            instances=agent_result.instances,
            entities=agent_result.entities,
            text=agent_result.text,
            meta={
                "agent_iterations": agent_result.iterations,
                "agent_text": agent_result.text,  # Final VLM synthesis
                "agent_tool_calls": [tc.to_dict() for tc in agent_result.tool_calls],
                "agent_tool_results": [tr.to_dict() for tr in agent_result.tool_results],
                "conversation": agent_result.conversation,
                **agent_result.meta,
            },
        )

        # Convert to Detections artifact
        from mata.core.artifacts.converters import vision_result_to_detections

        return vision_result_to_detections(vision_result)
