"""VLMQuery node — generic VLM query with multi-image support.

Provides a flexible VLM query node supporting single and multi-image input,
configurable output modes, and arbitrary prompt text.
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


class VLMQuery(Node):
    """Generic VLM query with multi-image support.

    Looks up a ``VisionLanguageModel`` provider and queries it with one or
    more images plus a text prompt.  Supports all VLM output modes
    (``None``, ``"json"``, ``"detect"``, ``"classify"``, ``"describe"``).

    When additional images are provided via the ``images`` input, they are
    combined with the primary ``image`` and passed as a list to the VLM.

    Args:
        using: Name of the VLM provider registered in the context.
        prompt: Text prompt sent to the VLM.
        output_mode: Optional output mode for structured responses
            (``None``, ``"json"``, ``"detect"``, ``"classify"``, ``"describe"``).
        out: Key under which the output artifact is stored
            (default ``"vlm_result"``).
        tools: Optional list of tool names to enable agent mode.
            When provided, the VLM can call these tools iteratively
            to gather information. Note: multi-image support is not
            available in agent mode (only uses primary image).
            (default ``None`` = standard single-call mode).
        max_iterations: Maximum number of agent loop iterations
            when tools are enabled (default ``5``).
        on_error: Error handling mode for tool execution failures:
            ``"retry"`` = retry malformed calls, ``"skip"`` = continue,
            ``"fail"`` = raise immediately (default ``"retry"``).
        name: Optional human-readable node name.
        **vlm_kwargs: Extra keyword arguments forwarded to the provider's
            ``query()`` call.

    Inputs:
        image (Image): Primary input image artifact.
        images (Optional[List[Image]]): Additional images for multi-image
            queries.  If omitted, only the primary image is used.

    Outputs:
        result (Detections): VLM query result wrapping a VisionResult.

    Example:
        ```python
        from mata.nodes import VLMQuery

        # Single-image query
        node = VLMQuery(
            using="qwen3-vl",
            prompt="What objects are visible?",
        )
        result = node.run(ctx, image=img)

        # Multi-image query (standard mode only)
        node = VLMQuery(
            using="qwen3-vl",
            prompt="Compare these images and list differences.",
        )
        result = node.run(ctx, image=img1, images=[img2, img3])

        # Agent mode with tools
        node = VLMQuery(
            using="qwen3-vl",
            prompt="Analyze this scene in detail.",
            tools=["detect", "classify"],
            max_iterations=5,
        )
        result = node.run(ctx, image=img)
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"result": Detections}

    def __init__(
        self,
        using: str,
        prompt: str,
        output_mode: str | None = None,
        out: str = "vlm_result",
        tools: list[str] | None = None,
        max_iterations: int = 5,
        on_error: str = "retry",
        name: str | None = None,
        **vlm_kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.prompt = prompt
        self.output_mode = output_mode
        self.output_name = out
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_error = on_error
        self.vlm_kwargs = vlm_kwargs

    def run(
        self,
        ctx: ExecutionContext,
        image: Image,
        images: list[Image] | None = None,
    ) -> dict[str, Artifact]:
        """Execute VLM query with optional multi-image input.

        Args:
            ctx: Execution context with providers and metrics.
            image: Primary input image artifact.
            images: Optional additional images for multi-image queries.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Detections artifact containing the VLM response.

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
            # Note: AgentLoop currently supports single image only
            # Multi-image support in agent mode is future work
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
            # Combine primary + additional images
            if images:
                all_images = [image] + list(images)
            else:
                all_images = None

            start = time.time()
            result = vlm.query(
                all_images if all_images else image,
                prompt=self.prompt,
                output_mode=self.output_mode,
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
            ctx.record_metric(
                self.name,
                "num_images",
                float(len(all_images)) if all_images else 1.0,
            )
            ctx.record_metric(
                self.name,
                "output_mode",
                0.0,  # Stored as string in meta, numeric placeholder for metrics
            )

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
