"""VLMDescribe node — generate natural language description using VLM.

Queries a Vision-Language Model provider to produce a natural language
description of an input image, returned as a VisionResult artifact
containing the text response and optional entities.
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


class VLMDescribe(Node):
    """Generate natural language description using a VLM.

    Looks up a ``VisionLanguageModel`` provider from the execution context,
    queries it with the input image and a descriptive prompt, and returns
    the VLM's response as a Detections artifact (wrapping VisionResult).

    Args:
        using: Name of the VLM provider registered in the context
            (e.g. ``"qwen3-vl"``, ``"gemini"``).
        prompt: Text prompt sent to the VLM
            (default ``"Describe this image in detail."``).
        out: Key under which the output artifact is stored
            (default ``"description"``).
        tools: Optional list of tool names to enable agent mode.
            When provided, the VLM can call these tools iteratively
            to gather information before generating the description.
            (default ``None`` = standard single-call mode).
        max_iterations: Maximum number of agent loop iterations
            when tools are enabled (default ``5``).
        on_error: Error handling mode for tool execution failures:
            ``"retry"`` = retry malformed calls, ``"skip"`` = continue,
            ``"fail"`` = raise immediately (default ``"retry"``).
        name: Optional human-readable node name.
        **vlm_kwargs: Extra keyword arguments forwarded to the provider's
            ``query()`` call (e.g. ``max_tokens``, ``temperature``).

    Inputs:
        image (Image): Input image artifact.

    Outputs:
        description (Detections): VLM description result containing text,
            optional entities, and metadata.

    Example:
        ```python
        from mata.nodes import VLMDescribe

        # Standard mode
        node = VLMDescribe(using="qwen3-vl", prompt="What is in this image?")
        result = node.run(ctx, image=img)
        desc = result["description"]

        # Agent mode - VLM can use tools to gather info first
        node = VLMDescribe(
            using="qwen3-vl",
            prompt="Provide a detailed description of this scene.",
            tools=["detect", "classify", "depth"],
            max_iterations=5,
        )
        result = node.run(ctx, image=img)
        desc = result["description"]  # Enhanced with tool-gathered info
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"description": Detections}

    def __init__(
        self,
        using: str,
        prompt: str = "Describe this image in detail.",
        out: str = "description",
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
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_error = on_error
        self.vlm_kwargs = vlm_kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Execute VLM description on the input image.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict with a single key (``self.output_name``) mapping to
            a Detections artifact containing the VLM description.

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
            ctx.record_metric(
                self.name,
                "response_length",
                float(len(agent_result.text)) if agent_result.text else 0.0,
            )
        else:
            # Standard mode — single VLM call (unchanged)
            start = time.time()
            result = vlm.query(
                image,
                prompt=self.prompt,
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
                "response_length",
                float(len(result.text)) if hasattr(result, "text") and result.text else 0.0,
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
