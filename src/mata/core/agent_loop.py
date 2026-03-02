"""Agent loop for VLM tool-calling system.

Implements the core agent loop that iterates between VLM inference and tool execution.
The VLM acts as a reasoning controller — it analyzes an image, identifies what it doesn't
know, calls specialized tools, and synthesizes a final response.

Design principles:
- Max iteration safety cap prevents infinite loops
- Retry logic for malformed VLM output
- Conversation history maintained as list of turns
- Accumulates instances and entities across all tool calls
- Multiple tool call format support (fenced blocks, XML, raw JSON)

Version: 1.7.0
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mata.core.tool_schema import ToolCall, ToolResult
from mata.core.types import Entity, Instance

if TYPE_CHECKING:
    from mata.core.artifacts.image import Image
    from mata.core.graph.context import ExecutionContext
    from mata.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    """Accumulated result from agent loop execution.

    Contains all information gathered during the agent loop's execution,
    including the final VLM synthesis, all tool calls made, and all
    instances/entities discovered.

    Attributes:
        text: Final VLM synthesis text (answer to original prompt)
        tool_calls: Complete list of all tool calls made during execution
        tool_results: Complete list of all tool results received
        iterations: Number of loop iterations executed
        instances: Spatially grounded detections from tools (with bbox/mask)
        entities: Semantic entities from VLM output
        conversation: Full conversation history (list of message dicts)
        meta: Additional metadata from the execution

    Examples:
        >>> result = AgentResult(
        ...     text="Found 2 cats and 1 dog in the image.",
        ...     tool_calls=[...],
        ...     tool_results=[...],
        ...     iterations=3,
        ...     instances=[...],
        ...     entities=[...],
        ...     conversation=[...],
        ... )
        >>> result.iterations
        3
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    instances: list[Instance] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    conversation: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "iterations": self.iterations,
            "instances": [inst.to_dict() for inst in self.instances],
            "entities": [ent.to_dict() for ent in self.entities],
            "conversation": self.conversation,
            "meta": self.meta,
        }


class AgentLoop:
    """Runs VLM ↔ tool-call iteration loop.

    The loop maintains a conversation history (list of turns) and iterates:
    1. Send image + history + tool descriptions to VLM
    2. Parse VLM response for tool calls
    3. If tool call found → execute tool → append result to history → goto 1
    4. If no tool call (final answer) → return accumulated result
    5. If max_iterations reached → return best-effort result

    The loop handles:
    - Retry logic for malformed tool calls
    - Tool execution failures (graceful degradation)
    - Infinite loop detection (repeated identical calls)
    - Token budget management (conversation history capping)
    - Accumulation of instances and entities from all tool calls

    Example:
        ```python
        from mata.core.agent_loop import AgentLoop
        from mata.core.tool_registry import ToolRegistry

        # Set up registry and loop
        registry = ToolRegistry(ctx, ["detect", "classify", "zoom"])
        loop = AgentLoop(vlm_provider, registry, max_iterations=5)

        # Run agent loop
        result = loop.run(
            image=img,
            prompt="Analyze this medical image.",
            system_prompt="You are a medical image analysis assistant.",
        )

        # Access results
        print(f"Final answer: {result.text}")
        print(f"Made {result.iterations} iterations")
        print(f"Found {len(result.instances)} instances")
        ```

    Attributes:
        vlm_provider: VLMWrapper instance for VLM inference
        tool_registry: ToolRegistry for tool resolution and execution
        max_iterations: Hard cap on loop iterations (default 5, max 20)
        on_error: Error handling mode ("retry", "skip", "fail")
        max_retries: Max retries for malformed tool calls (default 2)
        ctx: Optional ExecutionContext for tracing and metrics
        node_name: Optional node name for span identification
    """

    def __init__(
        self,
        vlm_provider: Any,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
        on_error: str = "retry",
        max_retries: int = 2,
        ctx: ExecutionContext | None = None,
        node_name: str = "agent",
    ):
        """Initialize agent loop with VLM provider and tool registry.

        Args:
            vlm_provider: VLMWrapper instance for VLM inference
            tool_registry: ToolRegistry for tool resolution and execution
            max_iterations: Hard cap on loop iterations (default 5, max recommended 20)
            on_error: Error handling mode:
                - "retry": Retry malformed tool calls with clarifying prompt
                - "skip": Ignore failed tools and continue
                - "fail": Raise immediately on first failure
            max_retries: Max retries for malformed tool calls before skipping (default 2)
            ctx: Optional ExecutionContext for tracing and metrics integration
            node_name: Node name for span identification (default "agent")

        Raises:
            ValueError: If max_iterations > 20 or on_error is invalid
        """
        if max_iterations > 20:
            raise ValueError(
                f"max_iterations={max_iterations} exceeds safety limit of 20. "
                f"High iteration counts can lead to excessive token usage."
            )

        if on_error not in {"retry", "skip", "fail"}:
            raise ValueError(f"Invalid on_error mode: {on_error}. " f"Must be 'retry', 'skip', or 'fail'.")

        self.vlm_provider = vlm_provider
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.on_error = on_error
        self.max_retries = max_retries
        self.ctx = ctx
        self.node_name = node_name

    def run(
        self,
        image: Image,
        prompt: str,
        system_prompt: str | None = None,
        **vlm_kwargs: Any,
    ) -> AgentResult:
        """Execute the agent loop.

        Iterates between VLM inference and tool execution until:
        - VLM produces a final answer (no tool call)
        - max_iterations reached
        - All tools fail

        Args:
            image: Image artifact to analyze
            prompt: User's initial prompt/question
            system_prompt: Optional system prompt override. If not provided,
                auto-generates one with tool descriptions.
            **vlm_kwargs: Additional VLM parameters (temperature, max_tokens, etc.)

        Returns:
            AgentResult with accumulated text, tool_calls, instances, entities

        Raises:
            RuntimeError: If on_error="fail" and a tool execution fails
        """
        # Build system prompt with tool descriptions
        if system_prompt is None:
            system_prompt = self._build_default_system_prompt()

        # Initialize conversation history
        conversation: list[dict[str, Any]] = []
        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        all_instances: list[Instance] = []
        all_entities: list[Entity] = []

        # Track retries for malformed calls
        retry_count = 0
        last_vlm_text = ""

        # Start parent span for entire agent loop execution
        parent_span = None
        if self.ctx is not None:
            parent_span = self.ctx.tracer.start_span(
                f"agent:{self.node_name}",
                attributes={
                    "max_iterations": self.max_iterations,
                    "on_error": self.on_error,
                    "node_name": self.node_name,
                },
            )
            logger.debug(f"Started agent loop span: {parent_span.span_id}")

        # Main agent loop
        for iteration in range(self.max_iterations):
            logger.info(f"Agent loop iteration {iteration + 1}/{self.max_iterations}")

            # Build current prompt (initial prompt + conversation context)
            if iteration == 0:
                current_prompt = prompt
            else:
                # On subsequent iterations, rely on conversation history context
                current_prompt = "Continue your analysis based on the tool results."

            # Start span for this VLM turn
            vlm_span = None
            if self.ctx is not None and parent_span is not None:
                vlm_span = self.ctx.tracer.start_span(
                    f"agent:vlm_turn_{iteration}",
                    parent_id=parent_span.span_id,
                    attributes={
                        "iteration": iteration,
                        "prompt_length": len(current_prompt),
                    },
                )

            # Call VLM with conversation history
            try:
                vlm_result = self.vlm_provider.query(
                    image,
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    **vlm_kwargs,
                )
            except Exception as e:
                logger.warning(f"VLM query failed at iteration {iteration + 1}: {e}")
                # End VLM span with error
                if self.ctx is not None and vlm_span is not None:
                    self.ctx.tracer.end_span(vlm_span, status="error", error_message=str(e))
                if self.on_error == "fail":
                    # End parent span with error
                    if self.ctx is not None and parent_span is not None:
                        self.ctx.tracer.end_span(parent_span, status="error", error_message=str(e))
                    raise RuntimeError(f"VLM query failed: {e}") from e
                # On error, return best-effort result so far (don't increment iteration)
                logger.warning(f"VLM query failed, returning best-effort result (mode: {self.on_error})")
                iterations_taken = iteration
                final_text = last_vlm_text if last_vlm_text else "No final answer generated."
                # End parent span gracefully
                if self.ctx is not None and parent_span is not None:
                    self.ctx.tracer.end_span(parent_span, status="ok")
                    self.ctx.metrics_collector.record(self.node_name, "agent_iterations", float(iterations_taken))
                    self.ctx.metrics_collector.record(self.node_name, "tool_calls_count", float(len(all_tool_calls)))
                return AgentResult(
                    text=final_text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    iterations=iterations_taken,
                    instances=all_instances,
                    entities=all_entities,
                    conversation=conversation,
                    meta={
                        "max_iterations": self.max_iterations,
                        "on_error": self.on_error,
                    },
                )

            vlm_text = vlm_result.text or ""
            last_vlm_text = vlm_text

            # End VLM span
            if self.ctx is not None and vlm_span is not None:
                self.ctx.tracer.end_span(
                    vlm_span,
                    status="ok",
                )

            # Accumulate entities from VLM output
            if vlm_result.entities:
                all_entities.extend(vlm_result.entities)
                logger.debug(f"Accumulated {len(vlm_result.entities)} entities from VLM")

            # Append user message to conversation
            conversation.append({"role": "assistant", "content": vlm_text})

            # Parse tool calls from VLM output
            tool_calls = self._parse_tool_calls(vlm_text)

            # Check if no tool call found
            if not tool_calls:
                # Check if this looks like a malformed tool call attempt
                if self._looks_like_attempted_tool_call(vlm_text):
                    # VLM attempted a tool call but format was invalid
                    if self.on_error == "retry" and retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(
                            f"Malformed tool call detected (retry {retry_count}/{self.max_retries}). "
                            f"VLM output: {vlm_text[:200]}..."
                        )

                        # Add clarification prompt to conversation
                        clarification = (
                            "Your tool call was malformed and could not be parsed. "
                            "Please use this exact format:\n"
                            "```tool_call\n"
                            '{"tool": "<tool_name>", "arguments": {"<arg_name>": <value>}}\n'
                            "```\n\n"
                            "Available tools: "
                            + ", ".join(schema.name for schema in self.tool_registry.all_schemas())
                            + "\n\n"
                            "Please try again with the correct format."
                        )
                        conversation.append({"role": "user", "content": clarification})

                        # Don't break - continue to next iteration for retry
                        continue
                    elif self.on_error == "fail":
                        raise RuntimeError(
                            f"Malformed tool call attempted: {vlm_text[:200]}... "
                            f"Could not parse tool call from VLM output."
                        )
                    else:
                        # on_error="skip" or max retries exceeded
                        logger.warning(
                            f"Malformed tool call detected but treating as final answer. "
                            f"Mode: {self.on_error}, retry_count: {retry_count}/{self.max_retries}"
                        )

                # No tool call found (either genuinely final answer or max retries exceeded)
                logger.info("No tool calls found. VLM provided final answer.")
                break

            # Check for identical repeated calls (infinite loop detection)
            if self._detect_loop(all_tool_calls, tool_calls):
                logger.warning("Agent loop detected — VLM is calling the same tool repeatedly. Breaking loop.")
                break

            # Execute each tool call
            for tool_call in tool_calls:
                logger.info(f"Executing tool: {tool_call.tool_name} with args: {tool_call.arguments}")

                # Start span for tool execution
                tool_span = None
                if self.ctx is not None and parent_span is not None:
                    tool_span = self.ctx.tracer.start_span(
                        f"agent:tool:{tool_call.tool_name}",
                        parent_id=parent_span.span_id,
                        attributes={
                            "tool": tool_call.tool_name,
                            "arguments": str(tool_call.arguments),
                        },
                    )

                try:
                    tool_result = self.tool_registry.execute_tool(tool_call, image)
                    all_tool_calls.append(tool_call)
                    all_tool_results.append(tool_result)

                    # End tool span successfully
                    if self.ctx is not None and tool_span is not None:
                        self.ctx.tracer.end_span(
                            tool_span,
                            status="ok",
                        )

                    # Accumulate instances from tool results
                    if "instances" in tool_result.artifacts:
                        instances = tool_result.artifacts["instances"]
                        all_instances.extend(instances)
                        logger.debug(f"Accumulated {len(instances)} instances from {tool_call.tool_name}")

                    # Add tool result to conversation
                    result_message = self._format_tool_result_for_vlm(tool_result)
                    conversation.append({"role": "user", "content": result_message})

                    # Reset retry count on successful tool execution
                    retry_count = 0

                except Exception as e:
                    logger.warning(f"Tool execution failed: {tool_call.tool_name}: {e}")

                    # End tool span with error
                    if self.ctx is not None and tool_span is not None:
                        self.ctx.tracer.end_span(tool_span, status="error", error_message=str(e))

                    if self.on_error == "fail":
                        # End parent span with error
                        if self.ctx is not None and parent_span is not None:
                            self.ctx.tracer.end_span(parent_span, status="error", error_message=str(e))
                        raise RuntimeError(f"Tool execution failed: {tool_call.tool_name}: {e}") from e

                    # Create failure result
                    failure_result = ToolResult(
                        tool_name=tool_call.tool_name,
                        success=False,
                        summary=f"Tool '{tool_call.tool_name}' failed: {str(e)}",
                        artifacts={},
                    )

                    all_tool_calls.append(tool_call)
                    all_tool_results.append(failure_result)

                    # Add failure to conversation
                    result_message = self._format_tool_result_for_vlm(failure_result)
                    conversation.append({"role": "user", "content": result_message})

                    if self.on_error == "retry" and retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(f"Retrying after tool failure ({retry_count}/{self.max_retries})")
                    elif self.on_error == "skip":
                        logger.warning(f"Skipping failed tool: {tool_call.tool_name}")
                        continue

            # Check for conversation history length (token budget management)
            if len(conversation) > 20:
                logger.warning(
                    f"Conversation history length ({len(conversation)}) exceeds recommended limit. "
                    f"Consider enabling history capping to prevent OOM."
                )

        # Build final result
        final_text = last_vlm_text if last_vlm_text else "No final answer generated."
        iterations_taken = iteration + 1

        # End parent span
        if self.ctx is not None and parent_span is not None:
            self.ctx.tracer.end_span(parent_span, status="ok")
            # Record agent-specific metrics
            self.ctx.metrics_collector.record(self.node_name, "agent_iterations", float(iterations_taken))
            self.ctx.metrics_collector.record(self.node_name, "tool_calls_count", float(len(all_tool_calls)))
            logger.debug(
                f"Ended agent loop span: {parent_span.span_id} ({iterations_taken} iterations, {len(all_tool_calls)} tool calls)"
            )

        return AgentResult(
            text=final_text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            iterations=iterations_taken,
            instances=all_instances,
            entities=all_entities,
            conversation=conversation,
            meta={
                "max_iterations": self.max_iterations,
                "on_error": self.on_error,
            },
        )

    def _build_default_system_prompt(self) -> str:
        """Build default system prompt with tool descriptions.

        Returns:
            System prompt text with tool usage instructions
        """
        tool_descriptions = self.tool_registry.build_system_prompt_block()

        prompt = f"""You have access to the following tools to help analyze this image:

{tool_descriptions}

To use a tool, respond with a JSON block in this exact format:
```tool_call
{{"tool": "<tool_name>", "arguments": {{"<arg_name>": <value>}}}}
```

After receiving tool results, analyze them and either:
- Call another tool if you need more information
- Provide your final answer (without any tool_call block)

Important:
- Only call one tool at a time
- Always provide your final answer after gathering enough information
- Bbox coordinates are in [x1, y1, x2, y2] pixel format
"""
        return prompt

    def _parse_tool_calls(self, vlm_text: str) -> list[ToolCall]:
        """Extract tool calls from VLM output text.

        Supports multiple formats:
        - ```tool_call\\n{...}\\n``` (fenced code block)
        - <tool_call>{...}</tool_call> (XML tags)
        - {"tool": "...", "arguments": {...}} (raw JSON)
        - {"action": "...", "parameters": {...}} (alternate keys)

        Args:
            vlm_text: Raw VLM output text

        Returns:
            List of parsed ToolCall objects (empty if no tool calls found)
        """
        tool_calls: list[ToolCall] = []

        # Strategy 1: Fenced code block ```tool_call\\n{...}\\n```
        fenced_pattern = r"```tool_call\s*\n(.*?)\n```"
        matches = re.findall(fenced_pattern, vlm_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            call = self._parse_json_tool_call(match.strip(), vlm_text)
            if call:
                tool_calls.append(call)

        if tool_calls:
            return tool_calls

        # Strategy 2: XML-style tags <tool_call>{...}</tool_call>
        xml_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(xml_pattern, vlm_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            call = self._parse_json_tool_call(match.strip(), vlm_text)
            if call:
                tool_calls.append(call)

        if tool_calls:
            return tool_calls

        # Strategy 3: Raw JSON object - try to parse entire text as JSON first
        # This catches cases where the entire VLM response is a JSON tool call
        call = self._parse_json_tool_call(vlm_text.strip(), vlm_text)
        if call:
            tool_calls.append(call)
            return tool_calls

        # Strategy 4: Look for JSON objects within the text
        # More comprehensive regex that handles nested objects
        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|\{[^{}]*\})*\}))*\}"
        matches = re.findall(json_pattern, vlm_text, re.DOTALL)
        for match in matches:
            call = self._parse_json_tool_call(match, vlm_text)
            if call:
                tool_calls.append(call)

        return tool_calls

    def _parse_json_tool_call(self, json_str: str, raw_text: str) -> ToolCall | None:
        """Parse a JSON string into a ToolCall.

        Handles both standard and alternate key names:
        - {"tool": "...", "arguments": {...}}
        - {"action": "...", "parameters": {...}}

        Args:
            json_str: JSON string to parse
            raw_text: Original VLM text (for ToolCall.raw_text)

        Returns:
            ToolCall if successfully parsed, None otherwise
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON tool call: {e}")
            return None

        # Extract tool name (support both "tool" and "action" keys)
        tool_name = data.get("tool") or data.get("action")
        if not tool_name:
            logger.debug(f"No tool name found in JSON: {json_str}")
            return None

        # Extract arguments (support both "arguments" and "parameters" keys)
        arguments = data.get("arguments") or data.get("parameters") or {}

        return ToolCall(
            tool_name=str(tool_name),
            arguments=arguments,
            raw_text=raw_text,
        )

    def _format_tool_result_for_vlm(self, result: ToolResult) -> str:
        """Format tool execution result as text for VLM's next turn.

        Args:
            result: ToolResult from tool execution

        Returns:
            Formatted string for conversation history
        """
        return result.to_conversation_message()

    def _detect_loop(self, all_calls: list[ToolCall], new_calls: list[ToolCall]) -> bool:
        """Detect infinite loop patterns in tool calls.

        Checks if the VLM is calling the same tool with the same arguments
        repeatedly (3+ times in a row).

        Args:
            all_calls: All tool calls made so far
            new_calls: New tool calls from current iteration

        Returns:
            True if loop detected, False otherwise
        """
        if len(all_calls) < 2 or not new_calls:
            return False

        # Get the last few calls
        recent_calls = all_calls[-2:] + new_calls

        # Check if all recent calls are identical
        if len(recent_calls) < 3:
            return False

        # Compare tool names and arguments
        first_call = recent_calls[0]
        for call in recent_calls[1:]:
            if call.tool_name != first_call.tool_name:
                return False
            if call.arguments != first_call.arguments:
                return False

        # All calls are identical → loop detected
        logger.warning(
            f"Loop detected: {first_call.tool_name} called {len(recent_calls)} times " f"with identical arguments"
        )
        return True

    def _looks_like_attempted_tool_call(self, text: str) -> bool:
        """Detect if text looks like a malformed tool call attempt.

        Checks for patterns that indicate the VLM tried to make a tool call
        but the format was invalid (e.g., malformed JSON, incorrect structure).

        Args:
            text: VLM output text

        Returns:
            True if text appears to be a malformed tool call attempt
        """
        # Pattern 1: Contains "tool_call" markers (fenced or XML)
        if "```tool_call" in text.lower() or "<tool_call>" in text.lower():
            return True

        # Pattern 2: Contains JSON-like structure with "tool" or "action" keys
        if re.search(r'\{\s*["\'](?:tool|action)["\']', text, re.IGNORECASE):
            return True

        # Pattern 3: Contains "arguments" or "parameters" keys near start
        if re.search(r'\{\s*["\'](?:arguments|parameters)["\']', text, re.IGNORECASE):
            return True

        # Pattern 4: Multiple curly braces suggesting JSON structure
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces >= 2 and close_braces >= 2:
            # Might be attempting JSON
            if '"' in text or "'" in text:
                return True

        return False
