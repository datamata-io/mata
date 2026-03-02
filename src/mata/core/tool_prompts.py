"""System prompt templates for VLM tool-calling.

This module provides templates and utilities for instructing VLMs how to use tools.
Different VLM families may need different prompt formats, but we start with a universal
template optimized for Qwen3-VL and similar open-weight vision-language models.

The prompt engineering approach:
- Explicit format instructions with examples
- Single tool call per turn (simplifies parsing)
- Clear termination signal (no tool_call block = final answer)
- Coordinate format specification (xyxy pixel format)

Version: 1.7.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mata.core.tool_schema import ToolResult, ToolSchema


# Universal system prompt template for tool-calling VLMs
# Works with Qwen3-VL, LLaVA, and similar open-weight models
TOOL_SYSTEM_PROMPT_TEMPLATE = """
You have access to the following tools to help analyze this image:

{tool_descriptions}

To use a tool, respond with a JSON block in this exact format:
```tool_call
{{"tool": "<tool_name>", "arguments": {{<args>}}}}
```

After receiving tool results, analyze them and either:
- Call another tool if you need more information
- Provide your final answer (without any tool_call block)

Important guidelines:
- Only call one tool at a time
- Always provide your final answer after gathering enough information
- Bbox coordinates are in [x1, y1, x2, y2] pixel format (absolute pixels)
- Use null for optional parameters if not needed
- Tool names must match exactly (case-sensitive)

Example valid tool call:
```tool_call
{{"tool": "detect", "arguments": {{"threshold": 0.5, "region": null}}}}
```

Example final answer:
There are 3 objects in the image: 2 cats (confidence 0.95, 0.89) and 1 dog (confidence 0.87).
"""


def build_tool_system_prompt(
    schemas: list[ToolSchema],
    base_prompt: str | None = None,
) -> str:
    """Build complete system prompt with tool descriptions and instructions.

    Generates a system prompt that instructs the VLM how to use available tools.
    The prompt includes:
    - Tool descriptions (from ToolSchema.to_prompt_str())
    - Format instructions for tool calls
    - Termination conditions
    - Example usage

    Args:
        schemas: List of available tool schemas (1-4 tools recommended)
        base_prompt: Optional custom base prompt to replace default template.
                    Use {tool_descriptions} placeholder for tool list insertion.

    Returns:
        Complete system prompt string ready for VLM consumption.

    Raises:
        ValueError: If schemas is empty or base_prompt is missing {tool_descriptions}.

    Examples:
        >>> from mata.core.tool_schema import schema_for_task
        >>> schemas = [schema_for_task("detect"), schema_for_task("classify")]
        >>> prompt = build_tool_system_prompt(schemas)
        >>> "tool_call" in prompt
        True
        >>> "detect" in prompt
        True

        >>> # With custom base prompt
        >>> custom = "Custom instructions\\n{tool_descriptions}\\nEnd."
        >>> prompt = build_tool_system_prompt(schemas, base_prompt=custom)
        >>> "Custom instructions" in prompt
        True
    """
    if not schemas:
        raise ValueError("At least one ToolSchema is required to build system prompt")

    # Generate tool descriptions block
    tool_descriptions = "\n\n".join(schema.to_prompt_str() for schema in schemas)

    # Use custom prompt or default template
    if base_prompt is not None:
        if "{tool_descriptions}" not in base_prompt:
            raise ValueError("Custom base_prompt must contain {tool_descriptions} placeholder")
        return base_prompt.format(tool_descriptions=tool_descriptions)

    # Use default template
    return TOOL_SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)


def format_tool_result_message(result: ToolResult) -> str:
    """Format a tool result as a user-turn message for conversation history.

    Converts a tool execution result into a text message suitable for adding to
    the VLM's conversation history. The message includes:
    - Tool name
    - Success/failure status (✓/✗)
    - Human-readable summary of results

    This is the message the VLM sees in the next turn after calling a tool.

    Args:
        result: The tool execution result to format.

    Returns:
        Formatted string for conversation history.

    Examples:
        >>> from mata.core.tool_schema import ToolResult
        >>> result = ToolResult(
        ...     tool_name="detect",
        ...     success=True,
        ...     summary="Found 3 objects: cat (0.95), dog (0.87), person (0.72)",
        ...     artifacts={}
        ... )
        >>> msg = format_tool_result_message(result)
        >>> msg
        '[Tool: detect] ✓ Found 3 objects: cat (0.95), dog (0.87), person (0.72)'

        >>> # Failed tool call
        >>> error_result = ToolResult(
        ...     tool_name="classify",
        ...     success=False,
        ...     summary="Error: region [1000, 1000, 2000, 2000] exceeds image bounds (800x600)",
        ...     artifacts={}
        ... )
        >>> msg = format_tool_result_message(error_result)
        >>> "✗" in msg
        True
        >>> "Error:" in msg
        True
    """
    # Delegate to ToolResult's built-in method for consistency
    return result.to_conversation_message()


def build_tool_call_retry_prompt(
    original_output: str,
    parse_error: str,
    schemas: list[ToolSchema],
) -> str:
    """Build a retry prompt for malformed tool calls.

    When the VLM produces a tool call that can't be parsed, this generates
    a clarifying prompt with:
    - What went wrong (parse error)
    - The correct format (with examples)
    - Available tool names

    Args:
        original_output: The VLM's malformed output
        parse_error: Description of parsing failure
        schemas: Available tool schemas

    Returns:
        Retry prompt to send back to VLM.

    Examples:
        >>> from mata.core.tool_schema import schema_for_task
        >>> schemas = [schema_for_task("detect")]
        >>> prompt = build_tool_call_retry_prompt(
        ...     original_output='detect objects',
        ...     parse_error='No JSON found in output',
        ...     schemas=schemas
        ... )
        >>> "format" in prompt.lower()
        True
        >>> "tool_call" in prompt
        True
    """
    tool_names = [s.name for s in schemas]
    tool_list = ", ".join(f'"{name}"' for name in tool_names)

    return f"""Your previous response could not be parsed as a tool call.

**Error**: {parse_error}

**Your output**:
{original_output[:200]}{'...' if len(original_output) > 200 else ''}

**Correct format** (use exactly this structure):
```tool_call
{{"tool": "<tool_name>", "arguments": {{<args>}}}}
```

**Available tools**: {tool_list}

Please retry the tool call using the correct format, or provide your final answer without a tool_call block.
"""
