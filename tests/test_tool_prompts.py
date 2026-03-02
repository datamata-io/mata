"""Unit tests for tool prompt templates and formatting.

Tests cover:
- System prompt generation with default template
- System prompt generation with custom template
- Tool result message formatting
- Retry prompt generation
- Edge cases and error handling

Test count: 15 tests
"""

import pytest

from mata.core.tool_prompts import (
    TOOL_SYSTEM_PROMPT_TEMPLATE,
    build_tool_call_retry_prompt,
    build_tool_system_prompt,
    format_tool_result_message,
)
from mata.core.tool_schema import ToolResult, schema_for_task


class TestToolSystemPromptTemplate:
    """Tests for TOOL_SYSTEM_PROMPT_TEMPLATE constant."""

    def test_template_contains_placeholder(self):
        """Template should contain {tool_descriptions} placeholder."""
        assert "{tool_descriptions}" in TOOL_SYSTEM_PROMPT_TEMPLATE

    def test_template_contains_format_instructions(self):
        """Template should include format instructions."""
        assert "tool_call" in TOOL_SYSTEM_PROMPT_TEMPLATE
        assert "tool_name" in TOOL_SYSTEM_PROMPT_TEMPLATE or "tool" in TOOL_SYSTEM_PROMPT_TEMPLATE
        assert "arguments" in TOOL_SYSTEM_PROMPT_TEMPLATE

    def test_template_contains_coordinate_format(self):
        """Template should specify coordinate format."""
        assert "x1" in TOOL_SYSTEM_PROMPT_TEMPLATE or "pixel" in TOOL_SYSTEM_PROMPT_TEMPLATE.lower()

    def test_template_contains_termination_signal(self):
        """Template should explain termination (final answer)."""
        assert "final answer" in TOOL_SYSTEM_PROMPT_TEMPLATE.lower()


class TestBuildToolSystemPrompt:
    """Tests for build_tool_system_prompt()."""

    def test_single_tool_renders_correctly(self):
        """System prompt should render with single tool."""
        schemas = [schema_for_task("detect")]
        prompt = build_tool_system_prompt(schemas)

        assert "detect" in prompt.lower()
        assert "tool_call" in prompt
        assert "threshold" in prompt.lower()  # detect has threshold parameter

    def test_multiple_tools_all_included(self):
        """System prompt should include all provided tools."""
        schemas = [
            schema_for_task("detect"),
            schema_for_task("classify"),
            schema_for_task("segment"),
        ]
        prompt = build_tool_system_prompt(schemas)

        assert "detect" in prompt.lower()
        assert "classify" in prompt.lower()
        assert "segment" in prompt.lower()

    def test_four_tools_renders_correctly(self):
        """System prompt should handle 4 tools (max recommended)."""
        schemas = [
            schema_for_task("detect"),
            schema_for_task("classify"),
            schema_for_task("segment"),
            schema_for_task("depth"),
        ]
        prompt = build_tool_system_prompt(schemas)

        assert "detect" in prompt.lower()
        assert "classify" in prompt.lower()
        assert "segment" in prompt.lower()
        assert "depth" in prompt.lower()
        assert len(prompt) > 0

    def test_custom_base_prompt_with_placeholder(self):
        """Custom base prompt should work when it contains placeholder."""
        custom = "Custom instructions\n\n{tool_descriptions}\n\nEnd of instructions."
        schemas = [schema_for_task("detect")]
        prompt = build_tool_system_prompt(schemas, base_prompt=custom)

        assert "Custom instructions" in prompt
        assert "End of instructions" in prompt
        assert "detect" in prompt.lower()

    def test_custom_base_prompt_missing_placeholder_raises(self):
        """Custom base prompt without placeholder should raise ValueError."""
        custom = "No placeholder here"
        schemas = [schema_for_task("detect")]

        with pytest.raises(ValueError, match="tool_descriptions"):
            build_tool_system_prompt(schemas, base_prompt=custom)

    def test_empty_schemas_raises(self):
        """Empty schema list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            build_tool_system_prompt([])

    def test_tool_descriptions_formatted_correctly(self):
        """Tool descriptions should be formatted with proper structure."""
        schemas = [schema_for_task("detect")]
        prompt = build_tool_system_prompt(schemas)

        # Should contain tool description format from ToolSchema.to_prompt_str()
        assert "Tool:" in prompt or "tool:" in prompt.lower()
        assert "Description:" in prompt or "description:" in prompt.lower()
        assert "Parameters:" in prompt or "parameters:" in prompt.lower()


class TestFormatToolResultMessage:
    """Tests for format_tool_result_message()."""

    def test_successful_result_formatting(self):
        """Successful tool result should format with checkmark."""
        result = ToolResult(
            tool_name="detect",
            success=True,
            summary="Found 3 objects: cat (0.95), dog (0.87), person (0.72)",
            artifacts={},
        )
        msg = format_tool_result_message(result)

        assert "detect" in msg.lower()
        assert "✓" in msg
        assert "Found 3 objects" in msg

    def test_failed_result_formatting(self):
        """Failed tool result should format with X mark."""
        result = ToolResult(
            tool_name="classify",
            success=False,
            summary="Error: region exceeds image bounds",
            artifacts={},
        )
        msg = format_tool_result_message(result)

        assert "classify" in msg.lower()
        assert "✗" in msg
        assert "Error:" in msg

    def test_result_message_is_concise(self):
        """Result message should be reasonably concise."""
        result = ToolResult(
            tool_name="segment",
            success=True,
            summary="Segmented 5 instances",
            artifacts={},
        )
        msg = format_tool_result_message(result)

        # Should not add excessive wrapper text
        assert len(msg) < len(result.summary) + 100


class TestBuildToolCallRetryPrompt:
    """Tests for build_tool_call_retry_prompt()."""

    def test_retry_prompt_includes_error(self):
        """Retry prompt should include parse error."""
        schemas = [schema_for_task("detect")]
        prompt = build_tool_call_retry_prompt(
            original_output="detect objects please",
            parse_error="No JSON found in output",
            schemas=schemas,
        )

        assert "No JSON found" in prompt
        assert "error" in prompt.lower()

    def test_retry_prompt_includes_available_tools(self):
        """Retry prompt should list available tool names."""
        schemas = [schema_for_task("detect"), schema_for_task("classify")]
        prompt = build_tool_call_retry_prompt(
            original_output="bad format",
            parse_error="Invalid JSON",
            schemas=schemas,
        )

        assert "detect" in prompt.lower()
        assert "classify" in prompt.lower()
        assert "Available" in prompt or "tools" in prompt.lower()

    def test_retry_prompt_includes_format_example(self):
        """Retry prompt should show correct format."""
        schemas = [schema_for_task("detect")]
        prompt = build_tool_call_retry_prompt(
            original_output="bad",
            parse_error="Invalid",
            schemas=schemas,
        )

        assert "tool_call" in prompt
        assert "tool" in prompt.lower()
        assert "arguments" in prompt.lower()

    def test_retry_prompt_truncates_long_output(self):
        """Retry prompt should truncate very long original output."""
        schemas = [schema_for_task("detect")]
        long_output = "x" * 500
        prompt = build_tool_call_retry_prompt(
            original_output=long_output,
            parse_error="Too long",
            schemas=schemas,
        )

        # Should not include full 500 chars
        assert prompt.count("x") < 300
        assert "..." in prompt or len(prompt) < len(long_output) + 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
