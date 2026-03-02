"""Unit tests for tool-call parsing (Task D2).

Tests parse_tool_calls() and validate_tool_call() functions in src/mata/core/parsers.py.
All tests run in isolation without model dependencies.
"""

from __future__ import annotations

import pytest

from mata.core.parsers import parse_tool_calls, validate_tool_call
from mata.core.tool_schema import ToolCall, ToolParameter, ToolSchema


class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_parse_fenced_block_format(self):
        """Test parsing tool call from fenced code block."""
        text = '```tool_call\n{"tool": "detect", "arguments": {"threshold": 0.5}}\n```'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "detect"
        assert result[0].arguments == {"threshold": 0.5}

    def test_parse_fenced_block_case_insensitive(self):
        """Test that fenced block tag is case-insensitive."""
        text = '```TOOL_CALL\n{"tool": "classify", "arguments": {}}\n```'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "classify"

    def test_parse_xml_format(self):
        """Test parsing tool call from XML-style tags."""
        text = '<tool_call>{"tool": "segment", "arguments": {"threshold": 0.3}}</tool_call>'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "segment"
        assert result[0].arguments == {"threshold": 0.3}

    def test_parse_xml_with_surrounding_text(self):
        """Test parsing XML format with surrounding text."""
        text = 'I will call a tool: <tool_call>{"tool": "depth", "arguments": {}}</tool_call> to analyze this.'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "depth"

    def test_parse_raw_json_tool_key(self):
        """Test parsing raw JSON with 'tool' and 'arguments' keys."""
        text = '{"tool": "detect", "arguments": {"threshold": 0.7, "region": [0, 0, 100, 100]}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "detect"
        assert result[0].arguments["threshold"] == 0.7
        assert result[0].arguments["region"] == [0, 0, 100, 100]

    def test_parse_raw_json_action_key(self):
        """Test parsing raw JSON with 'action' and 'parameters' keys."""
        text = '{"action": "classify", "parameters": {"top_k": 5}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "classify"
        assert result[0].arguments["top_k"] == 5

    def test_parse_raw_json_name_key(self):
        """Test parsing raw JSON with 'name' and 'arguments' keys."""
        text = '{"name": "zoom", "arguments": {"region": [100, 100, 200, 200], "scale": 2.0}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "zoom"
        assert result[0].arguments["scale"] == 2.0

    def test_parse_flat_argument_format(self):
        """Test parsing JSON with flat argument format (no nested 'arguments' key)."""
        text = '{"tool": "detect", "threshold": 0.5, "region": null}'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "detect"
        assert result[0].arguments["threshold"] == 0.5
        assert result[0].arguments["region"] is None

    def test_parse_final_answer_no_tool_call(self):
        """Test that text without tool call returns None (final answer mode)."""
        text = "I found 3 cats and 2 dogs in the image. The cats are orange and the dogs are brown."
        result = parse_tool_calls(text)

        assert result is None

    def test_parse_empty_text(self):
        """Test that empty text returns None."""
        assert parse_tool_calls("") is None
        assert parse_tool_calls("   ") is None
        assert parse_tool_calls("\n\t") is None

    def test_parse_malformed_json(self):
        """Test that malformed JSON returns None."""
        text = '```tool_call\n{"tool": "detect", "arguments": {threshold: 0.5}}\n```'  # Missing quotes
        result = parse_tool_calls(text)

        assert result is None

    def test_parse_json_without_tool_key(self):
        """Test that JSON without tool/action/name key returns None."""
        text = '{"threshold": 0.5, "region": [0, 0, 100, 100]}'
        result = parse_tool_calls(text)

        assert result is None

    def test_parse_priority_fenced_over_xml(self):
        """Test that fenced blocks are prioritized over XML tags."""
        text = '```tool_call\n{"tool": "detect", "arguments": {}}\n```<tool_call>{"tool": "classify", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "detect"  # Fenced block wins

    def test_parse_priority_xml_over_raw_json(self):
        """Test that XML tags are prioritized over raw JSON."""
        # Craft text with both XML and raw JSON (no fenced block)
        text = 'Here is my call: <tool_call>{"tool": "segment", "arguments": {}}</tool_call> or {"tool": "detect", "arguments": {}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert len(result) == 1
        assert result[0].tool_name == "segment"  # XML wins

    def test_parse_arguments_as_args_key(self):
        """Test parsing with 'args' key instead of 'arguments'."""
        text = '{"tool": "classify", "args": {"top_k": 3}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert result[0].arguments["top_k"] == 3

    def test_parse_arguments_as_params_key(self):
        """Test parsing with 'params' key instead of 'arguments'."""
        text = '{"action": "zoom", "params": {"scale": 3.0}}'
        result = parse_tool_calls(text)

        assert result is not None
        assert result[0].arguments["scale"] == 3.0


class TestValidateToolCall:
    """Tests for validate_tool_call function."""

    def test_exact_match_tool_name(self):
        """Test validation with exact tool name match."""
        schema = ToolSchema(
            name="detect",
            description="Run detection",
            task="detect",
            parameters=[
                ToolParameter("threshold", "float", "Confidence threshold", required=False, default=0.3),
            ],
        )

        call = ToolCall("detect", {"threshold": 0.5})
        validated = validate_tool_call(call, [schema])

        assert validated.tool_name == "detect"
        assert validated.arguments["threshold"] == 0.5

    def test_fuzzy_match_tool_name_prefix(self):
        """Test fuzzy matching with prefix (detection → detect)."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("detection", {})
        validated = validate_tool_call(call, [schema])

        assert validated.tool_name == "detect"

    def test_fuzzy_match_tool_name_suffix(self):
        """Test fuzzy matching with suffix (classify → classifier)."""
        schema = ToolSchema("classify", "Run classification", "classify", [])

        call = ToolCall("classifier", {})
        validated = validate_tool_call(call, [schema])

        assert validated.tool_name == "classify"

    def test_fuzzy_match_tool_name_edit_distance(self):
        """Test fuzzy matching with small edit distance (ditect → detect)."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("ditect", {})  # 1 character substitution
        validated = validate_tool_call(call, [schema])

        assert validated.tool_name == "detect"

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy matching is case-insensitive."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("DETECT", {})
        validated = validate_tool_call(call, [schema])

        assert validated.tool_name == "detect"

    def test_unknown_tool_name_raises_error(self):
        """Test that unknown tool name raises ValueError."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("unknown_tool", {})

        with pytest.raises(ValueError) as exc_info:
            validate_tool_call(call, [schema])

        assert "unknown_tool" in str(exc_info.value)
        assert "detect" in str(exc_info.value)  # Lists available tools

    def test_type_coercion_string_to_float(self):
        """Test type coercion from string to float."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("threshold", "float", "Threshold", required=False, default=0.3)],
        )

        call = ToolCall("detect", {"threshold": "0.7"})  # String instead of float
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["threshold"] == 0.7
        assert isinstance(validated.arguments["threshold"], float)

    def test_type_coercion_string_to_int(self):
        """Test type coercion from string to int."""
        schema = ToolSchema(
            "classify",
            "Run classification",
            "classify",
            [ToolParameter("top_k", "int", "Top K", required=False, default=5)],
        )

        call = ToolCall("classify", {"top_k": "10"})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["top_k"] == 10
        assert isinstance(validated.arguments["top_k"], int)

    def test_type_coercion_string_to_bool(self):
        """Test type coercion from string to bool."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("use_nms", "bool", "Use NMS", required=False, default=True)],
        )

        call = ToolCall("detect", {"use_nms": "false"})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["use_nms"] is False

    def test_type_coercion_string_to_list(self):
        """Test type coercion from string to list[str]."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("text_prompts", "list[str]", "Classes", required=False, default=None)],
        )

        call = ToolCall("detect", {"text_prompts": "cat, dog, bird"})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["text_prompts"] == ["cat", "dog", "bird"]

    def test_type_coercion_json_string_to_list(self):
        """Test type coercion from JSON string to list."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("text_prompts", "list[str]", "Classes", required=False)],
        )

        call = ToolCall("detect", {"text_prompts": '["cat", "dog", "bird"]'})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["text_prompts"] == ["cat", "dog", "bird"]

    def test_type_coercion_string_to_bbox(self):
        """Test type coercion from string to bbox (list of 4 floats)."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("region", "bbox", "Region", required=False, default=None)],
        )

        call = ToolCall("detect", {"region": "[100, 50, 300, 250]"})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["region"] == [100.0, 50.0, 300.0, 250.0]

    def test_fill_default_values(self):
        """Test that missing optional parameters are filled with defaults."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [
                ToolParameter("threshold", "float", "Threshold", required=False, default=0.3),
                ToolParameter("region", "bbox", "Region", required=False, default=None),
            ],
        )

        call = ToolCall("detect", {})  # No arguments provided
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["threshold"] == 0.3
        assert validated.arguments["region"] is None

    def test_unknown_parameter_kept_as_is(self):
        """Test that unknown parameters are kept as-is (not dropped)."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("detect", {"unknown_param": "value"})
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["unknown_param"] == "value"

    def test_validation_creates_new_instance(self):
        """Test that validation returns a new ToolCall instance (immutable)."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        original_call = ToolCall("detect", {"threshold": 0.5})
        validated = validate_tool_call(original_call, [schema])

        # Should be a new instance
        assert validated is not original_call
        # Original unchanged
        assert original_call.tool_name == "detect"

    def test_raw_text_preserved(self):
        """Test that raw_text field is preserved during validation."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        call = ToolCall("detect", {}, raw_text='{"tool": "detect", "arguments": {}}')
        validated = validate_tool_call(call, [schema])

        assert validated.raw_text == '{"tool": "detect", "arguments": {}}'

    def test_multiple_schemas_fuzzy_match(self):
        """Test fuzzy matching across multiple available schemas."""
        schemas = [
            ToolSchema("detect", "Detection", "detect", []),
            ToolSchema("classify", "Classification", "classify", []),
            ToolSchema("segment", "Segmentation", "segment", []),
        ]

        call = ToolCall("classification", {})
        validated = validate_tool_call(call, schemas)

        assert validated.tool_name == "classify"

    def test_type_coercion_already_correct_type(self):
        """Test that values with correct types are not re-coerced."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("threshold", "float", "Threshold", required=False)],
        )

        call = ToolCall("detect", {"threshold": 0.5})  # Already float
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["threshold"] == 0.5
        assert isinstance(validated.arguments["threshold"], float)

    def test_type_coercion_int_to_float(self):
        """Test that int is coerced to float when expected type is float."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("threshold", "float", "Threshold", required=False)],
        )

        call = ToolCall("detect", {"threshold": 1})  # Int instead of float
        validated = validate_tool_call(call, [schema])

        assert validated.arguments["threshold"] == 1.0
        assert isinstance(validated.arguments["threshold"], float)

    def test_type_coercion_failure_keeps_original(self):
        """Test that failed type coercion keeps original value."""
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("threshold", "float", "Threshold", required=False)],
        )

        call = ToolCall("detect", {"threshold": "not_a_number"})
        validated = validate_tool_call(call, [schema])

        # Should keep original value if coercion fails
        assert validated.arguments["threshold"] == "not_a_number"


class TestLevenshteinDistance:
    """Tests for _levenshtein_distance helper (implicitly tested via fuzzy matching)."""

    def test_fuzzy_match_single_substitution(self):
        """Test fuzzy matching with 1 character substitution."""
        from mata.core.parsers import _levenshtein_distance

        assert _levenshtein_distance("detect", "ditect") == 1

    def test_fuzzy_match_single_insertion(self):
        """Test fuzzy matching with 1 character insertion."""
        from mata.core.parsers import _levenshtein_distance

        assert _levenshtein_distance("detect", "deetect") == 1

    def test_fuzzy_match_single_deletion(self):
        """Test fuzzy matching with 1 character deletion."""
        from mata.core.parsers import _levenshtein_distance

        assert _levenshtein_distance("detect", "detec") == 1

    def test_fuzzy_match_identical_strings(self):
        """Test fuzzy matching with identical strings."""
        from mata.core.parsers import _levenshtein_distance

        assert _levenshtein_distance("detect", "detect") == 0

    def test_fuzzy_match_too_distant(self):
        """Test that tools with edit distance > 2 don't match."""
        schema = ToolSchema("detect", "Run detection", "detect", [])

        # "xyz" has edit distance 6 from "detect"
        call = ToolCall("xyz", {})

        with pytest.raises(ValueError):
            validate_tool_call(call, [schema])


class TestCoerceArgumentType:
    """Tests for _coerce_argument_type helper."""

    def test_bool_coercion_true_variations(self):
        """Test bool coercion for various 'true' representations."""
        from mata.core.parsers import _coerce_argument_type

        assert _coerce_argument_type("true", "bool") is True
        assert _coerce_argument_type("True", "bool") is True
        assert _coerce_argument_type("TRUE", "bool") is True
        assert _coerce_argument_type("1", "bool") is True
        assert _coerce_argument_type("yes", "bool") is True

    def test_bool_coercion_false_variations(self):
        """Test bool coercion for various 'false' representations."""
        from mata.core.parsers import _coerce_argument_type

        assert _coerce_argument_type("false", "bool") is False
        assert _coerce_argument_type("False", "bool") is False
        assert _coerce_argument_type("FALSE", "bool") is False
        assert _coerce_argument_type("0", "bool") is False
        assert _coerce_argument_type("no", "bool") is False

    def test_list_coercion_comma_separated(self):
        """Test list coercion from comma-separated string."""
        from mata.core.parsers import _coerce_argument_type

        result = _coerce_argument_type("cat, dog, bird", "list[str]")
        assert result == ["cat", "dog", "bird"]

    def test_list_coercion_with_extra_spaces(self):
        """Test list coercion handles extra whitespace."""
        from mata.core.parsers import _coerce_argument_type

        result = _coerce_argument_type("  cat  ,  dog  ,  bird  ", "list[str]")
        assert result == ["cat", "dog", "bird"]

    def test_bbox_coercion_from_json_array(self):
        """Test bbox coercion from JSON array string."""
        from mata.core.parsers import _coerce_argument_type

        result = _coerce_argument_type("[100, 50, 300, 250]", "bbox")
        assert result == [100.0, 50.0, 300.0, 250.0]

    def test_bbox_coercion_truncates_extra_values(self):
        """Test bbox coercion truncates arrays with > 4 values."""
        from mata.core.parsers import _coerce_argument_type

        result = _coerce_argument_type("[100, 50, 300, 250, 999, 888]", "bbox")
        assert result == [100.0, 50.0, 300.0, 250.0]  # Only first 4


class TestIntegration:
    """Integration tests combining parsing and validation."""

    def test_full_workflow_parse_and_validate(self):
        """Test complete workflow: parse → validate → execute."""
        # VLM output with fuzzy tool name and string arguments
        vlm_output = '```tool_call\n{"tool": "detection", "arguments": {"threshold": "0.7"}}\n```'

        # Parse
        calls = parse_tool_calls(vlm_output)
        assert calls is not None
        assert len(calls) == 1

        # Validate
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [ToolParameter("threshold", "float", "Threshold", required=False, default=0.3)],
        )
        validated = validate_tool_call(calls[0], [schema])

        # Check results
        assert validated.tool_name == "detect"  # Fuzzy matched
        assert validated.arguments["threshold"] == 0.7  # Type coerced
        assert isinstance(validated.arguments["threshold"], float)

    def test_workflow_with_defaults_filled(self):
        """Test workflow where defaults are filled for missing params."""
        vlm_output = '<tool_call>{"tool": "classify", "arguments": {}}</tool_call>'

        calls = parse_tool_calls(vlm_output)
        schema = ToolSchema(
            "classify",
            "Run classification",
            "classify",
            [ToolParameter("top_k", "int", "Top K", required=False, default=5)],
        )
        validated = validate_tool_call(calls[0], [schema])

        assert validated.arguments["top_k"] == 5  # Default filled

    def test_workflow_final_answer_no_validation(self):
        """Test that final answer (no tool call) doesn't require validation."""
        vlm_output = "I found 3 cats in the image."

        calls = parse_tool_calls(vlm_output)
        assert calls is None  # Final answer mode, no validation needed

    def test_workflow_with_region_parameter(self):
        """Test workflow with bbox region parameter."""
        vlm_output = '{"tool": "detect", "arguments": {"region": "[100, 50, 300, 250]", "threshold": "0.6"}}'

        calls = parse_tool_calls(vlm_output)
        schema = ToolSchema(
            "detect",
            "Run detection",
            "detect",
            [
                ToolParameter("region", "bbox", "Region", required=False, default=None),
                ToolParameter("threshold", "float", "Threshold", required=False, default=0.3),
            ],
        )
        validated = validate_tool_call(calls[0], [schema])

        assert validated.arguments["region"] == [100.0, 50.0, 300.0, 250.0]
        assert validated.arguments["threshold"] == 0.6
