"""Unit tests for tool schema definitions (Task A1).

Tests cover:
- ToolParameter construction, defaults, immutability
- ToolSchema construction, serialization methods
- ToolCall dataclass
- ToolResult dataclass and conversation formatting
- schema_for_task() factory function
- to_prompt_str() and to_openai_schema() output formats

Version: 1.7.0
"""

from __future__ import annotations

import pytest

from mata.core.tool_schema import (
    TASK_SCHEMA_DEFAULTS,
    ToolCall,
    ToolParameter,
    ToolResult,
    ToolSchema,
    schema_for_task,
)

# ============================================================================
# ToolParameter Tests
# ============================================================================


def test_tool_parameter_construction():
    """Test basic ToolParameter construction."""
    param = ToolParameter(
        name="threshold",
        type="float",
        description="Confidence threshold",
        required=True,
        default=None,
    )

    assert param.name == "threshold"
    assert param.type == "float"
    assert param.description == "Confidence threshold"
    assert param.required is True
    assert param.default is None


def test_tool_parameter_defaults():
    """Test ToolParameter with default values."""
    param = ToolParameter(
        name="top_k",
        type="int",
        description="Number of results",
        required=False,
        default=5,
    )

    assert param.required is False
    assert param.default == 5


def test_tool_parameter_immutability():
    """Test that ToolParameter is frozen (immutable)."""
    param = ToolParameter("name", "str", "A parameter")

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        param.name = "new_name"  # type: ignore


def test_tool_parameter_to_dict():
    """Test ToolParameter.to_dict() serialization."""
    param = ToolParameter("threshold", "float", "Min confidence", False, 0.3)
    d = param.to_dict()

    assert d["name"] == "threshold"
    assert d["type"] == "float"
    assert d["description"] == "Min confidence"
    assert d["required"] is False
    assert d["default"] == 0.3


# ============================================================================
# ToolSchema Tests
# ============================================================================


def test_tool_schema_construction():
    """Test basic ToolSchema construction."""
    schema = ToolSchema(
        name="detect",
        description="Run object detection",
        task="detect",
        parameters=[],
        builtin=False,
    )

    assert schema.name == "detect"
    assert schema.description == "Run object detection"
    assert schema.task == "detect"
    assert schema.parameters == []
    assert schema.builtin is False


def test_tool_schema_with_parameters():
    """Test ToolSchema with parameters."""
    params = [
        ToolParameter("threshold", "float", "Min confidence", False, 0.3),
        ToolParameter("region", "bbox", "Crop region", False, None),
    ]

    schema = ToolSchema(
        name="detect",
        description="Run detection",
        task="detect",
        parameters=params,
    )

    assert len(schema.parameters) == 2
    assert schema.parameters[0].name == "threshold"
    assert schema.parameters[1].name == "region"


def test_tool_schema_immutability():
    """Test that ToolSchema is frozen (immutable)."""
    schema = ToolSchema("detect", "Run detection", "detect")

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        schema.name = "classify"  # type: ignore


def test_tool_schema_to_prompt_str_no_params():
    """Test to_prompt_str() with no parameters."""
    schema = ToolSchema(
        name="zoom",
        description="Zoom into a region",
        task="image",
        builtin=True,
    )

    prompt = schema.to_prompt_str()

    assert "Tool: zoom" in prompt
    assert "Description: Zoom into a region" in prompt
    assert "Parameters: None" in prompt


def test_tool_schema_to_prompt_str_with_params():
    """Test to_prompt_str() with parameters."""
    params = [
        ToolParameter("threshold", "float", "Min confidence", False, 0.3),
        ToolParameter("region", "bbox", "Crop region", True, None),
    ]

    schema = ToolSchema(
        name="detect",
        description="Run object detection",
        task="detect",
        parameters=params,
    )

    prompt = schema.to_prompt_str()

    assert "Tool: detect" in prompt
    assert "Description: Run object detection" in prompt
    assert "Parameters:" in prompt
    assert "threshold (float, optional, default=0.3): Min confidence" in prompt
    assert "region (bbox, required): Crop region" in prompt


def test_tool_schema_to_prompt_str_format():
    """Test that to_prompt_str() produces parseable format."""
    schema = TASK_SCHEMA_DEFAULTS["detect"]
    prompt = schema.to_prompt_str()

    # Check structure
    lines = prompt.split("\n")
    assert lines[0].startswith("Tool:")
    assert lines[1].startswith("Description:")
    assert "Parameters:" in prompt

    # Check all parameters are listed
    assert "region" in prompt
    assert "threshold" in prompt
    assert "text_prompts" in prompt


def test_tool_schema_to_openai_schema_basic():
    """Test to_openai_schema() basic structure."""
    schema = ToolSchema(
        name="classify",
        description="Classify image",
        task="classify",
        parameters=[
            ToolParameter("top_k", "int", "Number of results", False, 5),
        ],
    )

    openai_schema = schema.to_openai_schema()

    assert openai_schema["name"] == "classify"
    assert openai_schema["description"] == "Classify image"
    assert openai_schema["parameters"]["type"] == "object"
    assert "top_k" in openai_schema["parameters"]["properties"]


def test_tool_schema_to_openai_schema_type_mapping():
    """Test to_openai_schema() type mapping."""
    params = [
        ToolParameter("threshold", "float", "Float param", False, 0.5),
        ToolParameter("count", "int", "Int param", True),
        ToolParameter("label", "str", "String param", False, "default"),
        ToolParameter("classes", "list[str]", "List param", False),
        ToolParameter("region", "bbox", "Bbox param", False),
        ToolParameter("flag", "bool", "Bool param", False, True),
    ]

    schema = ToolSchema("test", "Test schema", "detect", params)
    openai_schema = schema.to_openai_schema()

    props = openai_schema["parameters"]["properties"]

    # Check type mappings
    assert props["threshold"]["type"] == "number"
    assert props["count"]["type"] == "integer"
    assert props["label"]["type"] == "string"
    assert props["classes"]["type"] == "array"
    assert props["classes"]["items"]["type"] == "string"
    assert props["region"]["type"] == "array"
    assert props["region"]["items"]["type"] == "number"
    assert props["region"]["minItems"] == 4
    assert props["region"]["maxItems"] == 4
    assert props["flag"]["type"] == "boolean"

    # Check default values
    assert props["threshold"]["default"] == 0.5
    assert props["label"]["default"] == "default"
    assert props["flag"]["default"] is True

    # Check required array
    assert "count" in openai_schema["parameters"]["required"]
    assert "threshold" not in openai_schema["parameters"]["required"]


def test_tool_schema_to_dict():
    """Test ToolSchema.to_dict() serialization."""
    params = [ToolParameter("threshold", "float", "Min confidence", False, 0.3)]
    schema = ToolSchema("detect", "Run detection", "detect", params, False)

    d = schema.to_dict()

    assert d["name"] == "detect"
    assert d["description"] == "Run detection"
    assert d["task"] == "detect"
    assert len(d["parameters"]) == 1
    assert d["parameters"][0]["name"] == "threshold"
    assert d["builtin"] is False


# ============================================================================
# ToolCall Tests
# ============================================================================


def test_tool_call_construction():
    """Test ToolCall construction."""
    call = ToolCall(
        tool_name="detect",
        arguments={"threshold": 0.5, "region": [10, 20, 100, 200]},
        raw_text='{"tool": "detect", "arguments": {...}}',
    )

    assert call.tool_name == "detect"
    assert call.arguments["threshold"] == 0.5
    assert "region" in call.arguments
    assert call.raw_text.startswith('{"tool"')


def test_tool_call_immutability():
    """Test that ToolCall is frozen (immutable)."""
    call = ToolCall("detect", {"threshold": 0.5})

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        call.tool_name = "classify"  # type: ignore


def test_tool_call_to_dict():
    """Test ToolCall.to_dict() serialization."""
    call = ToolCall("detect", {"threshold": 0.5}, "raw text")
    d = call.to_dict()

    assert d["tool_name"] == "detect"
    assert d["arguments"]["threshold"] == 0.5
    assert d["raw_text"] == "raw text"


# ============================================================================
# ToolResult Tests
# ============================================================================


def test_tool_result_construction():
    """Test ToolResult construction."""
    result = ToolResult(
        tool_name="detect",
        success=True,
        summary="Found 3 objects: person, car, dog",
        artifacts={"instances": []},
    )

    assert result.tool_name == "detect"
    assert result.success is True
    assert "Found 3 objects" in result.summary
    assert "instances" in result.artifacts


def test_tool_result_failure():
    """Test ToolResult for failed execution."""
    result = ToolResult(
        tool_name="classify",
        success=False,
        summary="Classification failed: model not loaded",
        artifacts={},
    )

    assert result.success is False
    assert "failed" in result.summary


def test_tool_result_immutability():
    """Test that ToolResult is frozen (immutable)."""
    result = ToolResult("detect", True, "Success")

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        result.success = False  # type: ignore


def test_tool_result_to_conversation_message_success():
    """Test to_conversation_message() for successful execution."""
    result = ToolResult("detect", True, "Found 2 cats and 1 dog")
    message = result.to_conversation_message()

    assert "[Tool: detect]" in message
    assert "✓" in message
    assert "Found 2 cats and 1 dog" in message


def test_tool_result_to_conversation_message_failure():
    """Test to_conversation_message() for failed execution."""
    result = ToolResult("classify", False, "Model not found")
    message = result.to_conversation_message()

    assert "[Tool: classify]" in message
    assert "✗" in message
    assert "Model not found" in message


def test_tool_result_to_dict():
    """Test ToolResult.to_dict() serialization."""
    result = ToolResult("detect", True, "Found 3 objects", {"count": 3})
    d = result.to_dict()

    assert d["tool_name"] == "detect"
    assert d["success"] is True
    assert d["summary"] == "Found 3 objects"
    assert d["artifacts"]["count"] == 3


# ============================================================================
# schema_for_task() Factory Tests
# ============================================================================


def test_schema_for_task_detect():
    """Test schema_for_task() for detect task."""
    schema = schema_for_task("detect")

    assert schema.name == "detect"
    assert schema.task == "detect"
    assert schema.builtin is False
    assert len(schema.parameters) > 0

    # Check expected parameters
    param_names = [p.name for p in schema.parameters]
    assert "region" in param_names
    assert "threshold" in param_names
    assert "text_prompts" in param_names


def test_schema_for_task_classify():
    """Test schema_for_task() for classify task."""
    schema = schema_for_task("classify")

    assert schema.name == "classify"
    assert schema.task == "classify"

    param_names = [p.name for p in schema.parameters]
    assert "region" in param_names
    assert "text_prompts" in param_names
    assert "top_k" in param_names


def test_schema_for_task_segment():
    """Test schema_for_task() for segment task."""
    schema = schema_for_task("segment")

    assert schema.name == "segment"
    assert schema.task == "segment"

    param_names = [p.name for p in schema.parameters]
    assert "region" in param_names
    assert "text_prompts" in param_names
    assert "threshold" in param_names


def test_schema_for_task_depth():
    """Test schema_for_task() for depth task."""
    schema = schema_for_task("depth")

    assert schema.name == "depth"
    assert schema.task == "depth"

    param_names = [p.name for p in schema.parameters]
    assert "region" in param_names


def test_schema_for_task_unknown():
    """Test schema_for_task() raises for unknown task."""
    with pytest.raises(ValueError, match="Unknown task type"):
        schema_for_task("nonexistent")


def test_schema_for_task_error_message():
    """Test schema_for_task() error message lists available tasks."""
    with pytest.raises(ValueError) as exc_info:
        schema_for_task("invalid")

    error_msg = str(exc_info.value)
    assert "detect" in error_msg
    assert "classify" in error_msg
    assert "segment" in error_msg
    assert "depth" in error_msg


# ============================================================================
# Default Schemas Tests
# ============================================================================


def test_task_schema_defaults_complete():
    """Test that TASK_SCHEMA_DEFAULTS contains all 4 core tasks."""
    assert "detect" in TASK_SCHEMA_DEFAULTS
    assert "classify" in TASK_SCHEMA_DEFAULTS
    assert "segment" in TASK_SCHEMA_DEFAULTS
    assert "depth" in TASK_SCHEMA_DEFAULTS


def test_task_schema_defaults_all_have_region():
    """Test that all task schemas include the 'region' parameter."""
    for task_name, schema in TASK_SCHEMA_DEFAULTS.items():
        param_names = [p.name for p in schema.parameters]
        assert "region" in param_names, f"Task '{task_name}' missing 'region' parameter"

        # Check region parameter details
        region_param = next(p for p in schema.parameters if p.name == "region")
        assert region_param.type == "bbox"
        assert region_param.required is False
        assert region_param.default is None


def test_task_schema_defaults_descriptions():
    """Test that all default schemas have non-empty descriptions."""
    for task_name, schema in TASK_SCHEMA_DEFAULTS.items():
        assert len(schema.description) > 20, f"Schema '{task_name}' has too short description"
        assert schema.task == task_name


# ============================================================================
# Round-trip Serialization Tests
# ============================================================================


def test_tool_schema_roundtrip():
    """Test that ToolSchema can be serialized and reconstructed."""
    original = ToolSchema(
        name="test",
        description="Test tool",
        task="detect",
        parameters=[
            ToolParameter("param1", "float", "A float param", False, 0.5),
            ToolParameter("param2", "int", "An int param", True),
        ],
        builtin=False,
    )

    # Serialize to dict
    d = original.to_dict()

    # Reconstruct (manual reconstruction for test purposes)
    params = [ToolParameter(**p) for p in d["parameters"]]
    reconstructed = ToolSchema(
        name=d["name"],
        description=d["description"],
        task=d["task"],
        parameters=params,
        builtin=d["builtin"],
    )

    assert reconstructed.name == original.name
    assert reconstructed.description == original.description
    assert reconstructed.task == original.task
    assert len(reconstructed.parameters) == len(original.parameters)
    assert reconstructed.builtin == original.builtin


def test_openai_schema_validity():
    """Test that to_openai_schema() produces valid OpenAI function schema."""
    schema = TASK_SCHEMA_DEFAULTS["detect"]
    openai_schema = schema.to_openai_schema()

    # Check required top-level keys
    assert "name" in openai_schema
    assert "description" in openai_schema
    assert "parameters" in openai_schema

    # Check parameters structure
    params = openai_schema["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params
    assert isinstance(params["properties"], dict)
    assert isinstance(params["required"], list)

    # Check that all properties are properly formatted
    for prop_name, prop_schema in params["properties"].items():
        assert "type" in prop_schema
        assert "description" in prop_schema
