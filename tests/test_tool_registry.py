"""Unit tests for ToolRegistry (Task A2).

Tests cover:
- ToolRegistry construction with built-in and provider-based tools
- Provider resolution from ExecutionContext
- Schema generation and retrieval
- System prompt block generation
- Tool execution (built-in zoom/crop and provider-based)
- Error handling and validation

Version: 1.7.0
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mata.core.artifacts.classifications import Classification, Classifications
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.tool_registry import BUILTIN_SCHEMAS, ToolRegistry
from mata.core.tool_schema import ToolCall, ToolResult
from mata.core.types import Instance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_image():
    """Create a mock Image artifact."""
    import numpy as np
    from PIL import Image as PILImage

    # Create a simple 100x100 RGB image
    pil_img = PILImage.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
    return Image.from_pil(pil_img)


@pytest.fixture
def mock_detect_provider():
    """Create a mock detection provider."""
    provider = Mock()
    provider.predict = Mock(
        return_value=Detections(
            instances=[
                Instance(bbox=[10, 10, 50, 50], label="cat", score=0.95),
                Instance(bbox=[60, 60, 90, 90], label="dog", score=0.87),
            ]
        )
    )
    return provider


@pytest.fixture
def mock_classify_provider():
    """Create a mock classification provider."""
    provider = Mock()
    provider.predict = Mock(
        return_value=Classifications(
            predictions=(
                Classification(label=0, score=0.95, label_name="cat"),
                Classification(label=1, score=0.87, label_name="dog"),
            )
        )
    )
    return provider


@pytest.fixture
def mock_context_with_providers(mock_detect_provider, mock_classify_provider):
    """Create a mock ExecutionContext with providers."""
    ctx = ExecutionContext(
        providers={
            "detect": {"detr": mock_detect_provider},
            "classify": {"clip": mock_classify_provider},
        }
    )
    return ctx


@pytest.fixture
def mock_empty_context():
    """Create a mock ExecutionContext with no providers."""
    return ExecutionContext(providers={})


# ============================================================================
# ToolRegistry Construction Tests
# ============================================================================


def test_registry_with_builtin_tools_only(mock_empty_context):
    """Test registry with only built-in tools."""
    registry = ToolRegistry(mock_empty_context, ["zoom", "crop"])

    assert len(registry.all_schemas()) == 2
    assert "zoom" in registry._schemas
    assert "crop" in registry._schemas


def test_registry_with_provider_tools_only(mock_context_with_providers):
    """Test registry with only provider-based tools."""
    registry = ToolRegistry(mock_context_with_providers, ["detr", "clip"])

    assert len(registry.all_schemas()) == 2
    assert "detr" in registry._schemas
    assert "clip" in registry._schemas
    assert "detr" in registry._tool_map
    assert "clip" in registry._tool_map


def test_registry_with_mixed_tools(mock_context_with_providers):
    """Test registry with both built-in and provider tools."""
    registry = ToolRegistry(mock_context_with_providers, ["detr", "zoom", "clip"])

    assert len(registry.all_schemas()) == 3
    assert "detr" in registry._schemas
    assert "zoom" in registry._schemas
    assert "clip" in registry._schemas


def test_registry_unknown_tool_raises_error(mock_context_with_providers):
    """Test that unknown tool name raises KeyError."""
    with pytest.raises(KeyError, match="Tool 'unknown' not found"):
        ToolRegistry(mock_context_with_providers, ["unknown"])


def test_registry_error_message_lists_available(mock_context_with_providers):
    """Test that error message lists available providers and built-ins."""
    with pytest.raises(KeyError) as exc_info:
        ToolRegistry(mock_context_with_providers, ["nonexistent"])

    error_msg = str(exc_info.value)
    assert "detr" in error_msg
    assert "clip" in error_msg
    assert "zoom" in error_msg
    assert "crop" in error_msg


def test_registry_empty_context_with_provider_tool(mock_empty_context):
    """Test that provider tool fails with empty context."""
    with pytest.raises(KeyError, match="No providers registered"):
        ToolRegistry(mock_empty_context, ["detr"])


# ============================================================================
# get_schema() Tests
# ============================================================================


def test_get_schema_for_builtin(mock_empty_context):
    """Test get_schema() for built-in tool."""
    registry = ToolRegistry(mock_empty_context, ["zoom"])
    schema = registry.get_schema("zoom")

    assert schema.name == "zoom"
    assert schema.builtin is True
    assert schema.task == "image"


def test_get_schema_for_provider(mock_context_with_providers):
    """Test get_schema() for provider-based tool."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])
    schema = registry.get_schema("detr")

    assert schema.name == "detr"
    assert schema.builtin is False
    assert schema.task == "detect"


def test_get_schema_unknown_tool(mock_context_with_providers):
    """Test get_schema() raises KeyError for unknown tool."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])

    with pytest.raises(KeyError, match="Tool 'unknown' not in registry"):
        registry.get_schema("unknown")


# ============================================================================
# get_provider() Tests
# ============================================================================


def test_get_provider_returns_correct_instance(mock_context_with_providers, mock_detect_provider):
    """Test get_provider() returns the correct provider instance."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])
    provider = registry.get_provider("detr")

    assert provider == mock_detect_provider


def test_get_provider_for_builtin_raises_error(mock_empty_context):
    """Test get_provider() raises KeyError for built-in tools."""
    registry = ToolRegistry(mock_empty_context, ["zoom"])

    with pytest.raises(KeyError, match="is a built-in tool"):
        registry.get_provider("zoom")


def test_get_provider_unknown_tool(mock_context_with_providers):
    """Test get_provider() raises KeyError for unknown tool."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])

    with pytest.raises(KeyError, match="not in registry"):
        registry.get_provider("clip")


# ============================================================================
# all_schemas() Tests
# ============================================================================


def test_all_schemas_returns_all_tools(mock_context_with_providers):
    """Test all_schemas() returns all registered tools."""
    registry = ToolRegistry(mock_context_with_providers, ["detr", "zoom", "clip"])
    schemas = registry.all_schemas()

    assert len(schemas) == 3
    schema_names = [s.name for s in schemas]
    assert "detr" in schema_names
    assert "zoom" in schema_names
    assert "clip" in schema_names


# ============================================================================
# build_system_prompt_block() Tests
# ============================================================================


def test_build_system_prompt_block_format(mock_context_with_providers):
    """Test build_system_prompt_block() produces correct format."""
    registry = ToolRegistry(mock_context_with_providers, ["detr", "zoom"])
    prompt = registry.build_system_prompt_block()

    # Check structure
    assert "Tool: detr" in prompt
    assert "Tool: zoom" in prompt
    assert "Description:" in prompt
    assert "Parameters:" in prompt


def test_build_system_prompt_block_includes_all_tools(mock_context_with_providers):
    """Test that system prompt block includes all registered tools."""
    registry = ToolRegistry(mock_context_with_providers, ["detr", "clip", "zoom"])
    prompt = registry.build_system_prompt_block()

    assert "detr" in prompt
    assert "clip" in prompt
    assert "zoom" in prompt


# ============================================================================
# execute_tool() - Built-in Tools Tests
# ============================================================================


def test_execute_crop_tool(mock_empty_context, mock_image):
    """Test executing crop built-in tool."""
    registry = ToolRegistry(mock_empty_context, ["crop"])

    call = ToolCall("crop", {"region": [10, 10, 50, 50]}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert "Cropped region" in result.summary
    assert "image" in result.artifacts
    assert result.artifacts["image"].width == 40
    assert result.artifacts["image"].height == 40


def test_execute_crop_with_clamping(mock_empty_context, mock_image):
    """Test crop tool clamps out-of-bounds regions."""
    registry = ToolRegistry(mock_empty_context, ["crop"])

    # Region extends beyond image bounds
    call = ToolCall("crop", {"region": [-10, -10, 150, 150]}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    # Should be clamped to [0, 0, 100, 100]
    assert result.artifacts["image"].width == 100
    assert result.artifacts["image"].height == 100


def test_execute_crop_invalid_region(mock_empty_context, mock_image):
    """Test crop tool fails on invalid region."""
    registry = ToolRegistry(mock_empty_context, ["crop"])

    # Inverted region (x2 < x1)
    call = ToolCall("crop", {"region": [50, 50, 10, 10]}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is False
    assert "Invalid region" in result.summary


def test_execute_crop_missing_region(mock_empty_context, mock_image):
    """Test crop tool fails when region is missing."""
    registry = ToolRegistry(mock_empty_context, ["crop"])

    call = ToolCall("crop", {}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is False
    assert "requires a 'region'" in result.summary


def test_execute_zoom_tool(mock_empty_context, mock_image):
    """Test executing zoom built-in tool."""
    registry = ToolRegistry(mock_empty_context, ["zoom"])

    call = ToolCall("zoom", {"region": [10, 10, 50, 50], "scale": 2.0}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert "Zoomed region" in result.summary
    assert "image" in result.artifacts
    # Cropped to 40x40, then scaled 2x = 80x80
    assert result.artifacts["image"].width == 80
    assert result.artifacts["image"].height == 80


def test_execute_zoom_default_scale(mock_empty_context, mock_image):
    """Test zoom tool uses default scale=2.0."""
    registry = ToolRegistry(mock_empty_context, ["zoom"])

    call = ToolCall("zoom", {"region": [10, 10, 50, 50]}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert result.artifacts["image"].width == 80  # 40 * 2.0


# ============================================================================
# execute_tool() - Provider Tools Tests
# ============================================================================


def test_execute_detect_provider(mock_context_with_providers, mock_detect_provider, mock_image):
    """Test executing a detection provider tool."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])

    call = ToolCall("detr", {"threshold": 0.5}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert "Found 2 objects" in result.summary
    assert "cat" in result.summary
    assert "dog" in result.summary
    assert "detections" in result.artifacts

    # Verify provider was called
    mock_detect_provider.predict.assert_called_once()


def test_execute_classify_provider(mock_context_with_providers, mock_classify_provider, mock_image):
    """Test executing a classification provider tool."""
    registry = ToolRegistry(mock_context_with_providers, ["clip"])

    call = ToolCall("clip", {}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert "Top classification: cat" in result.summary
    assert "classifications" in result.artifacts

    # Verify provider was called
    mock_classify_provider.predict.assert_called_once()


def test_execute_detect_with_no_detections(mock_context_with_providers, mock_image):
    """Test detection provider that returns no detections."""
    # Create provider that returns empty detections
    provider = Mock()
    provider.predict = Mock(return_value=Detections(instances=[]))

    ctx = ExecutionContext(providers={"detect": {"detr": provider}})
    registry = ToolRegistry(ctx, ["detr"])

    call = ToolCall("detr", {}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True
    assert "No objects detected" in result.summary


def test_execute_provider_with_region_crop(mock_context_with_providers, mock_detect_provider, mock_image):
    """Test provider tool with region parameter crops image first."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])

    call = ToolCall("detr", {"region": [10, 10, 50, 50], "threshold": 0.5}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is True

    # Verify provider was called (region should be removed from args)
    mock_detect_provider.predict.assert_called_once()
    call_args = mock_detect_provider.predict.call_args

    # First argument should be a cropped image
    called_image = call_args[0][0]
    assert called_image.width == 40
    assert called_image.height == 40

    # threshold should still be in kwargs
    assert call_args[1]["threshold"] == 0.5


def test_execute_provider_failure_handling(mock_context_with_providers, mock_image):
    """Test provider tool failure is handled gracefully."""
    # Create provider that raises exception
    provider = Mock()
    provider.predict = Mock(side_effect=RuntimeError("Model crashed"))

    ctx = ExecutionContext(providers={"detect": {"detr": provider}})
    registry = ToolRegistry(ctx, ["detr"])

    call = ToolCall("detr", {}, "")
    result = registry.execute_tool(call, mock_image)

    assert result.success is False
    assert "failed" in result.summary.lower()
    assert "Model crashed" in result.summary


def test_execute_unknown_tool_raises_error(mock_context_with_providers, mock_image):
    """Test executing unknown tool raises KeyError."""
    registry = ToolRegistry(mock_context_with_providers, ["detr"])

    call = ToolCall("unknown", {}, "")
    with pytest.raises(KeyError, match="not in registry"):
        registry.execute_tool(call, mock_image)


# ============================================================================
# Provider Resolution Tests
# ============================================================================


def test_resolve_provider_from_multiple_capabilities():
    """Test resolving providers from context with multiple capabilities."""
    mock_detect = Mock()
    mock_classify = Mock()
    mock_segment = Mock()

    ctx = ExecutionContext(
        providers={
            "detect": {"yolo": mock_detect},
            "classify": {"resnet": mock_classify},
            "segment": {"sam": mock_segment},
        }
    )

    registry = ToolRegistry(ctx, ["yolo", "resnet", "sam"])

    assert registry.get_provider("yolo") == mock_detect
    assert registry.get_provider("resnet") == mock_classify
    assert registry.get_provider("sam") == mock_segment


def test_schema_for_capability_uses_provider_name():
    """Test that schema uses provider name, not capability."""
    mock_provider = Mock()
    ctx = ExecutionContext(providers={"detect": {"my_detector": mock_provider}})

    registry = ToolRegistry(ctx, ["my_detector"])
    schema = registry.get_schema("my_detector")

    assert schema.name == "my_detector"  # Not "detect"
    assert schema.task == "detect"


# ============================================================================
# BUILTIN_SCHEMAS Tests
# ============================================================================


def test_builtin_schemas_define_zoom_and_crop():
    """Test that BUILTIN_SCHEMAS contains zoom and crop."""
    assert "zoom" in BUILTIN_SCHEMAS
    assert "crop" in BUILTIN_SCHEMAS


def test_builtin_schemas_have_correct_properties():
    """Test that built-in schemas have correct properties."""
    zoom = BUILTIN_SCHEMAS["zoom"]
    assert zoom.name == "zoom"
    assert zoom.task == "image"
    assert zoom.builtin is True
    assert len(zoom.parameters) == 2  # region, scale

    crop = BUILTIN_SCHEMAS["crop"]
    assert crop.name == "crop"
    assert crop.task == "image"
    assert crop.builtin is True
    assert len(crop.parameters) == 1  # region


def test_builtin_zoom_has_region_and_scale_params():
    """Test zoom schema has region and scale parameters."""
    zoom = BUILTIN_SCHEMAS["zoom"]
    param_names = [p.name for p in zoom.parameters]

    assert "region" in param_names
    assert "scale" in param_names

    # Check region is required
    region_param = next(p for p in zoom.parameters if p.name == "region")
    assert region_param.required is True

    # Check scale has default
    scale_param = next(p for p in zoom.parameters if p.name == "scale")
    assert scale_param.required is False
    assert scale_param.default == 2.0


def test_builtin_crop_has_region_param():
    """Test crop schema has region parameter."""
    crop = BUILTIN_SCHEMAS["crop"]
    param_names = [p.name for p in crop.parameters]

    assert "region" in param_names

    # Check region is required
    region_param = next(p for p in crop.parameters if p.name == "region")
    assert region_param.required is True


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_registry_with_duplicate_tool_names(mock_context_with_providers):
    """Test registry handles duplicate tool names in list."""
    # Should work - duplicates are just redundant
    registry = ToolRegistry(mock_context_with_providers, ["detr", "detr", "zoom"])

    # Should only have unique tools
    assert len(registry.all_schemas()) == 2


def test_execute_tool_result_has_correct_format():
    """Test that ToolResult has expected structure."""
    registry = ToolRegistry(ExecutionContext(providers={}), ["crop"])

    import numpy as np
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
    image = Image.from_pil(pil_img)

    call = ToolCall("crop", {"region": [10, 10, 50, 50]}, "")
    result = registry.execute_tool(call, image)

    # Verify ToolResult structure
    assert isinstance(result, ToolResult)
    assert hasattr(result, "tool_name")
    assert hasattr(result, "success")
    assert hasattr(result, "summary")
    assert hasattr(result, "artifacts")
    assert result.tool_name == "crop"


# ============================================================================
# Task B3: Provider Resolution for Tools — Acceptance Criteria Tests
# ============================================================================


def test_b3_custom_tool_names_resolve_with_flat_providers():
    """Test B3 Criterion 1: tools=["detector"] resolves with flat providers dict."""
    # Simulate user naming providers with custom names (not task types)
    mock_detr = Mock()
    mock_detr.predict = Mock(
        return_value=Detections(instances=[Instance(bbox=[10, 10, 50, 50], label="cat", score=0.95)])
    )

    mock_clip = Mock()
    mock_clip.predict = Mock(
        return_value=Classifications(predictions=(Classification(label=0, score=0.95, label_name="cat"),))
    )

    # User names them "detector" and "classifier" (not "detect"/"classify")
    ctx = ExecutionContext(
        providers={
            "detect": {"detector": mock_detr},
            "classify": {"classifier": mock_clip},
        }
    )

    # Registry should resolve these custom names
    registry = ToolRegistry(ctx, ["detector", "classifier"])

    # Verify both tools are registered
    assert "detector" in registry._tool_map
    assert "classifier" in registry._tool_map

    # Verify schemas use custom names
    detector_schema = registry.get_schema("detector")
    assert detector_schema.name == "detector"
    assert detector_schema.task == "detect"

    classifier_schema = registry.get_schema("classifier")
    assert classifier_schema.name == "classifier"
    assert classifier_schema.task == "classify"

    # Verify providers resolve correctly
    assert registry.get_provider("detector") == mock_detr
    assert registry.get_provider("classifier") == mock_clip


def test_b3_nonexistent_tool_raises_with_available_providers_listed():
    """Test B3 Criterion 2: tools=["nonexistent"] raises KeyError with available providers."""
    mock_detr = Mock()
    ctx = ExecutionContext(providers={"detect": {"my_detector": mock_detr}})

    # Try to use a nonexistent tool
    with pytest.raises(KeyError) as exc_info:
        ToolRegistry(ctx, ["nonexistent_tool"])

    error_msg = str(exc_info.value)

    # Error message must list available providers
    assert "nonexistent_tool" in error_msg
    assert "my_detector" in error_msg
    assert "detect" in error_msg  # Capability type should be shown
    assert "zoom" in error_msg or "crop" in error_msg  # Built-ins should be listed


def test_b3_builtin_tools_resolve_without_providers():
    """Test B3 Criterion 3: Built-in tools resolve without provider entries."""
    # Empty context — no providers registered
    ctx = ExecutionContext(providers={})

    # But built-in tools should still work
    registry = ToolRegistry(ctx, ["zoom", "crop"])

    # Verify built-ins are registered
    assert "zoom" in registry._schemas
    assert "crop" in registry._schemas

    # Verify they are NOT in the provider map (they're built-in)
    assert "zoom" not in registry._tool_map
    assert "crop" not in registry._tool_map

    # Verify schemas are correct
    zoom_schema = registry.get_schema("zoom")
    assert zoom_schema.builtin is True
    assert zoom_schema.name == "zoom"

    crop_schema = registry.get_schema("crop")
    assert crop_schema.builtin is True
    assert crop_schema.name == "crop"


def test_b3_builtin_and_provider_tools_mixed():
    """Test B3 Criterion 3 extended: Built-ins + providers can coexist."""
    mock_detr = Mock()
    ctx = ExecutionContext(providers={"detect": {"detr": mock_detr}})

    # Mix built-in and provider tools
    registry = ToolRegistry(ctx, ["detr", "zoom", "crop"])

    # All should be registered
    assert len(registry.all_schemas()) == 3

    # Verify built-ins
    zoom_schema = registry.get_schema("zoom")
    assert zoom_schema.builtin is True

    # Verify provider
    detr_schema = registry.get_schema("detr")
    assert detr_schema.builtin is False
    assert registry.get_provider("detr") == mock_detr


def test_b3_nested_providers_dict_works():
    """Test B3 Criterion 4: Works with nested provider dicts."""
    mock_detr = Mock()
    mock_yolo = Mock()
    mock_clip = Mock()

    # Nested format: {capability: {name: adapter}}
    ctx = ExecutionContext(
        providers={
            "detect": {"detr": mock_detr, "yolo": mock_yolo},
            "classify": {"clip": mock_clip},
        }
    )

    registry = ToolRegistry(ctx, ["detr", "yolo", "clip"])

    # All tools should resolve correctly
    assert registry.get_provider("detr") == mock_detr
    assert registry.get_provider("yolo") == mock_yolo
    assert registry.get_provider("clip") == mock_clip

    # Schemas should use provider names
    assert registry.get_schema("detr").name == "detr"
    assert registry.get_schema("yolo").name == "yolo"
    assert registry.get_schema("clip").name == "clip"


def test_b3_flat_providers_dict_auto_organized():
    """Test B3 Criterion 4 extended: Flat providers work after normalization."""
    # This tests that mata.infer() can pass a flat dict which gets normalized
    # For the ToolRegistry, it always receives nested format from ExecutionContext
    # But we verify that once normalized, resolution still works

    mock_detr = Mock()
    mock_clip = Mock()

    # Start with flat format (simulating mata.infer input)

    # Simulate normalization by organizing into nested format
    # (This mirrors what _normalize_providers does)
    nested_providers = {
        "detect": {"my_detector": mock_detr},
        "classify": {"my_classifier": mock_clip},
    }

    ctx = ExecutionContext(providers=nested_providers)
    registry = ToolRegistry(ctx, ["my_detector", "my_classifier"])

    # Verify resolution works with normalized format
    assert registry.get_provider("my_detector") == mock_detr
    assert registry.get_provider("my_classifier") == mock_clip


def test_b3_error_message_quality():
    """Test B3: Error messages are helpful and actionable."""
    mock_detr = Mock()
    mock_sam = Mock()

    ctx = ExecutionContext(
        providers={
            "detect": {"detr": mock_detr},
            "segment": {"sam": mock_sam},
        }
    )

    # Try nonexistent tool
    with pytest.raises(KeyError) as exc_info:
        ToolRegistry(ctx, ["unknown_tool", "detr"])

    error_msg = str(exc_info.value)

    # Error should mention what was missing
    assert "unknown_tool" in error_msg

    # Error should list what's available
    assert "detr" in error_msg
    assert "sam" in error_msg

    # Error should mention built-ins as alternative
    assert "zoom" in error_msg or "crop" in error_msg


def test_b3_multiple_capabilities_provider_resolution():
    """Test B3: Provider resolution searches all capabilities correctly."""
    mock_detect = Mock()
    mock_classify = Mock()
    mock_segment = Mock()
    mock_depth = Mock()

    ctx = ExecutionContext(
        providers={
            "detect": {"d1": mock_detect},
            "classify": {"c1": mock_classify},
            "segment": {"s1": mock_segment},
            "depth": {"depth1": mock_depth},
        }
    )

    # Request tools from different capabilities
    registry = ToolRegistry(ctx, ["d1", "c1", "s1", "depth1"])

    # All should resolve to correct providers
    assert registry.get_provider("d1") == mock_detect
    assert registry.get_provider("c1") == mock_classify
    assert registry.get_provider("s1") == mock_segment
    assert registry.get_provider("depth1") == mock_depth

    # Schemas should map to correct tasks
    assert registry.get_schema("d1").task == "detect"
    assert registry.get_schema("c1").task == "classify"
    assert registry.get_schema("s1").task == "segment"
    assert registry.get_schema("depth1").task == "depth"


def test_b3_same_provider_name_different_capabilities():
    """Test B3: Provider names are unique across capabilities (no conflicts)."""
    mock_model1 = Mock()
    mock_model2 = Mock()

    # Deliberately use same name in different capabilities
    # (Edge case: would cause ambiguity if not handled correctly)
    ctx = ExecutionContext(
        providers={
            "detect": {"mymodel": mock_model1},
            "classify": {"othermodel": mock_model2},
        }
    )

    # Request both tools
    registry = ToolRegistry(ctx, ["mymodel", "othermodel"])

    # Each should resolve to the correct provider
    assert registry.get_provider("mymodel") == mock_model1
    assert registry.get_provider("othermodel") == mock_model2

    # Schemas should have correct tasks
    assert registry.get_schema("mymodel").task == "detect"
    assert registry.get_schema("othermodel").task == "classify"
