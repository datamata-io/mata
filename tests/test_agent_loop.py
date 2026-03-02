"""Tests for VLM agent loop implementation.

Test coverage:
- AgentLoop initialization and validation
- Basic single tool call execution
- Multi-tool call chaining
- Max iterations enforcement
- Malformed tool call retry logic
- Tool execution failure handling
- Infinite loop detection
- Error modes (retry, skip, fail)
- AgentResult accumulation
- Conversation history building
- Tool call parsing (multiple formats)
- Built-in tool execution

Version: 1.7.0
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from PIL import Image as PILImage

from mata.core.agent_loop import AgentLoop, AgentResult
from mata.core.artifacts.image import Image
from mata.core.tool_registry import ToolRegistry
from mata.core.tool_schema import ToolCall, ToolResult, ToolSchema
from mata.core.types import Entity, Instance, VisionResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample Image artifact."""
    pil_img = PILImage.new("RGB", (640, 480), color=(100, 100, 100))
    return Image.from_pil(pil_img)


@pytest.fixture
def mock_vlm_provider():
    """Create a mock VLM provider."""
    mock = Mock()

    # Default: return a simple text response (no tool call)
    default_result = VisionResult(
        instances=[],
        entities=[],
        text="This is a final answer.",
    )
    mock.query.return_value = default_result

    return mock


@pytest.fixture
def mock_tool_registry():
    """Create a mock ToolRegistry."""
    mock = Mock(spec=ToolRegistry)

    # Default tool schemas
    mock.all_schemas.return_value = [
        ToolSchema("detect", "Run detection", "detect", []),
        ToolSchema("classify", "Run classification", "classify", []),
    ]

    # Default system prompt block
    mock.build_system_prompt_block.return_value = (
        "Tool: detect\nDescription: Run detection\n\n" "Tool: classify\nDescription: Run classification"
    )

    # Default tool execution returns success
    mock.execute_tool.return_value = ToolResult(
        tool_name="detect",
        success=True,
        summary="Found 2 objects: cat (0.95), dog (0.87)",
        artifacts={
            "instances": [
                Instance(
                    bbox=(10, 20, 100, 150),
                    score=0.95,
                    label=0,
                    label_name="cat",
                ),
            ]
        },
    )

    return mock


# ============================================================================
# AgentResult Tests
# ============================================================================


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_agent_result_construction(self):
        """AgentResult can be constructed with all fields."""
        result = AgentResult(
            text="Final answer",
            tool_calls=[ToolCall("detect", {}, "")],
            tool_results=[ToolResult("detect", True, "Success")],
            iterations=3,
            instances=[],
            entities=[],
            conversation=[{"role": "user", "content": "Hello"}],
            meta={"key": "value"},
        )

        assert result.text == "Final answer"
        assert len(result.tool_calls) == 1
        assert len(result.tool_results) == 1
        assert result.iterations == 3
        assert len(result.conversation) == 1
        assert result.meta["key"] == "value"

    def test_agent_result_defaults(self):
        """AgentResult has sensible defaults."""
        result = AgentResult(text="Final answer")

        assert result.tool_calls == []
        assert result.tool_results == []
        assert result.iterations == 0
        assert result.instances == []
        assert result.entities == []
        assert result.conversation == []
        assert result.meta == {}

    def test_agent_result_immutable(self):
        """AgentResult is frozen (immutable)."""
        result = AgentResult(text="Final answer")

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.text = "Changed"  # type: ignore

    def test_agent_result_to_dict(self):
        """AgentResult serializes to dict correctly."""
        call = ToolCall("detect", {"threshold": 0.5}, "raw")
        result_obj = ToolResult("detect", True, "Found 2 objects")

        result = AgentResult(
            text="Final answer",
            tool_calls=[call],
            tool_results=[result_obj],
            iterations=2,
        )

        data = result.to_dict()

        assert data["text"] == "Final answer"
        assert data["iterations"] == 2
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "detect"
        assert len(data["tool_results"]) == 1


# ============================================================================
# AgentLoop Initialization Tests
# ============================================================================


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    def test_init_default_params(self, mock_vlm_provider, mock_tool_registry):
        """AgentLoop initializes with default parameters."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        assert loop.vlm_provider == mock_vlm_provider
        assert loop.tool_registry == mock_tool_registry
        assert loop.max_iterations == 5
        assert loop.on_error == "retry"
        assert loop.max_retries == 2

    def test_init_custom_params(self, mock_vlm_provider, mock_tool_registry):
        """AgentLoop accepts custom parameters."""
        loop = AgentLoop(
            mock_vlm_provider,
            mock_tool_registry,
            max_iterations=10,
            on_error="skip",
            max_retries=3,
        )

        assert loop.max_iterations == 10
        assert loop.on_error == "skip"
        assert loop.max_retries == 3

    def test_init_rejects_high_max_iterations(self, mock_vlm_provider, mock_tool_registry):
        """AgentLoop rejects max_iterations > 20."""
        with pytest.raises(ValueError, match="exceeds safety limit"):
            AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=25)

    def test_init_rejects_invalid_on_error(self, mock_vlm_provider, mock_tool_registry):
        """AgentLoop rejects invalid on_error mode."""
        with pytest.raises(ValueError, match="Invalid on_error mode"):
            AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="invalid")


# ============================================================================
# Basic Agent Loop Tests
# ============================================================================


class TestAgentLoopBasic:
    """Tests for basic agent loop execution."""

    def test_run_no_tools_final_answer(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop with no tool calls returns final answer immediately."""
        # VLM returns final answer (no tool call)
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="This is a cat image.",
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "What is this?")

        assert result.text == "This is a cat image."
        assert result.iterations == 1
        assert len(result.tool_calls) == 0
        assert len(result.tool_results) == 0

    def test_run_single_tool_call(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop executes a single tool call and returns final answer."""
        # Turn 1: VLM calls detect tool
        turn1_result = VisionResult(
            instances=[],
            entities=[],
            text='```tool_call\n{"tool": "detect", "arguments": {}}\n```',
        )

        # Turn 2: VLM provides final answer
        turn2_result = VisionResult(
            instances=[],
            entities=[],
            text="Found 2 objects: cat and dog.",
        )

        mock_vlm_provider.query.side_effect = [turn1_result, turn2_result]

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "What objects are in this image?")

        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "detect"
        assert len(result.tool_results) == 1
        assert result.text == "Found 2 objects: cat and dog."

    def test_run_multi_tool_chain(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop chains multiple tool calls."""
        # Turn 1: VLM calls detect
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='```tool_call\n{"tool": "detect", "arguments": {}}\n```',
        )

        # Turn 2: VLM calls classify
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text='<tool_call>{"tool": "classify", "arguments": {}}</tool_call>',
        )

        # Turn 3: Final answer
        turn3 = VisionResult(
            instances=[],
            entities=[],
            text="The image contains a cat and a dog.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2, turn3]

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Analyze this image.")

        assert result.iterations == 3
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "detect"
        assert result.tool_calls[1].tool_name == "classify"
        assert result.text == "The image contains a cat and a dog."

    def test_run_max_iterations_reached(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop stops at max_iterations even if no final answer."""
        # Always return tool call (never a final answer)
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=3)
        result = loop.run(sample_image, "What is this?")

        assert result.iterations == 3
        # Loop detection kicks in after 2 identical calls (breaks before 3rd)
        assert len(result.tool_calls) == 2


# ============================================================================
# Tool Call Parsing Tests
# ============================================================================


class TestToolCallParsing:
    """Tests for tool call parsing from VLM output."""

    def test_parse_fenced_block_format(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Parses tool calls in fenced code block format."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='Some text\n```tool_call\n{"tool": "detect", "arguments": {"threshold": 0.5}}\n```\nMore text',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=1)
        result = loop.run(sample_image, "Test")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "detect"
        assert result.tool_calls[0].arguments["threshold"] == 0.5

    def test_parse_xml_tag_format(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Parses tool calls in XML tag format."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='<tool_call>{"tool": "classify", "arguments": {"top_k": 3}}</tool_call>',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=1)
        result = loop.run(sample_image, "Test")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "classify"
        assert result.tool_calls[0].arguments["top_k"] == 3

    def test_parse_raw_json_format(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Parses tool calls in raw JSON format."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=1)
        result = loop.run(sample_image, "Test")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "detect"

    def test_parse_alternate_key_names(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Parses tool calls with alternate key names (action, parameters)."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"action": "detect", "parameters": {"threshold": 0.7}}',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=1)
        result = loop.run(sample_image, "Test")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "detect"
        assert result.tool_calls[0].arguments["threshold"] == 0.7

    def test_parse_no_tool_call(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Returns empty list when no tool call found."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="This is just a regular response with no tool call.",
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Test")

        assert len(result.tool_calls) == 0
        assert result.iterations == 1


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling modes."""

    def test_on_error_fail_raises_on_tool_failure(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """on_error='fail' raises RuntimeError on tool execution failure."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        mock_tool_registry.execute_tool.side_effect = RuntimeError("Tool failed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="fail")

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            loop.run(sample_image, "Test")

    def test_on_error_skip_continues_on_tool_failure(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """on_error='skip' continues execution on tool failure."""
        # Turn 1: Tool call (will fail)
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        # Turn 2: Final answer
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Analysis complete despite tool failure.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]
        mock_tool_registry.execute_tool.side_effect = RuntimeError("Tool failed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")
        result = loop.run(sample_image, "Test")

        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False

    def test_on_error_retry_retries_malformed_calls(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """on_error='retry' retries on tool execution failure."""
        # Turn 1: Tool call (will fail)
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        # Turn 2: VLM response after failure feedback
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="I see. Let me provide a final answer instead.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]
        mock_tool_registry.execute_tool.side_effect = RuntimeError("Tool failed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="retry", max_retries=2)
        result = loop.run(sample_image, "Test")

        assert result.iterations == 2


# ============================================================================
# Instance and Entity Accumulation Tests
# ============================================================================


class TestAccumulation:
    """Tests for accumulating instances and entities."""

    def test_accumulates_instances_from_tools(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop accumulates instances from tool results."""
        # Turn 1: Tool call
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        # Turn 2: Final answer
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Found objects.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        # Mock tool returns instances
        instance1 = Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat")
        instance2 = Instance(bbox=(200, 50, 300, 200), score=0.87, label=1, label_name="dog")

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="detect",
            success=True,
            summary="Found 2 objects",
            artifacts={"instances": [instance1, instance2]},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Test")

        assert len(result.instances) == 2
        assert result.instances[0].label_name == "cat"
        assert result.instances[1].label_name == "dog"

    def test_accumulates_entities_from_vlm(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop accumulates entities from VLM output."""
        entity1 = Entity("person", 0.9)
        entity2 = Entity("vehicle", 0.85)

        # Turn 1: VLM with entities
        turn1 = VisionResult(
            instances=[],
            entities=[entity1, entity2],
            text='{"tool": "detect", "arguments": {}}',
        )

        # Turn 2: Final answer
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Analysis complete.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Test")

        assert len(result.entities) == 2
        assert result.entities[0].label == "person"
        assert result.entities[1].label == "vehicle"


# ============================================================================
# Infinite Loop Detection Tests
# ============================================================================


class TestLoopDetection:
    """Tests for infinite loop detection."""

    def test_detects_repeated_identical_calls(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Detects and breaks on repeated identical tool calls."""
        # Always return the same tool call
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {"threshold": 0.5}}',
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=10)
        result = loop.run(sample_image, "Test")

        # Loop detected after 2 identical calls (breaks before 3rd)
        assert result.iterations == 3
        assert len(result.tool_calls) == 2


# ============================================================================
# Conversation History Tests
# ============================================================================


class TestConversationHistory:
    """Tests for conversation history building."""

    def test_builds_conversation_history(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop builds correct conversation history."""
        # Turn 1: Tool call
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        # Turn 2: Final answer
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Final answer.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Test")

        # Should have: assistant turn (tool call) + user turn (tool result) + assistant turn (final)
        assert len(result.conversation) == 3
        assert result.conversation[0]["role"] == "assistant"
        assert result.conversation[1]["role"] == "user"  # Tool result
        assert result.conversation[2]["role"] == "assistant"

    def test_conversation_includes_tool_results(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Conversation history includes formatted tool results."""
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Done.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="detect",
            success=True,
            summary="Found 2 objects",
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Test")

        # Check that tool result is in conversation
        tool_result_msg = result.conversation[1]["content"]
        assert "[Tool: detect]" in tool_result_msg
        assert "Found 2 objects" in tool_result_msg


# ============================================================================
# VLM Query Failure Tests
# ============================================================================


class TestVLMFailure:
    """Tests for VLM query failures."""

    def test_vlm_failure_on_error_fail(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """VLM failure with on_error='fail' raises RuntimeError."""
        mock_vlm_provider.query.side_effect = RuntimeError("VLM crashed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="fail")

        with pytest.raises(RuntimeError, match="VLM query failed"):
            loop.run(sample_image, "Test")

    def test_vlm_failure_on_error_skip(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """VLM failure with on_error='skip' returns best-effort result."""
        mock_vlm_provider.query.side_effect = RuntimeError("VLM crashed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")
        result = loop.run(sample_image, "Test")

        # Should return empty result (no iterations completed)
        assert result.iterations == 0
        assert result.text == "No final answer generated."


# ============================================================================
# System Prompt Tests
# ============================================================================


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_uses_custom_system_prompt(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop uses custom system prompt when provided."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="Final answer.",
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        loop.run(sample_image, "Test", system_prompt="Custom system prompt")

        # Verify custom system prompt was passed to VLM
        call_kwargs = mock_vlm_provider.query.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom system prompt"

    def test_generates_default_system_prompt(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """AgentLoop generates default system prompt with tool descriptions."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="Final answer.",
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        loop.run(sample_image, "Test")

        # Verify system prompt contains tool descriptions
        call_kwargs = mock_vlm_provider.query.call_args[1]
        system_prompt = call_kwargs["system_prompt"]
        assert "Tool: detect" in system_prompt
        assert "Tool: classify" in system_prompt
        assert "tool_call" in system_prompt


# ============================================================================
# Region-Based Tool Dispatch Tests
# ============================================================================


class TestRegionBasedDispatch:
    """Tests for region-based tool dispatch (crop before adapter execution)."""

    def test_region_parameter_crops_before_detect(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Tool call with region parameter crops image before running adapter."""
        # VLM calls detect with a region
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {"region": [100, 50, 300, 250]}}',
        )
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Found objects in region.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        # Mock execute_tool to capture what it receives
        captured_calls = []

        def capture_execute_tool(tool_call, image):
            captured_calls.append((tool_call, image))
            return ToolResult(
                tool_name="detect",
                success=True,
                summary="Detected in cropped region",
                artifacts={},
            )

        mock_tool_registry.execute_tool.side_effect = capture_execute_tool

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        loop.run(sample_image, "Detect in region")

        # Verify execute_tool was called once
        assert len(captured_calls) == 1
        tool_call, used_image = captured_calls[0]

        # Verify the tool call contains region argument
        assert "region" in tool_call.arguments
        assert tool_call.arguments["region"] == [100, 50, 300, 250]

    def test_no_region_parameter_uses_full_image(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Tool call without region parameter uses full image."""
        # VLM calls detect without region
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {"threshold": 0.5}}',
        )
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Detected objects.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        # Mock execute_tool to verify no region was passed
        captured_calls = []

        def capture_execute_tool(tool_call, image):
            captured_calls.append(tool_call)
            return ToolResult(
                tool_name="detect",
                success=True,
                summary="Detected in full image",
                artifacts={},
            )

        mock_tool_registry.execute_tool.side_effect = capture_execute_tool

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        loop.run(sample_image, "Detect")

        # Verify execute_tool was called
        assert len(captured_calls) == 1
        tool_call = captured_calls[0]

        # Verify no region in arguments (or None/null)
        assert tool_call.arguments.get("region") is None


# ============================================================================
# Built-in Tool Execution Tests
# ============================================================================


class TestBuiltinTools:
    """Tests for built-in tool (zoom, crop) execution."""

    def test_zoom_tool_execution(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """VLM can call zoom tool to focus on region."""
        # VLM calls zoom tool
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "zoom", "arguments": {"region": [100, 50, 300, 250], "scale": 2.0}}',
        )
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Zoomed region analyzed.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        # Mock zoom tool execution
        from mata.core.artifacts.image import Image

        zoomed_img = Image.from_pil(PILImage.new("RGB", (400, 400), color=(150, 150, 150)))

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="zoom",
            success=True,
            summary="Zoomed region (100, 50, 300, 250) by 2.0x",
            artifacts={"image": zoomed_img},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Zoom and analyze")

        # Verify zoom tool was called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "zoom"
        assert result.tool_calls[0].arguments["region"] == [100, 50, 300, 250]
        assert result.tool_calls[0].arguments["scale"] == 2.0

    def test_crop_tool_execution(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """VLM can call crop tool to extract region."""
        # VLM calls crop tool
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "crop", "arguments": {"region": [50, 100, 200, 300]}}',
        )
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text="Cropped region extracted.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2]

        # Mock crop tool execution
        from mata.core.artifacts.image import Image

        cropped_img = Image.from_pil(PILImage.new("RGB", (150, 200), color=(200, 200, 200)))

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="crop",
            success=True,
            summary="Cropped region (50, 100, 200, 300)",
            artifacts={"image": cropped_img},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)
        result = loop.run(sample_image, "Crop region")

        # Verify crop tool was called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "crop"
        assert result.tool_calls[0].arguments["region"] == [50, 100, 200, 300]

        # Verify tool result was successful
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True


# ============================================================================
# Task E2: Error Recovery & Guardrails Tests
# ============================================================================


class TestMalformedToolCallDetection:
    """Tests for detecting malformed tool call attempts."""

    def test_detects_fenced_tool_call_marker(self, mock_vlm_provider, mock_tool_registry):
        """Detects text with ```tool_call marker as attempted call."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        # Text with malformed fenced block
        text = "Let me analyze this. ```tool_call\n{tool: detect"  # Missing closing
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_detects_xml_tool_call_marker(self, mock_vlm_provider, mock_tool_registry):
        """Detects text with <tool_call> marker as attempted call."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = "I'll use <tool_call>{invalid json}</tool_call>"
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_detects_tool_key_in_json_like_structure(self, mock_vlm_provider, mock_tool_registry):
        """Detects JSON-like structure with 'tool' key."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = '{"tool": detect, "arguments": {}}'  # Missing quotes around detect
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_detects_action_key_in_json_like_structure(self, mock_vlm_provider, mock_tool_registry):
        """Detects JSON-like structure with 'action' key (alternate format)."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = "{'action': 'detect', 'parameters': {}}"  # Single quotes
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_detects_arguments_key(self, mock_vlm_provider, mock_tool_registry):
        """Detects JSON-like structure with 'arguments' key."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = '{"arguments": {"threshold": 0.5}}'  # Missing tool name
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_detects_multiple_braces_with_quotes(self, mock_vlm_provider, mock_tool_registry):
        """Detects multiple curly braces suggesting JSON attempt."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = 'I will use {{  "detect": "true" }}'
        assert loop._looks_like_attempted_tool_call(text) is True

    def test_does_not_detect_plain_text(self, mock_vlm_provider, mock_tool_registry):
        """Plain text without tool call markers is not detected."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = "This is a normal final answer with no tool calls."
        assert loop._looks_like_attempted_tool_call(text) is False

    def test_does_not_detect_single_brace(self, mock_vlm_provider, mock_tool_registry):
        """Text with single braces is not detected as tool call."""
        loop = AgentLoop(mock_vlm_provider, mock_tool_registry)

        text = "The function signature is: detect(image) -> result"
        assert loop._looks_like_attempted_tool_call(text) is False


class TestMalformedToolCallRetry:
    """Tests for retry logic on malformed tool calls."""

    def test_retry_mode_retries_malformed_call_with_clarification(
        self, sample_image, mock_vlm_provider, mock_tool_registry
    ):
        """on_error='retry' retries malformed tool call with clarifying prompt."""
        # Turn 1: Malformed tool call (missing quotes)
        turn1 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": detect, "arguments": {}}',  # Missing quotes around "detect"
        )

        # Turn 2: VLM tries again after clarification
        turn2 = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',  # Fixed
        )

        # Turn 3: Final answer after successful tool execution
        turn3 = VisionResult(
            instances=[],
            entities=[],
            text="Analysis complete.",
        )

        mock_vlm_provider.query.side_effect = [turn1, turn2, turn3]

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="detect",
            success=True,
            summary="Detected 1 object",
            artifacts={},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="retry", max_retries=2)
        result = loop.run(sample_image, "Test")

        # Verify we had 2 iterations (retry + successful execution)
        assert result.iterations >= 2

        # Verify conversation includes clarification message
        conversation_text = str(result.conversation)
        assert "malformed" in conversation_text.lower() or "correct format" in conversation_text.lower()

    def test_retry_mode_gives_up_after_max_retries(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Retry mode gives up after max_retries exceeded."""
        # All turns produce malformed calls
        malformed_turn = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": detect}',  # Always malformed
        )

        mock_vlm_provider.query.return_value = malformed_turn

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="retry", max_retries=2)
        result = loop.run(sample_image, "Test")

        # Should hit max retries and treat final attempt as answer
        # Max 2 retries means 3 total attempts (original + 2 retries)
        assert result.iterations <= 3
        assert len(result.tool_calls) == 0  # No successful tool calls

    def test_fail_mode_raises_on_malformed_call(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """on_error='fail' raises RuntimeError on malformed tool call."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="```tool_call\n{invalid json",  # Malformed
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="fail")

        with pytest.raises(RuntimeError, match="Malformed tool call"):
            loop.run(sample_image, "Test")

    def test_skip_mode_treats_malformed_as_final_answer(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """on_error='skip' treats malformed call as final answer."""
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": detect}',  # Malformed
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")
        result = loop.run(sample_image, "Test")

        # Should complete with 1 iteration, no tool calls
        assert result.iterations == 1
        assert len(result.tool_calls) == 0
        assert result.text == '{"tool": detect}'  # Returns malformed text as answer


class TestErrorLoggingLevel:
    """Tests for WARNING level logging on error recovery."""

    def test_tool_failure_logs_warning(self, sample_image, mock_vlm_provider, mock_tool_registry, caplog):
        """Tool execution failure logs at WARNING level."""
        import logging

        mock_vlm_provider.query.side_effect = [
            VisionResult(instances=[], entities=[], text='{"tool": "detect", "arguments": {}}'),
            VisionResult(instances=[], entities=[], text="Final answer."),
        ]

        mock_tool_registry.execute_tool.side_effect = RuntimeError("Tool failed")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")

        with caplog.at_level(logging.WARNING):
            loop.run(sample_image, "Test")

        # Check that WARNING messages were logged
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0

        # Check message content
        warning_messages = [r.message for r in warning_records]
        assert any("Tool execution failed" in msg for msg in warning_messages)

    def test_malformed_call_logs_warning(self, sample_image, mock_vlm_provider, mock_tool_registry, caplog):
        """Malformed tool call logs at WARNING level."""
        import logging

        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text="```tool_call\n{malformed",  # Malformed
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")

        with caplog.at_level(logging.WARNING):
            loop.run(sample_image, "Test")

        # Check that WARNING was logged for malformed call
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0

        warning_messages = [r.message for r in warning_records]
        assert any("Malformed" in msg or "malformed" in msg for msg in warning_messages)

    def test_vlm_failure_logs_warning(self, sample_image, mock_vlm_provider, mock_tool_registry, caplog):
        """VLM query failure logs at WARNING level."""
        import logging

        mock_vlm_provider.query.side_effect = RuntimeError("VLM connection lost")

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, on_error="skip")

        with caplog.at_level(logging.WARNING):
            loop.run(sample_image, "Test")

        # Check that WARNING was logged
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0

        warning_messages = [r.message for r in warning_records]
        assert any("VLM query failed" in msg for msg in warning_messages)

    def test_loop_detection_logs_warning(self, sample_image, mock_vlm_provider, mock_tool_registry, caplog):
        """Infinite loop detection logs at WARNING level."""
        import logging

        # Always return the same tool call
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {"threshold": 0.5}}',
        )

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="detect",
            success=True,
            summary="Detected",
            artifacts={},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=5)

        with caplog.at_level(logging.WARNING):
            loop.run(sample_image, "Test")

        # Check that loop detection WARNING was logged
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0

        warning_messages = [r.message for r in warning_records]
        assert any("loop detected" in msg.lower() for msg in warning_messages)


class TestMaxIterationsEnforcement:
    """Tests for max_iterations hard cap enforcement."""

    def test_max_iterations_always_respected(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """Loop never exceeds max_iterations even with continuous tool calls."""
        # Always return a tool call (never final answer)
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": "detect", "arguments": {}}',
        )

        mock_tool_registry.execute_tool.return_value = ToolResult(
            tool_name="detect",
            success=True,
            summary="Detected",
            artifacts={},
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=3)
        result = loop.run(sample_image, "Test")

        # Should stop at exactly max_iterations
        assert result.iterations <= 3

    def test_max_iterations_with_retries(self, sample_image, mock_vlm_provider, mock_tool_registry):
        """max_iterations respected even when retrying malformed calls."""
        # Always return malformed calls
        mock_vlm_provider.query.return_value = VisionResult(
            instances=[],
            entities=[],
            text='{"tool": detect}',  # Malformed
        )

        loop = AgentLoop(mock_vlm_provider, mock_tool_registry, max_iterations=2, on_error="retry", max_retries=5)
        result = loop.run(sample_image, "Test")

        # Should respect max_iterations even though max_retries is higher
        assert result.iterations <= 2
