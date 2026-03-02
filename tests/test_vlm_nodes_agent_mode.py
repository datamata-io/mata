"""Tests for VLM nodes with agent mode (tools parameter).

Verifies Task B2 implementation:
- VLMDetect, VLMQuery, VLMDescribe accept tools= parameter
- When tools=None (default), behavior is unchanged (backward compatibility)
- When tools=["detect", ...], nodes delegate to AgentLoop
- Agent mode produces valid Detections artifacts
"""

from unittest.mock import Mock, patch

import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.types import Entity, Instance, VisionResult
from mata.nodes.vlm_describe import VLMDescribe
from mata.nodes.vlm_detect import VLMDetect
from mata.nodes.vlm_query import VLMQuery


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    ctx = Mock(spec=ExecutionContext)
    ctx.record_metric = Mock()
    return ctx


@pytest.fixture
def mock_image():
    """Create a mock image artifact."""
    img = Mock(spec=Image)
    img.pil_image = Mock()
    img.pil_image.size = (640, 480)
    return img


@pytest.fixture
def mock_vlm_provider():
    """Create mock VLM provider."""
    provider = Mock()
    # Mock standard VLM query response
    provider.query.return_value = VisionResult(
        instances=[],
        entities=[Entity("cat", 0.95), Entity("dog", 0.87)],
        text="Found a cat and a dog.",
    )
    return provider


class TestVLMDetectAgentMode:
    """Tests for VLMDetect with agent mode."""

    def test_without_tools_standard_mode(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMDetect without tools= uses standard single-call mode."""
        mock_context.get_provider.return_value = mock_vlm_provider

        # No tools= parameter (default behavior)
        node = VLMDetect(using="vlm", prompt="Detect objects.")
        result = node.run(mock_context, mock_image)

        # Verify standard VLM query was called (not agent loop)
        mock_vlm_provider.query.assert_called_once()
        assert "vlm_dets" in result
        assert isinstance(result["vlm_dets"], Detections)

        # Verify metrics recorded
        assert mock_context.record_metric.call_count >= 3  # latency_ms, num_entities, num_instances

    def test_with_tools_agent_mode_called(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMDetect with tools= delegates to AgentLoop."""
        from mata.core.agent_loop import AgentResult

        mock_context.get_provider.return_value = mock_vlm_provider

        # Mock AgentLoop and ToolRegistry at the module level where they're imported
        with (
            patch("mata.core.agent_loop.AgentLoop") as mock_agent_loop_class,
            patch("mata.core.tool_registry.ToolRegistry") as mock_registry_class,
        ):

            # Create mock AgentResult
            mock_agent_result = AgentResult(
                text="Found 2 cats and 1 dog.",
                tool_calls=[],
                tool_results=[],
                iterations=3,
                instances=[Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")],
                entities=[Entity("cat", 0.95)],
                conversation=[],
            )

            # Configure mocks
            mock_loop_instance = Mock()
            mock_loop_instance.run.return_value = mock_agent_result
            mock_agent_loop_class.return_value = mock_loop_instance
            mock_registry_class.return_value = Mock()

            # Create node with tools
            node = VLMDetect(
                using="vlm",
                prompt="Analyze this image.",
                tools=["detect", "classify"],
                max_iterations=5,
                on_error="retry",
            )

            # Run node
            result = node.run(mock_context, mock_image)

            # Verify AgentLoop was instantiated
            mock_agent_loop_class.assert_called_once()
            args, kwargs = mock_agent_loop_class.call_args
            assert args[0] == mock_vlm_provider  # vlm provider
            assert args[2] == 5  # max_iterations
            assert args[3] == "retry"  # on_error

            # Verify loop.run() was called
            mock_loop_instance.run.assert_called_once()

            # Verify result is Detections artifact
            assert "vlm_dets" in result
            assert isinstance(result["vlm_dets"], Detections)

            # Verify metrics include agent-specific metrics
            metric_calls = [call[0] for call in mock_context.record_metric.call_args_list]
            metric_names = [call[1] for call in metric_calls]
            assert "agent_iterations" in metric_names
            assert "tool_calls_count" in metric_names


class TestVLMQueryAgentMode:
    """Tests for VLMQuery with agent mode."""

    def test_without_tools_standard_mode(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMQuery without tools= uses standard mode."""
        mock_context.get_provider.return_value = mock_vlm_provider

        node = VLMQuery(using="vlm", prompt="What do you see?")
        result = node.run(mock_context, mock_image)

        mock_vlm_provider.query.assert_called_once()
        assert "vlm_result" in result
        assert isinstance(result["vlm_result"], Detections)

    def test_with_tools_agent_mode(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMQuery with tools= uses agent mode."""
        from mata.core.agent_loop import AgentResult

        mock_context.get_provider.return_value = mock_vlm_provider

        with (
            patch("mata.core.agent_loop.AgentLoop") as mock_agent_loop_class,
            patch("mata.core.tool_registry.ToolRegistry"),
        ):

            mock_agent_result = AgentResult(
                text="Analysis complete.",
                iterations=2,
                instances=[],
                entities=[],
            )
            mock_loop_instance = Mock()
            mock_loop_instance.run.return_value = mock_agent_result
            mock_agent_loop_class.return_value = mock_loop_instance

            node = VLMQuery(
                using="vlm",
                prompt="Analyze this scene.",
                tools=["detect"],
            )
            result = node.run(mock_context, mock_image)

            mock_agent_loop_class.assert_called_once()
            assert "vlm_result" in result
            assert isinstance(result["vlm_result"], Detections)


class TestVLMDescribeAgentMode:
    """Tests for VLMDescribe with agent mode."""

    def test_without_tools_standard_mode(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMDescribe without tools= uses standard mode."""
        mock_context.get_provider.return_value = mock_vlm_provider

        node = VLMDescribe(using="vlm")
        result = node.run(mock_context, mock_image)

        mock_vlm_provider.query.assert_called_once()
        assert "description" in result
        assert isinstance(result["description"], Detections)

    def test_with_tools_agent_mode(self, mock_context, mock_image, mock_vlm_provider):
        """Test VLMDescribe with tools= uses agent mode."""
        from mata.core.agent_loop import AgentResult

        mock_context.get_provider.return_value = mock_vlm_provider

        with (
            patch("mata.core.agent_loop.AgentLoop") as mock_agent_loop_class,
            patch("mata.core.tool_registry.ToolRegistry"),
        ):

            mock_agent_result = AgentResult(
                text="Detailed description of the scene.",
                iterations=1,
                instances=[],
                entities=[Entity("tree", 0.98)],
            )
            mock_loop_instance = Mock()
            mock_loop_instance.run.return_value = mock_agent_result
            mock_agent_loop_class.return_value = mock_loop_instance

            node = VLMDescribe(
                using="vlm",
                tools=["detect", "classify", "depth"],
                max_iterations=10,
            )
            result = node.run(mock_context, mock_image)

            mock_agent_loop_class.assert_called_once()
            args, _ = mock_agent_loop_class.call_args
            assert args[2] == 10  # max_iterations

            assert "description" in result
            assert isinstance(result["description"], Detections)


class TestAgentResultConversion:
    """Test _agent_result_to_detections() helper method."""

    def test_converts_agent_result_to_detections(self):
        """Test conversion preserves instances, entities, text, and metadata."""
        from mata.core.agent_loop import AgentResult
        from mata.core.tool_schema import ToolCall, ToolResult

        # Create AgentResult with full data
        tool_call = ToolCall(
            tool_name="detect",
            arguments={"threshold": 0.5},
            raw_text='{"tool": "detect", "arguments": {"threshold": 0.5}}',
        )
        tool_result = ToolResult(
            tool_name="detect",
            success=True,
            summary="Found 2 objects",
            artifacts={},
        )

        agent_result = AgentResult(
            text="Final analysis: found cats and dogs.",
            tool_calls=[tool_call],
            tool_results=[tool_result],
            iterations=3,
            instances=[Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")],
            entities=[Entity("cat", 0.95), Entity("dog", 0.87)],
            conversation=[
                {"role": "user", "content": "Analyze this image."},
                {"role": "assistant", "content": "I'll use the detect tool."},
            ],
            meta={"extra_key": "extra_value"},
        )

        # Use VLMDetect to test conversion
        node = VLMDetect(using="vlm")
        detections = node._agent_result_to_detections(agent_result)

        # Verify instances and entities preserved
        assert len(detections.instances) == 1
        assert detections.instances[0].label_name == "cat"
        assert len(detections.entities) == 2

        # Verify metadata preserved (text is stored in meta)
        assert "agent_iterations" in detections.meta
        assert detections.meta["agent_iterations"] == 3
        assert "agent_text" in detections.meta  # Changed from "tool_calls" to "agent_text"
        assert detections.meta["agent_text"] == "Final analysis: found cats and dogs."
        assert "agent_tool_calls" in detections.meta  # Changed from "tool_calls" to "agent_tool_calls"
        assert len(detections.meta["agent_tool_calls"]) == 1
        assert "agent_tool_results" in detections.meta  # Changed from "tool_results" to "agent_tool_results"
        assert len(detections.meta["agent_tool_results"]) == 1
        assert "conversation" in detections.meta
        assert len(detections.meta["conversation"]) == 2
        assert "extra_key" in detections.meta
        assert detections.meta["extra_key"] == "extra_value"

        # Verify text can be accessed via to_vision_result()
        # Note: text is stored in meta, not as a direct field on VisionResult
        vision_result = detections.to_vision_result()
        assert len(vision_result.instances) == 1
        assert len(vision_result.entities) == 2


class TestBackwardCompatibility:
    """Ensure tools=None (default) maintains 100% backward compatibility."""

    def test_vlm_detect_default_params(self):
        """Test VLMDetect() without tools= has same signature as before."""
        node = VLMDetect(using="vlm")
        assert node.tools is None
        assert node.max_iterations == 5  # default
        assert node.on_error == "retry"  # default

    def test_vlm_query_default_params(self):
        """Test VLMQuery() without tools= has same signature as before."""
        node = VLMQuery(using="vlm", prompt="Test")
        assert node.tools is None
        assert node.max_iterations == 5
        assert node.on_error == "retry"

    def test_vlm_describe_default_params(self):
        """Test VLMDescribe() without tools= has same signature as before."""
        node = VLMDescribe(using="vlm")
        assert node.tools is None
        assert node.max_iterations == 5
        assert node.on_error == "retry"

    def test_existing_code_still_works(self, mock_context, mock_image, mock_vlm_provider):
        """Test that all existing VLM node usage patterns still work."""
        mock_context.get_provider.return_value = mock_vlm_provider

        # Old-style VLMDetect usage
        node1 = VLMDetect(using="vlm", prompt="Detect.", auto_promote=True)
        result1 = node1.run(mock_context, mock_image)
        assert "vlm_dets" in result1

        # Old-style VLMQuery usage
        node2 = VLMQuery(using="vlm", prompt="Query?", output_mode="detect")
        result2 = node2.run(mock_context, mock_image)
        assert "vlm_result" in result2

        # Old-style VLMDescribe usage
        node3 = VLMDescribe(using="vlm", out="desc")
        result3 = node3.run(mock_context, mock_image)
        assert "desc" in result3
