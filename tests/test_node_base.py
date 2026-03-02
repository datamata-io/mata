"""Unit tests for Node base class.

Tests cover:
- Abstract enforcement
- Input validation (correct/incorrect types)
- Output validation
- Config storage
- Error messages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

from mata.core.artifacts.base import Artifact
from mata.core.exceptions import ValidationError
from mata.core.graph.node import Node


# Mock artifacts for testing
@dataclass(frozen=True)
class MockImageArtifact(Artifact):
    """Mock image artifact for testing."""

    width: int
    height: int
    data: str = "mock_data"

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "data": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockImageArtifact:
        return cls(width=data["width"], height=data["height"], data=data.get("data", "mock_data"))

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")


@dataclass(frozen=True)
class MockDetectionsArtifact(Artifact):
    """Mock detections artifact for testing."""

    num_objects: int
    labels: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        return {"num_objects": self.num_objects, "labels": list(self.labels)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockDetectionsArtifact:
        return cls(num_objects=data["num_objects"], labels=tuple(data.get("labels", [])))

    def validate(self) -> None:
        if self.num_objects < 0:
            raise ValueError("Number of objects must be non-negative")


@dataclass(frozen=True)
class MockMasksArtifact(Artifact):
    """Mock masks artifact for testing."""

    num_masks: int

    def to_dict(self) -> dict[str, Any]:
        return {"num_masks": self.num_masks}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockMasksArtifact:
        return cls(num_masks=data["num_masks"])


# Mock nodes for testing
class MockDetectNode(Node):
    """Mock detection node with proper type signatures."""

    inputs = {"image": MockImageArtifact}
    outputs = {"detections": MockDetectionsArtifact}

    def run(self, ctx: ExecutionContext, image: MockImageArtifact) -> dict[str, Artifact]:
        # Simple mock implementation
        return {"detections": MockDetectionsArtifact(num_objects=3, labels=("cat", "dog", "bird"))}


class MockSegmentNode(Node):
    """Mock segmentation node with multiple inputs."""

    inputs = {"image": MockImageArtifact, "detections": MockDetectionsArtifact}
    outputs = {"masks": MockMasksArtifact}

    def run(
        self, ctx: ExecutionContext, image: MockImageArtifact, detections: MockDetectionsArtifact
    ) -> dict[str, Artifact]:
        return {"masks": MockMasksArtifact(num_masks=detections.num_objects)}


class MockMultiOutputNode(Node):
    """Mock node with multiple outputs."""

    inputs = {"image": MockImageArtifact}
    outputs = {"detections": MockDetectionsArtifact, "masks": MockMasksArtifact}

    def run(self, ctx: ExecutionContext, image: MockImageArtifact) -> dict[str, Artifact]:
        return {"detections": MockDetectionsArtifact(num_objects=2), "masks": MockMasksArtifact(num_masks=2)}


class MockOptionalInputNode(Node):
    """Mock node with optional input (using Optional type)."""

    inputs = {"image": MockImageArtifact, "detections": MockDetectionsArtifact | None}
    outputs = {"masks": MockMasksArtifact}

    def run(self, ctx: ExecutionContext, **inputs) -> dict[str, Artifact]:
        num_masks = inputs.get("detections").num_objects if "detections" in inputs else 0
        return {"masks": MockMasksArtifact(num_masks=num_masks)}


# Test class
class TestNodeBase:
    """Test suite for Node base class."""

    def test_abstract_enforcement(self):
        """Test that Node cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Node()

    def test_missing_run_method(self):
        """Test that subclass without run() method cannot be instantiated."""
        with pytest.raises(TypeError):

            class BadNode(Node):
                inputs = {}
                outputs = {}
                # Missing run() method

            BadNode()

    def test_node_initialization_default_name(self):
        """Test node initialization with default name."""
        node = MockDetectNode()
        assert node.name == "MockDetectNode"
        assert node.config == {}

    def test_node_initialization_custom_name(self):
        """Test node initialization with custom name."""
        node = MockDetectNode(name="my_detector")
        assert node.name == "my_detector"
        assert node.config == {}

    def test_node_initialization_with_config(self):
        """Test node initialization with configuration."""
        node = MockDetectNode(name="detector_1", threshold=0.5, device="cuda", model_id="detr-r50")
        assert node.name == "detector_1"
        assert node.config == {"threshold": 0.5, "device": "cuda", "model_id": "detr-r50"}

    def test_required_inputs_property(self):
        """Test required_inputs property returns correct input names."""
        node = MockDetectNode()
        assert node.required_inputs == {"image"}

        node2 = MockSegmentNode()
        assert node2.required_inputs == {"image", "detections"}

    def test_provided_outputs_property(self):
        """Test provided_outputs property returns correct output names."""
        node = MockDetectNode()
        assert node.provided_outputs == {"detections"}

        node2 = MockMultiOutputNode()
        assert node2.provided_outputs == {"detections", "masks"}

    def test_validate_inputs_correct_types(self):
        """Test input validation passes with correct types."""
        node = MockDetectNode()
        inputs = {"image": MockImageArtifact(width=640, height=480)}

        # Should not raise
        node.validate_inputs(inputs)

    def test_validate_inputs_missing_required(self):
        """Test input validation fails with missing required inputs."""
        node = MockDetectNode()
        inputs = {}  # Missing 'image'

        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs(inputs)

        assert "missing required inputs" in str(exc_info.value).lower()
        assert "image" in str(exc_info.value)

    def test_validate_inputs_wrong_type(self):
        """Test input validation fails with wrong artifact type."""
        node = MockDetectNode()
        # Pass detections instead of image
        inputs = {"image": MockDetectionsArtifact(num_objects=3)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs(inputs)

        assert "wrong type" in str(exc_info.value).lower()
        assert "MockImageArtifact" in str(exc_info.value)
        assert "MockDetectionsArtifact" in str(exc_info.value)

    def test_validate_inputs_unexpected_input(self):
        """Test input validation fails with unexpected inputs."""
        node = MockDetectNode()
        inputs = {"image": MockImageArtifact(width=640, height=480), "extra_input": MockMasksArtifact(num_masks=5)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs(inputs)

        assert "unexpected inputs" in str(exc_info.value).lower()
        assert "extra_input" in str(exc_info.value)

    def test_validate_inputs_multiple_inputs(self):
        """Test input validation with multiple inputs."""
        node = MockSegmentNode()
        inputs = {
            "image": MockImageArtifact(width=640, height=480),
            "detections": MockDetectionsArtifact(num_objects=3),
        }

        # Should not raise
        node.validate_inputs(inputs)

    def test_validate_inputs_calls_artifact_validate(self):
        """Test input validation calls artifact.validate() method."""
        node = MockDetectNode()
        # Create invalid artifact (negative dimensions)
        inputs = {"image": MockImageArtifact(width=-1, height=480)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs(inputs)

        assert "invalid artifact" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_validate_outputs_correct_types(self):
        """Test output validation passes with correct types."""
        node = MockDetectNode()
        outputs = {"detections": MockDetectionsArtifact(num_objects=3)}

        # Should not raise
        node.validate_outputs(outputs)

    def test_validate_outputs_missing_required(self):
        """Test output validation fails with missing outputs."""
        node = MockDetectNode()
        outputs = {}  # Missing 'detections'

        with pytest.raises(ValidationError) as exc_info:
            node.validate_outputs(outputs)

        assert "missing required outputs" in str(exc_info.value).lower()
        assert "detections" in str(exc_info.value)

    def test_validate_outputs_wrong_type(self):
        """Test output validation fails with wrong artifact type."""
        node = MockDetectNode()
        # Return masks instead of detections
        outputs = {"detections": MockMasksArtifact(num_masks=3)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_outputs(outputs)

        assert "wrong type" in str(exc_info.value).lower()
        assert "MockDetectionsArtifact" in str(exc_info.value)
        assert "MockMasksArtifact" in str(exc_info.value)

    def test_validate_outputs_unexpected_output(self):
        """Test output validation fails with unexpected outputs."""
        node = MockDetectNode()
        outputs = {"detections": MockDetectionsArtifact(num_objects=3), "extra_output": MockMasksArtifact(num_masks=5)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_outputs(outputs)

        assert "unexpected outputs" in str(exc_info.value).lower()
        assert "extra_output" in str(exc_info.value)

    def test_validate_outputs_multiple_outputs(self):
        """Test output validation with multiple outputs."""
        node = MockMultiOutputNode()
        outputs = {"detections": MockDetectionsArtifact(num_objects=2), "masks": MockMasksArtifact(num_masks=2)}

        # Should not raise
        node.validate_outputs(outputs)

    def test_validate_outputs_calls_artifact_validate(self):
        """Test output validation calls artifact.validate() method."""
        node = MockDetectNode()
        # Create invalid artifact (negative objects)
        outputs = {"detections": MockDetectionsArtifact(num_objects=-1)}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_outputs(outputs)

        assert "invalid artifact" in str(exc_info.value).lower()
        assert "non-negative" in str(exc_info.value).lower()

    def test_repr(self):
        """Test string representation of node."""
        node = MockDetectNode(name="detector_1")
        repr_str = repr(node)

        assert "MockDetectNode" in repr_str
        assert "name='detector_1'" in repr_str
        assert "inputs=" in repr_str
        assert "MockImageArtifact" in repr_str
        assert "outputs=" in repr_str
        assert "MockDetectionsArtifact" in repr_str

    def test_repr_multi_io(self):
        """Test string representation with multiple inputs/outputs."""
        node = MockMultiOutputNode()
        repr_str = repr(node)

        assert "MockMultiOutputNode" in repr_str
        assert "MockImageArtifact" in repr_str
        assert "MockDetectionsArtifact" in repr_str
        assert "MockMasksArtifact" in repr_str

    def test_error_messages_clear(self):
        """Test that error messages include helpful context."""
        node = MockDetectNode(name="my_detector")

        # Test missing input error message
        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs({})

        error_msg = str(exc_info.value)
        assert "my_detector" in error_msg  # Node name
        assert "Required:" in error_msg  # Shows what was required
        assert "Provided:" in error_msg  # Shows what was provided

        # Test wrong type error message
        with pytest.raises(ValidationError) as exc_info:
            node.validate_inputs({"image": MockDetectionsArtifact(num_objects=1)})

        error_msg = str(exc_info.value)
        assert "my_detector" in error_msg
        assert "image" in error_msg  # Input name
        assert "Expected" in error_msg
        assert "got" in error_msg

    def test_node_with_no_declared_inputs(self):
        """Test node with no declared inputs (e.g., source nodes)."""

        class SourceNode(Node):
            inputs = {}
            outputs = {"image": MockImageArtifact}

            def run(self, ctx: ExecutionContext) -> dict[str, Artifact]:
                return {"image": MockImageArtifact(width=640, height=480)}

        node = SourceNode()
        assert node.required_inputs == set()

        # Should accept no inputs
        node.validate_inputs({})

        # Should also accept inputs gracefully if not strictly enforcing
        # (This behavior can be adjusted based on requirements)

    def test_node_with_no_declared_outputs(self):
        """Test node with no declared outputs (e.g., sink nodes)."""

        class SinkNode(Node):
            inputs = {"image": MockImageArtifact}
            outputs = {}

            def run(self, ctx: ExecutionContext, image: MockImageArtifact) -> dict[str, Artifact]:
                # Process but don't output
                return {}

        node = SinkNode()
        assert node.provided_outputs == set()

        # Should accept empty outputs
        node.validate_outputs({})

    def test_subclass_inheritance(self):
        """Test that artifact type checking supports subclasses."""

        # Create a subclass of MockImageArtifact
        @dataclass(frozen=True)
        class EnhancedImageArtifact(MockImageArtifact):
            metadata: str = ""

        node = MockDetectNode()

        # Should accept subclass
        enhanced_image = EnhancedImageArtifact(width=640, height=480, metadata="test")
        inputs = {"image": enhanced_image}

        node.validate_inputs(inputs)  # Should not raise


class TestNodeIntegration:
    """Integration tests for Node with mock execution context."""

    def test_node_execution_flow(self):
        """Test complete node execution flow with validation."""

        # Create mock execution context
        class MockContext:
            def get_provider(self, capability, name):
                return None

        node = MockDetectNode(name="detector", threshold=0.5)
        ctx = MockContext()

        # Prepare inputs
        inputs = {"image": MockImageArtifact(width=640, height=480)}

        # Validate inputs
        node.validate_inputs(inputs)

        # Execute node
        outputs = node.run(ctx, **inputs)

        # Validate outputs
        node.validate_outputs(outputs)

        # Check results
        assert "detections" in outputs
        assert isinstance(outputs["detections"], MockDetectionsArtifact)
        assert outputs["detections"].num_objects == 3

    def test_multi_input_node_execution(self):
        """Test node with multiple inputs."""

        class MockContext:
            def get_provider(self, capability, name):
                return None

        node = MockSegmentNode()
        ctx = MockContext()

        inputs = {
            "image": MockImageArtifact(width=640, height=480),
            "detections": MockDetectionsArtifact(num_objects=5),
        }

        # Validate and execute
        node.validate_inputs(inputs)
        outputs = node.run(ctx, **inputs)
        node.validate_outputs(outputs)

        # Check that masks count matches detections
        assert outputs["masks"].num_masks == 5

    def test_multi_output_node_execution(self):
        """Test node with multiple outputs."""

        class MockContext:
            def get_provider(self, capability, name):
                return None

        node = MockMultiOutputNode()
        ctx = MockContext()

        inputs = {"image": MockImageArtifact(width=640, height=480)}

        node.validate_inputs(inputs)
        outputs = node.run(ctx, **inputs)
        node.validate_outputs(outputs)

        # Check both outputs present
        assert "detections" in outputs
        assert "masks" in outputs
        assert outputs["detections"].num_objects == 2
        assert outputs["masks"].num_masks == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
