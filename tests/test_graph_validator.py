"""Tests for graph validator.

Comprehensive test suite for GraphValidator covering:
- Type compatibility checking
- Dependency resolution
- Cycle detection
- Name collision detection
- Provider capability verification
- Error message clarity
"""

from typing import Any

import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.exceptions import ValidationError
from mata.core.graph.node import Node
from mata.core.graph.validator import GraphValidator, ValidationResult


# Mock Artifact types for testing
class MockMasks(Artifact):
    """Mock Masks artifact for testing."""

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockMasks":
        return cls()


class MockClassifications(Artifact):
    """Mock Classifications artifact for testing."""

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockClassifications":
        return cls()


# Mock Node implementations for testing
class MockDetectNode(Node):
    """Mock detection node."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, provider_name: str = "detr", name: str = "MockDetect"):
        super().__init__(name=name)
        self.provider_name = provider_name

    def run(self, ctx, **inputs):
        return {}


class MockFilterNode(Node):
    """Mock filter node."""

    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "MockFilter"):
        super().__init__(name=name)

    def run(self, ctx, **inputs):
        return {}


class MockSegmentNode(Node):
    """Mock segmentation node."""

    inputs = {"image": Image, "detections": Detections}
    outputs = {"masks": MockMasks}

    def __init__(self, provider_name: str = "sam", name: str = "MockSegment"):
        super().__init__(name=name)
        self.provider_name = provider_name

    def run(self, ctx, **inputs):
        return {}


class MockClassifyNode(Node):
    """Mock classification node."""

    inputs = {"image": Image}
    outputs = {"classifications": MockClassifications}

    def __init__(self, provider_name: str = "resnet", name: str = "MockClassify"):
        super().__init__(name=name)
        self.provider_name = provider_name

    def run(self, ctx, **inputs):
        return {}


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result with no errors."""
        result = ValidationResult(valid=True, errors=[], warnings=[])

        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_invalid_result(self):
        """Test invalid result with errors."""
        result = ValidationResult(valid=False, errors=["Error 1", "Error 2"], warnings=["Warning 1"])

        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_raise_if_invalid_with_valid_result(self):
        """Test raise_if_invalid does not raise for valid result."""
        result = ValidationResult(valid=True, errors=[], warnings=[])

        # Should not raise
        result.raise_if_invalid()

    def test_raise_if_invalid_with_invalid_result(self):
        """Test raise_if_invalid raises ValidationError for invalid result."""
        result = ValidationResult(valid=False, errors=["Error 1", "Error 2"], warnings=[])

        with pytest.raises(ValidationError) as exc_info:
            result.raise_if_invalid()

        assert "Graph validation failed" in str(exc_info.value)
        assert "Error 1" in str(exc_info.value)
        assert "Error 2" in str(exc_info.value)

    def test_string_representation_valid(self):
        """Test string representation of valid result."""
        result = ValidationResult(valid=True, errors=[], warnings=[])
        s = str(result)

        assert "✓ Valid" in s

    def test_string_representation_valid_with_warnings(self):
        """Test string representation of valid result with warnings."""
        result = ValidationResult(valid=True, errors=[], warnings=["Warning 1"])
        s = str(result)

        assert "✓ Valid" in s
        assert "1 warnings" in s
        assert "Warning 1" in s

    def test_string_representation_invalid(self):
        """Test string representation of invalid result."""
        result = ValidationResult(valid=False, errors=["Error 1", "Error 2"], warnings=["Warning 1"])
        s = str(result)

        assert "✗ Invalid" in s
        assert "2 errors" in s
        assert "1 warnings" in s
        assert "Error 1" in s
        assert "Error 2" in s
        assert "Warning 1" in s


class TestGraphValidator:
    """Tests for GraphValidator class."""

    def test_empty_graph_validation(self):
        """Test validation fails for empty graph."""
        validator = GraphValidator()
        result = validator.validate(nodes=[], wiring={})

        assert result.valid is False
        assert len(result.errors) == 1
        assert "must contain at least one node" in result.errors[0]

    def test_simple_valid_graph(self):
        """Test validation passes for simple valid graph."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "detect.detections"}

        result = validator.validate(nodes=[detect, filter_node], wiring=wiring)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_multi_node_valid_graph(self):
        """Test validation passes for multi-node graph."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        segment = MockSegmentNode(name="segment")
        classify = MockClassifyNode(name="classify")

        wiring = {
            "detect.image": "input.image",
            "segment.image": "input.image",
            "segment.detections": "detect.detections",
            "classify.image": "input.image",
        }

        result = validator.validate(nodes=[detect, segment, classify], wiring=wiring)

        assert result.valid is True
        assert len(result.errors) == 0


class TestTypeCompatibility:
    """Tests for type compatibility checking."""

    def test_compatible_types(self):
        """Test type checking passes for compatible types."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "detect.detections"}

        errors = validator.check_type_compatibility([detect, filter_node], wiring)

        assert len(errors) == 0

    def test_type_mismatch(self):
        """Test type checking fails for incompatible types."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        classify = MockClassifyNode(name="classify")

        # Wire detections to image input (type mismatch!)
        wiring = {"detect.image": "input.image", "classify.image": "detect.detections"}  # Wrong type!

        errors = validator.check_type_compatibility([detect, classify], wiring)

        assert len(errors) == 1
        assert "Type mismatch" in errors[0]
        assert "Detections" in errors[0]
        assert "Image" in errors[0]

    def test_unknown_node_in_wiring(self):
        """Test error when wiring references unknown node."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")

        wiring = {"detect.image": "input.image", "unknown_node.input": "detect.detections"}  # Unknown node!

        errors = validator.check_type_compatibility([detect], wiring)

        assert len(errors) == 1
        assert "Unknown node 'unknown_node'" in errors[0]

    def test_unknown_input_in_wiring(self):
        """Test error when wiring references unknown input."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.wrong_input": "detect.detections"}  # Wrong input name!

        errors = validator.check_type_compatibility([detect, filter_node], wiring)

        assert len(errors) == 1
        assert "has no input 'wrong_input'" in errors[0]

    def test_unknown_artifact_source(self):
        """Test error when wiring references non-existent artifact."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "nonexistent.artifact"}  # Doesn't exist!

        errors = validator.check_type_compatibility([detect, filter_node], wiring)

        assert len(errors) == 1
        assert "Artifact 'nonexistent.artifact' not found" in errors[0]

    def test_external_input_skipped(self):
        """Test external inputs (input.*) are skipped in type checking."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")

        wiring = {"detect.image": "input.image"}  # External input, no type checking

        errors = validator.check_type_compatibility([detect], wiring)

        # Should not raise errors for external inputs
        assert len(errors) == 0

    def test_invalid_wiring_format(self):
        """Test error for invalid wiring target format."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")

        wiring = {"invalid_target": "detect.detections"}  # Missing node.input format

        errors = validator.check_type_compatibility([detect], wiring)

        assert len(errors) == 1
        assert "Invalid wiring target" in errors[0]
        assert "node_name.input_name" in errors[0]


class TestDependencyResolution:
    """Tests for dependency resolution checking."""

    def test_all_dependencies_satisfied(self):
        """Test dependency checking passes when all inputs wired."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "detect.detections"}

        errors = validator.check_dependencies([detect, filter_node], wiring)

        assert len(errors) == 0

    def test_missing_required_input(self):
        """Test error when required input is not wired."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        # Missing wiring for filter.detections
        wiring = {
            "detect.image": "input.image"
            # filter.detections not wired!
        }

        errors = validator.check_dependencies([detect, filter_node], wiring)

        assert len(errors) == 1
        assert "filter" in errors[0]
        assert "requires input 'detections'" in errors[0]
        assert "not wired" in errors[0]

    def test_missing_source_artifact(self):
        """Test error when wired source doesn't exist."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "nonexistent.output"}  # Source doesn't exist

        errors = validator.check_dependencies([detect, filter_node], wiring)

        assert len(errors) == 1
        assert "is wired to 'nonexistent.output'" in errors[0]
        assert "not available" in errors[0]

    def test_multi_input_node(self):
        """Test dependency checking for node with multiple inputs."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        segment = MockSegmentNode(name="segment")

        # Segment requires both image and detections
        wiring = {
            "detect.image": "input.image",
            "segment.image": "input.image",
            "segment.detections": "detect.detections",
        }

        errors = validator.check_dependencies([detect, segment], wiring)

        assert len(errors) == 0

    def test_multi_input_node_missing_one_input(self):
        """Test error when multi-input node missing one input."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        segment = MockSegmentNode(name="segment")

        # Segment missing detections input
        wiring = {
            "detect.image": "input.image",
            "segment.image": "input.image",
            # segment.detections not wired!
        }

        errors = validator.check_dependencies([detect, segment], wiring)

        assert len(errors) == 1
        assert "segment" in errors[0]
        assert "requires input 'detections'" in errors[0]


class TestCycleDetection:
    """Tests for cycle detection in graphs."""

    def test_acyclic_graph(self):
        """Test cycle detection returns None for valid DAG."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        wiring = {"detect.image": "input.image", "filter.detections": "detect.detections"}

        cycle = validator.detect_cycles([detect, filter_node], wiring)

        assert cycle is None

    def test_simple_cycle(self):
        """Test cycle detection finds simple A→B→A cycle."""
        validator = GraphValidator()

        # Create two nodes that depend on each other (impossible!)
        node_a = MockFilterNode(name="node_a")
        node_b = MockFilterNode(name="node_b")

        # A → B → A (cycle)
        wiring = {"node_a.detections": "node_b.detections", "node_b.detections": "node_a.detections"}

        cycle = validator.detect_cycles([node_a, node_b], wiring)

        assert cycle is not None
        assert len(cycle) >= 2
        # Cycle should contain both nodes
        assert "node_a" in cycle
        assert "node_b" in cycle

    def test_self_cycle(self):
        """Test cycle detection finds self-referential cycle."""
        validator = GraphValidator()

        node = MockFilterNode(name="self_node")

        # Node depends on itself (cycle)
        wiring = {"self_node.detections": "self_node.detections"}

        cycle = validator.detect_cycles([node], wiring)

        assert cycle is not None
        assert "self_node" in cycle

    def test_three_node_cycle(self):
        """Test cycle detection finds A→B→C→A cycle."""
        validator = GraphValidator()

        node_a = MockFilterNode(name="node_a")
        node_b = MockFilterNode(name="node_b")
        node_c = MockFilterNode(name="node_c")

        # A → B → C → A (cycle)
        wiring = {
            "node_a.detections": "node_c.detections",
            "node_b.detections": "node_a.detections",
            "node_c.detections": "node_b.detections",
        }

        cycle = validator.detect_cycles([node_a, node_b, node_c], wiring)

        assert cycle is not None
        assert len(cycle) >= 3

    def test_complex_dag_no_cycle(self):
        """Test cycle detection on complex DAG with multiple paths."""
        validator = GraphValidator()

        # Diamond pattern: A → B → D
        #                   A → C → D
        node_a = MockDetectNode(name="node_a")
        node_b = MockFilterNode(name="node_b")
        node_c = MockFilterNode(name="node_c")
        node_d = MockSegmentNode(name="node_d")

        wiring = {
            "node_a.image": "input.image",
            "node_b.detections": "node_a.detections",
            "node_c.detections": "node_a.detections",
            "node_d.image": "input.image",
            "node_d.detections": "node_b.detections",  # Could also use node_c
        }

        cycle = validator.detect_cycles([node_a, node_b, node_c, node_d], wiring)

        assert cycle is None

    def test_external_inputs_no_cycle(self):
        """Test external inputs don't create false cycles."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")

        wiring = {"detect.image": "input.image"}  # External input, not a cycle

        cycle = validator.detect_cycles([detect], wiring)

        assert cycle is None


class TestNameCollisions:
    """Tests for name collision detection."""

    def test_unique_node_names(self):
        """Test no collision when all node names unique."""
        validator = GraphValidator()

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        errors = validator.check_name_collisions([detect, filter_node])

        assert len(errors) == 0

    def test_duplicate_node_names(self):
        """Test error when node names are duplicated."""
        validator = GraphValidator()

        detect1 = MockDetectNode(name="detect")
        detect2 = MockDetectNode(name="detect")  # Same name!

        errors = validator.check_name_collisions([detect1, detect2])

        # Should have 2 errors: duplicate node name + duplicate output artifact
        assert len(errors) == 2
        error_str = " ".join(errors)
        assert "Duplicate node name 'detect'" in error_str
        assert "2 nodes" in error_str

    def test_multiple_duplicate_node_names(self):
        """Test multiple sets of duplicate names detected."""
        validator = GraphValidator()

        detect1 = MockDetectNode(name="detect")
        detect2 = MockDetectNode(name="detect")
        filter1 = MockFilterNode(name="filter")
        filter2 = MockFilterNode(name="filter")

        errors = validator.check_name_collisions([detect1, detect2, filter1, filter2])

        # Should have 4 errors: 2 duplicate node names + 2 duplicate output artifacts
        assert len(errors) == 4
        # Should have errors for both "detect" and "filter"
        error_str = " ".join(errors)
        assert "detect" in error_str
        assert "filter" in error_str

    def test_triple_duplicate_name(self):
        """Test error message for 3+ nodes with same name."""
        validator = GraphValidator()

        node1 = MockDetectNode(name="duplicate")
        node2 = MockDetectNode(name="duplicate")
        node3 = MockDetectNode(name="duplicate")

        errors = validator.check_name_collisions([node1, node2, node3])

        # Should have 2 errors: duplicate node name + duplicate output artifact
        assert len(errors) == 2
        error_str = " ".join(errors)
        assert "Duplicate node name 'duplicate'" in error_str
        assert "3 nodes" in error_str


class TestProviderCapabilities:
    """Tests for provider capability verification."""

    def test_all_providers_available(self):
        """Test validation passes when all providers exist."""
        validator = GraphValidator()

        detect = MockDetectNode(provider_name="detr", name="detect")
        segment = MockSegmentNode(provider_name="sam", name="segment")

        providers = {"detr": {"type": "detector"}, "sam": {"type": "segmenter"}}

        errors = validator.check_provider_capabilities([detect, segment], providers)

        assert len(errors) == 0

    def test_missing_provider(self):
        """Test error when required provider is missing."""
        validator = GraphValidator()

        detect = MockDetectNode(provider_name="yolo", name="detect")

        providers = {
            "detr": {"type": "detector"}
            # "yolo" not available!
        }

        errors = validator.check_provider_capabilities([detect], providers)

        assert len(errors) == 1
        assert "requires provider 'yolo'" in errors[0]
        assert "not available" in errors[0]
        assert "detr" in errors[0]  # Should list available providers

    def test_multiple_missing_providers(self):
        """Test errors for multiple missing providers."""
        validator = GraphValidator()

        detect = MockDetectNode(provider_name="yolo", name="detect")
        segment = MockSegmentNode(provider_name="sam2", name="segment")

        providers = {
            "detr": {"type": "detector"}
            # Both "yolo" and "sam2" missing
        }

        errors = validator.check_provider_capabilities([detect, segment], providers)

        assert len(errors) == 2
        error_str = " ".join(errors)
        assert "yolo" in error_str
        assert "sam2" in error_str

    def test_node_without_provider(self):
        """Test nodes without provider_name are skipped."""
        validator = GraphValidator()

        # Node without provider_name attribute
        filter_node = MockFilterNode(name="filter")

        providers = {}

        errors = validator.check_provider_capabilities([filter_node], providers)

        # Should not error (node doesn't require provider)
        assert len(errors) == 0


class TestIntegratedValidation:
    """Tests for integrated validation (validate method)."""

    def test_valid_graph_full_validation(self):
        """Test full validation passes for valid graph."""
        validator = GraphValidator()

        detect = MockDetectNode(provider_name="detr", name="detect")
        filter_node = MockFilterNode(name="filter")
        segment = MockSegmentNode(provider_name="sam", name="segment")

        wiring = {
            "detect.image": "input.image",
            "filter.detections": "detect.detections",
            "segment.image": "input.image",
            "segment.detections": "filter.detections",
        }

        providers = {"detr": {"type": "detector"}, "sam": {"type": "segmenter"}}

        result = validator.validate(nodes=[detect, filter_node, segment], wiring=wiring, providers=providers)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_invalid_graph_multiple_errors(self):
        """Test full validation collects multiple errors."""
        validator = GraphValidator()

        # Create invalid graph with multiple issues
        detect1 = MockDetectNode(provider_name="missing_provider", name="detect")
        detect2 = MockDetectNode(provider_name="detr", name="detect")  # Duplicate name

        # Missing required input + type mismatch
        wiring = {"detect.image": "nonexistent.output"}  # Wrong source

        providers = {"detr": {}}  # "missing_provider" not available

        result = validator.validate(nodes=[detect1, detect2], wiring=wiring, providers=providers)

        assert result.valid is False
        assert len(result.errors) > 1  # Should have multiple errors

        error_str = " ".join(result.errors)
        assert "Duplicate node name" in error_str  # Name collision
        assert "missing_provider" in error_str  # Missing provider

    def test_validation_with_cycle_error(self):
        """Test validation detects cycle."""
        validator = GraphValidator()

        node_a = MockFilterNode(name="node_a")
        node_b = MockFilterNode(name="node_b")

        # Create cycle
        wiring = {"node_a.detections": "node_b.detections", "node_b.detections": "node_a.detections"}

        result = validator.validate(nodes=[node_a, node_b], wiring=wiring)

        assert result.valid is False
        assert any("Circular dependency" in err for err in result.errors)

    def test_validation_without_providers(self):
        """Test validation works without provider checking."""
        validator = GraphValidator()

        detect = MockDetectNode(provider_name="detr", name="detect")

        wiring = {"detect.image": "input.image"}

        # Don't provide providers (skip provider checking)
        result = validator.validate(nodes=[detect], wiring=wiring)

        # Should still validate structure/types/etc
        assert result.valid is True


class TestPerformance:
    """Tests for validation performance."""

    def test_validation_performance_small_graph(self):
        """Test validation completes quickly for small graph."""
        import time

        validator = GraphValidator()

        # Create 5-node graph
        nodes = [MockDetectNode(name=f"node_{i}") for i in range(5)]

        wiring = {f"node_{i}.image": f"node_{i-1}.detections" if i > 0 else "input.image" for i in range(5)}

        start = time.time()
        validator.validate(nodes=nodes, wiring=wiring)
        elapsed = time.time() - start

        # Should complete in <100ms
        assert elapsed < 0.1
        # Note: May fail if structure doesn't match, but timing should be fast

    def test_validation_performance_medium_graph(self):
        """Test validation completes quickly for medium graph."""
        import time

        validator = GraphValidator()

        # Create 20-node graph
        nodes = [MockDetectNode(name=f"node_{i}") for i in range(20)]

        wiring = {f"node_{i}.image": f"node_{i-1}.detections" if i > 0 else "input.image" for i in range(20)}

        start = time.time()
        validator.validate(nodes=nodes, wiring=wiring)
        elapsed = time.time() - start

        # Should complete in <100ms even for 20 nodes
        assert elapsed < 0.1
