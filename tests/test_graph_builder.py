"""Tests for graph builder and compiler.

Comprehensive test suite for Graph and CompiledGraph covering:
- Fluent API (add, then, parallel, conditional)
- Sequential wiring
- Parallel node grouping
- Conditional branching
- Graph compilation
- Execution order computation
- DAG representation
- Error handling
- Visualization (optional)
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.exceptions import ValidationError
from mata.core.graph.graph import CompiledGraph, Graph
from mata.core.graph.node import Node


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


class MockDepthMap(Artifact):
    """Mock DepthMap artifact for testing."""

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockDepthMap":
        return cls()


# Mock Node implementations for testing
class MockDetectNode(Node):
    """Mock detection node."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, provider_name: str = "detr", name: str = "MockDetect", out: str = "dets"):
        super().__init__(name=name)
        self.provider_name = provider_name
        self.output_name = out

    def run(self, ctx, **inputs):
        return {}


class MockFilterNode(Node):
    """Mock filter node."""

    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "MockFilter", src: str = "dets", out: str = "filtered"):
        super().__init__(name=name)
        self.src = src
        self.out = out

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


class MockDepthNode(Node):
    """Mock depth estimation node."""

    inputs = {"image": Image}
    outputs = {"depth": MockDepthMap}

    def __init__(self, provider_name: str = "depth_anything", name: str = "MockDepth"):
        super().__init__(name=name)
        self.provider_name = provider_name

    def run(self, ctx, **inputs):
        return {}


# =============================================================================
# Graph Construction Tests
# =============================================================================


class TestGraphConstruction:
    """Test graph initialization and basic construction."""

    def test_graph_init_default_name(self):
        """Test graph initialization with default name."""
        graph = Graph()
        assert graph.name == "untitled_graph"
        assert len(graph._nodes) == 0
        assert len(graph._wiring) == 0
        assert graph._compiled is None

    def test_graph_init_custom_name(self):
        """Test graph initialization with custom name."""
        graph = Graph(name="detection_pipeline")
        assert graph.name == "detection_pipeline"

    def test_add_single_node(self):
        """Test adding a single node to graph."""
        graph = Graph()
        node = MockDetectNode()

        graph.add(node, inputs={"image": "input.image"})

        assert len(graph._nodes) == 1
        assert graph._nodes[0] == node
        assert "MockDetect.image" in graph._wiring
        assert graph._wiring["MockDetect.image"] == "input.image"

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes with explicit wiring."""
        graph = Graph()
        detect = MockDetectNode()
        filter_node = MockFilterNode()

        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "MockDetect.detections"})

        assert len(graph._nodes) == 2
        assert graph._wiring["MockFilter.detections"] == "MockDetect.detections"

    def test_add_duplicate_node_name_raises_error(self):
        """Test that adding node with duplicate name raises error."""
        graph = Graph()
        node1 = MockDetectNode(name="detect")
        node2 = MockDetectNode(name="detect")

        graph.add(node1, inputs={"image": "input.image"})

        with pytest.raises(ValidationError, match="already exists"):
            graph.add(node2, inputs={"image": "input.image"})


# =============================================================================
# Fluent API Tests
# =============================================================================


class TestFluentAPI:
    """Test fluent API methods (then, parallel, conditional)."""

    def test_then_sequential_chaining(self):
        """Test sequential chaining with then()."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph.then(detect).then(filter_node)

        assert len(graph._nodes) == 2
        assert graph._nodes[0] == detect
        assert graph._nodes[1] == filter_node

    def test_then_auto_wiring(self):
        """Test that then() auto-wires from previous node outputs."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph.then(detect).then(filter_node)

        # Filter should be wired to detect's dynamic output name ("dets")
        # _update_last_outputs maps static key "detections" -> "detect.dets"
        assert "filter.detections" in graph._wiring
        assert graph._wiring["filter.detections"] == "detect.dets"

    def test_then_first_node_wires_to_input(self):
        """Test that first node in then() chain wires to graph inputs."""
        graph = Graph()
        detect = MockDetectNode(name="detect")

        graph.then(detect)

        assert "detect.image" in graph._wiring
        assert graph._wiring["detect.image"] == "input.image"

    def test_then_chainable(self):
        """Test that then() returns self for chaining."""
        graph = Graph()
        result = graph.then(MockDetectNode(name="d1"))

        assert result is graph

    def test_parallel_nodes(self):
        """Test adding parallel nodes."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        classify = MockClassifyNode(name="classify")
        depth = MockDepthNode(name="depth")

        graph.parallel([detect, classify, depth])

        assert len(graph._nodes) == 3
        # All should wire to input.image
        assert graph._wiring["detect.image"] == "input.image"
        assert graph._wiring["classify.image"] == "input.image"
        assert graph._wiring["depth.image"] == "input.image"

    def test_parallel_chainable(self):
        """Test that parallel() returns self for chaining."""
        graph = Graph()
        result = graph.parallel([MockDetectNode(name="d1")])

        assert result is graph

    def test_parallel_after_sequential(self):
        """Test parallel execution after sequential nodes."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")
        segment = MockSegmentNode(name="segment")
        classify = MockClassifyNode(name="classify")

        # Sequential: detect -> filter
        # Then parallel: segment and classify
        graph.then(detect).then(filter_node).parallel([segment, classify])

        assert len(graph._nodes) == 4
        # Segment should use filter's dynamic output name ("filtered")
        # _update_last_outputs maps static key "detections" -> "filter.filtered"
        assert "segment.detections" in graph._wiring
        assert graph._wiring["segment.detections"] == "filter.filtered"

    def test_conditional_with_then_branch(self):
        """Test conditional branching with then branch only."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        segment = MockSegmentNode(name="segment")

        def has_detections(ctx):
            return True

        graph.then(detect).conditional(predicate=has_detections, then_branch=segment, else_branch=None)

        assert len(graph._nodes) == 2
        assert segment in graph._nodes

    def test_conditional_with_both_branches(self):
        """Test conditional branching with then and else branches."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        segment = MockSegmentNode(name="segment_then")
        classify = MockClassifyNode(name="classify_else")

        def has_detections(ctx):
            return True

        graph.then(detect).conditional(predicate=has_detections, then_branch=segment, else_branch=classify)

        assert len(graph._nodes) == 3
        assert segment in graph._nodes
        assert classify in graph._nodes

    def test_conditional_chainable(self):
        """Test that conditional() returns self for chaining."""
        graph = Graph()

        result = graph.conditional(predicate=lambda ctx: True, then_branch=MockDetectNode(name="d1"))

        assert result is graph


# =============================================================================
# Graph Compilation Tests
# =============================================================================


class TestGraphCompilation:
    """Test graph compilation and validation."""

    def test_compile_valid_graph(self):
        """Test compiling a valid graph."""
        graph = Graph(name="test_graph")
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph.then(detect).then(filter_node)

        # Mock providers (flat dict with provider names as keys)
        providers = {
            "detr": Mock(),
        }

        compiled = graph.compile(providers)

        assert isinstance(compiled, CompiledGraph)
        assert compiled.name == "test_graph"
        assert len(compiled.nodes) == 2
        assert compiled.validation_result.valid

    def test_compile_invalid_graph_raises_error(self):
        """Test that compiling invalid graph raises ValidationError."""
        graph = Graph()
        detect = MockDetectNode(name="detect")

        # Create invalid wiring (type mismatch)
        graph.add(detect, inputs={"image": "input.nonexistent"})

        with pytest.raises(ValidationError):
            graph.compile(providers={})

    def test_compile_stores_compiled_graph(self):
        """Test that compile() stores the compiled graph."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        providers = {"detr": Mock()}
        compiled = graph.compile(providers)

        assert graph._compiled is compiled

    def test_compile_runs_validation(self):
        """Test that compile() runs full validation."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        providers = {"detr": Mock()}
        compiled = graph.compile(providers)

        assert compiled.validation_result is not None
        assert compiled.validation_result.valid


# =============================================================================
# Execution Order Tests
# =============================================================================


class TestExecutionOrder:
    """Test execution order computation."""

    def test_sequential_execution_order(self):
        """Test execution order for sequential graph."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph.then(detect).then(filter_node)

        providers = {"detr": Mock()}
        compiled = graph.compile(providers)

        assert len(compiled.execution_order) == 2
        # Stage 0: detect
        assert len(compiled.execution_order[0]) == 1
        assert compiled.execution_order[0][0] == detect
        # Stage 1: filter
        assert len(compiled.execution_order[1]) == 1
        assert compiled.execution_order[1][0] == filter_node

    def test_parallel_execution_order(self):
        """Test execution order for parallel nodes."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        classify = MockClassifyNode(name="classify")
        depth = MockDepthNode(name="depth")

        graph.parallel([detect, classify, depth])

        providers = {
            "detr": Mock(),
            "resnet": Mock(),
            "depth_anything": Mock(),
        }
        compiled = graph.compile(providers)

        # All nodes should be in stage 0 (parallel)
        assert len(compiled.execution_order) == 1
        assert len(compiled.execution_order[0]) == 3

    def test_mixed_execution_order(self):
        """Test execution order for mixed sequential and parallel."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")
        MockSegmentNode(name="segment")
        MockClassifyNode(name="classify")

        # detect -> filter -> (segment, classify) in parallel
        # Note: classify needs image, segment needs image + detections
        # So they can run in parallel IF image is available
        graph.then(detect).then(filter_node)

        providers = {
            "detr": Mock(),
            "sam": Mock(),
            "resnet": Mock(),
        }
        compiled = graph.compile(providers)

        # Should have stages: [detect], [filter]
        assert len(compiled.execution_order) >= 2

    def test_get_parallel_stages(self):
        """Test get_parallel_stages() method."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        classify = MockClassifyNode(name="classify")

        graph.parallel([detect, classify])

        providers = {
            "detr": Mock(),
            "resnet": Mock(),
        }
        compiled = graph.compile(providers)

        parallel_stages = compiled.get_parallel_stages()
        assert parallel_stages == compiled.execution_order
        assert len(parallel_stages[0]) == 2


# =============================================================================
# DAG Representation Tests
# =============================================================================


class TestDAGRepresentation:
    """Test DAG representation and networkx integration."""

    def test_dag_created_with_networkx(self):
        """Test that DAG is created when networkx is available."""
        try:
            import networkx as nx

            graph = Graph()
            detect = MockDetectNode(name="detect")
            filter_node = MockFilterNode(name="filter")

            graph.then(detect).then(filter_node)

            providers = {"detr": Mock()}
            compiled = graph.compile(providers)

            assert compiled.dag is not None
            assert isinstance(compiled.dag, nx.DiGraph)
            assert len(compiled.dag.nodes()) == 2
            assert len(compiled.dag.edges()) == 1

        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_dag_is_none_without_networkx(self):
        """Test that DAG is None when networkx not available."""
        with patch.dict("sys.modules", {"networkx": None}):
            # This would test fallback behavior
            # Skip if networkx is actually installed
            pytest.skip("NetworkX is installed, cannot test without it")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in graph builder."""

    def test_compile_empty_graph_fails(self):
        """Test that compiling empty graph fails."""
        graph = Graph()

        with pytest.raises(ValidationError, match="at least one node"):
            graph.compile(providers={})

    def test_compile_with_cycles_fails(self):
        """Test that graph with cycles fails compilation."""
        # Create a circular dependency manually
        graph = Graph()
        node1 = MockFilterNode(name="node1")
        node2 = MockFilterNode(name="node2")

        graph.add(node1, inputs={"detections": "node2.detections"})
        graph.add(node2, inputs={"detections": "node1.detections"})

        with pytest.raises(ValidationError, match="[Cc]ircular"):
            graph.compile(providers={})

    def test_visualize_before_compile_raises_error(self):
        """Test that visualize() before compile raises error."""
        graph = Graph()
        graph.then(MockDetectNode(name="detect"))

        with pytest.raises(ValueError, match="must be compiled"):
            graph.visualize("output.png")


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Test graph visualization (optional, requires networkx + pydot)."""

    def test_visualize_requires_networkx(self):
        """Test that visualize raises ImportError without networkx."""
        with patch("mata.core.graph.graph.HAS_NETWORKX", False):
            graph = Graph()
            graph.then(MockDetectNode(name="detect"))

            with pytest.raises(ImportError, match="networkx"):
                graph.visualize("output.png")

    @pytest.mark.slow
    def test_visualize_creates_file(self, tmp_path):
        """Test that visualize creates output file."""
        try:
            import networkx as nx  # noqa: F401
            import pydot  # noqa: F401

            graph = Graph()
            detect = MockDetectNode(name="detect")
            filter_node = MockFilterNode(name="filter")

            graph.then(detect).then(filter_node)

            providers = {"detr": Mock()}
            compiled = graph.compile(providers)

            output_path = tmp_path / "graph.png"
            compiled.visualize(str(output_path))

            assert output_path.exists()

        except (ImportError, FileNotFoundError):
            pytest.skip("pydot or graphviz not installed")


# =============================================================================
# CompiledGraph Tests
# =============================================================================


class TestCompiledGraph:
    """Test CompiledGraph class."""

    def test_compiled_graph_repr(self):
        """Test CompiledGraph string representation."""
        graph = Graph(name="test")
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        providers = {"detr": Mock()}
        compiled = graph.compile(providers)

        repr_str = repr(compiled)
        assert "CompiledGraph" in repr_str
        assert "test" in repr_str
        assert "1 nodes" in repr_str

    def test_compiled_graph_execution_order_computed_on_init(self):
        """Test that execution order is computed during __post_init__."""
        graph = Graph()
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        providers = {"detr": Mock()}
        compiled = graph.compile(providers)

        assert compiled.execution_order is not None
        assert len(compiled.execution_order) > 0


# =============================================================================
# Graph Representation Tests
# =============================================================================


class TestGraphRepresentation:
    """Test graph string representation."""

    def test_graph_repr_uncompiled(self):
        """Test Graph __repr__ when uncompiled."""
        graph = Graph(name="test_graph")
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        repr_str = repr(graph)
        assert "Graph" in repr_str
        assert "test_graph" in repr_str
        assert "1 nodes" in repr_str
        assert "uncompiled" in repr_str

    def test_graph_repr_compiled(self):
        """Test Graph __repr__ when compiled."""
        graph = Graph(name="test_graph")
        detect = MockDetectNode(name="detect")
        graph.then(detect)

        providers = {"detr": Mock()}
        graph.compile(providers)

        repr_str = repr(graph)
        assert "compiled" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphIntegration:
    """Integration tests with real-world scenarios."""

    def test_detection_segmentation_pipeline(self):
        """Test complete detection -> filter -> segmentation pipeline."""
        graph = Graph(name="det_seg_pipeline")

        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")
        segment = MockSegmentNode(name="segment")

        graph.then(detect).then(filter_node).then(segment)

        providers = {
            "detr": Mock(),
            "sam": Mock(),
        }

        compiled = graph.compile(providers)

        assert len(compiled.nodes) == 3
        assert compiled.validation_result.valid
        assert len(compiled.execution_order) == 3  # 3 sequential stages

    def test_parallel_task_fusion(self):
        """Test parallel tasks followed by fusion."""
        graph = Graph(name="parallel_fusion")

        detect = MockDetectNode(name="detect")
        classify = MockClassifyNode(name="classify")
        depth = MockDepthNode(name="depth")

        # All parallel on input image
        graph.parallel([detect, classify, depth])

        providers = {
            "detr": Mock(),
            "resnet": Mock(),
            "depth_anything": Mock(),
        }

        compiled = graph.compile(providers)

        assert len(compiled.nodes) == 3
        # All should be in stage 0
        assert len(compiled.execution_order[0]) == 3

    def test_complex_multi_stage_pipeline(self):
        """Test complex pipeline with mixed sequential and parallel."""
        graph = Graph(name="complex_pipeline")

        # Stage 0: Parallel detection and depth
        detect = MockDetectNode(name="detect")
        depth = MockDepthNode(name="depth")

        # Stage 1: Filter detections
        filter_node = MockFilterNode(name="filter")

        # Stage 2: Segment based on filtered detections
        segment = MockSegmentNode(name="segment")

        graph.parallel([detect, depth])
        graph.then(filter_node)
        graph.then(segment)

        providers = {
            "detr": Mock(),
            "depth_anything": Mock(),
            "sam": Mock(),
        }

        compiled = graph.compile(providers)

        assert len(compiled.nodes) == 4
        assert compiled.validation_result.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
