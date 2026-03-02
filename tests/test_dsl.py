"""Tests for DSL helpers.

Tests the ergonomic DSL helpers for graph construction including:
- NodePipe and >> operator
- out() for named outputs
- bind() for input binding
- sequential() for building sequential graphs
- parallel_tasks() for building parallel graphs
- pipeline() for multi-stage pipelines
"""

from unittest.mock import Mock

import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.dsl import (
    NodePipe,
    bind,
    out,
    parallel_tasks,
    pipeline,
    sequential,
)
from mata.core.graph.graph import Graph
from mata.core.graph.node import Node


# Mock nodes for testing
class MockNode(Node):
    """Mock node for testing."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, name=None, output_name="out", **config):
        super().__init__(name=name, **config)
        self.output_name = output_name

    def run(self, ctx, **inputs):
        return {self.output_name: Mock(spec=Detections)}


class MockNodeMultiInput(Node):
    """Mock node with multiple inputs."""

    inputs = {"image": Image, "detections": Detections}
    outputs = {"result": Detections}

    def __init__(self, name=None, output_name="result", **config):
        super().__init__(name=name, **config)
        self.output_name = output_name

    def run(self, ctx, **inputs):
        return {self.output_name: Mock(spec=Detections)}


class TestNodePipe:
    """Test NodePipe class and >> operator."""

    def test_init_with_node(self):
        """Test NodePipe initialization with a node."""
        node = MockNode(name="node1")
        pipe = NodePipe(node)

        assert len(pipe.nodes) == 1
        assert pipe.nodes[0] is node

    def test_init_with_non_node_raises_error(self):
        """Test NodePipe initialization with non-node raises TypeError."""
        with pytest.raises(TypeError, match="NodePipe requires a Node instance"):
            NodePipe("not a node")

    def test_rshift_operator_single(self):
        """Test >> operator with single node."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        pipe = NodePipe(node1) >> node2

        assert len(pipe.nodes) == 2
        assert pipe.nodes[0] is node1
        assert pipe.nodes[1] is node2

    def test_rshift_operator_chain(self):
        """Test >> operator with multiple nodes."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNode(name="node3")

        pipe = NodePipe(node1) >> node2 >> node3

        assert len(pipe.nodes) == 3
        assert pipe.nodes[0] is node1
        assert pipe.nodes[1] is node2
        assert pipe.nodes[2] is node3

    def test_rshift_operator_with_non_node_raises_error(self):
        """Test >> operator with non-node raises TypeError."""
        node = MockNode(name="node1")
        pipe = NodePipe(node)

        with pytest.raises(TypeError, match="Can only pipe Node instances"):
            pipe >> "not a node"

    def test_build_creates_graph(self):
        """Test build() creates a Graph."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        pipe = NodePipe(node1) >> node2
        graph = pipe.build()

        assert isinstance(graph, Graph)
        assert len(graph._nodes) == 2

    def test_build_with_name(self):
        """Test build() with custom graph name."""
        node = MockNode(name="node1")
        pipe = NodePipe(node)

        graph = pipe.build(name="my_pipeline")

        assert graph.name == "my_pipeline"

    def test_build_without_name(self):
        """Test build() without custom name uses default."""
        node = MockNode(name="node1")
        pipe = NodePipe(node)

        graph = pipe.build()

        # Graph uses "untitled_graph" as default when name is None
        assert graph.name == "untitled_graph" or graph.name is None

    def test_repr(self):
        """Test string representation."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        pipe = NodePipe(node1) >> node2
        repr_str = repr(pipe)

        assert "NodePipe" in repr_str
        assert "node1" in repr_str
        assert "node2" in repr_str
        assert ">>" in repr_str


class TestOutHelper:
    """Test out() helper for named outputs."""

    def test_out_sets_output_name(self):
        """Test out() sets output_name attribute."""
        node = MockNode(name="node1", output_name="default")

        result = out(node, "custom_output")

        assert result is node  # Returns the same node
        assert node.output_name == "custom_output"

    def test_out_with_node_without_output_name(self):
        """Test out() with node that doesn't have output_name attribute."""

        # Create a node class without output_name
        class MinimalNode(Node):
            inputs = {}
            outputs = {"result": Detections}

            def run(self, ctx, **inputs):
                return {"result": Mock(spec=Detections)}

        node = MinimalNode(name="minimal")
        result = out(node, "custom")

        assert result is node
        assert hasattr(node, "_dsl_output_name")
        assert node._dsl_output_name == "custom"

    def test_out_with_non_node_raises_error(self):
        """Test out() with non-node raises TypeError."""
        with pytest.raises(TypeError, match="out\\(\\) requires a Node instance"):
            out("not a node", "output")

    def test_out_in_graph_context(self):
        """Test out() helper in graph building."""
        node = MockNode(name="detector")
        node_with_custom_output = out(node, "my_detections")

        Graph().then(node_with_custom_output)

        assert node.output_name == "my_detections"


class TestBindHelper:
    """Test bind() helper for input binding."""

    def test_bind_sets_input_bindings(self):
        """Test bind() sets input bindings."""
        node = MockNode(name="node1")

        result = bind(node, image="input.image", detections="detector.dets")

        assert result is node  # Returns the same node
        assert hasattr(node, "_dsl_input_bindings")
        assert node._dsl_input_bindings == {"image": "input.image", "detections": "detector.dets"}

    def test_bind_with_single_input(self):
        """Test bind() with single input."""
        node = MockNode(name="node1")

        bind(node, image="custom.source")

        assert node._dsl_input_bindings == {"image": "custom.source"}

    def test_bind_multiple_calls(self):
        """Test multiple bind() calls accumulate bindings."""
        node = MockNode(name="node1")

        bind(node, image="source1")
        bind(node, detections="source2")

        assert node._dsl_input_bindings == {"image": "source1", "detections": "source2"}

    def test_bind_with_non_node_raises_error(self):
        """Test bind() with non-node raises TypeError."""
        with pytest.raises(TypeError, match="bind\\(\\) requires a Node instance"):
            bind("not a node", image="source")

    def test_bind_in_graph_context(self):
        """Test bind() helper in graph building."""
        node = MockNodeMultiInput(name="processor")
        bound_node = bind(node, image="img.data", detections="det.output")

        Graph().add(bound_node)

        assert node._dsl_input_bindings == {"image": "img.data", "detections": "det.output"}


class TestSequential:
    """Test sequential() helper."""

    def test_sequential_with_single_node(self):
        """Test sequential() with single node."""
        node = MockNode(name="node1")

        graph = sequential(node)

        assert isinstance(graph, Graph)
        assert len(graph._nodes) == 1
        assert graph._nodes[0] is node

    def test_sequential_with_multiple_nodes(self):
        """Test sequential() with multiple nodes."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNode(name="node3")

        graph = sequential(node1, node2, node3)

        assert len(graph._nodes) == 3
        assert graph._nodes[0] is node1
        assert graph._nodes[1] is node2
        assert graph._nodes[2] is node3

    def test_sequential_with_name(self):
        """Test sequential() with custom graph name."""
        node = MockNode(name="node1")

        graph = sequential(node, name="my_pipeline")

        assert graph.name == "my_pipeline"

    def test_sequential_without_name(self):
        """Test sequential() without name creates default name."""
        node = MockNode(name="node1")

        graph = sequential(node)

        assert graph.name == "sequential_pipeline"

    def test_sequential_with_no_nodes_raises_error(self):
        """Test sequential() with no nodes raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one node"):
            sequential()

    def test_sequential_with_non_node_raises_error(self):
        """Test sequential() with non-node raises TypeError."""
        node = MockNode(name="node1")

        with pytest.raises(TypeError, match="requires Node instances"):
            sequential(node, "not a node")


class TestParallelTasks:
    """Test parallel_tasks() helper."""

    def test_parallel_tasks_with_single_node(self):
        """Test parallel_tasks() with single node."""
        node = MockNode(name="node1")

        graph = parallel_tasks(node, fuse=False)

        assert isinstance(graph, Graph)
        assert len(graph._nodes) == 1

    def test_parallel_tasks_with_multiple_nodes(self):
        """Test parallel_tasks() with multiple nodes."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNode(name="node3")

        graph = parallel_tasks(node1, node2, node3, fuse=False)

        assert len(graph._nodes) == 3

    def test_parallel_tasks_with_name(self):
        """Test parallel_tasks() with custom graph name."""
        node = MockNode(name="node1")

        graph = parallel_tasks(node, name="parallel_graph", fuse=False)

        assert graph.name == "parallel_graph"

    def test_parallel_tasks_without_name(self):
        """Test parallel_tasks() without name creates default name."""
        node = MockNode(name="node1")

        graph = parallel_tasks(node, fuse=False)

        assert graph.name == "parallel_tasks"

    def test_parallel_tasks_with_no_nodes_raises_error(self):
        """Test parallel_tasks() with no nodes raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one node"):
            parallel_tasks()

    def test_parallel_tasks_with_non_node_raises_error(self):
        """Test parallel_tasks() with non-node raises TypeError."""
        node = MockNode(name="node1")

        with pytest.raises(TypeError, match="requires Node instances"):
            parallel_tasks(node, "not a node")

    def test_parallel_tasks_fuse_parameter(self):
        """Test parallel_tasks() fuse parameter (without actual Fuse node)."""
        # Test that fuse parameter is accepted (actual fusion tested separately)
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        # With fuse=True (won't add Fuse node since it's not imported in test)
        graph_with_fuse = parallel_tasks(node1, node2, fuse=True)

        # With fuse=False
        graph_without_fuse = parallel_tasks(node1, node2, fuse=False)

        # Both should succeed
        assert isinstance(graph_with_fuse, Graph)
        assert isinstance(graph_without_fuse, Graph)


class TestPipeline:
    """Test pipeline() helper."""

    def test_pipeline_with_single_stage(self):
        """Test pipeline() with single stage."""
        node = MockNode(name="node1")

        graph = pipeline([[node]])

        assert isinstance(graph, Graph)
        assert len(graph._nodes) == 1

    def test_pipeline_with_multiple_stages(self):
        """Test pipeline() with multiple stages."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNode(name="node3")
        node4 = MockNode(name="node4")

        graph = pipeline([[node1, node2], [node3], [node4]])  # Stage 0: parallel  # Stage 1: single  # Stage 2: single

        assert len(graph._nodes) == 4

    def test_pipeline_with_parallel_stages(self):
        """Test pipeline() with parallel nodes in stages."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNode(name="node3")

        graph = pipeline([[node1, node2], [node3]])  # Parallel stage  # Sequential stage

        assert len(graph._nodes) == 3

    def test_pipeline_with_name(self):
        """Test pipeline() with custom graph name."""
        node = MockNode(name="node1")

        graph = pipeline([[node]], name="my_pipeline")

        assert graph.name == "my_pipeline"

    def test_pipeline_without_name(self):
        """Test pipeline() without name creates default name."""
        node = MockNode(name="node1")

        graph = pipeline([[node]])

        assert graph.name == "multi_stage_pipeline"

    def test_pipeline_with_no_stages_raises_error(self):
        """Test pipeline() with no stages raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one stage"):
            pipeline([])

    def test_pipeline_with_empty_stage_raises_error(self):
        """Test pipeline() with empty stage raises ValueError."""
        node = MockNode(name="node1")

        with pytest.raises(ValueError, match="Stage 1 is empty"):
            pipeline([[node], []])

    def test_pipeline_with_non_list_stage_raises_error(self):
        """Test pipeline() with non-list stage raises TypeError."""
        node = MockNode(name="node1")

        with pytest.raises(TypeError, match="Each stage must be a list"):
            pipeline([[node], node])

    def test_pipeline_with_non_node_raises_error(self):
        """Test pipeline() with non-node raises TypeError."""
        node = MockNode(name="node1")

        with pytest.raises(TypeError, match="requires Node instances"):
            pipeline([[node, "not a node"]])


class TestDSLIntegration:
    """Integration tests combining multiple DSL helpers."""

    def test_pipe_operator_with_out(self):
        """Test combining >> operator with out() helper."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        (NodePipe(out(node1, "custom1")) >> out(node2, "custom2")).build()

        assert node1.output_name == "custom1"
        assert node2.output_name == "custom2"

    def test_pipe_operator_with_bind(self):
        """Test combining >> operator with bind() helper."""
        node1 = MockNode(name="node1")
        node2 = MockNodeMultiInput(name="node2")

        (NodePipe(node1) >> bind(node2, image="source.img", detections="source.dets")).build()

        assert node2._dsl_input_bindings == {"image": "source.img", "detections": "source.dets"}

    def test_sequential_with_out(self):
        """Test sequential() with out() helper."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        sequential(out(node1, "output1"), out(node2, "output2"))

        assert node1.output_name == "output1"
        assert node2.output_name == "output2"

    def test_parallel_tasks_with_bind(self):
        """Test parallel_tasks() with bind() helper."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")

        parallel_tasks(bind(node1, image="img1"), bind(node2, image="img2"), fuse=False)

        assert node1._dsl_input_bindings == {"image": "img1"}
        assert node2._dsl_input_bindings == {"image": "img2"}

    def test_pipeline_with_combined_helpers(self):
        """Test pipeline() with multiple DSL helpers."""
        node1 = MockNode(name="node1")
        node2 = MockNode(name="node2")
        node3 = MockNodeMultiInput(name="node3")

        pipeline([[out(node1, "det1"), out(node2, "det2")], [bind(node3, image="source.img", detections="det1")]])

        assert node1.output_name == "det1"
        assert node2.output_name == "det2"
        assert node3._dsl_input_bindings["detections"] == "det1"


class TestDSLDocumentationExamples:
    """Test that documentation examples work correctly."""

    def test_nodepipe_example(self):
        """Test NodePipe example from docstring."""
        node1 = MockNode(name="detect", output_name="dets")
        node2 = MockNode(name="filter", output_name="filtered")
        node3 = MockNode(name="segment", output_name="masks")

        pipeline_obj = NodePipe(node1) >> node2 >> node3

        graph = pipeline_obj.build()

        assert isinstance(graph, Graph)
        assert len(graph._nodes) == 3

    def test_out_example(self):
        """Test out() example from docstring."""
        detector = out(MockNode(name="Detect"), "my_detections")

        Graph().then(detector)

        assert detector.output_name == "my_detections"

    def test_sequential_example(self):
        """Test sequential() example from docstring."""
        graph = sequential(
            MockNode(name="Detect", output_name="dets"),
            MockNode(name="Filter", output_name="filtered"),
            MockNode(name="TopK", output_name="top5"),
            name="detect_filter_topk",
        )

        assert graph.name == "detect_filter_topk"
        assert len(graph._nodes) == 3

    def test_parallel_tasks_example(self):
        """Test parallel_tasks() example from docstring."""
        graph = parallel_tasks(
            MockNode(name="Detect", output_name="dets"),
            MockNode(name="EstimateDepth", output_name="depth"),
            MockNode(name="Classify", output_name="class"),
            name="multi_task_parallel",
            fuse=False,  # Disabled since Fuse node not available in test
        )

        assert graph.name == "multi_task_parallel"
        assert len(graph._nodes) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
