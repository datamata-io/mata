"""DSL helpers for ergonomic graph construction.

Provides syntactic sugar and helper functions for building graphs with
less boilerplate and more readable code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mata.core.graph.graph import Graph
    from mata.core.graph.node import Node


class NodePipe:
    """Enable node >> node syntax for sequential graph building.

    The NodePipe class wraps nodes and provides the >> operator for
    fluent, readable graph construction. Nodes are automatically chained
    with auto-wiring.

    Example:
        ```python
        from mata.nodes import Detect, Filter, SegmentImage
        from mata.core.graph.dsl import NodePipe

        # Create pipeline with >> operator
        pipeline = (
            NodePipe(Detect(using="detr", out="dets"))
            >> Filter(src="dets", out="filtered", score_gt=0.5)
            >> SegmentImage(using="sam", image="input.image", dets="filtered")
        )

        # Build graph
        graph = pipeline.build()
        ```

    Attributes:
        nodes: List of nodes in the pipeline
    """

    def __init__(self, node: Node):
        """Initialize pipe with first node.

        Args:
            node: First node in the pipeline

        Example:
            ```python
            pipe = NodePipe(Detect(using="detr", out="dets"))
            ```
        """
        from mata.core.graph.node import Node

        if not isinstance(node, Node):
            raise TypeError(f"NodePipe requires a Node instance, got {type(node).__name__}")

        self.nodes: list[Node] = [node]

    def __rshift__(self, other: Node) -> NodePipe:
        """Implement >> operator for chaining nodes.

        Args:
            other: Next node in the pipeline

        Returns:
            Self for continued chaining

        Raises:
            TypeError: If other is not a Node instance

        Example:
            ```python
            pipe = NodePipe(node1) >> node2 >> node3
            ```
        """
        from mata.core.graph.node import Node

        if not isinstance(other, Node):
            raise TypeError(f"Can only pipe Node instances, got {type(other).__name__}")

        self.nodes.append(other)
        return self

    def build(self, name: str | None = None) -> Graph:
        """Build graph from pipe chain.

        Converts the node chain to a Graph with sequential execution
        using auto-wiring between consecutive nodes.

        Args:
            name: Optional graph name

        Returns:
            Graph ready for compilation

        Example:
            ```python
            graph = (NodePipe(node1) >> node2 >> node3).build(name="my_pipeline")
            ```
        """
        from mata.core.graph.graph import Graph

        graph = Graph(name=name)

        # Add all nodes sequentially
        for node in self.nodes:
            graph.then(node)

        return graph

    def __repr__(self) -> str:
        """String representation of pipe."""
        node_names = " >> ".join(n.name for n in self.nodes)
        return f"<NodePipe: {node_names}>"


class _NodeWrapper:
    """Internal wrapper for nodes with DSL metadata.

    Used by out() and bind() helpers to attach metadata to nodes
    before adding them to graphs.
    """

    def __init__(self, node: Node):
        """Initialize wrapper.

        Args:
            node: Node to wrap
        """
        self.node = node
        self._output_name: str | None = None
        self._input_bindings: dict[str, str] = {}

    def set_output_name(self, name: str) -> None:
        """Set custom output name."""
        self._output_name = name

    def set_input_bindings(self, bindings: dict[str, str]) -> None:
        """Set input artifact bindings."""
        self._input_bindings = bindings

    def apply_to_node(self) -> Node:
        """Apply metadata to node and return it.

        Modifies the node's configuration to reflect the DSL metadata.
        """
        # Apply output name if set
        if self._output_name and hasattr(self.node, "output_name"):
            self.node.output_name = self._output_name

        # Apply input bindings if set
        if self._input_bindings:
            # Store bindings in node config for Graph to use
            if not hasattr(self.node, "_dsl_input_bindings"):
                self.node._dsl_input_bindings = {}
            self.node._dsl_input_bindings.update(self._input_bindings)

        return self.node


def out(node: Node, name: str) -> Node:
    """Set output name for node.

    Convenience helper to override the node's default output name.
    Useful when you need specific artifact names for wiring.

    Args:
        node: Node instance
        name: Output artifact name

    Returns:
        Node with modified output name

    Example:
        ```python
        from mata.nodes import Detect
        from mata.core.graph.dsl import out

        # Set custom output name
        detector = out(Detect(using="detr"), "my_detections")

        # Use in graph
        graph = Graph().then(detector)
        # Detections will be available as "my_detections" instead of default
        ```
    """
    from mata.core.graph.node import Node

    if not isinstance(node, Node):
        raise TypeError(f"out() requires a Node instance, got {type(node).__name__}")

    # Modify the node's output_name if it exists
    if hasattr(node, "output_name"):
        node.output_name = name
    else:
        # Store in config for nodes that use different attributes
        if not hasattr(node, "_dsl_output_name"):
            node._dsl_output_name = name

    return node


def bind(node: Node, **inputs: str) -> Node:
    """Bind inputs to artifact names.

    Explicitly specify where each node input should come from.
    Overrides auto-wiring for finer control.

    Args:
        node: Node instance
        **inputs: Keyword arguments mapping input names to artifact sources
                 Format: input_name="artifact_source"

    Returns:
        Node with input bindings

    Example:
        ```python
        from mata.nodes import SegmentImage
        from mata.core.graph.dsl import bind

        # Explicitly bind inputs
        segmenter = bind(
            SegmentImage(using="sam"),
            image="input.image",
            dets="detector.detections"
        )

        # Use in graph
        graph = Graph().add(segmenter)
        ```
    """
    from mata.core.graph.node import Node

    if not isinstance(node, Node):
        raise TypeError(f"bind() requires a Node instance, got {type(node).__name__}")

    # Store bindings in node for Graph to use during wiring
    if not hasattr(node, "_dsl_input_bindings"):
        node._dsl_input_bindings = {}
    node._dsl_input_bindings.update(inputs)

    return node


def sequential(*nodes: Node, name: str | None = None) -> Graph:
    """Build sequential graph from nodes.

    Convenience function for creating a simple sequential pipeline.
    Nodes are executed in order with auto-wiring.

    Args:
        *nodes: Variable number of nodes to chain
        name: Optional graph name

    Returns:
        Graph with nodes in sequence

    Example:
        ```python
        from mata.nodes import Detect, Filter, TopK
        from mata.core.graph.dsl import sequential

        # Create sequential pipeline
        graph = sequential(
            Detect(using="detr", out="dets"),
            Filter(src="dets", out="filtered", score_gt=0.5),
            TopK(src="filtered", out="top5", k=5),
            name="detect_filter_topk"
        )

        # Ready for compilation
        compiled = graph.compile(providers)
        ```
    """
    from mata.core.graph.graph import Graph
    from mata.core.graph.node import Node

    # Validate inputs
    for i, node in enumerate(nodes):
        if not isinstance(node, Node):
            raise TypeError(f"sequential() requires Node instances, got {type(node).__name__} " f"at position {i}")

    if not nodes:
        raise ValueError("sequential() requires at least one node")

    # Build graph
    graph = Graph(name=name or "sequential_pipeline")

    for node in nodes:
        graph.then(node)

    return graph


def parallel_tasks(*nodes: Node, name: str | None = None, fuse: bool = True) -> Graph:
    """Build parallel graph with optional fusion.

    Creates a graph where nodes execute in parallel (no dependencies).
    Optionally adds a Fuse node at the end to bundle results.

    Args:
        *nodes: Variable number of nodes to execute in parallel
        name: Optional graph name
        fuse: If True, adds Fuse node to bundle results into MultiResult

    Returns:
        Graph with parallel execution

    Example:
        ```python
        from mata.nodes import Detect, EstimateDepth, Classify
        from mata.core.graph.dsl import parallel_tasks

        # Parallel tasks with auto-fusion
        graph = parallel_tasks(
            Detect(using="detr", out="dets"),
            EstimateDepth(using="depth_anything", out="depth"),
            Classify(using="resnet50", out="class"),
            name="multi_task_parallel",
            fuse=True
        )

        # All tasks run in parallel, then fused into MultiResult
        ```
    """
    from mata.core.graph.graph import Graph
    from mata.core.graph.node import Node

    # Validate inputs
    for i, node in enumerate(nodes):
        if not isinstance(node, Node):
            raise TypeError(f"parallel_tasks() requires Node instances, got {type(node).__name__} " f"at position {i}")

    if not nodes:
        raise ValueError("parallel_tasks() requires at least one node")

    # Build graph
    graph = Graph(name=name or "parallel_tasks")

    # Add nodes in parallel
    graph.parallel(list(nodes))

    # Optionally add fusion node
    if fuse:
        try:
            from mata.nodes import Fuse

            # Collect output names from all nodes
            channel_sources = {}
            for node in nodes:
                if hasattr(node, "output_name"):
                    output_name = node.output_name
                    channel_sources[output_name] = f"{node.name}.{output_name}"

            # Remove 'out' key if present to avoid collision with Fuse's own parameter
            fuse_out = "result"
            if "out" in channel_sources:
                # Rename the channel to avoid clashing with Fuse(out=...) parameter
                channel_sources["out_channel"] = channel_sources.pop("out")

            # Add fuse node
            if channel_sources:
                fuse_node = Fuse(out=fuse_out, **channel_sources)
                graph.then(fuse_node)

        except ImportError:
            # Fuse node not available yet - skip fusion
            pass

    return graph


def pipeline(stages: list[list[Node]], name: str | None = None) -> Graph:
    """Build multi-stage pipeline.

    Creates a graph with multiple execution stages. Nodes within each stage
    execute in parallel, while stages execute sequentially.

    Args:
        stages: List of stages, where each stage is a list of parallel nodes
        name: Optional graph name

    Returns:
        Graph with staged execution

    Example:
        ```python
        from mata.nodes import Detect, EstimateDepth, Filter, TopK, SegmentImage
        from mata.core.graph.dsl import pipeline

        # Multi-stage pipeline
        graph = pipeline([
            # Stage 0: Parallel detection and depth
            [
                Detect(using="detr", out="dets"),
                EstimateDepth(using="depth_anything", out="depth")
            ],

            # Stage 1: Filter detections
            [
                Filter(src="dets", out="filtered", score_gt=0.5)
            ],

            # Stage 2: Parallel top-k and segmentation
            [
                TopK(src="filtered", out="top5", k=5),
                SegmentImage(using="sam", dets="filtered", out="masks")
            ]
        ], name="multi_stage_pipeline")
        ```
    """
    from mata.core.graph.graph import Graph
    from mata.core.graph.node import Node

    # Validate inputs
    if not stages:
        raise ValueError("pipeline() requires at least one stage")

    for stage_idx, stage in enumerate(stages):
        if not isinstance(stage, list):
            raise TypeError(f"Each stage must be a list of nodes, got {type(stage).__name__} " f"at stage {stage_idx}")

        if not stage:
            raise ValueError(f"Stage {stage_idx} is empty")

        for node_idx, node in enumerate(stage):
            if not isinstance(node, Node):
                raise TypeError(
                    f"pipeline() requires Node instances, got {type(node).__name__} "
                    f"at stage {stage_idx}, position {node_idx}"
                )

    # Build graph
    graph = Graph(name=name or "multi_stage_pipeline")

    # Process each stage
    for stage in stages:
        if len(stage) == 1:
            # Single node - add with then()
            graph.then(stage[0])
        else:
            # Multiple nodes - add in parallel
            graph.parallel(stage)

    return graph


# Convenience exports
__all__ = [
    "NodePipe",
    "out",
    "bind",
    "sequential",
    "parallel_tasks",
    "pipeline",
]
