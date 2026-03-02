"""Graph builder and compiler.

Provides fluent API for building and compiling task graphs with validation,
DAG representation, and execution order computation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.graph.node import Node

from mata.core.exceptions import ValidationError
from mata.core.graph.validator import GraphValidator, ValidationResult

# Optional networkx for DAG and visualization
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class Graph:
    """Fluent graph builder with validation and compilation.

    Graph provides a fluent API for constructing multi-task computer vision workflows.
    It supports:
    - Sequential chaining with auto-wiring (then)
    - Explicit input wiring (add)
    - Parallel node execution (parallel)
    - Conditional branching (conditional)
    - Graph compilation with validation
    - DAG representation and visualization

    The graph builder tracks nodes and their connections (wiring), then compiles
    to a validated, executable CompiledGraph with optimized execution order.

    Example:
        ```python
        from mata.core.graph import Graph
        from mata.nodes import Detect, Filter, SegmentImage

        # Sequential workflow
        graph = (Graph()
            .then(Detect(using="detr", out="dets"))
            .then(Filter(src="dets", out="filtered", score_gt=0.5))
            .then(SegmentImage(using="sam", image="input.image", dets="filtered"))
        )

        # Compile with providers
        compiled = graph.compile(providers={
            "detect": {"detr": detr_adapter},
            "segment": {"sam": sam_adapter}
        })

        # Parallel execution
        graph = Graph().parallel([
            Detect(using="detr", out="dets"),
            EstimateDepth(using="depth_anything", out="depth")
        ])
        ```

    Attributes:
        name: Human-readable graph name
    """

    def __init__(self, name: str | None = None):
        """Initialize graph builder.

        Args:
            name: Optional human-readable graph name (defaults to "untitled_graph")

        Example:
            ```python
            graph = Graph(name="detection_segmentation_pipeline")
            ```
        """
        self.name = name or "untitled_graph"
        self._nodes: list[Node] = []
        self._wiring: dict[str, str] = {}  # {node.input: artifact_name}
        self._compiled: CompiledGraph | None = None
        self._last_outputs: dict[str, str] = {}  # Track last node's outputs for auto-wiring
        self._conditionals: list[tuple[Callable, Node, Node | None]] = []  # For future conditional support

    def add(self, node: Node, inputs: dict[str, str] | None = None) -> Graph:
        """Add node with explicit input wiring.

        Adds a node to the graph with explicit specification of where each input
        should come from. Input sources can reference:
        - Other node outputs: "node_name.output_name"
        - Graph inputs: "input.artifact_name"

        Args:
            node: Node instance to add
            inputs: Dictionary mapping node input names to artifact sources.
                   If None, attempts to auto-wire from previous node outputs.
                   Format: {input_name: artifact_source}

        Returns:
            Self for method chaining

        Raises:
            ValidationError: If node name conflicts with existing node

        Example:
            ```python
            # Explicit wiring
            graph = Graph()
            graph.add(Detect(using="detr", out="dets"), inputs={"image": "input.image"})
            graph.add(Filter(src="dets", out="filtered"), inputs={"detections": "Detect.dets"})

            # Auto-wiring (if previous node output matches)
            graph.add(Detect(using="detr", out="dets"), inputs={"image": "input.image"})
            graph.add(Filter(src="dets", out="filtered"))  # Auto-wires from Detect.dets
            ```
        """
        # Check for name collision
        if any(n.name == node.name for n in self._nodes):
            raise ValidationError(
                f"Node name '{node.name}' already exists in graph. "
                f"Use unique names or pass name parameter: Node(name='unique_name')"
            )

        self._nodes.append(node)

        # Wire inputs
        if inputs:
            for input_name, artifact_source in inputs.items():
                target = f"{node.name}.{input_name}"
                self._wiring[target] = artifact_source
        else:
            # Attempt auto-wiring from last outputs
            self._auto_wire_node(node)

        # Track this node's outputs for next auto-wiring
        self._update_last_outputs(node)

        return self

    def then(self, node: Node) -> Graph:
        """Add node in sequence (auto-wire from previous).

        Convenience method for sequential workflows. Automatically wires the node's
        inputs to the previous node's outputs when names match, or to any available
        artifact in the last output set.

        Args:
            node: Node instance to add in sequence

        Returns:
            Self for method chaining

        Example:
            ```python
            graph = (Graph()
                .then(Detect(using="detr", out="dets"))          # First node
                .then(Filter(src="dets", out="filtered"))        # Auto-wired
                .then(TopK(src="filtered", out="top5", k=5))     # Auto-wired
            )
            ```
        """
        return self.add(node, inputs=None)

    def parallel(self, nodes: list[Node]) -> Graph:
        """Add nodes to execute in parallel.

        Adds multiple nodes that can execute simultaneously (no dependencies
        between them). Each node must have its inputs explicitly wired or
        auto-wired from previous stages.

        Parallel nodes are tagged in the compiled graph for scheduler optimization,
        allowing multi-threaded or asynchronous execution.

        Args:
            nodes: List of nodes to execute in parallel

        Returns:
            Self for method chaining

        Example:
            ```python
            # Parallel detection and depth estimation
            graph = Graph().parallel([
                Detect(using="detr", out="dets"),
                EstimateDepth(using="depth_anything", out="depth")
            ])

            # Both nodes operate on input.image in parallel
            ```
        """
        # Add all nodes with current auto-wiring state
        for node in nodes:
            self.add(node, inputs=None)

        return self

    def conditional(self, predicate: Callable, then_branch: Node, else_branch: Node | None = None) -> Graph:
        """Add conditional execution.

        Adds a conditional branch to the graph. The predicate function is evaluated
        at runtime to determine which branch to execute.

        Note: Full conditional support requires runtime evaluation during execution.
              This method stores the conditional for compilation but requires
              scheduler support (planned for future releases).

        Args:
            predicate: Callable that returns bool, evaluated at runtime
            then_branch: Node to execute if predicate returns True
            else_branch: Optional node to execute if predicate returns False

        Returns:
            Self for method chaining

        Example:
            ```python
            def has_detections(ctx):
                dets = ctx.retrieve("dets")
                return len(dets.instances) > 0

            graph = (Graph()
                .then(Detect(using="detr", out="dets"))
                .conditional(
                    predicate=has_detections,
                    then_branch=SegmentImage(using="sam", dets="dets"),
                    else_branch=None  # Skip segmentation if no detections
                )
            )
            ```
        """
        # Store conditional for future implementation
        self._conditionals.append((predicate, then_branch, else_branch))

        # For now, add the then_branch unconditionally
        # TODO: Full conditional support in scheduler (v2.0)
        self.add(then_branch, inputs=None)

        if else_branch:
            # Also add else branch (scheduler should handle exclusivity)
            self.add(else_branch, inputs=None)

        return self

    def compile(self, providers: dict[str, Any]) -> CompiledGraph:
        """Validate and compile to executable DAG.

        Performs comprehensive validation (type checking, dependency resolution,
        cycle detection) and compiles the graph into an optimized DAG structure
        with execution order computed.

        Args:
            providers: Provider registry for capability checking
                      Format: {capability: {name: provider}}

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValidationError: If graph validation fails

        Example:
            ```python
            compiled = graph.compile(providers={
                "detect": {"detr": detr_adapter},
                "segment": {"sam": sam_adapter}
            })

            # Now ready for execution
            result = scheduler.execute(compiled, ctx, initial_artifacts)
            ```
        """
        # Validate graph
        validator = GraphValidator()
        result = validator.validate(nodes=self._nodes, wiring=self._wiring, providers=providers)

        # Raise if invalid
        result.raise_if_invalid()

        # Build DAG
        if HAS_NETWORKX:
            dag = self._build_networkx_dag()
        else:
            dag = None

        # Create compiled graph
        compiled = CompiledGraph(
            name=self.name, nodes=self._nodes, wiring=self._wiring, dag=dag, validation_result=result
        )

        self._compiled = compiled
        return compiled

    def visualize(self, output_path: str) -> None:
        """Generate graph visualization (DOT format).

        Creates a visual representation of the graph using GraphViz DOT format.
        Nodes are shown with their types, edges represent data flow.

        Requires: networkx and pygraphviz/pydot

        Args:
            output_path: Path to save visualization (e.g., "graph.png", "graph.pdf")

        Raises:
            ImportError: If networkx is not installed
            ValueError: If graph has not been compiled yet

        Example:
            ```python
            graph.compile(providers)
            graph.visualize("pipeline.png")
            ```
        """
        if not HAS_NETWORKX:
            raise ImportError("Graph visualization requires networkx. Install with: pip install networkx")

        if self._compiled is None:
            raise ValueError("Graph must be compiled before visualization. Call graph.compile() first.")

        self._compiled.visualize(output_path)

    # --- Private methods ---

    def _auto_wire_node(self, node: Node) -> None:
        """Attempt to auto-wire node inputs from last outputs.

        Matches node input names with available artifact names from previous nodes.
        """
        if not self._last_outputs:
            # First node - wire to graph inputs
            for input_name in node.inputs.keys():
                target = f"{node.name}.{input_name}"
                self._wiring[target] = f"input.{input_name}"
        else:
            # Wire from last outputs
            for input_name in node.inputs.keys():
                target = f"{node.name}.{input_name}"

                # Try to find matching output
                if input_name in self._last_outputs:
                    self._wiring[target] = self._last_outputs[input_name]
                else:
                    # Fallback: wire to input if not found
                    self._wiring[target] = f"input.{input_name}"

    def _update_last_outputs(self, node: Node) -> None:
        """Update tracking of last node's outputs for auto-wiring.

        When a node uses a dynamic output name (via ``output_name`` or ``out``
        attribute), those names are used instead of the static class-level
        ``outputs`` dict keys.  This ensures auto-wiring resolves correctly
        for nodes like ``Detect(using="detr", out="dets")``.
        """
        # Collect the dynamic output name(s) the node will actually produce
        dynamic_name = getattr(node, "output_name", None) or getattr(node, "out", None)

        if dynamic_name is not None and len(node.outputs) == 1:
            # Single-output node with a dynamic name
            # Map the static type key to the dynamic artifact name
            static_key = next(iter(node.outputs.keys()))
            artifact_name = f"{node.name}.{dynamic_name}"
            # Store under *both* the static type key and the dynamic name
            # so downstream nodes can match by either convention.
            self._last_outputs[static_key] = artifact_name
            if dynamic_name != static_key:
                self._last_outputs[dynamic_name] = artifact_name
        else:
            # Multi-output or no dynamic name — use class-level keys
            for output_name in node.outputs.keys():
                artifact_name = f"{node.name}.{output_name}"
                self._last_outputs[output_name] = artifact_name

    def _build_networkx_dag(self) -> nx.DiGraph:
        """Build NetworkX directed graph from nodes and wiring."""
        dag = nx.DiGraph()

        # Add nodes
        for node in self._nodes:
            dag.add_node(node.name, node=node)

        # Add edges from wiring
        for target, source in self._wiring.items():
            # Parse target: node_name.input_name
            if "." in target:
                target_node, _ = target.rsplit(".", 1)
            else:
                continue

            # Parse source: node_name.output_name or input.artifact_name
            if "." in source:
                source_node, _ = source.rsplit(".", 1)
                if source_node != "input" and source_node in [n.name for n in self._nodes]:
                    # Add edge between nodes
                    dag.add_edge(source_node, target_node)

        return dag

    def run(
        self,
        image: Any,
        providers: dict[str, Any],
        *,
        scheduler: Any | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Execute this graph on an image (convenience wrapper).

        Thin delegation layer over ``mata.infer()``. Allows fluent execution
        directly from the graph builder without importing the top-level API.

        Args:
            image: Input image. Accepts:
                - ``str`` or ``Path``: file path to an image on disk
                - ``PIL.Image.Image``: a Pillow image object
                - ``np.ndarray``: a numpy array (HWC, uint8, RGB or BGR)
            providers: Provider instances keyed by name.
                Keys must match the ``using`` parameter of nodes in the graph.
                Values are loaded adapters (e.g. from ``mata.load()``).
                Accepts flat ``{"name": adapter}`` or nested
                ``{"capability": {"name": adapter}}``.
            scheduler: Optional scheduler instance for execution strategy.
                Defaults to ``SyncScheduler`` (sequential execution).
                Pass ``ParallelScheduler()`` for concurrent independent stages.
            device: Device placement. One of ``"auto"``, ``"cuda"``, ``"cpu"``.
            **kwargs: Additional keyword arguments forwarded to ``mata.infer()``.

        Returns:
            MultiResult with all task outputs accessible as attributes.

        Raises:
            ValueError: If image type is unsupported or graph is empty.
            ValidationError: If graph compilation fails.
            RuntimeError: If graph execution fails.

        Examples:
            Fluent chained execution::

                result = (Graph("top5")
                    .then(Detect(using="detector", out="dets"))
                    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
                    .then(TopK(k=5, src="filtered", out="top5"))
                    .then(Fuse(dets="top5", out="final"))
                    .run("photo.jpg", providers={"detector": detector})
                )

            Separate build and run::

                graph = Graph("pipeline").then(...).then(...)
                result = graph.run("photo.jpg", providers={"detector": detector})

            With parallel scheduler::

                result = graph.run(
                    "scene.jpg",
                    providers={...},
                    scheduler=ParallelScheduler(),
                )
        """
        from mata.api import infer

        return infer(
            image=image,
            graph=self,
            providers=providers,
            scheduler=scheduler,
            device=device,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation of graph."""
        status = "compiled" if self._compiled else "uncompiled"
        return f"<Graph '{self.name}' ({len(self._nodes)} nodes, {status})>"


@dataclass
class CompiledGraph:
    """Compiled, validated graph ready for execution.

    A CompiledGraph is the result of compiling a Graph. It contains:
    - Validated nodes and wiring
    - DAG representation (if networkx available)
    - Computed execution order with parallelization opportunities
    - Validation results

    The scheduler uses the execution order to run nodes efficiently,
    potentially parallelizing independent nodes within each stage.

    Attributes:
        name: Graph name
        nodes: List of validated nodes
        wiring: Validated wiring dictionary
        dag: NetworkX DiGraph (if available)
        validation_result: Results from validation
        execution_order: Computed execution stages (list of parallel node groups)

    Example:
        ```python
        compiled = graph.compile(providers)

        # Inspect execution order
        for stage_num, parallel_nodes in enumerate(compiled.execution_order):
            print(f"Stage {stage_num}: {[n.name for n in parallel_nodes]}")

        # Get parallel stages
        parallel_stages = compiled.get_parallel_stages()
        ```
    """

    name: str
    nodes: list[Node]
    wiring: dict[str, str]
    dag: Any | None  # nx.DiGraph if networkx available
    validation_result: ValidationResult
    execution_order: list[list[Node]] = None

    def __post_init__(self):
        """Compute execution order after initialization."""
        if self.execution_order is None:
            self.execution_order = self._compute_order()

    def _compute_order(self) -> list[list[Node]]:
        """Compute execution order with parallelization opportunities.

        Uses topological sorting to determine safe execution order, then groups
        nodes into stages where nodes in the same stage can execute in parallel.

        Returns:
            List of stages, where each stage is a list of nodes that can run in parallel

        Example:
            Stage 0: [Detect, EstimateDepth]  # Both operate on input.image
            Stage 1: [Filter]                  # Depends on Detect
            Stage 2: [SegmentImage]            # Depends on Filter
        """
        if HAS_NETWORKX and self.dag:
            return self._compute_order_networkx()
        else:
            return self._compute_order_manual()

    def _compute_order_networkx(self) -> list[list[Node]]:
        """Compute execution order using NetworkX topological generations."""
        # Use topological generations to find parallel stages
        node_map = {node.name: node for node in self.nodes}

        stages = []
        for generation in nx.topological_generations(self.dag):
            stage_nodes = [node_map[name] for name in generation]
            stages.append(stage_nodes)

        return stages

    def _compute_order_manual(self) -> list[list[Node]]:
        """Compute execution order manually using Kahn's algorithm.

        Fallback when networkx is not available.
        """
        # Build dependency graph
        node_map = {node.name: node for node in self.nodes}
        dependencies: dict[str, set[str]] = defaultdict(set)

        for target, source in self.wiring.items():
            # Parse target and source
            if "." in target:
                target_node, _ = target.rsplit(".", 1)
            else:
                continue

            if "." in source:
                source_node, _ = source.rsplit(".", 1)
                if source_node != "input" and source_node in node_map:
                    dependencies[target_node].add(source_node)

        # Kahn's algorithm for topological sort with stages
        in_degree = {name: len(deps) for name, deps in dependencies.items()}
        for node in self.nodes:
            if node.name not in in_degree:
                in_degree[node.name] = 0

        stages = []
        remaining = set(node.name for node in self.nodes)

        while remaining:
            # Find all nodes with in_degree 0 (can run in parallel)
            ready = [name for name in remaining if in_degree[name] == 0]

            if not ready:
                # Shouldn't happen if graph is valid DAG
                break

            # Add stage
            stage_nodes = [node_map[name] for name in ready]
            stages.append(stage_nodes)

            # Remove ready nodes and update in_degrees
            for name in ready:
                remaining.remove(name)

                # Decrease in_degree for dependent nodes
                for dep_name, deps in dependencies.items():
                    if name in deps:
                        in_degree[dep_name] -= 1

        return stages

    def get_parallel_stages(self) -> list[list[Node]]:
        """Get nodes that can run in parallel.

        Returns the execution order where each inner list contains nodes
        that can be executed simultaneously.

        Returns:
            List of stages with parallel nodes

        Example:
            ```python
            for stage_num, parallel_nodes in enumerate(compiled.get_parallel_stages()):
                print(f"Stage {stage_num}: Can run {len(parallel_nodes)} nodes in parallel")
                for node in parallel_nodes:
                    print(f"  - {node.name}")
            ```
        """
        return self.execution_order

    def visualize(self, output_path: str) -> None:
        """Generate graph visualization.

        Creates a visual representation of the compiled graph showing nodes,
        their types, and data flow edges.

        Args:
            output_path: Path to save visualization (e.g., "graph.png")

        Raises:
            ImportError: If visualization dependencies not installed
        """
        if not HAS_NETWORKX or self.dag is None:
            raise ImportError("Visualization requires networkx. Install with: pip install networkx")

        try:
            # Try pydot for visualization
            import pydot  # noqa: F401
            from networkx.drawing.nx_pydot import write_dot

            # Create a copy with labels
            viz_graph = self.dag.copy()
            for node_name in viz_graph.nodes():
                node = viz_graph.nodes[node_name].get("node")
                if node:
                    viz_graph.nodes[node_name]["label"] = f"{node.name}\n({node.__class__.__name__})"

            # Write to file
            if output_path.endswith(".dot"):
                write_dot(viz_graph, output_path)
            else:
                # Convert to image format
                dot_data = nx.nx_pydot.to_pydot(viz_graph)

                # Determine format from extension
                ext = output_path.rsplit(".", 1)[-1].lower()
                if ext in ["png", "pdf", "svg", "jpg", "jpeg"]:
                    dot_data.write(output_path, format=ext)
                else:
                    raise ValueError(f"Unsupported output format: {ext}")

        except ImportError:
            raise ImportError("Graph visualization requires pydot. Install with: pip install pydot")

    def __repr__(self) -> str:
        """String representation of compiled graph."""
        return f"<CompiledGraph '{self.name}' " f"({len(self.nodes)} nodes, {len(self.execution_order)} stages)>"
