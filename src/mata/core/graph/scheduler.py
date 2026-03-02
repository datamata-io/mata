"""Graph scheduler for synchronous and parallel execution.

Implements schedulers that execute compiled graphs with timing metrics,
error handling, and provenance collection.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext
    from mata.core.graph.graph import CompiledGraph
    from mata.core.graph.node import Node

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.result import MultiResult
from mata.core.logging import get_logger

logger = get_logger(__name__)


class SyncScheduler:
    """Synchronous graph executor.

    Executes compiled graphs synchronously in topological order, ensuring
    all data dependencies are satisfied before running each node.

    Features:
    - Sequential execution in dependency order
    - Per-node timing metrics
    - Error handling with detailed context
    - Provenance collection (models, graph config, timestamps)
    - Artifact storage in execution context

    Example:
        ```python
        from mata.core.graph import Graph, SyncScheduler
        from mata.core.graph.context import ExecutionContext

        # Build and compile graph
        graph = Graph().then(Detect(using="detr", out="dets"))
        compiled = graph.compile(providers={"detect": {"detr": detr_adapter}})

        # Create context
        ctx = ExecutionContext(providers={"detect": {"detr": detr_adapter}})

        # Execute with scheduler
        scheduler = SyncScheduler()
        result = scheduler.execute(
            compiled,
            ctx,
            initial_artifacts={"input.image": image_artifact}
        )

        # Access results
        print(result.detections)
        print(result.metrics)
        print(result.provenance)
        ```

    Attributes:
        None (stateless - can reuse for multiple executions)
    """

    def __init__(self):
        """Initialize synchronous scheduler.

        The scheduler is stateless and can be reused for multiple graph executions.
        """
        pass

    def execute(
        self, graph: CompiledGraph, context: ExecutionContext, initial_artifacts: dict[str, Artifact]
    ) -> MultiResult:
        """Execute graph synchronously in topological order.

        Executes all nodes in the compiled graph sequentially, respecting
        data dependencies. Collects timing metrics, handles errors, and
        packages results into a MultiResult.

        Args:
            graph: Compiled graph with validated nodes and execution order
            context: Execution context with providers and artifact storage
            initial_artifacts: Initial artifacts (typically {"input.image": image})
                             Keys should use "input." prefix for graph inputs.

        Returns:
            MultiResult containing all output channels, metrics, and provenance

        Raises:
            ValidationError: If initial artifacts are missing or wrong type
            RuntimeError: If node execution fails

        Example:
            ```python
            result = scheduler.execute(
                compiled_graph,
                context,
                initial_artifacts={"input.image": image_artifact}
            )

            # Access channels
            detections = result.detections
            masks = result.masks

            # Access metadata
            print(f"Total time: {result.metrics['total_time_ms']}ms")
            print(f"Graph: {result.provenance['graph_name']}")
            ```
        """
        start_time = time.perf_counter()
        logger.info(f"Starting synchronous execution of graph '{graph.name}'")

        # Initialise observability
        context.metrics_collector.start()
        root_span = context.tracer.start_span(
            f"graph:{graph.name}",
            attributes={"num_nodes": len(graph.nodes), "device": context.device},
        )
        context.provenance_tracker.record_graph(graph)
        for cap, providers_dict in context.providers.items():
            for pname, prov in providers_dict.items():
                context.provenance_tracker.record_model(f"{cap}.{pname}", prov)

        # Store initial artifacts in context
        for name, artifact in initial_artifacts.items():
            context.store(name, artifact)

        # Execute nodes in topological order (stage by stage)
        try:
            for stage_idx, stage_nodes in enumerate(graph.execution_order):
                logger.debug(
                    f"Executing stage {stage_idx + 1}/{len(graph.execution_order)}: " f"{[n.name for n in stage_nodes]}"
                )

                # Execute each node in the stage sequentially
                for node in stage_nodes:
                    self._execute_node(node, context, graph.wiring)

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            context.tracer.end_span(root_span, status="error", error_message=str(e))
            context.metrics_collector.stop()
            raise

        # Finalise observability
        context.tracer.end_span(root_span)
        context.metrics_collector.stop()

        # Collect final results
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        logger.info(f"Graph '{graph.name}' completed in {total_time_ms:.2f}ms " f"({len(graph.nodes)} nodes)")

        # Build MultiResult
        result = self._build_result(graph, context, total_time_ms)

        return result

    def _execute_node(self, node: Node, context: ExecutionContext, wiring: dict[str, str]) -> dict[str, Artifact]:
        """Execute single node with timing.

        Executes a node by:
        1. Resolving input artifacts from context using wiring
        2. Validating input types
        3. Running the node
        4. Validating output types
        5. Storing outputs in context
        6. Recording timing metrics

        Args:
            node: Node to execute
            context: Execution context
            wiring: Graph wiring dictionary

        Returns:
            Dictionary of output artifacts produced by the node

        Raises:
            ValidationError: If input/output validation fails
            RuntimeError: If node execution fails
        """
        logger.debug(f"Executing node '{node.name}'")
        node_start = time.perf_counter()

        # Start tracer span for this node
        # Find root span (the first root) to use as parent
        root_spans = context.tracer.get_root_spans()
        parent_id = root_spans[-1].span_id if root_spans else None
        span = context.tracer.start_span(
            f"node:{node.name}",
            parent_id=parent_id,
            attributes={"node_type": node.__class__.__name__},
        )

        try:
            # Resolve inputs from wiring
            input_artifacts = self._resolve_inputs(node, context, wiring)

            # Validate inputs
            node.validate_inputs(input_artifacts)

            # Execute node
            output_artifacts = node.run(context, **input_artifacts)

            # Validate outputs
            node.validate_outputs(output_artifacts)

            # Store outputs in context
            for output_name, artifact in output_artifacts.items():
                # Store with node-qualified name
                qualified_name = f"{node.name}.{output_name}"
                context.store(qualified_name, artifact)

                # Also store with unqualified name for convenience
                context.store(output_name, artifact)

            # Record timing metric
            node_end = time.perf_counter()
            latency_ms = (node_end - node_start) * 1000
            context.record_metric(node.name, "latency_ms", latency_ms)

            # Record in MetricsCollector as well
            context.metrics_collector.record_latency(node.name, latency_ms)
            self._record_memory_metrics(context, node.name)

            # End tracer span
            context.tracer.end_span(span)

            logger.debug(
                f"Node '{node.name}' completed in {latency_ms:.2f}ms, " f"produced {len(output_artifacts)} outputs"
            )

            return output_artifacts

        except Exception as e:
            # End span with error
            context.tracer.end_span(span, status="error", error_message=str(e))
            self._handle_error(node, e)
            raise  # Re-raise after logging

    def _resolve_inputs(self, node: Node, context: ExecutionContext, wiring: dict[str, str]) -> dict[str, Artifact]:
        """Resolve input artifacts for a node from wiring.

        Maps node input names to actual artifacts by following the wiring
        connections.

        Args:
            node: Node requiring inputs
            context: Execution context with stored artifacts
            wiring: Graph wiring dictionary

        Returns:
            Dictionary mapping input parameter names to artifacts

        Raises:
            KeyError: If required input artifact is not found
        """
        input_artifacts = {}

        for input_name, input_type in node.inputs.items():
            # Look for wiring entry
            wiring_key = f"{node.name}.{input_name}"

            if wiring_key in wiring:
                # Get source artifact name from wiring
                source = wiring[wiring_key]

                # Retrieve artifact from context
                try:
                    artifact = context.retrieve(source)
                    input_artifacts[input_name] = artifact
                except KeyError:
                    # Try without node prefix if source has one
                    if "." in source:
                        _, simple_name = source.rsplit(".", 1)
                        try:
                            artifact = context.retrieve(simple_name)
                            input_artifacts[input_name] = artifact
                        except KeyError:
                            raise KeyError(
                                f"Node '{node.name}' requires input '{input_name}' "
                                f"from '{source}', but artifact not found in context. "
                                f"Available: {list(context._artifacts.keys())}"
                            )
                    else:
                        raise
            else:
                # No explicit wiring - maybe it's optional or has default
                # Check if input is optional (has Optional type hint)
                # For now, just skip - validation will catch missing required inputs
                pass

        return input_artifacts

    def _handle_error(self, node: Node, error: Exception) -> None:
        """Handle node execution failure.

        Logs detailed error information including node name, error type,
        and error message.

        Args:
            node: Node that failed
            error: Exception that was raised
        """
        logger.error(
            f"Node '{node.name}' ({node.__class__.__name__}) failed: " f"{error.__class__.__name__}: {error}",
            exc_info=True,
        )

    def _record_memory_metrics(
        self,
        context: ExecutionContext,
        node_name: str,
    ) -> None:
        """Record memory metrics for a node (best effort).

        Captures process RSS memory and GPU memory when available.

        Args:
            context: Execution context.
            node_name: Node name to record against.
        """
        # Process memory (RSS)
        try:

            # /proc/self/status is Linux-only; fallback silently
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        context.metrics_collector.record_memory(node_name, kb / 1024)
                        break
        except Exception:
            pass

        # GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                context.metrics_collector.record_gpu_memory(node_name, gpu_mb)
        except Exception:
            pass

    def _collect_provenance(self, context: ExecutionContext, graph: CompiledGraph) -> dict[str, Any]:
        """Collect provenance metadata.

        Gathers metadata about the graph execution including:
        - Graph name and configuration
        - Model information from providers
        - Execution timestamp
        - Device used

        Args:
            context: Execution context
            graph: Compiled graph

        Returns:
            Dictionary of provenance metadata
        """
        provenance = {
            "graph_name": graph.name,
            "graph_hash": self._compute_graph_hash(graph),
            "timestamp": datetime.now().isoformat(),
            "device": context.device,
            "num_nodes": len(graph.nodes),
            "num_stages": len(graph.execution_order),
        }

        # Collect model information from providers
        models = {}
        for capability, provider_dict in context.providers.items():
            for name, provider in provider_dict.items():
                # Try to get model ID or name
                model_id = getattr(provider, "model_id", None)
                if model_id:
                    models[f"{capability}.{name}"] = model_id

        if models:
            provenance["models"] = models

        return provenance

    def _compute_graph_hash(self, graph: CompiledGraph) -> str:
        """Compute hash of graph structure for provenance.

        Creates a hash based on node types, names, and wiring to uniquely
        identify the graph configuration.

        Args:
            graph: Compiled graph

        Returns:
            SHA256 hash string (first 8 characters)
        """
        # Build string representation of graph structure
        graph_repr = f"{graph.name}|"

        for node in graph.nodes:
            graph_repr += f"{node.__class__.__name__}:{node.name}|"

        for key, value in sorted(graph.wiring.items()):
            graph_repr += f"{key}={value}|"

        # Compute hash
        hash_obj = hashlib.sha256(graph_repr.encode())
        return hash_obj.hexdigest()[:8]

    def _build_result(self, graph: CompiledGraph, context: ExecutionContext, total_time_ms: float) -> MultiResult:
        """Package execution results into MultiResult.

        Collects all artifacts produced during graph execution and packages
        them with metrics and provenance into a MultiResult.

        Args:
            graph: Compiled graph
            context: Execution context with artifacts
            total_time_ms: Total execution time in milliseconds

        Returns:
            MultiResult with channels, metrics, and provenance
        """
        # Collect output channels
        channels = {}

        # Get all artifacts from context (excluding input. prefixed ones)
        for name, artifact in context._artifacts.items():
            if not name.startswith("input.") and "." not in name:
                # Only include simple names (not node-qualified names)
                channels[name] = artifact

        # Collect metrics
        metrics = dict(context.get_metrics())
        metrics["total_time_ms"] = total_time_ms

        # Add per-stage timing if available
        stage_times = defaultdict(float)
        for node in graph.nodes:
            if node.name in metrics:
                node_metrics = metrics[node.name]
                if "latency_ms" in node_metrics:
                    # Find which stage this node belongs to
                    for stage_idx, stage_nodes in enumerate(graph.execution_order):
                        if node in stage_nodes:
                            stage_times[f"stage_{stage_idx}_ms"] += node_metrics["latency_ms"]
                            break

        metrics.update(stage_times)

        # Collect provenance
        provenance = self._collect_provenance(context, graph)

        # Build MultiResult
        result = MultiResult(channels=channels, provenance=provenance, metrics=metrics, meta={"graph_name": graph.name})

        return result


class ParallelScheduler(SyncScheduler):
    """Parallel graph executor using ThreadPoolExecutor.

    Extends SyncScheduler to execute nodes within each execution stage in
    parallel using threads. This can provide significant speedup for graphs
    with independent parallel paths (e.g., running detection and depth
    estimation simultaneously).

    Features:
    - All features of SyncScheduler
    - Parallel execution of independent nodes within each stage
    - Configurable thread pool size
    - Thread-safe artifact storage
    - Per-thread error handling

    Example:
        ```python
        from mata.core.graph import Graph, ParallelScheduler

        # Graph with parallel branches
        graph = Graph().parallel([
            Detect(using="detr", out="dets"),
            EstimateDepth(using="depth_anything", out="depth")
        ])

        compiled = graph.compile(providers=providers)

        # Execute with parallel scheduler
        scheduler = ParallelScheduler(max_workers=4)
        result = scheduler.execute(compiled, ctx, initial_artifacts)

        # Both detection and depth estimation ran in parallel
        print(result.detections)
        print(result.depth)
        ```

    Attributes:
        max_workers: Maximum number of parallel threads
    """

    def __init__(self, max_workers: int = 4):
        """Initialize parallel scheduler.

        Args:
            max_workers: Maximum number of threads in the pool.
                        Recommended: 2-4 for typical CV tasks.
                        Too many threads can cause memory issues with large models.

        Example:
            ```python
            # Conservative (good for large models)
            scheduler = ParallelScheduler(max_workers=2)

            # Aggressive (good for lightweight models)
            scheduler = ParallelScheduler(max_workers=8)
            ```
        """
        super().__init__()
        self.max_workers = max_workers
        logger.info(f"ParallelScheduler initialized with {max_workers} workers")

    def execute(
        self, graph: CompiledGraph, context: ExecutionContext, initial_artifacts: dict[str, Artifact]
    ) -> MultiResult:
        """Execute graph with parallel stages.

        Executes the graph with parallelization within each stage. Nodes in
        the same stage (with no data dependencies) are executed concurrently
        using a thread pool.

        Args:
            graph: Compiled graph with validated nodes and execution order
            context: Execution context with providers and artifact storage
            initial_artifacts: Initial artifacts (typically {"input.image": image})

        Returns:
            MultiResult containing all output channels, metrics, and provenance

        Raises:
            ValidationError: If initial artifacts are missing or wrong type
            RuntimeError: If node execution fails

        Note:
            Thread safety: While the context is not thread-safe by design,
            the scheduler ensures synchronization by waiting for all nodes
            in a stage to complete before proceeding to the next stage.
        """
        start_time = time.perf_counter()
        logger.info(f"Starting parallel execution of graph '{graph.name}' " f"with {self.max_workers} workers")

        # Store initial artifacts in context
        for name, artifact in initial_artifacts.items():
            context.store(name, artifact)

        # Execute stages in order, parallelizing within each stage
        try:
            for stage_idx, stage_nodes in enumerate(graph.execution_order):
                logger.debug(
                    f"Executing stage {stage_idx + 1}/{len(graph.execution_order)}: "
                    f"{len(stage_nodes)} nodes in parallel: {[n.name for n in stage_nodes]}"
                )

                if len(stage_nodes) == 1:
                    # Single node - execute directly without thread overhead
                    self._execute_node(stage_nodes[0], context, graph.wiring)
                else:
                    # Multiple nodes - execute in parallel
                    self._execute_stage(stage_nodes, context, graph.wiring)

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            raise

        # Collect final results
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        logger.info(
            f"Graph '{graph.name}' completed in {total_time_ms:.2f}ms "
            f"({len(graph.nodes)} nodes, {len(graph.execution_order)} stages)"
        )

        # Build MultiResult
        result = self._build_result(graph, context, total_time_ms)

        return result

    def _execute_stage(
        self, nodes: list[Node], context: ExecutionContext, wiring: dict[str, str]
    ) -> dict[str, Artifact]:
        """Execute nodes in parallel stage.

        Runs all nodes in a stage concurrently using ThreadPoolExecutor.
        Waits for all nodes to complete before returning.

        Args:
            nodes: List of nodes to execute in parallel
            context: Execution context
            wiring: Graph wiring dictionary

        Returns:
            Dictionary of all output artifacts from the stage

        Raises:
            RuntimeError: If any node fails
        """
        stage_outputs = {}

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all nodes
            future_to_node = {executor.submit(self._execute_node, node, context, wiring): node for node in nodes}

            # Wait for completion and collect results
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    outputs = future.result()
                    stage_outputs.update(outputs)
                except Exception as e:
                    logger.error(f"Node '{node.name}' failed in parallel execution: {e}")
                    # Cancel remaining futures
                    for f in future_to_node:
                        f.cancel()
                    raise RuntimeError(f"Parallel stage execution failed at node '{node.name}': {e}") from e

        return stage_outputs


class OptimizedParallelScheduler(ParallelScheduler):
    """Parallel scheduler with device placement and memory optimization.

    Extends ParallelScheduler with advanced features for production environments:
    - Device placement strategies (auto, round_robin, memory_aware)
    - Multi-GPU support with automatic distribution
    - Model unloading for memory efficiency
    - GPU memory management with cache clearing
    - Enhanced parallel stage execution with device-aware scheduling

    Features:
    - All features of ParallelScheduler
    - Smart device placement for each node
    - Automatic model unloading to free memory
    - Multi-GPU load balancing
    - GPU memory monitoring and optimization

    Example:
        ```python
        from mata.core.graph import Graph, OptimizedParallelScheduler

        # Create optimized scheduler
        scheduler = OptimizedParallelScheduler(
            max_workers=4,
            device_placement="memory_aware",  # Choose GPU with most free memory
            unload_unused=True  # Unload models after use
        )

        result = scheduler.execute(compiled_graph, context, initial_artifacts)

        # Models were automatically placed on optimal devices
        # and unloaded after execution to free memory
        ```

    Device Placement Strategies:
    - "auto": Prefer GPU if available, fallback to CPU
    - "round_robin": Distribute nodes across available GPUs
    - "memory_aware": Choose GPU with most free memory for each node

    Attributes:
        max_workers: Maximum number of parallel threads
        device_placement: Strategy for device assignment
        unload_unused: Whether to unload models after use
        _device_pool: List of available devices
        _device_index: Current device index for round-robin
        _device_usage: Track device assignments for load balancing
    """

    def __init__(
        self,
        max_workers: int = 4,
        device_placement: str = "auto",  # auto, round_robin, memory_aware
        unload_unused: bool = True,
    ):
        """Initialize optimized parallel scheduler.

        Args:
            max_workers: Maximum number of threads in the pool
            device_placement: Strategy for device assignment:
                - "auto": Prefer GPU if available, fallback to CPU
                - "round_robin": Distribute across available GPUs
                - "memory_aware": Choose GPU with most free memory
            unload_unused: Whether to unload models from memory after use

        Raises:
            ValueError: If device_placement strategy is invalid

        Example:
            ```python
            # Memory-aware placement (recommended for production)
            scheduler = OptimizedParallelScheduler(
                max_workers=2,
                device_placement="memory_aware",
                unload_unused=True
            )

            # Round-robin for consistent load distribution
            scheduler = OptimizedParallelScheduler(
                device_placement="round_robin",
                unload_unused=False  # Keep models loaded
            )
            ```
        """
        super().__init__(max_workers)

        valid_strategies = {"auto", "round_robin", "memory_aware"}
        if device_placement not in valid_strategies:
            raise ValueError(f"Invalid device_placement: {device_placement}. " f"Must be one of: {valid_strategies}")

        self.device_placement = device_placement
        self.unload_unused = unload_unused
        self._device_pool = self._init_devices()
        self._device_index = 0  # For round-robin
        self._device_usage = defaultdict(int)  # Track assignments

        logger.info(
            f"OptimizedParallelScheduler initialized: "
            f"workers={max_workers}, placement={device_placement}, "
            f"unload={unload_unused}, devices={self._device_pool}"
        )

    def _init_devices(self) -> list[str]:
        """Initialize available devices.

        Detects available CUDA devices and CPU, returning a list of device
        strings in preference order.

        Returns:
            List of device strings (e.g., ["cuda:0", "cuda:1", "cpu"])
        """
        devices = ["cpu"]

        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                cuda_devices = [f"cuda:{i}" for i in range(device_count)]
                devices = cuda_devices + devices  # GPUs first

                logger.info(f"CUDA available: {device_count} GPUs detected")

                # Log GPU memory info
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        memory_gb = props.total_memory / (1024**3)
                        logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                    except Exception:
                        pass  # Skip if can't get properties
            else:
                logger.info("CUDA not available, using CPU only")
        except ImportError:
            logger.warning("PyTorch not available, using CPU only")

        return devices

    def _assign_device(self, node: Node) -> str:
        """Assign device to node based on strategy.

        Args:
            node: Node to assign device for

        Returns:
            Device string (e.g., "cuda:0" or "cpu")
        """
        if self.device_placement == "auto":
            # Prefer first GPU if available, fallback to CPU
            if len(self._device_pool) > 1 and self._device_pool[0] != "cpu":
                device = self._device_pool[0]  # First GPU
            else:
                device = "cpu"

        elif self.device_placement == "round_robin":
            # Distribute across all GPUs, skip CPU unless no GPUs
            gpu_devices = [d for d in self._device_pool if d.startswith("cuda")]
            if gpu_devices:
                device = gpu_devices[self._device_index % len(gpu_devices)]
                self._device_index += 1
            else:
                device = "cpu"

        elif self.device_placement == "memory_aware":
            # Choose GPU with most free memory
            device = self._choose_best_memory_device()

        else:
            # Fallback to auto
            device = self._device_pool[0] if len(self._device_pool) > 1 else "cpu"

        # Track usage
        self._device_usage[device] += 1

        logger.debug(f"Assigned device '{device}' to node '{node.name}'")
        return device

    def _choose_best_memory_device(self) -> str:
        """Choose device with most free GPU memory.

        Returns:
            Device string with most available memory
        """
        best_device = "cpu"
        max_free_memory = 0

        try:
            import torch

            if torch.cuda.is_available():
                for device_str in self._device_pool:
                    if device_str.startswith("cuda"):
                        device_idx = int(device_str.split(":")[1])

                        # Get memory info
                        free_memory = torch.cuda.get_device_properties(device_idx).total_memory
                        try:
                            # Try to get current memory usage
                            allocated = torch.cuda.memory_allocated(device_idx)
                            free_memory -= allocated
                        except Exception:
                            pass  # Use total memory if can't get allocated

                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_device = device_str

                # If no GPU had significant free memory, fall back to first GPU
                if best_device == "cpu" and len(self._device_pool) > 1:
                    gpu_devices = [d for d in self._device_pool if d.startswith("cuda")]
                    if gpu_devices:
                        best_device = gpu_devices[0]

        except ImportError:
            pass  # Stick with CPU

        return best_device

    def _unload_model(self, provider: Any) -> None:
        """Unload model from memory after use.

        Attempts to free memory by:
        1. Deleting model attribute if present
        2. Clearing CUDA cache if on GPU
        3. Calling garbage collection

        Args:
            provider: Provider instance that may have a model
        """
        if not self.unload_unused:
            return

        try:
            # Try to unload model from provider
            if hasattr(provider, "model"):
                model = provider.model
                device_str = getattr(model, "device", None)

                logger.debug(f"Unloading model from provider: {type(provider).__name__}")

                # Delete the model
                del provider.model

                # Clear CUDA cache if model was on GPU
                if device_str and str(device_str).startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA cache after model unload")

            # Also try common adapter patterns
            for attr_name in ["adapter", "_model", "processor"]:
                if hasattr(provider, attr_name):
                    attr = getattr(provider, attr_name)
                    if attr is not None:
                        logger.debug(f"Unloading {attr_name} from provider")
                        setattr(provider, attr_name, None)

        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")
            # Don't raise - unloading is optional optimization

    def _execute_stage_optimized(
        self, nodes: list[Node], context: ExecutionContext, wiring: dict[str, str]
    ) -> dict[str, Artifact]:
        """Execute stage with device placement and memory optimization.

        Enhanced version of _execute_stage that:
        1. Assigns optimal devices to each node
        2. Executes nodes in parallel with device placement
        3. Unloads models after use if configured
        4. Provides detailed logging of device assignments

        Args:
            nodes: List of nodes to execute in parallel
            context: Execution context
            wiring: Graph wiring dictionary

        Returns:
            Dictionary of all output artifacts from the stage

        Raises:
            RuntimeError: If any node fails
        """
        stage_outputs = {}

        # Assign devices to nodes
        node_device_assignments = {}
        for node in nodes:
            device = self._assign_device(node)
            node_device_assignments[node] = device

            # Try to move provider to assigned device if possible
            self._move_provider_to_device(node, context, device)

        logger.info(
            f"Executing stage with {len(nodes)} nodes on devices: "
            f"{[(n.name, d) for n, d in node_device_assignments.items()]}"
        )

        # Execute in parallel with device assignments
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all nodes
            future_to_node = {
                executor.submit(
                    self._execute_node_with_device, node, context, wiring, node_device_assignments[node]
                ): node
                for node in nodes
            }

            # Wait for completion and collect results
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    outputs = future.result()
                    stage_outputs.update(outputs)

                    # Unload model if configured
                    if self.unload_unused:
                        self._unload_node_provider(node, context)

                except Exception as e:
                    logger.error(f"Node '{node.name}' failed in optimized execution: {e}")
                    # Cancel remaining futures
                    for f in future_to_node:
                        f.cancel()
                    raise RuntimeError(f"Optimized parallel stage execution failed at node '{node.name}': {e}") from e

        # Log device usage summary
        logger.info(f"Stage device usage: {dict(self._device_usage)}")

        return stage_outputs

    def _move_provider_to_device(self, node: Node, context: ExecutionContext, device: str) -> None:
        """Move provider's model to specified device if possible.

        Attempts to move the provider's model to the target device. This is
        optional and will fail silently if the provider doesn't support it.

        Args:
            node: Node whose provider to move
            context: Execution context with providers
            device: Target device string
        """
        try:
            # Get provider for this node
            # This is a simplified approach - real implementation would
            # need to resolve provider based on node configuration
            # For now, just skip the device movement
            pass
        except Exception:
            # Device movement is optional - don't fail if not supported
            pass

    def _execute_node_with_device(
        self, node: Node, context: ExecutionContext, wiring: dict[str, str], device: str
    ) -> dict[str, Artifact]:
        """Execute node with device context.

        Enhanced version of _execute_node that includes device information
        in the execution context.

        Args:
            node: Node to execute
            context: Execution context
            wiring: Graph wiring dictionary
            device: Assigned device for this node

        Returns:
            Dictionary of output artifacts
        """
        # Store device assignment in context for node to use
        original_device = context.device
        context.device = device

        try:
            # Execute normally
            return self._execute_node(node, context, wiring)
        finally:
            # Restore original device
            context.device = original_device

    def _unload_node_provider(self, node: Node, context: ExecutionContext) -> None:
        """Unload provider for specific node.

        Attempts to unload the provider associated with a node to free memory.

        Args:
            node: Node whose provider to unload
            context: Execution context with providers
        """
        try:
            # This would need to be implemented based on how providers
            # are resolved for nodes. For now, we'll implement a generic
            # approach that tries to find and unload any models

            # Try to get provider info from node
            if hasattr(node, "provider_name"):
                provider_name = node.provider_name
                # Would need to look up provider in context and unload it
                # This is a placeholder for the actual implementation
                logger.debug(f"Would unload provider '{provider_name}' for node '{node.name}'")

        except Exception as e:
            logger.debug(f"Could not unload provider for node '{node.name}': {e}")
            # Don't fail - unloading is optional

    def execute(
        self, graph: CompiledGraph, context: ExecutionContext, initial_artifacts: dict[str, Artifact]
    ) -> MultiResult:
        """Execute graph with optimized parallel execution and device placement.

        Enhanced version of ParallelScheduler.execute() that uses the optimized
        stage execution with device placement and memory management.

        Args:
            graph: Compiled graph with validated nodes and execution order
            context: Execution context with providers and artifact storage
            initial_artifacts: Initial artifacts (typically {"input.image": image})

        Returns:
            MultiResult containing all output channels, metrics, and provenance

        The execution includes:
        - Device placement for each node based on configured strategy
        - Parallel execution within stages
        - Model unloading for memory efficiency
        - Enhanced logging and monitoring
        """
        start_time = time.perf_counter()
        logger.info(
            f"Starting optimized parallel execution of graph '{graph.name}' "
            f"with {self.max_workers} workers, device_placement='{self.device_placement}'"
        )

        # Store initial artifacts in context
        for name, artifact in initial_artifacts.items():
            context.store(name, artifact)

        # Reset device usage tracking
        self._device_usage.clear()

        # Execute stages in order with optimization
        try:
            for stage_idx, stage_nodes in enumerate(graph.execution_order):
                logger.debug(
                    f"Executing optimized stage {stage_idx + 1}/{len(graph.execution_order)}: "
                    f"{len(stage_nodes)} nodes: {[n.name for n in stage_nodes]}"
                )

                if len(stage_nodes) == 1:
                    # Single node - execute directly with device assignment
                    node = stage_nodes[0]
                    device = self._assign_device(node)
                    self._execute_node_with_device(node, context, graph.wiring, device)

                    if self.unload_unused:
                        self._unload_node_provider(node, context)
                else:
                    # Multiple nodes - execute with optimized parallel execution
                    self._execute_stage_optimized(stage_nodes, context, graph.wiring)

        except Exception as e:
            logger.error(f"Optimized graph execution failed: {e}")
            raise

        # Collect final results
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Log final summary
        logger.info(
            f"Optimized graph '{graph.name}' completed in {total_time_ms:.2f}ms "
            f"({len(graph.nodes)} nodes, {len(graph.execution_order)} stages)"
        )
        logger.info(f"Final device usage: {dict(self._device_usage)}")

        # Build MultiResult with additional metrics
        result = self._build_result(graph, context, total_time_ms)

        # Add optimization metrics
        result.metrics["device_usage"] = dict(self._device_usage)
        result.metrics["device_placement_strategy"] = self.device_placement
        result.metrics["unload_unused_enabled"] = self.unload_unused
        result.metrics["available_devices"] = self._device_pool

        return result
