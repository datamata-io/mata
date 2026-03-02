"""Execution context for graph runtime.

Provides runtime context for graph execution including artifact storage,
provider access, metrics collection, tracing, provenance, and device management.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from mata.core.artifacts.base import Artifact
from mata.core.exceptions import ValidationError
from mata.core.observability.metrics import MetricsCollector
from mata.core.observability.provenance import ProvenanceTracker
from mata.core.observability.tracing import ExecutionTracer


class ExecutionContext:
    """Runtime context for graph execution.

    ExecutionContext manages the runtime state during graph execution, providing:
    - Artifact storage and retrieval by name
    - Provider registry access by capability
    - Device management (auto-detection for CUDA/CPU)
    - Metrics collection (latency, memory, custom metrics)
    - Optional artifact caching
    - Cleanup utilities

    The context is passed to every node's run() method, allowing nodes to:
    - Store/retrieve intermediate artifacts
    - Access task providers (detectors, classifiers, etc.)
    - Record execution metrics
    - Query the execution device

    Thread Safety:
        ExecutionContext is NOT thread-safe by design. Each graph execution
        should use its own context instance. For parallel node execution,
        the scheduler should handle synchronization externally.

    Example:
        ```python
        from mata.core.graph.context import ExecutionContext

        # Create context with providers
        ctx = ExecutionContext(
            providers={"detect": {"detr": detr_adapter}},
            device="cuda",
            cache_artifacts=True
        )

        # Store artifacts
        ctx.store("image", image_artifact)

        # Retrieve artifacts
        image = ctx.retrieve("image")

        # Get providers
        detector = ctx.get_provider("detect", "detr")

        # Record metrics
        ctx.record_metric("detect_node", "latency_ms", 45.2)
        ctx.record_metric("detect_node", "num_detections", 5)

        # Get all metrics
        metrics = ctx.get_metrics()
        # {"detect_node": {"latency_ms": 45.2, "num_detections": 5}}
        ```

    Attributes:
        providers: Nested dict of providers by capability and name
        device: Resolved device string ("cuda" or "cpu")
    """

    def __init__(
        self,
        providers: dict[str, dict[str, Any]] | None = None,
        device: str = "auto",
        cache_artifacts: bool = False,
    ):
        """Initialize execution context.

        Args:
            providers: Nested dictionary of providers {capability: {name: provider}}.
                      Example: {"detect": {"detr": detr_adapter, "yolo": yolo_adapter}}
            device: Device to use for execution. Options:
                   - "auto": Auto-detect (CUDA if available, otherwise CPU)
                   - "cuda": Force CUDA (will fail if not available)
                   - "cpu": Force CPU
            cache_artifacts: If True, artifacts are cached in memory for reuse.
                           If False, artifacts may be garbage collected after use.

        Example:
            ```python
            ctx = ExecutionContext(
                providers={
                    "detect": {"detr": detr_adapter},
                    "segment": {"sam": sam_adapter}
                },
                device="auto",
                cache_artifacts=True
            )
            ```
        """
        self.providers = providers or {}
        self.device = self._resolve_device(device)
        self._artifacts: dict[str, Artifact] = {}
        self._metrics: dict[str, dict[str, float]] = defaultdict(dict)
        self._cache_enabled = cache_artifacts
        self._start_time = time.time()

        # Observability sub-systems
        self.metrics_collector = MetricsCollector()
        self.tracer = ExecutionTracer()
        self.provenance_tracker = ProvenanceTracker()

    def store(self, name: str, artifact: Artifact) -> None:
        """Store artifact by name.

        Artifacts are stored in the context for later retrieval by nodes.
        If caching is disabled, artifacts may be overwritten or removed.

        Args:
            name: Unique name for the artifact (e.g., "detections", "masks")
            artifact: Artifact instance to store

        Raises:
            ValueError: If name is empty or artifact is not an Artifact instance

        Example:
            ```python
            ctx.store("detections", detections_artifact)
            ctx.store("filtered_dets", filtered_detections)
            ```
        """
        if not name:
            raise ValueError("Artifact name cannot be empty")

        if not isinstance(artifact, Artifact):
            raise ValueError(f"Can only store Artifact instances, got {type(artifact).__name__}")

        # Validate artifact before storing
        try:
            artifact.validate()
        except Exception as e:
            raise ValidationError(f"Cannot store invalid artifact '{name}': {e}")

        self._artifacts[name] = artifact

    def retrieve(self, name: str) -> Artifact:
        """Retrieve artifact by name.

        Args:
            name: Name of the artifact to retrieve

        Returns:
            The stored artifact

        Raises:
            KeyError: If artifact with given name does not exist

        Example:
            ```python
            detections = ctx.retrieve("detections")
            image = ctx.retrieve("input_image")
            ```
        """
        if name not in self._artifacts:
            available = ", ".join(self._artifacts.keys()) if self._artifacts else "none"
            raise KeyError(f"Artifact '{name}' not found in context. " f"Available artifacts: {available}")

        return self._artifacts[name]

    def has(self, name: str) -> bool:
        """Check if artifact exists in context.

        Args:
            name: Name of the artifact to check

        Returns:
            True if artifact exists, False otherwise

        Example:
            ```python
            if ctx.has("detections"):
                dets = ctx.retrieve("detections")
            else:
                # Run detection node first
                ...
            ```
        """
        return name in self._artifacts

    def get_provider(self, capability: str, name: str) -> Any:
        """Get provider by capability and name.

        Providers are task-specific adapters (e.g., detectors, classifiers)
        registered in the context. Nodes use this method to access the
        appropriate provider for their task.

        Args:
            capability: Provider capability type (e.g., "detect", "segment")
            name: Provider name (e.g., "detr", "sam")

        Returns:
            The provider instance

        Raises:
            KeyError: If capability or provider name not found

        Example:
            ```python
            # In a detect node
            detector = ctx.get_provider("detect", "detr")
            result = detector.predict(image)

            # In a segment node
            segmenter = ctx.get_provider("segment", "sam")
            masks = segmenter.segment(image, prompts=boxes)
            ```
        """
        if capability not in self.providers:
            available_caps = ", ".join(self.providers.keys()) if self.providers else "none"
            raise KeyError(
                f"Capability '{capability}' not found in provider registry. "
                f"Available capabilities: {available_caps}"
            )

        capability_providers = self.providers[capability]

        if name not in capability_providers:
            available_names = ", ".join(capability_providers.keys()) if capability_providers else "none"
            raise KeyError(
                f"Provider '{name}' not found for capability '{capability}'. " f"Available providers: {available_names}"
            )

        return capability_providers[name]

    def record_metric(self, node: str, metric: str, value: float) -> None:
        """Record execution metric for a node.

        Metrics are stored per-node and can include latency, memory usage,
        custom counters, etc. All values are stored as floats.

        Args:
            node: Node name (typically node.name)
            metric: Metric name (e.g., "latency_ms", "num_detections")
            value: Metric value (will be converted to float)

        Example:
            ```python
            # Record latency
            start = time.time()
            result = detector.predict(image)
            latency_ms = (time.time() - start) * 1000
            ctx.record_metric("detect_node", "latency_ms", latency_ms)

            # Record custom metrics
            ctx.record_metric("detect_node", "num_detections", len(result.instances))
            ctx.record_metric("filter_node", "filtered_count", 3)
            ```
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Metric value must be numeric, got {type(value).__name__}: {value}")

        self._metrics[node][metric] = numeric_value

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Get all recorded metrics.

        Returns:
            Dictionary mapping node names to their metrics.
            Format: {node_name: {metric_name: value}}

        Example:
            ```python
            metrics = ctx.get_metrics()
            # {
            #     "detect_node": {"latency_ms": 45.2, "num_detections": 5},
            #     "filter_node": {"latency_ms": 0.5, "filtered_count": 3}
            # }

            # Calculate total latency
            total_latency = sum(
                m.get("latency_ms", 0)
                for m in metrics.values()
            )
            ```
        """
        # Return a dict (not defaultdict) to avoid side effects
        return dict(self._metrics)

    def get_node_metrics(self, node: str) -> dict[str, float]:
        """Get metrics for a specific node.

        Args:
            node: Node name

        Returns:
            Dictionary of metrics for the node, empty dict if no metrics recorded

        Example:
            ```python
            detect_metrics = ctx.get_node_metrics("detect_node")
            latency = detect_metrics.get("latency_ms", 0)
            ```
        """
        return dict(self._metrics.get(node, {}))

    def clear_artifacts(self) -> None:
        """Clear all stored artifacts.

        Useful for cleanup after graph execution or to free memory.
        Metrics are NOT cleared.

        Example:
            ```python
            # After graph execution
            result = scheduler.execute(graph, ctx, initial_artifacts)

            # Free memory
            ctx.clear_artifacts()
            ```
        """
        self._artifacts.clear()

    def clear_metrics(self) -> None:
        """Clear all recorded metrics.

        Artifacts are NOT cleared.

        Example:
            ```python
            # Reset metrics for new execution
            ctx.clear_metrics()
            ```
        """
        self._metrics.clear()

    def clear(self) -> None:
        """Clear both artifacts and metrics.

        Complete reset of the context state.

        Example:
            ```python
            # Complete cleanup
            ctx.clear()
            ```
        """
        self.clear_artifacts()
        self.clear_metrics()

    def get_execution_time(self) -> float:
        """Get total execution time since context creation.

        Returns:
            Elapsed time in seconds

        Example:
            ```python
            ctx = ExecutionContext(...)
            # ... execute graph ...
            print(f"Total time: {ctx.get_execution_time():.2f}s")
            ```
        """
        return time.time() - self._start_time

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device.

        Args:
            device: Device specification ("auto", "cuda", "cpu")

        Returns:
            Resolved device string ("cuda" or "cpu")

        Raises:
            ValueError: If device is invalid or CUDA requested but not available

        Example:
            ```python
            # Auto-detection
            device = ctx._resolve_device("auto")  # "cuda" if available, else "cpu"

            # Explicit
            device = ctx._resolve_device("cuda")  # Raises if CUDA not available
            device = ctx._resolve_device("cpu")   # Always works
            ```
        """
        if device.lower() == "auto":
            # Auto-detect: prefer CUDA if available
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                # PyTorch not available, default to CPU
                return "cpu"

        elif device.lower() == "cuda":
            # Force CUDA: verify availability
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA device requested but CUDA is not available. " "Use device='auto' or device='cpu' instead."
                    )
                return "cuda"
            except ImportError:
                raise ValueError(
                    "CUDA device requested but PyTorch is not installed. "
                    "Install PyTorch with CUDA support or use device='cpu'."
                )

        elif device.lower() == "cpu":
            return "cpu"

        else:
            raise ValueError(f"Invalid device '{device}'. " f"Valid options: 'auto', 'cuda', 'cpu'")

    def __repr__(self) -> str:
        """String representation of context.

        Returns:
            Human-readable string with context state

        Example:
            ```python
            print(ctx)
            # ExecutionContext(device='cuda', artifacts=3, providers=2, metrics=1)
            ```
        """
        num_caps = len(self.providers)
        num_providers = sum(len(p) for p in self.providers.values())
        num_artifacts = len(self._artifacts)
        num_nodes_with_metrics = len(self._metrics)

        return (
            f"ExecutionContext("
            f"device='{self.device}', "
            f"artifacts={num_artifacts}, "
            f"capabilities={num_caps}, "
            f"providers={num_providers}, "
            f"metrics={num_nodes_with_metrics})"
        )
