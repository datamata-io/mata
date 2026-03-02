"""Metrics collection for MATA graph execution.

Collects and aggregates execution metrics including latency, memory usage,
GPU memory, and custom counters for monitoring graph performance.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NodeMetrics:
    """Metrics recorded for a single node execution.

    Attributes:
        latency_ms: Node execution time in milliseconds.
        memory_mb: Process memory usage during execution in MB.
        gpu_memory_mb: GPU memory usage during execution in MB.
        custom: Additional user-defined metrics.
    """

    latency_ms: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
        }
        if self.custom:
            d["custom"] = dict(self.custom)
        return d


class MetricsCollector:
    """Collect and aggregate execution metrics for graph nodes.

    Tracks per-node latency, memory, GPU memory, and custom metrics.
    Provides aggregated summaries and export to JSON/CSV formats.

    Example:
        ```python
        from mata.core.observability import MetricsCollector

        collector = MetricsCollector()

        # Record metrics
        collector.record_latency("detect_node", 45.2)
        collector.record_memory("detect_node", 512.0)
        collector.record_gpu_memory("detect_node", 1024.0)

        # Record custom metric
        collector.record("detect_node", "num_detections", 5)

        # Get summary
        summary = collector.get_summary()
        print(summary["total_latency_ms"])
        print(summary["per_node"]["detect_node"]["latency_ms"])

        # Export
        json_str = collector.export("json")
        csv_str = collector.export("csv")
        ```
    """

    def __init__(self) -> None:
        """Initialize empty metrics collector."""
        self._metrics: dict[str, NodeMetrics] = {}
        self._start_time: float | None = None
        self._end_time: float | None = None

    def _ensure_node(self, node: str) -> NodeMetrics:
        """Get or create NodeMetrics for a node."""
        if node not in self._metrics:
            self._metrics[node] = NodeMetrics()
        return self._metrics[node]

    # ── Recording methods ────────────────────────────────────────────

    def record_latency(self, node: str, latency_ms: float) -> None:
        """Record execution latency for a node.

        Args:
            node: Node name.
            latency_ms: Execution time in milliseconds.
        """
        self._ensure_node(node).latency_ms = float(latency_ms)

    def record_memory(self, node: str, memory_mb: float) -> None:
        """Record process memory usage for a node.

        Args:
            node: Node name.
            memory_mb: Memory usage in megabytes.
        """
        self._ensure_node(node).memory_mb = float(memory_mb)

    def record_gpu_memory(self, node: str, gpu_memory_mb: float) -> None:
        """Record GPU memory usage for a node.

        Args:
            node: Node name.
            gpu_memory_mb: GPU memory usage in megabytes.
        """
        self._ensure_node(node).gpu_memory_mb = float(gpu_memory_mb)

    def record(self, node: str, metric: str, value: float) -> None:
        """Record a custom metric for a node.

        Args:
            node: Node name.
            metric: Metric name.
            value: Metric value (will be converted to float).
        """
        self._ensure_node(node).custom[metric] = float(value)

    # ── Timing helpers ───────────────────────────────────────────────

    def start(self) -> None:
        """Mark the start of a graph execution."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Mark the end of a graph execution."""
        self._end_time = time.perf_counter()

    @property
    def wall_time_ms(self) -> float:
        """Wall-clock time between start() and stop() in milliseconds.

        Returns 0 if start/stop were not called.
        """
        if self._start_time is None or self._end_time is None:
            return 0.0
        return (self._end_time - self._start_time) * 1000

    # ── Querying ─────────────────────────────────────────────────────

    def get_node_metrics(self, node: str) -> NodeMetrics | None:
        """Get metrics for a specific node.

        Args:
            node: Node name.

        Returns:
            NodeMetrics or None if the node has no recorded metrics.
        """
        return self._metrics.get(node)

    @property
    def nodes(self) -> list[str]:
        """List of nodes with recorded metrics."""
        return list(self._metrics.keys())

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated metrics summary.

        Returns:
            Dictionary with:
            - total_latency_ms: Sum of all node latencies.
            - peak_memory_mb: Maximum memory usage across nodes.
            - peak_gpu_memory_mb: Maximum GPU memory across nodes.
            - wall_time_ms: Wall-clock time (if start/stop called).
            - num_nodes: Number of nodes with metrics.
            - per_node: Per-node metrics dictionaries.
        """
        latencies = [m.latency_ms for m in self._metrics.values()]
        memories = [m.memory_mb for m in self._metrics.values() if m.memory_mb > 0]
        gpu_memories = [m.gpu_memory_mb for m in self._metrics.values() if m.gpu_memory_mb > 0]

        return {
            "total_latency_ms": sum(latencies) if latencies else 0.0,
            "peak_memory_mb": max(memories) if memories else 0.0,
            "peak_gpu_memory_mb": max(gpu_memories) if gpu_memories else 0.0,
            "wall_time_ms": self.wall_time_ms,
            "num_nodes": len(self._metrics),
            "per_node": {name: metrics.to_dict() for name, metrics in self._metrics.items()},
        }

    # ── Export ────────────────────────────────────────────────────────

    def export(self, fmt: str = "json") -> str:
        """Export metrics in the given format.

        Args:
            fmt: Export format — ``"json"`` or ``"csv"``.

        Returns:
            Formatted string.

        Raises:
            ValueError: If format is not supported.
        """
        fmt = fmt.lower()
        if fmt == "json":
            return self._export_json()
        elif fmt == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'csv'.")

    def _export_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.get_summary(), indent=2)

    def _export_csv(self) -> str:
        """Export metrics as CSV.

        Columns: node, latency_ms, memory_mb, gpu_memory_mb, plus any custom
        metric columns found across all nodes.
        """
        # Collect all custom metric keys across nodes
        custom_keys: list[str] = sorted({k for m in self._metrics.values() for k in m.custom})

        buf = io.StringIO()
        fieldnames = ["node", "latency_ms", "memory_mb", "gpu_memory_mb"] + custom_keys
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for name, metrics in self._metrics.items():
            row: dict[str, Any] = {
                "node": name,
                "latency_ms": metrics.latency_ms,
                "memory_mb": metrics.memory_mb,
                "gpu_memory_mb": metrics.gpu_memory_mb,
            }
            for ck in custom_keys:
                row[ck] = metrics.custom.get(ck, "")
            writer.writerow(row)

        return buf.getvalue()

    # ── Utilities ────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._start_time = None
        self._end_time = None

    def merge(self, other: MetricsCollector) -> None:
        """Merge metrics from another collector into this one.

        Existing nodes are overwritten by *other*'s data.

        Args:
            other: Another MetricsCollector instance.
        """
        for node, m in other._metrics.items():
            self._metrics[node] = m

    def __len__(self) -> int:
        """Number of nodes with recorded metrics."""
        return len(self._metrics)

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"MetricsCollector("
            f"nodes={summary['num_nodes']}, "
            f"total_latency_ms={summary['total_latency_ms']:.2f}, "
            f"peak_memory_mb={summary['peak_memory_mb']:.1f})"
        )
