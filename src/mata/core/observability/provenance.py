"""Provenance tracking for MATA graph execution.

Records model versions, configurations, and graph metadata to enable
reproducibility and auditability of inference results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRecord:
    """Record of a model used during graph execution.

    Attributes:
        name: Model identifier (HuggingFace ID, file path, or alias).
        model_type: Adapter/runtime type (e.g. ``"huggingface"``, ``"onnx"``).
        task: Task the model serves (e.g. ``"detect"``, ``"segment"``).
        version: Model version string, if known.
        hash: Content hash for reproducibility.
        extra: Additional metadata (device, dtype, etc.).
    """

    name: str = ""
    model_type: str = ""
    task: str = ""
    version: str = ""
    hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serialisable dictionary."""
        d: dict[str, Any] = {"name": self.name}
        if self.model_type:
            d["model_type"] = self.model_type
        if self.task:
            d["task"] = self.task
        if self.version:
            d["version"] = self.version
        if self.hash:
            d["hash"] = self.hash
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


@dataclass
class GraphRecord:
    """Record of a compiled graph's structure.

    Attributes:
        name: Graph name.
        graph_hash: Hash of the graph structure.
        num_nodes: Number of nodes in the graph.
        num_stages: Number of execution stages.
        node_names: Ordered list of node names.
        extra: Additional metadata.
    """

    name: str = ""
    graph_hash: str = ""
    num_nodes: int = 0
    num_stages: int = 0
    node_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serialisable dictionary."""
        d: dict[str, Any] = {
            "name": self.name,
            "graph_hash": self.graph_hash,
            "num_nodes": self.num_nodes,
            "num_stages": self.num_stages,
            "node_names": list(self.node_names),
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


class ProvenanceTracker:
    """Track model versions, configs, and graph metadata.

    Provides a structured record of everything used during graph execution
    so results can be understood, reproduced, or audited.

    Example:
        ```python
        from mata.core.observability import ProvenanceTracker

        tracker = ProvenanceTracker()

        # Record a model
        tracker.record_model("detector", adapter)
        tracker.record_model("segmenter", sam_adapter)

        # Record the graph
        tracker.record_graph(compiled_graph)

        # Get full provenance
        prov = tracker.get_provenance()
        print(prov["models"])
        print(prov["graph"])
        print(prov["timestamp"])

        # Export
        json_str = tracker.export("json")
        ```
    """

    def __init__(self) -> None:
        """Initialise empty provenance tracker."""
        self._models: dict[str, ModelRecord] = {}
        self._graph: GraphRecord | None = None
        self._extra: dict[str, Any] = {}
        self._timestamp: str = datetime.now().isoformat()

    # ── Model recording ──────────────────────────────────────────────

    def record_model(self, name: str, adapter: Any) -> None:
        """Record a model/adapter used during execution.

        Automatically extracts metadata such as ``model_id``, ``model_type``,
        ``task``, and ``device`` from common adapter attributes.

        Args:
            name: Logical name for the model (e.g. ``"detector"``).
            adapter: Adapter instance (any object with typical adapter attrs).
        """
        record = ModelRecord(name=name)

        # Extract model_id
        model_id = getattr(adapter, "model_id", None)
        if model_id:
            record.name = str(model_id)

        # Extract model type / runtime
        model_type = getattr(adapter, "model_type", None) or getattr(adapter, "runtime", None)
        if model_type:
            record.model_type = str(model_type)

        # Extract task
        task = getattr(adapter, "task", None)
        if task:
            record.task = str(task)

        # Extract version
        version = getattr(adapter, "version", None) or getattr(adapter, "model_version", None)
        if version:
            record.version = str(version)

        # Compute hash from repr or id
        record.hash = self._compute_adapter_hash(adapter)

        # Additional metadata
        device = getattr(adapter, "device", None)
        if device:
            record.extra["device"] = str(device)

        dtype = getattr(adapter, "dtype", None)
        if dtype:
            record.extra["dtype"] = str(dtype)

        self._models[name] = record
        logger.debug(f"Recorded model '{name}': {record.name}")

    def record_model_info(
        self,
        name: str,
        model_id: str,
        *,
        model_type: str = "",
        task: str = "",
        version: str = "",
        model_hash: str = "",
        **extra: Any,
    ) -> None:
        """Record model metadata directly (no adapter introspection).

        Args:
            name: Logical name for the model.
            model_id: Model identifier (HuggingFace ID, path, etc.).
            model_type: Runtime type.
            task: Task served by the model.
            version: Model version.
            model_hash: Content hash.
            **extra: Additional key-value metadata.
        """
        record = ModelRecord(
            name=model_id,
            model_type=model_type,
            task=task,
            version=version,
            hash=model_hash,
            extra=dict(extra),
        )
        self._models[name] = record

    # ── Graph recording ──────────────────────────────────────────────

    def record_graph(self, graph: Any) -> None:
        """Record compiled graph metadata.

        Extracts name, number of nodes, execution stages, and computes
        a structural hash.

        Args:
            graph: A ``CompiledGraph`` instance (or any object with
                   ``name``, ``nodes``, ``execution_order``, ``wiring``).
        """
        name = getattr(graph, "name", "unknown")
        nodes = getattr(graph, "nodes", [])
        execution_order = getattr(graph, "execution_order", [])
        wiring = getattr(graph, "wiring", {})

        node_names = [getattr(n, "name", str(n)) for n in nodes]
        graph_hash = self._compute_graph_hash(name, node_names, wiring)

        self._graph = GraphRecord(
            name=str(name),
            graph_hash=graph_hash,
            num_nodes=len(nodes),
            num_stages=len(execution_order),
            node_names=node_names,
        )
        logger.debug(f"Recorded graph '{name}' ({len(nodes)} nodes)")

    def record_graph_info(
        self,
        name: str,
        graph_hash: str = "",
        num_nodes: int = 0,
        num_stages: int = 0,
        node_names: list[str] | None = None,
        **extra: Any,
    ) -> None:
        """Record graph metadata directly.

        Args:
            name: Graph name.
            graph_hash: Structural hash.
            num_nodes: Number of nodes.
            num_stages: Number of execution stages.
            node_names: List of node names.
            **extra: Additional key-value metadata.
        """
        self._graph = GraphRecord(
            name=name,
            graph_hash=graph_hash,
            num_nodes=num_nodes,
            num_stages=num_stages,
            node_names=node_names or [],
            extra=dict(extra),
        )

    # ── Extra metadata ───────────────────────────────────────────────

    def record_extra(self, key: str, value: Any) -> None:
        """Record additional provenance metadata.

        Args:
            key: Metadata key.
            value: Metadata value (should be JSON-serialisable).
        """
        self._extra[key] = value

    # ── Querying ─────────────────────────────────────────────────────

    def get_provenance(self) -> dict[str, Any]:
        """Get full provenance record.

        Returns:
            Dictionary containing:
            - models: Per-model metadata.
            - graph: Graph structure metadata (if recorded).
            - timestamp: ISO timestamp of tracker creation.
            - extra: Any additional metadata.
        """
        result: dict[str, Any] = {
            "models": {name: record.to_dict() for name, record in self._models.items()},
            "timestamp": self._timestamp,
        }

        if self._graph is not None:
            result["graph"] = self._graph.to_dict()

        if self._extra:
            result.update(self._extra)

        return result

    def get_model(self, name: str) -> ModelRecord | None:
        """Get a specific model record.

        Args:
            name: Logical model name.

        Returns:
            ``ModelRecord`` or ``None``.
        """
        return self._models.get(name)

    @property
    def model_names(self) -> list[str]:
        """List of recorded model names."""
        return list(self._models.keys())

    # ── Export ────────────────────────────────────────────────────────

    def export(self, fmt: str = "json") -> str:
        """Export provenance in the given format.

        Args:
            fmt: ``"json"`` or ``"csv"``.

        Returns:
            Formatted string.

        Raises:
            ValueError: If format is not supported.
        """
        fmt = fmt.lower()
        if fmt == "json":
            return json.dumps(self.get_provenance(), indent=2)
        elif fmt == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'csv'.")

    def _export_csv(self) -> str:
        """Export models portion as CSV."""
        import csv as _csv
        import io

        buf = io.StringIO()
        fieldnames = ["logical_name", "model_name", "model_type", "task", "version", "hash"]
        writer = _csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for logical_name, record in self._models.items():
            writer.writerow(
                {
                    "logical_name": logical_name,
                    "model_name": record.name,
                    "model_type": record.model_type,
                    "task": record.task,
                    "version": record.version,
                    "hash": record.hash,
                }
            )

        return buf.getvalue()

    # ── Hashing helpers ──────────────────────────────────────────────

    @staticmethod
    def _compute_adapter_hash(adapter: Any) -> str:
        """Compute a hash representing an adapter's identity.

        Uses ``model_id`` + class name as the hash source, falling back
        to ``repr``.
        """
        model_id = getattr(adapter, "model_id", "")
        class_name = type(adapter).__name__
        raw = f"{class_name}:{model_id}" if model_id else repr(adapter)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _compute_graph_hash(
        name: str,
        node_names: list[str],
        wiring: Any,
    ) -> str:
        """Compute a structural hash of a graph."""
        parts = [name] + node_names
        if isinstance(wiring, dict):
            for k, v in sorted(wiring.items()):
                parts.append(f"{k}={v}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ── Utilities ────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all provenance data."""
        self._models.clear()
        self._graph = None
        self._extra.clear()
        self._timestamp = datetime.now().isoformat()

    def __repr__(self) -> str:
        graph_info = f", graph='{self._graph.name}'" if self._graph else ""
        return f"ProvenanceTracker(models={len(self._models)}{graph_info})"
