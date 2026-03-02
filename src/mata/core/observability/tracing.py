"""Execution tracing for MATA graph execution.

Records hierarchical spans for each node execution, enabling
timing analysis and execution flow visualisation.
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """A single tracing span representing a unit of work.

    Attributes:
        span_id: Unique span identifier.
        name: Human-readable span name (typically the node name).
        parent_id: ID of the parent span, if any.
        start_time: Unix timestamp when the span started.
        end_time: Unix timestamp when the span ended.
        attributes: User-defined key-value attributes.
        status: Span status (``"ok"`` or ``"error"``).
        error_message: Error details (set when status is ``"error"``).
    """

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    parent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        """Duration of the span in milliseconds."""
        if self.end_time <= 0 or self.start_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def is_finished(self) -> bool:
        """Whether the span has been ended."""
        return self.end_time > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert span to serialisable dictionary."""
        d: dict[str, Any] = {
            "span_id": self.span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.attributes:
            d["attributes"] = dict(self.attributes)
        if self.error_message:
            d["error_message"] = self.error_message
        return d

    def __repr__(self) -> str:
        status_icon = "✓" if self.status == "ok" else "✗"
        finished = f"{self.duration_ms:.2f}ms" if self.is_finished else "running"
        return f"Span({status_icon} {self.name}, {finished})"


class ExecutionTracer:
    """Trace graph execution with hierarchical spans.

    Creates a tree of spans that mirrors the execution flow of a graph,
    with one root span per graph execution and child spans for each node.

    Example:
        ```python
        from mata.core.observability import ExecutionTracer

        tracer = ExecutionTracer()

        # Root span
        root = tracer.start_span("graph_execution")

        # Child span for a node
        node_span = tracer.start_span("detect_node", parent_id=root.span_id)
        # ... run detection ...
        tracer.end_span(node_span)

        tracer.end_span(root)

        # Export
        print(tracer.export_trace("json"))
        ```
    """

    def __init__(self) -> None:
        """Initialise empty execution tracer."""
        self._spans: list[Span] = []
        self._active_spans: dict[str, Span] = {}  # span_id -> Span

    # ── Span lifecycle ───────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new tracing span.

        Args:
            name: Span name (typically node name or stage name).
            parent_id: Parent span ID for hierarchical tracing.
            attributes: Initial key-value attributes for the span.

        Returns:
            The newly created ``Span``.
        """
        span = Span(
            name=name,
            parent_id=parent_id,
            start_time=time.perf_counter(),
            attributes=attributes or {},
        )
        self._spans.append(span)
        self._active_spans[span.span_id] = span
        logger.debug(f"Span started: {span.name} ({span.span_id})")
        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        error_message: str | None = None,
    ) -> None:
        """End a tracing span.

        Args:
            span: The span to end.
            status: Final status (``"ok"`` or ``"error"``).
            error_message: Error detail string if status is ``"error"``.
        """
        # Using object.__setattr__ because Span is a plain dataclass
        # (not frozen), so direct assignment works just fine.
        span.end_time = time.perf_counter()
        span.status = status
        if error_message:
            span.error_message = error_message
        self._active_spans.pop(span.span_id, None)
        logger.debug(f"Span ended: {span.name} ({span.span_id}) " f"[{span.duration_ms:.2f}ms, {status}]")

    def add_span_attribute(self, span: Span, key: str, value: Any) -> None:
        """Add an attribute to an existing span.

        Args:
            span: Target span.
            key: Attribute key.
            value: Attribute value (should be JSON-serialisable).
        """
        span.attributes[key] = value

    # ── Querying ─────────────────────────────────────────────────────

    @property
    def spans(self) -> list[Span]:
        """All recorded spans (finished and active)."""
        return list(self._spans)

    @property
    def active_spans(self) -> list[Span]:
        """Currently active (unfinished) spans."""
        return list(self._active_spans.values())

    def get_span(self, span_id: str) -> Span | None:
        """Get span by ID.

        Args:
            span_id: Span identifier.

        Returns:
            The ``Span`` or ``None`` if not found.
        """
        for s in self._spans:
            if s.span_id == span_id:
                return s
        return None

    def get_children(self, parent_id: str) -> list[Span]:
        """Get child spans of a given parent.

        Args:
            parent_id: Parent span ID.

        Returns:
            List of child spans.
        """
        return [s for s in self._spans if s.parent_id == parent_id]

    def get_root_spans(self) -> list[Span]:
        """Get root spans (spans with no parent)."""
        return [s for s in self._spans if s.parent_id is None]

    # ── Export ────────────────────────────────────────────────────────

    def export_trace(self, fmt: str = "json") -> str:
        """Export the entire trace.

        Args:
            fmt: Export format — ``"json"`` or ``"csv"``.

        Returns:
            Formatted string.

        Raises:
            ValueError: If the format is not supported.
        """
        fmt = fmt.lower()
        if fmt == "json":
            return self._export_json()
        elif fmt == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'csv'.")

    def _export_json(self) -> str:
        """Export trace as JSON."""
        data = {
            "spans": [s.to_dict() for s in self._spans],
            "total_spans": len(self._spans),
            "root_spans": len(self.get_root_spans()),
        }
        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        """Export trace as CSV."""
        buf = io.StringIO()
        fieldnames = [
            "span_id",
            "name",
            "parent_id",
            "start_time",
            "end_time",
            "duration_ms",
            "status",
            "error_message",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for span in self._spans:
            writer.writerow(
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "parent_id": span.parent_id or "",
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "error_message": span.error_message or "",
                }
            )

        return buf.getvalue()

    # ── Utilities ────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all spans."""
        self._spans.clear()
        self._active_spans.clear()

    def __len__(self) -> int:
        """Total number of recorded spans."""
        return len(self._spans)

    def __repr__(self) -> str:
        finished = sum(1 for s in self._spans if s.is_finished)
        active = len(self._active_spans)
        return f"ExecutionTracer(spans={len(self._spans)}, " f"finished={finished}, active={active})"
