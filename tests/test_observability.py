"""Tests for MATA observability system (Task 6.4).

Covers MetricsCollector, ExecutionTracer, ProvenanceTracker,
export formats, and integration with ExecutionContext / scheduler.
"""

from __future__ import annotations

import csv
import io
import json
import time
from unittest.mock import MagicMock

import pytest

from mata.core.observability.metrics import MetricsCollector, NodeMetrics
from mata.core.observability.provenance import (
    GraphRecord,
    ModelRecord,
    ProvenanceTracker,
)
from mata.core.observability.tracing import ExecutionTracer, Span

# ══════════════════════════════════════════════════════════════════════
# MetricsCollector tests
# ══════════════════════════════════════════════════════════════════════


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_default_values(self):
        m = NodeMetrics()
        assert m.latency_ms == 0.0
        assert m.memory_mb == 0.0
        assert m.gpu_memory_mb == 0.0
        assert m.custom == {}

    def test_to_dict(self):
        m = NodeMetrics(latency_ms=10.5, memory_mb=256.0, gpu_memory_mb=512.0)
        d = m.to_dict()
        assert d["latency_ms"] == 10.5
        assert d["memory_mb"] == 256.0
        assert d["gpu_memory_mb"] == 512.0
        assert "custom" not in d  # empty custom omitted

    def test_to_dict_with_custom(self):
        m = NodeMetrics(custom={"num_detections": 5.0})
        d = m.to_dict()
        assert d["custom"] == {"num_detections": 5.0}


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_empty_collector(self):
        mc = MetricsCollector()
        assert len(mc) == 0
        assert mc.nodes == []
        summary = mc.get_summary()
        assert summary["total_latency_ms"] == 0.0
        assert summary["num_nodes"] == 0

    def test_record_latency(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.2)
        assert mc.get_node_metrics("detect").latency_ms == 45.2

    def test_record_memory(self):
        mc = MetricsCollector()
        mc.record_memory("detect", 512.0)
        assert mc.get_node_metrics("detect").memory_mb == 512.0

    def test_record_gpu_memory(self):
        mc = MetricsCollector()
        mc.record_gpu_memory("detect", 1024.0)
        assert mc.get_node_metrics("detect").gpu_memory_mb == 1024.0

    def test_record_custom(self):
        mc = MetricsCollector()
        mc.record("detect", "num_detections", 5)
        nm = mc.get_node_metrics("detect")
        assert nm.custom["num_detections"] == 5.0

    def test_multiple_nodes(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.0)
        mc.record_latency("segment", 120.0)
        mc.record_latency("classify", 10.0)
        assert len(mc) == 3
        assert set(mc.nodes) == {"detect", "segment", "classify"}

    def test_get_summary_aggregates(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.0)
        mc.record_latency("segment", 120.0)
        mc.record_memory("detect", 512.0)
        mc.record_memory("segment", 1024.0)
        mc.record_gpu_memory("detect", 200.0)
        mc.record_gpu_memory("segment", 800.0)

        summary = mc.get_summary()
        assert summary["total_latency_ms"] == pytest.approx(165.0)
        assert summary["peak_memory_mb"] == pytest.approx(1024.0)
        assert summary["peak_gpu_memory_mb"] == pytest.approx(800.0)
        assert summary["num_nodes"] == 2
        assert "detect" in summary["per_node"]
        assert "segment" in summary["per_node"]

    def test_wall_time(self):
        mc = MetricsCollector()
        mc.start()
        time.sleep(0.05)  # 50 ms
        mc.stop()
        assert mc.wall_time_ms > 40  # allow some tolerance
        assert mc.wall_time_ms < 500

    def test_wall_time_not_started(self):
        mc = MetricsCollector()
        assert mc.wall_time_ms == 0.0

    def test_get_node_metrics_not_found(self):
        mc = MetricsCollector()
        assert mc.get_node_metrics("nonexistent") is None

    def test_clear(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 10.0)
        mc.start()
        mc.stop()
        mc.clear()
        assert len(mc) == 0
        assert mc.wall_time_ms == 0.0

    def test_merge(self):
        mc1 = MetricsCollector()
        mc1.record_latency("detect", 10.0)

        mc2 = MetricsCollector()
        mc2.record_latency("segment", 20.0)

        mc1.merge(mc2)
        assert len(mc1) == 2
        assert mc1.get_node_metrics("segment").latency_ms == 20.0

    def test_merge_overwrites(self):
        mc1 = MetricsCollector()
        mc1.record_latency("detect", 10.0)

        mc2 = MetricsCollector()
        mc2.record_latency("detect", 99.0)

        mc1.merge(mc2)
        assert mc1.get_node_metrics("detect").latency_ms == 99.0

    def test_repr(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.0)
        r = repr(mc)
        assert "MetricsCollector" in r
        assert "nodes=1" in r

    # ── Export tests ─────────────────────────────────────────

    def test_export_json(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.0)
        mc.record("detect", "count", 3)

        raw = mc.export("json")
        data = json.loads(raw)
        assert data["total_latency_ms"] == pytest.approx(45.0)
        assert data["per_node"]["detect"]["latency_ms"] == 45.0

    def test_export_csv(self):
        mc = MetricsCollector()
        mc.record_latency("detect", 45.0)
        mc.record_memory("detect", 512.0)
        mc.record("detect", "count", 3)

        raw = mc.export("csv")
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["node"] == "detect"
        assert float(rows[0]["latency_ms"]) == pytest.approx(45.0)
        assert float(rows[0]["memory_mb"]) == pytest.approx(512.0)
        assert float(rows[0]["count"]) == pytest.approx(3.0)

    def test_export_csv_multiple_custom_keys(self):
        mc = MetricsCollector()
        mc.record("detect", "count", 3)
        mc.record("segment", "mask_count", 5)

        raw = mc.export("csv")
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        assert len(rows) == 2
        # Both custom columns should be present
        assert "count" in reader.fieldnames
        assert "mask_count" in reader.fieldnames

    def test_export_unsupported_format(self):
        mc = MetricsCollector()
        with pytest.raises(ValueError, match="Unsupported"):
            mc.export("xml")


# ══════════════════════════════════════════════════════════════════════
# ExecutionTracer tests
# ══════════════════════════════════════════════════════════════════════


class TestSpan:
    """Tests for Span dataclass."""

    def test_default_span(self):
        s = Span()
        assert s.span_id  # auto-generated
        assert s.name == ""
        assert s.parent_id is None
        assert s.status == "ok"
        assert s.is_finished is False
        assert s.duration_ms == 0.0

    def test_duration_calculation(self):
        s = Span(start_time=100.0, end_time=100.05)
        assert s.duration_ms == pytest.approx(50.0)

    def test_to_dict_minimal(self):
        s = Span(name="test", start_time=1.0, end_time=2.0)
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["duration_ms"] == pytest.approx(1000.0)
        assert "parent_id" not in d
        assert "attributes" not in d
        assert "error_message" not in d

    def test_to_dict_full(self):
        s = Span(
            name="test",
            parent_id="abc",
            start_time=1.0,
            end_time=2.0,
            attributes={"key": "val"},
            status="error",
            error_message="boom",
        )
        d = s.to_dict()
        assert d["parent_id"] == "abc"
        assert d["attributes"] == {"key": "val"}
        assert d["error_message"] == "boom"
        assert d["status"] == "error"

    def test_repr(self):
        s = Span(name="detect", start_time=1.0, end_time=1.045)
        r = repr(s)
        assert "detect" in r
        assert "✓" in r


class TestExecutionTracer:
    """Tests for ExecutionTracer."""

    def test_empty_tracer(self):
        t = ExecutionTracer()
        assert len(t) == 0
        assert t.spans == []
        assert t.active_spans == []

    def test_start_and_end_span(self):
        t = ExecutionTracer()
        span = t.start_span("detect_node")
        assert span.name == "detect_node"
        assert span.start_time > 0
        assert not span.is_finished
        assert len(t.active_spans) == 1

        t.end_span(span)
        assert span.is_finished
        assert span.status == "ok"
        assert len(t.active_spans) == 0
        assert len(t) == 1

    def test_span_with_parent(self):
        t = ExecutionTracer()
        root = t.start_span("graph")
        child = t.start_span("detect", parent_id=root.span_id)
        assert child.parent_id == root.span_id

    def test_span_with_attributes(self):
        t = ExecutionTracer()
        span = t.start_span("detect", attributes={"model": "detr"})
        assert span.attributes["model"] == "detr"

    def test_add_span_attribute(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        t.add_span_attribute(span, "count", 5)
        assert span.attributes["count"] == 5

    def test_end_span_with_error(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        t.end_span(span, status="error", error_message="CUDA OOM")
        assert span.status == "error"
        assert span.error_message == "CUDA OOM"

    def test_get_span_by_id(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        found = t.get_span(span.span_id)
        assert found is span

    def test_get_span_not_found(self):
        t = ExecutionTracer()
        assert t.get_span("nonexistent") is None

    def test_get_children(self):
        t = ExecutionTracer()
        root = t.start_span("graph")
        c1 = t.start_span("detect", parent_id=root.span_id)
        c2 = t.start_span("segment", parent_id=root.span_id)
        _unrelated = t.start_span("other")

        children = t.get_children(root.span_id)
        assert len(children) == 2
        assert c1 in children
        assert c2 in children

    def test_get_root_spans(self):
        t = ExecutionTracer()
        r1 = t.start_span("graph1")
        _child = t.start_span("node", parent_id=r1.span_id)
        r2 = t.start_span("graph2")

        roots = t.get_root_spans()
        assert len(roots) == 2
        assert r1 in roots
        assert r2 in roots

    def test_clear(self):
        t = ExecutionTracer()
        t.start_span("detect")
        t.clear()
        assert len(t) == 0
        assert t.active_spans == []

    def test_repr(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        t.end_span(span)
        t.start_span("segment")  # still active
        r = repr(t)
        assert "spans=2" in r
        assert "finished=1" in r
        assert "active=1" in r

    # ── Hierarchical tracing ─────────────────────────────────

    def test_hierarchical_trace(self):
        t = ExecutionTracer()
        root = t.start_span("graph:main")
        d = t.start_span("node:detect", parent_id=root.span_id)
        time.sleep(0.01)
        t.end_span(d)
        s = t.start_span("node:segment", parent_id=root.span_id)
        time.sleep(0.01)
        t.end_span(s)
        t.end_span(root)

        assert len(t) == 3
        assert root.duration_ms > d.duration_ms
        children = t.get_children(root.span_id)
        assert len(children) == 2

    # ── Export tests ─────────────────────────────────────────

    def test_export_json(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        t.end_span(span)

        raw = t.export_trace("json")
        data = json.loads(raw)
        assert data["total_spans"] == 1
        assert data["root_spans"] == 1
        assert len(data["spans"]) == 1
        assert data["spans"][0]["name"] == "detect"

    def test_export_csv(self):
        t = ExecutionTracer()
        span = t.start_span("detect")
        t.end_span(span)

        raw = t.export_trace("csv")
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "detect"
        assert rows[0]["status"] == "ok"

    def test_export_unsupported_format(self):
        t = ExecutionTracer()
        with pytest.raises(ValueError, match="Unsupported"):
            t.export_trace("xml")


# ══════════════════════════════════════════════════════════════════════
# ProvenanceTracker tests
# ══════════════════════════════════════════════════════════════════════


class TestModelRecord:
    """Tests for ModelRecord dataclass."""

    def test_default(self):
        r = ModelRecord()
        assert r.name == ""
        assert r.hash == ""

    def test_to_dict_minimal(self):
        r = ModelRecord(name="detr")
        d = r.to_dict()
        assert d == {"name": "detr"}

    def test_to_dict_full(self):
        r = ModelRecord(
            name="detr",
            model_type="huggingface",
            task="detect",
            version="1.0",
            hash="abc123",
            extra={"device": "cuda"},
        )
        d = r.to_dict()
        assert d["model_type"] == "huggingface"
        assert d["task"] == "detect"
        assert d["extra"]["device"] == "cuda"


class TestGraphRecord:
    """Tests for GraphRecord dataclass."""

    def test_to_dict(self):
        r = GraphRecord(
            name="test_graph",
            graph_hash="def456",
            num_nodes=3,
            num_stages=2,
            node_names=["detect", "filter", "segment"],
        )
        d = r.to_dict()
        assert d["name"] == "test_graph"
        assert d["num_nodes"] == 3
        assert len(d["node_names"]) == 3


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""

    def test_empty_tracker(self):
        pt = ProvenanceTracker()
        prov = pt.get_provenance()
        assert prov["models"] == {}
        assert "timestamp" in prov

    def test_record_model_from_adapter(self):
        """Test recording model from a mock adapter with common attributes."""
        adapter = MagicMock()
        adapter.model_id = "facebook/detr-resnet-50"
        adapter.model_type = "huggingface"
        adapter.task = "detect"
        adapter.version = "1.0.0"
        adapter.device = "cuda"
        adapter.dtype = "float32"

        pt = ProvenanceTracker()
        pt.record_model("detector", adapter)

        record = pt.get_model("detector")
        assert record is not None
        assert record.name == "facebook/detr-resnet-50"
        assert record.model_type == "huggingface"
        assert record.task == "detect"
        assert record.version == "1.0.0"
        assert record.extra["device"] == "cuda"
        assert record.extra["dtype"] == "float32"
        assert record.hash  # non-empty hash

    def test_record_model_minimal_adapter(self):
        """Test recording model from adapter with no common attributes."""
        adapter = object()  # bare object

        pt = ProvenanceTracker()
        pt.record_model("model", adapter)

        record = pt.get_model("model")
        assert record is not None
        assert record.hash  # still gets a hash from repr

    def test_record_model_info_direct(self):
        pt = ProvenanceTracker()
        pt.record_model_info(
            "detector",
            "facebook/detr-resnet-50",
            model_type="huggingface",
            task="detect",
            version="1.0.0",
            model_hash="abc123",
        )

        record = pt.get_model("detector")
        assert record.name == "facebook/detr-resnet-50"
        assert record.hash == "abc123"

    def test_record_graph(self):
        """Test recording graph from a mock CompiledGraph."""
        graph = MagicMock()
        graph.name = "detect_segment"

        node1 = MagicMock()
        node1.name = "detect"
        node2 = MagicMock()
        node2.name = "segment"
        graph.nodes = [node1, node2]
        graph.execution_order = [[node1], [node2]]
        graph.wiring = {"segment.image": "input.image"}

        pt = ProvenanceTracker()
        pt.record_graph(graph)

        prov = pt.get_provenance()
        assert "graph" in prov
        assert prov["graph"]["name"] == "detect_segment"
        assert prov["graph"]["num_nodes"] == 2
        assert prov["graph"]["num_stages"] == 2
        assert prov["graph"]["graph_hash"]  # non-empty

    def test_record_graph_info_direct(self):
        pt = ProvenanceTracker()
        pt.record_graph_info(
            "my_graph",
            graph_hash="xyz",
            num_nodes=5,
            num_stages=3,
            node_names=["a", "b", "c", "d", "e"],
        )

        prov = pt.get_provenance()
        assert prov["graph"]["name"] == "my_graph"
        assert prov["graph"]["graph_hash"] == "xyz"

    def test_record_extra(self):
        pt = ProvenanceTracker()
        pt.record_extra("experiment", "baseline_v1")
        prov = pt.get_provenance()
        assert prov["experiment"] == "baseline_v1"

    def test_model_names(self):
        pt = ProvenanceTracker()
        pt.record_model_info("detector", "detr")
        pt.record_model_info("segmenter", "sam")
        assert set(pt.model_names) == {"detector", "segmenter"}

    def test_get_model_not_found(self):
        pt = ProvenanceTracker()
        assert pt.get_model("nonexistent") is None

    def test_multiple_models(self):
        pt = ProvenanceTracker()
        a1 = MagicMock()
        a1.model_id = "detr"
        a2 = MagicMock()
        a2.model_id = "sam"

        pt.record_model("detector", a1)
        pt.record_model("segmenter", a2)

        prov = pt.get_provenance()
        assert len(prov["models"]) == 2

    def test_clear(self):
        pt = ProvenanceTracker()
        pt.record_model_info("detector", "detr")
        pt.record_graph_info("graph", num_nodes=3)
        pt.record_extra("env", "test")
        pt.clear()

        prov = pt.get_provenance()
        assert prov["models"] == {}
        assert "graph" not in prov

    def test_repr(self):
        pt = ProvenanceTracker()
        pt.record_model_info("detector", "detr")
        r = repr(pt)
        assert "ProvenanceTracker" in r
        assert "models=1" in r

    # ── Export tests ─────────────────────────────────────────

    def test_export_json(self):
        pt = ProvenanceTracker()
        pt.record_model_info("detector", "detr", task="detect")

        raw = pt.export("json")
        data = json.loads(raw)
        assert "models" in data
        assert data["models"]["detector"]["name"] == "detr"

    def test_export_csv(self):
        pt = ProvenanceTracker()
        pt.record_model_info("detector", "detr", task="detect", version="1.0")
        pt.record_model_info("segmenter", "sam", task="segment")

        raw = pt.export("csv")
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        assert len(rows) == 2
        names = {r["logical_name"] for r in rows}
        assert names == {"detector", "segmenter"}

    def test_export_unsupported_format(self):
        pt = ProvenanceTracker()
        with pytest.raises(ValueError, match="Unsupported"):
            pt.export("xml")


# ══════════════════════════════════════════════════════════════════════
# Integration with ExecutionContext
# ══════════════════════════════════════════════════════════════════════


class TestObservabilityContextIntegration:
    """Test that observability objects are available on ExecutionContext."""

    def test_context_has_metrics_collector(self):
        from mata.core.graph.context import ExecutionContext

        ctx = ExecutionContext()
        assert isinstance(ctx.metrics_collector, MetricsCollector)

    def test_context_has_tracer(self):
        from mata.core.graph.context import ExecutionContext

        ctx = ExecutionContext()
        assert isinstance(ctx.tracer, ExecutionTracer)

    def test_context_has_provenance_tracker(self):
        from mata.core.graph.context import ExecutionContext

        ctx = ExecutionContext()
        assert isinstance(ctx.provenance_tracker, ProvenanceTracker)

    def test_context_observability_is_per_instance(self):
        from mata.core.graph.context import ExecutionContext

        ctx1 = ExecutionContext()
        ctx2 = ExecutionContext()
        ctx1.metrics_collector.record_latency("node_a", 10)
        assert ctx2.metrics_collector.get_node_metrics("node_a") is None

    def test_context_record_metric_and_collector_independent(self):
        """Existing record_metric stays untouched; collector is separate."""
        from mata.core.graph.context import ExecutionContext

        ctx = ExecutionContext()
        ctx.record_metric("node_a", "latency_ms", 42)
        # The existing dict-based metrics still work
        assert ctx.get_metrics()["node_a"]["latency_ms"] == 42.0
        # MetricsCollector is independent if not used via scheduler
        assert ctx.metrics_collector.get_node_metrics("node_a") is None


# ══════════════════════════════════════════════════════════════════════
# Integration with Scheduler
# ══════════════════════════════════════════════════════════════════════


class TestObservabilitySchedulerIntegration:
    """Test that observability is populated during scheduler execution."""

    def _make_simple_graph(self):
        """Create minimal graph + context for scheduler tests."""
        from mata.core.artifacts.image import Image as ImageArtifact
        from mata.core.graph.context import ExecutionContext
        from mata.core.graph.graph import CompiledGraph
        from mata.core.graph.node import Node
        from mata.core.graph.validator import ValidationResult

        # A trivial node that passes through the image
        class PassthroughNode(Node):
            inputs = {"image": ImageArtifact}
            outputs = {"image": ImageArtifact}

            def __init__(self):
                super().__init__()
                self.name = "passthrough"

            def run(self, ctx, **kwargs):
                return {"image": kwargs["image"]}

        node = PassthroughNode()

        compiled = CompiledGraph(
            name="test_graph",
            nodes=[node],
            execution_order=[[node]],
            wiring={"passthrough.image": "input.image"},
            dag=None,
            validation_result=ValidationResult(valid=True),
        )

        # Create a tiny image artifact
        import numpy as np

        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        image_artifact = ImageArtifact(data=img_array, width=10, height=10)

        ctx = ExecutionContext(providers={})

        return compiled, ctx, image_artifact

    def test_scheduler_populates_metrics_collector(self):
        from mata.core.graph.scheduler import SyncScheduler

        compiled, ctx, image = self._make_simple_graph()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {"input.image": image})

        mc = ctx.metrics_collector
        assert mc.get_node_metrics("passthrough") is not None
        assert mc.get_node_metrics("passthrough").latency_ms > 0

    def test_scheduler_populates_tracer(self):
        from mata.core.graph.scheduler import SyncScheduler

        compiled, ctx, image = self._make_simple_graph()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {"input.image": image})

        tracer = ctx.tracer
        assert len(tracer) >= 2  # root span + node span
        roots = tracer.get_root_spans()
        assert len(roots) == 1
        assert "graph:test_graph" in roots[0].name
        # Node span is child of root
        children = tracer.get_children(roots[0].span_id)
        assert len(children) == 1
        assert "passthrough" in children[0].name

    def test_scheduler_populates_provenance(self):
        from mata.core.graph.scheduler import SyncScheduler

        compiled, ctx, image = self._make_simple_graph()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {"input.image": image})

        prov = ctx.provenance_tracker.get_provenance()
        assert "graph" in prov
        assert prov["graph"]["name"] == "test_graph"

    def test_scheduler_records_wall_time(self):
        from mata.core.graph.scheduler import SyncScheduler

        compiled, ctx, image = self._make_simple_graph()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {"input.image": image})

        assert ctx.metrics_collector.wall_time_ms > 0

    def test_scheduler_traces_error(self):
        import numpy as np

        from mata.core.artifacts.image import Image as ImageArtifact
        from mata.core.graph.context import ExecutionContext
        from mata.core.graph.graph import CompiledGraph
        from mata.core.graph.node import Node
        from mata.core.graph.scheduler import SyncScheduler
        from mata.core.graph.validator import ValidationResult

        class FailingNode(Node):
            inputs = {"image": ImageArtifact}
            outputs = {"image": ImageArtifact}

            def __init__(self):
                super().__init__()
                self.name = "failing"

            def run(self, ctx, **kwargs):
                raise RuntimeError("intentional failure")

        node = FailingNode()
        compiled = CompiledGraph(
            name="fail_graph",
            nodes=[node],
            execution_order=[[node]],
            wiring={"failing.image": "input.image"},
            dag=None,
            validation_result=ValidationResult(valid=True),
        )

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        image_artifact = ImageArtifact(data=img, width=10, height=10)
        ctx = ExecutionContext(providers={})

        scheduler = SyncScheduler()
        with pytest.raises(RuntimeError, match="intentional failure"):
            scheduler.execute(compiled, ctx, {"input.image": image_artifact})

        # Root span should be marked error
        roots = ctx.tracer.get_root_spans()
        assert len(roots) == 1
        assert roots[0].status == "error"

        # Node span should also be error
        children = ctx.tracer.get_children(roots[0].span_id)
        assert len(children) == 1
        assert children[0].status == "error"
        assert "intentional failure" in children[0].error_message

    def test_scheduler_records_provider_provenance(self):
        import numpy as np

        from mata.core.artifacts.image import Image as ImageArtifact
        from mata.core.graph.context import ExecutionContext
        from mata.core.graph.graph import CompiledGraph
        from mata.core.graph.node import Node
        from mata.core.graph.scheduler import SyncScheduler
        from mata.core.graph.validator import ValidationResult

        class DummyNode(Node):
            inputs = {"image": ImageArtifact}
            outputs = {"image": ImageArtifact}

            def __init__(self):
                super().__init__()
                self.name = "dummy"

            def run(self, ctx, **kwargs):
                return {"image": kwargs["image"]}

        node = DummyNode()
        compiled = CompiledGraph(
            name="prov_graph",
            nodes=[node],
            execution_order=[[node]],
            wiring={"dummy.image": "input.image"},
            dag=None,
            validation_result=ValidationResult(valid=True),
        )

        # Mock adapter with model_id
        mock_adapter = MagicMock()
        mock_adapter.model_id = "facebook/detr-resnet-50"
        mock_adapter.task = "detect"

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        image_artifact = ImageArtifact(data=img, width=10, height=10)
        ctx = ExecutionContext(providers={"detect": {"detr": mock_adapter}})

        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {"input.image": image_artifact})

        prov = ctx.provenance_tracker.get_provenance()
        assert "detect.detr" in prov["models"]
        assert prov["models"]["detect.detr"]["name"] == "facebook/detr-resnet-50"


# ══════════════════════════════════════════════════════════════════════
# Import convenience tests
# ══════════════════════════════════════════════════════════════════════


class TestObservabilityImports:
    """Test that observability is importable from expected locations."""

    def test_import_from_observability_package(self):
        pass

    def test_import_from_graph_package(self):
        pass
