"""Comprehensive tests for graph control-flow: EarlyExit, While, conditional edges.

Tests cover:
- EarlyExitException attributes
- EarlyExit node (triggers / pass-through)
- While loop semantics (basic, do-while, max_iterations cap, empty condition)
- Graph.add(condition=...) — conditional edges
- Cascade-skip: downstream nodes skipped when dependency is skipped
- Graph.conditional() creates a single If node
- SyncScheduler end-to-end with EarlyExit + conditional edges
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mata.core.artifacts.base import Artifact
from mata.core.graph.conditionals import (
    EarlyExit,
    EarlyExitException,
    If,
    Pass,
    While,
)
from mata.core.graph.context import ExecutionContext
from mata.core.graph.graph import CompiledGraph, Graph
from mata.core.graph.node import Node
from mata.core.graph.scheduler import SyncScheduler

# ---------------------------------------------------------------------------
# Shared test artifacts / nodes
# ---------------------------------------------------------------------------


class FakeArtifact(Artifact):
    def __init__(self, value=None):
        self.value = value

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get("value"))


class SimpleNode(Node):
    """Node that stores what it was called with and stores an output."""

    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {"result": FakeArtifact}

    def __init__(self, name: str, output_value=None, side_effect=None):
        super().__init__(name=name)
        self.output_value = output_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args = []

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        self.call_count += 1
        self.call_args.append(dict(inputs))
        if self.side_effect:
            self.side_effect(ctx)
        artifact = FakeArtifact(self.output_value)
        ctx.store(f"{self.name}.result", artifact)
        return {"result": artifact}


class DownstreamNode(Node):
    """Node that declares a wired dependency on an upstream artifact."""

    inputs: dict[str, Any] = {"result": FakeArtifact}
    outputs: dict[str, Any] = {"out": FakeArtifact}

    def __init__(self, name: str):
        super().__init__(name=name)
        self.call_count = 0

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        self.call_count += 1
        artifact = FakeArtifact("downstream")
        ctx.store(f"{self.name}.out", artifact)
        return {"out": artifact}


def _make_context() -> ExecutionContext:
    return ExecutionContext(providers={})


# ===========================================================================
# EarlyExitException
# ===========================================================================


class TestEarlyExitException:
    def test_default_attributes(self):
        exc = EarlyExitException()
        assert exc.reason == "EarlyExit condition met"
        assert exc.node_name == "EarlyExit"

    def test_custom_reason_and_node_name(self):
        exc = EarlyExitException(reason="no detections", node_name="gate")
        assert exc.reason == "no detections"
        assert exc.node_name == "gate"

    def test_is_exception(self):
        exc = EarlyExitException()
        assert isinstance(exc, Exception)

    def test_str_contains_node_name(self):
        exc = EarlyExitException(reason="empty", node_name="my_gate")
        assert "my_gate" in str(exc)

    def test_str_contains_reason(self):
        exc = EarlyExitException(reason="low confidence", node_name="q")
        assert "low confidence" in str(exc)


# ===========================================================================
# EarlyExit node
# ===========================================================================


class TestEarlyExitNode:
    def test_name_default(self):
        node = EarlyExit(predicate=lambda ctx: False)
        assert node.name == "EarlyExit"

    def test_name_custom(self):
        node = EarlyExit(predicate=lambda ctx: False, name="gate")
        assert node.name == "gate"

    def test_passthrough_when_false(self):
        ctx = _make_context()
        node = EarlyExit(predicate=lambda ctx: False)
        result = node.run(ctx)
        assert result == {}

    def test_raises_when_true(self):
        ctx = _make_context()
        node = EarlyExit(predicate=lambda ctx: True, reason="stop now", name="gate")
        with pytest.raises(EarlyExitException) as exc_info:
            node.run(ctx)
        assert exc_info.value.reason == "stop now"
        assert exc_info.value.node_name == "gate"

    def test_records_metrics(self):
        ctx = _make_context()
        node = EarlyExit(predicate=lambda ctx: False, name="q")
        node.run(ctx)
        metrics = ctx.get_metrics().get("q", {})
        assert "predicate_latency_ms" in metrics
        assert metrics["condition_result"] == 0.0  # stored as float

    def test_records_early_exit_metric_when_triggered(self):
        ctx = _make_context()
        node = EarlyExit(predicate=lambda ctx: True, name="q")
        with pytest.raises(EarlyExitException):
            node.run(ctx)
        metrics = ctx.get_metrics().get("q", {})
        assert metrics["early_exit_triggered"] == 1.0  # stored as float

    def test_no_inputs_no_outputs(self):
        node = EarlyExit(predicate=lambda ctx: False)
        assert node.inputs == {}
        assert node.outputs == {}

    def test_predicate_receives_context(self):
        received = []
        ctx = _make_context()
        ctx.store("flag", FakeArtifact("yes"))

        def pred(c):
            received.append(c)
            return False

        node = EarlyExit(predicate=pred)
        node.run(ctx)
        assert received[0] is ctx

    def test_repr(self):
        node = EarlyExit(predicate=lambda ctx: False, reason="test", name="q")
        assert "q" in repr(node)
        assert "test" in repr(node)


# ===========================================================================
# While node
# ===========================================================================


class TestWhileNode:
    def test_requires_nonempty_body(self):
        with pytest.raises(ValueError, match="at least one"):
            While(body=[], condition=lambda ctx: False)

    def test_max_iterations_must_be_positive(self):
        node = SimpleNode("n")
        with pytest.raises(ValueError):
            While(body=[node], condition=lambda ctx: False, max_iterations=0)

    def test_basic_single_iteration(self):
        """Condition False after first pass: body runs exactly once."""
        node = SimpleNode("body", output_value="v1")
        ctx = _make_context()

        while_node = While(
            body=[node],
            condition=lambda ctx: False,  # stop after 1st pass
            max_iterations=5,
        )
        while_node.run(ctx)
        assert node.call_count == 1

    def test_do_while_semantics_body_always_runs_once(self):
        """Even if condition would be False immediately, body runs at least once."""
        node = SimpleNode("body")
        ctx = _make_context()

        # Condition checked AFTER first pass — body must run at least once
        iterations = []

        def cond(c):
            iterations.append(len(iterations))
            return False  # stop immediately after first body run

        while_node = While(body=[node], condition=cond, max_iterations=10)
        while_node.run(ctx)
        assert node.call_count == 1

    def test_multi_iteration(self):
        """Loop runs N times until condition returns False."""
        counts = [0]
        node = SimpleNode("body")

        def cond(ctx):
            counts[0] += 1
            return counts[0] < 3  # run 3 times total

        ctx = _make_context()
        while_node = While(body=[node], condition=cond, max_iterations=10)
        while_node.run(ctx)
        assert node.call_count == 3

    def test_max_iterations_cap(self):
        """Loop stops at max_iterations even if condition stays True."""
        node = SimpleNode("body")
        ctx = _make_context()
        while_node = While(
            body=[node],
            condition=lambda ctx: True,  # never stop naturally
            max_iterations=4,
        )
        while_node.run(ctx)
        assert node.call_count == 4

    def test_multi_node_body(self):
        """All body nodes run each iteration."""
        a = SimpleNode("a")
        b = SimpleNode("b")
        ctx = _make_context()

        iters = [0]

        def cond(c):
            iters[0] += 1
            return iters[0] < 2

        while_node = While(body=[a, b], condition=cond, max_iterations=5)
        while_node.run(ctx)
        assert a.call_count == 2
        assert b.call_count == 2

    def test_returns_outputs(self):
        """run() should return a non-empty dict of last-produced outputs."""
        node = SimpleNode("body", output_value="final")
        ctx = _make_context()
        while_node = While(body=[node], condition=lambda ctx: False, max_iterations=3)
        result = while_node.run(ctx)
        assert isinstance(result, dict)

    def test_default_name(self):
        node = SimpleNode("x")
        w = While(body=[node], condition=lambda ctx: False)
        assert w.name == "While"

    def test_custom_name(self):
        node = SimpleNode("x")
        w = While(body=[node], condition=lambda ctx: False, name="my_loop")
        assert w.name == "my_loop"

    def test_key_error_in_condition_stops_loop(self):
        """If condition raises KeyError the loop stops gracefully."""
        node = SimpleNode("body")
        ctx = _make_context()

        def cond(c):
            raise KeyError("missing")

        while_node = While(body=[node], condition=cond, max_iterations=5)
        while_node.run(ctx)
        # Should have run at least once but not crashed
        assert node.call_count >= 1

    def test_repr(self):
        node = SimpleNode("x")
        w = While(body=[node], condition=lambda ctx: False, name="loop")
        assert "loop" in repr(w) or "While" in repr(w)


# ===========================================================================
# Graph.conditional() — now wraps into a single If node
# ===========================================================================


class TestGraphConditional:
    def test_creates_if_node(self):
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("segment")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg)
        assert any(isinstance(n, If) for n in graph._nodes)

    def test_branch_not_added_directly(self):
        """Branch nodes must NOT appear as top-level graph nodes."""
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("segment")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg)
        assert seg not in graph._nodes

    def test_then_branch_attached(self):
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("segment")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg)
        if_node = next(n for n in graph._nodes if isinstance(n, If))
        assert if_node.then_branch is seg

    def test_else_branch_defaults_to_pass(self):
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("segment")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg, else_branch=None)
        if_node = next(n for n in graph._nodes if isinstance(n, If))
        assert isinstance(if_node.else_branch, Pass)

    def test_else_branch_explicit(self):
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("seg")
        cls = SimpleNode("cls")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg, else_branch=cls)
        if_node = next(n for n in graph._nodes if isinstance(n, If))
        assert if_node.else_branch is cls

    def test_graph_size_with_conditional(self):
        """detect + if_node = 2 nodes."""
        graph = Graph()
        det = SimpleNode("detect")
        seg = SimpleNode("segment")
        graph.then(det).conditional(predicate=lambda ctx: True, then_branch=seg)
        assert len(graph._nodes) == 2

    def test_chainable(self):
        graph = Graph()
        result = graph.conditional(predicate=lambda ctx: True, then_branch=SimpleNode("d1"))
        assert result is graph


# ===========================================================================
# Graph.add(condition=...) — conditional edges
# ===========================================================================


class TestConditionalEdges:
    def test_condition_stored_in_edge_conditions(self):
        graph = Graph()
        node = SimpleNode("check")

        def cond(ctx):
            return True

        graph.add(node, condition=cond)
        assert graph._edge_conditions[node.name] is cond

    def test_no_condition_not_stored(self):
        graph = Graph()
        node = SimpleNode("plain")
        graph.add(node)
        assert node.name not in graph._edge_conditions

    def test_edge_conditions_passed_to_compiled_graph(self):
        graph = Graph()
        node = SimpleNode("n")

        def cond(ctx):
            return True

        graph.add(node, condition=cond)
        compiled = graph.compile(providers={})
        assert compiled.edge_conditions[node.name] is cond

    def test_compiled_graph_edge_conditions_default_empty(self):
        graph = Graph()
        node = SimpleNode("n")
        graph.add(node)
        compiled = graph.compile(providers={})
        assert compiled.edge_conditions == {}


# ===========================================================================
# CompiledGraph.edge_conditions dataclass field
# ===========================================================================


class TestCompiledGraphEdgeConditions:
    def test_field_defaults_to_empty_dict(self):
        """CompiledGraph without edge_conditions gets an empty dict."""
        node = SimpleNode("n")
        vr = MagicMock()
        vr.valid = True
        compiled = CompiledGraph(
            name="test",
            nodes=[node],
            wiring={},
            dag=None,
            validation_result=vr,
        )
        assert compiled.edge_conditions == {}

    def test_field_set(self):
        def cond(ctx):
            return True

        vr = MagicMock()
        vr.valid = True
        node = SimpleNode("n")
        compiled = CompiledGraph(
            name="test",
            nodes=[node],
            wiring={},
            dag=None,
            validation_result=vr,
            edge_conditions={"n": cond},
        )
        assert compiled.edge_conditions["n"] is cond


# ===========================================================================
# SyncScheduler — EarlyExit integration (end-to-end)
# ===========================================================================


class TestSyncSchedulerEarlyExit:
    def _compile(self, nodes, wiring=None, edge_conditions=None):
        """Helper to produce a minimal CompiledGraph."""
        vr = MagicMock()
        vr.valid = True
        return CompiledGraph(
            name="g",
            nodes=nodes,
            execution_order=[[n] for n in nodes],  # sequential stages
            wiring=wiring or {},
            dag=None,
            validation_result=vr,
            edge_conditions=edge_conditions or {},
        )

    def test_early_exit_stops_downstream_nodes(self):
        """Nodes after EarlyExit are not executed."""
        gate = EarlyExit(predicate=lambda ctx: True, name="gate")
        after = SimpleNode("after")

        compiled = self._compile([gate, after])
        ctx = _make_context()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {})

        assert after.call_count == 0

    def test_early_exit_passthrough_executes_downstream(self):
        """When predicate is False, downstream nodes continue."""
        gate = EarlyExit(predicate=lambda ctx: False, name="gate")
        after = SimpleNode("after")

        compiled = self._compile([gate, after])
        ctx = _make_context()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {})

        assert after.call_count == 1

    def test_early_exit_returns_partial_result(self):
        gate = EarlyExit(predicate=lambda ctx: True, name="gate")
        after = SimpleNode("after")

        compiled = self._compile([gate, after])
        ctx = _make_context()
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, ctx, {})

        # Should not raise; partial result returned
        assert result is not None

    def test_early_exit_before_the_gate_runs(self):
        """Nodes before the gate should always run."""
        before = SimpleNode("before")
        gate = EarlyExit(predicate=lambda ctx: True, name="gate")
        after = SimpleNode("after")

        compiled = self._compile([before, gate, after])
        ctx = _make_context()
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, {})

        assert before.call_count == 1
        assert after.call_count == 0


# ===========================================================================
# SyncScheduler — conditional edge integration (end-to-end)
# ===========================================================================


class TestSyncSchedulerConditionalEdges:
    def _compile(self, nodes, wiring=None, edge_conditions=None):
        vr = MagicMock()
        vr.valid = True
        return CompiledGraph(
            name="g",
            nodes=nodes,
            execution_order=[[n] for n in nodes],
            wiring=wiring or {},
            dag=None,
            validation_result=vr,
            edge_conditions=edge_conditions or {},
        )

    def test_node_skipped_when_condition_false(self):
        node = SimpleNode("expensive")
        compiled = self._compile([node], edge_conditions={"expensive": lambda ctx: False})
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})
        assert node.call_count == 0

    def test_node_runs_when_condition_true(self):
        node = SimpleNode("expensive")
        compiled = self._compile([node], edge_conditions={"expensive": lambda ctx: True})
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})
        assert node.call_count == 1

    def test_node_without_condition_always_runs(self):
        a = SimpleNode("a")
        b = SimpleNode("b")
        compiled = self._compile([a, b], edge_conditions={"b": lambda ctx: False})
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})
        assert a.call_count == 1
        assert b.call_count == 0

    def test_condition_receives_context(self):
        ctx_received = []

        def cond(c):
            ctx_received.append(c)
            return True

        node = SimpleNode("n")
        compiled = self._compile([node], edge_conditions={"n": cond})
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})
        assert ctx_received[0] is ctx


# ===========================================================================
# SyncScheduler — cascade-skip integration (end-to-end)
# ===========================================================================


class TestSyncSchedulerCascadeSkip:
    def _compile(self, nodes, wiring=None, edge_conditions=None):
        vr = MagicMock()
        vr.valid = True
        return CompiledGraph(
            name="g",
            nodes=nodes,
            execution_order=[[n] for n in nodes],
            wiring=wiring or {},
            dag=None,
            validation_result=vr,
            edge_conditions=edge_conditions or {},
        )

    def test_downstream_cascade_skipped(self):
        """When A is skipped, B which depends on A.result is also skipped."""
        a = SimpleNode("a")
        b = SimpleNode("b")

        wiring = {"b.result": "a.result"}
        compiled = self._compile(
            [a, b],
            wiring=wiring,
            edge_conditions={"a": lambda ctx: False},
        )
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})

        assert a.call_count == 0
        assert b.call_count == 0

    def test_independent_node_not_cascade_skipped(self):
        """C with no dependency on skipped A should still run."""
        a = SimpleNode("a")
        c = SimpleNode("c")  # unrelated to a

        compiled = self._compile(
            [a, c],
            wiring={},  # no dependency between a and c
            edge_conditions={"a": lambda ctx: False},
        )
        ctx = _make_context()
        SyncScheduler().execute(compiled, ctx, {})

        assert a.call_count == 0
        assert c.call_count == 1


# ===========================================================================
# SyncScheduler._is_dependency_skipped helper
# ===========================================================================


class TestIsDependencySkipped:
    def _make_compiled(self, wiring):
        vr = MagicMock()
        vr.valid = True
        return CompiledGraph(
            name="g",
            nodes=[],
            execution_order=[],
            wiring=wiring,
            dag=None,
            validation_result=vr,
        )

    def test_returns_false_for_empty_skipped(self):
        scheduler = SyncScheduler()
        node = SimpleNode("b")
        compiled = self._make_compiled({"b.result": "a.result"})
        assert scheduler._is_dependency_skipped(node, compiled, set()) is False

    def test_returns_true_when_dependency_skipped(self):
        scheduler = SyncScheduler()
        node = SimpleNode("b")
        compiled = self._make_compiled({"b.result": "a.result"})
        assert scheduler._is_dependency_skipped(node, compiled, {"a"}) is True

    def test_returns_false_when_dependency_is_input(self):
        """Wiring to 'input.*' namespace should NOT trigger cascade-skip."""
        scheduler = SyncScheduler()
        node = SimpleNode("b")
        compiled = self._make_compiled({"b.result": "input.image"})
        assert scheduler._is_dependency_skipped(node, compiled, {"input"}) is False

    def test_returns_false_when_no_matching_wiring(self):
        scheduler = SyncScheduler()
        node = SimpleNode("b")
        compiled = self._make_compiled({"c.result": "a.result"})
        assert scheduler._is_dependency_skipped(node, compiled, {"a"}) is False

    def test_returns_false_when_dependency_not_in_skipped(self):
        scheduler = SyncScheduler()
        node = SimpleNode("b")
        compiled = self._make_compiled({"b.result": "a.result"})
        assert scheduler._is_dependency_skipped(node, compiled, {"other"}) is False
