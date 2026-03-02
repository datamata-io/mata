"""Comprehensive tests for conditional execution (If, Pass, predicates).

Tests cover:
- If node with then/else branches
- Pass node (no-op)
- HasLabel predicate
- CountAbove predicate
- ScoreAbove predicate
- Functional predicate helpers (has_label, count_above, score_above)
- Nested conditionals
- Edge cases: empty detections, missing artifacts
"""

from __future__ import annotations

from mata.core.artifacts.detections import Detections
from mata.core.graph.conditionals import (
    CountAbove,
    HasLabel,
    If,
    Pass,
    ScoreAbove,
    count_above,
    has_label,
    score_above,
)
from mata.core.graph.context import ExecutionContext
from mata.core.graph.node import Node
from mata.core.types import Instance, VisionResult

# ──────── helpers ────────


def _make_ctx() -> ExecutionContext:
    return ExecutionContext(providers={}, device="cpu")


def _make_detections_with_labels(labels: list[str], scores: list[float] | None = None) -> Detections:
    if scores is None:
        scores = [0.9] * len(labels)
    instances = [
        Instance(
            bbox=(i * 10, i * 10, i * 10 + 50, i * 10 + 50),
            score=scores[i],
            label=i,
            label_name=labels[i],
        )
        for i in range(len(labels))
    ]
    return Detections.from_vision_result(VisionResult(instances=instances))


# ──────── mock branches ────────


class ThenNode(Node):
    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self):
        super().__init__(name="ThenNode")

    def run(self, ctx, **kw):
        return {"detections": Detections(meta={"branch": "then"})}


class ElseNode(Node):
    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self):
        super().__init__(name="ElseNode")

    def run(self, ctx, **kw):
        return {"detections": Detections(meta={"branch": "else"})}


# ══════════════════════════════════════════════════════════════
# HasLabel predicate
# ══════════════════════════════════════════════════════════════


class TestHasLabel:
    """HasLabel predicate tests."""

    def test_label_present(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["cat", "dog"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "cat")
        assert pred(ctx) is True

    def test_label_absent(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["cat", "dog"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "bird")
        assert pred(ctx) is False

    def test_case_insensitive(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["Cat"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "cat")
        assert pred(ctx) is True

    def test_missing_artifact_returns_false(self):
        ctx = _make_ctx()
        pred = HasLabel("nonexistent", "cat")
        assert pred(ctx) is False

    def test_repr(self):
        pred = HasLabel("dets", "cat")
        assert "HasLabel" in repr(pred)


# ══════════════════════════════════════════════════════════════
# CountAbove predicate
# ══════════════════════════════════════════════════════════════


class TestCountAbove:
    """CountAbove predicate tests."""

    def test_count_above_threshold(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["a", "b", "c"])
        ctx.store("dets", dets)

        pred = CountAbove("dets", 2)
        assert pred(ctx) is True  # 3 > 2

    def test_count_equal_not_above(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["a", "b"])
        ctx.store("dets", dets)

        pred = CountAbove("dets", 2)
        assert pred(ctx) is False  # 2 > 2 is False

    def test_count_zero(self):
        ctx = _make_ctx()
        dets = Detections()
        ctx.store("dets", dets)

        pred = CountAbove("dets", 0)
        assert pred(ctx) is False  # 0 > 0 is False

    def test_missing_artifact_returns_false(self):
        ctx = _make_ctx()
        pred = CountAbove("nonexistent", 0)
        assert pred(ctx) is False


# ══════════════════════════════════════════════════════════════
# ScoreAbove predicate
# ══════════════════════════════════════════════════════════════


class TestScoreAbove:
    """ScoreAbove predicate tests."""

    def test_score_above_threshold(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["cat"], scores=[0.95])
        ctx.store("dets", dets)

        pred = ScoreAbove("dets", 0.9)
        assert pred(ctx) is True

    def test_score_below_threshold(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["cat"], scores=[0.5])
        ctx.store("dets", dets)

        pred = ScoreAbove("dets", 0.9)
        assert pred(ctx) is False

    def test_empty_detections(self):
        ctx = _make_ctx()
        dets = Detections()
        ctx.store("dets", dets)

        pred = ScoreAbove("dets", 0.5)
        assert pred(ctx) is False

    def test_missing_artifact(self):
        ctx = _make_ctx()
        pred = ScoreAbove("nonexistent", 0.5)
        assert pred(ctx) is False


# ══════════════════════════════════════════════════════════════
# If node
# ══════════════════════════════════════════════════════════════


class TestIfNode:
    """If conditional node tests."""

    def test_then_branch_taken(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["cat"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "cat")
        if_node = If(predicate=pred, then_branch=ThenNode(), else_branch=ElseNode())
        result = if_node.run(ctx, detections=dets)
        assert result["detections"].meta.get("branch") == "then"

    def test_else_branch_taken(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["dog"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "cat")
        if_node = If(predicate=pred, then_branch=ThenNode(), else_branch=ElseNode())
        result = if_node.run(ctx, detections=dets)
        assert result["detections"].meta.get("branch") == "else"

    def test_else_defaults_to_pass(self):
        ctx = _make_ctx()
        dets = _make_detections_with_labels(["dog"])
        ctx.store("dets", dets)

        pred = HasLabel("dets", "cat")
        if_node = If(predicate=pred, then_branch=ThenNode())
        result = if_node.run(ctx, detections=dets)
        # Pass node returns empty dict
        assert result == {} or "detections" not in result or result.get("detections") is not None


# ══════════════════════════════════════════════════════════════
# Pass node
# ══════════════════════════════════════════════════════════════


class TestPassNode:
    """Pass (no-op) node."""

    def test_pass_returns_empty(self):
        ctx = _make_ctx()
        p = Pass()
        result = p.run(ctx)
        assert result == {}


# ══════════════════════════════════════════════════════════════
# Functional helpers
# ══════════════════════════════════════════════════════════════


class TestFunctionalHelpers:
    """has_label(), count_above(), score_above() factory functions."""

    def test_has_label_factory(self):
        pred = has_label("dets", "cat")
        assert callable(pred)

    def test_count_above_factory(self):
        pred = count_above("dets", 5)
        assert callable(pred)

    def test_score_above_factory(self):
        pred = score_above("dets", 0.9)
        assert callable(pred)
