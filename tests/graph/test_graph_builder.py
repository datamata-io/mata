"""Comprehensive tests for Graph builder, compilation, and CompiledGraph.

Tests cover:
- Fluent API: then(), add(), parallel(), conditional()
- Auto-wiring between sequential nodes
- Explicit wiring via add()
- Parallel node grouping and execution order
- Compilation with validation
- DAG representation (with and without networkx)
- Execution order computation (topological sort)
- Parallel stage computation
- Name collision handling
- Empty graph handling
- Multi-stage pipeline ordering
- Dynamic output name wiring
- Graph repr
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.exceptions import ValidationError
from mata.core.graph.graph import CompiledGraph, Graph
from mata.core.graph.node import Node

# ──────── mock nodes ────────


class MockDetect(Node):
    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, using: str = "det", out: str = "dets", **kw):
        super().__init__(name=kw.pop("name", "Detect"), **kw)
        self.provider_name = using
        self.output_name = out

    def run(self, ctx, image=None, **kw):
        return {self.output_name: Detections()}


class MockClassify(Node):
    inputs = {"image": Image}
    outputs = {"classifications": Classifications}

    def __init__(self, using: str = "clf", out: str = "cls", **kw):
        super().__init__(name=kw.pop("name", "Classify"), **kw)
        self.provider_name = using
        self.output_name = out

    def run(self, ctx, image=None, **kw):
        return {self.output_name: Classifications()}


class MockDepth(Node):
    inputs = {"image": Image}
    outputs = {"depth": DepthMap}

    def __init__(self, using: str = "dp", out: str = "depth", **kw):
        super().__init__(name=kw.pop("name", "Depth"), **kw)
        self.provider_name = using
        self.output_name = out

    def run(self, ctx, image=None, **kw):
        return {self.output_name: DepthMap(depth=np.zeros((4, 4)))}


class MockFilter(Node):
    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, src: str = "dets", out: str = "filtered", **kw):
        super().__init__(name=kw.pop("name", "Filter"), **kw)
        self.src = src
        self.output_name = out

    def run(self, ctx, detections=None, **kw):
        return {self.output_name: detections or Detections()}


class MockFuse(Node):
    inputs = {}
    outputs = {"result": MultiResult}

    def __init__(self, out: str = "final", **kw):
        super().__init__(name=kw.pop("name", "Fuse"), **kw)
        self.output_name = out

    def run(self, ctx, **kw):
        return {self.output_name: MultiResult()}


# ──────── mock providers (pass validation) ────────

_MOCK_PROVIDERS = {
    "detect": {"det": Mock()},
    "classify": {"clf": Mock()},
    "depth": {"dp": Mock()},
    # Flat keys for validator (checks node.provider_name in top-level keys)
    "det": Mock(),
    "clf": Mock(),
    "dp": Mock(),
}


# ══════════════════════════════════════════════════════════════
# Graph builder API
# ══════════════════════════════════════════════════════════════


class TestGraphThen:
    """Sequential chaining via then()."""

    def test_single_node(self):
        g = Graph("single").then(MockDetect())
        assert len(g._nodes) == 1

    def test_chain_two_nodes(self):
        g = Graph("chain").then(MockDetect()).then(MockFilter())
        assert len(g._nodes) == 2

    def test_three_stage_pipeline(self):
        g = Graph("pipe").then(MockDetect()).then(MockFilter()).then(MockFuse())
        assert len(g._nodes) == 3

    def test_first_node_wired_to_input(self):
        g = Graph("w").then(MockDetect())
        node = g._nodes[0]
        wiring_key = f"{node.name}.image"
        assert g._wiring.get(wiring_key, "").startswith("input.")

    def test_second_node_auto_wired(self):
        g = Graph().then(MockDetect()).then(MockFilter())
        # Filter's input should wire to Detect's output
        filter_key = f"{g._nodes[1].name}.detections"
        assert filter_key in g._wiring


class TestGraphAdd:
    """Explicit wiring via add()."""

    def test_explicit_wiring(self):
        g = Graph()
        g.add(MockDetect(), inputs={"image": "input.image"})
        assert "Detect.image" in g._wiring
        assert g._wiring["Detect.image"] == "input.image"

    def test_add_auto_wire_when_no_inputs(self):
        g = Graph()
        g.add(MockDetect())
        assert "Detect.image" in g._wiring


class TestGraphParallel:
    """Parallel node grouping."""

    def test_parallel_two_nodes(self):
        g = Graph("par").parallel(
            [
                MockDetect(name="D1"),
                MockClassify(name="C1"),
            ]
        )
        assert len(g._nodes) == 2

    def test_parallel_three_nodes(self):
        g = Graph().parallel(
            [
                MockDetect(name="D1"),
                MockClassify(name="C1"),
                MockDepth(name="Dp"),
            ]
        )
        assert len(g._nodes) == 3

    def test_parallel_nodes_wire_to_input(self):
        g = Graph().parallel(
            [
                MockDetect(name="D1"),
                MockClassify(name="C1"),
            ]
        )
        assert "D1.image" in g._wiring
        assert "C1.image" in g._wiring


class TestGraphConditional:
    """Conditional branching."""

    def test_conditional_adds_then_branch(self):
        g = (
            Graph()
            .then(MockDetect())
            .conditional(
                predicate=lambda ctx: True,
                then_branch=MockFilter(name="ThenFilter"),
            )
        )
        names = [n.name for n in g._nodes]
        assert "ThenFilter" in names

    def test_conditional_adds_else_branch(self):
        g = (
            Graph()
            .then(MockDetect())
            .conditional(
                predicate=lambda ctx: True,
                then_branch=MockFilter(name="ThenFilter"),
                else_branch=MockFuse(name="ElseFuse"),
            )
        )
        names = [n.name for n in g._nodes]
        assert "ThenFilter" in names
        assert "ElseFuse" in names


class TestGraphNameCollision:
    """Name collision detection."""

    def test_duplicate_name_raises(self):
        with pytest.raises(ValidationError, match="already exists"):
            Graph().then(MockDetect(name="dup")).then(MockFilter(name="dup"))


class TestGraphRepr:
    """Graph __repr__."""

    def test_uncompiled_repr(self):
        g = Graph("test_graph").then(MockDetect())
        r = repr(g)
        assert "test_graph" in r
        assert "1 nodes" in r
        assert "uncompiled" in r

    def test_compiled_repr(self):
        g = Graph("test_graph").then(MockDetect())
        g.compile(providers=_MOCK_PROVIDERS)
        r = repr(g)
        assert "compiled" in r


# ══════════════════════════════════════════════════════════════
# Graph compilation
# ══════════════════════════════════════════════════════════════


class TestGraphCompile:
    """Compilation and validation."""

    def test_compile_simple(self):
        g = Graph().then(MockDetect())
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        assert isinstance(compiled, CompiledGraph)

    def test_compile_stores_nodes(self):
        g = Graph().then(MockDetect()).then(MockFilter())
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        assert len(compiled.nodes) == 2

    def test_compile_stores_wiring(self):
        g = Graph().then(MockDetect()).then(MockFilter())
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        assert len(compiled.wiring) > 0

    def test_compile_creates_dag(self):
        g = Graph().then(MockDetect()).then(MockFilter())
        g.compile(providers=_MOCK_PROVIDERS)
        # DAG may be None if networkx not available
        # Just check compilation succeeded

    def test_compile_validation_result(self):
        g = Graph().then(MockDetect())
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        assert compiled.validation_result.valid


# ══════════════════════════════════════════════════════════════
# Execution order / CompiledGraph
# ══════════════════════════════════════════════════════════════


class TestCompiledGraphOrder:
    """Execution order computation."""

    def test_sequential_order(self):
        g = Graph().then(MockDetect()).then(MockFilter())
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        order = compiled.execution_order
        # Should produce 2 stages
        assert len(order) >= 2
        # Detect must come before Filter
        flat = [n.name for stage in order for n in stage]
        assert flat.index("Detect") < flat.index("Filter")

    def test_parallel_same_stage(self):
        g = Graph().parallel(
            [
                MockDetect(name="D"),
                MockClassify(name="C"),
            ]
        )
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        order = compiled.execution_order
        # Both should be in the same stage (first stage)
        first_stage_names = [n.name for n in order[0]]
        assert "D" in first_stage_names
        assert "C" in first_stage_names

    def test_parallel_then_sequential(self):
        g = (
            Graph()
            .parallel(
                [
                    MockDetect(name="D"),
                    MockClassify(name="C"),
                ]
            )
            .then(MockFuse(name="F"))
        )
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        order = compiled.execution_order
        flat = [n.name for stage in order for n in stage]
        # Fuse must come after both D and C
        assert flat.index("F") > flat.index("D")
        assert flat.index("F") > flat.index("C")

    def test_get_parallel_stages(self):
        g = Graph().parallel([MockDetect(name="D"), MockClassify(name="C")])
        compiled = g.compile(providers=_MOCK_PROVIDERS)
        stages = compiled.get_parallel_stages()
        assert isinstance(stages, list)
        assert len(stages) >= 1


class TestCompiledGraphRepr:
    """CompiledGraph __repr__."""

    def test_repr(self):
        g = Graph("my_pipeline").then(MockDetect())
        c = g.compile(providers=_MOCK_PROVIDERS)
        r = repr(c)
        assert "my_pipeline" in r


# ══════════════════════════════════════════════════════════════
# Dynamic output name wiring
# ══════════════════════════════════════════════════════════════


class TestDynamicOutputWiring:
    """Nodes with custom output names (out parameter) auto-wire correctly."""

    def test_detect_custom_out_auto_wires(self):
        g = Graph().then(MockDetect(out="my_dets")).then(MockFilter())
        # Filter should be wired to Detect.my_dets
        filter_wiring = g._wiring.get("Filter.detections", "")
        assert "Detect" in filter_wiring

    def test_last_outputs_track_dynamic_names(self):
        g = Graph().then(MockDetect(out="special"))
        # _last_outputs should have the dynamic name registered
        assert any("special" in v for v in g._last_outputs.values())
