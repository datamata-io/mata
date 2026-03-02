"""Comprehensive tests for GraphValidator.

Tests cover:
- Type compatibility checking across connections
- Dependency resolution (required inputs satisfied)
- Cycle detection (DAG enforcement)
- Name collision detection
- Provider capability verification
- Optional type handling
- Dynamic output name validation
- Complex graph topologies
- Error message quality
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.exceptions import ValidationError
from mata.core.graph.node import Node
from mata.core.graph.validator import GraphValidator, ValidationResult

# ──────── mock nodes ────────


class DetectNode(Node):
    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "Detect", using: str = "det"):
        super().__init__(name=name)
        self.provider_name = using

    def run(self, ctx, **kw):
        return {}


class FilterNode(Node):
    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "Filter"):
        super().__init__(name=name)

    def run(self, ctx, **kw):
        return {}


class ClassifyNode(Node):
    inputs = {"image": Image}
    outputs = {"classifications": Classifications}

    def __init__(self, name: str = "Classify", using: str = "clf"):
        super().__init__(name=name)
        self.provider_name = using

    def run(self, ctx, **kw):
        return {}


class DepthNode(Node):
    inputs = {"image": Image}
    outputs = {"depth": DepthMap}

    def __init__(self, name: str = "Depth", using: str = "dp"):
        super().__init__(name=name)
        self.provider_name = using

    def run(self, ctx, **kw):
        return {}


class DynOutputNode(Node):
    """Node with dynamic output name."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "DynOut", out: str = "custom"):
        super().__init__(name=name)
        self.output_name = out

    def run(self, ctx, **kw):
        return {}


# ══════════════════════════════════════════════════════════════
# ValidationResult
# ══════════════════════════════════════════════════════════════


class TestValidationResult:
    """ValidationResult dataclass behaviour."""

    def test_valid_result(self):
        vr = ValidationResult(valid=True)
        assert vr.valid
        assert len(vr.errors) == 0

    def test_raise_if_invalid(self):
        vr = ValidationResult(valid=False, errors=["err1"])
        with pytest.raises(ValidationError, match="err1"):
            vr.raise_if_invalid()

    def test_raise_if_valid_does_nothing(self):
        vr = ValidationResult(valid=True)
        vr.raise_if_invalid()  # no-op

    def test_str_valid(self):
        vr = ValidationResult(valid=True)
        assert "Valid" in str(vr)

    def test_str_invalid(self):
        vr = ValidationResult(valid=False, errors=["bad"])
        s = str(vr)
        assert "Invalid" in s
        assert "bad" in s

    def test_str_warnings(self):
        vr = ValidationResult(valid=True, warnings=["warn1"])
        s = str(vr)
        assert "warn1" in s


# ══════════════════════════════════════════════════════════════
# Type compatibility
# ══════════════════════════════════════════════════════════════


class TestTypeCompatibility:
    """check_type_compatibility()."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_compatible_connection(self):
        nodes = [DetectNode(), FilterNode()]
        wiring = {
            "Detect.image": "input.image",
            "Filter.detections": "Detect.detections",
        }
        errors = self.v.check_type_compatibility(nodes, wiring)
        assert len(errors) == 0

    def test_incompatible_types(self):
        nodes = [DetectNode(), ClassifyNode()]
        wiring = {
            "Detect.image": "input.image",
            "Classify.image": "Detect.detections",  # Detections → Image mismatch
        }
        errors = self.v.check_type_compatibility(nodes, wiring)
        assert len(errors) > 0
        assert any("Type mismatch" in e or "mismatch" in e.lower() for e in errors)

    def test_external_input_skipped(self):
        nodes = [DetectNode()]
        wiring = {"Detect.image": "input.image"}
        errors = self.v.check_type_compatibility(nodes, wiring)
        assert len(errors) == 0

    def test_unknown_artifact_flagged(self):
        nodes = [FilterNode()]
        wiring = {"Filter.detections": "NonExistent.output"}
        errors = self.v.check_type_compatibility(nodes, wiring)
        assert len(errors) > 0

    def test_dynamic_output_name_resolved(self):
        nodes = [DynOutputNode(out="my_dets"), FilterNode()]
        wiring = {
            "DynOut.image": "input.image",
            "Filter.detections": "DynOut.my_dets",
        }
        errors = self.v.check_type_compatibility(nodes, wiring)
        assert len(errors) == 0


# ══════════════════════════════════════════════════════════════
# Dependency resolution
# ══════════════════════════════════════════════════════════════


class TestDependencyResolution:
    """check_dependencies()."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_all_inputs_satisfied(self):
        nodes = [DetectNode(), FilterNode()]
        wiring = {
            "Detect.image": "input.image",
            "Filter.detections": "Detect.detections",
        }
        errors = self.v.check_dependencies(nodes, wiring)
        assert len(errors) == 0

    def test_missing_input_flagged(self):
        nodes = [FilterNode()]
        wiring = {}  # Filter needs detections but none wired
        errors = self.v.check_dependencies(nodes, wiring)
        assert len(errors) > 0


# ══════════════════════════════════════════════════════════════
# Cycle detection
# ══════════════════════════════════════════════════════════════


class TestCycleDetection:
    """detect_cycles()."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_no_cycle(self):
        nodes = [DetectNode(), FilterNode()]
        wiring = {
            "Detect.image": "input.image",
            "Filter.detections": "Detect.detections",
        }
        cycle = self.v.detect_cycles(nodes, wiring)
        assert cycle is None

    def test_self_loop_detected(self):
        nodes = [FilterNode()]
        wiring = {"Filter.detections": "Filter.detections"}
        cycle = self.v.detect_cycles(nodes, wiring)
        assert cycle is not None

    def test_cycle_between_two_nodes(self):
        # Can't really create a true cycle with strict typing,
        # but we can wire in a way that creates dependency loop
        a = DetectNode(name="A")
        b = DetectNode(name="B")
        wiring = {
            "A.image": "B.detections",
            "B.image": "A.detections",
        }
        cycle = self.v.detect_cycles([a, b], wiring)
        assert cycle is not None


# ══════════════════════════════════════════════════════════════
# Name collisions
# ══════════════════════════════════════════════════════════════


class TestNameCollisions:
    """check_name_collisions()."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_no_collisions(self):
        nodes = [DetectNode(name="A"), FilterNode(name="B")]
        errors = self.v.check_name_collisions(nodes)
        assert len(errors) == 0

    def test_duplicate_names(self):
        nodes = [DetectNode(name="A"), FilterNode(name="A")]
        errors = self.v.check_name_collisions(nodes)
        assert len(errors) > 0
        assert any("Duplicate" in e for e in errors)


# ══════════════════════════════════════════════════════════════
# Provider capabilities
# ══════════════════════════════════════════════════════════════


class TestProviderCapabilities:
    """check_provider_capabilities()."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_provider_available(self):
        nodes = [DetectNode(using="det")]
        providers = {"det": Mock()}
        errors = self.v.check_provider_capabilities(nodes, providers)
        assert len(errors) == 0

    def test_provider_missing(self):
        nodes = [DetectNode(using="missing_det")]
        providers = {}
        errors = self.v.check_provider_capabilities(nodes, providers)
        assert len(errors) > 0


# ══════════════════════════════════════════════════════════════
# Full validate()
# ══════════════════════════════════════════════════════════════


class TestFullValidation:
    """End-to-end validate() method."""

    def setup_method(self):
        self.v = GraphValidator()

    def test_valid_pipeline(self):
        nodes = [DetectNode(), FilterNode()]
        wiring = {
            "Detect.image": "input.image",
            "Filter.detections": "Detect.detections",
        }
        result = self.v.validate(nodes, wiring, providers={"det": Mock()})
        assert result.valid

    def test_empty_graph_invalid(self):
        result = self.v.validate([], {})
        assert not result.valid
        assert any("at least one node" in e for e in result.errors)

    def test_multiple_errors_collected(self):
        # Filter with no wiring + unknown provider
        nodes = [DetectNode(using="unknown"), FilterNode()]
        wiring = {}
        result = self.v.validate(nodes, wiring, providers={})
        assert not result.valid
        assert len(result.errors) >= 1
