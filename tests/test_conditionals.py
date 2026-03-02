"""Unit tests for conditional execution nodes and predicates.

Tests cover:
- Predicate implementations (HasLabel, CountAbove, ScoreAbove)
- If node execution with then/else branches
- Pass node behavior
- Conditional graph integration
- Error handling and edge cases
- Metrics collection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.graph.conditionals import CountAbove, HasLabel, If, Pass, ScoreAbove, count_above, has_label, score_above
from mata.core.graph.context import ExecutionContext
from mata.core.graph.node import Node
from mata.core.types import Entity, Instance


# Mock artifacts for testing
@dataclass(frozen=True)
class MockImageArtifact(Artifact):
    """Mock image artifact for testing."""

    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockImageArtifact:
        return cls(width=data["width"], height=data["height"])

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid image dimensions")


@dataclass(frozen=True)
class MockResultArtifact(Artifact):
    """Mock result artifact for testing."""

    value: str

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockResultArtifact:
        return cls(value=data["value"])

    def validate(self) -> None:
        pass


# Mock nodes for testing branches
class MockNodeA(Node):
    """Mock node returning result A."""

    inputs = {"image": MockImageArtifact}
    outputs = {"result": MockResultArtifact}

    def __init__(self, name: str = "MockNodeA"):
        super().__init__(name=name)

    def run(self, ctx: ExecutionContext, **inputs) -> dict[str, Artifact]:
        return {"result": MockResultArtifact("result_A")}


class MockNodeB(Node):
    """Mock node returning result B."""

    inputs = {"image": MockImageArtifact}
    outputs = {"result": MockResultArtifact}

    def __init__(self, name: str = "MockNodeB"):
        super().__init__(name=name)

    def run(self, ctx: ExecutionContext, **inputs) -> dict[str, Artifact]:
        return {"result": MockResultArtifact("result_B")}


class MockErrorNode(Node):
    """Mock node that always raises an error."""

    inputs = {"image": MockImageArtifact}
    outputs = {"result": MockResultArtifact}

    def __init__(self, error_message: str = "Mock error"):
        super().__init__(name="MockErrorNode")
        self.error_message = error_message

    def run(self, ctx: ExecutionContext, **inputs) -> dict[str, Artifact]:
        raise RuntimeError(self.error_message)


# Helper function to create test detections
def create_test_detections(
    instances: list[tuple[str, float, tuple[float, float, float, float]]] = None,
    entities: list[tuple[str, float]] = None,
) -> Detections:
    """Create test Detections artifact.

    Args:
        instances: List of (label, score, bbox) tuples
        entities: List of (label, score) tuples

    Returns:
        Detections artifact with test data
    """
    inst_list = []
    if instances:
        for label, score, bbox in instances:
            inst = Instance(bbox=bbox, label=0, label_name=label, score=score)  # Dummy label ID
            inst_list.append(inst)

    entity_list = []
    if entities:
        for label, score in entities:
            entity = Entity(label=label, score=score)
            entity_list.append(entity)

    return Detections(instances=inst_list, entities=entity_list, meta={})


class TestHasLabel:
    """Tests for HasLabel predicate."""

    def test_init(self):
        """Test HasLabel initialization."""
        predicate = HasLabel("detections", "cat")

        assert predicate.src == "detections"
        assert predicate.label == "cat"

    def test_init_case_normalization(self):
        """Test case normalization during init."""
        predicate = HasLabel("dets", "CAT")

        assert predicate.label == "cat"  # Normalized to lowercase

    def test_label_found_in_instances(self):
        """Test label found in instance labels."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("cat", 0.9, (10, 10, 50, 50)), ("dog", 0.8, (60, 60, 100, 100))])
        ctx.retrieve.return_value = dets

        predicate = HasLabel("dets", "cat")
        result = predicate(ctx)

        assert result is True
        ctx.retrieve.assert_called_once_with("dets")

    def test_label_found_in_entities(self):
        """Test label found in entity labels."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(entities=[("cat", 0.9), ("dog", 0.8)])
        ctx.retrieve.return_value = dets

        predicate = HasLabel("dets", "cat")
        result = predicate(ctx)

        assert result is True

    def test_label_found_case_insensitive(self):
        """Test case-insensitive label matching."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("Cat", 0.9, (10, 10, 50, 50))])
        ctx.retrieve.return_value = dets

        predicate = HasLabel("dets", "cat")  # lowercase query
        result = predicate(ctx)

        assert result is True

    def test_label_not_found(self):
        """Test label not found returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("dog", 0.8, (60, 60, 100, 100))], entities=[("bird", 0.7)])
        ctx.retrieve.return_value = dets

        predicate = HasLabel("dets", "cat")
        result = predicate(ctx)

        assert result is False

    def test_empty_detections(self):
        """Test empty detections returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections()  # Empty
        ctx.retrieve.return_value = dets

        predicate = HasLabel("dets", "cat")
        result = predicate(ctx)

        assert result is False

    def test_artifact_not_found(self):
        """Test missing artifact returns False."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.side_effect = KeyError("Artifact not found")

        predicate = HasLabel("missing", "cat")
        result = predicate(ctx)

        assert result is False

    def test_wrong_artifact_type(self):
        """Test wrong artifact type raises TypeError."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.return_value = MockImageArtifact(100, 100)  # Not Detections

        predicate = HasLabel("image", "cat")

        with pytest.raises(TypeError, match="HasLabel predicate expects Detections"):
            predicate(ctx)

    def test_repr(self):
        """Test string representation."""
        predicate = HasLabel("detections", "cat")

        assert repr(predicate) == "HasLabel(src='detections', label='cat')"


class TestCountAbove:
    """Tests for CountAbove predicate."""

    def test_init(self):
        """Test CountAbove initialization."""
        predicate = CountAbove("detections", 5)

        assert predicate.src == "detections"
        assert predicate.n == 5

    def test_count_above_threshold_instances_only(self):
        """Test count above threshold with instances only."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(
            instances=[
                ("cat", 0.9, (10, 10, 50, 50)),
                ("dog", 0.8, (60, 60, 100, 100)),
                ("bird", 0.7, (110, 110, 150, 150)),
            ]
        )
        ctx.retrieve.return_value = dets

        predicate = CountAbove("dets", 2)  # 3 > 2
        result = predicate(ctx)

        assert result is True

    def test_count_above_threshold_with_entities(self):
        """Test count above threshold with instances + entities."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(
            instances=[("cat", 0.9, (10, 10, 50, 50))], entities=[("dog", 0.8), ("bird", 0.7)]
        )
        ctx.retrieve.return_value = dets

        predicate = CountAbove("dets", 2)  # 1 + 2 = 3 > 2
        result = predicate(ctx)

        assert result is True

    def test_count_equal_threshold(self):
        """Test count equal to threshold returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("cat", 0.9, (10, 10, 50, 50)), ("dog", 0.8, (60, 60, 100, 100))])
        ctx.retrieve.return_value = dets

        predicate = CountAbove("dets", 2)  # 2 == 2 (not >)
        result = predicate(ctx)

        assert result is False

    def test_count_below_threshold(self):
        """Test count below threshold returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("cat", 0.9, (10, 10, 50, 50))])
        ctx.retrieve.return_value = dets

        predicate = CountAbove("dets", 2)  # 1 < 2
        result = predicate(ctx)

        assert result is False

    def test_empty_detections(self):
        """Test empty detections returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections()  # Empty
        ctx.retrieve.return_value = dets

        predicate = CountAbove("dets", 0)  # 0 <= 0
        result = predicate(ctx)

        assert result is False

    def test_artifact_not_found(self):
        """Test missing artifact returns False."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.side_effect = KeyError("Artifact not found")

        predicate = CountAbove("missing", 1)
        result = predicate(ctx)

        assert result is False

    def test_wrong_artifact_type(self):
        """Test wrong artifact type raises TypeError."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.return_value = MockImageArtifact(100, 100)  # Not Detections

        predicate = CountAbove("image", 1)

        with pytest.raises(TypeError, match="CountAbove predicate expects Detections"):
            predicate(ctx)

    def test_repr(self):
        """Test string representation."""
        predicate = CountAbove("detections", 5)

        assert repr(predicate) == "CountAbove(src='detections', n=5)"


class TestScoreAbove:
    """Tests for ScoreAbove predicate."""

    def test_init(self):
        """Test ScoreAbove initialization."""
        predicate = ScoreAbove("detections", 0.8)

        assert predicate.src == "detections"
        assert predicate.threshold == 0.8

    def test_max_score_above_threshold(self):
        """Test max score above threshold."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("cat", 0.9, (10, 10, 50, 50)), ("dog", 0.7, (60, 60, 100, 100))])
        ctx.retrieve.return_value = dets

        predicate = ScoreAbove("dets", 0.8)  # max 0.9 > 0.8
        result = predicate(ctx)

        assert result is True

    def test_max_score_from_entities(self):
        """Test max score found in entities."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(
            instances=[("cat", 0.7, (10, 10, 50, 50))], entities=[("dog", 0.95)]  # Higher score in entities
        )
        ctx.retrieve.return_value = dets

        predicate = ScoreAbove("dets", 0.9)  # max 0.95 > 0.9
        result = predicate(ctx)

        assert result is True

    def test_max_score_equal_threshold(self):
        """Test max score equal to threshold returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(
            instances=[("cat", 0.5, (10, 10, 50, 50))]  # Use 0.5 to avoid floating point precision issues
        )
        ctx.retrieve.return_value = dets

        predicate = ScoreAbove("dets", 0.5)  # 0.5 == 0.5 (not >)
        result = predicate(ctx)

        assert result is False

    def test_max_score_below_threshold(self):
        """Test max score below threshold returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections(instances=[("cat", 0.7, (10, 10, 50, 50)), ("dog", 0.6, (60, 60, 100, 100))])
        ctx.retrieve.return_value = dets

        predicate = ScoreAbove("dets", 0.8)  # max 0.7 < 0.8
        result = predicate(ctx)

        assert result is False

    def test_empty_detections(self):
        """Test empty detections returns False."""
        ctx = Mock(spec=ExecutionContext)
        dets = create_test_detections()  # Empty
        ctx.retrieve.return_value = dets

        predicate = ScoreAbove("dets", 0.5)
        result = predicate(ctx)

        assert result is False

    def test_artifact_not_found(self):
        """Test missing artifact returns False."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.side_effect = KeyError("Artifact not found")

        predicate = ScoreAbove("missing", 0.5)
        result = predicate(ctx)

        assert result is False

    def test_wrong_artifact_type(self):
        """Test wrong artifact type raises TypeError."""
        ctx = Mock(spec=ExecutionContext)
        ctx.retrieve.return_value = MockImageArtifact(100, 100)  # Not Detections

        predicate = ScoreAbove("image", 0.5)

        with pytest.raises(TypeError, match="ScoreAbove predicate expects Detections"):
            predicate(ctx)

    def test_repr(self):
        """Test string representation."""
        predicate = ScoreAbove("detections", 0.8)

        assert repr(predicate) == "ScoreAbove(src='detections', threshold=0.8)"


class TestIf:
    """Tests for If conditional node."""

    def test_init_with_else_branch(self):
        """Test If node initialization with explicit else branch."""
        then_node = MockNodeA()
        else_node = MockNodeB()
        predicate = HasLabel("dets", "cat")

        if_node = If(predicate, then_node, else_node)

        assert if_node.predicate is predicate
        assert if_node.then_branch is then_node
        assert if_node.else_branch is else_node
        assert if_node.name == "If"

    def test_init_without_else_branch(self):
        """Test If node initialization with default Pass else branch."""
        then_node = MockNodeA()
        predicate = HasLabel("dets", "cat")

        if_node = If(predicate, then_node)

        assert if_node.predicate is predicate
        assert if_node.then_branch is then_node
        assert isinstance(if_node.else_branch, Pass)

    def test_init_custom_name(self):
        """Test If node with custom name."""
        then_node = MockNodeA()
        predicate = HasLabel("dets", "cat")

        if_node = If(predicate, then_node, name="ConditionalProcessor")

        assert if_node.name == "ConditionalProcessor"

    @patch("mata.core.graph.conditionals.time")
    def test_run_then_branch(self, mock_time):
        """Test If node executes then branch when predicate is True."""
        # Mock time for metrics
        mock_time.time.side_effect = [0.0, 0.001, 0.001, 0.002]  # Start, predicate end, branch start, branch end

        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        # Setup predicate to return True
        predicate = Mock()
        predicate.return_value = True

        then_node = MockNodeA("ThenNode")
        else_node = MockNodeB("ElseNode")

        if_node = If(predicate, then_node, else_node, name="TestIf")

        # Test inputs
        inputs = {"image": MockImageArtifact(100, 100)}

        # Execute
        result = if_node.run(ctx, **inputs)

        # Check result is from then branch
        assert result == {"result": MockResultArtifact("result_A")}

        # Check predicate was called
        predicate.assert_called_once_with(ctx)

        # Check metrics recorded
        ctx.record_metric.assert_any_call("TestIf", "predicate_latency_ms", 1.0)
        ctx.record_metric.assert_any_call("TestIf", "condition_result", True)
        ctx.record_metric.assert_any_call("TestIf", "selected_branch", 1.0)
        ctx.record_metric.assert_any_call("TestIf", "then_branch_latency_ms", 1.0)

    @patch("mata.core.graph.conditionals.time")
    def test_run_else_branch(self, mock_time):
        """Test If node executes else branch when predicate is False."""
        # Mock time for metrics
        mock_time.time.side_effect = [0.0, 0.001, 0.001, 0.002]

        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        # Setup predicate to return False
        predicate = Mock()
        predicate.return_value = False

        then_node = MockNodeA("ThenNode")
        else_node = MockNodeB("ElseNode")

        if_node = If(predicate, then_node, else_node, name="TestIf")

        # Test inputs
        inputs = {"image": MockImageArtifact(100, 100)}

        # Execute
        result = if_node.run(ctx, **inputs)

        # Check result is from else branch
        assert result == {"result": MockResultArtifact("result_B")}

        # Check predicate was called
        predicate.assert_called_once_with(ctx)

        # Check metrics recorded
        ctx.record_metric.assert_any_call("TestIf", "condition_result", False)
        ctx.record_metric.assert_any_call("TestIf", "selected_branch", 0.0)
        ctx.record_metric.assert_any_call("TestIf", "else_branch_latency_ms", 1.0)

    def test_run_predicate_error(self):
        """Test If node handles predicate errors."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        # Setup predicate to raise error
        predicate = Mock()
        predicate.side_effect = RuntimeError("Predicate error")

        then_node = MockNodeA()
        if_node = If(predicate, then_node, name="TestIf")

        inputs = {"image": MockImageArtifact(100, 100)}

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Predicate error"):
            if_node.run(ctx, **inputs)

        # Check error was recorded
        ctx.record_metric.assert_any_call("TestIf", "error", 1.0)

    def test_run_branch_error(self):
        """Test If node handles branch execution errors."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        # Setup predicate to return True
        predicate = Mock()
        predicate.return_value = True

        error_node = MockErrorNode("Branch error")
        if_node = If(predicate, error_node, name="TestIf")

        inputs = {"image": MockImageArtifact(100, 100)}

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Branch error"):
            if_node.run(ctx, **inputs)

        # Check error was recorded
        ctx.record_metric.assert_any_call("TestIf", "error", 1.0)

    def test_repr(self):
        """Test string representation."""
        then_node = MockNodeA("ThenNode")
        else_node = MockNodeB("ElseNode")
        predicate = HasLabel("dets", "cat")

        if_node = If(predicate, then_node, else_node)

        expected = "If(predicate=HasLabel(src='dets', label='cat'), then=ThenNode, else=ElseNode)"
        assert repr(if_node) == expected


class TestPass:
    """Tests for Pass no-op node."""

    def test_init(self):
        """Test Pass node initialization."""
        pass_node = Pass()

        assert pass_node.name == "Pass"
        assert pass_node.inputs == {}
        assert pass_node.outputs == {}

    def test_init_custom_name(self):
        """Test Pass node with custom name."""
        pass_node = Pass(name="NoOp")

        assert pass_node.name == "NoOp"

    def test_run(self):
        """Test Pass node execution."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        pass_node = Pass(name="TestPass")

        # Execute with some inputs
        inputs = {
            "image": MockImageArtifact(100, 100),
            "detections": create_test_detections([("cat", 0.9, (10, 10, 50, 50))]),
        }

        result = pass_node.run(ctx, **inputs)

        # Should return empty dict
        assert result == {}

        # Should record metrics
        ctx.record_metric.assert_any_call("TestPass", "executed", True)
        ctx.record_metric.assert_any_call("TestPass", "input_count", 2)

    def test_run_no_inputs(self):
        """Test Pass node with no inputs."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        pass_node = Pass()

        result = pass_node.run(ctx)

        assert result == {}
        ctx.record_metric.assert_any_call("Pass", "input_count", 0)

    def test_repr(self):
        """Test string representation."""
        pass_node = Pass(name="TestPass")

        assert repr(pass_node) == "Pass(name='TestPass')"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_has_label_function(self):
        """Test has_label convenience function."""
        predicate = has_label("detections", "cat")

        assert isinstance(predicate, HasLabel)
        assert predicate.src == "detections"
        assert predicate.label == "cat"

    def test_count_above_function(self):
        """Test count_above convenience function."""
        predicate = count_above("detections", 5)

        assert isinstance(predicate, CountAbove)
        assert predicate.src == "detections"
        assert predicate.n == 5

    def test_score_above_function(self):
        """Test score_above convenience function."""
        predicate = score_above("detections", 0.8)

        assert isinstance(predicate, ScoreAbove)
        assert predicate.src == "detections"
        assert predicate.threshold == 0.8


class TestIntegration:
    """Integration tests for conditional execution."""

    def test_conditional_workflow_real_predicates(self):
        """Test complete conditional workflow with real predicates."""
        # Create real context
        ctx = ExecutionContext()

        # Store test detection artifacts
        high_conf_dets = create_test_detections(
            instances=[("cat", 0.95, (10, 10, 50, 50)), ("dog", 0.9, (60, 60, 100, 100))]
        )
        low_conf_dets = create_test_detections(instances=[("cat", 0.4, (10, 10, 50, 50))])

        # Test ScoreAbove with high confidence detections
        ctx.store("high_conf_dets", high_conf_dets)
        predicate = ScoreAbove("high_conf_dets", 0.8)
        assert predicate(ctx) is True

        # Test ScoreAbove with low confidence detections
        ctx.store("low_conf_dets", low_conf_dets)
        predicate = ScoreAbove("low_conf_dets", 0.8)
        assert predicate(ctx) is False

        # Test HasLabel
        predicate = HasLabel("high_conf_dets", "cat")
        assert predicate(ctx) is True

        predicate = HasLabel("high_conf_dets", "bird")
        assert predicate(ctx) is False

        # Test CountAbove
        predicate = CountAbove("high_conf_dets", 1)  # 2 > 1
        assert predicate(ctx) is True

        predicate = CountAbove("high_conf_dets", 2)  # 2 == 2 (not >)
        assert predicate(ctx) is False

    def test_nested_conditionals(self):
        """Test nested If nodes."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        # Setup predicates
        outer_predicate = Mock()
        outer_predicate.return_value = True

        inner_predicate = Mock()
        inner_predicate.return_value = False

        # Create nested structure
        inner_if = If(inner_predicate, MockNodeA("InnerThen"), MockNodeB("InnerElse"))

        outer_if = If(outer_predicate, inner_if, Pass())  # Then branch is another If node

        inputs = {"image": MockImageArtifact(100, 100)}
        result = outer_if.run(ctx, **inputs)

        # Should execute: outer=True -> inner -> inner=False -> InnerElse
        assert result == {"result": MockResultArtifact("result_B")}

        # Both predicates should be called
        outer_predicate.assert_called_once()
        inner_predicate.assert_called_once()

    def test_if_with_pass_else(self):
        """Test If node with Pass() as else branch."""
        ctx = Mock(spec=ExecutionContext)
        ctx.record_metric = Mock()

        predicate = Mock()
        predicate.return_value = False  # Should execute else branch (Pass)

        if_node = If(predicate, MockNodeA(), Pass("DefaultPass"))

        inputs = {"image": MockImageArtifact(100, 100)}
        result = if_node.run(ctx, **inputs)

        # Pass node returns empty dict
        assert result == {}

    def test_multiple_predicates_composition(self):
        """Test combining multiple predicates in decision logic."""
        ctx = ExecutionContext()

        # Store test data
        dets = create_test_detections(instances=[("cat", 0.95, (10, 10, 50, 50)), ("dog", 0.9, (60, 60, 100, 100))])
        ctx.store("detections", dets)

        # Test multiple separate predicates
        has_cat = HasLabel("detections", "cat")
        high_score = ScoreAbove("detections", 0.8)
        many_objects = CountAbove("detections", 1)

        assert has_cat(ctx) is True
        assert high_score(ctx) is True
        assert many_objects(ctx) is True

        # Different detections
        single_low_det = create_test_detections(instances=[("bird", 0.3, (10, 10, 50, 50))])
        ctx.store("single_low", single_low_det)

        low_score = ScoreAbove("single_low", 0.8)
        no_cat = HasLabel("single_low", "cat")
        few_objects = CountAbove("single_low", 1)

        assert low_score(ctx) is False
        assert no_cat(ctx) is False
        assert few_objects(ctx) is False
