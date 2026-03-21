"""Conditional execution nodes and predicates for the MATA graph system.

Provides conditional branching capabilities allowing graphs to make decisions
based on detection results, confidence scores, and label presence.

Example usage:
    ```python
    from mata.core.graph.conditionals import If, HasLabel, CountAbove, ScoreAbove, Pass
    from mata.nodes import Detect, Segment

    # Conditional segmentation: only segment if cats are detected
    graph = Graph("conditional_segmentation").then(
        Detect(using="detector", out="dets")
    ).then(
        If(
            predicate=HasLabel("dets", "cat"),
            then_branch=Segment(using="segmenter", image="image", dets="dets", out="masks"),
            else_branch=Pass()
        )
    )

    # Quality-based processing: different post-processing for high vs low confidence
    graph = Graph("quality_based_processing").then(
        Detect(using="detector", out="dets")
    ).then(
        If(
            predicate=ScoreAbove("dets", 0.8),
            then_branch=TopK(src="dets", k=5, out="final"),
            else_branch=Filter(src="dets", score_gt=0.3, out="final")
        )
    )
    ```
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.graph.node import Node


class Predicate(ABC):
    """Base class for conditional predicates.

    Predicates are callables that take an ExecutionContext and return boolean
    values to determine conditional branching. They examine intermediate
    artifacts stored in the context to make decisions.
    """

    @abstractmethod
    def __call__(self, ctx: ExecutionContext) -> bool:
        """Evaluate predicate against execution context.

        Args:
            ctx: Execution context containing artifacts and providers

        Returns:
            True if condition is met, False otherwise
        """
        pass


class HasLabel(Predicate):
    """Check if detections contain a specific label.

    Useful for conditional processing based on object presence.
    Searches through both instance labels and entity labels.

    Args:
        src: Name of the Detections artifact in context
        label: Label name to search for (case-insensitive)

    Example:
        ```python
        # Only segment if cats are detected
        predicate = HasLabel("detections", "cat")

        # Multiple ways to match:
        # - Instance with label_name="cat"
        # - Entity with label="cat"
        # - Case insensitive: "Cat", "CAT" all match "cat"
        ```
    """

    def __init__(self, src: str, label: str):
        """Initialize label checking predicate.

        Args:
            src: Name of Detections artifact in execution context
            label: Label to check for (case-insensitive)
        """
        self.src = src
        self.label = label.lower()  # Normalize for case-insensitive matching

    def __call__(self, ctx: ExecutionContext) -> bool:
        """Check if label exists in detections.

        Args:
            ctx: Execution context

        Returns:
            True if label found in instances or entities

        Raises:
            KeyError: If source artifact not found in context
            TypeError: If source artifact is not Detections
        """
        try:
            dets = ctx.retrieve(self.src)
        except KeyError:
            # If artifact doesn't exist, treat as no label found
            return False

        if not isinstance(dets, Detections):
            raise TypeError(f"HasLabel predicate expects Detections artifact, got {type(dets).__name__}")

        # Check labels (case-insensitive)
        target_label = self.label.lower()

        # Check instance labels
        for label in dets.labels:
            if label.lower() == target_label:
                return True

        return False

    def __repr__(self) -> str:
        return f"HasLabel(src='{self.src}', label='{self.label}')"


class CountAbove(Predicate):
    """Check if detection count exceeds threshold.

    Counts total detections (instances + entities) and compares
    against threshold. Useful for branching based on scene complexity.

    Args:
        src: Name of the Detections artifact in context
        n: Minimum number of detections required (exclusive)

    Example:
        ```python
        # Different processing for busy vs simple scenes
        predicate = CountAbove("detections", 5)  # More than 5 objects

        # Apply NMS to busy scenes, keep all for simple scenes
        if_node = If(
            predicate=CountAbove("dets", 3),
            then_branch=NMS(src="dets", iou_threshold=0.5, out="filtered"),
            else_branch=Pass()  # Keep all detections
        )
        ```
    """

    def __init__(self, src: str, n: int):
        """Initialize count checking predicate.

        Args:
            src: Name of Detections artifact in execution context
            n: Threshold count (condition is count > n)
        """
        self.src = src
        self.n = n

    def __call__(self, ctx: ExecutionContext) -> bool:
        """Check if detection count exceeds threshold.

        Args:
            ctx: Execution context

        Returns:
            True if total detection count > threshold

        Raises:
            KeyError: If source artifact not found in context
            TypeError: If source artifact is not Detections
        """
        try:
            dets = ctx.retrieve(self.src)
        except KeyError:
            # If artifact doesn't exist, treat as count = 0
            return False

        if not isinstance(dets, Detections):
            raise TypeError(f"CountAbove predicate expects Detections artifact, got {type(dets).__name__}")

        # Count total detections (instances + entities)
        total_count = len(dets.instances) + len(dets.entities)
        return total_count > self.n

    def __repr__(self) -> str:
        return f"CountAbove(src='{self.src}', n={self.n})"


class ScoreAbove(Predicate):
    """Check if maximum confidence score exceeds threshold.

    Examines all confidence scores (instances + entities) and checks
    if the highest score exceeds the threshold. Useful for quality-based
    conditional processing.

    Args:
        src: Name of the Detections artifact in context
        threshold: Minimum confidence score required (exclusive)

    Example:
        ```python
        # High confidence: use strict filtering, Low confidence: use relaxed filtering
        predicate = ScoreAbove("detections", 0.8)

        if_node = If(
            predicate=ScoreAbove("dets", 0.7),
            then_branch=Filter(src="dets", score_gt=0.5, out="filtered"),   # Strict
            else_branch=Filter(src="dets", score_gt=0.2, out="filtered")    # Relaxed
        )
        ```
    """

    def __init__(self, src: str, threshold: float):
        """Initialize score checking predicate.

        Args:
            src: Name of Detections artifact in execution context
            threshold: Score threshold (condition is max_score > threshold)
        """
        self.src = src
        self.threshold = threshold

    def __call__(self, ctx: ExecutionContext) -> bool:
        """Check if maximum score exceeds threshold.

        Args:
            ctx: Execution context

        Returns:
            True if max score > threshold, False if no detections or all below threshold

        Raises:
            KeyError: If source artifact not found in context
            TypeError: If source artifact is not Detections
        """
        try:
            dets = ctx.retrieve(self.src)
        except KeyError:
            # If artifact doesn't exist, treat as no scores
            return False

        if not isinstance(dets, Detections):
            raise TypeError(f"ScoreAbove predicate expects Detections artifact, got {type(dets).__name__}")

        # Get all scores
        scores = dets.scores

        # Handle empty detections
        if len(scores) == 0:
            return False

        # Check max score against threshold
        max_score = float(scores.max())
        return max_score > self.threshold

    def __repr__(self) -> str:
        return f"ScoreAbove(src='{self.src}', threshold={self.threshold})"


class If(Node):
    """Conditional execution node with then/else branches.

    Evaluates a predicate and executes either the then_branch or else_branch
    based on the result. Both branches are Nodes that will be executed with
    the same inputs passed to the If node.

    The If node has dynamic input/output types that match its branches. Input
    validation ensures all branches can accept the provided inputs.

    Args:
        predicate: Callable that takes ExecutionContext and returns bool
        then_branch: Node to execute if predicate returns True
        else_branch: Node to execute if predicate returns False (defaults to Pass())
        name: Optional node name for debugging and metrics

    Example:
        ```python
        from mata.nodes import Filter, TopK

        # Conditional filtering based on detection quality
        conditional_filter = If(
            predicate=ScoreAbove("dets", 0.8),
            then_branch=TopK(src="dets", k=3, out="filtered"),      # High quality: top 3
            else_branch=Filter(src="dets", score_gt=0.3, out="filtered"),  # Low quality: filter
            name="quality_filter"
        )

        # Conditional segmentation only if target objects detected
        conditional_segment = If(
            predicate=HasLabel("dets", "person"),
            then_branch=PromptBoxes(using="sam", dets="dets", out="masks"),
            else_branch=Pass()  # No segmentation needed
        )
        ```
    """

    # Dynamic inputs/outputs - determined by branches
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    def __init__(
        self, predicate: Predicate, then_branch: Node, else_branch: Node | None = None, name: str | None = None
    ):
        """Initialize conditional node.

        Args:
            predicate: Predicate to evaluate for branching decision
            then_branch: Node to execute if predicate is True
            else_branch: Node to execute if predicate is False (defaults to Pass())
            name: Optional node name (defaults to "If")
        """
        super().__init__(name=name or "If")
        self.predicate = predicate
        self.then_branch = then_branch
        self.else_branch = else_branch if else_branch is not None else Pass()

        # Derive If's effective outputs from branch declarations so that
        # validate_outputs() in the scheduler can match produced artifacts.
        self.outputs = self._compute_combined_outputs()

        # Validate branch compatibility during construction
        self._validate_branch_compatibility()

    def _compute_combined_outputs(self) -> dict:
        """Compute the effective output signature from both branches.

        Accounts for nodes that use a dynamic ``out`` / ``output_name`` attribute
        to remap a fixed class-level key to a user-supplied artifact name
        (e.g. ``TopK(out="final_dets")`` declares ``outputs = {"detections": …}``
        but actually emits ``{"final_dets": …}``).
        """

        def _effective(node: Node) -> dict:
            dynamic_name = getattr(node, "output_name", None) or getattr(node, "out", None)
            if dynamic_name is not None and len(node.outputs) == 1:
                return {dynamic_name: next(iter(node.outputs.values()))}
            return dict(node.outputs)

        combined: dict = {}
        combined.update(_effective(self.then_branch))
        combined.update(_effective(self.else_branch))
        return combined

    def _validate_branch_compatibility(self) -> None:
        """Validate that both branches have compatible input/output signatures.

        This ensures the If node can be properly typed and validated.
        For now, we do basic checks - full type validation happens at execution time.

        Raises:
            ValueError: If branches have incompatible signatures
        """
        # Both branches should have inputs defined
        if not hasattr(self.then_branch, "inputs") or not hasattr(self.else_branch, "inputs"):
            raise ValueError("Both branches must define input types")

        # For now, we'll do runtime type checking during execution
        # More sophisticated static analysis can be added later

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Execute conditional logic: evaluate predicate and run appropriate branch.

        Args:
            ctx: Execution context containing artifacts and providers
            **inputs: Input artifacts to pass to the selected branch

        Returns:
            Output artifacts from the executed branch

        Raises:
            Exception: Any exception from predicate evaluation or branch execution
        """
        # Record predicate evaluation start
        predicate_start = time.time()

        try:
            # Evaluate predicate
            condition = self.predicate(ctx)

            # Record predicate evaluation time
            predicate_time = (time.time() - predicate_start) * 1000
            ctx.record_metric(self.name, "predicate_latency_ms", predicate_time)
            ctx.record_metric(self.name, "condition_result", condition)

            # Select and execute branch
            selected_branch = self.then_branch if condition else self.else_branch
            branch_name = "then_branch" if condition else "else_branch"

            # Record branch selection (1.0 = then, 0.0 = else)
            ctx.record_metric(self.name, "selected_branch", 1.0 if condition else 0.0)

            # Execute selected branch
            branch_start = time.time()
            branch_inputs = self._resolve_branch_inputs(ctx, selected_branch, inputs)
            result = selected_branch.run(ctx, **branch_inputs)

            # Record branch execution time
            branch_time = (time.time() - branch_start) * 1000
            ctx.record_metric(self.name, f"{branch_name}_latency_ms", branch_time)

            return result

        except Exception:
            # Record error occurrence (1.0 = error occurred)
            ctx.record_metric(self.name, "error", 1.0)
            raise

    def _resolve_branch_inputs(
        self, ctx: ExecutionContext, branch: "Node", passed_inputs: dict
    ) -> dict:
        """Resolve a branch node's inputs, filling gaps from the execution context.

        When the scheduler calls ``If.run()`` with no kwargs (because ``If.inputs``
        is empty), branch nodes that declare their own inputs cannot receive them
        through the normal scheduler path.  This method resolves those missing
        inputs directly from the context using the branch node's ``src``/``*_src``
        attributes and type-aware fallbacks.

        Args:
            ctx: Execution context.
            branch: The branch node to resolve inputs for.
            passed_inputs: Artifacts already passed to ``If.run()`` by the scheduler.

        Returns:
            Dict of resolved input kwargs ready to pass to ``branch.run()``.
        """
        resolved = dict(passed_inputs)

        for input_name, input_type in branch.inputs.items():
            if input_name in resolved:
                continue  # already provided

            # Strategy 1: direct context lookup (works for Filter-style nodes where
            # the input key IS the context artifact name, e.g. inputs={"dets": Detections})
            try:
                resolved[input_name] = ctx.retrieve(input_name)
                continue
            except (KeyError, AttributeError):
                pass

            # Strategy 2: branch has a 'src' attribute pointing to the context key
            # (TopK pattern: inputs={"detections": Detections}, src="dets")
            src = getattr(branch, "src", None)
            if isinstance(src, str):
                try:
                    resolved[input_name] = ctx.retrieve(src)
                    continue
                except (KeyError, AttributeError):
                    pass

            # Strategy 3: scan *_src / *_src_* attributes with type check
            # (PromptBoxes pattern: image_src="image", dets_src="filtered")
            for attr, val in vars(branch).items():
                if not isinstance(val, str):
                    continue
                if not (attr == "src" or attr.endswith("_src")):
                    continue
                try:
                    artifact = ctx.retrieve(val)
                    if isinstance(artifact, input_type):
                        resolved[input_name] = artifact
                        break
                except (KeyError, AttributeError):
                    continue

            if input_name in resolved:
                continue

            # Strategy 4: input namespace (e.g. "input.image")
            try:
                resolved[input_name] = ctx.retrieve(f"input.{input_name}")
            except (KeyError, AttributeError):
                pass  # Missing inputs will be caught by validate_inputs

        return resolved

    def validate_outputs(self, outputs: dict) -> None:
        """Validate If node outputs leniently.

        The If node may execute either branch, so only artifacts that were
        actually produced are type-checked. Missing outputs are allowed —
        e.g. when else_branch=Pass() produces nothing.
        """
        from mata.core.exceptions import ValidationError

        for output_name, artifact in outputs.items():
            if output_name in self.outputs:
                expected_type = self.outputs[output_name]
                self._assert_artifact_type(output_name, artifact, expected_type, "output")
            try:
                artifact.validate()
            except Exception as e:
                raise ValidationError(f"Node '{self.name}' produced invalid artifact: {e}")

    def __repr__(self) -> str:
        return f"If(predicate={self.predicate}, then={self.then_branch.name}, else={self.else_branch.name})"


class Pass(Node):
    """No-operation node that returns empty outputs.

    Useful as a default else_branch in conditional nodes or as a placeholder
    in graph construction. Accepts any inputs and produces no outputs.

    Example:
        ```python
        # Conditional segmentation - segment only if people detected
        conditional_node = If(
            predicate=HasLabel("dets", "person"),
            then_branch=PromptBoxes(using="sam", dets="dets", out="masks"),
            else_branch=Pass()  # Do nothing if no people
        )

        # Explicit no-op in sequential chain
        graph = Graph("example").then(
            Detect(using="detector", out="dets")
        ).then(
            Pass()  # Placeholder for future processing
        ).then(
            Filter(src="dets", score_gt=0.5, out="filtered")
        )
        ```
    """

    # Accept any inputs, produce no outputs
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    def __init__(self, name: str | None = None):
        """Initialize pass-through node.

        Args:
            name: Optional node name (defaults to "Pass")
        """
        super().__init__(name=name or "Pass")

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """No-operation: accept inputs and return empty outputs.

        Args:
            ctx: Execution context (unused)
            **inputs: Input artifacts (ignored)

        Returns:
            Empty dictionary (no outputs)
        """
        # Record that pass node was executed
        ctx.record_metric(self.name, "executed", True)
        ctx.record_metric(self.name, "input_count", len(inputs))

        return {}

    def __repr__(self) -> str:
        return f"Pass(name='{self.name}')"


# Convenience functions for simpler predicate creation


def has_label(src: str, label: str) -> HasLabel:
    """Create HasLabel predicate (convenience function).

    Args:
        src: Source artifact name
        label: Label to check for

    Returns:
        HasLabel predicate
    """
    return HasLabel(src, label)


def count_above(src: str, n: int) -> CountAbove:
    """Create CountAbove predicate (convenience function).

    Args:
        src: Source artifact name
        n: Count threshold

    Returns:
        CountAbove predicate
    """
    return CountAbove(src, n)


def score_above(src: str, threshold: float) -> ScoreAbove:
    """Create ScoreAbove predicate (convenience function).

    Args:
        src: Source artifact name
        threshold: Score threshold

    Returns:
        ScoreAbove predicate
    """
    return ScoreAbove(src, threshold)


# Type alias for backwards compatibility and convenience
ConditionalNode = If
