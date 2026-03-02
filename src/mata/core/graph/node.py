"""Node base class for graph system.

Provides the foundational Node base class with typed I/O signatures,
execution interface, and validation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

from mata.core.artifacts.base import Artifact
from mata.core.exceptions import ValidationError


class Node(ABC):
    """Base class for all graph nodes with strong typing.

    Nodes are the fundamental building blocks of MATA graphs. Each node:
    - Declares typed input/output artifacts at the class level
    - Implements a run() method for execution
    - Validates inputs/outputs match declared types
    - Stores configuration and metadata

    Subclasses must:
    - Define class-level `inputs` and `outputs` dicts with artifact types
    - Implement the abstract `run()` method

    Example:
        ```python
        from mata.core.artifacts.image import Image
        from mata.core.artifacts.detections import Detections

        class Detect(Node):
            '''Object detection node.'''

            inputs = {"image": Image}
            outputs = {"detections": Detections}

            def __init__(self, using: str, out: str = "dets", **kwargs):
                super().__init__(name="Detect")
                self.provider_name = using
                self.output_name = out
                self.kwargs = kwargs

            def run(self, ctx: ExecutionContext, image: Image) -> Dict[str, Artifact]:
                detector = ctx.get_provider(Detector, self.provider_name)
                detections = detector.predict(image, **self.kwargs)
                return {self.output_name: detections}
        ```

    Attributes:
        name: Human-readable node name (defaults to class name)
        config: Configuration dictionary passed to __init__
        inputs: Class-level dict mapping input names to artifact types
        outputs: Class-level dict mapping output names to artifact types
    """

    # Class-level type declarations (override in subclasses)
    inputs: dict[str, type[Artifact]] = {}
    outputs: dict[str, type[Artifact]] = {}

    def __init__(self, name: str | None = None, **config):
        """Initialize node with optional name and configuration.

        Args:
            name: Human-readable node name. If None, uses class name.
            **config: Arbitrary configuration parameters stored for later use.
                     Common examples: threshold, device, model_id, etc.

        Example:
            ```python
            node = MyNode(name="detector_1", threshold=0.5, device="cuda")
            assert node.name == "detector_1"
            assert node.config == {"threshold": 0.5, "device": "cuda"}
            ```
        """
        self.name = name or self.__class__.__name__
        self.config = config

    @abstractmethod
    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Execute node with typed inputs, return typed outputs.

        This is the core execution method that must be implemented by all subclasses.
        The scheduler calls this method during graph execution.

        Args:
            ctx: Execution context providing access to providers, artifacts,
                and metrics collection.
            **inputs: Input artifacts matching the declared `inputs` type signature.
                     Names must match keys in the class-level `inputs` dict.

        Returns:
            Dictionary mapping output names to artifact instances.
            Keys should match the declared `outputs` type signature.

        Raises:
            Any exceptions raised during execution will be caught by the scheduler
            and reported with node context.

        Example:
            ```python
            def run(self, ctx: ExecutionContext, image: Image) -> Dict[str, Artifact]:
                # Get provider from context
                detector = ctx.get_provider(Detector, self.provider_name)

                # Execute task
                result = detector.predict(image, threshold=self.config.get("threshold", 0.5))

                # Record metrics
                ctx.record_metric(self.name, "num_detections", len(result.instances))

                # Return outputs
                return {"detections": result}
            ```
        """
        pass

    def validate_inputs(self, inputs: dict[str, Artifact]) -> None:
        """Validate input artifacts match declared types.

        Checks that:
        1. All required inputs are present
        2. Input artifact types match declared types (including subclasses)
        3. No unexpected inputs are provided

        Args:
            inputs: Dictionary of input artifacts to validate

        Raises:
            ValidationError: If validation fails with detailed error message

        Example:
            ```python
            node = Detect(using="detr")
            inputs = {"image": Image(...)}
            node.validate_inputs(inputs)  # OK

            inputs = {"wrong_name": Image(...)}
            node.validate_inputs(inputs)  # Raises ValidationError
            ```
        """
        # Check all required inputs are present
        required = self.required_inputs
        provided = set(inputs.keys())

        missing = required - provided
        if missing:
            raise ValidationError(
                f"Node '{self.name}' missing required inputs: {sorted(missing)}. "
                f"Required: {sorted(required)}, Provided: {sorted(provided)}"
            )

        # Check for unexpected inputs (only if inputs are strictly defined)
        if self.inputs:  # Only validate if node has declared inputs
            unexpected = provided - set(self.inputs.keys())
            if unexpected:
                raise ValidationError(
                    f"Node '{self.name}' received unexpected inputs: {sorted(unexpected)}. "
                    f"Expected: {sorted(self.inputs.keys())}, Received: {sorted(provided)}"
                )

        # Check input types match declared types
        for input_name, artifact in inputs.items():
            if input_name in self.inputs:
                expected_type = self.inputs[input_name]

                # Handle Optional types (unwrap)
                if hasattr(expected_type, "__origin__"):
                    # Handle Optional[T] and Union types
                    import typing

                    origin = getattr(expected_type, "__origin__", None)
                    if origin is typing.Union:
                        args = getattr(expected_type, "__args__", ())
                        # Check if artifact matches any of the union types
                        if not any(isinstance(artifact, arg) for arg in args if arg is not type(None)):
                            raise ValidationError(
                                f"Node '{self.name}' input '{input_name}' has wrong type. "
                                f"Expected one of {args}, got {type(artifact).__name__}"
                            )
                        continue

                # Standard type check (supports subclasses)
                if not isinstance(artifact, expected_type):
                    raise ValidationError(
                        f"Node '{self.name}' input '{input_name}' has wrong type. "
                        f"Expected {expected_type.__name__}, got {type(artifact).__name__}"
                    )

        # Validate each artifact
        for artifact in inputs.values():
            try:
                artifact.validate()
            except Exception as e:
                raise ValidationError(f"Node '{self.name}' received invalid artifact: {e}")

    def validate_outputs(self, outputs: dict[str, Artifact]) -> None:
        """Validate output artifacts match declared types.

        Checks that:
        1. Output count matches declared count (or at least one output produced)
        2. Output artifact types match declared types (including subclasses)

        Nodes may use dynamic output names via ``out`` or ``output_name``
        parameters.  When the declared class-level output key doesn't match the
        runtime key, the validator falls back to a positional type check:
        the *i*-th declared type is checked against the *i*-th produced artifact
        (in insertion order).

        Args:
            outputs: Dictionary of output artifacts to validate

        Raises:
            ValidationError: If validation fails with detailed error message

        Example:
            ```python
            node = Detect(using="detr")
            outputs = {"detections": Detections(...)}
            node.validate_outputs(outputs)  # OK

            outputs = {}  # Missing required output
            node.validate_outputs(outputs)  # Raises ValidationError
            ```
        """
        expected_keys = self.provided_outputs
        provided_keys = set(outputs.keys())

        # Fast path: keys match exactly – strict check
        if expected_keys == provided_keys:
            self._check_output_types_by_name(outputs)
            return

        # Dynamic output names: many nodes let users override the output key
        # via ``out`` / ``output_name``.  When names diverge, validate by
        # checking that each produced artifact is an instance of one of the
        # declared output types.
        dynamic_name = getattr(self, "output_name", None) or getattr(self, "out", None)
        if dynamic_name is not None and len(outputs) == len(self.outputs):
            self._check_output_types_positional(outputs)
            return

        # No dynamic name detected – fall back to strict check
        missing = expected_keys - provided_keys
        if missing:
            raise ValidationError(
                f"Node '{self.name}' missing required outputs: {sorted(missing)}. "
                f"Expected: {sorted(expected_keys)}, Provided: {sorted(provided_keys)}"
            )

        unexpected = provided_keys - expected_keys
        if unexpected:
            raise ValidationError(
                f"Node '{self.name}' produced unexpected outputs: {sorted(unexpected)}. "
                f"Expected: {sorted(expected_keys)}, Produced: {sorted(provided_keys)}"
            )

        self._check_output_types_by_name(outputs)

    def _check_output_types_by_name(self, outputs: dict[str, Artifact]) -> None:
        """Check output types matching by name."""
        for output_name, artifact in outputs.items():
            if output_name in self.outputs:
                expected_type = self.outputs[output_name]
                self._assert_artifact_type(output_name, artifact, expected_type, "output")

        # Validate each artifact
        for artifact in outputs.values():
            try:
                artifact.validate()
            except Exception as e:
                raise ValidationError(f"Node '{self.name}' produced invalid artifact: {e}")

    def _check_output_types_positional(self, outputs: dict[str, Artifact]) -> None:
        """Check output types positionally when keys don't match."""
        declared_types = list(self.outputs.values())
        produced_artifacts = list(outputs.values())

        for idx, (expected_type, artifact) in enumerate(zip(declared_types, produced_artifacts)):
            output_name = list(outputs.keys())[idx]
            self._assert_artifact_type(output_name, artifact, expected_type, "output")

        # Validate each artifact
        for artifact in outputs.values():
            try:
                artifact.validate()
            except Exception as e:
                raise ValidationError(f"Node '{self.name}' produced invalid artifact: {e}")

    def _assert_artifact_type(self, name: str, artifact: Artifact, expected_type: type, direction: str) -> None:
        """Assert an artifact matches the expected type."""
        # Handle Optional types (unwrap)
        if hasattr(expected_type, "__origin__"):
            import typing

            origin = getattr(expected_type, "__origin__", None)
            if origin is typing.Union:
                args = getattr(expected_type, "__args__", ())
                if not any(isinstance(artifact, arg) for arg in args if arg is not type(None)):
                    raise ValidationError(
                        f"Node '{self.name}' {direction} '{name}' has wrong type. "
                        f"Expected one of {args}, got {type(artifact).__name__}"
                    )
                return

        # Standard type check (supports subclasses)
        if not isinstance(artifact, expected_type):
            raise ValidationError(
                f"Node '{self.name}' {direction} '{name}' has wrong type. "
                f"Expected {expected_type.__name__}, got {type(artifact).__name__}"
            )

    @property
    def required_inputs(self) -> set[str]:
        """Get set of required input artifact names.

        Returns:
            Set of input names declared in the class-level `inputs` dict.

        Example:
            ```python
            class MyNode(Node):
                inputs = {"image": Image, "detections": Detections}
                outputs = {"masks": Masks}

            node = MyNode()
            assert node.required_inputs == {"image", "detections"}
            ```
        """
        return set(self.inputs.keys())

    @property
    def provided_outputs(self) -> set[str]:
        """Get set of provided output artifact names.

        Returns:
            Set of output names declared in the class-level `outputs` dict.

        Example:
            ```python
            class MyNode(Node):
                inputs = {"image": Image}
                outputs = {"detections": Detections, "masks": Masks}

            node = MyNode()
            assert node.provided_outputs == {"detections", "masks"}
            ```
        """
        return set(self.outputs.keys())

    def __repr__(self) -> str:
        """Return string representation of node.

        Format: ClassName(name='node_name', inputs={...}, outputs={...})

        Returns:
            Human-readable string representation

        Example:
            ```python
            node = Detect(using="detr", out="dets")
            print(node)
            # Detect(name='Detect', inputs={'image': Image}, outputs={'detections': Detections})
            ```
        """
        input_types = {k: v.__name__ for k, v in self.inputs.items()}
        output_types = {k: v.__name__ for k, v in self.outputs.items()}

        return (
            f"{self.__class__.__name__}(" f"name='{self.name}', " f"inputs={input_types}, " f"outputs={output_types})"
        )
