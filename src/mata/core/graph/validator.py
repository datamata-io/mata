"""Graph validation system.

Provides comprehensive validation for graph structure, type safety,
dependencies, and correctness before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.graph.node import Node

from mata.core.artifacts.base import Artifact
from mata.core.exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of graph validation with errors and warnings.

    Attributes:
        valid: True if validation passed (no errors)
        errors: List of error messages (prevent execution)
        warnings: List of warning messages (may proceed with caution)

    Example:
        ```python
        result = validator.validate(nodes, wiring)
        if not result.valid:
            print("Validation failed:")
            for error in result.errors:
                print(f"  - {error}")
            result.raise_if_invalid()  # Raises ValidationError
        ```
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed.

        Raises:
            ValidationError: If valid=False, includes all error messages

        Example:
            ```python
            result = validator.validate(nodes, wiring)
            result.raise_if_invalid()  # OK if valid=True, raises if valid=False
            ```
        """
        if not self.valid:
            error_msg = "Graph validation failed:\n"
            for i, error in enumerate(self.errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise ValidationError(error_msg.strip())

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.valid:
            status = "✓ Valid"
            if self.warnings:
                status += f" ({len(self.warnings)} warnings)"
        else:
            status = f"✗ Invalid ({len(self.errors)} errors"
            if self.warnings:
                status += f", {len(self.warnings)} warnings"
            status += ")"

        result = [status]

        if self.errors:
            result.append("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                result.append(f"  {i}. {error}")

        if self.warnings:
            result.append("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                result.append(f"  {i}. {warning}")

        return "\n".join(result)


class GraphValidator:
    """Validates graph structure and type safety.

    Performs comprehensive validation checks on graphs before execution:
    - Type compatibility between connected nodes
    - Dependency resolution (all required inputs satisfied)
    - Cycle detection (enforces DAG structure)
    - Name collision detection (unique output names)
    - Provider capability verification

    Example:
        ```python
        validator = GraphValidator()

        # Create nodes
        detect_node = Detect(using="detr")
        segment_node = SegmentImage(using="sam")

        # Define wiring
        wiring = {
            "segment_node.image": "input.image",
            "segment_node.detections": "detect_node.detections"
        }

        # Validate
        result = validator.validate(
            nodes=[detect_node, segment_node],
            wiring=wiring
        )

        if not result.valid:
            print(result)
            result.raise_if_invalid()
        ```

    Performance:
        Designed to complete in <100ms for typical graphs (5-20 nodes).
        Uses efficient algorithms (topological sort with DFS for cycles).
    """

    def validate(
        self, nodes: list[Node], wiring: dict[str, str], providers: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Run all validation checks on graph.

        Args:
            nodes: List of nodes in the graph
            wiring: Dictionary mapping node inputs to artifact names
                   Format: {"node_name.input_name": "artifact_name"}
            providers: Optional provider registry for capability checking

        Returns:
            ValidationResult with errors and warnings

        Example:
            ```python
            result = validator.validate(
                nodes=[detect_node, filter_node],
                wiring={"filter_node.detections": "detect_node.detections"}
            )

            assert result.valid
            assert len(result.errors) == 0
            ```
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Basic structure validation
        if not nodes:
            errors.append("Graph must contain at least one node")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        # Check for name collisions
        collision_errors = self.check_name_collisions(nodes)
        errors.extend(collision_errors)

        # Check dependencies
        dependency_errors = self.check_dependencies(nodes, wiring)
        errors.extend(dependency_errors)

        # Check type compatibility
        type_errors = self.check_type_compatibility(nodes, wiring)
        errors.extend(type_errors)

        # Detect cycles
        cycle_path = self.detect_cycles(nodes, wiring)
        if cycle_path:
            cycle_str = " → ".join(cycle_path)
            errors.append(f"Circular dependency detected: {cycle_str}")

        # Check provider capabilities (if providers given)
        if providers is not None:
            provider_errors = self.check_provider_capabilities(nodes, providers)
            errors.extend(provider_errors)

        # Determine validity
        valid = len(errors) == 0

        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    def check_type_compatibility(self, nodes: list[Node], wiring: dict[str, str]) -> list[str]:
        """Verify artifact types match across connections.

        For each connection in the wiring, validates that:
        - The source node produces an output of the expected type
        - The target node expects an input of a compatible type
        - Types are compatible (exact match or subclass)

        Args:
            nodes: List of nodes in the graph
            wiring: Dictionary mapping node inputs to artifact names

        Returns:
            List of error messages (empty if all types are compatible)

        Example:
            ```python
            # Valid connection: Detect outputs Detections, Filter expects Detections
            wiring = {"filter.detections": "detect.detections"}
            errors = validator.check_type_compatibility([detect, filter], wiring)
            assert len(errors) == 0

            # Invalid connection: type mismatch
            wiring = {"classify.image": "detect.detections"}  # Wrong type!
            errors = validator.check_type_compatibility([detect, classify], wiring)
            assert len(errors) > 0
            ```
        """
        errors: list[str] = []

        # Build artifact type map: {artifact_name: artifact_type}
        artifact_types: dict[str, type[Artifact]] = {}

        # Map node outputs
        node_map = {node.name: node for node in nodes}
        for node in nodes:
            for output_name, output_type in node.outputs.items():
                artifact_name = f"{node.name}.{output_name}"
                artifact_types[artifact_name] = output_type

            # Register dynamic output names so wiring like "Detect.dets" resolves
            dynamic_name = getattr(node, "output_name", None) or getattr(node, "out", None)
            if dynamic_name is not None and len(node.outputs) == 1:
                output_type = next(iter(node.outputs.values()))
                artifact_types[f"{node.name}.{dynamic_name}"] = output_type

        # Check each wired connection
        for target, source in wiring.items():
            # Parse target (node_name.input_name)
            if "." not in target:
                errors.append(f"Invalid wiring target '{target}': must be 'node_name.input_name'")
                continue

            node_name, input_name = target.rsplit(".", 1)

            # Get target node
            if node_name not in node_map:
                errors.append(f"Unknown node '{node_name}' in wiring target '{target}'")
                continue

            target_node = node_map[node_name]

            # Get expected input type
            if input_name not in target_node.inputs:
                errors.append(
                    f"Node '{node_name}' has no input '{input_name}'. "
                    f"Available inputs: {list(target_node.inputs.keys())}"
                )
                continue

            expected_type = target_node.inputs[input_name]

            # Handle Optional types
            if hasattr(expected_type, "__origin__"):
                import typing

                origin = getattr(expected_type, "__origin__", None)
                if origin is typing.Union:
                    args = getattr(expected_type, "__args__", ())
                    if type(None) in args:
                        # Unwrap Optional[T] to T
                        expected_type = next(arg for arg in args if arg is not type(None))

            # Get source artifact type
            if source not in artifact_types:
                # Could be an external input (e.g., "input.image")
                # Skip type checking for external inputs
                if not source.startswith("input."):
                    errors.append(
                        f"Artifact '{source}' not found in graph outputs. " f"Available: {list(artifact_types.keys())}"
                    )
                continue

            source_type = artifact_types[source]

            # Check type compatibility
            if not self._is_compatible_type(source_type, expected_type):
                errors.append(
                    f"Type mismatch in wiring '{target}': "
                    f"source '{source}' produces {source_type.__name__}, "
                    f"but target expects {expected_type.__name__}"
                )

        return errors

    def check_dependencies(self, nodes: list[Node], wiring: dict[str, str]) -> list[str]:
        """Ensure all required inputs are provided.

        For each node, validates that:
        - All required inputs are wired to a source
        - Sources are either node outputs or external inputs

        Args:
            nodes: List of nodes in the graph
            wiring: Dictionary mapping node inputs to artifact names

        Returns:
            List of error messages (empty if all dependencies satisfied)

        Example:
            ```python
            # Valid: all inputs wired
            wiring = {
                "detect.image": "input.image",
                "filter.detections": "detect.detections"
            }
            errors = validator.check_dependencies(nodes, wiring)
            assert len(errors) == 0

            # Invalid: missing required input
            wiring = {}  # detect.image not wired!
            errors = validator.check_dependencies(nodes, wiring)
            assert len(errors) > 0
            ```
        """
        errors: list[str] = []

        # Build set of available artifacts
        available_artifacts: set[str] = set()

        # Add node outputs (both static declared names and dynamic output names)
        for node in nodes:
            for output_name in node.outputs.keys():
                artifact_name = f"{node.name}.{output_name}"
                available_artifacts.add(artifact_name)

            # Also register dynamic output names from ``output_name`` / ``out`` attrs
            dynamic_name = getattr(node, "output_name", None) or getattr(node, "out", None)
            if dynamic_name is not None:
                available_artifacts.add(f"{node.name}.{dynamic_name}")

        # Add external inputs (anything starting with "input.")
        for source in wiring.values():
            if source.startswith("input."):
                available_artifacts.add(source)

        # Check each node's required inputs
        for node in nodes:
            for input_name in node.required_inputs:
                target = f"{node.name}.{input_name}"

                # Check if input is wired
                if target not in wiring:
                    errors.append(
                        f"Node '{node.name}' requires input '{input_name}' "
                        f"but it is not wired. Available artifacts: {sorted(available_artifacts)}"
                    )
                    continue

                # Check if source exists
                source = wiring[target]
                if source not in available_artifacts:
                    errors.append(
                        f"Node '{node.name}' input '{input_name}' is wired to '{source}', "
                        f"but '{source}' is not available. "
                        f"Available artifacts: {sorted(available_artifacts)}"
                    )

        return errors

    def detect_cycles(self, nodes: list[Node], wiring: dict[str, str]) -> list[str] | None:
        """Detect circular dependencies using DFS.

        Graphs must be Directed Acyclic Graphs (DAGs). This method detects
        cycles using depth-first search with a recursive stack.

        Args:
            nodes: List of nodes in the graph
            wiring: Dictionary mapping node inputs to artifact names

        Returns:
            List of node names forming a cycle (if found), otherwise None

        Example:
            ```python
            # Valid DAG: A → B → C
            wiring = {"B.input": "A.output", "C.input": "B.output"}
            cycle = validator.detect_cycles(nodes, wiring)
            assert cycle is None

            # Invalid: A → B → A (cycle!)
            wiring = {"B.input": "A.output", "A.input": "B.output"}
            cycle = validator.detect_cycles(nodes, wiring)
            assert cycle == ["A", "B", "A"]
            ```
        """
        # Build adjacency list: node -> list of dependent nodes
        graph: dict[str, set[str]] = {node.name: set() for node in nodes}

        for target, source in wiring.items():
            # Parse target (node_name.input_name)
            if "." not in target:
                continue

            target_node = target.rsplit(".", 1)[0]

            # Parse source (node_name.output_name or input.*)
            if "." not in source or source.startswith("input."):
                continue

            source_node = source.rsplit(".", 1)[0]

            # Add edge: source_node -> target_node
            if source_node in graph and target_node in graph:
                graph[source_node].add(target_node)

        # DFS cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            """Depth-first search for cycle detection."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Cycle found! Build cycle path
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            path.pop()
            return None

        # Check each node (handles disconnected graphs)
        for node in nodes:
            if node.name not in visited:
                cycle = dfs(node.name)
                if cycle:
                    return cycle

        return None

    def check_name_collisions(self, nodes: list[Node]) -> list[str]:
        """Find duplicate output names across nodes.

        Each node must have a unique name, and each node's outputs must
        have unique artifact names (node_name.output_name).

        Args:
            nodes: List of nodes in the graph

        Returns:
            List of error messages (empty if no collisions)

        Example:
            ```python
            # Valid: unique node names
            node1 = Detect(using="detr")  # name="Detect"
            node2 = Filter(...)            # name="Filter"
            errors = validator.check_name_collisions([node1, node2])
            assert len(errors) == 0

            # Invalid: duplicate node names
            node1 = Detect(using="detr")   # name="Detect"
            node2 = Detect(using="rtdetr") # name="Detect" (collision!)
            errors = validator.check_name_collisions([node1, node2])
            assert len(errors) > 0
            ```
        """
        errors: list[str] = []

        # Check for duplicate node names
        node_names: dict[str, int] = {}
        for node in nodes:
            if node.name in node_names:
                node_names[node.name] += 1
            else:
                node_names[node.name] = 1

        duplicates = [name for name, count in node_names.items() if count > 1]
        if duplicates:
            for name in duplicates:
                count = node_names[name]
                errors.append(f"Duplicate node name '{name}' ({count} nodes). " f"Each node must have a unique name.")

        # Check for duplicate output artifact names
        output_artifacts: dict[str, list[str]] = {}
        for node in nodes:
            for output_name in node.outputs.keys():
                artifact_name = f"{node.name}.{output_name}"
                if artifact_name in output_artifacts:
                    output_artifacts[artifact_name].append(node.name)
                else:
                    output_artifacts[artifact_name] = [node.name]

        for artifact_name, producing_nodes in output_artifacts.items():
            if len(producing_nodes) > 1:
                errors.append(f"Duplicate output artifact '{artifact_name}' produced by nodes: " f"{producing_nodes}")

        return errors

    def check_provider_capabilities(self, nodes: list[Node], providers: dict[str, Any]) -> list[str]:
        """Verify providers implement required capabilities.

        For nodes that use providers (e.g., Detect using "detr"), validates that:
        - The required provider exists in the registry
        - The provider implements the expected capability protocol

        Args:
            nodes: List of nodes in the graph
            providers: Provider registry mapping names to provider instances

        Returns:
            List of error messages (empty if all providers available)

        Example:
            ```python
            providers = {
                "detr": detr_detector,
                "sam": sam_segmenter
            }

            # Valid: all providers exist
            node = Detect(using="detr")
            errors = validator.check_provider_capabilities([node], providers)
            assert len(errors) == 0

            # Invalid: missing provider
            node = Detect(using="yolo")  # "yolo" not in providers!
            errors = validator.check_provider_capabilities([node], providers)
            assert len(errors) > 0
            ```
        """
        errors: list[str] = []

        for node in nodes:
            # Check if node has a provider requirement (common pattern: node.provider_name)
            if hasattr(node, "provider_name"):
                provider_name = node.provider_name

                if provider_name not in providers:
                    errors.append(
                        f"Node '{node.name}' requires provider '{provider_name}', "
                        f"but it is not available. "
                        f"Available providers: {sorted(providers.keys())}"
                    )
                # Note: Protocol checking could be added here if we have capability info
                # For now, we just check existence

        return errors

    def _is_compatible_type(self, source_type: type[Artifact], target_type: type[Artifact]) -> bool:
        """Check if source type is compatible with target type.

        Types are compatible if:
        - They are the same type
        - source_type is a subclass of target_type
        - source_type is the base Artifact class (wildcard: runtime type unknown)

        Args:
            source_type: Type produced by source node
            target_type: Type expected by target node

        Returns:
            True if types are compatible
        """
        # Base Artifact is a wildcard (e.g. ValkeyLoad can produce any subtype);
        # defer the concrete type check to runtime.
        if source_type is Artifact:
            return True
        try:
            return issubclass(source_type, target_type)
        except TypeError:
            # Handle non-class types (generics, etc.)
            return source_type == target_type
