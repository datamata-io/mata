"""Base classes for graph artifact system.

Provides the foundational Artifact base class with type safety, validation,
and serialization interfaces, plus a type registry for runtime checking.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Artifact(ABC):
    """Base class for all graph artifacts with type safety and immutability.

    All artifacts in the MATA graph system inherit from this base class, which provides:
    - Immutability through frozen dataclass
    - Serialization interface (to_dict/from_dict)
    - Validation hooks for subclasses
    - Type-safe artifact handling

    Subclasses must implement:
    - to_dict(): Convert artifact to dictionary representation
    - from_dict(): Construct artifact from dictionary representation

    Subclasses may override:
    - validate(): Custom validation logic (default: no-op)

    Example:
        ```python
        @dataclass(frozen=True)
        class MyArtifact(Artifact):
            value: int
            label: str

            def to_dict(self) -> Dict[str, Any]:
                return {"value": self.value, "label": self.label}

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "MyArtifact":
                return cls(value=data["value"], label=data["label"])

            def validate(self) -> None:
                if self.value < 0:
                    raise ValueError("Value must be non-negative")
        ```
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary representation.

        Returns:
            Dictionary containing all artifact data, serializable to JSON.

        Note:
            Implementation should handle nested artifacts, numpy arrays,
            and other complex types appropriately.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Construct artifact from dictionary representation.

        Args:
            data: Dictionary containing artifact data (typically from to_dict())

        Returns:
            New artifact instance

        Raises:
            ValueError: If data is invalid or missing required fields
            TypeError: If data types are incorrect
        """
        pass

    def validate(self) -> None:
        """Validate artifact data.

        Override this method to implement custom validation logic.
        Called automatically by the graph system before node execution.

        Raises:
            ValueError: If validation fails

        Example:
            ```python
            def validate(self) -> None:
                if len(self.instances) == 0:
                    raise ValueError("Artifact must contain at least one instance")
                if any(inst.score < 0 or inst.score > 1 for inst in self.instances):
                    raise ValueError("Instance scores must be in [0, 1]")
            ```
        """
        pass  # Default: no validation

    def __post_init__(self):
        """Auto-validate on construction (if not frozen)."""
        # Note: This won't be called for frozen dataclasses in normal usage,
        # but is here for documentation and potential unfrozen subclasses
        pass


class ArtifactTypeRegistry:
    """Registry for artifact type management and compatibility checking.

    Provides:
    - Type registration: Associate string names with artifact types
    - Type lookup: Retrieve artifact types by name
    - Compatibility checking: Verify if source artifact can be used as target type

    The registry uses a singleton pattern - all instances share the same type data.
    This ensures consistency across the graph system.

    Example:
        ```python
        # Register custom artifact types
        registry = ArtifactTypeRegistry()
        registry.register("detections", Detections)
        registry.register("masks", Masks)

        # Look up types
        det_type = registry.get("detections")

        # Check compatibility
        is_ok = registry.is_compatible(Detections, Artifact)  # True (subclass)
        is_ok = registry.is_compatible(Detections, Masks)     # False (unrelated)
        ```
    """

    # Class-level storage (singleton pattern)
    _registry: dict[str, type[Artifact]] = {}
    _initialized: bool = False

    def __init__(self):
        """Initialize the registry.

        Uses shared class-level storage to maintain consistency across instances.
        """
        if not ArtifactTypeRegistry._initialized:
            ArtifactTypeRegistry._registry = {}
            ArtifactTypeRegistry._initialized = True

    def register(self, name: str, artifact_type: type[Artifact]) -> None:
        """Register an artifact type with a string name.

        Args:
            name: String identifier for the artifact type (e.g., "detections")
            artifact_type: Class type inheriting from Artifact

        Raises:
            TypeError: If artifact_type is not a subclass of Artifact
            ValueError: If name is already registered with a different type

        Example:
            ```python
            registry.register("detections", Detections)
            registry.register("masks", Masks)
            ```
        """
        # Validate that artifact_type is an Artifact subclass
        if not (inspect.isclass(artifact_type) and issubclass(artifact_type, Artifact)):
            raise TypeError(f"artifact_type must be a subclass of Artifact, got {artifact_type}")

        # Check for naming conflicts
        if name in self._registry:
            existing = self._registry[name]
            if existing != artifact_type:
                raise ValueError(
                    f"Name '{name}' already registered with type {existing.__name__}, "
                    f"cannot register {artifact_type.__name__}"
                )
            # Same type, silently accept (idempotent)
            return

        self._registry[name] = artifact_type

    def get(self, name: str) -> type[Artifact]:
        """Retrieve an artifact type by name.

        Args:
            name: String identifier for the artifact type

        Returns:
            Artifact class type

        Raises:
            KeyError: If name is not registered

        Example:
            ```python
            DetectionsType = registry.get("detections")
            instance = DetectionsType.from_dict(data)
            ```
        """
        if name not in self._registry:
            raise KeyError(f"Artifact type '{name}' not registered. " f"Available types: {list(self._registry.keys())}")
        return self._registry[name]

    def has(self, name: str) -> bool:
        """Check if an artifact type is registered.

        Args:
            name: String identifier for the artifact type

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry

    def is_compatible(self, source: type[Artifact], target: type[Artifact]) -> bool:
        """Check if source artifact type is compatible with target type.

        Compatibility is defined as:
        - Exact match: source == target
        - Subclass: source is a subclass of target
        - Superclass: target is a subclass of source (relaxed compatibility)

        Args:
            source: Source artifact type (what we have)
            target: Target artifact type (what we need)

        Returns:
            True if compatible, False otherwise

        Example:
            ```python
            # Exact match
            registry.is_compatible(Detections, Detections)  # True

            # Subclass
            registry.is_compatible(DetectionsV2, Detections)  # True

            # Unrelated
            registry.is_compatible(Detections, Masks)  # False
            ```
        """
        # Validate inputs
        if not (inspect.isclass(source) and inspect.isclass(target)):
            return False

        # Exact match
        if source == target:
            return True

        # Check subclass relationship (both directions for flexibility)
        try:
            if issubclass(source, target) or issubclass(target, source):
                return True
        except TypeError:
            # Not classes or not in same hierarchy
            return False

        return False

    def list_types(self) -> set[str]:
        """List all registered artifact type names.

        Returns:
            Set of registered type names

        Example:
            ```python
            types = registry.list_types()
            # {'detections', 'masks', 'image', ...}
            ```
        """
        return set(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered types.

        Warning:
            This is primarily for testing. Clearing the registry during
            normal operation may cause errors if graphs reference the types.
        """
        self._registry.clear()

    def __repr__(self) -> str:
        """String representation of registry state."""
        type_names = sorted(self._registry.keys())
        return f"ArtifactTypeRegistry({len(type_names)} types: {type_names})"
