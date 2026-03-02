"""Unit tests for artifact base classes.

Tests the foundational Artifact base class and ArtifactTypeRegistry,
ensuring proper type safety, validation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mata.core.artifacts.base import Artifact, ArtifactTypeRegistry


# Test artifact implementations
@dataclass(frozen=True)
class SimpleArtifact(Artifact):
    """Simple test artifact with basic fields."""

    value: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleArtifact:
        return cls(
            value=data["value"],
            label=data["label"],
        )


@dataclass(frozen=True)
class ValidatedArtifact(Artifact):
    """Test artifact with validation."""

    score: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidatedArtifact:
        artifact = cls(
            score=data["score"],
            count=data["count"],
        )
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if self.count < 0:
            raise ValueError(f"Count must be non-negative, got {self.count}")


@dataclass(frozen=True)
class NestedArtifact(Artifact):
    """Test artifact with nested data."""

    name: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NestedArtifact:
        return cls(
            name=data["name"],
            metadata=data["metadata"],
        )


@dataclass(frozen=True)
class ExtendedArtifact(SimpleArtifact):
    """Subclass of SimpleArtifact for inheritance tests."""

    extra: str = "default"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtendedArtifact:
        return cls(
            value=data["value"],
            label=data["label"],
            extra=data.get("extra", "default"),
        )


class TestArtifactBase:
    """Test suite for Artifact base class."""

    def test_artifact_immutability(self):
        """Test that artifacts are immutable (frozen)."""
        artifact = SimpleArtifact(value=42, label="test")

        # Should not be able to modify fields
        with pytest.raises(Exception):  # FrozenInstanceError
            artifact.value = 100

    def test_artifact_to_dict(self):
        """Test artifact serialization to dict."""
        artifact = SimpleArtifact(value=42, label="test")
        data = artifact.to_dict()

        assert data == {"value": 42, "label": "test"}
        assert isinstance(data, dict)

    def test_artifact_from_dict(self):
        """Test artifact deserialization from dict."""
        data = {"value": 42, "label": "test"}
        artifact = SimpleArtifact.from_dict(data)

        assert artifact.value == 42
        assert artifact.label == "test"
        assert isinstance(artifact, SimpleArtifact)

    def test_artifact_round_trip(self):
        """Test serialization round-trip (to_dict → from_dict)."""
        original = SimpleArtifact(value=42, label="test")
        data = original.to_dict()
        restored = SimpleArtifact.from_dict(data)

        assert restored.value == original.value
        assert restored.label == original.label

    def test_artifact_validation_pass(self):
        """Test artifact validation with valid data."""
        artifact = ValidatedArtifact(score=0.75, count=10)
        artifact.validate()  # Should not raise

    def test_artifact_validation_fail_score(self):
        """Test artifact validation fails on invalid score."""
        artifact = ValidatedArtifact(score=1.5, count=10)

        with pytest.raises(ValueError, match="Score must be in"):
            artifact.validate()

    def test_artifact_validation_fail_count(self):
        """Test artifact validation fails on negative count."""
        artifact = ValidatedArtifact(score=0.5, count=-1)

        with pytest.raises(ValueError, match="Count must be non-negative"):
            artifact.validate()

    def test_artifact_validation_in_from_dict(self):
        """Test validation called during deserialization."""
        data = {"score": 2.0, "count": 5}

        with pytest.raises(ValueError, match="Score must be in"):
            ValidatedArtifact.from_dict(data)

    def test_artifact_nested_data(self):
        """Test artifact with nested dictionary data."""
        artifact = NestedArtifact(name="test", metadata={"key1": "value1", "key2": 42})

        data = artifact.to_dict()
        assert data["name"] == "test"
        assert data["metadata"]["key1"] == "value1"
        assert data["metadata"]["key2"] == 42

        restored = NestedArtifact.from_dict(data)
        assert restored.name == artifact.name
        assert restored.metadata == artifact.metadata

    def test_artifact_inheritance(self):
        """Test artifact subclass behavior."""
        artifact = ExtendedArtifact(value=42, label="test", extra="bonus")

        # Should have all fields
        assert artifact.value == 42
        assert artifact.label == "test"
        assert artifact.extra == "bonus"

        # Serialization should include all fields
        data = artifact.to_dict()
        assert data == {"value": 42, "label": "test", "extra": "bonus"}

        # Deserialization should work
        restored = ExtendedArtifact.from_dict(data)
        assert restored.value == artifact.value
        assert restored.label == artifact.label
        assert restored.extra == artifact.extra


class TestArtifactTypeRegistry:
    """Test suite for ArtifactTypeRegistry."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Clear registry before each test."""
        registry = ArtifactTypeRegistry()
        registry.clear()
        yield
        registry.clear()

    def test_registry_register_type(self):
        """Test registering an artifact type."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)

        assert registry.has("simple")
        assert registry.get("simple") == SimpleArtifact

    def test_registry_register_multiple_types(self):
        """Test registering multiple artifact types."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("validated", ValidatedArtifact)
        registry.register("nested", NestedArtifact)

        assert registry.has("simple")
        assert registry.has("validated")
        assert registry.has("nested")
        assert registry.get("simple") == SimpleArtifact
        assert registry.get("validated") == ValidatedArtifact
        assert registry.get("nested") == NestedArtifact

    def test_registry_register_idempotent(self):
        """Test that registering the same type twice is idempotent."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("simple", SimpleArtifact)  # Should not raise

        assert registry.get("simple") == SimpleArtifact

    def test_registry_register_conflict(self):
        """Test that registering different types with same name raises error."""
        registry = ArtifactTypeRegistry()
        registry.register("artifact", SimpleArtifact)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("artifact", ValidatedArtifact)

    def test_registry_register_invalid_type(self):
        """Test that registering non-Artifact type raises error."""
        registry = ArtifactTypeRegistry()

        with pytest.raises(TypeError, match="must be a subclass of Artifact"):
            registry.register("invalid", dict)

    def test_registry_register_non_class(self):
        """Test that registering non-class raises error."""
        registry = ArtifactTypeRegistry()

        with pytest.raises(TypeError, match="must be a subclass of Artifact"):
            registry.register("invalid", "not a class")

    def test_registry_get_missing_type(self):
        """Test getting unregistered type raises error."""
        registry = ArtifactTypeRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_registry_get_error_message(self):
        """Test that get error message lists available types."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("validated", ValidatedArtifact)

        with pytest.raises(KeyError, match="Available types"):
            registry.get("missing")

    def test_registry_has_type(self):
        """Test checking if type is registered."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)

        assert registry.has("simple") is True
        assert registry.has("nonexistent") is False

    def test_registry_list_types(self):
        """Test listing all registered types."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("validated", ValidatedArtifact)
        registry.register("nested", NestedArtifact)

        types = registry.list_types()
        assert types == {"simple", "validated", "nested"}

    def test_registry_list_types_empty(self):
        """Test listing types when registry is empty."""
        registry = ArtifactTypeRegistry()
        types = registry.list_types()
        assert types == set()

    def test_registry_clear(self):
        """Test clearing the registry."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("validated", ValidatedArtifact)

        assert len(registry.list_types()) == 2

        registry.clear()
        assert len(registry.list_types()) == 0
        assert not registry.has("simple")

    def test_registry_is_compatible_exact_match(self):
        """Test compatibility checking with exact type match."""
        registry = ArtifactTypeRegistry()

        assert registry.is_compatible(SimpleArtifact, SimpleArtifact) is True
        assert registry.is_compatible(ValidatedArtifact, ValidatedArtifact) is True

    def test_registry_is_compatible_subclass(self):
        """Test compatibility checking with subclass."""
        registry = ArtifactTypeRegistry()

        # ExtendedArtifact is a subclass of SimpleArtifact
        assert registry.is_compatible(ExtendedArtifact, SimpleArtifact) is True

    def test_registry_is_compatible_superclass(self):
        """Test compatibility checking with superclass (relaxed)."""
        registry = ArtifactTypeRegistry()

        # Relaxed compatibility: superclass also compatible
        assert registry.is_compatible(SimpleArtifact, ExtendedArtifact) is True

    def test_registry_is_compatible_unrelated(self):
        """Test compatibility checking with unrelated types."""
        registry = ArtifactTypeRegistry()

        assert registry.is_compatible(SimpleArtifact, ValidatedArtifact) is False
        assert registry.is_compatible(ValidatedArtifact, NestedArtifact) is False

    def test_registry_is_compatible_to_base(self):
        """Test compatibility checking against Artifact base class."""
        registry = ArtifactTypeRegistry()

        # All artifacts should be compatible with Artifact base
        assert registry.is_compatible(SimpleArtifact, Artifact) is True
        assert registry.is_compatible(ValidatedArtifact, Artifact) is True
        assert registry.is_compatible(ExtendedArtifact, Artifact) is True

    def test_registry_is_compatible_invalid_input(self):
        """Test compatibility checking with invalid inputs."""
        registry = ArtifactTypeRegistry()

        # Non-class inputs should return False
        assert registry.is_compatible("not a class", SimpleArtifact) is False
        assert registry.is_compatible(SimpleArtifact, "not a class") is False
        assert registry.is_compatible(42, SimpleArtifact) is False

    def test_registry_singleton_behavior(self):
        """Test that registry instances share the same state."""
        registry1 = ArtifactTypeRegistry()
        registry2 = ArtifactTypeRegistry()

        registry1.register("simple", SimpleArtifact)

        # registry2 should see the registration from registry1
        assert registry2.has("simple")
        assert registry2.get("simple") == SimpleArtifact

    def test_registry_repr(self):
        """Test string representation of registry."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)
        registry.register("validated", ValidatedArtifact)

        repr_str = repr(registry)
        assert "ArtifactTypeRegistry" in repr_str
        assert "2 types" in repr_str
        assert "simple" in repr_str
        assert "validated" in repr_str


class TestArtifactAdvanced:
    """Advanced test cases for artifact system."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Clear registry before each test."""
        registry = ArtifactTypeRegistry()
        registry.clear()
        yield
        registry.clear()

    def test_artifact_type_lookup_workflow(self):
        """Test complete workflow: register, lookup, instantiate."""
        registry = ArtifactTypeRegistry()
        registry.register("simple", SimpleArtifact)

        # Lookup type
        ArtifactType = registry.get("simple")  # noqa: N806

        # Create instance
        data = {"value": 42, "label": "test"}
        artifact = ArtifactType.from_dict(data)

        assert isinstance(artifact, SimpleArtifact)
        assert artifact.value == 42
        assert artifact.label == "test"

    def test_artifact_polymorphism(self):
        """Test polymorphic behavior through base class."""
        artifacts: list[Artifact] = [
            SimpleArtifact(value=1, label="a"),
            ValidatedArtifact(score=0.5, count=10),
            NestedArtifact(name="test", metadata={}),
        ]

        # All should serialize
        for artifact in artifacts:
            data = artifact.to_dict()
            assert isinstance(data, dict)

        # All should have validate method
        for artifact in artifacts:
            artifact.validate()  # Should not raise

    def test_artifact_type_checking(self):
        """Test type checking with isinstance."""
        simple = SimpleArtifact(value=42, label="test")
        extended = ExtendedArtifact(value=42, label="test", extra="bonus")

        # Both are Artifacts
        assert isinstance(simple, Artifact)
        assert isinstance(extended, Artifact)

        # Both are SimpleArtifact (extended is subclass)
        assert isinstance(simple, SimpleArtifact)
        assert isinstance(extended, SimpleArtifact)

        # Only extended is ExtendedArtifact
        assert not isinstance(simple, ExtendedArtifact)
        assert isinstance(extended, ExtendedArtifact)

    def test_registry_with_inheritance_hierarchy(self):
        """Test registry with class inheritance."""
        registry = ArtifactTypeRegistry()
        registry.register("base", SimpleArtifact)
        registry.register("extended", ExtendedArtifact)

        # Both should be retrievable
        base_type = registry.get("base")
        extended_type = registry.get("extended")

        # Compatibility should work both ways
        assert registry.is_compatible(base_type, base_type)
        assert registry.is_compatible(extended_type, extended_type)
        assert registry.is_compatible(extended_type, base_type)
        assert registry.is_compatible(base_type, extended_type)

    def test_artifact_default_validation(self):
        """Test that default validate() does nothing."""
        # SimpleArtifact doesn't override validate()
        artifact = SimpleArtifact(value=-999, label="")
        artifact.validate()  # Should not raise even with unusual values

    def test_artifact_from_dict_with_extra_fields(self):
        """Test deserialization ignores extra fields gracefully."""
        data = {
            "value": 42,
            "label": "test",
            "extra_field": "ignored",
            "another_field": 100,
        }

        # Should work, ignoring extra fields
        artifact = SimpleArtifact.from_dict(data)
        assert artifact.value == 42
        assert artifact.label == "test"

    def test_artifact_from_dict_with_missing_fields(self):
        """Test deserialization fails on missing required fields."""
        data = {"value": 42}  # Missing 'label'

        with pytest.raises(KeyError):
            SimpleArtifact.from_dict(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
