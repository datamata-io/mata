"""Unit tests for ExecutionContext.

Tests cover:
- Artifact storage/retrieval
- Provider lookup
- Device resolution (auto/cuda/cpu)
- Metrics recording
- Caching behavior
- Cleanup
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from mata.core.artifacts.base import Artifact
from mata.core.exceptions import ValidationError
from mata.core.graph.context import ExecutionContext


# Mock artifacts for testing
@dataclass(frozen=True)
class MockArtifact(Artifact):
    """Mock artifact for testing."""

    value: int
    label: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockArtifact:
        return cls(value=data["value"], label=data.get("label", "test"))

    def validate(self) -> None:
        if self.value < 0:
            raise ValueError("Value must be non-negative")


@dataclass(frozen=True)
class InvalidArtifact(Artifact):
    """Mock artifact that always fails validation."""

    value: int

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidArtifact:
        return cls(value=data["value"])

    def validate(self) -> None:
        raise ValueError("Always invalid")


# Mock provider for testing
class MockDetector:
    """Mock detector provider."""

    def predict(self, image, **kwargs):
        return f"Detected with {kwargs}"


class MockSegmenter:
    """Mock segmenter provider."""

    def segment(self, image, **kwargs):
        return f"Segmented with {kwargs}"


class TestExecutionContextBasics:
    """Test basic ExecutionContext functionality."""

    def test_init_default(self):
        """Test default initialization."""
        ctx = ExecutionContext()

        assert ctx.providers == {}
        assert ctx.device in ["cuda", "cpu"]  # Auto-detected
        assert not hasattr(ctx, "_artifacts") or len(ctx._artifacts) == 0

    def test_init_with_providers(self):
        """Test initialization with providers."""
        providers = {"detect": {"detr": MockDetector()}, "segment": {"sam": MockSegmenter()}}

        ctx = ExecutionContext(providers=providers, device="cpu")

        assert ctx.providers == providers
        assert ctx.device == "cpu"

    def test_init_with_device(self):
        """Test initialization with explicit device."""
        ctx = ExecutionContext(device="cpu")
        assert ctx.device == "cpu"

    def test_repr(self):
        """Test string representation."""
        providers = {"detect": {"detr": MockDetector(), "yolo": MockDetector()}, "segment": {"sam": MockSegmenter()}}
        ctx = ExecutionContext(providers=providers)

        repr_str = repr(ctx)
        assert "ExecutionContext" in repr_str
        assert "device=" in repr_str
        assert "artifacts=" in repr_str
        assert "providers=" in repr_str


class TestArtifactStorage:
    """Test artifact storage and retrieval."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving artifacts."""
        ctx = ExecutionContext()
        artifact = MockArtifact(value=42, label="test")

        ctx.store("my_artifact", artifact)
        retrieved = ctx.retrieve("my_artifact")

        assert retrieved == artifact
        assert retrieved.value == 42
        assert retrieved.label == "test"

    def test_store_multiple_artifacts(self):
        """Test storing multiple artifacts."""
        ctx = ExecutionContext()

        art1 = MockArtifact(value=1, label="first")
        art2 = MockArtifact(value=2, label="second")
        art3 = MockArtifact(value=3, label="third")

        ctx.store("art1", art1)
        ctx.store("art2", art2)
        ctx.store("art3", art3)

        assert ctx.retrieve("art1") == art1
        assert ctx.retrieve("art2") == art2
        assert ctx.retrieve("art3") == art3

    def test_store_overwrites(self):
        """Test that storing with same name overwrites."""
        ctx = ExecutionContext()

        art1 = MockArtifact(value=1)
        art2 = MockArtifact(value=2)

        ctx.store("artifact", art1)
        ctx.store("artifact", art2)  # Overwrite

        retrieved = ctx.retrieve("artifact")
        assert retrieved == art2
        assert retrieved.value == 2

    def test_retrieve_missing_artifact(self):
        """Test retrieving non-existent artifact raises KeyError."""
        ctx = ExecutionContext()

        with pytest.raises(KeyError) as exc_info:
            ctx.retrieve("missing")

        assert "missing" in str(exc_info.value).lower()
        assert "not found" in str(exc_info.value).lower()

    def test_retrieve_missing_with_available_list(self):
        """Test error message includes available artifacts."""
        ctx = ExecutionContext()
        ctx.store("art1", MockArtifact(value=1))
        ctx.store("art2", MockArtifact(value=2))

        with pytest.raises(KeyError) as exc_info:
            ctx.retrieve("missing")

        error = str(exc_info.value).lower()
        assert "available" in error
        assert "art1" in str(exc_info.value)
        assert "art2" in str(exc_info.value)

    def test_has_artifact(self):
        """Test checking artifact existence."""
        ctx = ExecutionContext()

        assert not ctx.has("artifact")

        ctx.store("artifact", MockArtifact(value=42))

        assert ctx.has("artifact")
        assert not ctx.has("other")

    def test_store_invalid_artifact_fails_validation(self):
        """Test storing invalid artifact raises ValidationError."""
        ctx = ExecutionContext()
        invalid = InvalidArtifact(value=1)

        with pytest.raises(ValidationError) as exc_info:
            ctx.store("invalid", invalid)

        assert "invalid artifact" in str(exc_info.value).lower()
        assert "Always invalid" in str(exc_info.value)

    def test_store_empty_name_fails(self):
        """Test storing with empty name raises ValueError."""
        ctx = ExecutionContext()

        with pytest.raises(ValueError) as exc_info:
            ctx.store("", MockArtifact(value=42))

        assert "empty" in str(exc_info.value).lower()

    def test_store_non_artifact_fails(self):
        """Test storing non-Artifact raises ValueError."""
        ctx = ExecutionContext()

        with pytest.raises(ValueError) as exc_info:
            ctx.store("invalid", {"not": "artifact"})

        assert "artifact" in str(exc_info.value).lower()

    def test_clear_artifacts(self):
        """Test clearing artifacts."""
        ctx = ExecutionContext()

        ctx.store("art1", MockArtifact(value=1))
        ctx.store("art2", MockArtifact(value=2))
        assert ctx.has("art1")
        assert ctx.has("art2")

        ctx.clear_artifacts()

        assert not ctx.has("art1")
        assert not ctx.has("art2")


class TestProviderAccess:
    """Test provider registry access."""

    def test_get_provider(self):
        """Test getting provider by capability and name."""
        detector = MockDetector()
        providers = {"detect": {"detr": detector}}
        ctx = ExecutionContext(providers=providers)

        retrieved = ctx.get_provider("detect", "detr")
        assert retrieved is detector

    def test_get_provider_multiple_capabilities(self):
        """Test getting providers from multiple capabilities."""
        detector = MockDetector()
        segmenter = MockSegmenter()
        providers = {"detect": {"detr": detector}, "segment": {"sam": segmenter}}
        ctx = ExecutionContext(providers=providers)

        assert ctx.get_provider("detect", "detr") is detector
        assert ctx.get_provider("segment", "sam") is segmenter

    def test_get_provider_multiple_per_capability(self):
        """Test multiple providers per capability."""
        detr = MockDetector()
        yolo = MockDetector()
        providers = {"detect": {"detr": detr, "yolo": yolo}}
        ctx = ExecutionContext(providers=providers)

        assert ctx.get_provider("detect", "detr") is detr
        assert ctx.get_provider("detect", "yolo") is yolo

    def test_get_provider_missing_capability(self):
        """Test getting provider with missing capability raises KeyError."""
        ctx = ExecutionContext(providers={"detect": {}})

        with pytest.raises(KeyError) as exc_info:
            ctx.get_provider("segment", "sam")

        error = str(exc_info.value).lower()
        assert "segment" in error
        assert "not found" in error
        assert "detect" in str(exc_info.value)  # Shows available

    def test_get_provider_missing_name(self):
        """Test getting provider with missing name raises KeyError."""
        providers = {"detect": {"detr": MockDetector()}}
        ctx = ExecutionContext(providers=providers)

        with pytest.raises(KeyError) as exc_info:
            ctx.get_provider("detect", "yolo")

        error = str(exc_info.value).lower()
        assert "yolo" in error
        assert "not found" in error
        assert "detr" in str(exc_info.value)  # Shows available

    def test_get_provider_empty_registry(self):
        """Test error message when provider registry is empty."""
        ctx = ExecutionContext()

        with pytest.raises(KeyError) as exc_info:
            ctx.get_provider("detect", "detr")

        assert "none" in str(exc_info.value).lower()


class TestMetricsCollection:
    """Test metrics recording and retrieval."""

    def test_record_metric(self):
        """Test recording single metric."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "latency_ms", 45.2)

        metrics = ctx.get_metrics()
        assert "node1" in metrics
        assert metrics["node1"]["latency_ms"] == 45.2

    def test_record_multiple_metrics_per_node(self):
        """Test recording multiple metrics for same node."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "latency_ms", 45.2)
        ctx.record_metric("node1", "num_detections", 5)
        ctx.record_metric("node1", "memory_mb", 128.5)

        metrics = ctx.get_node_metrics("node1")
        assert metrics["latency_ms"] == 45.2
        assert metrics["num_detections"] == 5.0
        assert metrics["memory_mb"] == 128.5

    def test_record_metrics_multiple_nodes(self):
        """Test recording metrics for multiple nodes."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "latency_ms", 45.2)
        ctx.record_metric("node2", "latency_ms", 12.3)
        ctx.record_metric("node3", "latency_ms", 78.9)

        metrics = ctx.get_metrics()
        assert len(metrics) == 3
        assert metrics["node1"]["latency_ms"] == 45.2
        assert metrics["node2"]["latency_ms"] == 12.3
        assert metrics["node3"]["latency_ms"] == 78.9

    def test_record_metric_overwrites(self):
        """Test recording same metric twice overwrites."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "latency_ms", 45.2)
        ctx.record_metric("node1", "latency_ms", 50.0)  # Overwrite

        metrics = ctx.get_node_metrics("node1")
        assert metrics["latency_ms"] == 50.0

    def test_record_metric_converts_to_float(self):
        """Test metric values are converted to float."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "count", 5)  # int
        ctx.record_metric("node2", "value", 3.14)  # float
        ctx.record_metric("node3", "flag", True)  # bool

        metrics = ctx.get_metrics()
        assert metrics["node1"]["count"] == 5.0
        assert metrics["node2"]["value"] == 3.14
        assert metrics["node3"]["flag"] == 1.0

    def test_record_metric_invalid_value(self):
        """Test recording non-numeric metric raises ValueError."""
        ctx = ExecutionContext()

        with pytest.raises(ValueError) as exc_info:
            ctx.record_metric("node1", "name", "invalid")

        assert "numeric" in str(exc_info.value).lower()

    def test_get_node_metrics_missing_node(self):
        """Test getting metrics for node with no metrics returns empty dict."""
        ctx = ExecutionContext()

        metrics = ctx.get_node_metrics("missing_node")
        assert metrics == {}

    def test_clear_metrics(self):
        """Test clearing metrics."""
        ctx = ExecutionContext()

        ctx.record_metric("node1", "latency_ms", 45.2)
        ctx.record_metric("node2", "count", 5)
        assert len(ctx.get_metrics()) == 2

        ctx.clear_metrics()

        assert len(ctx.get_metrics()) == 0

    def test_clear_metrics_preserves_artifacts(self):
        """Test clearing metrics doesn't affect artifacts."""
        ctx = ExecutionContext()

        ctx.store("artifact", MockArtifact(value=42))
        ctx.record_metric("node1", "latency_ms", 45.2)

        ctx.clear_metrics()

        assert ctx.has("artifact")
        assert len(ctx.get_metrics()) == 0


class TestDeviceResolution:
    """Test device resolution logic."""

    def test_device_cpu_explicit(self):
        """Test explicit CPU device."""
        ctx = ExecutionContext(device="cpu")
        assert ctx.device == "cpu"

    def test_device_cpu_case_insensitive(self):
        """Test device specification is case-insensitive."""
        ctx = ExecutionContext(device="CPU")
        assert ctx.device == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_available(self, mock_cuda):
        """Test explicit CUDA when available."""
        ctx = ExecutionContext(device="cuda")
        assert ctx.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_cuda_unavailable(self, mock_cuda):
        """Test explicit CUDA when unavailable raises error."""
        with pytest.raises(ValueError) as exc_info:
            ExecutionContext(device="cuda")

        assert "cuda" in str(exc_info.value).lower()
        assert "not available" in str(exc_info.value).lower()

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_auto_with_cuda(self, mock_cuda):
        """Test auto device selection with CUDA available."""
        ctx = ExecutionContext(device="auto")
        assert ctx.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_auto_without_cuda(self, mock_cuda):
        """Test auto device selection without CUDA."""
        ctx = ExecutionContext(device="auto")
        assert ctx.device == "cpu"

    @patch("builtins.__import__", side_effect=ImportError)
    def test_device_auto_no_torch(self, mock_import):
        """Test auto device when PyTorch not installed defaults to CPU."""
        ctx = ExecutionContext(device="auto")
        assert ctx.device == "cpu"

    def test_device_invalid(self):
        """Test invalid device raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExecutionContext(device="gpu")

        error = str(exc_info.value).lower()
        assert "invalid" in error
        assert "gpu" in error
        assert "auto" in error
        assert "cuda" in error
        assert "cpu" in error


class TestCleanupAndUtilities:
    """Test cleanup and utility methods."""

    def test_clear_all(self):
        """Test complete clear of artifacts and metrics."""
        ctx = ExecutionContext()

        ctx.store("artifact", MockArtifact(value=42))
        ctx.record_metric("node1", "latency_ms", 45.2)

        ctx.clear()

        assert not ctx.has("artifact")
        assert len(ctx.get_metrics()) == 0

    def test_execution_time(self):
        """Test execution time tracking."""
        ctx = ExecutionContext()

        time.sleep(0.1)  # Sleep for 100ms

        elapsed = ctx.get_execution_time()
        assert elapsed >= 0.1
        assert elapsed < 0.2  # Should be close to 100ms

    def test_execution_time_increases(self):
        """Test execution time increases over time."""
        ctx = ExecutionContext()

        time1 = ctx.get_execution_time()
        time.sleep(0.05)
        time2 = ctx.get_execution_time()

        assert time2 > time1


class TestCachingBehavior:
    """Test artifact caching behavior."""

    def test_cache_enabled(self):
        """Test caching enabled stores artifacts."""
        ctx = ExecutionContext(cache_artifacts=True)

        artifact = MockArtifact(value=42)
        ctx.store("artifact", artifact)

        # Artifact should be retrievable
        assert ctx.has("artifact")
        retrieved = ctx.retrieve("artifact")
        assert retrieved == artifact

    def test_cache_disabled(self):
        """Test caching disabled still stores artifacts (for now)."""
        # Note: In current implementation, caching flag doesn't affect storage
        # This is intentional - the flag is for future optimization
        ctx = ExecutionContext(cache_artifacts=False)

        artifact = MockArtifact(value=42)
        ctx.store("artifact", artifact)

        # Artifact should still be retrievable
        assert ctx.has("artifact")


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_typical_graph_execution_flow(self):
        """Test typical graph execution workflow."""
        # Setup providers
        detector = MockDetector()
        segmenter = MockSegmenter()
        providers = {"detect": {"detr": detector}, "segment": {"sam": segmenter}}

        # Create context
        ctx = ExecutionContext(providers=providers, device="cpu")

        # Store input artifact
        input_artifact = MockArtifact(value=0, label="input")
        ctx.store("input_image", input_artifact)

        # Simulate detect node execution
        start = time.time()
        detector_result = MockArtifact(value=5, label="detections")
        ctx.store("detections", detector_result)
        ctx.record_metric("detect_node", "latency_ms", (time.time() - start) * 1000)
        ctx.record_metric("detect_node", "num_detections", 5)

        # Simulate segment node execution
        start = time.time()
        segment_result = MockArtifact(value=5, label="masks")
        ctx.store("masks", segment_result)
        ctx.record_metric("segment_node", "latency_ms", (time.time() - start) * 1000)

        # Verify results
        assert ctx.has("input_image")
        assert ctx.has("detections")
        assert ctx.has("masks")

        metrics = ctx.get_metrics()
        assert "detect_node" in metrics
        assert "segment_node" in metrics
        assert "latency_ms" in metrics["detect_node"]
        assert "num_detections" in metrics["detect_node"]

    def test_multiple_nodes_same_provider(self):
        """Test multiple nodes using same provider."""
        detector = MockDetector()
        providers = {"detect": {"detr": detector}}
        ctx = ExecutionContext(providers=providers)

        # Node 1 uses detector
        det1 = ctx.get_provider("detect", "detr")
        ctx.store("dets1", MockArtifact(value=3))
        ctx.record_metric("node1", "count", 3)

        # Node 2 uses same detector
        det2 = ctx.get_provider("detect", "detr")
        ctx.store("dets2", MockArtifact(value=5))
        ctx.record_metric("node2", "count", 5)

        # Should be same provider instance
        assert det1 is det2

        # Both artifacts stored
        assert ctx.retrieve("dets1").value == 3
        assert ctx.retrieve("dets2").value == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
