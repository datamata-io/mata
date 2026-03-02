"""Tests for ProviderRegistry with capability-based lookup and lazy loading.

Tests cover:
- Registration (lazy and eager)
- Lookup (get, has)
- Listing (providers, capabilities)
- Unregistration
- Capability verification
- Multiple providers per capability
- Error handling (not found, duplicate, factory failure)
- Thread safety
- Edge cases (empty registry, clear)
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import Mock

import pytest

from mata.core.registry.protocols import (
    Classifier,
    DepthEstimator,
    Detector,
    Segmenter,
)
from mata.core.registry.providers import (
    ProviderConfig,
    ProviderError,
    ProviderNotFoundError,
    ProviderRegistry,
)

# ---------------------------------------------------------------------------
# Test helpers / mock providers
# ---------------------------------------------------------------------------


class MockDetector:
    """Mock that structurally matches the Detector protocol."""

    def predict(self, image, **kwargs):
        return {"detections": []}


class MockClassifier:
    """Mock that structurally matches the Classifier protocol."""

    def classify(self, image, **kwargs):
        return {"class": "cat", "score": 0.95}


class MockSegmenter:
    """Mock that structurally matches the Segmenter protocol."""

    def segment(self, image, **kwargs):
        return {"masks": []}


class MockDepthEstimator:
    """Mock that structurally matches the DepthEstimator protocol."""

    def estimate(self, image, **kwargs):
        return {"depth_map": None}


class NotADetector:
    """Intentionally does NOT implement Detector protocol."""

    def do_something(self):
        return "nope"


# ---------------------------------------------------------------------------
# ProviderConfig tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_basic_creation(self):
        config = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=MockDetector,
            lazy=True,
        )
        assert config.name == "detr"
        assert config.capability is Detector
        assert config.lazy is True
        assert config.instance is None

    def test_is_loaded_false_initially(self):
        config = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=MockDetector,
            lazy=True,
        )
        assert config.is_loaded is False

    def test_is_loaded_true_with_instance(self):
        instance = MockDetector()
        config = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=MockDetector,
            lazy=True,
            instance=instance,
        )
        assert config.is_loaded is True

    def test_repr_does_not_include_instance(self):
        """instance has repr=False, should not appear in repr."""
        config = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=MockDetector,
            lazy=True,
            instance=MockDetector(),
        )
        r = repr(config)
        assert "instance" not in r
        assert "detr" in r


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for provider registration."""

    def test_register_lazy(self):
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())

        registry.register("detr", Detector, factory, lazy=True)

        assert registry.has(Detector, "detr")
        # Factory not called yet
        factory.assert_not_called()

    def test_register_eager(self):
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())

        registry.register("detr", Detector, factory, lazy=False)

        assert registry.has(Detector, "detr")
        # Factory called immediately
        factory.assert_called_once()

    def test_register_default_lazy_true(self):
        """Default lazy=True when not specified."""
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())

        registry.register("detr", Detector, factory)

        factory.assert_not_called()
        assert registry.has(Detector, "detr")

    def test_register_duplicate_raises(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)

        with pytest.raises(ProviderError, match="already registered"):
            registry.register("detr", Detector, MockDetector, lazy=True)

    def test_register_same_name_different_capability(self):
        """Same name under different capabilities is allowed."""
        registry = ProviderRegistry()
        registry.register("model_a", Detector, MockDetector, lazy=True)
        registry.register("model_a", Classifier, MockClassifier, lazy=True)

        assert registry.has(Detector, "model_a")
        assert registry.has(Classifier, "model_a")

    def test_register_eager_factory_failure(self):
        registry = ProviderRegistry()

        def bad_factory():
            raise ValueError("Model file not found")

        with pytest.raises(ProviderError, match="Failed to create provider"):
            registry.register("broken", Detector, bad_factory, lazy=False)

        # Should NOT be registered after failure
        assert not registry.has(Detector, "broken")

    def test_register_multiple_providers_per_capability(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)
        registry.register("rtdetr", Detector, MockDetector, lazy=True)

        assert len(registry.list_providers(Detector)) == 3


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


class TestLookup:
    """Tests for provider lookup (get)."""

    def test_get_lazy_triggers_factory(self):
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())

        registry.register("detr", Detector, factory, lazy=True)
        result = registry.get(Detector, "detr")

        factory.assert_called_once()
        assert isinstance(result, MockDetector)

    def test_get_lazy_caches_instance(self):
        """Second get() should return same instance without calling factory again."""
        registry = ProviderRegistry()
        instance = MockDetector()
        factory = Mock(return_value=instance)

        registry.register("detr", Detector, factory, lazy=True)

        result1 = registry.get(Detector, "detr")
        result2 = registry.get(Detector, "detr")

        factory.assert_called_once()
        assert result1 is result2
        assert result1 is instance

    def test_get_eager_returns_instance(self):
        registry = ProviderRegistry()
        instance = MockDetector()
        factory = Mock(return_value=instance)

        registry.register("detr", Detector, factory, lazy=False)
        result = registry.get(Detector, "detr")

        assert result is instance

    def test_get_not_found_raises(self):
        registry = ProviderRegistry()

        with pytest.raises(ProviderNotFoundError, match="not found"):
            registry.get(Detector, "nonexistent")

    def test_get_not_found_shows_available(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)

        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.get(Detector, "unknown")

        assert "detr" in str(exc_info.value) or "yolo" in str(exc_info.value)
        assert exc_info.value.available  # available list is populated

    def test_get_wrong_capability_raises(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)

        with pytest.raises(ProviderNotFoundError):
            registry.get(Classifier, "detr")

    def test_get_lazy_factory_failure(self):
        registry = ProviderRegistry()

        def bad_factory():
            raise RuntimeError("GPU out of memory")

        registry.register("broken", Detector, bad_factory, lazy=True)

        with pytest.raises(ProviderError, match="Failed to lazy-load"):
            registry.get(Detector, "broken")


# ---------------------------------------------------------------------------
# has() tests
# ---------------------------------------------------------------------------


class TestHas:
    """Tests for has() method."""

    def test_has_registered(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        assert registry.has(Detector, "detr") is True

    def test_has_not_registered(self):
        registry = ProviderRegistry()
        assert registry.has(Detector, "detr") is False

    def test_has_wrong_capability(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        assert registry.has(Classifier, "detr") is False

    def test_has_does_not_trigger_lazy_load(self):
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())
        registry.register("detr", Detector, factory, lazy=True)

        registry.has(Detector, "detr")
        factory.assert_not_called()


# ---------------------------------------------------------------------------
# list_providers tests
# ---------------------------------------------------------------------------


class TestListProviders:
    """Tests for list_providers() method."""

    def test_list_all_empty(self):
        registry = ProviderRegistry()
        assert registry.list_providers() == []

    def test_list_all_providers(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)

        result = registry.list_providers()
        assert sorted(result) == ["detr", "resnet"]

    def test_list_by_capability(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)

        detectors = registry.list_providers(Detector)
        classifiers = registry.list_providers(Classifier)

        assert detectors == ["detr", "yolo"]
        assert classifiers == ["resnet"]

    def test_list_by_capability_empty(self):
        registry = ProviderRegistry()
        assert registry.list_providers(Detector) == []

    def test_list_returns_sorted(self):
        registry = ProviderRegistry()
        registry.register("zulu", Detector, MockDetector, lazy=True)
        registry.register("alpha", Detector, MockDetector, lazy=True)
        registry.register("mike", Detector, MockDetector, lazy=True)

        assert registry.list_providers(Detector) == ["alpha", "mike", "zulu"]


# ---------------------------------------------------------------------------
# list_capabilities tests
# ---------------------------------------------------------------------------


class TestListCapabilities:
    """Tests for list_capabilities() method."""

    def test_list_capabilities_empty(self):
        registry = ProviderRegistry()
        assert registry.list_capabilities() == []

    def test_list_capabilities(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)

        caps = registry.list_capabilities()
        assert "Detector" in caps
        assert "Classifier" in caps

    def test_list_capabilities_after_unregister_all(self):
        """Capability with no remaining providers should not appear."""
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.unregister(Detector, "detr")

        assert registry.list_capabilities() == []


# ---------------------------------------------------------------------------
# unregister tests
# ---------------------------------------------------------------------------


class TestUnregister:
    """Tests for unregister() method."""

    def test_unregister_existing(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)

        registry.unregister(Detector, "detr")
        assert not registry.has(Detector, "detr")

    def test_unregister_not_found_raises(self):
        registry = ProviderRegistry()

        with pytest.raises(ProviderNotFoundError):
            registry.unregister(Detector, "nonexistent")

    def test_unregister_allows_re_register(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.unregister(Detector, "detr")

        # Should succeed now
        registry.register("detr", Detector, MockDetector, lazy=True)
        assert registry.has(Detector, "detr")

    def test_unregister_one_does_not_affect_others(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)

        registry.unregister(Detector, "detr")

        assert not registry.has(Detector, "detr")
        assert registry.has(Detector, "yolo")


# ---------------------------------------------------------------------------
# verify_capability tests
# ---------------------------------------------------------------------------


class TestVerifyCapability:
    """Tests for verify_capability() method."""

    def test_verify_valid_detector(self):
        registry = ProviderRegistry()
        detector = MockDetector()
        assert registry.verify_capability(detector, Detector) is True

    def test_verify_valid_classifier(self):
        registry = ProviderRegistry()
        classifier = MockClassifier()
        assert registry.verify_capability(classifier, Classifier) is True

    def test_verify_invalid_provider(self):
        registry = ProviderRegistry()
        not_detector = NotADetector()
        assert registry.verify_capability(not_detector, Detector) is False

    def test_verify_with_non_protocol_type(self):
        """Should return False gracefully for non-protocol types."""
        registry = ProviderRegistry()
        assert registry.verify_capability("string", int) is False


# ---------------------------------------------------------------------------
# get_config tests
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for get_config() method."""

    def test_get_config_existing(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)

        config = registry.get_config(Detector, "detr")
        assert config is not None
        assert config.name == "detr"
        assert config.lazy is True
        assert config.is_loaded is False

    def test_get_config_not_found(self):
        registry = ProviderRegistry()
        assert registry.get_config(Detector, "missing") is None

    def test_get_config_does_not_trigger_lazy_load(self):
        registry = ProviderRegistry()
        factory = Mock(return_value=MockDetector())
        registry.register("detr", Detector, factory, lazy=True)

        config = registry.get_config(Detector, "detr")
        factory.assert_not_called()
        assert config.is_loaded is False


# ---------------------------------------------------------------------------
# clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for clear() method."""

    def test_clear_empties_registry(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)

        registry.clear()

        assert len(registry) == 0
        assert registry.list_providers() == []

    def test_clear_idempotent(self):
        registry = ProviderRegistry()
        registry.clear()
        registry.clear()
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# __len__ and __repr__ tests
# ---------------------------------------------------------------------------


class TestDunder:
    """Tests for __len__ and __repr__."""

    def test_len_empty(self):
        registry = ProviderRegistry()
        assert len(registry) == 0

    def test_len_with_providers(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)
        assert len(registry) == 3

    def test_repr_empty(self):
        registry = ProviderRegistry()
        assert "empty" in repr(registry)

    def test_repr_with_providers(self):
        registry = ProviderRegistry()
        registry.register("detr", Detector, MockDetector, lazy=True)
        r = repr(registry)
        assert "ProviderRegistry" in r
        assert "Detector" in r
        assert "detr" in r


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent access to ProviderRegistry."""

    def test_concurrent_registration(self):
        """Multiple threads registering different providers simultaneously."""
        registry = ProviderRegistry()
        errors: list[Exception] = []

        def register_provider(name: str, capability):
            try:
                registry.register(name, capability, MockDetector, lazy=True)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            t = threading.Thread(
                target=register_provider,
                args=(f"provider_{i}", Detector),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_providers(Detector)) == 20

    def test_concurrent_get(self):
        """Multiple threads calling get() on the same lazy provider."""
        call_count = 0
        lock = threading.Lock()

        def counting_factory():
            nonlocal call_count
            with lock:
                call_count += 1
            time.sleep(0.01)  # Simulate slow load
            return MockDetector()

        registry = ProviderRegistry()
        registry.register("detr", Detector, counting_factory, lazy=True)

        results: list[Any] = []
        result_lock = threading.Lock()

        def get_provider():
            result = registry.get(Detector, "detr")
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=get_provider) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get a result
        assert len(results) == 10
        # All results should be MockDetector instances
        for r in results:
            assert isinstance(r, MockDetector)

    def test_concurrent_register_and_get(self):
        """Mixed register/get operations from multiple threads."""
        registry = ProviderRegistry()
        errors: list[Exception] = []

        # Pre-register some providers
        for i in range(5):
            registry.register(f"pre_{i}", Detector, MockDetector, lazy=True)

        def register_task(idx):
            try:
                registry.register(f"new_{idx}", Detector, MockDetector, lazy=True)
            except Exception as e:
                errors.append(e)

        def get_task(idx):
            try:
                registry.get(Detector, f"pre_{idx % 5}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=register_task, args=(i,)))
            threads.append(threading.Thread(target=get_task, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# ProviderNotFoundError tests
# ---------------------------------------------------------------------------


class TestProviderNotFoundError:
    """Tests for ProviderNotFoundError exception."""

    def test_attributes(self):
        err = ProviderNotFoundError(Detector, "missing", ["detr", "yolo"])
        assert err.capability is Detector
        assert err.name == "missing"
        assert err.available == ["detr", "yolo"]

    def test_message_includes_name(self):
        err = ProviderNotFoundError(Detector, "missing", [])
        assert "missing" in str(err)
        assert "Detector" in str(err)

    def test_message_includes_available(self):
        err = ProviderNotFoundError(Detector, "missing", ["detr", "yolo"])
        msg = str(err)
        assert "detr" in msg
        assert "yolo" in msg

    def test_empty_available(self):
        err = ProviderNotFoundError(Detector, "missing")
        assert "none" in str(err)


# ---------------------------------------------------------------------------
# Integration-like tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration-style tests simulating real usage patterns."""

    def test_full_workflow(self):
        """Register → get → use → unregister lifecycle."""
        registry = ProviderRegistry()

        # Register
        registry.register("detr", Detector, MockDetector, lazy=True)
        assert len(registry) == 1

        # Get and use
        detector = registry.get(Detector, "detr")
        result = detector.predict("test.jpg")
        assert "detections" in result

        # Unregister
        registry.unregister(Detector, "detr")
        assert len(registry) == 0

    def test_multi_capability_registry(self):
        """Registry with multiple capabilities and providers."""
        registry = ProviderRegistry()

        registry.register("detr", Detector, MockDetector, lazy=True)
        registry.register("yolo", Detector, MockDetector, lazy=True)
        registry.register("resnet", Classifier, MockClassifier, lazy=True)
        registry.register("sam", Segmenter, MockSegmenter, lazy=True)
        registry.register("dpt", DepthEstimator, MockDepthEstimator, lazy=True)

        assert len(registry) == 5
        assert len(registry.list_providers(Detector)) == 2
        assert len(registry.list_providers(Classifier)) == 1
        assert len(registry.list_capabilities()) == 4

    def test_lazy_vs_eager_loading(self):
        """Verify lazy providers are not loaded until accessed."""
        registry = ProviderRegistry()
        lazy_factory = Mock(return_value=MockDetector())
        eager_factory = Mock(return_value=MockClassifier())

        registry.register("lazy_det", Detector, lazy_factory, lazy=True)
        registry.register("eager_cls", Classifier, eager_factory, lazy=False)

        # Eager already loaded
        eager_factory.assert_called_once()
        lazy_factory.assert_not_called()

        # Access lazy
        registry.get(Detector, "lazy_det")
        lazy_factory.assert_called_once()

    def test_replace_provider(self):
        """Unregister then re-register with different factory."""
        registry = ProviderRegistry()

        factory_v1 = Mock(return_value=MockDetector())
        factory_v2 = Mock(return_value=MockDetector())

        registry.register("detr", Detector, factory_v1, lazy=True)
        registry.unregister(Detector, "detr")
        registry.register("detr", Detector, factory_v2, lazy=True)

        registry.get(Detector, "detr")
        factory_v1.assert_not_called()
        factory_v2.assert_called_once()

    def test_verify_before_use(self):
        """Verify capability before using provider."""
        registry = ProviderRegistry()
        valid = MockDetector()
        invalid = NotADetector()

        assert registry.verify_capability(valid, Detector)
        assert not registry.verify_capability(invalid, Detector)
