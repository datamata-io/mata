"""Comprehensive tests for the provider registry and protocol system.

Tests cover:
- ProviderRegistry: register, get, has, list, unregister, clear
- Capability protocols: Detector, Segmenter, Classifier, DepthEstimator
- Lazy loading via lazy=True parameter
- Multiple providers per capability
- ProviderConfig dataclass
- Thread safety
- Error handling (ProviderNotFoundError)
- VisionLanguageModel protocol
- verify_capability
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from mata.core.registry.protocols import (
    Classifier,
    DepthEstimator,
    Detector,
    Segmenter,
    VisionLanguageModel,
)
from mata.core.registry.providers import (
    ProviderConfig,
    ProviderError,
    ProviderNotFoundError,
    ProviderRegistry,
)

# ──────── mock providers ────────


class MockDetector:
    """Structurally matches Detector protocol (predict method)."""

    def predict(self, image, **kwargs):
        return {"detections": []}


class MockClassifier:
    """Structurally matches Classifier protocol (classify method)."""

    def classify(self, image, **kwargs):
        return {"classifications": []}


class MockSegmenter:
    """Structurally matches Segmenter protocol (segment method)."""

    def segment(self, image, **kwargs):
        return {"masks": []}


class MockDepthEstimator:
    """Structurally matches DepthEstimator protocol (estimate method)."""

    def estimate(self, image, **kwargs):
        return {"depth_map": None}


class MockVLM:
    """Structurally matches VisionLanguageModel protocol (query method)."""

    def query(self, image, prompt, **kwargs):
        return {"text": "a cat"}


class NotADetector:
    """Does NOT match any protocol."""

    def unrelated_method(self):
        pass


# ══════════════════════════════════════════════════════════════
# ProviderRegistry basics
# ══════════════════════════════════════════════════════════════


class TestProviderRegistryBasics:
    """Basic registration and lookup."""

    def setup_method(self):
        self.reg = ProviderRegistry()

    def test_register_and_get_eager(self):
        """Register eagerly (lazy=False) and retrieve provider."""
        det = MockDetector()
        self.reg.register("detr", Detector, lambda: det, lazy=False)
        result = self.reg.get(Detector, "detr")
        assert result is det

    def test_has_provider(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        assert self.reg.has(Detector, "detr")
        assert not self.reg.has(Detector, "yolo")

    def test_get_nonexistent_raises(self):
        with pytest.raises(ProviderNotFoundError):
            self.reg.get(Detector, "nonexistent")

    def test_register_multiple_capabilities(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("resnet", Classifier, lambda: MockClassifier(), lazy=False)
        assert self.reg.has(Detector, "detr")
        assert self.reg.has(Classifier, "resnet")

    def test_register_multiple_per_capability(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("yolo", Detector, lambda: MockDetector(), lazy=False)
        assert self.reg.has(Detector, "detr")
        assert self.reg.has(Detector, "yolo")

    def test_len(self):
        """__len__ returns total provider count."""
        assert len(self.reg) == 0
        self.reg.register("det1", Detector, lambda: MockDetector(), lazy=False)
        assert len(self.reg) == 1
        self.reg.register("det2", Detector, lambda: MockDetector(), lazy=False)
        assert len(self.reg) == 2


class TestProviderRegistryUnregister:
    """Unregistration tests."""

    def setup_method(self):
        self.reg = ProviderRegistry()

    def test_unregister(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.unregister(Detector, "detr")
        assert not self.reg.has(Detector, "detr")

    def test_unregister_nonexistent_raises(self):
        with pytest.raises((ProviderNotFoundError, KeyError)):
            self.reg.unregister(Detector, "nonexistent")

    def test_clear(self):
        """clear() removes all providers."""
        self.reg.register("d1", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("c1", Classifier, lambda: MockClassifier(), lazy=False)
        self.reg.clear()
        assert len(self.reg) == 0


class TestProviderRegistryListing:
    """Provider listing."""

    def setup_method(self):
        self.reg = ProviderRegistry()

    def test_list_providers_for_capability(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("yolo", Detector, lambda: MockDetector(), lazy=False)
        providers = self.reg.list_providers(Detector)
        assert "detr" in providers
        assert "yolo" in providers

    def test_list_all_providers(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("resnet", Classifier, lambda: MockClassifier(), lazy=False)
        all_providers = self.reg.list_providers()
        assert "detr" in all_providers
        assert "resnet" in all_providers

    def test_list_capabilities(self):
        self.reg.register("detr", Detector, lambda: MockDetector(), lazy=False)
        self.reg.register("resnet", Classifier, lambda: MockClassifier(), lazy=False)
        caps = self.reg.list_capabilities()
        # Capability names should reflect the protocol
        assert len(caps) >= 2


# ══════════════════════════════════════════════════════════════
# Lazy registration
# ══════════════════════════════════════════════════════════════


class TestLazyRegistration:
    """Factory-based lazy provider loading via lazy=True."""

    def setup_method(self):
        self.reg = ProviderRegistry()

    def test_lazy_register_and_get(self):
        factory = Mock(return_value=MockDetector())
        self.reg.register("lazy_det", Detector, factory, lazy=True)
        result = self.reg.get(Detector, "lazy_det")
        factory.assert_called_once()
        assert isinstance(result, MockDetector)

    def test_lazy_factory_called_once(self):
        """Provider factory cached after first call."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return MockDetector()

        self.reg.register("lazy_det", Detector, factory, lazy=True)
        self.reg.get(Detector, "lazy_det")
        self.reg.get(Detector, "lazy_det")
        assert call_count == 1

    def test_lazy_vs_eager(self):
        """lazy=False invokes factory immediately at registration."""
        factory = Mock(return_value=MockDetector())
        self.reg.register("eager_det", Detector, factory, lazy=False)
        factory.assert_called_once()


# ══════════════════════════════════════════════════════════════
# ProviderConfig
# ══════════════════════════════════════════════════════════════


class TestProviderConfig:
    """ProviderConfig dataclass tests."""

    def test_config_fields(self):
        def factory():
            return MockDetector()

        cfg = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=factory,
            lazy=True,
        )
        assert cfg.name == "detr"
        assert cfg.capability is Detector
        assert cfg.lazy is True
        assert cfg.instance is None

    def test_is_loaded_property(self):
        cfg = ProviderConfig(
            name="detr",
            capability=Detector,
            factory=lambda: MockDetector(),
            lazy=True,
        )
        assert not cfg.is_loaded

    def test_get_config(self):
        """Registry exposes config for registered providers."""
        reg = ProviderRegistry()
        reg.register("detr", Detector, lambda: MockDetector(), lazy=True)
        cfg = reg.get_config(Detector, "detr")
        assert cfg is not None
        assert cfg.name == "detr"


# ══════════════════════════════════════════════════════════════
# Protocol compliance
# ══════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """Verify mock providers match runtime-checkable protocols."""

    def test_detector_isinstance(self):
        assert isinstance(MockDetector(), Detector)

    def test_classifier_isinstance(self):
        assert isinstance(MockClassifier(), Classifier)

    def test_segmenter_isinstance(self):
        assert isinstance(MockSegmenter(), Segmenter)

    def test_depth_estimator_isinstance(self):
        assert isinstance(MockDepthEstimator(), DepthEstimator)

    def test_vlm_isinstance(self):
        assert isinstance(MockVLM(), VisionLanguageModel)

    def test_non_matching_class(self):
        assert not isinstance(NotADetector(), Detector)

    def test_verify_capability(self):
        """ProviderRegistry.verify_capability checks protocol."""
        reg = ProviderRegistry()
        assert reg.verify_capability(MockDetector(), Detector) is True
        assert reg.verify_capability(NotADetector(), Detector) is False


# ══════════════════════════════════════════════════════════════
# Thread safety
# ══════════════════════════════════════════════════════════════


class TestProviderThreadSafety:
    """Concurrent access to registry."""

    def test_concurrent_registration(self):
        reg = ProviderRegistry()
        errors = []

        def register_provider(idx):
            try:
                reg.register(f"det_{idx}", Detector, lambda: MockDetector(), lazy=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_provider, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(10):
            assert reg.has(Detector, f"det_{i}")

    def test_concurrent_get_lazy(self):
        """Multiple threads calling get on lazy provider should be safe."""
        reg = ProviderRegistry()
        call_count = 0
        lock = threading.Lock()

        def factory():
            nonlocal call_count
            with lock:
                call_count += 1
            return MockDetector()

        reg.register("shared", Detector, factory, lazy=True)
        results = []

        def get_provider():
            results.append(reg.get(Detector, "shared"))

        threads = [threading.Thread(target=get_provider) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        # All should get the same cached instance
        assert all(r is results[0] for r in results)


# ══════════════════════════════════════════════════════════════
# Error messages
# ══════════════════════════════════════════════════════════════


class TestProviderErrors:
    """Error handling and messages."""

    def test_not_found_error_message(self):
        reg = ProviderRegistry()
        try:
            reg.get(Detector, "missing")
            pytest.fail("Should have raised")
        except ProviderNotFoundError as e:
            assert "missing" in str(e).lower() or "Detector" in str(e)

    def test_provider_error_base(self):
        err = ProviderError("something went wrong")
        assert "something went wrong" in str(err)
