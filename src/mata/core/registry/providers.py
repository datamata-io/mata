"""Provider registry with capability-based lookup and lazy loading.

This module implements the ProviderRegistry class, which manages task providers
(detectors, classifiers, segmenters, etc.) using capability-based registration
and lookup. Providers are registered against Protocol types and can be lazily
instantiated on first access.

Thread Safety:
    All public methods use a threading.Lock for thread-safe access to the
    internal registry. Multiple threads can safely register, lookup, and
    unregister providers concurrently.

Example:
    ```python
    from mata.core.registry.providers import ProviderRegistry
    from mata.core.registry.protocols import Detector, Classifier

    registry = ProviderRegistry()

    # Register with lazy loading (factory called on first get())
    registry.register(
        name="detr",
        capability=Detector,
        adapter_factory=lambda: load_detr_model(),
        lazy=True
    )

    # Register with eager loading (instance created immediately)
    registry.register(
        name="resnet",
        capability=Classifier,
        adapter_factory=lambda: load_resnet_model(),
        lazy=False
    )

    # Lookup
    detector = registry.get(Detector, "detr")  # Factory called here
    classifier = registry.get(Classifier, "resnet")  # Already loaded

    # List providers
    registry.list_providers()  # ["detr", "resnet"]
    registry.list_providers(Detector)  # ["detr"]

    # Unregister
    registry.unregister(Detector, "detr")
    ```
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mata.core.exceptions import MATAError


class ProviderNotFoundError(MATAError):
    """Raised when a requested provider is not found in the registry.

    This occurs when:
    - Provider name does not exist for the requested capability
    - Capability type has no registered providers
    """

    def __init__(
        self,
        capability: type[Protocol],
        name: str,
        available: list[str] | None = None,
    ) -> None:
        self.capability = capability
        self.name = name
        self.available = available or []

        cap_name = getattr(capability, "__name__", str(capability))
        avail_str = ", ".join(self.available) if self.available else "none"
        msg = f"Provider '{name}' not found for capability '{cap_name}'. " f"Available providers: {avail_str}"
        super().__init__(msg)


class ProviderError(MATAError):
    """Raised when a provider operation fails.

    Covers factory errors, capability verification failures, and
    duplicate registration attempts.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class ProviderConfig:
    """Configuration for a registered provider.

    Stores the provider's factory function, capability type, and optionally
    the lazily-loaded instance.

    Attributes:
        name: Unique provider name within its capability.
        capability: The Protocol type this provider implements.
        factory: Callable that creates the provider instance.
        lazy: If True, instance is created on first get(). If False,
              instance is created at registration time.
        instance: The loaded provider instance, or None if not yet loaded.
    """

    name: str
    capability: type[Protocol]
    factory: Callable[..., Any]
    lazy: bool
    instance: Any | None = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        """Check if the provider instance has been created."""
        return self.instance is not None


class ProviderRegistry:
    """Registry for capability-based provider lookup with lazy loading.

    The ProviderRegistry manages providers (adapters, models, wrappers) organized
    by their capability protocols. Each provider is registered under a specific
    capability (e.g., Detector, Classifier) and a unique name.

    Features:
    - **Capability-based registration**: Providers are indexed by Protocol type
    - **Lazy loading**: Factories are called only on first access (configurable)
    - **Protocol verification**: Optional runtime check that providers actually
      implement the declared capability
    - **Multiple providers per capability**: Multiple detectors, classifiers, etc.
    - **Thread-safe access**: All operations are protected by a lock
    - **Clean error messages**: ProviderNotFoundError with available alternatives

    Example:
        ```python
        registry = ProviderRegistry()

        # Lazy registration
        registry.register("yolo", Detector, lambda: YOLOAdapter(), lazy=True)

        # Eager registration
        registry.register("resnet", Classifier, lambda: ResNetAdapter(), lazy=False)

        # Lookup (triggers factory for lazy providers)
        detector = registry.get(Detector, "yolo")
        assert isinstance(detector, Detector)  # Runtime check

        # List available
        registry.list_providers(Detector)  # ["yolo"]
        ```
    """

    def __init__(self) -> None:
        """Initialize empty provider registry."""
        # {capability_name: {provider_name: ProviderConfig}}
        self._providers: dict[str, dict[str, ProviderConfig]] = defaultdict(dict)
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        capability: type[Protocol],
        adapter_factory: Callable[..., Any],
        lazy: bool = True,
    ) -> None:
        """Register a provider with a capability.

        If ``lazy=True`` (default), the factory is stored and called on the
        first ``get()`` call. If ``lazy=False``, the factory is called
        immediately and the instance is stored.

        Args:
            name: Unique name for this provider within the capability.
            capability: Protocol type the provider implements.
            adapter_factory: Callable that returns a provider instance.
            lazy: If True, defer instantiation until first get().

        Raises:
            ProviderError: If a provider with the same name is already
                registered for this capability.
            ProviderError: If ``lazy=False`` and the factory raises an error.

        Example:
            ```python
            registry.register(
                "detr",
                Detector,
                lambda: HuggingFaceDetectAdapter("facebook/detr-resnet-50"),
                lazy=True
            )
            ```
        """
        cap_key = self._capability_key(capability)

        with self._lock:
            if name in self._providers[cap_key]:
                cap_name = getattr(capability, "__name__", str(capability))
                raise ProviderError(
                    f"Provider '{name}' already registered for capability "
                    f"'{cap_name}'. Use unregister() first to replace it."
                )

            config = ProviderConfig(
                name=name,
                capability=capability,
                factory=adapter_factory,
                lazy=lazy,
            )

            if not lazy:
                # Eager: instantiate immediately
                try:
                    config.instance = adapter_factory()
                except Exception as e:
                    cap_name = getattr(capability, "__name__", str(capability))
                    raise ProviderError(
                        f"Failed to create provider '{name}' for capability " f"'{cap_name}': {e}"
                    ) from e

            self._providers[cap_key][name] = config

    def get(self, capability: type[Protocol], name: str) -> Any:
        """Get a provider by capability and name.

        For lazy providers, the factory is called on the first access and
        the resulting instance is cached for subsequent calls.

        Args:
            capability: Protocol type to look up.
            name: Name of the provider.

        Returns:
            The provider instance.

        Raises:
            ProviderNotFoundError: If no provider is registered with the
                given name for the specified capability.
            ProviderError: If lazy instantiation fails.

        Example:
            ```python
            detector = registry.get(Detector, "detr")
            detections = detector.predict(image)
            ```
        """
        cap_key = self._capability_key(capability)

        with self._lock:
            providers = self._providers.get(cap_key)
            if providers is None or name not in providers:
                available = list(providers.keys()) if providers else []
                raise ProviderNotFoundError(capability, name, available)

            config = providers[name]

            # Lazy load if needed
            if config.instance is None:
                try:
                    config.instance = config.factory()
                except Exception as e:
                    cap_name = getattr(capability, "__name__", str(capability))
                    raise ProviderError(
                        f"Failed to lazy-load provider '{name}' for capability " f"'{cap_name}': {e}"
                    ) from e

            return config.instance

    def has(self, capability: type[Protocol], name: str) -> bool:
        """Check if a provider exists for a capability.

        This does NOT trigger lazy loading — it only checks registration.

        Args:
            capability: Protocol type to check.
            name: Name of the provider.

        Returns:
            True if the provider is registered, False otherwise.

        Example:
            ```python
            if registry.has(Detector, "detr"):
                detector = registry.get(Detector, "detr")
            ```
        """
        cap_key = self._capability_key(capability)

        with self._lock:
            return name in self._providers.get(cap_key, {})

    def list_providers(self, capability: type[Protocol] | None = None) -> list[str]:
        """List registered provider names.

        Args:
            capability: If provided, list only providers for this capability.
                       If None, list all providers across all capabilities.

        Returns:
            Sorted list of provider names.

        Example:
            ```python
            # All providers
            all_names = registry.list_providers()

            # Only detectors
            detector_names = registry.list_providers(Detector)
            ```
        """
        with self._lock:
            if capability is not None:
                cap_key = self._capability_key(capability)
                providers = self._providers.get(cap_key, {})
                return sorted(providers.keys())

            # All providers across all capabilities
            all_names: set[str] = set()
            for providers in self._providers.values():
                all_names.update(providers.keys())
            return sorted(all_names)

    def list_capabilities(self) -> list[str]:
        """List all registered capability names.

        Returns:
            Sorted list of capability key strings.

        Example:
            ```python
            caps = registry.list_capabilities()
            # ["Classifier", "Detector"]
            ```
        """
        with self._lock:
            return sorted(cap_key for cap_key, providers in self._providers.items() if providers)  # skip empty

    def unregister(self, capability: type[Protocol], name: str) -> None:
        """Remove a provider from the registry.

        Args:
            capability: Protocol type of the provider.
            name: Name of the provider to remove.

        Raises:
            ProviderNotFoundError: If the provider is not registered.

        Example:
            ```python
            registry.unregister(Detector, "detr")
            assert not registry.has(Detector, "detr")
            ```
        """
        cap_key = self._capability_key(capability)

        with self._lock:
            providers = self._providers.get(cap_key)
            if providers is None or name not in providers:
                available = list(providers.keys()) if providers else []
                raise ProviderNotFoundError(capability, name, available)
            del providers[name]

    def verify_capability(self, provider: Any, capability: type[Protocol]) -> bool:
        """Verify that a provider implements a capability protocol.

        Uses ``isinstance()`` with runtime-checkable protocols to verify
        structural compatibility.

        Args:
            provider: The provider instance to check.
            capability: The Protocol type to verify against.

        Returns:
            True if the provider is structurally compatible with the
            protocol, False otherwise.

        Example:
            ```python
            class MyDetector:
                def predict(self, image, **kwargs):
                    return Detections(...)

            detector = MyDetector()
            assert registry.verify_capability(detector, Detector)
            ```
        """
        try:
            return isinstance(provider, capability)
        except TypeError:
            # Protocol not runtime-checkable or other type error
            return False

    def get_config(self, capability: type[Protocol], name: str) -> ProviderConfig | None:
        """Get the configuration for a registered provider.

        This is useful for inspecting registration details without
        triggering lazy loading.

        Args:
            capability: Protocol type of the provider.
            name: Name of the provider.

        Returns:
            ProviderConfig if found, None otherwise.
        """
        cap_key = self._capability_key(capability)

        with self._lock:
            providers = self._providers.get(cap_key)
            if providers is None:
                return None
            return providers.get(name)

    def clear(self) -> None:
        """Remove all registered providers.

        This is primarily useful for testing and cleanup.
        """
        with self._lock:
            self._providers.clear()

    def __len__(self) -> int:
        """Return total number of registered providers across all capabilities."""
        with self._lock:
            return sum(len(p) for p in self._providers.values())

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        with self._lock:
            entries = []
            for cap_key, providers in sorted(self._providers.items()):
                names = sorted(providers.keys())
                if names:
                    entries.append(f"{cap_key}: [{', '.join(names)}]")
            content = ", ".join(entries) if entries else "empty"
            return f"ProviderRegistry({content})"

    @staticmethod
    def _capability_key(capability: type[Protocol]) -> str:
        """Get a stable string key for a capability type.

        Uses the class name for readable keys in the internal dict.

        Args:
            capability: Protocol type.

        Returns:
            String key derived from the type name.
        """
        return getattr(capability, "__name__", str(capability))
