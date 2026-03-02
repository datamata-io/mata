"""Provider registry and capability protocols for MATA graph system.

This module provides the capability-based provider system for graph nodes,
including runtime-checkable protocols for all task types and a provider registry
for capability-based lookup.
"""

from __future__ import annotations

from mata.core.registry.protocols import (
    Classifier,
    DepthEstimator,
    Detector,
    Embedder,
    PoseEstimator,
    Segmenter,
    Tracker,
    VisionLanguageModel,
)
from mata.core.registry.providers import (
    ProviderConfig,
    ProviderError,
    ProviderNotFoundError,
    ProviderRegistry,
)

__all__ = [
    "Detector",
    "Segmenter",
    "Classifier",
    "PoseEstimator",
    "DepthEstimator",
    "Embedder",
    "Tracker",
    "VisionLanguageModel",
    "ProviderRegistry",
    "ProviderConfig",
    "ProviderNotFoundError",
    "ProviderError",
]
