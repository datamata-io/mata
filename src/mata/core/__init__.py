"""Core module initialization."""

from .config import MATAConfig, get_config, set_config
from .exceptions import (
    ConfigurationError,
    InvalidInputError,
    MATAError,
    MATARuntimeError,
    ModelLoadError,
    RuntimeError,
    TaskNotSupportedError,
)
from .types import (
    BarcodeRegion,
    BarcodeResult,
    BBox,
    ClassifyResult,
    DepthResult,
    Detection,
    DetectResult,
    EmbedResult,
    Entity,
    Instance,
    OCRResult,
    SegmentMask,
    SegmentResult,
    TextRegion,
    Track,
    TrackResult,
    VisionResult,
)

__all__ = [
    # Config
    "MATAConfig",
    "get_config",
    "set_config",
    # Exceptions
    "MATAError",
    "TaskNotSupportedError",
    "MATARuntimeError",
    "RuntimeError",  # Deprecated alias, use MATARuntimeError
    "InvalidInputError",
    "ModelLoadError",
    "ConfigurationError",
    # Types
    "BBox",
    "Instance",
    "Entity",
    "VisionResult",
    "DepthResult",
    "Detection",
    "DetectResult",
    "SegmentMask",
    "SegmentResult",
    "ClassifyResult",
    "OCRResult",
    "TextRegion",
    "BarcodeRegion",
    "BarcodeResult",
    "Track",
    "TrackResult",
    "EmbedResult",
]
