"""MATA - Model-Agnostic Task Architecture

A task-centric, model-agnostic framework for computer vision.

Example usage:
    >>> import mata

    >>> # One-shot detection
    >>> result = mata.run("detect", "image.jpg")
    >>> print(result.to_json(indent=2))

    >>> # Load adapter for repeated use
    >>> detector = mata.load("detect", "rtdetr")
    >>> result1 = detector.predict("image1.jpg")
    >>> result2 = detector.predict("image2.jpg")

    >>> # Segmentation with visualization
    >>> segmenter = mata.load("segment", "facebook/mask2former-swin-tiny-coco-instance")
    >>> result = segmenter.predict("image.jpg")
    >>> vis = mata.visualize_segmentation(result, "image.jpg")
    >>> vis.show()

    >>> # List available models
    >>> print(mata.list_models("detect"))
"""

__version__ = "1.9.2b1"

from .api import get_model_info, infer, list_models, load, register_model, run, track, val, verbose
from .core import (
    ClassifyResult,
    DepthResult,
    # Types
    Detection,
    DetectResult,
    Entity,
    Instance,
    InvalidInputError,
    # Config
    MATAConfig,
    # Exceptions
    MATAError,
    ModelLoadError,
    OCRResult,
    SegmentMask,
    SegmentResult,
    TaskNotSupportedError,
    TextRegion,
    Track,
    TrackResult,
    VisionResult,
    get_config,
    set_config,
)
from .eval import ClassifyMetrics, DepthMetrics, DetMetrics, OCRMetrics, SegmentMetrics

# Visualization (lazy import to avoid hard dependency)
try:
    from .visualization import create_panoptic_visualization, visualize_segmentation

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

    def visualize_segmentation(*args, **kwargs):
        raise ImportError("Visualization requires Pillow. Install with: pip install Pillow")

    def create_panoptic_visualization(*args, **kwargs):
        raise ImportError("Visualization requires Pillow. Install with: pip install Pillow")


__all__ = [
    # API
    "load",
    "run",
    "track",
    "infer",
    "val",
    "list_models",
    "get_model_info",
    "register_model",
    "verbose",
    # Metrics
    "DetMetrics",
    "SegmentMetrics",
    "ClassifyMetrics",
    "DepthMetrics",
    "OCRMetrics",
    # Types
    "Detection",
    "DetectResult",
    "SegmentMask",
    "SegmentResult",
    "ClassifyResult",
    "DepthResult",
    "OCRResult",
    "TextRegion",
    "Track",
    "TrackResult",
    "Entity",
    "Instance",
    "VisionResult",
    # Exceptions
    "MATAError",
    "TaskNotSupportedError",
    "InvalidInputError",
    "ModelLoadError",
    # Config
    "MATAConfig",
    "get_config",
    "set_config",
    # Visualization
    "visualize_segmentation",
    "create_panoptic_visualization",
    # Version
    "__version__",
]
