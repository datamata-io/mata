"""TensorRT adapter for GPU-optimized inference.

Supports TensorRT engines for maximum GPU performance.
Requires NVIDIA GPU and TensorRT installation.
"""

from __future__ import annotations

from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


class TensorRTDetectAdapter:
    """TensorRT detection adapter (placeholder).

    TensorRT support will be added in a future release.
    """

    def __init__(self, engine_path: str, device: str = "cuda", threshold: float = 0.3, **kwargs: Any) -> None:
        """Initialize TensorRT adapter.

        Args:
            engine_path: Path to TensorRT engine file
            device: Device (must be "cuda")
            threshold: Detection confidence threshold
            **kwargs: Additional arguments

        Raises:
            NotImplementedError: Always (not yet implemented)
        """
        raise NotImplementedError(
            "TensorRT adapter is not yet implemented. "
            "This feature will be available in a future release. "
            "For GPU-accelerated inference, please use:\n"
            "  - HuggingFace models with device='cuda'\n"
            "  - ONNX models with GPU execution provider\n\n"
            "Example:\n"
            "  detector = mata.load('detect', 'PekingU/rtdetr_v2_r18vd', device='cuda')"
        )
