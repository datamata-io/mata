"""PyTorch-specific base adapter for MATA framework.

This module provides the PyTorchBaseAdapter class that extends BaseAdapter
with PyTorch-specific functionality. Used by HuggingFace, PyTorch checkpoint,
and TorchScript adapters.
"""

from __future__ import annotations

from typing import Any

from mata.core.logging import get_logger

from .base import BaseAdapter

logger = get_logger(__name__)

# Lazy import for PyTorch
_torch = None
TORCH_AVAILABLE = None


def _ensure_torch():
    """Ensure PyTorch is imported (lazy loading).

    Defers torch import until actually needed to improve startup time
    and avoid import errors when PyTorch is not required.

    Returns:
        torch module if available

    Raises:
        ImportError: If torch cannot be imported
    """
    global _torch, TORCH_AVAILABLE
    if _torch is None:
        try:
            import torch

            _torch = torch
            TORCH_AVAILABLE = True
            logger.debug(f"PyTorch {torch.__version__} loaded successfully")
        except ImportError:
            TORCH_AVAILABLE = False
            raise ImportError("PyTorch is required for this adapter. " "Install with: pip install torch")
    return _torch


class PyTorchBaseAdapter(BaseAdapter):
    """Base adapter for PyTorch-based models.

    Extends BaseAdapter with PyTorch-specific functionality:
    - Lazy torch module loading
    - Device management (CPU/CUDA auto-detection)
    - Common PyTorch patterns and utilities

    Used by:
    - HuggingFaceDetectAdapter (transformers models)
    - PyTorchDetectAdapter (checkpoint files)
    - TorchScriptDetectAdapter (JIT compiled models)

    Attributes:
        torch: PyTorch module (lazily loaded)
        device: torch.device object (CPU or CUDA)
        threshold: Confidence threshold inherited from BaseAdapter
        id2label: Label mapping inherited from BaseAdapter

    Example:
        >>> class MyDetector(PyTorchBaseAdapter):
        ...     def __init__(self, model_path, **kwargs):
        ...         super().__init__(
        ...             device=kwargs.get('device', 'auto'),
        ...             threshold=kwargs.get('threshold', 0.5)
        ...         )
        ...         self.model = self.torch.load(model_path)
        ...         self.model.to(self.device)
        ...
        ...     def predict(self, image, **kwargs):
        ...         pil_image = self._load_image(image)
        ...         # Model inference...
    """

    def __init__(
        self, device: str = "auto", threshold: float = 0.3, id2label: dict[int, str] | None = None, **kwargs: Any
    ):
        """Initialize PyTorch base adapter.

        Args:
            device: Device specification ("cuda", "cpu", or "auto")
                - "auto": Use CUDA if available, otherwise CPU
                - "cuda": Force CUDA (will fail if unavailable)
                - "cpu": Force CPU
            threshold: Detection/segmentation confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping
            **kwargs: Additional arguments (passed to subclasses)

        Raises:
            ImportError: If PyTorch is not installed
            ValueError: If threshold is invalid
        """
        # Initialize base class (validates threshold)
        super().__init__(threshold=threshold, id2label=id2label)

        # Lazy load PyTorch
        self.torch = _ensure_torch()

        # Setup device
        self.device = self._setup_device(device)

        logger.info(f"Initialized {self.__class__.__name__} with device={self.device}, " f"threshold={self.threshold}")

    def _setup_device(self, device: str) -> Any:
        """Setup PyTorch device with auto-detection.

        Handles device specification and CUDA availability checks.

        Args:
            device: Device specification string
                - "auto": Auto-select CUDA if available
                - "cuda": Force CUDA
                - "cpu": Force CPU
                - "cuda:0", "cuda:1", etc.: Specific GPU

        Returns:
            torch.device object

        Raises:
            RuntimeError: If CUDA requested but not available
        """
        if device == "auto":
            # Auto-detect: prefer CUDA if available
            if self.torch.cuda.is_available():
                device_obj = self.torch.device("cuda")
                logger.info(f"Auto-selected CUDA device " f"({self.torch.cuda.get_device_name(0)})")
            else:
                device_obj = self.torch.device("cpu")
                logger.info("Auto-selected CPU device (CUDA not available)")
        else:
            # Explicit device specification
            device_obj = self.torch.device(device)

            # Validate CUDA availability if requested
            if device_obj.type == "cuda" and not self.torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device '{device}' requested but CUDA is not available. "
                    f"Use device='cpu' or device='auto' instead."
                )

            logger.info(f"Using specified device: {device_obj}")

        return device_obj
