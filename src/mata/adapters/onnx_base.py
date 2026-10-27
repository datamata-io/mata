"""ONNX Runtime-specific base adapter for MATA framework.

This module provides the ONNXBaseAdapter class that extends BaseAdapter
with ONNX Runtime-specific functionality.
"""

from __future__ import annotations

from typing import Any

from mata.core.exceptions import ConfigurationError
from mata.core.logging import get_logger

from .base import BaseAdapter

logger = get_logger(__name__)

#: SessionOptions attributes callers may override.
_ALLOWED_SESSION_OPTIONS = frozenset(
    {
        "intra_op_num_threads",
        "inter_op_num_threads",
        "execution_mode",
        "enable_mem_pattern",
        "enable_cpu_mem_arena",
        "log_severity_level",
    }
)

# Lazy import for ONNX Runtime
_ort = None
ONNX_AVAILABLE = None


def _ensure_onnxruntime():
    """Ensure ONNX Runtime is imported (lazy loading).

    Defers onnxruntime import until actually needed to improve startup time
    and avoid import errors when ONNX Runtime is not required.

    Returns:
        onnxruntime module if available

    Raises:
        ImportError: If onnxruntime cannot be imported
    """
    global _ort, ONNX_AVAILABLE
    if _ort is None:
        try:
            import onnxruntime as ort

            _ort = ort
            ONNX_AVAILABLE = True
            logger.debug(f"ONNX Runtime {ort.__version__} loaded successfully")
        except ImportError:
            ONNX_AVAILABLE = False
            raise ImportError(
                "ONNX Runtime is required for ONNX models. "
                "Install with: pip install datamata[onnx]  (CPU)  or  pip install datamata[onnx-gpu]  (CUDA)"
            )
    return _ort


class ONNXBaseAdapter(BaseAdapter):
    """Base adapter for ONNX Runtime models.

    Extends BaseAdapter with ONNX Runtime-specific functionality:
    - Lazy onnxruntime module loading
    - Execution provider management (CPU/CUDA)
    - Common ONNX patterns and utilities

    Used by:
    - ONNXDetectAdapter (ONNX detection models)
    - (Future) ONNXSegmentAdapter, ONNXClassifyAdapter

    Attributes:
        ort: ONNX Runtime module (lazily loaded)
        providers: List of execution providers in priority order
        threshold: Confidence threshold inherited from BaseAdapter
        id2label: Label mapping inherited from BaseAdapter

    Example:
        >>> class MyONNXDetector(ONNXBaseAdapter):
        ...     def __init__(self, model_path, **kwargs):
        ...         super().__init__(
        ...             device=kwargs.get('device', 'auto'),
        ...             threshold=kwargs.get('threshold', 0.5)
        ...         )
        ...         self.session = self.ort.InferenceSession(
        ...             model_path,
        ...             providers=self.providers
        ...         )
        ...
        ...     def predict(self, image, **kwargs):
        ...         pil_image = self._load_image(image)
        ...         # ONNX inference...
    """

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
        channel_order: str = "rgb",
        session_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize ONNX base adapter.

        Args:
            device: Device specification ("cuda", "cuda:0", "cpu", or "auto")
                - "auto": Use CUDA if available, otherwise CPU
                - "cuda" / "cuda:N": Require the CUDA execution provider
                - "cpu": Force CPU execution provider
            threshold: Detection/segmentation confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping
            channel_order: Channel order of numpy inputs ("rgb" or "bgr")
            session_options: Optional ONNX Runtime SessionOptions overrides
            **kwargs: Additional arguments (passed to subclasses)

        Raises:
            ImportError: If onnxruntime is not installed
            ValueError: If threshold is invalid
            ConfigurationError: If CUDA is requested but unavailable
        """
        # Initialize base class (validates threshold and channel_order)
        super().__init__(threshold=threshold, id2label=id2label, channel_order=channel_order)

        # Lazy load ONNX Runtime
        self.ort = _ensure_onnxruntime()

        # Setup execution providers
        self.providers = self._get_providers(device)
        self.session_options = dict(session_options or {})

        logger.info(
            f"Initialized {self.__class__.__name__} with " f"providers={self.providers}, threshold={self.threshold}"
        )

    def _make_session_options(self) -> Any:
        """Build ONNX Runtime SessionOptions, applying caller overrides."""
        options = self.ort.SessionOptions()
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        for key, value in self.session_options.items():
            if key not in _ALLOWED_SESSION_OPTIONS:
                raise ConfigurationError(
                    f"Unsupported session option {key!r}. Allowed: {sorted(_ALLOWED_SESSION_OPTIONS)}"
                )
            setattr(options, key, value)
        return options

    def _get_providers(self, device: str) -> list[str]:
        """Get ONNX Runtime execution providers based on device.

        Args:
            device: Device specification string
                - "auto": Auto-select CUDA if available
                - "cuda" / "cuda:N": Require the CUDA provider
                - "cpu": CPU provider only

        Returns:
            List of provider names in priority order

        Raises:
            ConfigurationError: If CUDA is explicitly requested but the
                installed onnxruntime has no CUDA execution provider.
        """
        available_providers = self.ort.get_available_providers()
        device_kind = str(device).split(":", 1)[0].strip().lower()
        logger.debug(f"Available ONNX providers: {available_providers}")

        if device_kind == "cuda":
            if "CUDAExecutionProvider" not in available_providers:
                raise ConfigurationError(
                    "CUDAExecutionProvider requested but onnxruntime reports "
                    f"{available_providers}. Install datamata[onnx-gpu] with a "
                    "matching CUDA runtime, or use device='cpu'/'auto'."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA execution provider")
        elif device_kind == "auto" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA execution provider (auto)")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU execution provider")

        return providers
