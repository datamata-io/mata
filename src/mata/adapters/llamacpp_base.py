"""llama-cpp-python base adapter for MATA framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.logging import get_logger

from .base import BaseAdapter

logger = get_logger(__name__)

_llama_cpp = None
LLAMA_CPP_AVAILABLE = None


def _ensure_llama_cpp():
    """Lazy import for llama-cpp-python. Raises ImportError with install hint."""
    global _llama_cpp, LLAMA_CPP_AVAILABLE
    if _llama_cpp is None:
        try:
            import llama_cpp

            _llama_cpp = llama_cpp
            LLAMA_CPP_AVAILABLE = True
            logger.debug("llama-cpp-python loaded successfully")
        except ImportError:
            LLAMA_CPP_AVAILABLE = False
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install datamata[gguf]  "
                "For GPU offloading see: https://llama-cpp-python.readthedocs.io/en/latest/#installation"
            )
    return _llama_cpp


class LlamaCppBaseAdapter(BaseAdapter):
    """Base adapter for llama-cpp-python GGUF models.

    Extends BaseAdapter directly (no PyTorch dependency) — follows the same
    isolation design as ONNXBaseAdapter.

    Attributes:
        llama_cpp: llama_cpp module (lazily loaded)
        model_path: Absolute path to the .gguf file
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU-only, -1 = all)
        n_ctx: Context window size in tokens
        verbose: Whether to enable llama.cpp verbose logging
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        verbose: bool = False,
        threshold: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold)
        self.llama_cpp = _ensure_llama_cpp()

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}. "
                f"Download a .gguf file from HuggingFace Hub or another source."
            )
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Expected .gguf file, got: {path.suffix}")

        self.model_path = str(path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose

    def _create_llm(self, **extra_kwargs: Any) -> Any:
        """Create a llama_cpp.Llama instance. Subclasses call with task-specific kwargs."""
        return self.llama_cpp.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
            **extra_kwargs,
        )

    def predict(self, image: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement predict()")

    def info(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "backend": "llama-cpp-python",
        }
