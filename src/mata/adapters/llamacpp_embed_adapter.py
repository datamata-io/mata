"""llama-cpp-python embed adapter for MATA framework."""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.exceptions import UnsupportedModelError
from mata.core.logging import get_logger

from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppEmbedAdapter(LlamaCppBaseAdapter):
    """Embedding adapter for GGUF files via llama-cpp-python.

    Uses llama_cpp.Llama(embedding=True) mode — works with CLIP GGUF
    and other embedding-capable GGUF models.

    Implements the ReIDAdapter duck-type interface so it can be wrapped
    by EmbedAdapter for the public embed task API. Does NOT subclass
    ReIDAdapter directly to avoid inheriting PyTorchBaseAdapter's torch dep.

    Args:
        model_path: Path to an embedding-capable .gguf file (e.g., CLIP GGUF)
        n_gpu_layers: Layers to offload to GPU (0 = CPU-only, -1 = all)
        n_ctx: Context window size (default 512 for embedding models)
    """

    task = "embed"
    name = "llamacpp_embed"

    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 512, **kwargs: Any):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, **kwargs)
        self._embedding_dim: int | None = None
        try:
            self._llm = self._create_llm(embedding=True)
        except Exception as e:
            raise UnsupportedModelError(
                f"GGUF model '{model_path}' does not support embedding mode. "
                f"Ensure the model is a CLIP or embedding-capable GGUF file. "
                f"Original error: {e}"
            ) from e

    @property
    def embedding_dim(self) -> int | None:
        """Embedding dimensionality; None until first predict() call."""
        return self._embedding_dim

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized embeddings for a batch of image crops.

        Args:
            crops: List of (H, W, 3) uint8 numpy arrays

        Returns:
            (N, D) float32 L2-normalized embedding array
        """
        from PIL import Image

        if not crops:
            empty = np.zeros((0, self._embedding_dim or 0), dtype=np.float32)
            return empty

        embeddings = []
        for crop in crops:
            pil = Image.fromarray(crop)
            embedding = self._llm.embed(pil)
            embeddings.append(np.array(embedding, dtype=np.float32))

        result = np.stack(embeddings, axis=0)  # (N, D)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result = result / norms

        self._embedding_dim = result.shape[1]
        return result

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "embedding_dim": self._embedding_dim})
        return d
