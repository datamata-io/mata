"""llama-cpp-python classify adapter for MATA framework."""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.exceptions import InvalidInputError
from mata.core.logging import get_logger
from mata.core.types import Classification, ClassifyResult

from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppClassifyAdapter(LlamaCppBaseAdapter):
    """Zero-shot classification adapter for CLIP GGUF files.

    Computes cosine similarity between image embedding and text prompt
    embeddings, matching the pattern of HuggingFaceCLIPAdapter.
    Requires a CLIP-capable GGUF model (embedding=True mode).

    Args:
        model_path: Path to CLIP GGUF file
        text_prompts: Class labels for zero-shot classification
        n_gpu_layers: Layers to offload to GPU (0 = CPU-only, -1 = all)
    """

    task = "classify"
    name = "llamacpp_classify"

    def __init__(
        self,
        model_path: str,
        text_prompts: list[str] | None = None,
        n_gpu_layers: int = 0,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=512, **kwargs)
        self.text_prompts = text_prompts or []
        self._llm = self._create_llm(embedding=True)

    def predict(
        self,
        image: Any,
        text_prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image via cosine similarity against text prompts.

        Args:
            image: PIL Image, file path, or numpy array
            text_prompts: Override constructor text_prompts for this call

        Returns:
            ClassifyResult with classifications sorted by score descending

        Raises:
            InvalidInputError: If no text_prompts available at call or constructor time
        """
        prompts = text_prompts or self.text_prompts
        if not prompts:
            raise InvalidInputError(
                "text_prompts required for GGUF classify. "
                "Pass at load time: mata.load('classify', 'model.gguf', text_prompts=[...]) "
                "or at run time: mata.run('classify', 'image.jpg', text_prompts=[...])"
            )

        pil_image, _ = self._load_image(image)

        img_emb = np.array(self._llm.embed(pil_image), dtype=np.float32)
        img_norm = np.linalg.norm(img_emb)
        img_emb = img_emb / (img_norm + 1e-8)

        classifications = []
        for idx, label in enumerate(prompts):
            text_emb = np.array(self._llm.embed(label), dtype=np.float32)
            text_norm = np.linalg.norm(text_emb)
            text_emb = text_emb / (text_norm + 1e-8)
            score = float(np.dot(img_emb, text_emb))
            classifications.append(Classification(label=idx, label_name=label, score=max(0.0, score)))

        classifications.sort(key=lambda c: c.score, reverse=True)
        return ClassifyResult(
            predictions=classifications,
            meta={"model_path": self.model_path, "backend": "llama-cpp-python"},
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "text_prompts": self.text_prompts})
        return d
