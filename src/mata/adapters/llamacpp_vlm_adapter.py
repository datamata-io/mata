"""llama-cpp-python VLM adapter for MATA framework."""

from __future__ import annotations

import base64
import io
from typing import Any

from mata.core.logging import get_logger
from mata.core.types import VisionResult

from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppVLMAdapter(LlamaCppBaseAdapter):
    """VLM adapter for GGUF files via llama-cpp-python.

    Supports LLaVA-style multimodal (mmproj) and self-contained
    multimodal GGUF files (e.g., Qwen2-VL GGUF).

    Args:
        model_path: Path to the .gguf VLM file
        mmproj: Optional path to the multimodal projector .gguf file
                (required for LLaVA-v1.5/1.6; not needed for Qwen2-VL GGUF)
        n_gpu_layers: Layers to offload to GPU (0 = CPU-only, -1 = all)
        n_ctx: Context window size in tokens
        max_tokens: Default max tokens to generate
    """

    task = "vlm"
    name = "llamacpp_vlm"

    def __init__(
        self,
        model_path: str,
        mmproj: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        max_tokens: int = 512,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, **kwargs)
        self.mmproj = mmproj
        self.max_tokens = max_tokens

        extra: dict[str, Any] = {}
        if mmproj:
            extra["chat_handler"] = self.llama_cpp.llava_chat_handler.LlavaLogitsProcessor(
                clip_model_path=mmproj, verbose=self.verbose
            )
        self._llm = self._create_llm(**extra)

    def predict(
        self,
        image: Any,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Generate text response for an image + text prompt.

        Args:
            image: PIL Image, file path, or numpy array
            prompt: Text prompt/question (required)
            max_new_tokens: Override default max_tokens

        Returns:
            VisionResult with .text = generated response

        Raises:
            InvalidInputError: If prompt is None or empty
        """
        from mata.core.exceptions import InvalidInputError

        if not prompt:
            raise InvalidInputError("prompt is required for VLM predict()")

        pil_image, image_path = self._load_image(image)

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        n_tokens = max_new_tokens or self.max_tokens
        response = self._llm.create_chat_completion(messages=messages, max_tokens=n_tokens)
        text = response["choices"][0]["message"]["content"]

        return VisionResult(
            instances=[],
            text=text,
            prompt=prompt,
            meta={
                "model_path": self.model_path,
                "backend": "llama-cpp-python",
                "n_gpu_layers": self.n_gpu_layers,
                "image_path": image_path,
            },
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "mmproj": self.mmproj, "max_tokens": self.max_tokens})
        return d
