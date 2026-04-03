from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.logging import get_logger

logger = get_logger(__name__)

# Optional qwen-vl-utils for enhanced preprocessing
_qwen_vl_utils = None
_QWEN_VL_UTILS_AVAILABLE = None


def _try_load_qwen_vl_utils():
    """Lazy-load qwen-vl-utils. Graceful fallback -- not required."""
    global _qwen_vl_utils, _QWEN_VL_UTILS_AVAILABLE
    if _QWEN_VL_UTILS_AVAILABLE is None:
        try:
            import qwen_vl_utils

            _qwen_vl_utils = qwen_vl_utils
            _QWEN_VL_UTILS_AVAILABLE = True
            logger.debug("qwen-vl-utils loaded -- using enhanced preprocessing")
        except ImportError:
            _QWEN_VL_UTILS_AVAILABLE = False
            logger.info(
                "qwen-vl-utils not found, using basic preprocessing. "
                "For optimal quality: pip install datamata[qwen3-embedding]"
            )
    return _qwen_vl_utils


class Qwen3VLEmbeddingAdapter:
    """Multimodal embedding encoder using Qwen3-VL-Embedding.

    Embeds text, images, video, and arbitrary combinations into a shared
    vector space (up to 4096D for 8B, 2048D for 2B), enabling cross-modal
    similarity search and retrieval.

    Args:
        model_id: HuggingFace model ID, e.g. "Qwen/Qwen3-VL-Embedding-8B".
        device: Torch device string ("auto", "cpu", "cuda", "mps").
        dtype: Torch dtype string ("float16", "bfloat16", "float32").
        embed_dim: Output embedding dimension (None -> full native dimension).
            Supports Matryoshka truncation from 64 to native max.
        fps: Frame sampling rate for video files (default 1.0).
        max_frames: Maximum frames to sample from video (default 64).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str | None = None,
        embed_dim: int | None = None,
        fps: float = 1.0,
        max_frames: int = 64,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self._embed_dim = embed_dim
        self._native_dim: int | None = None
        self._embedding_dim: int | None = None
        self.fps = fps
        self.max_frames = max_frames

        # Resolve device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Resolve dtype
        dtype_map: dict[str | None, Any] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            None: torch.float32,
        }
        self._dtype = dtype_map.get(dtype, torch.float32)

        logger.info(f"Loading Qwen3-VL-Embedding: {model_id} on {self._device} ({self._dtype})")

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, dtype=self._dtype)
        self._model.eval()
        self._model.to(self._device)

        # Try loading qwen-vl-utils for enhanced preprocessing
        _try_load_qwen_vl_utils()

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """Embed a single image (BGR uint8) into a vector.

        Returns:
            np.ndarray: (1, D) float32, L2-normalized.
        """
        return self.predict_multimodal({"image": image})

    def predict_text(self, text: str) -> np.ndarray:
        """Embed a text query into the same vector space.

        Returns:
            np.ndarray: (1, D) float32, L2-normalized.
        """
        return self.predict_multimodal({"text": text})

    def predict_video(self, frames: list[np.ndarray]) -> np.ndarray:
        """Embed video frames into a single vector.

        Args:
            frames: List of (H, W, 3) uint8 BGR frames.

        Returns:
            np.ndarray: (1, D) float32, L2-normalized.
        """
        return self.predict_multimodal({"video": frames})

    def predict_multimodal(self, input_dict: dict[str, Any]) -> np.ndarray:
        """Embed any combination of text, image, video into a single vector.

        Args:
            input_dict: Dict with optional keys "text", "image", "video".
                - "text": str
                - "image": np.ndarray (H,W,3) BGR uint8, PIL Image, or file path str
                - "video": list[np.ndarray] BGR frames, or file path str

        Returns:
            np.ndarray: (1, D) float32, L2-normalized.
        """
        import torch

        # Build conversation message content for processor
        content = []
        instruction = input_dict.get("instruction", "Represent the user's input.")

        # Process image input
        image_input = input_dict.get("image")
        if image_input is not None:
            pil_image = self._to_pil(image_input)
            content.append({"type": "image", "image": pil_image})

        # Process video input
        video_input = input_dict.get("video")
        if video_input is not None:
            pil_frames = self._process_video(video_input)
            content.append({"type": "video", "video": pil_frames})

        # Process text input
        text_input = input_dict.get("text")
        if text_input is not None:
            content.append({"type": "text", "text": text_input})

        if not content:
            raise ValueError(
                "predict_multimodal() received empty input dict -- " "provide at least one of: text, image, video"
            )

        # Build chat conversation for processor
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

        # Apply chat template to get prompt text
        prompt_text = self._processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Process inputs through processor
        inputs = self._prepare_inputs(prompt_text, content)
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        # Forward pass + EOS extraction
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            # Extract last hidden state at EOS token position
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
            # EOS is the last token
            eos_embedding = last_hidden[:, -1, :]  # (1, hidden_dim)

        emb = eos_embedding[0].cpu().float().numpy()
        emb = self._l2_normalize(emb)

        # Matryoshka truncation if embed_dim specified
        if self._embed_dim is not None and self._embed_dim < len(emb):
            emb = emb[: self._embed_dim]
            emb = self._l2_normalize(emb)  # Re-normalize after truncation

        self._native_dim = eos_embedding.shape[-1]
        self._embedding_dim = len(emb)
        return emb[np.newaxis, :]  # (1, D)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Batch-embed image crops (for EmbedAdapter Image/ROIs compatibility).

        Args:
            crops: List of (H, W, 3) uint8 BGR arrays.

        Returns:
            np.ndarray: (N, D) float32, each row L2-normalized.
        """
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = [self.predict_image(crop)[0] for crop in crops]
        return np.stack(embeddings, axis=0)

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def info(self) -> dict[str, Any]:
        return {
            "type": "qwen3_vl_embedding",
            "model_id": self.model_id,
            "native_dim": self._native_dim,
            "embedding_dim": self._embedding_dim,
            "embed_dim_requested": self._embed_dim,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "fps": self.fps,
            "max_frames": self.max_frames,
            "qwen_vl_utils_available": _QWEN_VL_UTILS_AVAILABLE or False,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_pil(self, image: Any) -> Any:
        """Convert image input to PIL Image."""
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            return PILImage.fromarray(image[:, :, ::-1])  # BGR -> RGB
        if isinstance(image, PILImage.Image):
            return image
        if isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _process_video(self, video: Any) -> list:
        """Convert video input to list of PIL frames."""
        from PIL import Image as PILImage

        if isinstance(video, list):
            # List of BGR numpy frames
            sampled = self._sample_frames(video)
            return [PILImage.fromarray(f[:, :, ::-1]) for f in sampled]
        if isinstance(video, str):
            # Video file path -- extract frames
            frames = self._extract_frames_from_file(video)
            return [PILImage.fromarray(f[:, :, ::-1]) for f in frames]
        raise TypeError(f"Unsupported video type: {type(video).__name__}")

    def _sample_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Uniformly sample frames to max_frames."""
        if not frames:
            raise ValueError("Empty frame list for video embedding")
        n = len(frames)
        if n <= self.max_frames:
            return frames
        indices = np.linspace(0, n - 1, self.max_frames, dtype=int)
        return [frames[i] for i in indices]

    def _extract_frames_from_file(self, video_path: str) -> list[np.ndarray]:
        """Extract frames from video file at configured fps."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(source_fps / self.fps))
        frames = []
        idx = 0
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        return frames

    def _prepare_inputs(self, prompt_text: str, content: list[dict]) -> dict:
        """Prepare model inputs, using qwen-vl-utils if available."""
        # Extract PIL images and video frames for processor
        images = []
        videos = []
        for item in content:
            if item["type"] == "image":
                images.append(item["image"])
            elif item["type"] == "video":
                videos.append(item["video"])

        kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos

        return self._processor(**kwargs)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v
