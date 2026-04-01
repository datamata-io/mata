from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.logging import get_logger

logger = get_logger(__name__)


class XCLIPAdapter:
    """Temporal video-language encoder using Microsoft X-CLIP.

    Embeds a list of video frames (a clip) and text queries into the same
    512-dimensional vector space, enabling direct cosine comparison for
    semantic video retrieval.

    Args:
        model_id: HuggingFace model ID, e.g. "microsoft/xclip-base-patch32".
        device: Torch device string ("auto", "cpu", "cuda", "mps").
        n_frames: Number of frames to sample per clip (default 8, matching
            X-CLIP-base training). Clips with more frames are uniformly
            subsampled; clips with fewer are repeated to fill.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        n_frames: int = 8,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.n_frames = n_frames
        self._embedding_dim: int | None = None

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info(f"Loading X-CLIP encoder: {model_id} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        self._model.to(self._device)

        # Cache dummy text inputs for video-only forward passes.
        # XCLIPModel requires both pixel_values and input_ids in its forward
        # pass (to compute cross-modal logits), so predict_video() uses a
        # neutral dummy text rather than get_video_features(), which returns
        # raw patch embeddings instead of projected features in transformers >=5.2.
        _dummy = self._processor(text=[""], return_tensors="pt", padding=True)
        self._dummy_text_inputs: dict = {k: v.to(self._device) for k, v in _dummy.items()}

    def predict_video(self, frames: list[np.ndarray]) -> np.ndarray:
        """Embed a video clip (list of BGR frames) into a 512D vector.

        Args:
            frames: List of (H, W, 3) uint8 numpy arrays in BGR order.
                    Any length — uniformly resampled to self.n_frames.

        Returns:
            np.ndarray: (1, 512) float32, L2-normalized.
        """
        import torch
        from PIL import Image as PILImage

        # Resample to n_frames
        frames = self._resample_frames(frames)

        # BGR→RGB and convert to PIL
        pil_frames = [PILImage.fromarray(f[:, :, ::-1]) for f in frames]

        with torch.no_grad():
            # Use images= instead of videos= — in transformers >=5.2 the
            # XCLIPProcessor.__call__ loop only iterates get_attributes(), which
            # returns ['image_processor', 'tokenizer']. The videos= kwarg is
            # routed to video_processor which is not in that list and is silently
            # dropped. Passing a flat list of frames via images= hits
            # image_processor (VideoMAEImageProcessor) which correctly returns
            # pixel_values of shape (1, n_frames, C, H, W).
            inputs = self._processor(images=pil_frames, return_tensors="pt")
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                raise RuntimeError(
                    f"XCLIPProcessor did not produce pixel_values for {len(pil_frames)} frames. "
                    f"Processor type: {type(self._processor).__name__}"
                )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            # get_video_features() returns raw patch embeddings in transformers >=5.2.
            # Run the full forward pass with a cached dummy text and read
            # out.video_embeds which contains the proper projected (B, D) features.
            merged = {**inputs, **self._dummy_text_inputs}
            out = self._model(**merged)
            video_features = out.video_embeds  # (B, D)

        emb = video_features[0].cpu().float().numpy()
        emb = self._l2_normalize(emb)
        self._embedding_dim = emb.shape[0]
        return emb[np.newaxis, :]  # (1, D)

    def predict_text(self, text: str) -> np.ndarray:
        """Embed a text query into the same 512D vector space as video clips.

        Args:
            text: Natural-language query string.

        Returns:
            np.ndarray: (1, 512) float32, L2-normalized.
        """
        import torch

        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            # get_text_features() returns raw token embeddings in transformers >=5.2.
            # Use the text sub-model + projection directly to get the proper
            # (batch, projection_dim) text features.
            txt_out = self._model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = self._model.text_projection(txt_out.pooler_output)  # (B, D)

        emb = text_features[0].cpu().float().numpy()
        emb = self._l2_normalize(emb)
        self._embedding_dim = emb.shape[0]
        return emb[np.newaxis, :]  # (1, D)

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def info(self) -> dict[str, Any]:
        return {
            "type": "xclip",
            "model_id": self.model_id,
            "n_frames": self.n_frames,
            "embedding_dim": self._embedding_dim,
            "device": str(self._device),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resample_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Uniformly resample frame list to exactly self.n_frames."""
        n = len(frames)
        if n == 0:
            raise ValueError("XCLIPAdapter.predict_video() received an empty frame list.")
        if n == self.n_frames:
            return frames
        indices = np.linspace(0, n - 1, self.n_frames, dtype=int)
        return [frames[i] for i in indices]

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v
