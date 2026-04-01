"""Public embed adapter — conforms to Embedder protocol.

Wraps ReIDAdapter subclasses to provide the public mata.load("embed", ...)
interface. Accepts Image or ROIs artifacts and returns np.ndarray embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.artifacts.image import Image
from mata.core.artifacts.rois import ROIs
from mata.core.logging import get_logger

logger = get_logger(__name__)


class EmbedAdapter:
    """Public embedding adapter — wraps any ReIDAdapter subclass.

    Conforms to the ``Embedder`` protocol defined in
    ``mata.core.registry.protocols``.

    Args:
        encoder: A ``ReIDAdapter`` instance (HuggingFace or ONNX).

    Example:
        >>> from mata.adapters.reid_adapter import HuggingFaceReIDAdapter
        >>> adapter = EmbedAdapter(encoder=HuggingFaceReIDAdapter("openai/clip-vit-base-patch32"))
        >>> img = Image.from_path("photo.jpg")
        >>> emb = adapter.embed(img)  # (1, 512) float32
        >>>
        >>> rois = ROIs(roi_images=[crop1, crop2], source_boxes=[(0,0,10,10),(0,0,10,10)])
        >>> embs = adapter.embed(rois)  # (2, 512) float32
    """

    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    def embed(self, input: Image | ROIs, **kwargs: Any) -> np.ndarray:
        """Extract feature embeddings.

        Args:
            input: Image artifact (whole-image embedding) or ROIs artifact
                   (per-region embeddings).
            **kwargs: Forwarded to encoder. Supported:
                - normalize (bool): L2 normalize output (default True,
                  already done by ReIDAdapter).

        Returns:
            np.ndarray: (N, D) float32 embeddings.
                - Image input: N=1
                - ROIs input: N=len(rois)

        Raises:
            TypeError: If input is not an Image or ROIs artifact.
        """
        if isinstance(input, Image):
            np_image = input.to_numpy()
            # X-CLIP has no predict(); embed a single frame as a 1-frame video clip
            # (_resample_frames inside predict_video repeats it to fill n_frames).
            if hasattr(self._encoder, "predict_video") and not hasattr(self._encoder, "predict"):
                return self._encoder.predict_video([np_image])
            return self._encoder.predict([np_image])
        elif isinstance(input, ROIs):
            crops = [np.array(roi) for roi in input.roi_images]
            if not crops:
                return np.empty((0, 0), dtype=np.float32)
            return self._encoder.predict(crops)
        elif isinstance(input, list):
            # Video clip: list of BGR numpy frames → (1, D)
            if not hasattr(self._encoder, "predict_video"):
                from mata.core.exceptions import UnsupportedModelError

                raise UnsupportedModelError(
                    "Video clip embedding (list of frames) requires an X-CLIP encoder. "
                    f"The current encoder ({type(self._encoder).__name__}) does not support predict_video(). "
                    "Load with: mata.load('embed', 'microsoft/xclip-base-patch32')"
                )
            return self._encoder.predict_video(input)
        elif isinstance(input, str):
            # Text query → (1, D) in same space as video/image embeddings
            if not hasattr(self._encoder, "predict_text"):
                from mata.core.exceptions import UnsupportedModelError

                raise UnsupportedModelError(
                    "Text embedding requires a CLIP or X-CLIP encoder. "
                    f"The current encoder ({type(self._encoder).__name__}) does not support predict_text(). "
                    "Load with: mata.load('embed', 'openai/clip-vit-base-patch32') or "
                    "mata.load('embed', 'microsoft/xclip-base-patch32')"
                )
            return self._encoder.predict_text(input)
        else:
            raise TypeError(f"EmbedAdapter.embed() expects Image or ROIs, got {type(input).__name__}")

    @property
    def embedding_dim(self) -> int | None:
        """Embedding dimensionality (available after first embed call)."""
        return self._encoder.embedding_dim

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query into the model's vector space.

        Requires an encoder that supports predict_text() (e.g. X-CLIP).

        Args:
            text: Natural-language query string.

        Returns:
            np.ndarray: (1, D) float32, L2-normalized.
        """
        return self.embed(text)

    def info(self) -> dict[str, Any]:
        """Adapter metadata."""
        base = self._encoder.info()
        base["type"] = "embed"
        return base
