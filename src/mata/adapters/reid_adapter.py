"""ReID feature extraction adapter.

Internal adapter used by TrackingAdapter to extract appearance embeddings
from detection crops.  Not a public task adapter — users access ReID
through mata.track(with_reid=True).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from mata.adapters.pytorch_base import PyTorchBaseAdapter
from mata.core.logging import get_logger

logger = get_logger(__name__)


class ReIDAdapter(PyTorchBaseAdapter):
    """Base class for ReID feature extraction adapters.

    Subclasses must implement:
        _load_model() — Load the encoder model and preprocessor.
        _extract_single(crop) — Extract embedding from one crop.

    The public interface is:
        predict(crops) — Batch-extract embeddings from N crops.
                         Returns (N, D) float32 array, L2-normalised.
    """

    def __init__(self, model_id: str, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device)
        self.model_id = model_id
        self._embedding_dim: int | None = None
        self._load_model(**kwargs)

    @abstractmethod
    def _load_model(self, **kwargs: Any) -> None:
        """Load encoder weights and preprocessor."""

    @abstractmethod
    def _extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract raw embedding from a single BGR/RGB crop.

        Args:
            crop: (H, W, 3) uint8 numpy array.

        Returns:
            1-D float32 embedding vector (unnormalised).
        """

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Batch-extract L2-normalised embeddings.

        Args:
            crops: List of (H, W, 3) uint8 numpy arrays.

        Returns:
            (N, D) float32 array, each row L2-normalised.
            Returns empty (0, 0) array if crops is empty.
        """
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = []
        for crop in crops:
            emb = self._extract_single(crop)
            embeddings.append(emb)

        result = np.stack(embeddings).astype(np.float32)

        # L2 normalise each row
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        result = result / norms

        self._embedding_dim = result.shape[1]
        return result

    @property
    def embedding_dim(self) -> int | None:
        """Embedding dimensionality (available after first predict call)."""
        return self._embedding_dim

    def info(self) -> dict[str, Any]:
        return {
            "type": "reid",
            "model_id": self.model_id,
            "embedding_dim": self._embedding_dim,
            "device": str(self.device),
        }


class ONNXReIDAdapter(ReIDAdapter):
    """ONNX Runtime ReID feature extractor.

    Loads a .onnx file with a single image input and single embedding output.
    Input shape: (1, 3, H, W) or (1, H, W, 3) — auto-detected from model metadata.
    Output shape: (1, D) — embedding dimension read from output spec.

    Args:
        model_id: Path to the .onnx file.
        device: Ignored for ONNX (use ``providers`` kwarg instead).
        providers: ONNX Runtime execution providers list.
                   Defaults to ``["CPUExecutionProvider"]``.

    Example::

        adapter = ONNXReIDAdapter("osnet.onnx")
        embeddings = adapter.predict([crop1, crop2])  # (2, D) float32
    """

    # ImageNet normalisation constants (RGB, [0, 1] range)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self, **kwargs: Any) -> None:
        import onnxruntime as ort

        providers = kwargs.get("providers", ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(self.model_id, providers=providers)

        inp = self._session.get_inputs()[0]
        self._input_name: str = inp.name
        self._input_shape: list = inp.shape  # e.g. [1, 3, 256, 128]

        self._layout: str = self._detect_layout(self._input_shape)
        logger.info(f"Loaded ONNX ReID model: {self.model_id} " f"(input={self._input_shape}, layout={self._layout})")

    @staticmethod
    def _detect_layout(shape: list) -> str:
        """Detect input tensor layout (NCHW or NHWC) from ONNX input shape.

        Inspects the channel dimension (index 1 for NCHW, index 3 for NHWC).
        When ambiguous, defaults to NCHW which is more common for ReID models.

        Args:
            shape: ONNX input shape list, e.g. ``[1, 3, 256, 128]``.

        Returns:
            ``"NCHW"`` or ``"NHWC"``.
        """
        if len(shape) != 4:
            return "NCHW"

        c_at_1 = shape[1]
        c_at_3 = shape[3]

        # NHWC: channels at index 3 == 3, channels at index 1 is not 3
        if isinstance(c_at_3, int) and c_at_3 == 3:
            if not (isinstance(c_at_1, int) and c_at_1 == 3):
                return "NHWC"
            # Both indices happen to be 3 — default NCHW (more common)
            return "NCHW"

        # NCHW: channels at index 1 == 3
        return "NCHW"

    def _get_spatial_dims(self) -> tuple[int, int]:
        """Return ``(height, width)`` expected by the model.

        Falls back to ``(256, 128)`` (a common ReID resolution) when the ONNX
        model uses dynamic/symbolic dimensions.
        """
        shape = self._input_shape
        if self._layout == "NCHW":
            h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 256
            w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 128
        else:  # NHWC
            h = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 256
            w = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 128
        return int(h), int(w)

    def _preprocess(self, crop: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize, normalise, and reshape a crop into an ONNX input tensor.

        Args:
            crop: ``(H, W, 3)`` uint8 RGB numpy array.
            height: Target height.
            width: Target width.

        Returns:
            ``(1, C, H, W)`` or ``(1, H, W, C)`` float32 array.
        """
        from PIL import Image

        pil_img = Image.fromarray(crop.astype(np.uint8))
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        img = np.array(pil_img, dtype=np.float32) / 255.0

        # ImageNet normalisation
        img = (img - self._MEAN) / self._STD  # (H, W, 3)

        if self._layout == "NCHW":
            tensor = img.transpose(2, 0, 1)[np.newaxis]  # (1, C, H, W)
        else:
            tensor = img[np.newaxis]  # (1, H, W, C)

        return tensor.astype(np.float32)

    def _extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Run a single crop through the ONNX session and return raw embedding.

        Args:
            crop: ``(H, W, 3)`` uint8 RGB numpy array.

        Returns:
            1-D float32 embedding vector (unnormalised).
        """
        h, w = self._get_spatial_dims()
        tensor = self._preprocess(crop, h, w)
        outputs = self._session.run(None, {self._input_name: tensor})
        # First output is typically (1, D); flatten to (D,)
        return outputs[0].flatten().astype(np.float32)

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "runtime": "onnx",
            "layout": getattr(self, "_layout", None),
            "input_shape": getattr(self, "_input_shape", None),
        }


class HuggingFaceReIDAdapter(ReIDAdapter):
    """HuggingFace-backed ReID feature extractor.

    Supports:
    - AutoModel (generic feature extraction via last hidden state mean pooling)
    - CLIPModel (image encoder branch, returns image embeddings)
    - ViT/DeiT/Swin image models (pooler_output if available, else mean pool)

    Architecture auto-detection order:
    1. Model ID contains 'clip' → use CLIPModel image encoder
    2. Config model_type in ('vit', 'deit', 'swin') → use pooler_output
    3. Fallback → AutoModel + mean pooling of last_hidden_state

    All transformers imports are lazy — the library is not imported at module
    import time.
    """

    # Architecture families that produce pooler_output via AutoModel
    _POOLER_ARCHS = {"vit", "deit", "swin", "beit", "convnext", "mobilevit", "efficientnet"}

    def _load_model(self, **kwargs: Any) -> None:
        """Load HuggingFace model with architecture auto-detection."""
        from PIL import Image as _PilImage  # noqa: F401 — ensure PIL available

        self._arch = self._detect_architecture()
        logger.info(f"Loading ReID encoder: {self.model_id} (arch={self._arch})")

        if self._arch == "clip":
            self._load_clip(**kwargs)
        else:
            self._load_automodel(**kwargs)

    def _detect_architecture(self) -> str:
        """Detect model architecture from model_id string, then config probe.

        Returns:
            One of: "clip", "vit_pooler", "generic"
        """
        model_id_lower = self.model_id.lower()
        if "clip" in model_id_lower:
            return "clip"

        # Probe config for model_type
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_id)
            model_type = getattr(config, "model_type", "").lower()
            if model_type in self._POOLER_ARCHS:
                return "vit_pooler"
        except Exception:
            pass

        return "generic"

    def _load_clip(self, **kwargs: Any) -> None:
        """Load CLIP model — use image encoder only."""
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

    def _load_automodel(self, **kwargs: Any) -> None:
        """Load generic AutoModel for feature extraction."""
        from transformers import AutoModel, AutoProcessor

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception:
            # Fallback: some ViT models only expose AutoFeatureExtractor
            from transformers import AutoFeatureExtractor

            self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)

        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

    def _extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Forward pass through the encoder and return a pooled feature vector.

        Args:
            crop: (H, W, 3) uint8 numpy array (RGB).

        Returns:
            1-D float32 embedding vector (unnormalised).
        """
        import torch
        from PIL import Image

        pil_image = Image.fromarray(crop)

        with torch.no_grad():
            if self._arch == "clip":
                inputs = self._processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Get image features from CLIP's vision encoder
                image_features = self._model.get_image_features(**inputs)
                embedding = image_features[0].cpu().float().numpy()
            else:
                inputs = self._processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)

                if (
                    self._arch == "vit_pooler"
                    and hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    # Use pooler_output for ViT/DeiT/Swin models
                    embedding = outputs.pooler_output[0].cpu().float().numpy()
                else:
                    # Generic fallback: mean-pool over the token sequence dimension
                    last_hidden = outputs.last_hidden_state  # (1, T, D)
                    embedding = last_hidden[0].mean(dim=0).cpu().float().numpy()

        return embedding
