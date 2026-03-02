"""HuggingFace depth estimation adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import DepthResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

_transformers = None
TRANSFORMERS_AVAILABLE = None


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading)."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            _transformers = {
                "AutoImageProcessor": AutoImageProcessor,
                "AutoModelForDepthEstimation": AutoModelForDepthEstimation,
            }
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


class HuggingFaceDepthAdapter(PyTorchBaseAdapter):
    """Depth estimation adapter for HuggingFace models."""

    name = "huggingface_depth"
    task = "depth"

    def __init__(self, model_id: str, device: str = "auto", normalize: bool = True, **kwargs: Any) -> None:
        super().__init__(device=device, threshold=0.0)

        transformers = _ensure_transformers()
        if not transformers:
            raise ImportError(
                "transformers is required for HuggingFace depth adapter. "
                "Install with: pip install transformers torch"
            )

        self.transformers = transformers
        self.model_id = model_id
        self.normalize = normalize

        self._load_model()

    def _load_model(self) -> None:
        """Load model and processor."""
        from mata.core.logging import suppress_third_party_logs

        try:
            logger.info(f"Loading HuggingFace depth model: {self.model_id}")
            with suppress_third_party_logs():
                self.processor = self.transformers["AutoImageProcessor"].from_pretrained(self.model_id)
                self.model = self.transformers["AutoModelForDepthEstimation"].from_pretrained(self.model_id)
            self.model = self.model.to(self.device).eval()
            logger.info(f"Depth model loaded successfully on {self.device}")
        except Exception as e:
            raise ModelLoadError(self.model_id, f"Failed to load HuggingFace depth model: {type(e).__name__}: {str(e)}")

    def estimate(self, image: Any, **kwargs: Any) -> DepthResult:
        """Estimate depth (EstimateDepth node interface).

        This method provides compatibility with the EstimateDepth graph node,
        which expects an estimate() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            **kwargs: Additional arguments passed to predict()

        Returns:
            DepthResult with depth map
        """
        return self.predict(image=image, **kwargs)

    def info(self) -> dict[str, Any]:
        """Return adapter information."""
        return {
            "name": self.name,
            "task": self.task,
            "model_id": self.model_id,
            "device": str(self.device),
            "backend": "transformers",
            "normalize": self.normalize,
        }

    def predict(self, image: Any, **kwargs: Any) -> DepthResult:
        """Perform depth estimation on an image."""
        pil_image, input_path = self._load_image(image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_size = kwargs.get("target_size")
        if target_size is None:
            target_sizes = [(pil_image.height, pil_image.width)]
        else:
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError("target_size must be a tuple/list of (height, width)")
            target_sizes = [tuple(target_size)]

        post_processed = self.processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)

        predicted_depth = post_processed[0]["predicted_depth"]
        depth = predicted_depth.squeeze().detach().cpu().numpy()

        normalize = kwargs.get("normalize", self.normalize)
        normalized = None
        if normalize:
            depth_min = float(np.nanmin(depth))
            depth_max = float(np.nanmax(depth))
            if depth_max - depth_min > 1e-8:
                normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                normalized = np.zeros_like(depth, dtype=np.float32)

        meta = {
            "model_id": self.model_id,
            "input_path": input_path,
            "device": str(self.device),
            "task": "depth",
            "normalized": normalize,
        }

        return DepthResult(depth=depth, normalized=normalized, meta=meta)
