"""PyTorch checkpoint adapter for image classification.

Supports loading PyTorch classification checkpoints (.pth, .pt, .bin) with
automatic architecture detection from state dict keys.

Note: This is a foundation implementation. Full architecture reconstruction
requires helper modules for specific model types (ResNet, ViT, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import ClassifyResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)


class PyTorchClassifyAdapter(PyTorchBaseAdapter):
    """PyTorch checkpoint classification adapter.

    Foundation implementation for loading PyTorch classification checkpoints.
    Currently provides basic structure with automatic architecture detection.

    Full support requires helper modules for specific architectures (ResNet, ViT, etc.)

    Examples:
        >>> # Load checkpoint
        >>> classifier = PyTorchClassifyAdapter("checkpoint.pth", top_k=5)
        >>> # With manual config
        >>> classifier = PyTorchClassifyAdapter(
        ...     "checkpoint.pth",
        ...     config="config.yaml",
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: str | None = None,
        device: str = "auto",
        top_k: int = 5,
        threshold: float = 0.0,
        input_size: int = 224,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize PyTorch classification checkpoint adapter.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pth, .pt, .bin)
            config: Optional path to config file for architecture
            device: Device ("cuda", "cpu", or "auto")
            top_k: Number of top predictions to return (default: 5)
            threshold: Minimum confidence threshold for predictions (default: 0.0)
            input_size: Model input size in pixels (default: 224)
            id2label: Optional custom label mapping

        Raises:
            ModelLoadError: If checkpoint loading fails
            UnsupportedModelError: If architecture cannot be detected
            FileNotFoundError: If checkpoint file not found
            NotImplementedError: Full inference not yet implemented
        """
        # Initialize base class (handles device setup and torch import)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Validate checkpoint path
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.config_path = Path(config) if config else None
        self.top_k = top_k
        self.input_size = input_size

        # Load checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load PyTorch checkpoint and detect architecture.

        Note: Full model reconstruction not yet implemented.
        Requires architecture-specific helper modules.
        """
        try:
            logger.info(f"Loading PyTorch classification checkpoint: {self.checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False  # Allow pickle for model loading
            )

            # Extract state dict (handle different checkpoint formats)
            state_dict = self._extract_state_dict(checkpoint)

            # Detect architecture from state dict
            architecture = self._detect_architecture(state_dict)
            logger.info(f"Detected architecture: {architecture}")

            # Store for info()
            self.architecture = architecture
            self.state_dict = state_dict

            logger.warning(
                f"PyTorch classification checkpoint loaded. "
                f"Full inference requires architecture-specific helper modules. "
                f"Detected: {architecture}"
            )

        except Exception as e:
            raise ModelLoadError(
                str(self.checkpoint_path), f"Failed to load PyTorch checkpoint: {type(e).__name__}: {str(e)}"
            )

    def _extract_state_dict(self, checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract state dict from checkpoint.

        Args:
            checkpoint: Loaded checkpoint dict

        Returns:
            State dict with model weights
        """
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for common keys
            for key in ["state_dict", "model", "model_state_dict", "ema"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    # Handle EMA wrapper
                    if key == "ema" and isinstance(state_dict, dict):
                        if "module" in state_dict:
                            return state_dict["module"]
                    return state_dict

            # Assume it's already a state dict
            return checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    def _detect_architecture(self, state_dict: dict[str, torch.Tensor]) -> str:
        """Detect model architecture from state dict keys.

        Args:
            state_dict: Model state dict

        Returns:
            Architecture name (resnet, vit, convnext, etc.)
        """
        keys = list(state_dict.keys())

        # Check for known architecture patterns
        if any("layer" in k and "downsample" in k for k in keys):
            return "resnet"
        elif any("blocks" in k and "attn" in k for k in keys):
            if any("patch_embed" in k for k in keys):
                return "vit"
            else:
                return "swin"
        elif any("stages" in k and "dwconv" in k for k in keys):
            return "convnext"
        elif any("_blocks" in k for k in keys):
            return "efficientnet"

        logger.warning("Could not detect specific architecture from state dict keys")
        return "unknown"

    def classify(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image (Classify node interface).

        This method provides compatibility with the Classify graph node,
        which expects a classify() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            top_k: Number of top predictions to return
            **kwargs: Additional arguments passed to predict()

        Returns:
            ClassifyResult with predictions
        """
        return self.predict(image=image, top_k=top_k, **kwargs)

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "PyTorchClassifyAdapter",
            "task": "classify",
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "architecture": getattr(self, "architecture", "unknown"),
            "top_k": self.top_k,
            "input_size": self.input_size,
            "backend": "pytorch",
            "status": "foundation_only",
        }

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, top_k: int | None = None, **kwargs: Any
    ) -> ClassifyResult:
        """Perform image classification on input image.

        Note: Full inference not yet implemented.
        Requires architecture-specific model reconstruction.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Number of top predictions to return
            **kwargs: Additional inference parameters

        Returns:
            ClassifyResult with predictions

        Raises:
            NotImplementedError: Full inference not yet implemented
        """
        raise NotImplementedError(
            f"PyTorch classification inference requires architecture-specific helper modules. "
            f"Detected architecture: {getattr(self, 'architecture', 'unknown')}. "
            f"Use HuggingFace, ONNX, or TorchScript adapters for classification instead."
        )
