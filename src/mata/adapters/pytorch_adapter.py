"""PyTorch checkpoint adapter with architecture auto-detection.

Supports loading PyTorch checkpoints (.pth, .pt, .bin) with automatic
architecture detection from state dict keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mata.core.exceptions import ModelLoadError, UnsupportedModelError
from mata.core.logging import get_logger
from mata.core.types import DetectResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)


class PyTorchDetectAdapter(PyTorchBaseAdapter):
    """PyTorch checkpoint detection adapter with auto-detection.

    Loads PyTorch checkpoints and automatically detects the model architecture
    from state dict keys. Supports:
    - RT-DETR checkpoints (hybrid_encoder key pattern)
    - DINO checkpoints (transformer.level_embed pattern)
    - Generic DETR checkpoints

    Requires a config file for model architecture reconstruction unless
    architecture can be reliably auto-detected.

    Examples:
        >>> # Load with auto-detection
        >>> detector = PyTorchDetectAdapter("checkpoint.pth")
        >>> # Load with manual config
        >>> detector = PyTorchDetectAdapter(
        ...     "checkpoint.pth",
        ...     config="config.yaml"
        ... )
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: str | None = None,
        device: str = "auto",
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize PyTorch checkpoint adapter.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pth, .pt, .bin)
            config: Optional path to config file for architecture
            device: Device ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping

        Raises:
            ModelLoadError: If checkpoint loading fails
            UnsupportedModelError: If architecture cannot be detected
            FileNotFoundError: If checkpoint file not found
        """
        # Initialize base class (handles device setup and torch import)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        # Validate checkpoint path
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.config_path = Path(config) if config else None

        # Load checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load PyTorch checkpoint and reconstruct model."""
        try:
            logger.info(f"Loading PyTorch checkpoint: {self.checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False  # Allow pickle for model loading
            )

            # Extract state dict (handle different checkpoint formats)
            state_dict = self._extract_state_dict(checkpoint)

            # Detect architecture from state dict
            architecture = self._detect_architecture(state_dict)
            logger.info(f"Detected architecture: {architecture}")

            # Reconstruct model based on architecture
            if self.config_path:
                # User provided config - use it to build model
                self.model = self._build_from_config(architecture, state_dict)
            else:
                # No config - attempt auto-reconstruction
                raise UnsupportedModelError(
                    f"PyTorch checkpoint loading requires a config file for architecture '{architecture}'. "
                    f"Please provide a config file path:\n"
                    f"  detector = mata.load('detect', '{self.checkpoint_path}', config='config.yaml')\n\n"
                    f"Alternatively, convert the checkpoint to ONNX format for config-free loading:\n"
                    f"  (ONNX export will be supported in a future version)"
                )

            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            if isinstance(e, (UnsupportedModelError, FileNotFoundError)):
                raise
            raise ModelLoadError(
                str(self.checkpoint_path), f"Failed to load PyTorch checkpoint: {type(e).__name__}: {str(e)}"
            )

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        """Extract state dict from checkpoint.

        Handles various checkpoint formats:
        - Direct state dict
        - Wrapped in 'model' key
        - EMA checkpoint ('ema' key)

        Args:
            checkpoint: Loaded checkpoint

        Returns:
            State dictionary
        """
        if isinstance(checkpoint, dict):
            # Check for EMA checkpoint (common in training)
            if "ema" in checkpoint:
                logger.info("Detected EMA checkpoint, using EMA weights")
                # EMA might be nested
                ema = checkpoint["ema"]
                if isinstance(ema, dict) and "module" in ema:
                    return ema["module"]
                elif isinstance(ema, dict) and "state_dict" in ema:
                    return ema["state_dict"]
                return ema

            # Check for 'model' key
            if "model" in checkpoint:
                logger.info("Detected wrapped checkpoint with 'model' key")
                return checkpoint["model"]

            # Check for 'state_dict' key
            if "state_dict" in checkpoint:
                logger.info("Detected wrapped checkpoint with 'state_dict' key")
                return checkpoint["state_dict"]

            # Assume it's the state dict itself
            return checkpoint
        else:
            raise ModelLoadError(str(self.checkpoint_path), f"Unexpected checkpoint format: {type(checkpoint)}")

    def _detect_architecture(self, state_dict: dict[str, torch.Tensor]) -> str:
        """Detect model architecture from state dict keys.

        Args:
            state_dict: Model state dictionary

        Returns:
            Architecture name
        """
        keys = list(state_dict.keys())

        # RT-DETR detection
        if any("hybrid_encoder" in k for k in keys):
            logger.info("Detected RT-DETR architecture (hybrid_encoder found)")
            return "rtdetr"

        # DINO detection
        if any("transformer.level_embed" in k for k in keys):
            logger.info("Detected DINO architecture (transformer.level_embed found)")
            return "dino"

        # Conditional DETR detection
        if any("query_head" in k for k in keys):
            logger.info("Detected Conditional DETR architecture (query_head found)")
            return "conditional_detr"

        # Generic DETR detection
        if any("transformer" in k and "decoder" in k for k in keys):
            logger.info("Detected generic DETR architecture")
            return "detr"

        # Unknown architecture
        logger.warning(f"Could not reliably detect architecture. " f"Sample keys: {keys[:5]}")
        return "unknown"

    def _build_from_config(self, architecture: str, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
        """Build model from config file.

        Args:
            architecture: Detected architecture name
            state_dict: State dictionary to load

        Returns:
            Reconstructed model

        Raises:
            UnsupportedModelError: If config loading fails
        """
        # Import architecture helpers
        try:
            if architecture == "rtdetr":
                from mata.adapters.helpers.rtdetr_helper import RTDETRHelper

                model = RTDETRHelper.build_model(self.config_path)
                RTDETRHelper.load_checkpoint(model, state_dict)
                return model

            elif architecture == "dino":
                from mata.adapters.helpers.dino_helper import DINOHelper

                model = DINOHelper.build_model(self.config_path)
                DINOHelper.load_checkpoint(model, state_dict)
                return model

            elif architecture in ("detr", "conditional_detr"):
                from mata.adapters.helpers.detr_helper import DETRHelper

                model = DETRHelper.build_model(self.config_path)
                DETRHelper.load_checkpoint(model, state_dict)
                return model

            else:
                raise UnsupportedModelError(
                    f"Architecture '{architecture}' not yet supported for PyTorch checkpoint loading. "
                    f"Supported: rtdetr, dino, detr, conditional_detr"
                )

        except ImportError as e:
            raise UnsupportedModelError(
                f"Architecture helper not implemented yet: {e}. "
                f"PyTorch checkpoint loading will be fully supported in the next release."
            )

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "PyTorchDetectAdapter",
            "task": "detect",
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "threshold": self.threshold,
            "backend": "pytorch",
        }

    @torch.no_grad()
    def predict(
        self, image: str | Path | Image.Image | np.ndarray, threshold: float | None = None, **kwargs: Any
    ) -> DetectResult:
        """Run object detection on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional threshold override
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            DetectResult with detections

        Raises:
            InvalidInputError: If image is invalid
            NotImplementedError: If preprocessing not yet implemented
        """
        raise NotImplementedError(
            "PyTorch checkpoint inference is not yet fully implemented. "
            "This feature will be available in the next release. "
            "For now, please use HuggingFace models or ONNX format:\n"
            "  detector = mata.load('detect', 'PekingU/rtdetr_v2_r18vd')  # HuggingFace\n"
            "  detector = mata.load('detect', 'model.onnx')  # ONNX"
        )
