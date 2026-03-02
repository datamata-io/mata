"""DETR architecture helper for PyTorch checkpoint loading.

Will be implemented in a future release.
"""

from pathlib import Path

import torch


class DETRHelper:
    """Helper for DETR model reconstruction."""

    @staticmethod
    def build_model(config_path: Path) -> torch.nn.Module:
        """Build DETR model from config.

        Args:
            config_path: Path to config file

        Returns:
            DETR model instance

        Raises:
            NotImplementedError: Not yet implemented
        """
        raise NotImplementedError(
            "DETR helper not yet implemented. "
            "Use HuggingFace models instead: mata.load('detect', 'facebook/detr-resnet-50')"
        )

    @staticmethod
    def load_checkpoint(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        """Load checkpoint into model.

        Args:
            model: Model instance
            state_dict: State dictionary
        """
        raise NotImplementedError("DETR helper not yet implemented")
