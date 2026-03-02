"""RT-DETR architecture helper for PyTorch checkpoint loading.

Will be implemented in a future release.
"""

from pathlib import Path

import torch


class RTDETRHelper:
    """Helper for RT-DETR model reconstruction."""

    @staticmethod
    def build_model(config_path: Path) -> torch.nn.Module:
        """Build RT-DETR model from config.

        Args:
            config_path: Path to config file

        Returns:
            RT-DETR model instance

        Raises:
            NotImplementedError: Not yet implemented
        """
        raise NotImplementedError(
            "RT-DETR helper not yet implemented. "
            "Use HuggingFace models instead: mata.load('detect', 'PekingU/rtdetr_v2_r18vd')"
        )

    @staticmethod
    def load_checkpoint(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        """Load checkpoint into model.

        Args:
            model: Model instance
            state_dict: State dictionary
        """
        raise NotImplementedError("RT-DETR helper not yet implemented")
