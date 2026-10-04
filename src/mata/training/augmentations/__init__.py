"""Data augmentation pipelines for training."""

from __future__ import annotations

from .albumentations import (
    AlbumentationsWrapper,
)  # class always importable; raises ImportError on instantiation if albumentations not installed
from .basic import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BasicClassificationAugmentation,
    BasicDetectionAugmentation,
    BasicSegmentationAugmentation,
)
from .factory import AugmentationFactory

__all__ = [
    "BasicDetectionAugmentation",
    "BasicClassificationAugmentation",
    "BasicSegmentationAugmentation",
    "AlbumentationsWrapper",
    "AugmentationFactory",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
