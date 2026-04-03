"""Data augmentation pipelines for training."""

from __future__ import annotations

from .basic import (
    BasicClassificationAugmentation,
    BasicDetectionAugmentation,
    BasicSegmentationAugmentation,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .albumentations import AlbumentationsWrapper  # class always importable; raises ImportError on instantiation if albumentations not installed
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
