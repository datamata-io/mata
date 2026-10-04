"""Training dataset loaders for COCO, Pascal VOC, and ImageFolder formats."""

from __future__ import annotations

from .coco_dataset import COCODetectionDataset, COCOSegmentationDataset
from .collators import (
    classification_collate_fn,
    detection_collate_fn,
    segmentation_collate_fn,
)
from .factory import DatasetFactory
from .imagefolder import ImageFolderDataset
from .voc_dataset import VOCDetectionDataset

__all__ = [
    "COCODetectionDataset",
    "COCOSegmentationDataset",
    "DatasetFactory",
    "ImageFolderDataset",
    "VOCDetectionDataset",
    "classification_collate_fn",
    "detection_collate_fn",
    "segmentation_collate_fn",
]
