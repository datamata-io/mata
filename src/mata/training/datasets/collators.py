"""Task-specific DataLoader collation functions for MATA training."""

from __future__ import annotations

from typing import Any


def detection_collate_fn(
    batch: list[tuple[Any, dict]],
) -> tuple[list[Any], list[dict]]:
    """Collate detection samples — variable-size boxes/labels per image.

    Returns ``(list[image], list[target])`` for torchvision-style training.
    Images are NOT stacked because object detection models accept variable
    spatial dimensions.

    Args:
        batch: List of ``(image, target)`` tuples from a detection dataset.

    Returns:
        Tuple of ``(images, targets)`` as plain Python lists.
    """
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


def segmentation_collate_fn(
    batch: list[tuple[Any, dict]],
) -> tuple[list[Any], list[dict]]:
    """Collate segmentation samples — variable-size masks per image.

    Returns ``(list[image], list[target])`` for torchvision-style training.
    Images are NOT stacked because segmentation models accept variable
    spatial dimensions.

    Args:
        batch: List of ``(image, target)`` tuples from a segmentation dataset.

    Returns:
        Tuple of ``(images, targets)`` as plain Python lists.
    """
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


def classification_collate_fn(
    batch: list[tuple[Any, dict]],
) -> tuple[Any, Any]:
    """Collate classification samples — stack images and stack labels.

    Classification datasets always produce fixed-size images (after resize
    transforms), so images can be stacked into a single ``[N, C, H, W]``
    tensor.

    Args:
        batch: List of ``(image, target)`` tuples where ``target["label"]``
               is an integer class index.

    Returns:
        Tuple of ``(images_tensor, labels_tensor)`` where images has shape
        ``[N, C, H, W]`` and labels has shape ``[N]``.
    """
    import torch

    if not batch:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    images, targets = zip(*batch)
    images = torch.stack(list(images), dim=0)
    labels = torch.tensor([t["label"] for t in targets], dtype=torch.long)
    return images, labels
