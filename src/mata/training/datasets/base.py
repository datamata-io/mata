from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

# Lazy torch import
_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TrainingDataset(ABC):
    """Abstract base class for MATA training datasets.

    Subclasses must implement __getitem__ and __len__.

    Target format by task:
    - detect: {"boxes": Tensor[N,4] xyxy, "labels": Tensor[N], "image_id": int}
    - classify: {"label": int}
    - segment: {"boxes": Tensor[N,4], "labels": Tensor[N], "masks": Tensor[N,H,W]}
    """

    def __init__(self, task: str, transforms: Any | None = None) -> None:
        self.task = task
        self.transforms = transforms

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]: ...

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _load_image(self, path: str | Path) -> Image.Image:
        """Load image as RGB PIL Image."""
        return Image.open(path).convert("RGB")
