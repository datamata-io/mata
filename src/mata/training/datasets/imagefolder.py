from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError
from mata.training.datasets.base import IMAGE_EXTENSIONS, TrainingDataset


class ImageFolderDataset(TrainingDataset):
    """Classification dataset that auto-discovers classes from directory structure.

    Expected layout::

        root/
            cat/
                image1.jpg
                image2.png
            dog/
                image3.jpg

    Class names are sorted alphabetically for deterministic label mapping.
    Hidden files and directories (starting with ``.``) are skipped.
    Non-image files are skipped silently.

    Args:
        root: Path to the root directory containing class subdirectories.
        transforms: Optional callable applied to ``(image, target)`` pairs.
    """

    def __init__(self, root: str | Path, transforms: Any | None = None) -> None:
        super().__init__(task="classify", transforms=transforms)
        self._root = Path(root)
        self._samples: list[tuple[Path, int]] = []
        self._class_names: dict[int, str] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Scan root for class subdirectories and collect image paths."""
        class_dirs = sorted(d for d in self._root.iterdir() if d.is_dir() and not d.name.startswith("."))

        if not class_dirs:
            raise TrainingError(
                f"No class subdirectories found in '{self._root}'. "
                "ImageFolderDataset requires at least one subdirectory "
                "(e.g. root/class_name/image.jpg)."
            )

        self._class_names = {idx: d.name for idx, d in enumerate(class_dirs)}

        for label, class_dir in enumerate(class_dirs):
            for file in sorted(class_dir.iterdir()):
                if file.is_file() and not file.name.startswith(".") and file.suffix.lower() in IMAGE_EXTENSIONS:
                    self._samples.append((file, label))

    # ------------------------------------------------------------------
    # TrainingDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        path, label = self._samples[index]
        image = self._load_image(path)
        target: dict[str, Any] = {"label": label}
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names
