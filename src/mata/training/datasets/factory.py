"""Auto-detecting dataset factory for MATA training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError

from .collators import (
    classification_collate_fn,
    detection_collate_fn,
    segmentation_collate_fn,
)

# Mapping from task to its canonical collation function
_COLLATE_FNS: dict[str, Callable] = {
    "detect": detection_collate_fn,
    "segment": segmentation_collate_fn,
    "classify": classification_collate_fn,
}

_SUPPORTED_TASKS = frozenset(_COLLATE_FNS.keys())


class DatasetFactory:
    """Auto-detect dataset format and build the correct dataset class.

    ``DatasetFactory.create()`` inspects the *data* argument and returns a
    ``(dataset, collate_fn)`` tuple ready for use with
    ``torch.utils.data.DataLoader``.

    **Detection order**:

    1. ``torch.utils.data.Dataset`` instance → pass through unchanged.
    2. Path ending in ``.yaml`` / ``.yml`` → parse YAML; if it contains any
       of the keys ``annotations``, ``train_annotations``, or
       ``val_annotations`` it is treated as a COCO config
       (``COCODetectionDataset`` / ``COCOSegmentationDataset``); otherwise
       a ``TrainingError`` is raised.
    3. Directory that contains an ``Annotations/`` sub-directory with
       ``*.xml`` files → ``VOCDetectionDataset`` (detect only).
    4. Directory that contains at least one ``*.json`` file directly inside
       → ``COCODetectionDataset`` / ``COCOSegmentationDataset``.  The first
       JSON file found is used as the annotation file.
    5. Directory whose immediate children are *all subdirectories* (no
       regular files at depth 0) → ``ImageFolderDataset`` (classify only).
    6. Otherwise → ``TrainingError`` with format suggestions.
    """

    @staticmethod
    def create(
        task: str,
        data: Any,
        split: str = "train",
        transforms: Any | None = None,
    ) -> tuple[Any, Callable]:
        """Create a dataset and its corresponding collate function.

        Args:
            task:       Task type — ``"detect"``, ``"segment"``, or
                        ``"classify"``.
            data:       One of:

                        - A string / :class:`~pathlib.Path` pointing to a
                          YAML config file, a directory, or a COCO JSON file.
                        - A ``torch.utils.data.Dataset`` instance to pass
                          through unchanged.

            split:      Split to load for YAML/COCO sources
                        (``"train"``, ``"val"``, ``"test"``).
            transforms: Optional callable applied to ``(image, target)``
                        pairs after loading.

        Returns:
            ``(dataset, collate_fn)`` tuple.

        Raises:
            :class:`~mata.core.exceptions.TrainingError`:
                When the data format cannot be detected or is incompatible
                with the requested task.
        """
        if task not in _SUPPORTED_TASKS:
            raise TrainingError(f"Unsupported task '{task}'. " f"DatasetFactory supports: {sorted(_SUPPORTED_TASKS)}.")

        collate_fn = _COLLATE_FNS[task]

        # ------------------------------------------------------------------
        # 1. torch.utils.data.Dataset pass-through
        # ------------------------------------------------------------------
        try:
            import torch.utils.data as _tud

            if isinstance(data, _tud.Dataset):
                return data, collate_fn
        except ImportError:
            pass  # torch not installed — can't be a Dataset, continue

        # ------------------------------------------------------------------
        # 2-6. Path-based detection
        # ------------------------------------------------------------------
        path = Path(data) if not isinstance(data, Path) else data

        # --- YAML config ---
        if path.suffix.lower() in {".yaml", ".yml"}:
            dataset = DatasetFactory._from_yaml(task, path, split, transforms)
            return dataset, collate_fn

        if not path.exists():
            raise TrainingError(
                f"Data path does not exist: '{path}'. " "Provide a valid YAML config, directory, or COCO JSON file."
            )

        if path.is_file() and path.suffix.lower() == ".json":
            # Explicit COCO JSON file without a separate images dir is not
            # yet supported — tell the user what to do.
            raise TrainingError(
                f"Received a bare JSON file '{path}'. "
                "For COCO datasets supply a directory that contains '*.json' "
                "annotations, or use a YAML config file."
            )

        if path.is_dir():
            dataset = DatasetFactory._from_directory(task, path, split, transforms)
            return dataset, collate_fn

        raise TrainingError(
            f"Cannot determine dataset format from '{path}'. "
            "Expected: a YAML config file, a directory with images/annotations, "
            "or a torch.utils.data.Dataset instance."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_yaml(task: str, yaml_path: Path, split: str, transforms: Any | None) -> Any:
        """Construct a dataset from a YAML config file."""
        import yaml  # lazy

        if not yaml_path.exists():
            raise TrainingError(f"Dataset YAML not found: '{yaml_path.resolve()}'.")

        with yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        coco_keys = {"annotations", "train_annotations", "val_annotations", "test_annotations"}
        is_coco = bool(coco_keys & set(cfg.keys()))

        if is_coco:
            return DatasetFactory._make_coco(task, str(yaml_path), split, transforms)

        raise TrainingError(
            f"Cannot determine dataset format from YAML '{yaml_path}'. "
            "For COCO datasets include an 'annotations' or "
            "'train_annotations'/'val_annotations' key. "
            "For ImageFolder datasets pass the root directory directly."
        )

    @staticmethod
    def _from_directory(task: str, directory: Path, split: str, transforms: Any | None) -> Any:
        """Auto-detect dataset type from directory layout."""

        # --- VOC: has Annotations/ sub-dir with *.xml files ---
        annotations_dir = directory / "Annotations"
        if annotations_dir.is_dir() and any(annotations_dir.glob("*.xml")):
            if task not in {"detect"}:
                raise TrainingError(
                    f"VOC dataset format detected but task is '{task}'. "
                    "VOCDetectionDataset only supports task='detect'."
                )
            from .voc_dataset import VOCDetectionDataset

            return VOCDetectionDataset(
                root=str(directory),
                image_set=split if split in {"train", "val", "trainval", "test"} else "trainval",
                transforms=transforms,
            )

        # --- COCO: has *.json annotation files at directory root ---
        json_files = sorted(directory.glob("*.json"))
        if json_files:
            # Use images dir as root and the first JSON as annotation file
            annotation_file = json_files[0]
            return DatasetFactory._make_coco(
                task,
                str(directory),
                split,
                transforms,
                annotation_file=str(annotation_file),
            )

        # --- ImageFolder: immediate children are all subdirectories ---
        children = [c for c in directory.iterdir() if not c.name.startswith(".")]
        child_dirs = [c for c in children if c.is_dir()]
        child_files = [c for c in children if c.is_file()]

        if child_dirs and not child_files:
            if task != "classify":
                raise TrainingError(
                    f"ImageFolder format detected (directory of class sub-directories), "
                    f"but task is '{task}'. ImageFolderDataset only supports task='classify'. "
                    f"For detection/segmentation provide COCO or VOC annotations."
                )
            from .imagefolder import ImageFolderDataset

            return ImageFolderDataset(root=str(directory), transforms=transforms)

        raise TrainingError(
            f"Cannot determine dataset format from directory '{directory}'. "
            "Expected one of:\n"
            "  - Annotations/*.xml   → VOC detection dataset\n"
            "  - *.json              → COCO detection/segmentation dataset\n"
            "  - subdirectories only → ImageFolder classification dataset\n"
            "Or use a YAML config file pointing to your dataset."
        )

    @staticmethod
    def _make_coco(
        task: str,
        root_or_yaml: str,
        split: str,
        transforms: Any | None,
        annotation_file: str | None = None,
    ) -> Any:
        """Instantiate the right COCO dataset class for the given task."""
        from .coco_dataset import COCODetectionDataset, COCOSegmentationDataset

        if task == "segment":
            cls = COCOSegmentationDataset
        elif task == "detect":
            cls = COCODetectionDataset
        else:
            raise TrainingError(
                f"COCO dataset format is not supported for task='{task}'. "
                "COCO datasets support 'detect' and 'segment' tasks."
            )

        if annotation_file is not None:
            # Explicit-paths mode: root_or_yaml is the images directory
            return cls(
                root=root_or_yaml,
                annotation_file=annotation_file,
                transforms=transforms,
            )
        # YAML mode: root_or_yaml is the YAML file path
        return cls(root=root_or_yaml, split=split, transforms=transforms)
