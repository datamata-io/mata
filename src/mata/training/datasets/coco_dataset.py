"""COCO-format training datasets for detection and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mata.eval.dataset import (
    _build_annotations_index,
    _extract_names_from_coco,
    _load_coco_json,
    _xywh_to_xyxy,
)
from mata.training.datasets.base import TrainingDataset, _ensure_torch


class COCODetectionDataset(TrainingDataset):
    """COCO-format detection dataset for training.

    Supports two construction modes:

    **Explicit paths**::

        dataset = COCODetectionDataset(
            root="/data/coco/train2017",
            annotation_file="/data/coco/annotations/instances_train2017.json",
        )

    **YAML config**::

        dataset = COCODetectionDataset("coco.yaml", split="train")

    YAML format (extended for training, backward-compatible with mata.val())::

        path: /data/coco
        train: train2017
        val: val2017
        train_annotations: annotations/instances_train2017.json
        val_annotations:   annotations/instances_val2017.json
        # Fall-back single key (backward compat):
        annotations: annotations/instances_val2017.json
        names:
          0: person
          1: bicycle

    Crowd annotations (``iscrowd=1``) are excluded from all splits.
    Images with zero annotations are supported and return empty tensors.

    Args:
        root:            Path to the images directory **or** a ``.yaml`` config file
                         (YAML mode is used automatically when ``annotation_file``
                         is ``None`` and the path has a ``.yaml``/``.yml`` suffix).
        annotation_file: Path to the COCO JSON annotation file.  Omit when using
                         YAML mode.
        split:           Dataset split to load — ``"train"``, ``"val"``, or
                         ``"test"``.  Only used in YAML mode.
        transforms:      Optional callable ``(image, target) → (image, target)``
                         applied after loading each sample.
    """

    def __init__(
        self,
        root: str | None = None,
        annotation_file: str | None = None,
        *,
        split: str = "train",
        transforms: Any | None = None,
    ) -> None:
        super().__init__(task="detect", transforms=transforms)
        self._split = split

        # Detect construction mode
        if root is not None and annotation_file is None and Path(root).suffix in {".yaml", ".yml"}:
            # YAML mode
            self._setup_from_yaml(root, split)
        elif root is not None and annotation_file is not None:
            # Explicit-paths mode
            self._images_dir = Path(root)
            self._annotations_path = Path(annotation_file)
            if not self._images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {self._images_dir.resolve()}")
            if not self._annotations_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {self._annotations_path.resolve()}")
        else:
            raise ValueError(
                "Provide either a YAML config path as 'root' (with no annotation_file), "
                "or both 'root' (images directory) and 'annotation_file'."
            )

        # Load and index COCO JSON eagerly
        coco_data = _load_coco_json(self._annotations_path)
        (
            self._images_by_id,
            self._anns_by_image,
            self._cat_id_to_label,
            self._has_segmentation,
        ) = _build_annotations_index(coco_data)
        self._coco_names: dict[int, str] = _extract_names_from_coco(coco_data)

        # Deterministically ordered image IDs
        self._image_ids: list[int] = sorted(self._images_by_id.keys())

    # ------------------------------------------------------------------
    # YAML config setup
    # ------------------------------------------------------------------

    def _setup_from_yaml(self, yaml_path: str, split: str) -> None:
        """Parse YAML config and resolve paths for the given split."""
        import yaml  # lazy — optional dep at top level

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path.resolve()}")
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        yaml_dir = path.parent.resolve()

        # Resolve dataset root
        root_str: str = cfg.get("path", "")
        root = (Path(root_str) if Path(root_str).is_absolute() else yaml_dir / root_str).resolve()

        # Resolve images subdirectory for the requested split
        images_subdir: str = cfg.get(split, "")
        images_dir = (root / images_subdir) if images_subdir else root
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found for split='{split}': {images_dir}")
        self._images_dir = images_dir

        # Resolve annotations: "{split}_annotations" then fallback to "annotations"
        ann_rel: str = cfg.get(f"{split}_annotations", "") or cfg.get("annotations", "")
        if not ann_rel:
            raise ValueError(
                f"YAML config must contain '{split}_annotations' or 'annotations' key " f"pointing to a COCO JSON file."
            )
        ann_path = root / ann_rel
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        self._annotations_path = ann_path

    # ------------------------------------------------------------------
    # TrainingDataset interface
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        return self._coco_names

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch = _ensure_torch()
        image_id = self._image_ids[index]
        image_info = self._images_by_id[image_id]

        # Load image
        image_path = self._images_dir / image_info["file_name"]
        image = self._load_image(image_path)

        # Filter crowd annotations
        anns = [a for a in self._anns_by_image.get(image_id, []) if not a.get("iscrowd", 0)]

        if anns:
            boxes = torch.tensor([_xywh_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32)
            labels = torch.tensor(
                [self._cat_id_to_label[a["category_id"]] for a in anns],
                dtype=torch.long,
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class COCOSegmentationDataset(COCODetectionDataset):
    """COCO-format segmentation dataset with per-instance binary mask loading.

    Extends :class:`COCODetectionDataset` to additionally return
    ``"masks": Tensor[N, H, W]`` (uint8 binary) in the target dict.

    Mask decoding strategy:

    1. **pycocotools** (preferred) — handles both RLE and polygon formats via
       ``pycocotools.mask.frPyObjects`` / ``decode``.
    2. **PIL fallback** — polygon-to-mask via ``PIL.ImageDraw`` when pycocotools
       is not installed.

    If an annotation has no segmentation field, a zero mask is returned for
    that instance.  If no annotations are present, ``masks`` is an empty
    ``Tensor[0, H, W]``.
    """

    def __init__(
        self,
        root: str | None = None,
        annotation_file: str | None = None,
        *,
        split: str = "train",
        transforms: Any | None = None,
    ) -> None:
        super().__init__(root=root, annotation_file=annotation_file, split=split, transforms=transforms)
        self._task = "segment"

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch = _ensure_torch()
        image_id = self._image_ids[index]
        image_info = self._images_by_id[image_id]

        # Load image
        image_path = self._images_dir / image_info["file_name"]
        image = self._load_image(image_path)
        img_w: int = image_info["width"]
        img_h: int = image_info["height"]

        # Filter crowd annotations
        anns = [a for a in self._anns_by_image.get(image_id, []) if not a.get("iscrowd", 0)]

        if anns:
            boxes = torch.tensor([_xywh_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32)
            labels = torch.tensor(
                [self._cat_id_to_label[a["category_id"]] for a in anns],
                dtype=torch.long,
            )
            masks_np = np.stack([self._decode_mask(a, img_h, img_w) for a in anns], axis=0)
            masks = torch.from_numpy(masks_np)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)

        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _decode_mask(self, ann: dict, height: int, width: int) -> np.ndarray:
        """Decode an annotation's segmentation field to a binary (H, W) uint8 mask.

        Tries pycocotools first; falls back to PIL polygon rasterisation.
        """
        seg = ann.get("segmentation")
        if not seg:
            return np.zeros((height, width), dtype=np.uint8)

        # ── pycocotools path (RLE or polygon) ──────────────────────────
        try:
            from pycocotools import mask as coco_mask  # type: ignore[import]

            rle = coco_mask.frPyObjects(seg, height, width)
            if isinstance(seg, list):
                # Polygon list → merge individual RLEs
                rle = coco_mask.merge(rle)
            return coco_mask.decode(rle).astype(np.uint8)
        except ImportError:
            pass

        # ── PIL polygon fallback ────────────────────────────────────────
        return self._polygon_to_mask(seg, height, width)

    @staticmethod
    def _polygon_to_mask(seg: Any, height: int, width: int) -> np.ndarray:
        """Rasterise polygon segmentation to a binary mask using PIL ImageDraw."""
        from PIL import Image as PILImage
        from PIL import ImageDraw

        mask = PILImage.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        if isinstance(seg, list):
            for polygon in seg:
                if len(polygon) >= 6:  # at least 3 (x, y) pairs
                    xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(xy, fill=1)
        return np.array(mask, dtype=np.uint8)
