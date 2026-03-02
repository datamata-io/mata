"""DatasetLoader — YAML config → COCO JSON ground truth + image paths.

Implemented in Task D1.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundTruth:
    """Ground truth annotations for a single image.

    Attributes:
        image_id:   COCO image ID.
        image_path: Absolute path to the image file on disk.
        boxes:      (N, 4) xyxy float32, absolute pixel coordinates.
        labels:     (N,) int32 class IDs (0-indexed).
        masks:      list of N masks (RLE dict / binary ndarray / polygon list) or
                    ``None`` for detection-only datasets.
        depth:      (H, W) float32 depth map or ``None``.
        image_size: ``(width, height)`` tuple.
        text:       list of N transcription strings or ``None``.
                    Populated only for OCR datasets.
    """

    image_id: int
    image_path: str
    boxes: np.ndarray  # (N, 4) xyxy float32
    labels: np.ndarray  # (N,) int32
    masks: list[Any] | None  # N masks or None
    depth: np.ndarray | None  # (H, W) float32 or None
    image_size: tuple[int, int]  # (width, height)
    text: list[str] | None = None  # 🆕 OCR transcriptions


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert COCO xywh bbox to xyxy format."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _load_coco_json(json_path: Path) -> dict:
    """Load and return a parsed COCO JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_annotations_index(
    coco_data: dict,
) -> tuple[dict[int, dict], dict[int, list], dict[int, int], bool]:
    """Build lookup structures from a parsed COCO JSON dict.

    Returns:
        images_by_id:    image_id → image dict (``file_name``, ``width``, ``height``)
        anns_by_image:   image_id → list of annotation dicts
        cat_id_to_label: COCO category ID (1-indexed) → 0-indexed class label
        has_segmentation: True if *any* annotation has a non-empty segmentation field
    """
    images_by_id: dict[int, dict] = {img["id"]: img for img in coco_data.get("images", [])}

    anns_by_image: dict[int, list] = {img_id: [] for img_id in images_by_id}
    has_segmentation = False

    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
        if ann.get("segmentation"):
            has_segmentation = True

    # Build 1-indexed category_id → 0-indexed label mapping
    categories = coco_data.get("categories", [])
    cat_id_to_label = {cat["id"]: idx for idx, cat in enumerate(categories)}

    return images_by_id, anns_by_image, cat_id_to_label, has_segmentation


def _extract_names_from_coco(coco_data: dict) -> dict[int, str]:
    """Return 0-indexed {label: name} mapping from COCO categories array."""
    return {idx: cat["name"] for idx, cat in enumerate(coco_data.get("categories", []))}


def _ground_truth_from_anns(
    image_info: dict,
    anns: list[dict],
    image_path: str,
    cat_id_to_label: dict[int, int],
    has_segmentation: bool,
    depth_map: np.ndarray | None,
) -> GroundTruth:
    """Build a :class:`GroundTruth` object from COCO annotation dicts."""
    boxes_list: list[list[float]] = []
    labels_list: list[int] = []
    masks_list: list[Any] = []
    text_list: list[str] = []
    has_text = False

    for ann in anns:
        label = cat_id_to_label.get(ann["category_id"])
        if label is None:
            logger.warning("Skipping annotation with unknown category_id=%d", ann["category_id"])
            continue
        boxes_list.append(_xywh_to_xyxy(ann["bbox"]))
        labels_list.append(label)

        if has_segmentation:
            seg = ann.get("segmentation")
            masks_list.append(seg if seg else None)

        if "text" in ann:
            text_list.append(ann["text"])
            has_text = True

    if boxes_list:
        boxes = np.array(boxes_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)

    masks: list[Any] | None = masks_list if has_segmentation else None

    return GroundTruth(
        image_id=image_info["id"],
        image_path=image_path,
        boxes=boxes,
        labels=labels,
        masks=masks,
        depth=depth_map,
        image_size=(image_info["width"], image_info["height"]),
        text=text_list if has_text else None,
    )


class DatasetLoader:
    """Loads image paths and ground-truth annotations for evaluation.

    Supports two construction modes:

    **YAML-driven** (full dataset config)::

        loader = DatasetLoader.from_yaml("coco.yaml")
        # or equivalently:
        loader = DatasetLoader(data="coco.yaml")

    **Standalone** (direct paths)::

        loader = DatasetLoader.from_coco_json("/data/coco/val2017",
                                               "instances_val2017.json")
        # or equivalently:
        loader = DatasetLoader(images="/data/coco/val2017",
                               annotations="instances_val2017.json")

    YAML format (MATA variant — COCO JSON instead of YOLO TXT labels)::

        path: /data/coco
        val:  val2017
        annotations: annotations/instances_val2017.json
        names:
          0: person
          1: bicycle
          # ... up to 79: toothbrush
        # Optional depth ground-truth directory (for depth tasks):
        # depth_gt: depth_maps/

    Args:
        data:        Path to a dataset YAML file **or** a pre-parsed ``dict``.
                     Mutually exclusive with ``images``/``annotations``.
        images:      Standalone mode: path to images directory or list of paths.
        annotations: Standalone mode: path to COCO JSON annotation file.
        split:       Which split to load — ``"val"`` (default), ``"train"``, or ``"test"``.
        task:        Task type — ``"detect"``, ``"segment"``, ``"classify"``,
                     ``"depth"``.  Affects which GT fields are populated.
    """

    def __init__(
        self,
        data: str | dict | None = None,
        images: str | list[str] | None = None,
        annotations: str | None = None,
        split: str = "val",
        task: str = "detect",
    ) -> None:
        self._split = split
        self._task = task

        # Internal state — populated by _setup()
        self._images_dir: Path | None = None
        self._annotations_path: Path | None = None
        self._depth_gt_dir: Path | None = None
        self._yaml_names: dict[int, str] = {}
        self._image_list: list[str] | None = None  # pre-computed image paths (standalone)

        # Parsed COCO JSON data — loaded lazily in _ensure_loaded()
        self._coco_data: dict | None = None
        self._images_by_id: dict[int, dict] = {}
        self._anns_by_image: dict[int, list] = {}
        self._cat_id_to_label: dict[int, int] = {}
        self._has_segmentation: bool = False
        self._coco_names: dict[int, str] = {}
        self._loaded: bool = False

        if data is not None and (images is not None or annotations is not None):
            raise ValueError("Provide either 'data' (YAML) or 'images'/'annotations', not both.")

        if data is not None:
            self._setup_from_yaml(data)
        elif images is not None or annotations is not None:
            self._setup_standalone(images, annotations)
        # else: empty loader — user must call _setup_* manually

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs: Any) -> DatasetLoader:
        """Construct a :class:`DatasetLoader` from a YAML dataset config.

        Args:
            yaml_path: Path to the ``.yaml`` dataset configuration file.
            **kwargs:  Forwarded to :meth:`__init__` (``split``, ``task``).
        """
        return cls(data=yaml_path, **kwargs)

    @classmethod
    def from_coco_json(cls, images_dir: str, json_path: str, **kwargs: Any) -> DatasetLoader:
        """Construct a :class:`DatasetLoader` from an images directory and COCO JSON.

        Args:
            images_dir: Path to the directory containing image files.
            json_path:  Path to the COCO annotation JSON file.
            **kwargs:   Forwarded to :meth:`__init__` (``split``, ``task``).
        """
        return cls(images=images_dir, annotations=json_path, **kwargs)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_from_yaml(self, data: str | dict) -> None:
        """Parse YAML config and resolve all paths."""
        if isinstance(data, dict):
            cfg = data
            yaml_dir = Path.cwd()
        else:
            yaml_path = Path(data)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Dataset YAML not found: {yaml_path.resolve()}")
            import yaml  # lazy import — optional heavy dep

            with yaml_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            yaml_dir = yaml_path.parent.resolve()

        # Resolve dataset root
        root_str: str = cfg.get("path", "")
        root = Path(root_str) if Path(root_str).is_absolute() else yaml_dir / root_str
        root = root.resolve()

        if root_str and not root.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root}")

        # Resolve images subdirectory for the requested split
        images_subdir: str = cfg.get(self._split, "")
        if images_subdir:
            images_dir = root / images_subdir
            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            self._images_dir = images_dir
        else:
            self._images_dir = root

        # Resolve annotations JSON
        ann_rel: str = cfg.get("annotations", "")
        if ann_rel:
            ann_path = root / ann_rel
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {ann_path}")
            self._annotations_path = ann_path
        else:
            raise ValueError("YAML config must contain an 'annotations' key pointing to a COCO JSON file.")

        # Optional depth GT directory
        depth_rel: str = cfg.get("depth_gt", "")
        if depth_rel:
            depth_dir = root / depth_rel
            if not depth_dir.exists():
                logger.warning("depth_gt directory not found: %s — depth GT disabled", depth_dir)
            else:
                self._depth_gt_dir = depth_dir

        # Class names from YAML (optional — COCO JSON categories used as fallback)
        names_cfg = cfg.get("names", {})
        if isinstance(names_cfg, dict):
            self._yaml_names = {int(k): str(v) for k, v in names_cfg.items()}
        elif isinstance(names_cfg, list):
            self._yaml_names = {i: str(n) for i, n in enumerate(names_cfg)}

    def _setup_standalone(
        self,
        images: str | list[str] | None,
        annotations: str | None,
    ) -> None:
        """Configure standalone mode from direct paths."""
        if isinstance(images, list):
            # Pre-computed list of image file paths
            self._image_list = images
        elif images is not None:
            images_path = Path(images)
            if not images_path.exists():
                raise FileNotFoundError(f"Images path not found: {images_path.resolve()}")
            self._images_dir = images_path.resolve()

        if annotations is not None:
            ann_path = Path(annotations)
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {ann_path.resolve()}")
            self._annotations_path = ann_path.resolve()

    # ------------------------------------------------------------------
    # Lazy COCO loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load and index the COCO JSON file (once)."""
        if self._loaded:
            return
        if self._annotations_path is None:
            raise RuntimeError(
                "No annotations path configured. " "Use DatasetLoader.from_yaml() or DatasetLoader.from_coco_json()."
            )
        self._coco_data = _load_coco_json(self._annotations_path)
        (
            self._images_by_id,
            self._anns_by_image,
            self._cat_id_to_label,
            self._has_segmentation,
        ) = _build_annotations_index(self._coco_data)
        self._coco_names = _extract_names_from_coco(self._coco_data)
        self._loaded = True

    # ------------------------------------------------------------------
    # Image catalogue helpers
    # ------------------------------------------------------------------

    _IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

    def _all_image_paths(self) -> list[str]:
        """Return sorted list of all image file paths for the configured split."""
        if self._image_list is not None:
            return self._image_list
        if self._images_dir is None:
            return []
        return sorted(str(p) for p in self._images_dir.iterdir() if p.suffix.lower() in self._IMAGE_EXTENSIONS)

    def _image_id_from_filename(self, filename: str) -> int | None:
        """Lookup COCO image_id by filename stem (without extension)."""
        stem = Path(filename).stem
        for img_id, img_info in self._images_by_id.items():
            if Path(img_info["file_name"]).stem == stem:
                return img_id
        return None

    def _depth_map_for(self, image_path: str) -> np.ndarray | None:
        """Load depth ground-truth ``.npy`` for *image_path* if available.

        Handles DIODE-style ``(H, W, 1)`` arrays by squeezing to ``(H, W)``.
        """
        if self._depth_gt_dir is None or self._task != "depth":
            return None
        stem = Path(image_path).stem
        npy_path = self._depth_gt_dir / f"{stem}.npy"
        if npy_path.exists():
            arr = np.load(str(npy_path)).astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            return arr
        logger.debug("No depth GT found for %s at %s", stem, npy_path)
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of images in the configured split."""
        self._ensure_loaded()
        images = self._all_image_paths()
        if images:
            return len(images)
        return len(self._images_by_id)

    def __iter__(self) -> Iterator[tuple[str, GroundTruth]]:
        """Yield ``(image_path, GroundTruth)`` tuples lazily."""
        self._ensure_loaded()

        image_paths = self._all_image_paths()

        if image_paths:
            # Iterate over real image files
            for img_path_str in image_paths:
                img_path = Path(img_path_str)
                img_id = self._image_id_from_filename(img_path.name)
                if img_id is None:
                    logger.debug("No COCO annotation found for image '%s' — skipping.", img_path.name)
                    continue
                img_info = self._images_by_id[img_id]
                anns = self._anns_by_image.get(img_id, [])
                depth_map = self._depth_map_for(img_path_str)
                gt = _ground_truth_from_anns(
                    img_info,
                    anns,
                    img_path_str,
                    self._cat_id_to_label,
                    self._has_segmentation,
                    depth_map,
                )
                yield img_path_str, gt
        else:
            # No local files — iterate over COCO image registry
            for img_id, img_info in self._images_by_id.items():
                img_path_str = img_info["file_name"]
                anns = self._anns_by_image.get(img_id, [])
                depth_map = self._depth_map_for(img_path_str)
                gt = _ground_truth_from_anns(
                    img_info,
                    anns,
                    img_path_str,
                    self._cat_id_to_label,
                    self._has_segmentation,
                    depth_map,
                )
                yield img_path_str, gt

    @property
    def names(self) -> dict[int, str]:
        """Class ID (0-indexed) → class name mapping.

        YAML ``names`` block takes precedence; falls back to COCO JSON ``categories``
        if YAML names were not provided.
        """
        if self._yaml_names:
            return self._yaml_names
        self._ensure_loaded()
        return self._coco_names

    @property
    def cat_id_to_label(self) -> dict[int, int]:
        """COCO category ID (1-indexed) → 0-indexed class label mapping.

        Use this to convert raw model output label IDs (which typically match
        COCO category IDs) to the 0-indexed label space used by :class:`GroundTruth`.
        """
        self._ensure_loaded()
        return dict(self._cat_id_to_label)

    # Keep backward-compat alias used by older code
    @property
    def class_names(self) -> dict[int, str]:
        """Alias for :attr:`names` (backward compatibility)."""
        return self.names
