"""COCO annotation utilities for MATA smoke tests.

Loads COCO val2017 annotations and provides helpers for iterating images
and fetching ground-truth data for lightweight stability comparisons.

Requires pycocotools (already a core MATA dependency).

Setup — download COCO val2017:
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip && unzip annotations_trainval2017.zip
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def load_coco(annotations_path: str | Path):
    """Load a COCO annotation file and return a pycocotools COCO object.

    Args:
        annotations_path: Path to instances_val2017.json (or any COCO JSON).

    Returns:
        pycocotools.coco.COCO instance.

    Raises:
        ImportError: If pycocotools is not installed.
        FileNotFoundError: If the annotations file does not exist.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO smoke tests. "
            "Install with: pip install pycocotools"
        ) from exc

    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    logger.info("Loading COCO annotations from %s ...", annotations_path)
    coco = COCO(str(annotations_path))
    logger.info(
        "Loaded %d images, %d annotations, %d categories.",
        len(coco.imgs),
        len(coco.anns),
        len(coco.cats),
    )
    return coco


def get_category_name_map(coco) -> dict[int, str]:
    """Return a dict mapping COCO category_id → category name.

    Uses the annotation file's own metadata, so no hardcoding needed.
    COCO category IDs are non-contiguous (1–90 with gaps).

    Args:
        coco: pycocotools.coco.COCO instance.

    Returns:
        e.g. {1: "person", 2: "bicycle", ..., 90: "toothbrush"}
    """
    return {cat_id: info["name"] for cat_id, info in coco.cats.items()}


# ---------------------------------------------------------------------------
# Per-image ground-truth
# ---------------------------------------------------------------------------


def get_image_ground_truth(coco, image_id: int, image_dir: Path) -> dict[str, Any]:
    """Return ground-truth data for a single COCO image.

    Args:
        coco: pycocotools.coco.COCO instance.
        image_id: COCO image ID.
        image_dir: Directory containing the image files.

    Returns:
        Dict with keys:
          - image_path  (Path)          absolute path to the image file
          - image_id    (int)           COCO image ID
          - width       (int)           image width from COCO metadata
          - height      (int)           image height from COCO metadata
          - category_ids (list[int])    unique COCO category IDs present
          - category_names (set[str])   unique category names present
          - bboxes_xyxy (list[tuple])   GT bboxes in xyxy pixel coords
          - instance_count (int)        number of annotated instances
          - ann_ids     (list[int])     raw annotation IDs for reference
    """
    img_info = coco.imgs[image_id]
    image_path = image_dir / img_info["file_name"]

    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    cat_name_map = get_category_name_map(coco)

    category_ids: list[int] = []
    category_names: set[str] = set()
    bboxes_xyxy: list[tuple[float, float, float, float]] = []

    for ann in anns:
        cat_id = ann["category_id"]
        category_ids.append(cat_id)
        category_names.add(cat_name_map.get(cat_id, str(cat_id)))

        # COCO bbox is [x, y, w, h] — convert to xyxy
        x, y, w, h = ann["bbox"]
        bboxes_xyxy.append((x, y, x + w, y + h))

    return {
        "image_path": image_path,
        "image_id": image_id,
        "width": img_info["width"],
        "height": img_info["height"],
        "category_ids": category_ids,
        "category_names": category_names,
        "bboxes_xyxy": bboxes_xyxy,
        "instance_count": len(anns),
        "ann_ids": ann_ids,
    }


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------


def iter_images(
    coco,
    image_dir: str | Path,
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Iterate over COCO images, yielding ground-truth dicts.

    Images whose file does not exist on disk are skipped with a warning.

    Args:
        coco: pycocotools.coco.COCO instance.
        image_dir: Directory containing COCO image files.
        limit: If set, stop after this many images.
        shuffle: If True, randomise iteration order (useful for sampling).

    Yields:
        Ground-truth dicts from :func:`get_image_ground_truth`.
    """
    image_dir = Path(image_dir)
    image_ids: list[int] = list(coco.imgs.keys())

    if shuffle:
        import random
        random.shuffle(image_ids)

    count = 0
    for image_id in image_ids:
        if limit is not None and count >= limit:
            break

        gt = get_image_ground_truth(coco, image_id, image_dir)
        if not gt["image_path"].exists():
            logger.warning("Image file not found, skipping: %s", gt["image_path"])
            continue

        yield gt
        count += 1


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compute_label_overlap(
    predicted_names: set[str],
    gt_names: set[str],
) -> dict[str, Any]:
    """Lightweight label-set comparison (stability proxy, not mAP).

    Args:
        predicted_names: Set of predicted category names (label_name values).
        gt_names: Set of GT category names for this image.

    Returns:
        Dict with keys:
          - matched      (set[str])  labels found in both prediction and GT
          - missed       (set[str])  GT labels absent from predictions
          - extra        (set[str])  predicted labels not in GT
          - overlap_ratio (float)   |matched| / |gt|, or 1.0 if gt is empty
    """
    matched = predicted_names & gt_names
    missed = gt_names - predicted_names
    extra = predicted_names - gt_names
    overlap_ratio = len(matched) / len(gt_names) if gt_names else 1.0

    return {
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "overlap_ratio": overlap_ratio,
    }
