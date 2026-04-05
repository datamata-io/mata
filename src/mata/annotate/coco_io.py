from __future__ import annotations

"""COCO JSON annotation I/O — read, write, and mutate COCO annotation files.

Schema reference (matches ``scripts/generate_coco_mini.py`` output):

    {
      "info":        {"description": str, "version": str},
      "licenses":    [],
      "images":      [{"id": int, "file_name": str, "width": int, "height": int}],
      "annotations": [{"id": int, "image_id": int, "category_id": int,
                        "bbox": [x, y, w, h],  ← xywh (COCO standard)
                        "area": float, "iscrowd": 0,
                        "segmentation": [[x1,y1,...]]}],
      "categories":  [{"id": int, "name": str, "supercategory": str}]
    }

All writes are atomic via ``tempfile`` + ``os.replace()``.
Category IDs are **1-indexed** (COCO standard).
Annotation IDs auto-increment from the current maximum.
"""

import json
import os
import random
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)

# Required top-level keys for a valid COCO dict
_REQUIRED_KEYS = frozenset({"images", "annotations", "categories"})


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_annotations(json_path: str | Path) -> dict:
    """Load a COCO JSON file and validate its top-level schema.

    Raises ``FileNotFoundError`` if the file is absent, ``ValueError`` if
    required keys are missing.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    missing = _REQUIRED_KEYS - set(coco.keys())
    if missing:
        raise ValueError(
            f"COCO JSON missing required keys: {sorted(missing)} in {path}"
        )

    return coco


def save_annotations(coco_dict: dict, json_path: str | Path) -> None:
    """Atomically write *coco_dict* to *json_path*.

    Writes to a sibling temporary file first, then renames it over the
    target so a mid-write crash cannot corrupt an existing valid file.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(coco_dict, indent=2, ensure_ascii=False)
    # Write to a temp file in the same directory (same filesystem → rename is atomic)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up orphaned temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug("Saved COCO annotations → %s", path)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def create_empty_coco(
    images: list[dict] | None = None,
    categories: list[dict] | None = None,
) -> dict:
    """Return an empty COCO dict with the canonical top-level structure."""
    return {
        "info": {"description": "MATA annotate dataset", "version": "1.0"},
        "licenses": [],
        "images": list(images) if images else [],
        "annotations": [],
        "categories": list(categories) if categories else [],
    }


# ---------------------------------------------------------------------------
# Annotation CRUD
# ---------------------------------------------------------------------------

def add_annotation(
    coco: dict,
    image_id: int,
    bbox_xywh: list[float],
    category_id: int,
    segmentation: list | None = None,
) -> int:
    """Add a new annotation to *coco* and return its auto-generated ID.

    The ID is ``max(existing_ids) + 1``, or ``1`` if there are none.
    ``area`` is computed from ``bbox_xywh[2] * bbox_xywh[3]``.
    """
    annotations = coco.setdefault("annotations", [])
    existing_ids = [a["id"] for a in annotations if isinstance(a.get("id"), int)]
    new_id = (max(existing_ids) + 1) if existing_ids else 1

    x, y, w, h = bbox_xywh
    ann: dict[str, Any] = {
        "id": new_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(w), float(h)],
        "area": float(w * h),
        "iscrowd": 0,
        "segmentation": segmentation if segmentation is not None else [],
    }
    annotations.append(ann)
    return new_id


def update_annotation(coco: dict, ann_id: int, **fields: Any) -> None:
    """Update arbitrary fields of the annotation with *ann_id*.

    Raises ``KeyError`` if the annotation does not exist.
    Protected fields (``id``, ``image_id``) are silently ignored to prevent
    accidental identity corruption.
    """
    _PROTECTED = {"id", "image_id"}
    for ann in coco.get("annotations", []):
        if ann["id"] == ann_id:
            for k, v in fields.items():
                if k not in _PROTECTED:
                    ann[k] = v
            return
    raise KeyError(f"Annotation id={ann_id} not found.")


def remove_annotation(coco: dict, ann_id: int) -> None:
    """Remove the annotation with *ann_id* from *coco*.

    Raises ``KeyError`` if the annotation is not found.
    """
    annotations = coco.get("annotations", [])
    new_list = [a for a in annotations if a["id"] != ann_id]
    if len(new_list) == len(annotations):
        raise KeyError(f"Annotation id={ann_id} not found.")
    coco["annotations"] = new_list


# ---------------------------------------------------------------------------
# Image / category helpers
# ---------------------------------------------------------------------------

def add_image(
    coco: dict,
    file_name: str,
    width: int,
    height: int,
) -> int:
    """Register an image entry in *coco* and return its auto-generated ID.

    The ID is ``max(existing_ids) + 1``, or ``1`` if there are none.
    """
    images = coco.setdefault("images", [])
    existing_ids = [i["id"] for i in images if isinstance(i.get("id"), int)]
    new_id = (max(existing_ids) + 1) if existing_ids else 1

    images.append({
        "id": new_id,
        "file_name": file_name,
        "width": int(width),
        "height": int(height),
    })
    return new_id


def add_category(
    coco: dict,
    name: str,
    supercategory: str | None = None,
    color: str | None = None,
) -> int:
    """Add a category to *coco* and return its auto-generated **1-indexed** ID.

    If a category with *name* already exists its ID is returned without
    adding a duplicate.
    """
    categories = coco.setdefault("categories", [])

    # Return existing ID if name already present
    for cat in categories:
        if cat["name"] == name:
            return cat["id"]

    existing_ids = [c["id"] for c in categories if isinstance(c.get("id"), int)]
    # 1-indexed: start at 1, or max+1
    new_id = (max(existing_ids) + 1) if existing_ids else 1

    entry: dict[str, Any] = {
        "id": new_id,
        "name": name,
        "supercategory": supercategory or name,
    }
    if color is not None:
        entry["color"] = color
    categories.append(entry)
    return new_id


def set_image_reviewed(coco: dict, image_filename: str, reviewed: bool) -> bool:
    """Set the *reviewed* flag on the image record matching *image_filename*.

    Matches on the full ``file_name`` or just the basename so that images
    stored with path prefixes (``train/image.jpg``) can be found by basename.

    Returns ``True`` if the image was found and updated, ``False`` otherwise.
    """
    base = image_filename.split("/")[-1]
    for img in coco.get("images", []):
        fn = img.get("file_name", "")
        if fn == image_filename or fn.split("/")[-1] == base:
            img["reviewed"] = bool(reviewed)
            return True
    return False


def update_category(
    coco: dict,
    cat_id: int,
    *,
    name: str | None = None,
    color: str | None = None,
    supercategory: str | None = None,
) -> dict:
    """Update mutable fields on the category with *cat_id*.

    Supports renaming (*name*), display colour (*color* hex string), and
    *supercategory*.  The protected ``id`` field is never changed.

    Returns the updated category dict.
    Raises ``KeyError`` if the category does not exist.
    """
    for cat in coco.get("categories", []):
        if cat["id"] == cat_id:
            if name is not None:
                cat["name"] = name
            if color is not None:
                cat["color"] = color
            if supercategory is not None:
                cat["supercategory"] = supercategory
            return cat
    raise KeyError(f"Category id={cat_id} not found.")


def delete_category(coco: dict, cat_id: int, reassign_to: int | None = None) -> int:
    """Remove the category with *cat_id* from *coco*.

    If *reassign_to* is given, annotations that reference *cat_id* are
    reassigned to that category rather than deleted.  Otherwise those
    annotations are removed.

    Returns the count of affected annotations (reassigned or deleted).
    Raises ``KeyError`` if *cat_id* is not found.
    Raises ``ValueError`` if *reassign_to* references a non-existent category.
    """
    existing_ids = {c["id"] for c in coco.get("categories", [])}
    if cat_id not in existing_ids:
        raise KeyError(f"Category id={cat_id} not found.")
    if reassign_to is not None and reassign_to not in existing_ids:
        raise ValueError(f"Reassign target category id={reassign_to} not found.")

    coco["categories"] = [c for c in coco.get("categories", []) if c["id"] != cat_id]

    count = 0
    if reassign_to is not None:
        for ann in coco.get("annotations", []):
            if ann.get("category_id") == cat_id:
                ann["category_id"] = reassign_to
                count += 1
    else:
        before = len(coco.get("annotations", []))
        coco["annotations"] = [
            a for a in coco.get("annotations", []) if a.get("category_id") != cat_id
        ]
        count = before - len(coco["annotations"])

    return count


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def xyxy_to_xywh(bbox: list[float]) -> list[float]:
    """Convert ``[x1, y1, x2, y2]`` → ``[x, y, w, h]`` (COCO storage format)."""
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert ``[x, y, w, h]`` → ``[x1, y1, x2, y2]`` (MATA adapter format)."""
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_coco(coco: dict) -> list[str]:
    """Return a list of human-readable warning strings for a COCO dict.

    Checks include:
    - Missing top-level keys
    - Duplicate image / annotation / category IDs
    - Annotations referencing unknown image IDs (orphans)
    - Annotations referencing unknown category IDs
    - Category IDs that are not 1-indexed (start at 0)
    - Annotations with non-positive bbox dimensions
    """
    warnings: list[str] = []

    # Top-level key presence
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            warnings.append(f"Missing top-level key: '{key}'")

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_ids = {img["id"] for img in images if "id" in img}
    category_ids = {cat["id"] for cat in categories if "id" in cat}

    # Duplicate image IDs
    seen_img: set[int] = set()
    for img in images:
        iid = img.get("id")
        if iid in seen_img:
            warnings.append(f"Duplicate image id: {iid}")
        seen_img.add(iid)

    # Duplicate annotation IDs
    seen_ann: set[int] = set()
    for ann in annotations:
        aid = ann.get("id")
        if aid in seen_ann:
            warnings.append(f"Duplicate annotation id: {aid}")
        seen_ann.add(aid)

    # Duplicate category IDs
    seen_cat: set[int] = set()
    for cat in categories:
        cid = cat.get("id")
        if cid in seen_cat:
            warnings.append(f"Duplicate category id: {cid}")
        seen_cat.add(cid)

    # Category IDs should be ≥ 1 (1-indexed COCO standard)
    for cat in categories:
        if cat.get("id", 1) == 0:
            warnings.append(
                f"Category '{cat.get('name')}' has id=0; COCO category IDs are 1-indexed."
            )

    # Orphan annotations (image_id not in images)
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is not None and img_id not in image_ids:
            warnings.append(
                f"Annotation id={ann.get('id')} references unknown image_id={img_id}."
            )
        cat_id = ann.get("category_id")
        if cat_id is not None and cat_id not in category_ids:
            warnings.append(
                f"Annotation id={ann.get('id')} references unknown category_id={cat_id}."
            )

    # Non-positive bbox dimensions
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            _, _, w, h = bbox
            if w <= 0 or h <= 0:
                warnings.append(
                    f"Annotation id={ann.get('id')} has non-positive bbox dimensions "
                    f"(w={w}, h={h})."
                )

    return warnings


# ---------------------------------------------------------------------------
# Dataset export helpers
# ---------------------------------------------------------------------------

def generate_yaml_config(
    dataset_path: str | Path,
    class_names: list[str],
    train_dir: str = "train",
    val_dir: str = "val",
    train_annotations: str = "annotations/instances_train.json",
    val_annotations: str = "annotations/instances_val.json",
) -> Path:
    """Write a training-ready ``dataset.yaml`` and return its path."""
    import yaml  # lazy import

    root = Path(dataset_path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    payload = {
        "path": str(root),
        "train": train_dir,
        "val": val_dir,
        "train_annotations": train_annotations,
        "val_annotations": val_annotations,
        "names": {index: name for index, name in enumerate(class_names)},
    }

    yaml_path = root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)

    return yaml_path


def split_dataset(
    coco: dict,
    ratio: float = 0.8,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Split a COCO dict into deterministic train/val subsets by image."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("Split ratio must be between 0 and 1 (exclusive).")

    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    if not images:
        empty = _make_coco_subset(coco, set())
        return empty, empty

    annotations_by_image: dict[int, list[dict]] = {img["id"]: [] for img in images}
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(ann)

    grouped_images: dict[tuple[int, ...], list[dict]] = {}
    for image in images:
        image_id = image["id"]
        category_signature = tuple(
            sorted({int(ann["category_id"]) for ann in annotations_by_image.get(image_id, [])})
        )
        grouped_images.setdefault(category_signature, []).append(image)

    rng = random.Random(seed)
    ordered_groups: list[tuple[tuple[int, ...], list[dict]]] = []
    for signature in sorted(grouped_images.keys(), key=lambda item: (len(item), item)):
        group = sorted(
            grouped_images[signature],
            key=lambda image: (str(image.get("file_name", "")), int(image.get("id", 0))),
        )
        rng.shuffle(group)
        ordered_groups.append((signature, group))

    target_train = int(round(len(images) * ratio))
    base_counts: list[int] = []
    remainders: list[tuple[float, int]] = []
    allocated = 0

    for index, (_, group) in enumerate(ordered_groups):
        exact = len(group) * ratio
        count = int(exact)
        base_counts.append(count)
        remainders.append((exact - count, index))
        allocated += count

    remaining = target_train - allocated
    for _, group_index in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if base_counts[group_index] < len(ordered_groups[group_index][1]):
            base_counts[group_index] += 1
            remaining -= 1

    train_image_ids: set[int] = set()
    for count, (_, group) in zip(base_counts, ordered_groups):
        train_image_ids.update(image["id"] for image in group[:count])

    val_image_ids = {image["id"] for image in images} - train_image_ids

    return _make_coco_subset(coco, train_image_ids), _make_coco_subset(coco, val_image_ids)


def _strip_split_prefix_from_coco(coco_subset: dict, split: str) -> None:
    """Strip the leading ``"<split>/"`` prefix from every ``file_name`` in *coco_subset*.

    ``COCODetectionDataset`` constructs image paths as ``images_dir / file_name``
    where ``images_dir`` is already the split subdirectory, so ``file_name`` values
    must be relative to that directory (no leading split prefix).
    """
    prefix = f"{split}/"
    for img in coco_subset.get("images", []):
        fn = img.get("file_name", "")
        if fn.startswith(prefix):
            img["file_name"] = fn[len(prefix):]


def _detect_split_from_path(file_name: str) -> str | None:
    """Return 'train', 'val', or 'test' if *file_name* has a matching path prefix."""
    for part in (p.lower() for p in Path(file_name).parts[:-1]):
        if "train" in part:
            return "train"
        if "val" in part:
            return "val"
        if "test" in part:
            return "test"
    return None


def _detect_split_from_fs(dataset_root: Path, file_name: str) -> str | None:
    """Check which split subdirectory contains the image file."""
    base = Path(file_name).name
    for split in ("train", "val", "test"):
        if (dataset_root / split / base).is_file():
            return split
    return None


def export_dataset(
    dataset_path: str | Path,
    coco: dict,
    class_names: list[str],
    split_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[Path, list[str]]:
    """Write ``dataset.yaml`` and split annotation JSONs for training.

    Detects the existing train/val split from ``file_name`` path prefixes or
    the filesystem and writes only the annotation metadata files — **images
    are never copied or deleted**.

    Raises:
        ValueError: If no split structure exists on disk.  The caller should
            direct the user to run Redistribute first.

    Returns ``(yaml_path, unassigned)`` where *unassigned* lists any
    ``file_name`` values that could not be assigned to train or val.
    """
    root = Path(dataset_path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Require at least one split directory to exist.
    has_splits = any((root / s).is_dir() for s in ("train", "val", "test"))
    if not has_splits:
        raise ValueError(
            "No split structure found (train/, val/, or test/ directories). "
            "Use Redistribute to organise images into splits first."
        )

    # Partition images by existing split (path prefix first, then FS lookup).
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    unassigned: list[str] = []

    for img in coco.get("images", []):
        img_id = img.get("id")
        file_name = img.get("file_name", "")
        split = _detect_split_from_path(file_name)
        if split is None:
            split = _detect_split_from_fs(root, file_name)
        if split == "train":
            train_ids.add(img_id)
        elif split in ("val", "valid"):
            val_ids.add(img_id)
        else:
            unassigned.append(file_name)

    train_coco = _make_coco_subset(coco, train_ids)
    val_coco = _make_coco_subset(coco, val_ids)

    # Strip the split-dir prefix so COCODetectionDataset (which prepends the
    # split directory itself) resolves to the correct image path.
    _strip_split_prefix_from_coco(train_coco, "train")
    for _v in ("val", "valid"):
        _strip_split_prefix_from_coco(val_coco, _v)
    _strip_split_prefix_from_coco(val_coco, "test")

    # Stage annotation JSONs so a write error never corrupts existing files.
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(dir=root, prefix=".export_tmp_"))
    try:
        tmp_annotations = tmp_dir / "annotations"
        tmp_annotations.mkdir(parents=True, exist_ok=True)
        save_annotations(train_coco, tmp_annotations / "instances_train.json")
        save_annotations(val_coco, tmp_annotations / "instances_val.json")

        annotations_dir = root / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("instances_train.json", "instances_val.json"):
            staged = tmp_annotations / fname
            if staged.exists():
                shutil.copy2(str(staged), str(annotations_dir / fname))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return generate_yaml_config(root, class_names), unassigned


def _make_coco_subset(coco: dict, image_ids: set[int]) -> dict:
    """Return a COCO dict subset containing only the requested image IDs."""
    subset = {
        "info": deepcopy(coco.get("info", {"description": "MATA annotate dataset", "version": "1.0"})),
        "licenses": deepcopy(coco.get("licenses", [])),
        "images": [deepcopy(image) for image in coco.get("images", []) if image.get("id") in image_ids],
        "annotations": [
            deepcopy(annotation)
            for annotation in coco.get("annotations", [])
            if annotation.get("image_id") in image_ids
        ],
        "categories": deepcopy(coco.get("categories", [])),
    }

    for key, value in coco.items():
        if key not in subset:
            subset[key] = deepcopy(value)

    return subset


