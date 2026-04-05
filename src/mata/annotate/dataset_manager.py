from __future__ import annotations

"""Dataset manager — safe file-system operations for the annotation data root.

Every method that accepts user-supplied path components routes through
``_safe_resolve()`` to prevent path traversal attacks.
"""

import json
import mimetypes
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class DatasetManager:
    """Safe CRUD operations on the annotation data root directory."""

    def __init__(self, data_root: str | Path) -> None:
        self._root = Path(data_root).resolve()

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def _safe_resolve(self, *parts: str) -> Path:
        """Resolve *parts* relative to the data root and verify the result
        stays inside ``self._root``.

        Raises ``ValueError`` on any path traversal attempt.
        """
        combined = Path(*parts)
        resolved = (self._root / combined).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ValueError(f"Path traversal blocked: {'/'.join(str(p) for p in parts)}")
        return resolved

    # ------------------------------------------------------------------
    # Dataset listing
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        """List subdirectories in data_root that look like datasets.

        Returns ``[{"name": "coco_mini", "image_count": 16,
        "has_annotations": True, "type": "coco",
        "cache_valid": True}, ...]``.

        If a ``.mata_cache.yaml`` file exists inside the dataset directory its
        values are used directly (sub-millisecond read).  Otherwise the slow
        ``_fast_image_count()`` fall-back runs and ``cache_valid`` is set to
        ``False`` so the UI can offer a Rescan button.
        """
        result: list[dict] = []
        if not self._root.exists():
            return result
        for item in sorted(self._root.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue

            cache = self.read_dataset_cache(item.name)
            if cache is not None:
                result.append({
                    "name": item.name,
                    "image_count": cache.get("image_count", 0),
                    "has_annotations": cache.get("has_annotations", False),
                    "type": cache.get("type", "unknown"),
                    "cache_valid": True,
                })
            else:
                _ds_type = self.detect_dataset_type(item.name)
                _img_count = _fast_image_count(item)
                _has_ann = bool(
                    (item / "annotations").is_dir()
                    or any(
                        (item / d / "_annotations.coco.json").is_file()
                        for d in ("train", "val", "valid", "test")
                    )
                )
                result.append({
                    "name": item.name,
                    "image_count": _img_count,
                    "has_annotations": _has_ann,
                    "type": _ds_type,
                    "cache_valid": False,
                })
        return result

    # ------------------------------------------------------------------
    # Dataset cache helpers
    # ------------------------------------------------------------------

    _CACHE_FILENAME = ".mata_cache.yaml"
    _CACHE_VERSION = 1

    def read_dataset_cache(self, name: str) -> dict | None:
        """Return parsed ``.mata_cache.yaml`` for *name*, or ``None`` on miss."""
        try:
            cache_path = self._safe_resolve(name, self._CACHE_FILENAME)
            if not cache_path.is_file():
                return None
            import yaml  # type: ignore[import-untyped]
            with cache_path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict) or data.get("version") != self._CACHE_VERSION:
                return None
            return data
        except Exception:
            return None

    def write_dataset_cache(self, name: str, data: dict) -> None:
        """Write *data* to ``.mata_cache.yaml`` inside dataset *name*.

        Silently no-ops on any ``OSError`` so callers never have to handle
        storage failures.
        """
        try:
            import yaml  # type: ignore[import-untyped]
            payload = {"version": self._CACHE_VERSION, **data}
            cache_path = self._safe_resolve(name, self._CACHE_FILENAME)
            tmp = cache_path.with_suffix(".yaml.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, default_flow_style=False, allow_unicode=True)
            tmp.replace(cache_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Image listing & serving
    # ------------------------------------------------------------------

    def list_images(
        self,
        dataset: str,
        *,
        page: int | None = None,
        per_page: int = 50,
        sort: str = "name_asc",
        split: str | None = None,
        annotated: str | None = None,
        search: str | None = None,
        coco: dict | None = None,
    ) -> dict:
        """List image files in a dataset directory (recursive).

        Returns a pagination envelope::

            {
                "images": [{"filename": "...", "width": 320, "height": 320,
                             "size_bytes": 1234, "annotation_count": 3,
                             "split": "train"}, ...],
                "total": 9400,
                "page": 1,
                "per_page": 50,
                "total_pages": 188,
            }

        When *page* is ``None`` all images are returned in one page
        (backward-compatible mode).

        Parameters
        ----------
        page:
            1-based page number.  ``None`` returns all images.
        per_page:
            Images per page (default 50).
        sort:
            Sort order — ``name_asc`` | ``name_desc`` | ``newest`` |
            ``oldest`` | ``size``.
        split:
            Restrict to a split: ``train`` | ``test`` | ``val``.
        annotated:
            ``"true"`` — only annotated images; ``"false"`` — only
            unannotated; ``None`` — all.
        search:
            Case-insensitive substring match on filename.
        coco:
            Pre-loaded COCO document used to compute ``annotation_count``
            per image.  When ``None`` all counts are 0.
        """
        dataset_dir = self._safe_resolve(dataset)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset '{dataset}' not found.")

        # Build annotation_count lookup from COCO if provided
        ann_count_by_filename: dict[str, int] = {}
        if coco:
            _id_to_filename = {
                img["id"]: img.get("file_name", "")
                for img in coco.get("images", [])
            }
            for ann in coco.get("annotations", []):
                fname = _id_to_filename.get(ann.get("image_id", -1), "")
                if fname:
                    # Only the base filename is used for matching
                    ann_count_by_filename[Path(fname).name] = (
                        ann_count_by_filename.get(Path(fname).name, 0) + 1
                    )

        # ------------------------------------------------------------------
        # Fast path: build image list directly from COCO metadata when available.
        # This avoids a full rglob scan which is O(N) on large COCO datasets.
        # Sorting by mtime is not supported in this path (falls back to name_asc).
        # Always scan the filesystem so that physical images not listed in the
        # COCO images array are still discovered (e.g. unannotated images added
        # to the folder without updating the COCO JSON).
        images = _build_image_list_from_fs(
            dataset_dir, ann_count_by_filename, sort,
            _needs_mtime=(sort in ("newest", "oldest")),
        )
        # Apply filters
        if split is not None:
            split_lower = split.lower()
            images = [img for img in images if img.get("split") == split_lower]
        if annotated == "true":
            images = [img for img in images if img["annotation_count"] > 0]
        elif annotated == "false":
            images = [img for img in images if img["annotation_count"] == 0]
        if search:
            search_lower = search.lower()
            images = [img for img in images if search_lower in img["filename"].lower()]

        total = len(images)

        # Apply pagination
        if page is None:
            # Fill dimensions only for the (typically small) result set
            for img in images:
                if "width" not in img and "_path" in img:
                    _try_fill_dimensions(img.pop("_path"), img)
                else:
                    img.pop("_path", None)
            return {
                "images": images,
                "total": total,
                "page": 1,
                "per_page": total,
                "total_pages": 1,
            }

        per_page = max(1, per_page)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = start + per_page
        page_slice = images[start:end]

        # Fill dimensions only for the current page — avoids PIL opening every image
        for img in page_slice:
            if "width" not in img and "_path" in img:
                _try_fill_dimensions(img["_path"], img)
            img.pop("_path", None)
        # Strip _path from images NOT in the page slice
        for img in images:
            img.pop("_path", None)

        return {
            "images": page_slice,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        }

    def serve_image(self, dataset: str, filename: str) -> tuple[bytes, str]:
        """Read raw image bytes for *filename* in *dataset*.

        Returns ``(data, content_type)``.  Raises ``ValueError`` on path
        traversal and ``FileNotFoundError`` when the image does not exist.
        """
        # First attempt: direct resolution (handles sub-paths like "train/img.jpg")
        path = self._safe_resolve(dataset, filename)

        if not path.is_file():
            # Fallback: recursive search by base name only (no slashes in name)
            if "/" not in filename and "\\" not in filename:
                dataset_dir = self._safe_resolve(dataset)
                matches = sorted(
                    f for f in dataset_dir.rglob(filename)
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
                )
                if matches:
                    path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"Image '{filename}' not found in dataset '{dataset}'."
                    )
            else:
                raise FileNotFoundError(
                    f"Image '{filename}' not found in dataset '{dataset}'."
                )

        data = path.read_bytes()
        ct = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        return data, ct

    def serve_thumbnail(
        self, dataset: str, filename: str, max_size: int = 256
    ) -> tuple[bytes, str]:
        """Serve the original image directly (no thumbnail cache)."""
        return self.serve_image(dataset, filename)

    # ------------------------------------------------------------------
    # Dataset CRUD
    # ------------------------------------------------------------------

    def create_dataset(self, name: str) -> dict:
        """Create a new dataset directory with standard sub-structure.

        Creates ``data/{name}/images/`` and ``data/{name}/annotations/``.
        Raises ``ValueError`` for invalid names or if the dataset exists.
        """
        if not _DATASET_NAME_RE.match(name):
            raise ValueError(
                f"Invalid dataset name '{name}'. "
                "Use alphanumeric characters, underscores, or hyphens (max 64 chars)."
            )
        dataset_dir = self._safe_resolve(name)
        if dataset_dir.exists():
            raise ValueError(f"Dataset '{name}' already exists.")

        (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)

        logger.info("Created dataset '%s' at %s", name, dataset_dir)
        return {"name": name, "path": str(dataset_dir), "created": True}

    def get_dataset_info(self, name: str, *, coco: dict | None = None) -> dict:
        """Return a summary dict for the named dataset.

        Parameters
        ----------
        name:
            Dataset directory name inside ``data_root``.
        coco:
            Pre-loaded COCO document.  When supplied **all** counts are derived
            directly from the JSON — no filesystem scan is performed.  This is
            the fast path for large COCO datasets (avoids O(N) rglob + stat).
        """
        dataset_dir = self._safe_resolve(name)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset '{name}' not found.")

        # Cheap path-only operations shared by both paths
        try:
            folder_path = dataset_dir.relative_to(Path.cwd()).as_posix()
        except ValueError:
            folder_path = dataset_dir.as_posix()

        has_train = any(
            (dataset_dir / d).is_dir() for d in ("train", "train2017", "train2014")
        )
        has_val = any(
            (dataset_dir / d).is_dir() for d in ("val", "valid", "val2017", "val2014")
        )
        dataset_type = self.detect_dataset_type(name)

        # Try to obtain COCO metadata when not supplied — load from disk if
        # annotations/ contains a JSON file, or merge per-split JSONs for
        # Roboflow-style datasets that co-locate _annotations.coco.json inside
        # each split directory (train/, valid/, test/).
        _coco = coco
        if _coco is None:
            ann_dir = dataset_dir / "annotations"
            if ann_dir.is_dir():
                for json_file in sorted(ann_dir.glob("*.json")):
                    try:
                        _coco = json.loads(json_file.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        pass
            # Roboflow / split-dir style: merge all per-split COCO JSONs
            if _coco is None:
                _coco = _merge_split_coco_jsons(dataset_dir)

        # ------------------------------------------------------------------
        # FAST PATH: derive everything from COCO JSON (no filesystem scan).
        # Used for COCO-format datasets — image counts can be in the 100 k+
        # range, making rglob + stat prohibitively slow (up to several minutes).
        # ------------------------------------------------------------------
        if _coco is not None and isinstance(_coco, dict):
            coco_images: list[dict] = _coco.get("images") or []
            coco_annotations: list[dict] = _coco.get("annotations") or []
            coco_categories: list[dict] = _coco.get("categories") or []

            image_count = len(coco_images)
            annotation_count = len(coco_annotations)
            classes: list[str] = [c["name"] for c in coco_categories if c.get("name")]

            # Annotated images = any image_id that has at least one annotation
            annotated_ids: set[int] = {
                a["image_id"] for a in coco_annotations if "image_id" in a
            }
            total_annotated = sum(
                1 for img in coco_images if img.get("id") in annotated_ids
            )
            total_unannotated = image_count - total_annotated
            browse_progress = (
                round((total_annotated / image_count) * 100, 1) if image_count > 0 else 0.0
            )

            # Splits: first try to detect from file_name path prefix (standard COCO
            # convention: file_name = "train2017/000001.jpg").
            splits_total: dict[str, int] = {}
            splits_annotated_d: dict[str, int] = {}
            for img in coco_images:
                fname = img.get("file_name", "")
                parts_lower = {p.lower() for p in Path(fname).parts[:-1]}
                image_split: str | None = None
                for s in ("train", "test", "val"):
                    if any(s in p for p in parts_lower):
                        image_split = s
                        break
                if image_split:
                    splits_total[image_split] = splits_total.get(image_split, 0) + 1
                    if img.get("id") in annotated_ids:
                        splits_annotated_d[image_split] = (
                            splits_annotated_d.get(image_split, 0) + 1
                        )

            # Fallback: COCO file_names have no path prefix — detect splits and
            # compute sizes by scanning known image subdirectories.
            _KNOWN_SPLIT_DIRS = (
                "train", "val", "valid", "test",
                "train2017", "val2017", "test2017",
                "train2014", "val2014", "test2014",
            )
            _IMG_SUBDIRS = ("images",) + _KNOWN_SPLIT_DIRS
            if not splits_total:
                _basename_to_id: dict[str, int] = {
                    Path(img.get("file_name", "")).name: img["id"]
                    for img in coco_images
                    if img.get("id") is not None
                }
                for _sd in _KNOWN_SPLIT_DIRS:
                    _sdir = dataset_dir / _sd
                    if not _sdir.is_dir():
                        continue
                    _split_key = (
                        "train" if "train" in _sd
                        else ("val" if "val" in _sd else "test")
                    )
                    for _f in _sdir.rglob("*"):
                        if _f.is_file() and _f.suffix.lower() in _IMAGE_EXTENSIONS:
                            splits_total[_split_key] = (
                                splits_total.get(_split_key, 0) + 1
                            )
                            _img_id = _basename_to_id.get(_f.name)
                            if _img_id is not None and _img_id in annotated_ids:
                                splits_annotated_d[_split_key] = (
                                    splits_annotated_d.get(_split_key, 0) + 1
                                )

            splits: dict[str, dict] = {
                s: {"total": splits_total[s], "annotated": splits_annotated_d.get(s, 0)}
                for s in splits_total
            }

            # File sizes: scan known image subdirs only (avoids root rglob and
            # skips large annotation JSON files in annotations/).
            total_size_bytes = 0
            for _sub in _IMG_SUBDIRS:
                _sub_dir = dataset_dir / _sub
                if _sub_dir.is_dir():
                    for _f in _sub_dir.rglob("*"):
                        if _f.is_file() and _f.suffix.lower() in _IMAGE_EXTENSIONS:
                            try:
                                total_size_bytes += _f.stat().st_size
                            except OSError:
                                pass

            return {
                "name": name,
                "type": dataset_type,
                "image_count": image_count,
                "annotation_count": annotation_count,
                "total_annotated": total_annotated,
                "total_unannotated": total_unannotated,
                "total_size_bytes": total_size_bytes,
                "folder_path": folder_path,
                "browse_progress": browse_progress,
                "splits": splits,
                "classes": classes,
                "has_train_split": has_train,
                "has_val_split": has_val,
            }

        # ------------------------------------------------------------------
        # SLOW PATH: full recursive filesystem scan.
        # Used only when no COCO JSON is available (e.g. ImageFolder datasets).
        # ------------------------------------------------------------------
        total_size_bytes: int = 0
        image_count = 0
        total_annotated = 0
        splits_total = {}
        splits_annotated_d = {}

        for f in sorted(dataset_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            rel = f.relative_to(dataset_dir)
            if any(p.name.startswith(".") for p in rel.parents):
                continue

            image_count += 1
            size = f.stat().st_size
            total_size_bytes += size

            rel_parts = {p.lower() for p in rel.parts[:-1]}
            image_split = None
            for s in ("train", "test", "val"):
                if any(s in part for part in rel_parts):
                    image_split = s
                    break

            if image_split:
                splits_total[image_split] = splits_total.get(image_split, 0) + 1
                splits_annotated_d.setdefault(image_split, 0)

        total_unannotated = image_count
        browse_progress = 0.0
        splits = {
            s: {"total": splits_total[s], "annotated": 0}
            for s in splits_total
        }

        return {
            "name": name,
            "type": dataset_type,
            "image_count": image_count,
            "annotation_count": 0,
            "total_annotated": 0,
            "total_unannotated": total_unannotated,
            "total_size_bytes": total_size_bytes,
            "folder_path": folder_path,
            "browse_progress": browse_progress,
            "splits": splits,
            "classes": [],
            "has_train_split": has_train,
            "has_val_split": has_val,
        }

    # ------------------------------------------------------------------
    # Classification dataset operations
    # ------------------------------------------------------------------

    def list_classes(self, dataset: str) -> list[dict]:
        """List class directories and image counts for an ImageFolder dataset.

        For split datasets such as ``train/`` + ``val/``, the primary class
        root is ``train/`` so counts match the editable training layout.
        """
        class_root = _get_primary_class_root(self, dataset)
        result: list[dict] = []

        for class_dir in _iter_class_dirs(class_root):
            count = sum(
                1 for file in class_dir.iterdir()
                if file.is_file()
                and not file.name.startswith(".")
                and file.suffix.lower() in _IMAGE_EXTENSIONS
            )
            result.append({"name": class_dir.name, "count": count})

        return result

    def reclassify_image(
        self, dataset: str, filename: str, from_class: str, to_class: str
    ) -> None:
        """Move an image between class directories, preserving its split root."""
        _validate_class_name(from_class)
        _validate_class_name(to_class)

        source_root: Path | None = None
        source_path: Path | None = None

        for class_root in _get_classification_roots(self, dataset):
            root_rel = class_root.relative_to(self._root)
            candidate = self._safe_resolve(*root_rel.parts, from_class, filename)
            if candidate.is_file():
                source_root = class_root
                source_path = candidate
                break

        if source_root is None or source_path is None:
            raise FileNotFoundError(
                f"Image '{filename}' not found in class '{from_class}' for dataset '{dataset}'."
            )

        root_rel = source_root.relative_to(self._root)
        destination_dir = self._safe_resolve(*root_rel.parts, to_class)
        if not destination_dir.is_dir():
            raise ValueError(
                f"Destination class '{to_class}' does not exist in dataset '{dataset}'."
            )

        destination_path = self._safe_resolve(*root_rel.parts, to_class, source_path.name)
        if destination_path.exists():
            raise ValueError(
                f"Destination image '{source_path.name}' already exists in class '{to_class}'."
            )

        shutil.move(str(source_path), str(destination_path))

    def create_class(self, dataset: str, class_name: str) -> None:
        """Create a class subdirectory in the dataset root or all split roots."""
        _validate_class_name(class_name)

        for class_root in _get_classification_roots(self, dataset):
            root_rel = class_root.relative_to(self._root)
            class_dir = self._safe_resolve(*root_rel.parts, class_name)
            class_dir.mkdir(parents=False, exist_ok=True)

    def delete_class(self, dataset: str, class_name: str) -> None:
        """Delete an empty class directory from the dataset root or all split roots."""
        _validate_class_name(class_name)

        class_dirs: list[Path] = []
        for class_root in _get_classification_roots(self, dataset):
            root_rel = class_root.relative_to(self._root)
            class_dir = self._safe_resolve(*root_rel.parts, class_name)
            if class_dir.exists():
                class_dirs.append(class_dir)

        if not class_dirs:
            raise FileNotFoundError(
                f"Class '{class_name}' not found in dataset '{dataset}'."
            )

        for class_dir in class_dirs:
            entries = [entry for entry in class_dir.iterdir() if not entry.name.startswith(".")]
            if entries:
                raise ValueError(f"Class '{class_name}' is not empty.")

        for class_dir in class_dirs:
            class_dir.rmdir()

    # ------------------------------------------------------------------
    # Split management
    # ------------------------------------------------------------------

    def move_to_split(self, dataset: str, filename: str, target_split: str) -> dict:
        """Move an image file to a different split directory (train/val/test).

        The image is located by basename search inside the dataset tree.
        The target split directory is created if it does not exist.

        Returns a dict ``{"moved": bool, "from": str, "to": str, "filename": str}``.

        Raises:
            ValueError: If *target_split* is not one of ``train``, ``val``, ``test``.
            FileNotFoundError: If *filename* is not found inside the dataset.
        """
        _VALID_SPLITS = {"train", "val", "test"}
        if target_split not in _VALID_SPLITS:
            raise ValueError(
                f"Invalid target_split '{target_split}'. "
                f"Must be one of: {', '.join(sorted(_VALID_SPLITS))}."
            )

        dataset_dir = self._safe_resolve(dataset)
        base = filename.split("/")[-1]

        # Find the image file anywhere inside the dataset directory.
        image_path: Path | None = None
        for candidate in dataset_dir.rglob(base):
            if candidate.is_file() and candidate.suffix.lower() in _IMAGE_EXTENSIONS:
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(
                f"Image '{filename}' not found in dataset '{dataset}'."
            )

        target_dir = self._safe_resolve(dataset, target_split)
        destination = self._safe_resolve(dataset, target_split, image_path.name)

        from_rel = str(image_path.relative_to(dataset_dir)).replace("\\", "/")
        to_rel = str(destination.relative_to(dataset_dir)).replace("\\", "/")

        if image_path == destination:
            return {"moved": False, "from": from_rel, "to": to_rel, "filename": image_path.name}

        if destination.exists():
            raise ValueError(
                f"Image '{image_path.name}' already exists in split '{target_split}'."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(destination))
        logger.info("Moved '%s' → %s/%s", image_path.name, dataset, target_split)
        return {"moved": True, "from": from_rel, "to": to_rel, "filename": image_path.name}

    # ------------------------------------------------------------------
    # Dataset type detection
    # ------------------------------------------------------------------

    def detect_dataset_type(self, name: str) -> str:
        """Infer dataset layout: ``"coco"``, ``"classification"``, ``"imagefolder"``, ``"voc"``, or ``"empty"``.

        Detection order (mirrors ``DatasetFactory._from_directory``):
        1. ``Annotations/*.xml`` present → ``"voc"``
        2. ``annotations/*.json`` present → ``"coco"``
        3. ``*.json`` at root level → ``"coco"``
        4. Split dirs (train/val/test) whose children are all subdirs → ``"classification"``
        5. All direct children are directories → ``"imagefolder"``
        6. Root contains image files directly → ``"imagefolder"``
        7. Otherwise → ``"empty"``
        """
        try:
            dataset_dir = self._safe_resolve(name)
        except ValueError:
            return "empty"

        if not dataset_dir.is_dir():
            return "empty"

        # VOC
        voc_ann = dataset_dir / "Annotations"
        if voc_ann.is_dir() and any(voc_ann.glob("*.xml")):
            return "voc"

        # COCO — annotations/ subdir with JSON
        coco_ann = dataset_dir / "annotations"
        if coco_ann.is_dir() and any(coco_ann.glob("*.json")):
            return "coco"

        # COCO — JSON files at root
        if any(dataset_dir.glob("*.json")):
            return "coco"

        # COCO — Roboflow style: _annotations.coco.json co-located in split dirs
        _SPLIT_NAMES_COCO = frozenset({"train", "val", "test", "valid"})
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name.lower() in _SPLIT_NAMES_COCO:
                if any(child.glob("*.json")):
                    return "coco"

        # ImageFolder / Classification
        children = [c for c in dataset_dir.iterdir() if not c.name.startswith(".")]
        if not children:
            return "empty"

        child_dirs = [c for c in children if c.is_dir()]
        child_files = [c for c in children if c.is_file()]

        # Classification — split dirs (train/val/test) whose children are
        # all subdirectories (class folders), not raw image files.
        # We require at least one qualifying split dir (train or val) with
        # class-folder layout, and no split dir may contain raw image files
        # mixed with subdirs.  A split whose top-level children are ALL files
        # (e.g. a flat test/ dump) is simply skipped so one bad split doesn't
        # veto detection for the rest.
        _SPLIT_NAMES = frozenset({"train", "val", "test", "valid"})
        _PRIMARY_SPLITS = frozenset({"train", "val"})
        split_dirs = [d for d in child_dirs if d.name.lower() in _SPLIT_NAMES]
        if split_dirs:
            def _split_layout(split_dir: Path) -> str:
                """Return 'class_folder', 'flat', or 'mixed'."""
                sc = [c for c in split_dir.iterdir() if not c.name.startswith(".")]
                if not sc:
                    return "flat"
                has_dirs = any(c.is_dir() for c in sc)
                has_imgs = any(c.is_file() and c.suffix.lower() in _IMAGE_EXTENSIONS for c in sc)
                if has_dirs and not has_imgs:
                    return "class_folder"
                if has_imgs and not has_dirs:
                    return "flat"
                return "mixed"

            layouts = {d: _split_layout(d) for d in split_dirs}
            # Any mixed split (subdirs + images at same level) disqualifies
            if not any(v == "mixed" for v in layouts.values()):
                class_folder_splits = [d for d, v in layouts.items() if v == "class_folder"]
                # At least one primary split (train/val) must be class-folder
                has_primary = any(d.name.lower() in _PRIMARY_SPLITS for d in class_folder_splits)
                if class_folder_splits and has_primary:
                    return "classification"

        if child_dirs and not child_files:
            return "imagefolder"

        # Flat folder with images at root
        if any(f.suffix.lower() in _IMAGE_EXTENSIONS for f in child_files):
            return "imagefolder"

        return "empty"

    # ------------------------------------------------------------------
    # Split redistribution
    # ------------------------------------------------------------------

    def redistribute_splits(
        self,
        dataset: str,
        train_pct: int,
        val_pct: int,
        test_pct: int,
        *,
        seed: int | None = None,
        annotated_first: bool = True,
    ) -> dict:
        """Redistribute all images in *dataset* across train/val/test by ratio.

        Collects every image in the dataset tree (all subdirs, including existing
        split dirs and unassigned root images), shuffles them, and moves each one
        to the appropriate ``<dataset>/<split>/`` subdirectory.

        Args:
            dataset: Dataset name.
            train_pct: Training split percentage (0–100).
            val_pct:   Validation split percentage (0–100).
            test_pct:  Test split percentage (0–100).
            seed:      Random seed for reproducibility.  ``None`` = fully random.
            annotated_first: When *True*, images that already have annotations
                             are stable-sorted to the front before ratios are
                             applied, so annotated images are preferentially
                             assigned to the training split.

        Returns:
            ``{"total": N, "moved": M, "splits": {"train": X, "val": Y, "test": Z}}``.

        Raises:
            ValueError: If percentages are invalid or do not sum to 100.
        """
        import random as _random

        if not (0 <= train_pct <= 100 and 0 <= val_pct <= 100 and 0 <= test_pct <= 100):
            raise ValueError("Each percentage must be between 0 and 100.")
        if train_pct + val_pct + test_pct != 100:
            raise ValueError(
                f"Percentages must sum to 100 (got {train_pct + val_pct + test_pct})."
            )

        dataset_dir = self._safe_resolve(dataset)

        # Build full image list (no annotation data needed, just paths + ann count)
        images = _build_image_list_from_fs(
            dataset_dir,
            ann_count_by_filename={},
            sort="name",
        )

        # Strip internal keys — we only need the Path
        paths: list[Path] = [img["_path"] for img in images]
        ann_counts: list[int] = [img.get("annotation_count", 0) for img in images]

        n = len(paths)
        rng = _random.Random(seed)
        # Pair paths with annotation counts, shuffle, then optionally stable-sort
        pairs = list(zip(paths, ann_counts))
        rng.shuffle(pairs)
        if annotated_first:
            # Stable sort: annotated images first, preserve internal shuffle order
            pairs.sort(key=lambda p: 0 if p[1] > 0 else 1)

        shuffled_paths = [p for p, _ in pairs]

        # Calculate split sizes (last bin absorbs rounding remainder)
        n_train = round(n * train_pct / 100)
        n_val = round(n * val_pct / 100)
        n_test = n - n_train - n_val

        # Clamp — rounding can produce negatives if a pct is 0
        n_test = max(0, n_test)

        assignments: list[tuple[Path, str]] = []
        for idx, path in enumerate(shuffled_paths):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments.append((path, split))

        _VALID_SPLITS_SET = {"train", "val", "test"}
        # Track basename → new relative file_name for COCO JSON update.
        moved_map: dict[str, str] = {}  # basename → "split/basename"
        moved = 0
        for image_path, target_split in assignments:
            if target_split not in _VALID_SPLITS_SET:
                continue
            target_dir = dataset_dir / target_split
            destination = target_dir / image_path.name
            if image_path == destination:
                # Already in the right place — still record so file_name is normalised.
                moved_map[image_path.name] = f"{target_split}/{image_path.name}"
                continue
            if destination.exists():
                logger.warning(
                    "Skipping '%s' → %s/ — destination already exists.",
                    image_path.name,
                    target_split,
                )
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(destination))
            moved_map[image_path.name] = f"{target_split}/{image_path.name}"
            moved += 1

        # Update COCO JSON file_name values to reflect new locations.
        _update_coco_file_names(dataset_dir, moved_map)

        logger.info(
            "redistribute_splits('%s'): moved %d/%d images  train=%d val=%d test=%d",
            dataset,
            moved,
            n,
            n_train,
            n_val,
            n_test,
        )
        return {
            "total": n,
            "moved": moved,
            "splits": {"train": n_train, "val": n_val, "test": n_test},
        }


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------


def _update_coco_file_names(dataset_dir: Path, moved_map: dict[str, str]) -> None:
    """Rewrite ``file_name`` in every COCO JSON under *dataset_dir* annotations.

    *moved_map* maps ``basename`` → ``"split/basename"`` for every image that
    was moved (or confirmed to be in its correct split directory).  Only entries
    whose current basename appears in *moved_map* are touched.
    """
    if not moved_map:
        return

    # Find all candidate COCO JSON files: annotations/ dir and split-dir roots.
    candidates: list[Path] = []
    ann_dir = dataset_dir / "annotations"
    if ann_dir.is_dir():
        candidates.extend(ann_dir.glob("*.json"))
    _SPLIT_NAMES = frozenset({"train", "val", "valid", "test"})
    for split_dir in sorted(dataset_dir.iterdir()):
        if split_dir.is_dir() and split_dir.name.lower() in _SPLIT_NAMES:
            candidates.extend(split_dir.glob("*.json"))

    for json_path in candidates:
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict) or "images" not in doc:
            continue

        changed = False
        for img in doc.get("images", []):
            current = img.get("file_name", "")
            base = Path(current).name
            new_file_name = moved_map.get(base)
            if new_file_name and current != new_file_name:
                img["file_name"] = new_file_name
                changed = True

        if changed:
            tmp = json_path.with_suffix(".json.tmp")
            try:
                tmp.write_text(
                    json.dumps(doc, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                tmp.replace(json_path)
            except Exception:
                tmp.unlink(missing_ok=True)

def _fast_image_count(dataset_dir: Path) -> int:
    """Return image count for *dataset_dir* without a full recursive scan.

    Strategy (in priority order):
    1. Sum ``len(doc["images"])`` across all COCO JSON files found under
       ``annotations/`` — fast, O(1) per JSON file header parse.
    2. ``dataset.yaml`` — resolve each referenced annotation JSON and sum.
    3. Fallback: recursive rglob scan (original slow path).
    """
    # -- Strategy 1: COCO annotations/ subdir ---------------------------------
    ann_dir = dataset_dir / "annotations"
    if ann_dir.is_dir():
        total = 0
        counted = False
        for json_path in sorted(ann_dir.glob("*.json")):
            try:
                with json_path.open(encoding="utf-8") as fh:
                    doc = json.load(fh)
                if isinstance(doc, dict) and "images" in doc:
                    total += len(doc["images"])
                    counted = True
            except Exception:
                pass
        if counted:
            return total

    # -- Strategy 1.5: Roboflow / co-located split-dir JSONs ----------------
    _SPLIT_NAMES_FIC = frozenset({"train", "val", "valid", "test"})
    total = 0
    counted = False
    for split_dir in sorted(dataset_dir.iterdir()):
        if not (split_dir.is_dir() and split_dir.name.lower() in _SPLIT_NAMES_FIC):
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            try:
                with json_path.open(encoding="utf-8") as fh:
                    doc = json.load(fh)
                if isinstance(doc, dict) and "images" in doc:
                    total += len(doc["images"])
                    counted = True
            except Exception:
                pass
    if counted:
        return total

    # -- Strategy 2: dataset.yaml --------------------------------------------
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.is_file():
        try:
            import yaml  # type: ignore[import-untyped]

            with yaml_path.open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            if isinstance(cfg, dict):
                total = 0
                counted = False
                for key in ("train_annotations", "val_annotations", "test_annotations"):
                    rel = cfg.get(key)
                    if rel:
                        candidate = dataset_dir / rel
                        if candidate.is_file():
                            try:
                                with candidate.open(encoding="utf-8") as fh:
                                    doc = json.load(fh)
                                if isinstance(doc, dict) and "images" in doc:
                                    total += len(doc["images"])
                                    counted = True
                            except Exception:
                                pass
                if counted:
                    return total
        except Exception:
            pass

    # -- Strategy 3: fallback rglob scan -------------------------------------
    return sum(
        1 for f in dataset_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        and not any(p.name.startswith(".") for p in f.relative_to(dataset_dir).parents)
    )


def _build_image_list_from_coco(
    coco: dict,
    ann_count_by_filename: dict[str, int],
    split: str | None,
    annotated: str | None,
    search: str | None,
    sort: str,
) -> list[dict]:
    """Build a filtered+sorted image list from COCO metadata (no filesystem scan)."""
    images: list[dict] = []
    for img_meta in coco.get("images", []):
        fname = img_meta.get("file_name", "")
        if not fname:
            continue
        base = Path(fname).name

        # Best-effort split detection from file_name path
        parts = {p.lower() for p in Path(fname).parts[:-1]}
        image_split: str | None = None
        for s in ("train", "test", "val"):
            if any(s in p for p in parts):
                image_split = s
                break

        ann_count = ann_count_by_filename.get(base, 0)

        entry: dict[str, Any] = {
            "filename": base,
            "size_bytes": img_meta.get("size_bytes", 0),
            "annotation_count": ann_count,
            "split": image_split,
        }
        if img_meta.get("width"):
            entry["width"] = img_meta["width"]
        if img_meta.get("height"):
            entry["height"] = img_meta["height"]

        # Apply filters inline — avoids building then filtering a large list
        if split is not None and image_split != split.lower():
            continue
        if annotated == "true" and ann_count == 0:
            continue
        if annotated == "false" and ann_count > 0:
            continue
        if search and search.lower() not in base.lower():
            continue

        images.append(entry)

    # Sort (mtime/size not available from COCO; callers should use fs path for those)
    if sort == "name_desc":
        images.sort(key=lambda x: x["filename"], reverse=True)
    else:
        images.sort(key=lambda x: x["filename"])

    return images


def _build_image_list_from_fs(
    dataset_dir: Path,
    ann_count_by_filename: dict[str, int],
    sort: str,
    *,
    _needs_mtime: bool = False,
) -> list[dict]:
    """Build an unsorted image list by scanning *dataset_dir* recursively.

    Internal ``_path`` and ``_mtime`` keys are retained for deferred sorting
    and dimension filling; callers must strip them before returning to clients.
    """
    images: list[dict] = []
    seen: set[str] = set()

    for f in sorted(dataset_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        rel = f.relative_to(dataset_dir)
        if any(p.name.startswith(".") for p in rel.parents):
            continue

        flat_name = f.name
        if flat_name in seen:
            flat_name = rel.as_posix()
        seen.add(flat_name)

        rel_parts = {p.lower() for p in rel.parts[:-1]}
        image_split: str | None = None
        for s in ("train", "test", "val"):
            if any(s in part for part in rel_parts):
                image_split = s
                break

        entry: dict[str, Any] = {
            "filename": flat_name,
            "size_bytes": f.stat().st_size,
            "annotation_count": ann_count_by_filename.get(f.name, 0),
            "split": image_split,
            "_path": f,
        }
        if _needs_mtime:
            entry["_mtime"] = os.path.getmtime(f)
        images.append(entry)

    # Sort
    if sort == "name_desc":
        images.sort(key=lambda x: x["filename"], reverse=True)
    elif sort == "newest":
        images.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
    elif sort == "oldest":
        images.sort(key=lambda x: x.get("_mtime", 0))
    elif sort == "size":
        images.sort(key=lambda x: x["size_bytes"], reverse=True)
    else:
        images.sort(key=lambda x: x["filename"])

    return images


def _has_annotation_files(dataset_dir: Path) -> bool:
    """Return True if *dataset_dir* contains any COCO or VOC annotation files."""
    ann_dir = dataset_dir / "annotations"
    if ann_dir.is_dir() and any(ann_dir.glob("*.json")):
        return True
    if any(dataset_dir.glob("*.json")):
        return True
    voc_ann = dataset_dir / "Annotations"
    if voc_ann.is_dir() and any(voc_ann.glob("*.xml")):
        return True
    return False


def _run_rescan_worker(
    dm: "DatasetManager",
    name: str,
    jobs: dict,
    lock: threading.Lock,
) -> None:
    """Background thread target: scan *name*, write cache, update *jobs*.

    This function runs inside a daemon thread started by
    ``AnnotateServer.start_rescan()``.  It reads only the metadata
    (image count, type, annotation presence) — NOT individual annotation
    counts — so the cache stays stable between annotation edits.
    """
    try:
        dataset_dir = dm._safe_resolve(name)
        image_count = _fast_image_count(dataset_dir)
        has_annotations = _has_annotation_files(dataset_dir)
        dataset_type = dm.detect_dataset_type(name)
        last_scanned = datetime.now(timezone.utc).isoformat()

        payload = {
            "image_count": image_count,
            "type": dataset_type,
            "has_annotations": has_annotations,
            "last_scanned": last_scanned,
        }
        dm.write_dataset_cache(name, payload)

        with lock:
            jobs[name] = {"status": "done", "cache": payload}
    except Exception as exc:
        with lock:
            jobs[name] = {"status": "error", "message": str(exc)}


def _run_redistribute_worker(
    dm: "DatasetManager",
    name: str,
    params: dict,
    jobs: dict,
    lock: threading.Lock,
) -> None:
    """Background thread target: run redistribute_splits, update *jobs*.

    *params* must contain ``train``, ``val``, ``test`` (int percentages) and
    optionally ``annotated_first`` (bool, default True) and ``seed`` (int|None).
    """
    try:
        result = dm.redistribute_splits(
            name,
            train_pct=int(params["train"]),
            val_pct=int(params["val"]),
            test_pct=int(params["test"]),
            seed=params.get("seed"),
            annotated_first=bool(params.get("annotated_first", True)),
        )
        with lock:
            jobs[name] = {"status": "done", "result": result}
    except Exception as exc:
        with lock:
            jobs[name] = {"status": "error", "message": str(exc)}


def _try_fill_dimensions(path: Path, entry: dict) -> None:
    """Attempt to add ``width`` and ``height`` keys to *entry* using Pillow."""
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(path) as img:
            entry["width"], entry["height"] = img.size
    except Exception:
        pass  # Pillow not available or image unreadable — dimensions omitted


def _merge_split_coco_jsons(dataset_dir: Path) -> dict | None:
    """Merge per-split COCO JSON files (Roboflow style) into a single document.

    Scans ``train/``, ``val/``, ``valid/``, and ``test/`` subdirectories for
    ``*.json`` files that look like COCO documents (contain an "images" key),
    and returns a merged dict with globally unique image/annotation IDs.

    Returns ``None`` if no qualifying JSON files are found.
    """
    _SPLIT_NAMES = frozenset({"train", "val", "valid", "test"})
    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    merged_categories: list[dict] = []
    seen_cat_ids: set[int] = set()
    img_id_offset = 0
    ann_id_offset = 0
    found_any = False

    for split_dir in sorted(dataset_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name.lower() not in _SPLIT_NAMES:
            continue
        for json_file in sorted(split_dir.glob("*.json")):
            try:
                split_coco = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(split_coco, dict) or "images" not in split_coco:
                continue
            found_any = True
            for cat in split_coco.get("categories") or []:
                if cat.get("id") not in seen_cat_ids:
                    merged_categories.append(cat)
                    seen_cat_ids.add(cat.get("id"))
            split_images = split_coco.get("images") or []
            split_anns = split_coco.get("annotations") or []
            old_to_new: dict[int, int] = {}
            for img in split_images:
                old_id = img.get("id", 0)
                new_id = old_id + img_id_offset
                old_to_new[old_id] = new_id
                merged_images.append({**img, "id": new_id})
            for ann in split_anns:
                old_img_id = ann.get("image_id", 0)
                merged_annotations.append({
                    **ann,
                    "id": ann.get("id", 0) + ann_id_offset,
                    "image_id": old_to_new.get(old_img_id, old_img_id + img_id_offset),
                })
            max_img_id = max((img.get("id", 0) for img in split_images), default=0)
            max_ann_id = max((ann.get("id", 0) for ann in split_anns), default=0)
            img_id_offset += max_img_id + 1
            ann_id_offset += max_ann_id + 1
            break  # one JSON per split dir is sufficient

    if not found_any:
        return None
    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }


def _resolve_image_path(dm: DatasetManager, dataset: str, filename: str) -> Path:
    """Shared helper: resolve an image path within *dataset*, with rglob fallback."""
    path = dm._safe_resolve(dataset, filename)
    if path.is_file():
        return path
    if "/" not in filename and "\\" not in filename:
        dataset_dir = dm._safe_resolve(dataset)
        matches = sorted(
            f for f in dataset_dir.rglob(filename)
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Image '{filename}' not found in dataset '{dataset}'.")


def _validate_class_name(class_name: str) -> None:
    """Validate a classification class name using the dataset naming rules."""
    if not _DATASET_NAME_RE.match(class_name):
        raise ValueError(
            f"Invalid class name '{class_name}'. "
            "Use alphanumeric characters, underscores, or hyphens (max 64 chars)."
        )


def _iter_class_dirs(root: Path):
    """Yield immediate non-hidden class directories in sorted order."""
    return (
        child for child in sorted(root.iterdir())
        if child.is_dir() and not child.name.startswith(".")
    )


def _get_classification_roots(dm: DatasetManager, dataset: str) -> list[Path]:
    """Return the editable roots for an ImageFolder dataset.

    If ``train/``, ``val/``, or ``test/`` split directories exist, those are
    treated as the classification roots. Otherwise the dataset directory itself
    is treated as the class root.
    """
    dataset_dir = dm._safe_resolve(dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset '{dataset}' not found.")

    split_roots = [
        dataset_dir / split
        for split in ("train", "val", "test")
        if (dataset_dir / split).is_dir()
    ]
    return split_roots or [dataset_dir]


def _get_primary_class_root(dm: DatasetManager, dataset: str) -> Path:
    """Return the primary class root used for class listing and editing UI."""
    roots = _get_classification_roots(dm, dataset)
    for root in roots:
        if root.name == "train":
            return root
    return roots[0]
