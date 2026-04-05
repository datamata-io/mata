from __future__ import annotations

"""REST API routing layer for the MATA annotation server.

``dispatch()`` is the single entry point called by ``AnnotateHandler`` for
every ``/api/*`` request.  It:

1. Parses and sanitizes the URL path into segments.
2. Validates dataset names and required request fields.
3. Delegates to ``dataset_manager``, ``coco_io``, or ``ai_assist`` modules.
4. Returns ``(status_code, response_body)`` — body is a dict (JSON) or a
   ``(bytes, content_type)`` tuple for binary responses.
5. Returns ``None`` for completely unknown routes (server sends 404).

Because backend modules (B1–E3) may not yet be implemented, calls that
require them are wrapped so they return 501 gracefully when the relevant
module hasn't been filled in yet.
"""

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    from mata.annotate.server import AnnotateServer

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_DANGEROUS_KEY_RE = re.compile(r"^__.*__$|^prototype$|\$", re.IGNORECASE)


def _valid_dataset_name(name: str) -> bool:
    """Return True iff *name* is a safe dataset/class identifier."""
    return bool(_DATASET_NAME_RE.match(name))


def _require_fields(body: dict, *fields: str) -> str | None:
    """Return an error message if any required field is absent, else None."""
    for f in fields:
        if f not in body:
            return f"Missing required field: '{f}'"
    return None


def _validate_patch_body(body: dict) -> str | None:
    """Validate a PATCH annotation request body.

    Returns an error message string on failure, or ``None`` on success.
    Rejects non-dict bodies, non-string keys, dangerous prototype-injection
    keys, and known fields with wrong types (category_id, bbox).
    """
    if not isinstance(body, dict):
        return "Request body must be a JSON object."
    for key in body:
        if not isinstance(key, str):
            return "All body keys must be strings."
        if _DANGEROUS_KEY_RE.search(key):
            return f"Key '{key}' is not permitted."
    if "category_id" in body and not isinstance(body["category_id"], int):
        return "'category_id' must be an integer."
    if "bbox" in body:
        bbox = body["bbox"]
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            return "'bbox' must be a list of 4 numbers."
    return None


def _parse_path(path: str) -> list[str]:
    """Split ``/api/datasets/foo/images`` into ``['datasets', 'foo', 'images']``.

    Each segment is percent-decoded so that filenames containing encoded
    slashes (e.g. ``train%2Fsample.jpg``) arrive as ``train/sample.jpg``.
    Path-traversal safety is enforced downstream by ``_safe_resolve``.
    """
    rel = path.lstrip("/")
    if rel.startswith("api/"):
        rel = rel[4:]
    return [unquote(s) for s in rel.split("/") if s]


# ---------------------------------------------------------------------------
# Backend accessor helpers — degrade gracefully before modules are complete
# ---------------------------------------------------------------------------

def _dm(server: "AnnotateServer") -> Any:
    """Return the DatasetManager attached to *server*, or raise NotImplementedError."""
    dm = getattr(server, "dataset_manager", None)
    if dm is None:
        raise NotImplementedError("DatasetManager not yet wired (Task B1 / C3).")
    return dm


def _ai(server: "AnnotateServer") -> Any:
    """Return the AIAssist instance attached to *server*, or raise NotImplementedError."""
    ai = getattr(server, "ai_assist", None)
    if ai is None:
        raise NotImplementedError("AIAssist not yet wired (Task E1 / C3).")
    return ai


def _coco_state(server: "AnnotateServer") -> dict[str, dict[str, Any]]:
    """Return the per-dataset COCO cache attached to *server*."""
    state = getattr(server, "coco_state", None)
    if state is None:
        state = {}
        setattr(server, "coco_state", state)
    return state


# Pattern for API image URLs: /api/datasets/<dataset>/images/<filename...>
_API_IMAGE_URL_RE = re.compile(
    r"^/?(?:api/)?datasets/(?P<dataset>[^/]+)/images/(?P<filename>.+)$"
)


def _resolve_assist_image_path(server: "AnnotateServer", image_path: str) -> str:
    """Resolve *image_path* to a real filesystem path.

    The frontend sends API-style URLs (``/api/datasets/<ds>/images/<file>``)
    for VLM and CLIP assist requests.  This helper detects that pattern and
    resolves the filename to an absolute path using ``_resolve_image_path``
    (which includes a rglob fallback for images nested in sub-directories).

    If *image_path* does not match the API URL pattern it is returned
    unchanged, allowing direct filesystem paths from API callers.

    Raises:
        FileNotFoundError: If the image cannot be found in the dataset.
    """
    m = _API_IMAGE_URL_RE.match(image_path)
    if m:
        from urllib.parse import unquote as _unquote
        from mata.annotate.dataset_manager import _resolve_image_path
        dataset = _unquote(m.group("dataset"))
        filename = _unquote(m.group("filename"))
        return str(_resolve_image_path(_dm(server), dataset, filename))
    return image_path


def _rank_annotation_paths(paths: list[Any]) -> list[Any]:
    """Sort candidate COCO JSON files by preferred editable filename.

    Priority:
      0 — annotations/instances.json (canonical MATA file)
      1 — annotations/instances*.json
      2 — train/_annotations*.json  (Roboflow primary split)
      3 — valid/_annotations*.json
      4 — test/_annotations*.json
      5 — everything else
    """
    _SPLIT_ORDER = {"train": 2, "val": 3, "valid": 3, "test": 4}

    def sort_key(path: Any) -> tuple[int, str]:
        name = str(getattr(path, "name", path)).lower()
        parent = str(getattr(path, "parent", "")).lower()
        parent_name = parent.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if name == "instances.json" and "annotations" in parent:
            return 0, name
        if name.startswith("instances"):
            return 1, name
        split_rank = _SPLIT_ORDER.get(parent_name)
        if split_rank is not None:
            return split_rank, name
        return 5, name

    return sorted(paths, key=sort_key)


def _annotation_path(server: "AnnotateServer", dataset: str, *, create: bool = False) -> Any:
    """Resolve the persisted COCO JSON file for *dataset*.

    Existing files are preferred over creating a new ``instances.json`` so split
    datasets such as ``coco_mini`` load a real annotation file immediately.
    """
    dm = _dm(server)
    state = _coco_state(server)
    cached = state.get(dataset)
    if cached:
        cached_path = dm._safe_resolve(dataset, "annotations", cached["path"])
        if cached_path.exists() or create:
            return cached_path

    preferred = dm._safe_resolve(dataset, "annotations", "instances.json")
    if preferred.exists():
        return preferred

    candidates: list[Any] = []
    annotation_dir = dm._safe_resolve(dataset, "annotations")
    if annotation_dir.is_dir():
        candidates.extend(path for path in annotation_dir.glob("*.json") if path.is_file())

    dataset_dir = dm._safe_resolve(dataset)
    candidates.extend(path for path in dataset_dir.glob("*.json") if path.is_file())

    # Roboflow style: _annotations.coco.json inside split dirs (train/valid/test)
    _SPLIT_NAMES = frozenset({"train", "val", "test", "valid"})
    for split_dir in sorted(dataset_dir.iterdir()):
        if split_dir.is_dir() and split_dir.name.lower() in _SPLIT_NAMES:
            candidates.extend(p for p in split_dir.glob("*.json") if p.is_file())

    ranked = _rank_annotation_paths(candidates)
    if ranked:
        return ranked[0]

    if create:
        preferred.parent.mkdir(parents=True, exist_ok=True)
    return preferred


def _load_dataset_coco(server: "AnnotateServer", dataset: str) -> tuple[Any, dict]:
    """Load and cache the COCO document for *dataset*.

    For Roboflow-style datasets whose annotation files live inside split
    subdirectories (``train/``, ``valid/``, ``test/``), all per-split COCO
    JSONs are merged so that annotations for val and test images are visible
    alongside train annotations.
    """
    from mata.annotate import coco_io
    from mata.annotate.dataset_manager import _merge_split_coco_jsons

    path = _annotation_path(server, dataset)
    state = _coco_state(server)
    cached = state.get(dataset)
    if cached and cached.get("path") == path.name:
        return path, cached["coco"]

    # When the selected annotation file lives inside a split dir, merge all
    # split-dir COCO JSONs so that val/test annotations are visible too.
    _SPLIT_NAMES = frozenset({"train", "val", "valid", "test"})
    if path.parent.name.lower() in _SPLIT_NAMES:
        dataset_dir = _dm(server)._safe_resolve(dataset)
        merged = _merge_split_coco_jsons(dataset_dir)
        if merged is not None:
            state[dataset] = {"path": path.name, "coco": merged}
            return path, merged

    if path.exists():
        coco = coco_io.load_annotations(path)
    else:
        coco = coco_io.create_empty_coco()

    state[dataset] = {"path": path.name, "coco": coco}
    return path, coco


def _save_dataset_coco(server: "AnnotateServer", dataset: str, coco: dict) -> Any:
    """Persist and cache the COCO document for *dataset*."""
    from mata.annotate import coco_io

    path = _annotation_path(server, dataset, create=True)
    coco_io.save_annotations(coco, path)
    _coco_state(server)[dataset] = {"path": path.name, "coco": coco}
    return path


# ---------------------------------------------------------------------------
# Dataset routes
# ---------------------------------------------------------------------------

def _handle_datasets(server: "AnnotateServer", method: str, parts: list[str], body: dict, query: dict | None = None) -> tuple[int, Any]:
    """Handle all /api/datasets/... routes."""
    if query is None:
        query = {}
    # GET /api/datasets
    if method == "GET" and len(parts) == 1:
        result = _dm(server).list_datasets()
        return 200, result

    # POST /api/datasets  — create new dataset
    if method == "POST" and len(parts) == 1:
        err = _require_fields(body, "name")
        if err:
            return 400, {"error": err, "code": 400}
        name = body["name"]
        if not _valid_dataset_name(name):
            return 400, {"error": f"Invalid dataset name '{name}'. Use alphanumeric, underscore, hyphen (max 64 chars).", "code": 400}
        result = _dm(server).create_dataset(name)
        return 201, result

    # POST /api/datasets/<name>  — create new dataset from route segment
    if method == "POST" and len(parts) == 2:
        dataset = parts[1]
        if not _valid_dataset_name(dataset):
            return 400, {"error": f"Invalid dataset name '{dataset}'.", "code": 400}
        result = _dm(server).create_dataset(dataset)
        return 201, result

    if len(parts) < 2:
        return 404, {"error": "Not found", "code": 404}

    dataset = parts[1]
    if not _valid_dataset_name(dataset):
        return 400, {"error": f"Invalid dataset name '{dataset}'.", "code": 400}

    tail = parts[2:]  # segments after /api/datasets/<name>/

    # GET /api/datasets/<name>/images
    if method == "GET" and tail == ["images"]:
        # Parse pagination/filter/sort params from query string
        try:
            page_raw = query.get("page")
            page = int(page_raw) if page_raw is not None else None
            per_page = int(query.get("per_page", 50))
        except (TypeError, ValueError):
            return 400, {"error": "'page' and 'per_page' must be integers.", "code": 400}
        if per_page <= 0:
            return 400, {"error": "'per_page' must be a positive integer.", "code": 400}
        sort = query.get("sort", "name_asc")
        valid_sorts = {"name_asc", "name_desc", "newest", "oldest", "size"}
        if sort not in valid_sorts:
            return 400, {"error": f"Invalid sort '{sort}'. Must be one of: {', '.join(sorted(valid_sorts))}.", "code": 400}
        split = query.get("split") or None
        annotated = query.get("annotated") or None
        search = query.get("search") or None
        # Load COCO for annotation counts if available
        try:
            _, coco = _load_dataset_coco(server, dataset)
        except Exception:
            coco = None
        result = _dm(server).list_images(
            dataset,
            page=page,
            per_page=per_page,
            sort=sort,
            split=split,
            annotated=annotated,
            search=search,
            coco=coco,
        )
        return 200, result

    # GET /api/datasets/<name>/images/<file>  (file may contain sub-path)
    if method == "GET" and len(tail) >= 2 and tail[0] == "images":
        filename = "/".join(tail[1:])
        data, ct = _dm(server).serve_image(dataset, filename)
        return 200, (data, ct)

    # GET /api/datasets/<name>/thumbnails/<file>  (file may contain sub-path)
    if method == "GET" and len(tail) >= 2 and tail[0] == "thumbnails":
        filename = "/".join(tail[1:])
        data, ct = _dm(server).serve_thumbnail(dataset, filename)
        return 200, (data, ct)

    # GET /api/datasets/<name>/annotations
    # Returns a *slim* envelope — images + categories only (no annotations array).
    # Annotation data can be large (500 MB+ for full COCO) so it is loaded lazily
    # per image via GET /annotations/image/<filename>.
    if method == "GET" and tail == ["annotations"]:
        _, coco = _load_dataset_coco(server, dataset)
        return 200, {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": coco.get("images", []),
            "categories": coco.get("categories", []),
            "annotations": [],
            "annotation_count": len(coco.get("annotations", [])),
        }

    # GET /api/datasets/<name>/annotations/image/<filename>
    # Returns the annotations for a single image (lazy per-image load).
    if method == "GET" and len(tail) >= 3 and tail[0] == "annotations" and tail[1] == "image":
        filename = "/".join(tail[2:])
        base = filename.split("/")[-1] if "/" in filename else filename
        _, coco = _load_dataset_coco(server, dataset)
        image_record = next(
            (img for img in coco.get("images", [])
             if img.get("file_name", "").split("/")[-1] == base
             or img.get("file_name", "") == filename),
            None,
        )
        if image_record is None:
            return 200, {"image_id": None, "annotations": []}
        image_id = image_record["id"]
        anns = [a for a in coco.get("annotations", []) if a.get("image_id") == image_id]
        return 200, {"image_id": image_id, "annotations": anns}

    # POST /api/datasets/<name>/annotations/image/<filename>
    # Replaces annotations for a single image in the stored COCO file.
    if method == "POST" and len(tail) >= 3 and tail[0] == "annotations" and tail[1] == "image":
        filename = "/".join(tail[2:])
        base = filename.split("/")[-1] if "/" in filename else filename
        err = _require_fields(body, "annotations")
        if err:
            return 400, {"error": err, "code": 400}
        new_annotations = body["annotations"]
        if not isinstance(new_annotations, list):
            return 400, {"error": "'annotations' must be a list.", "code": 400}
        _, coco = _load_dataset_coco(server, dataset)
        image_record = next(
            (img for img in coco.get("images", [])
             if img.get("file_name", "").split("/")[-1] == base
             or img.get("file_name", "") == filename),
            None,
        )
        if image_record is None:
            # Auto-create image entry when the COCO file doesn't list it yet
            new_image_id = max((img.get("id", 0) for img in coco.get("images", [])), default=0) + 1
            image_record = {"id": new_image_id, "file_name": filename, "width": 0, "height": 0}
            coco.setdefault("images", []).append(image_record)
        image_id = image_record["id"]
        # Update categories if provided
        if body.get("categories") and isinstance(body["categories"], list):
            coco["categories"] = body["categories"]
        # Replace all annotations for this image
        coco["annotations"] = [
            a for a in coco.get("annotations", []) if a.get("image_id") != image_id
        ]
        for ann in new_annotations:
            ann["image_id"] = image_id  # enforce consistency
            coco["annotations"].append(ann)
        _save_dataset_coco(server, dataset, coco)
        return 200, {"saved": True, "image_id": image_id, "annotation_count": len(new_annotations)}

    # POST /api/datasets/<name>/annotations  — full replace
    if method == "POST" and tail == ["annotations"]:
        err = _require_fields(body, "images", "annotations", "categories")
        if err:
            return 400, {"error": err, "code": 400}
        ann_path = _save_dataset_coco(server, dataset, body)
        return 200, {"saved": True, "path": str(ann_path)}

    # POST /api/datasets/<name>/annotations/add
    if method == "POST" and tail == ["annotations", "add"]:
        err = _require_fields(body, "image_id", "bbox_xywh", "category_id")
        if err:
            return 400, {"error": err, "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        new_id = coco_io.add_annotation(
            coco,
            image_id=int(body["image_id"]),
            bbox_xywh=list(body["bbox_xywh"]),
            category_id=int(body["category_id"]),
            segmentation=body.get("segmentation"),
        )
        _save_dataset_coco(server, dataset, coco)
        return 201, {"id": new_id}

    # DELETE /api/datasets/<name>/annotations/<id>
    if method == "DELETE" and len(tail) == 2 and tail[0] == "annotations":
        try:
            ann_id = int(tail[1])
        except ValueError:
            return 400, {"error": "Annotation ID must be an integer.", "code": 400}
        from mata.annotate import coco_io
        ann_path, coco = _load_dataset_coco(server, dataset)
        if not ann_path.exists():
            return 404, {"error": "No annotations file found.", "code": 404}
        coco_io.remove_annotation(coco, ann_id)
        _save_dataset_coco(server, dataset, coco)
        return 200, {"deleted": ann_id}

    # PATCH /api/datasets/<name>/annotations/<id>  — partial annotation update
    if method == "PATCH" and len(tail) == 2 and tail[0] == "annotations":
        try:
            ann_id = int(tail[1])
        except ValueError:
            return 400, {"error": "Annotation ID must be an integer.", "code": 400}
        err = _validate_patch_body(body)
        if err:
            return 400, {"error": err, "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        try:
            coco_io.update_annotation(coco, ann_id, **body)
        except KeyError:
            return 404, {"error": f"Annotation id={ann_id} not found.", "code": 404}
        _save_dataset_coco(server, dataset, coco)
        return 200, {"updated": ann_id}

    # POST /api/datasets/<name>/export
    if method == "POST" and tail == ["export"]:
        from mata.annotate import coco_io
        ann_path, coco = _load_dataset_coco(server, dataset)
        if not ann_path.exists():
            return 400, {"error": "No annotations to export.", "code": 400}
        dataset_path = _dm(server)._safe_resolve(dataset)
        # Warn if annotation export files already exist and will be overwritten.
        if not body.get("confirm"):
            ann_dir = dataset_path / "annotations"
            existing = [
                f for f in ("instances_train.json", "instances_val.json")
                if (ann_dir / f).is_file()
            ]
            if existing:
                listed = " and ".join(f"'{f}'" for f in existing)
                return 200, {
                    "confirm_required": True,
                    "warning": (
                        f"This will overwrite {listed} in the annotations/ folder. "
                        f"Your images in train/, val/, and test/ will NOT be moved."
                    ),
                }
        class_names = body.get("class_names") or [c["name"] for c in coco.get("categories", [])]
        split_ratio = float(body.get("split_ratio", 0.8))
        try:
            yaml_path, unassigned = coco_io.export_dataset(dataset_path, coco, class_names, split_ratio=split_ratio)
        except ValueError as exc:
            return 400, {"error": str(exc), "code": 400}
        return 200, {"yaml_path": str(yaml_path), "unassigned": unassigned}

    # POST /api/datasets/<name>/rescan  — start background cache rescan
    if method == "POST" and tail == ["rescan"]:
        result = server.start_rescan(dataset)
        if result.get("status") == "not_found":
            return 404, {"error": f"Dataset '{dataset}' not found.", "code": 404}
        return 202, result

    # GET /api/datasets/<name>/rescan  — poll rescan job status
    if method == "GET" and tail == ["rescan"]:
        return 200, server.get_rescan_status(dataset)

    # POST /api/datasets/<name>/redistribute  — start background redistribution
    if method == "POST" and tail == ["redistribute"]:
        err = _require_fields(body, "train", "val", "test")
        if err:
            return 400, {"error": err, "code": 400}
        try:
            train_pct = int(body["train"])
            val_pct = int(body["val"])
            test_pct = int(body["test"])
        except (ValueError, TypeError):
            return 400, {"error": "'train', 'val', 'test' must be integers.", "code": 400}
        if not (0 <= train_pct <= 100 and 0 <= val_pct <= 100 and 0 <= test_pct <= 100):
            return 400, {"error": "Each percentage must be between 0 and 100.", "code": 400}
        if train_pct + val_pct + test_pct != 100:
            return 400, {"error": f"Percentages must sum to 100 (got {train_pct + val_pct + test_pct}).", "code": 400}
        annotated_first = bool(body.get("annotated_first", True))
        seed = body.get("seed")
        result = server.start_redistribute(
            dataset, train_pct, val_pct, test_pct, seed=seed, annotated_first=annotated_first
        )
        if result.get("status") == "not_found":
            return 404, {"error": f"Dataset '{dataset}' not found.", "code": 404}
        return 202, result

    # GET /api/datasets/<name>/redistribute  — poll redistribute job status
    if method == "GET" and tail == ["redistribute"]:
        return 200, server.get_redistribute_status(dataset)

    # GET /api/datasets/<name>/stats
    if method == "GET" and tail == ["stats"]:
        result = _dm(server).get_dataset_info(dataset)
        return 200, result

    # GET /api/datasets/<name>/classes
    if method == "GET" and tail == ["classes"]:
        result = _dm(server).list_classes(dataset)
        return 200, result

    # POST /api/datasets/<name>/reclassify
    if method == "POST" and tail == ["reclassify"]:
        err = _require_fields(body, "filename", "from_class", "to_class")
        if err:
            return 400, {"error": err, "code": 400}
        from_class = body["from_class"]
        to_class = body["to_class"]
        if not _valid_dataset_name(from_class):
            return 400, {"error": f"Invalid class name '{from_class}'.", "code": 400}
        if not _valid_dataset_name(to_class):
            return 400, {"error": f"Invalid class name '{to_class}'.", "code": 400}
        _dm(server).reclassify_image(dataset, body["filename"], from_class, to_class)
        classes = _dm(server).list_classes(dataset)
        return 200, {"moved": True, "classes": classes}

    # PATCH /api/datasets/<name>/images/<filename.../reviewed
    # Body: {"reviewed": true|false}
    if method == "PATCH" and len(tail) >= 3 and tail[0] == "images" and tail[-1] == "reviewed":
        filename = "/".join(tail[1:-1])
        reviewed_val = body.get("reviewed")
        if not isinstance(reviewed_val, bool):
            return 400, {"error": "'reviewed' must be a boolean.", "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        found = coco_io.set_image_reviewed(coco, filename, reviewed_val)
        if not found:
            return 404, {"error": f"Image '{filename}' not found in annotations.", "code": 404}
        _save_dataset_coco(server, dataset, coco)
        return 200, {"updated": True, "filename": filename, "reviewed": reviewed_val}

    # POST /api/datasets/<name>/move
    # Body: {"filename": str, "target_split": "train"|"val"|"test"}
    if method == "POST" and tail == ["move"]:
        err = _require_fields(body, "filename", "target_split")
        if err:
            return 400, {"error": err, "code": 400}
        result = _dm(server).move_to_split(dataset, body["filename"], body["target_split"])
        return 200, result

    # POST /api/datasets/<name>/categories  — create a new category
    # Body: {"name": str, "color"?: str, "supercategory"?: str}
    if method == "POST" and tail == ["categories"]:
        err = _require_fields(body, "name")
        if err:
            return 400, {"error": err, "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        new_id = coco_io.add_category(
            coco,
            name=body["name"],
            supercategory=body.get("supercategory"),
            color=body.get("color"),
        )
        _save_dataset_coco(server, dataset, coco)
        category = next((c for c in coco["categories"] if c["id"] == new_id), None)
        return 201, {"id": new_id, "category": category}

    # PUT /api/datasets/<name>/categories/<id>  — rename / recolor a category
    # Body: {"name"?: str, "color"?: str, "supercategory"?: str}
    if method == "PUT" and len(tail) == 2 and tail[0] == "categories":
        try:
            cat_id = int(tail[1])
        except ValueError:
            return 400, {"error": "Category ID must be an integer.", "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        try:
            updated = coco_io.update_category(
                coco,
                cat_id,
                name=body.get("name"),
                color=body.get("color"),
                supercategory=body.get("supercategory"),
            )
        except KeyError:
            return 404, {"error": f"Category id={cat_id} not found.", "code": 404}
        _save_dataset_coco(server, dataset, coco)
        return 200, {"updated": True, "category": updated}

    # DELETE /api/datasets/<name>/categories/<id>
    # Query param: reassign_to=<int>  (optional)
    if method == "DELETE" and len(tail) == 2 and tail[0] == "categories":
        try:
            cat_id = int(tail[1])
        except ValueError:
            return 400, {"error": "Category ID must be an integer.", "code": 400}
        reassign_to: int | None = None
        raw_reassign = body.get("reassign_to") or query.get("reassign_to")
        if raw_reassign is not None:
            try:
                reassign_to = int(raw_reassign)
            except (ValueError, TypeError):
                return 400, {"error": "'reassign_to' must be an integer.", "code": 400}
        from mata.annotate import coco_io
        _, coco = _load_dataset_coco(server, dataset)
        try:
            affected = coco_io.delete_category(coco, cat_id, reassign_to=reassign_to)
        except KeyError:
            return 404, {"error": f"Category id={cat_id} not found.", "code": 404}
        except ValueError as exc:
            return 400, {"error": str(exc), "code": 400}
        _save_dataset_coco(server, dataset, coco)
        return 200, {"deleted": cat_id, "affected_annotations": affected}

    return 404, {"error": "Not found", "code": 404}


# ---------------------------------------------------------------------------
# AI-assist routes
# ---------------------------------------------------------------------------

def _handle_assist(server: "AnnotateServer", method: str, parts: list[str], body: dict) -> tuple[int, Any]:
    """Handle all /api/assist/... routes."""
    if len(parts) < 2:
        return 404, {"error": "Not found", "code": 404}

    action = parts[1]  # 'detect', 'vlm', 'classify', 'auto-annotate'

    if method == "POST" and action == "detect":
        err = _require_fields(body, "image_path")
        if err:
            return 400, {"error": err, "code": 400}
        try:
            image_path = _resolve_assist_image_path(server, body["image_path"])
        except FileNotFoundError as exc:
            return 404, {"error": str(exc), "code": 404}
        candidates = _ai(server).detect_assist(
            image_path,
            threshold=float(body.get("threshold", 0.3)),
            class_map=body.get("class_map"),
        )
        return 200, {"candidates": candidates}

    if method == "POST" and action == "vlm":
        err = _require_fields(body, "image_path")
        if err:
            return 400, {"error": err, "code": 400}
        try:
            image_path = _resolve_assist_image_path(server, body["image_path"])
        except FileNotFoundError as exc:
            return 404, {"error": str(exc), "code": 404}
        candidates = _ai(server).vlm_assist(
            image_path,
            class_names=body.get("class_names"),
            prompt=body.get("prompt"),
            max_new_tokens=int(body.get("max_new_tokens", 1024)),
        )
        return 200, {"candidates": candidates}

    if method == "POST" and action == "classify":
        err = _require_fields(body, "image_path", "class_names")
        if err:
            return 400, {"error": err, "code": 400}
        try:
            image_path = _resolve_assist_image_path(server, body["image_path"])
        except FileNotFoundError as exc:
            return 404, {"error": str(exc), "code": 404}
        suggestions = _ai(server).clip_classify(
            image_path,
            class_names=list(body["class_names"]),
            top_k=int(body.get("top_k", 5)),
        )
        return 200, {"suggestions": suggestions}

    if method == "POST" and action == "auto-annotate":
        err = _require_fields(body, "dataset", "image_filename")
        if err:
            return 400, {"error": err, "code": 400}
        dataset = body["dataset"]
        if not _valid_dataset_name(dataset):
            return 400, {"error": f"Invalid dataset name '{dataset}'.", "code": 400}
        ai = _ai(server)  # raises NotImplementedError → 501 if not configured
        from mata.annotate.dataset_manager import _resolve_image_path
        try:
            image_path = str(_resolve_image_path(_dm(server), dataset, body["image_filename"]))
        except FileNotFoundError as exc:
            return 404, {"error": str(exc), "code": 404}
        candidates = ai.detect_assist(
            image_path,
            threshold=float(body.get("threshold", 0.3)),
        )
        return 200, {"candidates": candidates}

    if method == "POST" and action == "zeroshot-detect":
        err = _require_fields(body, "image_path", "text_prompts")
        if err:
            return 400, {"error": err, "code": 400}
        try:
            image_path = _resolve_assist_image_path(server, body["image_path"])
        except FileNotFoundError as exc:
            return 404, {"error": str(exc), "code": 404}
        text_prompts = body["text_prompts"]
        if isinstance(text_prompts, list):
            text_prompts = [str(p) for p in text_prompts]
        else:
            text_prompts = str(text_prompts)
        candidates = _ai(server).zeroshot_detect_assist(
            image_path,
            text_prompts=text_prompts,
            threshold=float(body.get("threshold", 0.3)),
            model=body.get("model", "IDEA-Research/grounding-dino-tiny"),
        )
        return 200, {"candidates": candidates}

    return 404, {"error": "Not found", "code": 404}


# ---------------------------------------------------------------------------
# Training routes
# ---------------------------------------------------------------------------

def _handle_train(server: "AnnotateServer", method: str, parts: list[str], body: dict) -> tuple[int, Any]:
    """Handle all /api/train/... routes."""
    # POST /api/train  — start training
    if method == "POST" and len(parts) == 1:
        err = _require_fields(body, "task", "model", "data")
        if err:
            return 400, {"error": err, "code": 400}
        start_training = getattr(server, "start_training", None)
        if start_training is None:
            return 501, {"error": "Training bridge not yet implemented (Task F1).", "code": 501}
        return start_training(body)

    # GET /api/train/status
    if method == "GET" and len(parts) == 2 and parts[1] == "status":
        get_training_status = getattr(server, "get_training_status", None)
        if get_training_status is None:
            return 200, {"status": "idle"}
        return 200, get_training_status()

    # POST /api/train/stop
    if method == "POST" and len(parts) == 2 and parts[1] == "stop":
        stop_training = getattr(server, "stop_training", None)
        if stop_training is None:
            return 200, {"status": "idle"}
        return 200, stop_training()

    return 404, {"error": "Not found", "code": 404}


# ---------------------------------------------------------------------------
# Main dispatch entry point
# ---------------------------------------------------------------------------

def dispatch(
    server: "AnnotateServer",
    method: str,
    path: str,
    body: dict,
) -> tuple[int, Any] | None:
    """Route an ``/api/*`` request to the appropriate handler.

    Returns ``(status_code, response_body)`` where *response_body* is:

    - A ``dict`` — serialised to JSON by the server.
    - A ``(bytes, content_type)`` tuple — sent as a binary response.

    Returns ``None`` for completely unmatched routes (server sends 404).
    """
    # Parse path and query string (path may include a query string from the
    # server's self.path when passed through directly).
    _parsed = urlparse(path)
    clean_path = _parsed.path
    query: dict[str, str] = {k: v[0] for k, v in parse_qs(_parsed.query).items()}

    parts = _parse_path(clean_path)

    if not parts:
        return None

    try:
        section = parts[0]

        if section == "health":
            return 200, {"status": "ok"}

        if section == "datasets":
            return _handle_datasets(server, method, parts, body, query)

        if section == "assist":
            return _handle_assist(server, method, parts, body)

        if section == "train":
            return _handle_train(server, method, parts, body)

    except NotImplementedError as exc:
        return 501, {"error": str(exc), "code": 501}
    except ValueError as exc:
        return 400, {"error": str(exc), "code": 400}
    except PermissionError as exc:
        return 403, {"error": str(exc), "code": 403}
    except KeyError as exc:
        return 404, {"error": str(exc), "code": 404}
    except FileNotFoundError as exc:
        return 404, {"error": str(exc), "code": 404}
    except Exception as exc:  # noqa: BLE001
        from mata.core.logging import get_logger
        get_logger(__name__).exception("Unhandled error in API handler: %s", exc)
        return 500, {"error": "Internal server error", "code": 500}

    return None
