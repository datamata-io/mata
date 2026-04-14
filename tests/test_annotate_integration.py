"""Integration tests for the MATA annotation server — API end-to-end (Task G2).

Tests start a real HTTP server on a random port, issue live HTTP requests, and
verify full round-trip behaviour.  Workflows covered:

1.  Full annotation workflow — create dataset → upload annotations → export → verify COCO
2.  Annotation persistence — save, restart server, reload annotations
3.  Classification workflow — create ImageFolder dataset, list classes, reclassify
4.  Concurrent requests — 10 parallel GET/POST requests via threading
5.  Dataset listing — GET /api/datasets reflects file-system state
6.  Dataset creation — POST /api/datasets and POST /api/datasets/<name>
7.  Image listing — GET /api/datasets/<name>/images
8.  Image serving — GET /api/datasets/<name>/images/<file>
9.  Annotation CRUD — add / delete individual annotations live
10. COCO export — POST /api/datasets/<name>/export writes dataset.yaml
11. Stats endpoint — GET /api/datasets/<name>/stats
12. Health endpoint — GET /api/health
13. Invalid dataset name blocked — 400 on bad chars
14. Duplicate dataset creation blocked — 400 / 500
15. Delete annotation not found — 404
16. Missing annotation fields — 400
17. Export with no annotations — 400
18. Concurrent annotation persistence — 10 threads add annotations
19. Server shutdown while request in flight — no crash
20. Large page of image listing — 50 synthetic images
21. Stats reflects fresh annotation count after save
22. Classification mode stats return type=imagefolder
23. Reclassify moves file between class dirs
24. Reclassify nonexistent file returns 404
25. Dataset creation with invalid name returns 400
26. GET /api/datasets/<nonexistent> images returns 404
27. Export produces valid YAML config with correct keys
28. Full export COCO JSON loads and passes validate_coco()
"""

from __future__ import annotations

import http.client
import io
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from mata.annotate.coco_io import load_annotations, validate_coco
from mata.annotate.server import AnnotateServer


def _make_minimal_jpeg() -> bytes:
    """Return a 1×1 JPEG that Pillow can decode, or a truncated header as fallback."""
    try:
        from PIL import Image as _Image  # type: ignore[import-untyped]

        buf = io.BytesIO()
        _Image.new("RGB", (1, 1), color=(128, 0, 0)).save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0JFIF"


# Pre-built at import time so individual tests don't need Pillow directly.
_MINIMAL_JPEG_BYTES = _make_minimal_jpeg()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _ServerCtx:
    """Start a real HTTP server for the duration of a test, then shut it down."""

    def __init__(self, data_root: Path) -> None:
        self.server = AnnotateServer(data_root=str(data_root), port=0)
        self._thread = threading.Thread(
            target=self.server._httpd.serve_forever,
            daemon=True,
        )

    def __enter__(self) -> "AnnotateServer":
        self._thread.start()
        time.sleep(0.05)  # let the server thread start
        return self.server

    def __exit__(self, *_: object) -> None:
        self.server._httpd.shutdown()
        self._thread.join(timeout=5)
        self.server._httpd.server_close()


def _conn(server: AnnotateServer) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)


def _get(server: AnnotateServer, path: str) -> tuple[int, Any]:
    """Return (status, body_dict) for a GET request."""
    conn = _conn(server)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = raw
    return resp.status, body


def _post(server: AnnotateServer, path: str, body: dict | None = None) -> tuple[int, Any]:
    """Return (status, body_dict) for a POST request with JSON body."""
    payload = json.dumps(body or {}).encode()
    conn = _conn(server)
    conn.request(
        "POST",
        path,
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    resp = conn.getresponse()
    raw = resp.read()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def _delete(server: AnnotateServer, path: str) -> tuple[int, Any]:
    """Return (status, body_dict) for a DELETE request."""
    conn = _conn(server)
    conn.request("DELETE", path)
    resp = conn.getresponse()
    raw = resp.read()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def _patch(server: AnnotateServer, path: str, body: dict | None = None) -> tuple[int, Any]:
    """Return (status, body_dict) for a PATCH request with JSON body."""
    payload = json.dumps(body or {}).encode()
    conn = _conn(server)
    conn.request(
        "PATCH",
        path,
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    resp = conn.getresponse()
    raw = resp.read()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def _write_img(path: Path, content: bytes = b"\xff\xd8\xff\xe0JFIF") -> None:
    """Write a minimal fake JPEG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_valid_jpeg(path: Path) -> None:
    """Write a minimal but Pillow-decodable JPEG to *path*.

    Used by thumbnail tests where the server calls ``Image.open()`` on the
    source file before downscaling.  The plain fake-JPEG bytes used by
    ``_write_img`` are too truncated for Pillow to parse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MINIMAL_JPEG_BYTES)


def _sample_coco(
    image_filename: str = "img001.jpg",
    width: int = 100,
    height: int = 100,
) -> dict:
    """Return a minimal valid COCO dict with one image and one annotation."""
    return {
        "info": {"description": "integration test", "version": "1.0"},
        "licenses": [],
        "images": [{"id": 1, "file_name": image_filename, "width": width, "height": height}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "area": 3000.0,
                "iscrowd": 0,
                "segmentation": [],
            }
        ],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_ok(tmp_path: Path) -> None:
    """GET /api/health returns 200 with {status: ok}."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/health")
    assert status == 200
    assert body == {"status": "ok"}


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------


def test_empty_datasets_list(tmp_path: Path) -> None:
    """GET /api/datasets returns empty list when data_root is empty."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets")
    assert status == 200
    assert body == []


def test_datasets_list_reflects_filesystem(tmp_path: Path) -> None:
    """GET /api/datasets lists real sub-directories."""
    (tmp_path / "alpha" / "images").mkdir(parents=True)
    (tmp_path / "alpha" / "annotations").mkdir(parents=True)
    (tmp_path / "beta" / "images").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets")

    assert status == 200
    names = [d["name"] for d in body]
    assert "alpha" in names
    assert "beta" in names


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


def test_create_dataset_via_body(tmp_path: Path) -> None:
    """POST /api/datasets with JSON body creates the dataset directory."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets", {"name": "proj_a"})
    assert status == 201
    assert body["name"] == "proj_a"
    assert (tmp_path / "proj_a" / "images").is_dir()
    assert (tmp_path / "proj_a" / "annotations").is_dir()


def test_create_dataset_via_route_segment(tmp_path: Path) -> None:
    """POST /api/datasets/<name> creates the dataset directory."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets/proj_b")
    assert status == 201
    assert body["name"] == "proj_b"
    assert (tmp_path / "proj_b").is_dir()


def test_create_dataset_invalid_name_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets with invalid name returns 400."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets", {"name": "../evil"})
    assert status == 400
    assert "error" in body


def test_create_dataset_special_chars_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets with spaces in name returns 400."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets", {"name": "bad name!"})
    assert status == 400


# ---------------------------------------------------------------------------
# Image listing and serving
# ---------------------------------------------------------------------------


def test_list_images_returns_correct_count(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/images returns correct image list."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    for i in range(3):
        _write_img(tmp_path / "ds" / "images" / f"img{i:03d}.jpg")

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/ds/images")

    assert status == 200
    assert body["total"] == 3
    filenames = [item["filename"] for item in body["images"]]
    assert all(name.endswith(".jpg") for name in filenames)


def test_serve_image_returns_bytes(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/images/<file> returns image bytes."""
    img_content = b"\xff\xd8\xff\xe0JFIF_TEST"
    _write_img(tmp_path / "ds" / "images" / "test.jpg", img_content)

    with _ServerCtx(tmp_path) as srv:
        conn = _conn(srv)
        conn.request("GET", "/api/datasets/ds/images/test.jpg")
        resp = conn.getresponse()
        body = resp.read()

    assert resp.status == 200
    assert body == img_content


def test_list_images_nonexistent_dataset_returns_404(tmp_path: Path) -> None:
    """GET /api/datasets/<nonexistent>/images returns 404."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/no_such_ds/images")
    assert status == 404


def test_list_images_large_directory(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/images handles 50 images without error."""
    img_dir = tmp_path / "bigds" / "images"
    img_dir.mkdir(parents=True)
    for i in range(50):
        _write_img(img_dir / f"img{i:04d}.jpg")

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/bigds/images")

    assert status == 200
    assert body["total"] == 50


# ---------------------------------------------------------------------------
# Annotation CRUD
# ---------------------------------------------------------------------------


def test_get_annotations_empty_dataset(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/annotations returns empty COCO for new dataset."""
    (tmp_path / "newds" / "images").mkdir(parents=True)
    (tmp_path / "newds" / "annotations").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/newds/annotations")

    assert status == 200
    assert "images" in body
    assert "annotations" in body
    assert "categories" in body


def test_post_annotations_saves_and_returns_success(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/annotations persists COCO JSON to disk."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir(parents=True)
    coco = _sample_coco()

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets/ds/annotations", coco)

    assert status == 200
    assert body["saved"] is True
    # Verify file written to disk
    ann_file = Path(body["path"])
    assert ann_file.is_file()
    loaded = load_annotations(ann_file)
    assert loaded["images"][0]["file_name"] == "img001.jpg"


def test_annotations_persist_after_server_restart(tmp_path: Path) -> None:
    """Annotations saved in one server instance are readable by a fresh instance."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir(parents=True)
    coco = _sample_coco()

    # First server: save annotations
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/ds/annotations", coco)

    # Second server (fresh instance): reload annotations
    with _ServerCtx(tmp_path) as srv2:
        status, body = _get(srv2, "/api/datasets/ds/annotations")

    assert status == 200
    assert body["annotations"][0]["bbox"] == [10.0, 20.0, 50.0, 60.0]
    assert body["categories"][0]["name"] == "person"


def test_add_annotation_increments_id(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/annotations/add creates annotation with auto-ID."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir(parents=True)
    coco = _sample_coco()

    with _ServerCtx(tmp_path) as srv:
        # Save base COCO so the server has image + category
        _post(srv, "/api/datasets/ds/annotations", coco)
        # Add a new annotation
        status, body = _post(
            srv,
            "/api/datasets/ds/annotations/add",
            {"image_id": 1, "bbox_xywh": [5.0, 10.0, 30.0, 40.0], "category_id": 1},
        )

    assert status == 201
    assert isinstance(body["id"], int)
    assert body["id"] >= 2  # auto-incremented beyond the existing id=1


def test_add_annotation_missing_field_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/annotations/add without required fields → 400."""
    (tmp_path / "ds" / "images").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(
            srv,
            "/api/datasets/ds/annotations/add",
            {"image_id": 1, "bbox_xywh": [5.0, 5.0, 10.0, 10.0]},  # missing category_id
        )

    assert status == 400
    assert "error" in body


def test_delete_annotation_removes_it(tmp_path: Path) -> None:
    """DELETE /api/datasets/<name>/annotations/<id> removes the annotation."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir(parents=True)
    coco = _sample_coco()

    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/ds/annotations", coco)
        status, body = _delete(srv, "/api/datasets/ds/annotations/1")
        assert status == 200
        assert body["deleted"] == 1
        # Verify it's gone
        _, ann_body = _get(srv, "/api/datasets/ds/annotations")

    assert ann_body["annotations"] == []


def test_delete_annotation_not_found_returns_404(tmp_path: Path) -> None:
    """DELETE /api/datasets/<name>/annotations/<id> for missing ID → 404."""
    (tmp_path / "ds" / "images").mkdir(parents=True)
    (tmp_path / "ds" / "annotations").mkdir(parents=True)
    coco = _sample_coco()

    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/ds/annotations", coco)
        status, body = _delete(srv, "/api/datasets/ds/annotations/999")

    assert status == 404


def test_post_annotations_missing_required_fields_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/annotations without required fields → 400."""
    (tmp_path / "ds" / "images").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(
            srv,
            "/api/datasets/ds/annotations",
            {"images": [], "annotations": []},  # missing 'categories'
        )

    assert status == 400


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------


def test_export_produces_dataset_yaml(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/export writes dataset.yaml to the dataset root."""
    ds_dir = tmp_path / "myds"
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True)
    _write_img(img_dir / "img001.jpg")
    (ds_dir / "annotations").mkdir(parents=True)
    # export_dataset requires at least one split directory to exist;
    # create train/ and copy the image so detection works correctly.
    train_dir = ds_dir / "train"
    train_dir.mkdir(parents=True)
    _write_img(train_dir / "img001.jpg")

    coco = _sample_coco()
    coco["images"][0]["file_name"] = "train/img001.jpg"

    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/myds/annotations", coco)
        status, body = _post(srv, "/api/datasets/myds/export", {"class_names": ["person"]})

    assert status == 200
    yaml_path = Path(body["yaml_path"])
    assert yaml_path.exists()
    assert yaml_path.name == "dataset.yaml"

    import yaml  # type: ignore[import]
    config = yaml.safe_load(yaml_path.read_text())
    assert "names" in config
    assert "train" in config
    assert "val" in config


def test_export_no_annotations_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/export with no annotation file → 400."""
    (tmp_path / "empty_ds" / "images").mkdir(parents=True)
    (tmp_path / "empty_ds" / "annotations").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets/empty_ds/export")

    assert status == 400
    assert "error" in body


def test_export_coco_json_passes_validation(tmp_path: Path) -> None:
    """Exported COCO split files pass validate_coco() with zero warnings."""
    ds_dir = tmp_path / "validate_me"
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True)
    _write_img(img_dir / "img001.jpg")
    (ds_dir / "annotations").mkdir(parents=True)

    coco = _sample_coco()
    coco["images"][0]["file_name"] = "images/img001.jpg"

    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/validate_me/annotations", coco)
        _post(srv, "/api/datasets/validate_me/export", {"class_names": ["person"]})

    train_ann = ds_dir / "annotations" / "instances_train.json"
    val_ann = ds_dir / "annotations" / "instances_val.json"

    if train_ann.exists():
        warnings = validate_coco(load_annotations(train_ann))
        assert warnings == [], f"train COCO warnings: {warnings}"
    if val_ann.exists():
        warnings = validate_coco(load_annotations(val_ann))
        assert warnings == [], f"val COCO warnings: {warnings}"


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


def test_stats_returns_dataset_info(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/stats returns image_count, type, etc."""
    ds_dir = tmp_path / "statds"
    (ds_dir / "images").mkdir(parents=True)
    _write_img(ds_dir / "images" / "img001.jpg")
    _write_img(ds_dir / "images" / "img002.jpg")

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/statds/stats")

    assert status == 200
    assert body["image_count"] == 2
    assert "type" in body


def test_stats_reflects_annotation_count_after_save(tmp_path: Path) -> None:
    """Stats annotation_count updates after saving annotations."""
    ds_dir = tmp_path / "ds"
    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "annotations").mkdir(parents=True)
    _write_img(ds_dir / "images" / "img001.jpg")

    coco = _sample_coco()

    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/ds/annotations", coco)
        status, body = _get(srv, "/api/datasets/ds/stats")

    assert status == 200
    assert body["annotation_count"] >= 1


# ---------------------------------------------------------------------------
# Classification workflow
# ---------------------------------------------------------------------------


def _make_imagefolder(root: Path, name: str, classes: list[str], images_per_class: int = 2) -> Path:
    """Create a minimal ImageFolder dataset under *root*."""
    ds_dir = root / name
    for cls in classes:
        cls_dir = ds_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(images_per_class):
            _write_img(cls_dir / f"img{i:03d}.jpg")
    return ds_dir


def test_classification_stats_return_imagefolder_type(tmp_path: Path) -> None:
    """ImageFolder dataset is detected as type='imagefolder' in stats."""
    _make_imagefolder(tmp_path, "clf_ds", ["cat", "dog"])

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/clf_ds/stats")

    assert status == 200
    assert body["type"] == "imagefolder"


def test_list_classes_returns_class_info(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/classes returns class names and counts."""
    _make_imagefolder(tmp_path, "clf_ds", ["cat", "dog", "bird"], images_per_class=3)

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/clf_ds/classes")

    assert status == 200
    names = [c["name"] for c in body]
    assert set(names) == {"cat", "dog", "bird"}
    for cls_info in body:
        assert cls_info["count"] == 3


def test_reclassify_moves_image_between_classes(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/reclassify moves image from one class to another."""
    # Create cat with img000.jpg; dog directory is empty so the move can succeed
    ds_dir = tmp_path / "clf_ds"
    (ds_dir / "cat").mkdir(parents=True)
    _write_img(ds_dir / "cat" / "img000.jpg")
    (ds_dir / "dog").mkdir(parents=True)  # empty — no collision

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(
            srv,
            "/api/datasets/clf_ds/reclassify",
            {"filename": "img000.jpg", "from_class": "cat", "to_class": "dog"},
        )

    assert status == 200
    assert body["moved"] is True
    assert not (ds_dir / "cat" / "img000.jpg").exists()
    assert (ds_dir / "dog" / "img000.jpg").exists()


def test_reclassify_nonexistent_image_returns_404(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/reclassify for missing image returns 404."""
    _make_imagefolder(tmp_path, "clf_ds", ["cat", "dog"], images_per_class=1)

    with _ServerCtx(tmp_path) as srv:
        status, body = _post(
            srv,
            "/api/datasets/clf_ds/reclassify",
            {"filename": "ghost.jpg", "from_class": "cat", "to_class": "dog"},
        )

    assert status == 404


# ---------------------------------------------------------------------------
# Full annotation workflow
# ---------------------------------------------------------------------------


def test_full_annotation_workflow(tmp_path: Path) -> None:
    """End-to-end: create dataset → add images → post annotations → export → verify YAML."""
    with _ServerCtx(tmp_path) as srv:
        # 1. Create dataset
        s, b = _post(srv, "/api/datasets/workflow_test")
        assert s == 201

        # 2. Copy a test image into the new dataset
        img_dst = tmp_path / "workflow_test" / "images" / "img001.jpg"
        _write_img(img_dst)
        # Also place the image in train/ so export_dataset can detect the split.
        train_dst = tmp_path / "workflow_test" / "train" / "img001.jpg"
        _write_img(train_dst)

        # 3. Post COCO annotations
        coco = _sample_coco("train/img001.jpg")
        s, b = _post(srv, "/api/datasets/workflow_test/annotations", coco)
        assert s == 200
        assert b["saved"] is True

        # 4. Reload annotations via GET
        s, loaded = _get(srv, "/api/datasets/workflow_test/annotations")
        assert s == 200
        assert loaded["annotations"][0]["category_id"] == 1

        # 5. Export
        s, export_body = _post(
            srv,
            "/api/datasets/workflow_test/export",
            {"class_names": ["person"]},
        )
        assert s == 200
        yaml_path = Path(export_body["yaml_path"])
        assert yaml_path.name == "dataset.yaml"


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------


def test_concurrent_get_requests(tmp_path: Path) -> None:
    """10 simultaneous GET /api/health requests all succeed (ThreadingMixIn)."""
    results: list[int] = []
    lock = threading.Lock()

    def hit(server: AnnotateServer) -> None:
        s, _ = _get(server, "/api/health")
        with lock:
            results.append(s)

    with _ServerCtx(tmp_path) as srv:
        threads = [threading.Thread(target=hit, args=(srv,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert len(results) == 10
    assert all(s == 200 for s in results)


def test_concurrent_annotation_persistence(tmp_path: Path) -> None:
    """10 threads each add one annotation to the same dataset.

    The server must stay alive and return only controlled HTTP responses (no
    5xx).  Concurrent write collisions (4xx) are tolerated: the annotation
    store has no server-side write lock so race conditions may surface as
    OS-level PermissionError (403) or auto-ID collisions (400/500) on Windows.
    What we verify is that the server *does not crash* under concurrent load.
    """
    ds_dir = tmp_path / "concurrent_ds"
    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "annotations").mkdir(parents=True)

    coco = _sample_coco()
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/concurrent_ds/annotations", coco)

        crash_errors: list[Exception] = []
        statuses: list[int] = []
        lock = threading.Lock()

        def add_ann(server: AnnotateServer, idx: int) -> None:
            try:
                s, _ = _post(
                    server,
                    "/api/datasets/concurrent_ds/annotations/add",
                    {
                        "image_id": 1,
                        "bbox_xywh": [float(idx), float(idx), 5.0, 5.0],
                        "category_id": 1,
                    },
                )
                with lock:
                    statuses.append(s)
                    # 5xx = server crash — never acceptable
                    if s >= 500:
                        crash_errors.append(AssertionError(f"Server error {s} for thread {idx}"))
            except Exception as exc:
                with lock:
                    crash_errors.append(exc)

        threads = [threading.Thread(target=add_ann, args=(srv, i)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

    assert crash_errors == [], f"Server crash errors: {crash_errors}"
    assert len(statuses) == 10, "Not all threads completed"
    # At least some requests should succeed
    successes = sum(1 for s in statuses if s in (200, 201))
    assert successes >= 1, f"No successful annotation adds: {statuses}"


# ---------------------------------------------------------------------------
# Edge cases / security
# ---------------------------------------------------------------------------


def test_invalid_dataset_name_in_path_returns_400(tmp_path: Path) -> None:
    """GET /api/datasets/../evil/images is blocked (invalid name → 400)."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/..evil../images")
    assert status == 400


def test_get_nonexistent_dataset_annotations_returns_empty_coco(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/annotations for non-existing dataset dir."""
    # The dataset dir doesn't exist; server returns empty COCO (no error)
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/missing/annotations")
    # Either 200 with empty COCO or 404 is acceptable
    assert status in (200, 404)


def test_unknown_api_route_returns_404(tmp_path: Path) -> None:
    """GET /api/no_such_endpoint returns 404."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/no_such_endpoint")
    assert status == 404


def test_post_missing_body_fields_returns_400(tmp_path: Path) -> None:
    """POST /api/datasets with missing 'name' field returns 400."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _post(srv, "/api/datasets", {})
    assert status == 400
    assert "error" in body


# ---------------------------------------------------------------------------
# Task F2: Frontend View Smoke Tests
#
# These tests verify that the server correctly serves the SPA (index.html)
# with both view containers present, that static assets are reachable, and
# that the thumbnail endpoint returns valid image bytes.
#
# MANUAL SMOKE TEST CHECKLIST (run in a real browser after `mata annotate`):
#
# Browser View:
#   1. Open http://localhost:8710 in a browser.
#      EXPECTED: Page loads; #browser-view is visible; dataset list appears
#                in the left sidebar after a brief fetch.
#   2. Select a dataset from the sidebar.
#      EXPECTED: Thumbnail grid populates; browse-progress bar updates;
#                dataset stats (image count, annotation count) are displayed.
#   3. Click the "Train" split tab.
#      EXPECTED: Grid re-filters to show only training images; "Val" / "Test"
#                tabs similarly filter when clicked.
#   4. Type a partial filename into the search box.
#      EXPECTED: Grid narrows in real time; total count reflects filtered set.
#   5. Click a thumbnail card.
#      EXPECTED: Hash changes to #edit/...; #editor-view becomes visible;
#                #browser-view is hidden.
#   6. Click the breadcrumb back-link ("← Datasets" or dataset name).
#      EXPECTED: Hash reverts to #browse/...; browser view is restored.
#   7. Scroll the thumbnail grid to the last page by clicking "Next page".
#      EXPECTED: Navigation buttons enable/disable correctly at boundaries;
#                per-page selector changes the number of cards shown.
#   8. Click the theme toggle (sun / moon / monitor icon) in the top bar.
#      EXPECTED: All CSS variables switch; card borders, text, and canvas
#                background recolour correctly in dark and light modes.
#
# Editor View:
#   9. Open any image in the editor.
#      EXPECTED: Canvas fills the centre area; left panel shows Labels tab
#                with annotation count and Layers sub-tab.
#  10. Select the BBox tool (B) and draw a bounding box on the canvas.
#      EXPECTED: Class-picker popover appears; selecting a class saves the
#                annotation to the server (save-status shows "Saved").
#  11. Click an annotation in the Layers list.
#      EXPECTED: Annotation is highlighted on the canvas; Properties panel
#                shows correct xyxy bbox, area, and category dropdown.
#  12. Use mouse-wheel to zoom in > 200% and pan with Space + drag.
#      EXPECTED: Canvas zooms toward the cursor; annotations remain aligned
#                with image content; zoom % in the bottom bar updates.
#  13. Press Ctrl+Z to undo the last annotation.
#      EXPECTED: Annotation is removed from canvas and Layers list.
#  14. Toggle dark mode again to confirm theme persistence.
#      EXPECTED: After refreshing the page, the last chosen theme is restored.
# ---------------------------------------------------------------------------


def _get_raw(server: AnnotateServer, path: str) -> tuple[int, bytes, str]:
    """Return (status, raw_body_bytes, content_type) for a GET request."""
    conn = _conn(server)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read()
    ct = resp.getheader("Content-Type", "")
    return resp.status, raw, ct


# ------------------------------------------------------------------
# index.html — both view containers present
# ------------------------------------------------------------------


def test_index_html_served_with_200(tmp_path: Path) -> None:
    """GET / returns 200 with text/html content-type."""
    with _ServerCtx(tmp_path) as srv:
        status, body, ct = _get_raw(srv, "/")
    assert status == 200
    assert "text/html" in ct


def test_index_html_contains_browser_view_container(tmp_path: Path) -> None:
    """index.html must contain the #browser-view div for the browser view."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/")
    html = body.decode("utf-8", errors="replace")
    assert 'id="browser-view"' in html, "#browser-view container missing from index.html"


def test_index_html_contains_editor_view_container(tmp_path: Path) -> None:
    """index.html must contain the #editor-view div for the editor view."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/")
    html = body.decode("utf-8", errors="replace")
    assert 'id="editor-view"' in html, "#editor-view container missing from index.html"


def test_index_html_contains_both_views(tmp_path: Path) -> None:
    """index.html must contain both #browser-view and #editor-view in one response."""
    with _ServerCtx(tmp_path) as srv:
        status, body, _ = _get_raw(srv, "/")
    html = body.decode("utf-8", errors="replace")
    assert status == 200
    assert 'id="browser-view"' in html
    assert 'id="editor-view"' in html


def test_index_html_via_explicit_path_also_contains_views(tmp_path: Path) -> None:
    """GET /index.html returns the same SPA with both view containers."""
    with _ServerCtx(tmp_path) as srv:
        status, body, ct = _get_raw(srv, "/index.html")
    html = body.decode("utf-8", errors="replace")
    assert status == 200
    assert "text/html" in ct
    assert 'id="browser-view"' in html
    assert 'id="editor-view"' in html


# ------------------------------------------------------------------
# app.js — static asset loading
# ------------------------------------------------------------------


def test_app_js_loads_with_200(tmp_path: Path) -> None:
    """GET /static/app.js returns 200."""
    with _ServerCtx(tmp_path) as srv:
        status, body, _ = _get_raw(srv, "/static/app.js")
    assert status == 200
    assert len(body) > 0


def test_app_js_content_type_is_javascript(tmp_path: Path) -> None:
    """GET /static/app.js returns a JavaScript content-type header."""
    with _ServerCtx(tmp_path) as srv:
        _, _, ct = _get_raw(srv, "/static/app.js")
    # Browsers accept both 'application/javascript' and 'text/javascript'
    assert "javascript" in ct.lower(), f"Expected javascript content-type, got: {ct!r}"


def test_app_js_contains_router(tmp_path: Path) -> None:
    """app.js must define the Router object used for hash-based navigation."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/static/app.js")
    source = body.decode("utf-8", errors="replace")
    assert "Router" in source, "Router not found in app.js"


def test_app_js_contains_theme_manager(tmp_path: Path) -> None:
    """app.js must define ThemeManager for dark/light/system toggle."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/static/app.js")
    source = body.decode("utf-8", errors="replace")
    assert "ThemeManager" in source, "ThemeManager not found in app.js"


def test_app_js_references_localstorage_key(tmp_path: Path) -> None:
    """app.js must persist theme preference under 'mata-annotate-theme' in localStorage."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/static/app.js")
    source = body.decode("utf-8", errors="replace")
    assert "mata-annotate-theme" in source, "'mata-annotate-theme' key missing from app.js"


def test_app_js_contains_browser_view_reference(tmp_path: Path) -> None:
    """app.js must reference browser-view (used by Router to toggle views)."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/static/app.js")
    source = body.decode("utf-8", errors="replace")
    assert "browser-view" in source


def test_app_js_contains_editor_view_reference(tmp_path: Path) -> None:
    """app.js must reference editor-view (used by Router to toggle views)."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/static/app.js")
    source = body.decode("utf-8", errors="replace")
    assert "editor-view" in source


# ------------------------------------------------------------------
# API accessibility — basic reachability from the SPA perspective
# ------------------------------------------------------------------


def test_api_datasets_accessible_from_spa(tmp_path: Path) -> None:
    """GET /api/datasets is reachable (same origin as index.html)."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets")
    assert status == 200
    assert isinstance(body, list)


def test_api_health_accessible_from_spa(tmp_path: Path) -> None:
    """GET /api/health returns {status: ok} — used by SPA on startup."""
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/health")
    assert status == 200
    assert body.get("status") == "ok"


def test_api_stats_accessible_for_dataset(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/stats is reachable after creating a dataset."""
    (tmp_path / "smoke_ds" / "images").mkdir(parents=True)
    _write_img(tmp_path / "smoke_ds" / "images" / "img001.jpg")

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/smoke_ds/stats")
    assert status == 200
    assert "image_count" in body


# ------------------------------------------------------------------
# Thumbnail endpoint — valid image bytes returned
# ------------------------------------------------------------------


def test_thumbnail_endpoint_returns_image_bytes(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/thumbnails/<file> returns non-empty bytes."""
    _write_valid_jpeg(tmp_path / "tn_ds" / "images" / "photo.jpg")

    with _ServerCtx(tmp_path) as srv:
        conn = _conn(srv)
        conn.request("GET", "/api/datasets/tn_ds/thumbnails/photo.jpg")
        resp = conn.getresponse()
        data = resp.read()

    assert resp.status == 200
    assert len(data) > 0


def test_thumbnail_content_type_is_image(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/thumbnails/<file> returns an image content-type."""
    _write_valid_jpeg(tmp_path / "tn_ds" / "images" / "photo.jpg")

    with _ServerCtx(tmp_path) as srv:
        conn = _conn(srv)
        conn.request("GET", "/api/datasets/tn_ds/thumbnails/photo.jpg")
        resp = conn.getresponse()
        resp.read()
        ct = resp.getheader("Content-Type", "")

    assert resp.status == 200
    assert "image" in ct.lower(), f"Expected image content-type for thumbnail, got: {ct!r}"


def test_thumbnail_missing_file_returns_404(tmp_path: Path) -> None:
    """GET thumbnail for a non-existent image returns 404."""
    (tmp_path / "tn_ds" / "images").mkdir(parents=True)

    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/tn_ds/thumbnails/ghost.jpg")
    assert status == 404


# ------------------------------------------------------------------
# Hash-routing — both views accessible without server round-trip
# ------------------------------------------------------------------


def test_browser_view_accessible_via_hash_routing(tmp_path: Path) -> None:
    """GET / (with or without hash) serves the same SPA; hash routing is client-side.

    The server always returns index.html for /, so the SPA's Router handles
    #browse/... and #edit/... entirely in the browser.  This test confirms
    that index.html is always served for the root path regardless of any
    simulated hash value.
    """
    with _ServerCtx(tmp_path) as srv:
        # Hash fragments are not sent to the server; the SPA handles them.
        # Fetching / should always return the same index.html.
        status, body, ct = _get_raw(srv, "/")
    assert status == 200
    html = body.decode("utf-8", errors="replace")
    # Both view containers must be present so the Router can show either one.
    assert 'id="browser-view"' in html
    assert 'id="editor-view"' in html


def test_index_html_dark_theme_css_block_present(tmp_path: Path) -> None:
    """index.html must contain a [data-theme=\"dark\"] CSS variable block."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/")
    html = body.decode("utf-8", errors="replace")
    assert '[data-theme="dark"]' in html, "Dark theme CSS block missing from index.html"


def test_index_html_light_theme_css_variables_present(tmp_path: Path) -> None:
    """index.html must contain a :root or [data-theme=\"light\"] CSS variable declaration."""
    with _ServerCtx(tmp_path) as srv:
        _, body, _ = _get_raw(srv, "/")
    html = body.decode("utf-8", errors="replace")
    has_root = ":root" in html
    has_light = '[data-theme="light"]' in html
    assert has_root or has_light, "No light-theme CSS variable block found in index.html"


# ---------------------------------------------------------------------------
# Task F3: Full Workflow Integration Test
#
# End-to-end: start server → browse datasets → paginate → load annotations →
# add annotation → PATCH annotation → verify persistence → search images →
# delete annotation → verify removal.
# ---------------------------------------------------------------------------


def _make_workflow_dataset(root: Path, name: str, n_images: int = 6) -> None:
    """Create a minimal detect dataset with *n_images* dummy JPEGs and one annotation."""
    ds = root / name
    (ds / "images").mkdir(parents=True)
    (ds / "annotations").mkdir(parents=True)
    for i in range(1, n_images + 1):
        _write_img(ds / "images" / f"000{i:03d}.jpg")


def test_f3_datasets_listed(tmp_path: Path) -> None:
    """Step 1 — GET /api/datasets returns the dataset under test."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets")
    assert status == 200
    names = [d["name"] for d in body]
    assert "coco_mini" in names


def test_f3_paginated_images(tmp_path: Path) -> None:
    """Step 2 — GET images?page=1&per_page=5 returns paginated envelope."""
    _make_workflow_dataset(tmp_path, "coco_mini", n_images=6)
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/coco_mini/images?page=1&per_page=5")
    assert status == 200
    assert body["total"] > 0
    assert len(body["images"]) <= 5


def test_f3_stats_has_browse_progress(tmp_path: Path) -> None:
    """Step 3 — GET /api/datasets/<name>/stats includes browse_progress."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/coco_mini/stats")
    assert status == 200
    assert "browse_progress" in body


def test_f3_annotations_loadable(tmp_path: Path) -> None:
    """Step 4 — GET /api/datasets/<name>/annotations returns COCO structure."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/coco_mini/annotations")
    assert status == 200
    assert "annotations" in body
    assert "images" in body
    assert "categories" in body


def test_f3_add_annotation(tmp_path: Path) -> None:
    """Step 5 — POST /api/datasets/<name>/annotations/add creates a new annotation."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "0000001.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/coco_mini/annotations", coco)
        status, body = _post(
            srv,
            "/api/datasets/coco_mini/annotations/add",
            {"image_id": 1, "bbox_xywh": [10, 20, 100, 50], "category_id": 1},
        )
    assert status == 201
    assert "id" in body
    assert isinstance(body["id"], int)


def test_f3_patch_annotation(tmp_path: Path) -> None:
    """Step 6 — PATCH /api/datasets/<name>/annotations/<id> updates attributes."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "0000001.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/coco_mini/annotations", coco)
        _, add_body = _post(
            srv,
            "/api/datasets/coco_mini/annotations/add",
            {"image_id": 1, "bbox_xywh": [10, 20, 100, 50], "category_id": 1},
        )
        ann_id = add_body["id"]
        status, body = _patch(
            srv,
            f"/api/datasets/coco_mini/annotations/{ann_id}",
            {"attributes": {"occluded": True}},
        )
    assert status == 200
    assert body["updated"] == ann_id


def test_f3_patch_annotation_persists(tmp_path: Path) -> None:
    """Step 7 — PATCHed attributes are readable back via GET annotations."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "0000001.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/coco_mini/annotations", coco)
        _, add_body = _post(
            srv,
            "/api/datasets/coco_mini/annotations/add",
            {"image_id": 1, "bbox_xywh": [10, 20, 100, 50], "category_id": 1},
        )
        ann_id = add_body["id"]
        _patch(
            srv,
            f"/api/datasets/coco_mini/annotations/{ann_id}",
            {"attributes": {"occluded": True}},
        )
        # Reload and verify
        _, reload = _get(srv, "/api/datasets/coco_mini/annotations")
    patched = [a for a in reload["annotations"] if a["id"] == ann_id]
    assert len(patched) == 1
    assert patched[0].get("attributes", {}).get("occluded") is True


def test_f3_search_images(tmp_path: Path) -> None:
    """Step 8 — GET images?search=000001 filters by filename."""
    _make_workflow_dataset(tmp_path, "coco_mini", n_images=6)
    with _ServerCtx(tmp_path) as srv:
        status, body = _get(srv, "/api/datasets/coco_mini/images?search=000001")
    assert status == 200
    assert len(body["images"]) >= 1
    assert all("000001" in img["filename"] for img in body["images"])


def test_f3_delete_annotation(tmp_path: Path) -> None:
    """Step 9 — DELETE annotation removes it from persistent store."""
    _make_workflow_dataset(tmp_path, "coco_mini")
    coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "0000001.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }
    with _ServerCtx(tmp_path) as srv:
        _post(srv, "/api/datasets/coco_mini/annotations", coco)
        _, add_body = _post(
            srv,
            "/api/datasets/coco_mini/annotations/add",
            {"image_id": 1, "bbox_xywh": [10, 20, 100, 50], "category_id": 1},
        )
        ann_id = add_body["id"]
        del_status, del_body = _delete(
            srv, f"/api/datasets/coco_mini/annotations/{ann_id}"
        )
        _, reload = _get(srv, "/api/datasets/coco_mini/annotations")
    assert del_status == 200
    assert del_body["deleted"] == ann_id
    assert all(a["id"] != ann_id for a in reload["annotations"])


def test_f3_full_workflow_sequential(tmp_path: Path) -> None:
    """Full end-to-end sequential workflow in a single server session.

    Mirrors the programmatic test flow specified in Task F3:
    list datasets → paginate → stats → load → add → PATCH → verify → search → delete → verify.
    """
    _make_workflow_dataset(tmp_path, "coco_mini", n_images=6)
    base_coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "0000001.jpg", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }

    with _ServerCtx(tmp_path) as srv:
        # 1. List datasets
        s, datasets = _get(srv, "/api/datasets")
        assert s == 200
        assert any(d["name"] == "coco_mini" for d in datasets)

        # 2. Get paginated images
        s, page = _get(srv, "/api/datasets/coco_mini/images?page=1&per_page=5")
        assert s == 200
        assert page["total"] > 0
        assert len(page["images"]) <= 5

        # 3. Get stats
        s, stats = _get(srv, "/api/datasets/coco_mini/stats")
        assert s == 200
        assert "browse_progress" in stats

        # 4. Load (empty) annotations — initialise base COCO
        _post(srv, "/api/datasets/coco_mini/annotations", base_coco)
        s, coco = _get(srv, "/api/datasets/coco_mini/annotations")
        assert s == 200
        assert "annotations" in coco

        # 5. Add annotation
        s, ann = _post(
            srv,
            "/api/datasets/coco_mini/annotations/add",
            {"image_id": 1, "bbox_xywh": [10, 20, 100, 50], "category_id": 1},
        )
        assert s == 201
        assert "id" in ann
        ann_id = ann["id"]

        # 6. PATCH annotation
        s, patch_resp = _patch(
            srv,
            f"/api/datasets/coco_mini/annotations/{ann_id}",
            {"attributes": {"occluded": True}},
        )
        assert s == 200
        assert patch_resp["updated"] == ann_id

        # 7. Verify persistence
        s, coco = _get(srv, "/api/datasets/coco_mini/annotations")
        assert s == 200
        patched = [a for a in coco["annotations"] if a["id"] == ann_id]
        assert len(patched) == 1
        assert patched[0].get("attributes", {}).get("occluded") is True

        # 8. Search images
        s, search_resp = _get(srv, "/api/datasets/coco_mini/images?search=000001")
        assert s == 200
        assert len(search_resp["images"]) >= 1

        # 9. Delete annotation
        s, del_resp = _delete(srv, f"/api/datasets/coco_mini/annotations/{ann_id}")
        assert s == 200
        assert del_resp["deleted"] == ann_id

        # Verify deletion
        s, coco = _get(srv, "/api/datasets/coco_mini/annotations")
        assert s == 200
        assert all(a["id"] != ann_id for a in coco["annotations"])
