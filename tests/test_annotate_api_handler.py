from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mata.annotate.ai_assist import AIAssist
from mata.annotate import api_handler, coco_io
from mata.annotate.dataset_manager import DatasetManager
from mata.annotate.server import AnnotateServer


def _make_server(data_root: Path) -> SimpleNamespace:
    import threading as _threading
    from mata.annotate.dataset_manager import _run_rescan_worker as _rrw

    rescan_jobs: dict = {}
    rescan_lock = _threading.Lock()

    def _start_rescan(name: str) -> dict:
        dm = DatasetManager(data_root)
        dataset_dir = dm._safe_resolve(name)
        if not dataset_dir.is_dir():
            return {"status": "not_found"}
        with rescan_lock:
            if rescan_jobs.get(name, {}).get("status") == "running":
                return {"status": "already_running"}
            rescan_jobs[name] = {"status": "running"}
        t = _threading.Thread(
            target=_rrw,
            args=(dm, name, rescan_jobs, rescan_lock),
            daemon=True,
        )
        t.start()
        return {"status": "started"}

    def _get_rescan_status(name: str) -> dict:
        with rescan_lock:
            return dict(rescan_jobs.get(name, {"status": "idle"}))

    ns = SimpleNamespace(
        dataset_manager=DatasetManager(data_root),
        coco_state={},
        coco_io=None,
        ai_models={},
        ai_assist=None,
        training_bridge=None,
        _rescan_jobs=rescan_jobs,
        _rescan_lock=rescan_lock,
    )
    ns.start_rescan = _start_rescan
    ns.get_rescan_status = _get_rescan_status
    return ns


def _write_file(path: Path, content: bytes = b"img") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sample_coco() -> dict:
    coco = coco_io.create_empty_coco(
        images=[{"id": 1, "file_name": "000001.jpg", "width": 32, "height": 32}],
        categories=[{"id": 1, "name": "person", "supercategory": "person"}],
    )
    coco["annotations"] = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1, 2, 10, 12],
            "area": 120,
            "iscrowd": 0,
            "segmentation": [],
        }
    ]
    return coco


def _make_coco_dataset(root: Path, name: str = "coco_mini") -> Path:
    dataset_dir = root / name
    _write_file(dataset_dir / "train2017" / "000001.jpg")
    coco_io.save_annotations(_sample_coco(), dataset_dir / "annotations" / "instances_train2017.json")
    return dataset_dir


def test_annotate_server_initializes_backend_state(tmp_path: Path) -> None:
    server = AnnotateServer(data_root=str(tmp_path), port=0)
    try:
        assert isinstance(server.dataset_manager, DatasetManager)
        assert isinstance(server.ai_assist, AIAssist)
        assert server.ai_models == {}
        assert server.coco_state == {}
        assert server.coco_io == server.coco_state
    finally:
        server._httpd.server_close()


def test_dispatch_lists_real_datasets(tmp_path: Path) -> None:
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets", {})

    assert status == 200
    assert len(payload) == 1
    entry = payload[0]
    assert entry["name"] == "coco_mini"
    assert entry["image_count"] == 1
    assert entry["has_annotations"] is True
    assert entry["type"] == "coco"
    assert "cache_valid" in entry  # True if cache file was written, False otherwise


def test_dispatch_creates_dataset_from_route_segment(tmp_path: Path) -> None:
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "POST", "/api/datasets/test_project", {})

    assert status == 201
    assert payload["name"] == "test_project"
    assert (tmp_path / "test_project" / "images").is_dir()
    assert (tmp_path / "test_project" / "annotations").is_dir()


def test_dispatch_loads_existing_annotation_file_from_annotations_dir(tmp_path: Path) -> None:
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/coco_mini/annotations", {})

    assert status == 200
    assert payload["images"][0]["file_name"] == "000001.jpg"
    # GET /annotations returns the full annotations list (D1 fix: removed slim envelope).
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["bbox"] == [1, 2, 10, 12]
    assert payload["annotation_count"] == 1
    # Per-image endpoint also works.
    _, img_payload = api_handler.dispatch(
        server, "GET", "/api/datasets/coco_mini/annotations/image/000001.jpg", {}
    )
    assert img_payload["annotations"][0]["bbox"] == [1, 2, 10, 12]
    assert server.coco_state["coco_mini"]["path"] == "instances_train2017.json"


def test_dispatch_saves_annotations_back_to_existing_file(tmp_path: Path) -> None:
    dataset_dir = _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)
    updated_coco = coco_io.create_empty_coco(
        images=[{"id": 1, "file_name": "000001.jpg", "width": 32, "height": 32}],
        categories=[{"id": 1, "name": "car", "supercategory": "car"}],
    )

    status, payload = api_handler.dispatch(server, "POST", "/api/datasets/coco_mini/annotations", updated_coco)

    saved_path = dataset_dir / "annotations" / "instances_train2017.json"
    assert status == 200
    assert payload["saved"] is True
    assert payload["path"] == str(saved_path)
    assert coco_io.load_annotations(saved_path) == updated_coco


# ---------------------------------------------------------------------------
# Roboflow split-dir COCO: val/test annotations visible via _load_dataset_coco
# ---------------------------------------------------------------------------

def _make_roboflow_dataset(root: Path, name: str = "rfds") -> Path:
    """Create a Roboflow-style dataset with train/ valid/ test/ split COCO JSONs."""
    ds = root / name
    for split, img_name, ann_id, img_id in [
        ("train", "train_img.jpg", 1, 10),
        ("valid", "val_img.jpg",   2, 20),
        ("test",  "test_img.jpg",  3, 30),
    ]:
        (ds / split).mkdir(parents=True, exist_ok=True)
        _write_file(ds / split / img_name)
        coco_doc = {
            "images": [{"id": img_id, "file_name": img_name, "width": 8, "height": 8}],
            "annotations": [
                {"id": ann_id, "image_id": img_id, "category_id": 1,
                 "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0, "segmentation": []}
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        (ds / split / "_annotations.coco.json").write_text(
            json.dumps(coco_doc), encoding="utf-8"
        )
    return ds


def test_load_dataset_coco_merges_all_splits(tmp_path: Path) -> None:
    """_load_dataset_coco returns merged COCO containing train+valid+test images."""
    _make_roboflow_dataset(tmp_path)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/rfds/annotations", {}
    )

    assert status == 200
    assert len(payload["images"]) == 3
    assert payload["annotation_count"] == 3


def test_val_image_annotations_visible_after_merge(tmp_path: Path) -> None:
    """GET /annotations/image/val_img.jpg returns annotations from valid/ split."""
    _make_roboflow_dataset(tmp_path)
    server = _make_server(tmp_path)

    _, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/rfds/annotations/image/val_img.jpg", {}
    )

    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["bbox"] == [0, 0, 4, 4]


def test_test_image_annotations_visible_after_merge(tmp_path: Path) -> None:
    """GET /annotations/image/test_img.jpg returns annotations from test/ split."""
    _make_roboflow_dataset(tmp_path)
    server = _make_server(tmp_path)

    _, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/rfds/annotations/image/test_img.jpg", {}
    )

    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["bbox"] == [0, 0, 4, 4]


def test_merged_coco_is_cached_on_second_request(tmp_path: Path) -> None:
    """Second call to _load_dataset_coco returns the cached merged document."""
    _make_roboflow_dataset(tmp_path)
    server = _make_server(tmp_path)

    api_handler.dispatch(server, "GET", "/api/datasets/rfds/annotations", {})
    cached_coco = server.coco_state["rfds"]["coco"]

    # Second dispatch should return same object (cache hit)
    api_handler.dispatch(server, "GET", "/api/datasets/rfds/annotations", {})
    assert server.coco_state["rfds"]["coco"] is cached_coco


# ---------------------------------------------------------------------------
# Image listing — search query parameter (Task A4)
# ---------------------------------------------------------------------------


def _write_image(path: Path, content: bytes = b"img") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_image_dataset(root: Path, names: list[str], ds_name: str = "imgds") -> Path:
    """Create a flat image dataset with the given filenames."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for name in names:
        _write_image(dataset / "images" / name)
    return dataset


def test_dispatch_images_search_filters_results(tmp_path: Path) -> None:
    """GET /api/datasets/.../images?search=face returns only matching images."""
    _make_image_dataset(tmp_path, ["face_001.jpg", "face_002.jpg", "car_001.jpg"])
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/imgds/images?search=face", {}
    )

    assert status == 200
    assert payload["total"] == 2
    filenames = [img["filename"] for img in payload["images"]]
    assert all("face" in f.lower() for f in filenames)


def test_dispatch_images_search_is_case_insensitive(tmp_path: Path) -> None:
    """search param matching is case-insensitive via the API."""
    _make_image_dataset(tmp_path, ["Face_001.jpg", "FACE_002.jpg", "other.jpg"], "cids")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/cids/images?search=face", {}
    )

    assert status == 200
    assert payload["total"] == 2


def test_dispatch_images_search_total_reflects_filtered_count(tmp_path: Path) -> None:
    """total in the response reflects filtered count, not total dataset size."""
    _make_image_dataset(tmp_path, ["face.jpg", "dog.jpg", "cat.jpg"], "fds")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/fds/images?search=dog", {}
    )

    assert status == 200
    assert payload["total"] == 1
    assert payload["images"][0]["filename"] == "dog.jpg"


def test_dispatch_images_search_combined_with_pagination(tmp_path: Path) -> None:
    """search combines with page/per_page: total reflects filtered, page slices."""
    _make_image_dataset(
        tmp_path,
        ["face_1.jpg", "face_2.jpg", "face_3.jpg", "other.jpg"],
        "pgds",
    )
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pgds/images?search=face&page=1&per_page=2", {}
    )

    assert status == 200
    assert payload["total"] == 3
    assert payload["total_pages"] == 2
    assert len(payload["images"]) == 2


def test_dispatch_images_search_no_match_returns_empty(tmp_path: Path) -> None:
    """search with no matching images returns empty list with total=0."""
    _make_image_dataset(tmp_path, ["apple.jpg", "banana.jpg"], "eds")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/eds/images?search=zzz_nomatch", {}
    )

    assert status == 200
    assert payload["total"] == 0
    assert payload["images"] == []

# ---------------------------------------------------------------------------
# Task D7: Auto-annotate API endpoint  (/api/assist/auto-annotate)
# ---------------------------------------------------------------------------


class _FakeDetectAssist:
    """Minimal stand-in for AIAssist used in D7 tests."""

    def __init__(self, candidates: list) -> None:
        self._candidates = candidates
        self.calls: list[tuple[str, float]] = []
        self.vlm_calls: list[tuple[str, ...]] = []
        self.clip_calls: list[tuple[str, list]] = []

    def detect_assist(self, image_path: str, threshold: float = 0.3, **kwargs) -> list:
        self.calls.append((image_path, threshold))
        return self._candidates

    def vlm_assist(self, image_path: str, **kwargs) -> list:
        self.vlm_calls.append((image_path,))
        return self._candidates

    def clip_classify(self, image_path: str, class_names: list, **kwargs) -> list:
        self.clip_calls.append((image_path, class_names))
        return self._candidates


def _fake_server_with_assist(data_root: Path, assist: "_FakeDetectAssist") -> SimpleNamespace:
    ns = _make_server(data_root)
    ns.ai_assist = assist
    return ns


def test_auto_annotate_missing_dataset_returns_400(tmp_path: Path) -> None:
    server = _fake_server_with_assist(tmp_path, _FakeDetectAssist([]))

    status, payload = api_handler.dispatch(
        server, "POST", "/api/assist/auto-annotate", {"image_filename": "cat.jpg"}
    )

    assert status == 400
    assert "dataset" in payload["error"].lower()


def test_auto_annotate_missing_image_filename_returns_400(tmp_path: Path) -> None:
    server = _fake_server_with_assist(tmp_path, _FakeDetectAssist([]))

    status, payload = api_handler.dispatch(
        server, "POST", "/api/assist/auto-annotate", {"dataset": "myds"}
    )

    assert status == 400
    assert "image_filename" in payload["error"].lower()


def test_auto_annotate_invalid_dataset_name_returns_400(tmp_path: Path) -> None:
    server = _fake_server_with_assist(tmp_path, _FakeDetectAssist([]))

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "../evil", "image_filename": "img.jpg"},
    )

    assert status == 400


def test_auto_annotate_no_ai_assist_configured_returns_501(tmp_path: Path) -> None:
    server = _make_server(tmp_path)  # ai_assist=None

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "img.jpg"},
    )

    assert status == 501


def test_auto_annotate_returns_candidates_from_detect_assist(tmp_path: Path) -> None:
    candidates = [
        {
            "bbox_xywh": [10.0, 20.0, 50.0, 70.0],
            "bbox_xyxy": [10.0, 20.0, 60.0, 90.0],
            "label": "cat",
            "label_id": 1,
            "score": 0.9,
            "source": "detect",
        }
    ]
    assist = _FakeDetectAssist(candidates)
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds" / "cat.jpg").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "cat.jpg").write_bytes(b"x")

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "cat.jpg"},
    )

    assert status == 200
    assert payload["candidates"] == candidates
    assert len(assist.calls) == 1


def test_auto_annotate_default_threshold_is_0_3(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "img.jpg").write_bytes(b"x")

    api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "img.jpg"},
    )

    assert assist.calls[0][1] == pytest.approx(0.3)


def test_auto_annotate_passes_threshold_to_detect_assist(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "img.jpg").write_bytes(b"x")

    api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "img.jpg", "threshold": "0.65"},
    )

    assert assist.calls[0][1] == pytest.approx(0.65)


def test_auto_annotate_image_path_includes_dataset_and_filename(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "frame001.jpg").write_bytes(b"x")

    api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "frame001.jpg"},
    )

    assert len(assist.calls) == 1
    resolved_path = assist.calls[0][0]
    assert "myds" in resolved_path
    assert "frame001.jpg" in resolved_path


def test_auto_annotate_returns_empty_candidates_list(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "img.jpg").write_bytes(b"x")

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "img.jpg"},
    )

    assert status == 200
    assert payload["candidates"] == []


def test_auto_annotate_image_in_subdirectory_resolved_via_rglob(tmp_path: Path) -> None:
    """Images nested in sub-folders (e.g. train/) must be found via rglob fallback."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    # Place the image in a sub-folder to exercise the rglob fallback
    (tmp_path / "myds" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "train" / "nested.jpg").write_bytes(b"x")

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "nested.jpg"},
    )

    assert status == 200
    assert len(assist.calls) == 1
    assert "nested.jpg" in assist.calls[0][0]
    assert "train" in assist.calls[0][0]


def test_auto_annotate_missing_image_file_returns_404(tmp_path: Path) -> None:
    """When the image file does not exist anywhere in the dataset, return 404."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    # Dataset directory exists but image file does not
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/auto-annotate",
        {"dataset": "myds", "image_filename": "ghost.jpg"},
    )

    assert status == 404


# ---------------------------------------------------------------------------
# Task D7: /api/assist/detect, /api/assist/vlm, /api/assist/classify
#           — API URL path resolution
# ---------------------------------------------------------------------------


def test_assist_detect_resolves_api_url_to_filesystem_path(tmp_path: Path) -> None:
    """POST /api/assist/detect with an API URL resolves to the real file path."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "photo.jpg").write_bytes(b"x")

    status, _ = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/detect",
        {"image_path": "/api/datasets/myds/images/photo.jpg"},
    )

    assert status == 200
    assert len(assist.calls) == 1
    assert "photo.jpg" in assist.calls[0][0]
    assert assist.calls[0][0] != "/api/datasets/myds/images/photo.jpg"


def test_assist_detect_api_url_rglob_subdirectory(tmp_path: Path) -> None:
    """API URL path for /api/assist/detect finds images nested in sub-folders."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "train" / "deep.jpg").write_bytes(b"x")

    status, _ = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/detect",
        {"image_path": "/api/datasets/myds/images/deep.jpg"},
    )

    assert status == 200
    assert "deep.jpg" in assist.calls[0][0]
    assert "train" in assist.calls[0][0]


def test_assist_detect_api_url_missing_file_returns_404(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/detect",
        {"image_path": "/api/datasets/myds/images/ghost.jpg"},
    )

    assert status == 404


def test_assist_vlm_resolves_api_url_to_filesystem_path(tmp_path: Path) -> None:
    """POST /api/assist/vlm with an API URL resolves to the real file path."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "scene.jpg").write_bytes(b"x")

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/vlm",
        {"image_path": "/api/datasets/myds/images/scene.jpg"},
    )

    assert status == 200
    assert len(assist.vlm_calls) == 1
    assert "scene.jpg" in assist.vlm_calls[0][0]
    assert assist.vlm_calls[0][0] != "/api/datasets/myds/images/scene.jpg"


def test_assist_vlm_api_url_rglob_subdirectory(tmp_path: Path) -> None:
    """VLM endpoint finds images nested in sub-folders."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds" / "valid").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "valid" / "vlm_img.jpg").write_bytes(b"x")

    status, _ = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/vlm",
        {"image_path": "/api/datasets/myds/images/vlm_img.jpg"},
    )

    assert status == 200
    assert "vlm_img.jpg" in assist.vlm_calls[0][0]
    assert "valid" in assist.vlm_calls[0][0]


def test_assist_vlm_api_url_missing_file_returns_404(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/vlm",
        {"image_path": "/api/datasets/myds/images/ghost.jpg"},
    )

    assert status == 404


def test_assist_classify_resolves_api_url_to_filesystem_path(tmp_path: Path) -> None:
    """POST /api/assist/classify with an API URL resolves to the real file path."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "item.jpg").write_bytes(b"x")

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/classify",
        {
            "image_path": "/api/datasets/myds/images/item.jpg",
            "class_names": ["cat", "dog"],
        },
    )

    assert status == 200
    assert len(assist.clip_calls) == 1
    assert "item.jpg" in assist.clip_calls[0][0]
    assert assist.clip_calls[0][0] != "/api/datasets/myds/images/item.jpg"


def test_assist_classify_api_url_rglob_subdirectory(tmp_path: Path) -> None:
    """CLIP endpoint finds images nested in sub-folders."""
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds" / "test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "myds" / "test" / "clip_img.jpg").write_bytes(b"x")

    status, _ = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/classify",
        {
            "image_path": "/api/datasets/myds/images/clip_img.jpg",
            "class_names": ["cat", "dog"],
        },
    )

    assert status == 200
    assert "clip_img.jpg" in assist.clip_calls[0][0]
    assert "test" in assist.clip_calls[0][0]


def test_assist_classify_api_url_missing_file_returns_404(tmp_path: Path) -> None:
    assist = _FakeDetectAssist([])
    server = _fake_server_with_assist(tmp_path, assist)
    (tmp_path / "myds").mkdir(parents=True, exist_ok=True)

    status, payload = api_handler.dispatch(
        server,
        "POST",
        "/api/assist/classify",
        {
            "image_path": "/api/datasets/myds/images/ghost.jpg",
            "class_names": ["cat"],
        },
    )

    assert status == 404


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Pagination helpers
# ---------------------------------------------------------------------------


def _make_pageable_dataset(root: Path, count: int, ds_name: str = "pageds") -> Path:
    """Create a flat image dataset with *count* numbered image files."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        _write_file(dataset / "images" / f"img_{i:03d}.jpg", bytes([i % 256]) * max(1, i))
    return dataset


def _make_split_image_dataset(root: Path, ds_name: str = "splitds") -> Path:
    """Create a dataset with train/val/test split directories."""
    dataset = root / ds_name
    _write_file(dataset / "train" / "train_01.jpg")
    _write_file(dataset / "train" / "train_02.jpg")
    _write_file(dataset / "val" / "val_01.jpg")
    _write_file(dataset / "test" / "test_01.jpg")
    return dataset


def _make_size_varied_dataset(root: Path, ds_name: str = "sizeds") -> Path:
    """Create a dataset with images of different known sizes for sort tests."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_file(dataset / "images" / "b_medium.jpg", b"x" * 100)
    _write_file(dataset / "images" / "a_small.jpg", b"x" * 10)
    _write_file(dataset / "images" / "c_large.jpg", b"x" * 500)
    return dataset


def _make_annotated_image_dataset(root: Path, ds_name: str = "annds") -> Path:
    """Create a dataset with one annotated (2 anns) and one unannotated image."""
    dataset = root / ds_name
    _write_file(dataset / "images" / "annotated.jpg")
    _write_file(dataset / "images" / "unannotated.jpg")
    ann_coco = coco_io.create_empty_coco(
        images=[{"id": 1, "file_name": "annotated.jpg", "width": 32, "height": 32}],
        categories=[{"id": 1, "name": "cat", "supercategory": "cat"}],
    )
    ann_coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
         "area": 100, "iscrowd": 0, "segmentation": []},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 8, 8],
         "area": 64, "iscrowd": 0, "segmentation": []},
    ]
    coco_io.save_annotations(ann_coco, dataset / "annotations" / "instances.json")
    return dataset


def _make_stats_dataset(root: Path, ds_name: str = "statsds") -> Path:
    """Dataset with train/val dirs and a COCO that annotates only t1.jpg."""
    dataset = root / ds_name
    _write_file(dataset / "train" / "t1.jpg", b"x" * 50)
    _write_file(dataset / "train" / "t2.jpg", b"x" * 50)
    _write_file(dataset / "val" / "v1.jpg", b"x" * 50)
    stats_coco = coco_io.create_empty_coco(
        images=[
            {"id": 1, "file_name": "t1.jpg", "width": 32, "height": 32},
            {"id": 2, "file_name": "t2.jpg", "width": 32, "height": 32},
            {"id": 3, "file_name": "v1.jpg", "width": 32, "height": 32},
        ],
        categories=[{"id": 1, "name": "cat", "supercategory": "cat"}],
    )
    stats_coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
         "area": 100, "iscrowd": 0, "segmentation": []},
    ]
    coco_io.save_annotations(stats_coco, dataset / "annotations" / "instances.json")
    return dataset


def _setup_patch_dataset(root: Path, ds_name: str = "patchds") -> Path:
    """Create a COCO dataset suitable for PATCH annotation tests."""
    dataset = root / ds_name
    _write_file(dataset / "images" / "img.jpg")
    patch_coco = coco_io.create_empty_coco(
        images=[{"id": 1, "file_name": "img.jpg", "width": 64, "height": 64}],
        categories=[
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "car"},
        ],
    )
    patch_coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60],
         "area": 3000, "iscrowd": 0, "segmentation": []},
    ]
    coco_io.save_annotations(patch_coco, dataset / "annotations" / "instances.json")
    return dataset


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Pagination
# ---------------------------------------------------------------------------


def test_dispatch_images_no_params_returns_all_in_envelope(tmp_path: Path) -> None:
    """Default GET /images returns all images wrapped in pagination envelope."""
    _make_pageable_dataset(tmp_path, 5, "pg_a1")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/pg_a1/images", {})

    assert status == 200
    assert payload["total"] == 5
    assert payload["page"] == 1
    assert payload["per_page"] == 5
    assert payload["total_pages"] == 1
    assert len(payload["images"]) == 5


def test_dispatch_images_pagination_first_page_returns_correct_count(tmp_path: Path) -> None:
    """page=1&per_page=3 returns exactly 3 images."""
    _make_pageable_dataset(tmp_path, 10, "pg_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a2/images?page=1&per_page=3", {}
    )

    assert status == 200
    assert len(payload["images"]) == 3
    assert payload["per_page"] == 3


def test_dispatch_images_pagination_second_page_returns_next_slice(tmp_path: Path) -> None:
    """page=2&per_page=3 returns a disjoint set from page=1."""
    _make_pageable_dataset(tmp_path, 9, "pg_a3")
    server = _make_server(tmp_path)

    _, p1 = api_handler.dispatch(server, "GET", "/api/datasets/pg_a3/images?page=1&per_page=3", {})
    _, p2 = api_handler.dispatch(server, "GET", "/api/datasets/pg_a3/images?page=2&per_page=3", {})

    p1_names = {img["filename"] for img in p1["images"]}
    p2_names = {img["filename"] for img in p2["images"]}
    assert p1_names.isdisjoint(p2_names), "Pages must not overlap"


def test_dispatch_images_pagination_total_and_total_pages_correct(tmp_path: Path) -> None:
    """total and total_pages are computed correctly (7 images, per_page=3 → 3 pages)."""
    _make_pageable_dataset(tmp_path, 7, "pg_a4")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a4/images?page=1&per_page=3", {}
    )

    assert status == 200
    assert payload["total"] == 7
    assert payload["total_pages"] == 3


def test_dispatch_images_pagination_last_page_has_fewer_items(tmp_path: Path) -> None:
    """Last page returns the remainder when total is not divisible by per_page."""
    _make_pageable_dataset(tmp_path, 7, "pg_a5")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a5/images?page=3&per_page=3", {}
    )

    assert status == 200
    assert len(payload["images"]) == 1  # 7 - 6 = 1 image on last page


def test_dispatch_images_pagination_out_of_range_page_clamped(tmp_path: Path) -> None:
    """page beyond total_pages is clamped to the last page (returns images)."""
    _make_pageable_dataset(tmp_path, 5, "pg_a6")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a6/images?page=999&per_page=3", {}
    )

    assert status == 200
    assert len(payload["images"]) >= 1  # clamped to last page, not empty


def test_dispatch_images_pagination_non_integer_page_returns_400(tmp_path: Path) -> None:
    """Non-integer page param returns 400."""
    _make_pageable_dataset(tmp_path, 3, "pg_a7")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a7/images?page=abc", {}
    )

    assert status == 400


def test_dispatch_images_pagination_non_integer_per_page_returns_400(tmp_path: Path) -> None:
    """Non-integer per_page param returns 400."""
    _make_pageable_dataset(tmp_path, 3, "pg_a8")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a8/images?per_page=xyz", {}
    )

    assert status == 400


def test_dispatch_images_pagination_zero_per_page_returns_400(tmp_path: Path) -> None:
    """per_page=0 returns 400 (must be positive)."""
    _make_pageable_dataset(tmp_path, 3, "pg_a9")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a9/images?per_page=0", {}
    )

    assert status == 400


def test_dispatch_images_pagination_negative_per_page_returns_400(tmp_path: Path) -> None:
    """per_page=-1 returns 400 (must be positive)."""
    _make_pageable_dataset(tmp_path, 3, "pg_a10")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a10/images?per_page=-1", {}
    )

    assert status == 400


def test_dispatch_images_pagination_envelope_has_all_required_keys(tmp_path: Path) -> None:
    """Pagination envelope always includes images, total, page, per_page, total_pages."""
    _make_pageable_dataset(tmp_path, 2, "pg_a11")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a11/images?page=1&per_page=5", {}
    )

    assert status == 200
    for key in ("images", "total", "page", "per_page", "total_pages"):
        assert key in payload, f"Missing pagination key: {key}"


def test_dispatch_images_page_1_per_page_10_returns_10(tmp_path: Path) -> None:
    """page=1&per_page=10 returns exactly 10 images from a larger dataset."""
    _make_pageable_dataset(tmp_path, 25, "pg_a12")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a12/images?page=1&per_page=10", {}
    )

    assert status == 200
    assert len(payload["images"]) == 10
    assert payload["total"] == 25


def test_dispatch_images_page_2_per_page_10_returns_next_10(tmp_path: Path) -> None:
    """page=2&per_page=10 returns images 11-20; no overlap with page 1."""
    _make_pageable_dataset(tmp_path, 25, "pg_a13")
    server = _make_server(tmp_path)

    _, p1 = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a13/images?page=1&per_page=10", {}
    )
    _, p2 = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a13/images?page=2&per_page=10", {}
    )

    assert len(p2["images"]) == 10
    p1_files = {img["filename"] for img in p1["images"]}
    p2_files = {img["filename"] for img in p2["images"]}
    assert not p1_files & p2_files


def test_dispatch_images_backward_compat_no_page_returns_total(tmp_path: Path) -> None:
    """Without page param, total equals len(images) (backward compatible)."""
    _make_pageable_dataset(tmp_path, 8, "pg_a14")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/pg_a14/images", {})

    assert status == 200
    assert payload["total"] == len(payload["images"])


def test_dispatch_images_pagination_empty_dataset_returns_envelope(tmp_path: Path) -> None:
    """Empty dataset with pagination returns envelope with total=0, images=[]."""
    (tmp_path / "pg_a15" / "images").mkdir(parents=True, exist_ok=True)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/pg_a15/images?page=1&per_page=10", {}
    )

    assert status == 200
    assert payload["total"] == 0
    assert payload["images"] == []


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Split filter
# ---------------------------------------------------------------------------


def test_dispatch_images_split_train_returns_only_train(tmp_path: Path) -> None:
    """split=train returns only images whose path contains 'train'."""
    _make_split_image_dataset(tmp_path, "sp_a1")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a1/images?split=train", {}
    )

    assert status == 200
    assert payload["total"] == 2
    for img in payload["images"]:
        assert img["split"] == "train"


def test_dispatch_images_split_val_returns_only_val(tmp_path: Path) -> None:
    """split=val returns only images in the val directory."""
    _make_split_image_dataset(tmp_path, "sp_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a2/images?split=val", {}
    )

    assert status == 200
    assert payload["total"] == 1
    assert payload["images"][0]["split"] == "val"


def test_dispatch_images_split_test_returns_only_test(tmp_path: Path) -> None:
    """split=test returns only images in the test directory."""
    _make_split_image_dataset(tmp_path, "sp_a3")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a3/images?split=test", {}
    )

    assert status == 200
    assert payload["total"] == 1
    assert payload["images"][0]["split"] == "test"


def test_dispatch_images_split_invalid_value_returns_empty(tmp_path: Path) -> None:
    """split=nonexistent_split returns empty list (no images match)."""
    _make_split_image_dataset(tmp_path, "sp_a4")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a4/images?split=nonexistent", {}
    )

    assert status == 200
    assert payload["total"] == 0
    assert payload["images"] == []


def test_dispatch_images_per_image_split_field_is_set(tmp_path: Path) -> None:
    """Each image in a split dataset has its split field correctly populated."""
    _make_split_image_dataset(tmp_path, "sp_a5")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/sp_a5/images", {})

    assert status == 200
    split_values = {img["split"] for img in payload["images"]}
    assert "train" in split_values
    assert "val" in split_values
    assert "test" in split_values


def test_dispatch_images_no_split_param_returns_all(tmp_path: Path) -> None:
    """Without split param, all 4 images are returned regardless of split."""
    _make_split_image_dataset(tmp_path, "sp_a6")
    server = _make_server(tmp_path)

    _, all_payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a6/images", {}
    )
    _, train_payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a6/images?split=train", {}
    )

    assert all_payload["total"] == 4
    assert all_payload["total"] > train_payload["total"]


def test_dispatch_images_split_combined_with_pagination(tmp_path: Path) -> None:
    """split=train + page/per_page: total reflects split count, page slices it."""
    ds = tmp_path / "sp_a7"
    for i in range(5):
        _write_file(ds / "train" / f"t_{i}.jpg")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/sp_a7/images?split=train&page=1&per_page=3", {}
    )

    assert status == 200
    assert payload["total"] == 5
    assert len(payload["images"]) == 3
    for img in payload["images"]:
        assert img["split"] == "train"


def test_dispatch_images_flat_images_dir_split_is_none(tmp_path: Path) -> None:
    """Images in a flat /images/ directory have split=None."""
    _make_image_dataset(tmp_path, ["flat.jpg"], "sp_a8")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/sp_a8/images", {})

    assert status == 200
    assert payload["images"][0]["split"] is None


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Sort
# ---------------------------------------------------------------------------


def test_dispatch_images_sort_name_asc_is_alphabetical(tmp_path: Path) -> None:
    """sort=name_asc returns filenames in ascending alphabetical order."""
    _make_size_varied_dataset(tmp_path, "so_a1")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/so_a1/images?sort=name_asc", {}
    )

    assert status == 200
    filenames = [img["filename"] for img in payload["images"]]
    assert filenames == sorted(filenames)


def test_dispatch_images_sort_name_desc_is_reverse_alpha(tmp_path: Path) -> None:
    """sort=name_desc returns filenames in descending alphabetical order."""
    _make_size_varied_dataset(tmp_path, "so_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/so_a2/images?sort=name_desc", {}
    )

    assert status == 200
    filenames = [img["filename"] for img in payload["images"]]
    assert filenames == sorted(filenames, reverse=True)


def test_dispatch_images_sort_size_largest_first(tmp_path: Path) -> None:
    """sort=size returns images in descending size_bytes order."""
    _make_size_varied_dataset(tmp_path, "so_a3")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/so_a3/images?sort=size", {}
    )

    assert status == 200
    sizes = [img["size_bytes"] for img in payload["images"]]
    assert sizes == sorted(sizes, reverse=True)
    assert payload["images"][0]["filename"] == "c_large.jpg"


def test_dispatch_images_sort_newest_returns_200(tmp_path: Path) -> None:
    """sort=newest is accepted and returns 200 with correct count."""
    _make_size_varied_dataset(tmp_path, "so_a4")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/so_a4/images?sort=newest", {}
    )

    assert status == 200
    assert len(payload["images"]) == 3


def test_dispatch_images_invalid_sort_returns_400(tmp_path: Path) -> None:
    """An invalid sort value (e.g. 'random') returns 400."""
    _make_image_dataset(tmp_path, ["a.jpg"], "so_a5")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/so_a5/images?sort=random", {}
    )

    assert status == 400
    assert "sort" in payload["error"].lower()


def test_dispatch_images_default_sort_is_name_asc(tmp_path: Path) -> None:
    """Without sort param, default order is name_asc (alphabetical ascending)."""
    _make_size_varied_dataset(tmp_path, "so_a6")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/so_a6/images", {})

    assert status == 200
    filenames = [img["filename"] for img in payload["images"]]
    assert filenames == sorted(filenames)


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Annotation count
# ---------------------------------------------------------------------------


def test_dispatch_images_annotation_count_positive_for_annotated(tmp_path: Path) -> None:
    """annotated.jpg has annotation_count > 0 because it's in the loaded COCO."""
    _make_annotated_image_dataset(tmp_path, "ac_a1")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/ac_a1/images", {})

    assert status == 200
    img = next(img for img in payload["images"] if img["filename"] == "annotated.jpg")
    assert img["annotation_count"] > 0


def test_dispatch_images_annotation_count_zero_for_unannotated(tmp_path: Path) -> None:
    """unannotated.jpg has annotation_count=0 (not referenced in COCO)."""
    _make_annotated_image_dataset(tmp_path, "ac_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/ac_a2/images", {})

    assert status == 200
    img = next(img for img in payload["images"] if img["filename"] == "unannotated.jpg")
    assert img["annotation_count"] == 0


def test_dispatch_images_annotation_count_correct_number(tmp_path: Path) -> None:
    """annotation_count reflects the exact number of annotations (2 for annotated.jpg)."""
    _make_annotated_image_dataset(tmp_path, "ac_a3")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/ac_a3/images", {})

    assert status == 200
    img = next(img for img in payload["images"] if img["filename"] == "annotated.jpg")
    assert img["annotation_count"] == 2


def test_dispatch_images_annotation_count_key_present_always(tmp_path: Path) -> None:
    """annotation_count key is present in every image entry."""
    _make_image_dataset(tmp_path, ["plain.jpg"], "ac_a4")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/ac_a4/images", {})

    assert status == 200
    assert all("annotation_count" in img for img in payload["images"])


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Stats endpoint
# ---------------------------------------------------------------------------


def test_dispatch_stats_returns_200(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/stats returns HTTP 200."""
    _make_coco_dataset(tmp_path, "st_a1")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(server, "GET", "/api/datasets/st_a1/stats", {})

    assert status == 200


def test_dispatch_stats_includes_required_keys(tmp_path: Path) -> None:
    """Stats response includes total_annotated, total_unannotated, browse_progress."""
    _make_stats_dataset(tmp_path, "st_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a2/stats", {})

    assert status == 200
    for key in ("total_annotated", "total_unannotated", "browse_progress"):
        assert key in payload, f"Missing key: {key}"


def test_dispatch_stats_browse_progress_is_correct_percentage(tmp_path: Path) -> None:
    """browse_progress is (annotated/total)*100; 1 of 3 images annotated → ~33.3%."""
    _make_stats_dataset(tmp_path, "st_a3")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a3/stats", {})

    assert status == 200
    assert payload["total_annotated"] == 1
    assert payload["total_unannotated"] == 2
    assert pytest.approx(payload["browse_progress"], abs=0.2) == 33.3


def test_dispatch_stats_splits_dict_has_correct_counts(tmp_path: Path) -> None:
    """splits dict contains correct per-split total counts."""
    _make_stats_dataset(tmp_path, "st_a4")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a4/stats", {})

    assert status == 200
    assert "splits" in payload
    assert payload["splits"]["train"]["total"] == 2
    assert payload["splits"]["val"]["total"] == 1


def test_dispatch_stats_total_size_bytes_is_positive_integer(tmp_path: Path) -> None:
    """total_size_bytes is a positive integer when dataset has images."""
    _make_stats_dataset(tmp_path, "st_a5")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a5/stats", {})

    assert status == 200
    assert isinstance(payload["total_size_bytes"], int)
    assert payload["total_size_bytes"] > 0


def test_dispatch_stats_type_field_matches_detect_dataset_type(tmp_path: Path) -> None:
    """type field in stats matches detect_dataset_type() for the same dataset."""
    _make_coco_dataset(tmp_path, "st_a6")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a6/stats", {})

    assert status == 200
    assert "type" in payload
    expected_type = server.dataset_manager.detect_dataset_type("st_a6")
    assert payload["type"] == expected_type


def test_dispatch_stats_empty_dataset_dir_returns_zeros(tmp_path: Path) -> None:
    """An existing but empty dataset directory returns zeros for all count fields."""
    (tmp_path / "st-a7").mkdir()
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st-a7/stats", {})

    assert status == 200
    assert payload["image_count"] == 0
    assert payload["total_annotated"] == 0
    assert payload["browse_progress"] == 0.0


def test_dispatch_stats_browse_progress_100_when_all_annotated(tmp_path: Path) -> None:
    """browse_progress == 100.0 when every image has at least one annotation."""
    dataset = tmp_path / "st_a8"
    _write_file(dataset / "images" / "img1.jpg")
    full_coco = coco_io.create_empty_coco(
        images=[{"id": 1, "file_name": "img1.jpg", "width": 32, "height": 32}],
        categories=[{"id": 1, "name": "cat", "supercategory": "cat"}],
    )
    full_coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
         "area": 25, "iscrowd": 0, "segmentation": []},
    ]
    coco_io.save_annotations(full_coco, dataset / "annotations" / "instances.json")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/st_a8/stats", {})

    assert status == 200
    assert payload["browse_progress"] == 100.0
    assert payload["total_annotated"] == 1
    assert payload["total_unannotated"] == 0


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — PATCH annotation
# ---------------------------------------------------------------------------


def test_dispatch_patch_category_id_updates_annotation(tmp_path: Path) -> None:
    """PATCH with category_id=2 changes the annotation's category."""
    _setup_patch_dataset(tmp_path, "pt_a1")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a1/annotations", {})

    status, payload = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a1/annotations/1", {"category_id": 2}
    )

    assert status == 200
    assert payload["updated"] == 1
    _, img_payload = api_handler.dispatch(server, "GET", "/api/datasets/pt_a1/annotations/image/img.jpg", {})
    ann = next(a for a in img_payload["annotations"] if a["id"] == 1)
    assert ann["category_id"] == 2


def test_dispatch_patch_attributes_merges_correctly(tmp_path: Path) -> None:
    """PATCH with attributes dict stores the attributes on the annotation."""
    _setup_patch_dataset(tmp_path, "pt_a2")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a2/annotations", {})

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a2/annotations/1",
        {"attributes": {"occluded": True, "truncated": False}},
    )

    assert status == 200
    _, img_payload = api_handler.dispatch(server, "GET", "/api/datasets/pt_a2/annotations/image/img.jpg", {})
    ann = next(a for a in img_payload["annotations"] if a["id"] == 1)
    assert ann["attributes"]["occluded"] is True
    assert ann["attributes"]["truncated"] is False


def test_dispatch_patch_id_in_body_is_silently_ignored(tmp_path: Path) -> None:
    """PATCH body with 'id' field does not change the annotation's identity."""
    _setup_patch_dataset(tmp_path, "pt_a3")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a3/annotations", {})

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a3/annotations/1",
        {"id": 999, "category_id": 2},
    )

    assert status == 200
    _, img_payload = api_handler.dispatch(server, "GET", "/api/datasets/pt_a3/annotations/image/img.jpg", {})
    ids = [a["id"] for a in img_payload["annotations"]]
    assert 1 in ids
    assert 999 not in ids


def test_dispatch_patch_nonexistent_annotation_returns_404(tmp_path: Path) -> None:
    """PATCH on an annotation id that doesn't exist returns 404."""
    _setup_patch_dataset(tmp_path, "pt_a4")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a4/annotations", {})

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a4/annotations/9999", {"category_id": 1}
    )

    assert status == 404


def test_dispatch_patch_invalid_body_category_id_type_returns_400(tmp_path: Path) -> None:
    """PATCH with category_id as string (not int) returns 400."""
    _setup_patch_dataset(tmp_path, "pt_a5")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a5/annotations", {})

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a5/annotations/1",
        {"category_id": "not-an-int"},
    )

    assert status == 400


def test_dispatch_patch_bbox_updates_bbox_field(tmp_path: Path) -> None:
    """PATCH with a valid bbox list updates the annotation's bbox."""
    _setup_patch_dataset(tmp_path, "pt_a6")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a6/annotations", {})

    new_bbox = [5, 10, 30, 40]
    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a6/annotations/1", {"bbox": new_bbox}
    )

    assert status == 200
    _, img_payload = api_handler.dispatch(server, "GET", "/api/datasets/pt_a6/annotations/image/img.jpg", {})
    ann = next(a for a in img_payload["annotations"] if a["id"] == 1)
    assert ann["bbox"] == new_bbox


def test_dispatch_patch_non_integer_ann_id_in_url_returns_400(tmp_path: Path) -> None:
    """PATCH URL with a non-integer annotation ID returns 400."""
    _setup_patch_dataset(tmp_path, "pt_a7")
    server = _make_server(tmp_path)

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a7/annotations/abc", {"category_id": 1}
    )

    assert status == 400


def test_dispatch_patch_prototype_injection_key_returns_400(tmp_path: Path) -> None:
    """PATCH body with __proto__ key is rejected as 400."""
    _setup_patch_dataset(tmp_path, "pt_a8")
    server = _make_server(tmp_path)
    api_handler.dispatch(server, "GET", "/api/datasets/pt_a8/annotations", {})

    status, _ = api_handler.dispatch(
        server, "PATCH", "/api/datasets/pt_a8/annotations/1",
        {"__proto__": {"admin": True}},
    )

    assert status == 400


# ---------------------------------------------------------------------------
# Task F1: Backend API Tests — Query string parsing
# ---------------------------------------------------------------------------


def test_dispatch_images_query_string_in_path_is_parsed(tmp_path: Path) -> None:
    """Query string directly in the URL path is parsed and applied (search)."""
    _make_image_dataset(tmp_path, ["alpha.jpg", "beta.jpg", "alpha_2.jpg"], "qp_a1")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/qp_a1/images?search=alpha", {}
    )

    assert status == 200
    assert payload["total"] == 2


def test_dispatch_images_no_query_string_uses_defaults(tmp_path: Path) -> None:
    """Request without query string defaults: all images, name_asc, no filter."""
    _make_image_dataset(tmp_path, ["z.jpg", "a.jpg"], "qp_a2")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(server, "GET", "/api/datasets/qp_a2/images", {})

    assert status == 200
    assert payload["total"] == 2
    filenames = [img["filename"] for img in payload["images"]]
    assert filenames == sorted(filenames)


def test_dispatch_images_multiple_query_params_applied_together(tmp_path: Path) -> None:
    """search + sort + page are all applied simultaneously."""
    ds = tmp_path / "qp_a3"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    _write_file(ds / "images" / "face_b.jpg", b"x" * 100)
    _write_file(ds / "images" / "face_a.jpg", b"x" * 200)
    _write_file(ds / "images" / "dog.jpg", b"x" * 50)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server,
        "GET",
        "/api/datasets/qp_a3/images?search=face&sort=name_asc&page=1&per_page=1",
        {},
    )

    assert status == 200
    assert payload["total"] == 2  # 2 face images total
    assert len(payload["images"]) == 1
    assert "face" in payload["images"][0]["filename"].lower()


def test_dispatch_images_all_valid_sort_values_return_200(tmp_path: Path) -> None:
    """All documented sort values (name_asc, name_desc, newest, oldest, size) return 200."""
    _make_image_dataset(tmp_path, ["a.jpg", "b.jpg"], "qp_a4")
    server = _make_server(tmp_path)

    for sort_val in ("name_asc", "name_desc", "newest", "oldest", "size"):
        status, _ = api_handler.dispatch(
            server, "GET", f"/api/datasets/qp_a4/images?sort={sort_val}", {}
        )
        assert status == 200, f"Expected 200 for sort={sort_val}, got {status}"


def test_dispatch_images_query_search_case_insensitive_via_url(tmp_path: Path) -> None:
    """Case-insensitive search applied correctly when passed as URL query string."""
    _make_image_dataset(tmp_path, ["UPPER_001.jpg", "lower_001.jpg"], "qp_a5")
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/qp_a5/images?search=upper", {}
    )

    assert status == 200
    assert payload["total"] == 1
    assert "UPPER" in payload["images"][0]["filename"]


# ---------------------------------------------------------------------------
# Rescan API — cache + background worker
# ---------------------------------------------------------------------------


def test_dispatch_rescan_post_returns_202(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/rescan returns 202 for a known dataset."""
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "POST", "/api/datasets/coco_mini/rescan", {}
    )

    assert status == 202
    assert payload.get("status") in {"started", "already_running"}


def test_dispatch_rescan_post_unknown_dataset_returns_404(tmp_path: Path) -> None:
    """POST /api/datasets/<name>/rescan returns 404 when the dataset directory does not exist."""
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "POST", "/api/datasets/nonexistent_xyz/rescan", {}
    )

    assert status == 404
    assert "error" in payload


def test_dispatch_rescan_get_returns_idle_before_any_start(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/rescan returns idle when no rescan has been started."""
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    status, payload = api_handler.dispatch(
        server, "GET", "/api/datasets/coco_mini/rescan", {}
    )

    assert status == 200
    assert payload.get("status") == "idle"


def test_dispatch_rescan_get_returns_done_after_worker_finishes(tmp_path: Path) -> None:
    """GET /api/datasets/<name>/rescan returns done+cache after worker completes."""
    from mata.annotate.dataset_manager import _run_rescan_worker
    import threading

    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    # Run worker synchronously (no thread) to test the result shape
    jobs: dict = {}
    lock = threading.Lock()
    _run_rescan_worker(server.dataset_manager, "coco_mini", jobs, lock)

    assert jobs["coco_mini"]["status"] == "done"
    cache = jobs["coco_mini"]["cache"]
    assert cache["image_count"] >= 0
    assert "type" in cache
    assert "has_annotations" in cache
    assert "last_scanned" in cache


def test_dispatch_rescan_post_twice_returns_already_running(tmp_path: Path) -> None:
    """Second POST /rescan while a job is running returns already_running."""
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    # Manually mark as running
    with server._rescan_lock:
        server._rescan_jobs["coco_mini"] = {"status": "running"}

    status, payload = api_handler.dispatch(
        server, "POST", "/api/datasets/coco_mini/rescan", {}
    )

    assert status == 202
    assert payload.get("status") == "already_running"


def test_list_datasets_includes_cache_valid_field(tmp_path: Path) -> None:
    """list_datasets() always returns a cache_valid field for every entry."""
    _make_coco_dataset(tmp_path)
    server = _make_server(tmp_path)

    datasets = server.dataset_manager.list_datasets()

    assert len(datasets) == 1
    assert "cache_valid" in datasets[0]
    assert isinstance(datasets[0]["cache_valid"], bool)


def test_list_datasets_cache_valid_true_after_write(tmp_path: Path) -> None:
    """list_datasets() returns cache_valid=True when a valid cache file exists."""
    _make_coco_dataset(tmp_path)
    dm = DatasetManager(tmp_path)

    # Write cache
    dm.write_dataset_cache("coco_mini", {
        "image_count": 42,
        "type": "coco",
        "has_annotations": True,
        "last_scanned": "2025-01-01T00:00:00+00:00",
    })

    datasets = dm.list_datasets()
    entry = next(d for d in datasets if d["name"] == "coco_mini")
    assert entry["cache_valid"] is True
    assert entry["image_count"] == 42
