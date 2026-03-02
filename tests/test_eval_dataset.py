"""Unit tests for DatasetLoader and GroundTruth (Task D1).

Test coverage:
- GroundTruth dataclass construction and field types
- DatasetLoader.from_yaml() — YAML path resolution, names, iteration
- DatasetLoader.from_coco_json() — standalone mode
- DatasetLoader.__len__() — image count
- COCO xywh → xyxy conversion
- Category 1-indexing → 0-indexing
- Missing path raises FileNotFoundError with full path
- GroundTruth.masks is None for detection-only annotations
- names property from YAML and from COCO JSON
- Depth GT loading (.npy files)
- Edge cases: empty dataset, images without annotations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from mata.eval.dataset import DatasetLoader, GroundTruth, _xywh_to_xyxy

# ---------------------------------------------------------------------------
# Fixtures — COCO JSON builders
# ---------------------------------------------------------------------------

_SAMPLE_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
]


def _make_coco_json(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> dict:
    """Return a minimal COCO-JSON dict."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories or _SAMPLE_CATEGORIES,
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.dump(data), encoding="utf-8")


def _touch_image(dir_path: Path, name: str) -> Path:
    """Create an empty file to act as a placeholder image."""
    p = dir_path / name
    p.write_bytes(b"")
    return p


@pytest.fixture()
def simple_coco(tmp_path: Path):
    """
    Minimal COCO dataset on disk:
        tmp_path/
            images/        ← two placeholder images
            annotations/instances.json
    Returns (images_dir, json_path, coco_dict).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    _touch_image(images_dir, "000001.jpg")
    _touch_image(images_dir, "000002.jpg")

    coco = _make_coco_json(
        images=[
            {"id": 1, "file_name": "000001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "000002.jpg", "width": 320, "height": 240},
        ],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "area": 3000.0,
                "segmentation": [],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [100.0, 150.0, 80.0, 40.0],
                "area": 3200.0,
                "segmentation": [],
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 3,
                "bbox": [5.0, 5.0, 30.0, 30.0],
                "area": 900.0,
                "segmentation": [],
                "iscrowd": 0,
            },
        ],
    )
    json_path = ann_dir / "instances.json"
    _write_json(json_path, coco)
    return images_dir, json_path, coco


@pytest.fixture()
def seg_coco(tmp_path: Path):
    """COCO dataset with real segmentation polygons."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    _touch_image(images_dir, "img1.jpg")

    coco = _make_coco_json(
        images=[{"id": 10, "file_name": "img1.jpg", "width": 200, "height": 200}],
        annotations=[
            {
                "id": 1,
                "image_id": 10,
                "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "area": 2500.0,
                "segmentation": [[10.0, 10.0, 60.0, 10.0, 60.0, 60.0, 10.0, 60.0]],
                "iscrowd": 0,
            }
        ],
    )
    json_path = ann_dir / "seg_instances.json"
    _write_json(json_path, coco)
    return images_dir, json_path


@pytest.fixture()
def yaml_dataset(tmp_path: Path, simple_coco):
    """Full YAML-driven dataset fixture."""
    images_dir, json_path, coco = simple_coco
    root = tmp_path

    yaml_data = {
        "path": str(root),
        "val": "images",
        "annotations": "annotations/instances.json",
        "names": {0: "person", 1: "bicycle", 2: "car"},
    }
    yaml_path = tmp_path / "dataset.yaml"
    _write_yaml(yaml_path, yaml_data)
    return yaml_path, images_dir, json_path


# ---------------------------------------------------------------------------
# Unit-level helpers
# ---------------------------------------------------------------------------


class TestXywhToXyxy:
    def test_basic_conversion(self) -> None:
        result = _xywh_to_xyxy([10.0, 20.0, 50.0, 60.0])
        assert result == [10.0, 20.0, 60.0, 80.0]

    def test_zero_width_height(self) -> None:
        result = _xywh_to_xyxy([5.0, 5.0, 0.0, 0.0])
        assert result == [5.0, 5.0, 5.0, 5.0]

    def test_fractional_values(self) -> None:
        result = _xywh_to_xyxy([0.5, 0.5, 10.5, 20.5])
        assert pytest.approx(result) == [0.5, 0.5, 11.0, 21.0]


# ---------------------------------------------------------------------------
# GroundTruth dataclass
# ---------------------------------------------------------------------------


class TestGroundTruth:
    def test_construction_and_fields(self) -> None:
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)
        gt = GroundTruth(
            image_id=1,
            image_path="/tmp/img.jpg",
            boxes=boxes,
            labels=labels,
            masks=None,
            depth=None,
            image_size=(640, 480),
        )
        assert gt.image_id == 1
        assert gt.image_path == "/tmp/img.jpg"
        assert gt.boxes.shape == (1, 4)
        assert gt.boxes.dtype == np.float32
        assert gt.labels.dtype == np.int32
        assert gt.masks is None
        assert gt.depth is None
        assert gt.image_size == (640, 480)

    def test_empty_boxes(self) -> None:
        gt = GroundTruth(
            image_id=0,
            image_path="",
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int32),
            masks=None,
            depth=None,
            image_size=(100, 100),
        )
        assert gt.boxes.shape == (0, 4)
        assert gt.labels.shape == (0,)


# ---------------------------------------------------------------------------
# DatasetLoader.from_coco_json — standalone mode
# ---------------------------------------------------------------------------


class TestFromCocoJson:
    def test_len(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        assert len(loader) == 2

    def test_iteration_yields_tuples(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = list(loader)
        assert len(items) == 2
        for img_path, gt in items:
            assert isinstance(img_path, str)
            assert isinstance(gt, GroundTruth)

    def test_ground_truth_bbox_xyxy(self, simple_coco) -> None:
        """Verify xywh → xyxy conversion: [10,20,50,60] → [10,20,60,80]."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        gt1 = items["000001.jpg"]
        assert gt1.boxes.shape == (2, 4)
        # First annotation: bbox [10, 20, 50, 60] → xyxy [10, 20, 60, 80]
        np.testing.assert_allclose(gt1.boxes[0], [10.0, 20.0, 60.0, 80.0])

    def test_category_0_indexing(self, simple_coco) -> None:
        """COCO category IDs (1-indexed) must be converted to 0-indexed labels."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        gt1 = items["000001.jpg"]
        # category_id=1 → label 0, category_id=2 → label 1
        assert 0 in gt1.labels
        assert 1 in gt1.labels

    def test_image_size_populated(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        gt1 = items["000001.jpg"]
        assert gt1.image_size == (640, 480)

    def test_masks_none_for_detection_only(self, simple_coco) -> None:
        """When no annotation has a segmentation polygon, masks should be None."""
        images_dir, json_path, _ = simple_coco
        # Overwrite with annotations that have empty segmentation lists
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for _, gt in loader:
            assert gt.masks is None

    def test_masks_populated_for_segmentation(self, seg_coco) -> None:
        """When annotations contain polygons, masks should not be None."""
        images_dir, json_path = seg_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        results = list(loader)
        assert len(results) == 1
        _, gt = results[0]
        assert gt.masks is not None
        assert len(gt.masks) == 1

    def test_names_from_coco_categories(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        names = loader.names
        assert names[0] == "person"
        assert names[1] == "bicycle"
        assert names[2] == "car"

    def test_image_without_annotations_still_yielded(self, tmp_path: Path) -> None:
        """Images without annotations should still be yielded with empty boxes."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _touch_image(images_dir, "img.jpg")
        coco = _make_coco_json(
            images=[{"id": 99, "file_name": "img.jpg", "width": 100, "height": 100}],
            annotations=[],  # no annotations for this image
        )
        json_path = tmp_path / "ann.json"
        _write_json(json_path, coco)
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = list(loader)
        assert len(items) == 1
        _, gt = items[0]
        assert gt.boxes.shape == (0, 4)
        assert gt.labels.shape == (0,)

    def test_missing_images_dir_raises(self, tmp_path: Path) -> None:
        json_path = tmp_path / "ann.json"
        _write_json(json_path, _make_coco_json([], []))
        missing = str(tmp_path / "nonexistent_images")
        with pytest.raises(FileNotFoundError) as exc_info:
            DatasetLoader.from_coco_json(missing, str(json_path))
        assert "nonexistent_images" in str(exc_info.value)

    def test_missing_json_raises(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        missing_json = str(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            DatasetLoader.from_coco_json(str(images_dir), missing_json)
        assert "missing.json" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DatasetLoader.from_yaml — YAML mode
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_from_yaml_len(self, yaml_dataset) -> None:
        yaml_path, _, _ = yaml_dataset
        loader = DatasetLoader.from_yaml(str(yaml_path))
        assert len(loader) == 2

    def test_from_yaml_iterates(self, yaml_dataset) -> None:
        yaml_path, _, _ = yaml_dataset
        loader = DatasetLoader.from_yaml(str(yaml_path))
        items = list(loader)
        assert len(items) == 2

    def test_names_from_yaml(self, yaml_dataset) -> None:
        yaml_path, _, _ = yaml_dataset
        loader = DatasetLoader.from_yaml(str(yaml_path))
        names = loader.names
        assert names[0] == "person"
        assert names[1] == "bicycle"
        assert names[2] == "car"

    def test_from_yaml_alias_constructor(self, yaml_dataset) -> None:
        """DatasetLoader(data=...) is equivalent to from_yaml(...)."""
        yaml_path, _, _ = yaml_dataset
        loader = DatasetLoader(data=str(yaml_path))
        assert len(loader) == 2

    def test_split_parameter(self, tmp_path: Path) -> None:
        """The split parameter selects the correct images subdirectory."""
        images_dir = tmp_path / "val2017"
        images_dir.mkdir()
        _touch_image(images_dir, "img1.jpg")
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
            annotations=[],
        )
        _write_json(ann_dir / "instances.json", coco)
        yaml_data = {
            "path": str(tmp_path),
            "val": "val2017",
            "annotations": "annotations/instances.json",
        }
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, yaml_data)
        loader = DatasetLoader.from_yaml(str(yaml_path), split="val")
        assert len(loader) == 1

    def test_missing_yaml_raises(self) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            DatasetLoader.from_yaml("/nonexistent/path/dataset.yaml")
        assert "dataset.yaml" in str(exc_info.value)

    def test_missing_yaml_annotations_raises(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        yaml_data = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/missing.json",
        }
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, yaml_data)
        with pytest.raises(FileNotFoundError) as exc_info:
            DatasetLoader.from_yaml(str(yaml_path))
        assert "missing.json" in str(exc_info.value)

    def test_missing_yaml_images_dir_raises(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _write_json(ann_dir / "inst.json", _make_coco_json([], []))
        yaml_data = {
            "path": str(tmp_path),
            "val": "nonexistent_images",
            "annotations": "annotations/inst.json",
        }
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, yaml_data)
        with pytest.raises(FileNotFoundError) as exc_info:
            DatasetLoader.from_yaml(str(yaml_path))
        assert "nonexistent_images" in str(exc_info.value)

    def test_dict_config(self, tmp_path: Path) -> None:
        """DatasetLoader(data=<dict>) accepts a pre-parsed dict."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _touch_image(images_dir, "a.jpg")
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "a.jpg", "width": 50, "height": 50}],
            annotations=[],
        )
        _write_json(ann_dir / "inst.json", coco)
        cfg = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/inst.json",
            "names": {0: "person"},
        }
        loader = DatasetLoader(data=cfg)
        assert len(loader) == 1

    def test_names_list_format_in_yaml(self, tmp_path: Path) -> None:
        """YAML 'names' can be a list instead of a dict."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
            annotations=[],
            categories=[{"id": 1, "name": "cat", "supercategory": "animal"}],
        )
        _write_json(ann_dir / "inst.json", coco)
        _touch_image(images_dir, "x.jpg")
        yaml_data = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/inst.json",
            "names": ["cat"],  # list format
        }
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, yaml_data)
        loader = DatasetLoader.from_yaml(str(yaml_path))
        assert loader.names[0] == "cat"


# ---------------------------------------------------------------------------
# DatasetLoader general
# ---------------------------------------------------------------------------


class TestDatasetLoaderGeneral:
    def test_mutual_exclusion_raises(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        with pytest.raises(ValueError, match="not both"):
            DatasetLoader(
                data="some.yaml",
                images=str(images_dir),
                annotations=str(json_path),
            )

    def test_class_names_alias(self, simple_coco) -> None:
        """class_names is a backward-compat alias for names."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        assert loader.class_names == loader.names

    def test_image_id_in_ground_truth(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        assert items["000001.jpg"].image_id == 1
        assert items["000002.jpg"].image_id == 2

    def test_boxes_dtype_float32(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for _, gt in loader:
            assert gt.boxes.dtype == np.float32

    def test_labels_dtype_int32(self, simple_coco) -> None:
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for _, gt in loader:
            assert gt.labels.dtype == np.int32

    def test_image_list_standalone_mode(self, simple_coco) -> None:
        """DatasetLoader(images=[...], annotations=...) with a list of paths."""
        images_dir, json_path, _ = simple_coco
        image_paths = [
            str(images_dir / "000001.jpg"),
            str(images_dir / "000002.jpg"),
        ]
        loader = DatasetLoader(images=image_paths, annotations=str(json_path))
        assert len(loader) == 2
        results = list(loader)
        assert len(results) == 2

    def test_empty_coco_json(self, tmp_path: Path) -> None:
        """An empty COCO JSON (no images, no annotations) yields nothing."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        json_path = tmp_path / "empty.json"
        _write_json(json_path, _make_coco_json([], []))
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        assert len(loader) == 0
        assert list(loader) == []

    def test_image_without_coco_entry_skipped(self, tmp_path: Path) -> None:
        """Image files with no matching COCO entry are skipped silently."""
        images_dir = tmp_path / "img"
        images_dir.mkdir()
        _touch_image(images_dir, "known.jpg")
        _touch_image(images_dir, "unknown.jpg")  # not in COCO JSON
        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "known.jpg", "width": 100, "height": 100}],
            annotations=[],
        )
        json_path = tmp_path / "ann.json"
        _write_json(json_path, coco)
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        results = list(loader)
        assert len(results) == 1
        assert Path(results[0][0]).name == "known.jpg"


# ---------------------------------------------------------------------------
# DatasetLoader — depth GT loading
# ---------------------------------------------------------------------------


class TestDepthGt:
    def test_depth_loaded_when_task_is_depth(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        depth_dir = tmp_path / "depth_maps"
        depth_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()

        _touch_image(images_dir, "frame01.jpg")
        np.save(str(depth_dir / "frame01.npy"), np.ones((10, 10), dtype=np.float32))

        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "frame01.jpg", "width": 10, "height": 10}],
            annotations=[],
        )
        _write_json(ann_dir / "inst.json", coco)

        yaml_data = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/inst.json",
            "depth_gt": "depth_maps",
        }
        yaml_path = tmp_path / "depth.yaml"
        _write_yaml(yaml_path, yaml_data)

        loader = DatasetLoader.from_yaml(str(yaml_path), task="depth")
        results = list(loader)
        assert len(results) == 1
        _, gt = results[0]
        assert gt.depth is not None
        assert gt.depth.shape == (10, 10)
        assert gt.depth.dtype == np.float32

    def test_depth_none_for_detect_task(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        depth_dir = tmp_path / "depth_maps"
        depth_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()

        _touch_image(images_dir, "frame01.jpg")
        np.save(str(depth_dir / "frame01.npy"), np.ones((10, 10), dtype=np.float32))

        coco = _make_coco_json(
            images=[{"id": 1, "file_name": "frame01.jpg", "width": 10, "height": 10}],
            annotations=[],
        )
        _write_json(ann_dir / "inst.json", coco)

        yaml_data = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/inst.json",
            "depth_gt": "depth_maps",
        }
        yaml_path = tmp_path / "depth.yaml"
        _write_yaml(yaml_path, yaml_data)

        # task="detect" → depth should not be loaded
        loader = DatasetLoader.from_yaml(str(yaml_path), task="detect")
        _, gt = next(iter(loader))
        assert gt.depth is None


# ---------------------------------------------------------------------------
# DatasetLoader — import-level smoke test
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_eval(self) -> None:
        from mata.eval import DatasetLoader as DL  # noqa: N817
        from mata.eval import GroundTruth as GT  # noqa: N817

        assert DL is DatasetLoader
        assert GT is GroundTruth


# ---------------------------------------------------------------------------
# Task G5 — spec-named integration tests
# (names match those listed in VALIDATION_GUIDE.md exactly)
# ---------------------------------------------------------------------------


class TestSpecNamedDataset:
    """Named tests matching the Task G5 spec for traceability."""

    def test_from_yaml_resolves_paths(self, yaml_dataset) -> None:
        """from_yaml() resolves relative images/annotations paths correctly."""
        yaml_path, images_dir, json_path = yaml_dataset
        loader = DatasetLoader.from_yaml(str(yaml_path))
        # Internal _images_dir must resolve to the actual images directory
        assert loader._images_dir is not None
        assert loader._images_dir.exists()
        assert loader._annotations_path is not None
        assert loader._annotations_path.exists()

    def test_from_coco_json_loads_annotations(self, simple_coco) -> None:
        """from_coco_json() loads all annotations into GroundTruth objects."""
        images_dir, json_path, coco = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = list(loader)
        # 2 images, 3 annotations across them
        total_boxes = sum(len(gt.boxes) for _, gt in items)
        assert total_boxes == 3

    def test_bbox_xywh_to_xyxy_conversion(self, simple_coco) -> None:
        """COCO xywh bounding boxes are converted to xyxy on load."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        gt1 = items["000001.jpg"]
        # bbox [10, 20, 50, 60] → xyxy [10, 20, 60, 80]
        np.testing.assert_allclose(gt1.boxes[0], [10.0, 20.0, 60.0, 80.0])
        # bbox [100, 150, 80, 40] → xyxy [100, 150, 180, 190]
        np.testing.assert_allclose(gt1.boxes[1], [100.0, 150.0, 180.0, 190.0])

    def test_category_id_0_indexed(self, simple_coco) -> None:
        """COCO 1-indexed category IDs are mapped to 0-indexed class labels."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        items = {Path(p).name: gt for p, gt in loader}
        # category_id 1 → label 0, category_id 3 → label 2
        assert set(items["000001.jpg"].labels.tolist()).issubset({0, 1, 2})
        assert 2 in items["000002.jpg"].labels.tolist()

    def test_len_matches_image_count(self, simple_coco) -> None:
        """__len__() returns the number of image files in the images directory."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        assert len(loader) == 2

    def test_iteration_yields_tuples(self, simple_coco) -> None:
        """__iter__() yields (str, GroundTruth) tuples."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for item in loader:
            assert isinstance(item, tuple) and len(item) == 2
            path, gt = item
            assert isinstance(path, str)
            assert isinstance(gt, GroundTruth)

    def test_missing_images_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        """Passing a non-existent images dir raises FileNotFoundError."""
        json_path = tmp_path / "ann.json"
        _write_json(json_path, _make_coco_json([], []))
        with pytest.raises(FileNotFoundError):
            DatasetLoader.from_coco_json(str(tmp_path / "no_such_dir"), str(json_path))

    def test_names_from_yaml_block(self, yaml_dataset) -> None:
        """Class names defined in the YAML 'names' block are returned by .names."""
        yaml_path, _, _ = yaml_dataset
        loader = DatasetLoader.from_yaml(str(yaml_path))
        assert loader.names[0] == "person"
        assert loader.names[1] == "bicycle"
        assert loader.names[2] == "car"

    def test_names_from_coco_categories(self, simple_coco) -> None:
        """When no YAML names block is present, names come from COCO categories."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        names = loader.names
        assert names == {0: "person", 1: "bicycle", 2: "car"}


# ---------------------------------------------------------------------------
# Task B2 — OCR Annotation Loading in DatasetLoader
# ---------------------------------------------------------------------------

_OCR_CATEGORIES = [{"id": 1, "name": "text", "supercategory": "text"}]


def _make_ocr_coco_json(images: list[dict], annotations: list[dict]) -> dict:
    """Return a minimal COCO-Text-style JSON dict with OCR annotations."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": _OCR_CATEGORIES,
    }


@pytest.fixture()
def ocr_coco(tmp_path: Path):
    """
    COCO-Text-style dataset on disk:
        tmp_path/
            images/        ← two placeholder images
            annotations/cocotext.json
    Image 1 has two text annotations; image 2 has one.
    Returns (images_dir, json_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    _touch_image(images_dir, "img_ocr_1.jpg")
    _touch_image(images_dir, "img_ocr_2.jpg")

    coco = _make_ocr_coco_json(
        images=[
            {"id": 1, "file_name": "img_ocr_1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_ocr_2.jpg", "width": 320, "height": 240},
        ],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 200.0, 50.0, 20.0],
                "text": "STOP",
                "legibility": "legible",
                "language": "english",
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [200.0, 300.0, 60.0, 25.0],
                "text": "ONE WAY",
                "legibility": "legible",
                "language": "english",
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [10.0, 10.0, 80.0, 30.0],
                "text": "YIELD",
                "legibility": "legible",
                "language": "english",
            },
        ],
    )
    json_path = ann_dir / "cocotext.json"
    _write_json(json_path, coco)
    return images_dir, json_path


class TestOCRAnnotationLoading:
    """Task B2 — end-to-end COCO-Text annotation loading into GroundTruth.text."""

    def test_from_coco_json_ocr_text_is_list(self, ocr_coco) -> None:
        """from_coco_json() with OCR JSON → gt.text is a list[str]."""
        images_dir, json_path = ocr_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        results = {Path(p).name: gt for p, gt in loader}
        gt1 = results["img_ocr_1.jpg"]
        gt2 = results["img_ocr_2.jpg"]
        assert isinstance(gt1.text, list)
        assert isinstance(gt2.text, list)
        assert all(isinstance(t, str) for t in gt1.text)
        assert all(isinstance(t, str) for t in gt2.text)

    def test_from_coco_json_ocr_correct_text_values(self, ocr_coco) -> None:
        """Parsed gt.text contains the correct transcription strings."""
        images_dir, json_path = ocr_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        results = {Path(p).name: gt for p, gt in loader}
        assert sorted(results["img_ocr_1.jpg"].text) == ["ONE WAY", "STOP"]
        assert results["img_ocr_2.jpg"].text == ["YIELD"]

    def test_non_ocr_json_text_is_none(self, simple_coco) -> None:
        """Regular COCO detection JSON (no 'text' key) → gt.text is None."""
        images_dir, json_path, _ = simple_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for _, gt in loader:
            assert gt.text is None

    def test_text_parallel_to_boxes_and_labels(self, ocr_coco) -> None:
        """len(gt.text) == len(gt.boxes) == len(gt.labels) for all images."""
        images_dir, json_path = ocr_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        for _, gt in loader:
            assert gt.text is not None
            assert len(gt.text) == len(gt.boxes) == len(gt.labels)

    def test_empty_text_string_included(self, tmp_path: Path) -> None:
        """OCR annotation with 'text': '' → empty string included in gt.text."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _touch_image(images_dir, "blank.jpg")

        coco = _make_ocr_coco_json(
            images=[{"id": 1, "file_name": "blank.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 50.0, 20.0],
                    "text": "",
                    "legibility": "illegible",
                },
            ],
        )
        json_path = ann_dir / "empty_text.json"
        _write_json(json_path, coco)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        _, gt = next(iter(loader))
        assert gt.text is not None
        assert "" in gt.text
        assert len(gt.text) == 1

    def test_illegible_annotation_text_included(self, tmp_path: Path) -> None:
        """Annotations with legibility='illegible' are included as-is in gt.text."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _touch_image(images_dir, "sign.jpg")

        coco = _make_ocr_coco_json(
            images=[{"id": 1, "file_name": "sign.jpg", "width": 200, "height": 150}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [5.0, 5.0, 40.0, 15.0],
                    "text": "blurry",
                    "legibility": "illegible",
                },
            ],
        )
        json_path = ann_dir / "illegible.json"
        _write_json(json_path, coco)

        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        _, gt = next(iter(loader))
        assert gt.text == ["blurry"]

    def test_from_yaml_ocr_populates_text(self, tmp_path: Path) -> None:
        """DatasetLoader.from_yaml() with OCR YAML config populates gt.text."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        _touch_image(images_dir, "street.jpg")

        coco = _make_ocr_coco_json(
            images=[{"id": 1, "file_name": "street.jpg", "width": 400, "height": 300}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50.0, 100.0, 100.0, 30.0],
                    "text": "EXIT",
                    "legibility": "legible",
                    "language": "english",
                },
            ],
        )
        _write_json(ann_dir / "ocr_ann.json", coco)

        yaml_data = {
            "path": str(tmp_path),
            "val": "images",
            "annotations": "annotations/ocr_ann.json",
            "names": {0: "text"},
        }
        yaml_path = tmp_path / "coco_text.yaml"
        _write_yaml(yaml_path, yaml_data)

        loader = DatasetLoader.from_yaml(str(yaml_path))
        _, gt = next(iter(loader))
        assert gt.text == ["EXIT"]
        assert len(gt.text) == len(gt.boxes) == len(gt.labels) == 1

    def test_bbox_xywh_converted_for_ocr_annotations(self, ocr_coco) -> None:
        """COCO xywh bbox in OCR annotations is correctly converted to xyxy."""
        images_dir, json_path = ocr_coco
        loader = DatasetLoader.from_coco_json(str(images_dir), str(json_path))
        results = {Path(p).name: gt for p, gt in loader}
        gt2 = results["img_ocr_2.jpg"]
        # bbox [10, 10, 80, 30] → xyxy [10, 10, 90, 40]
        np.testing.assert_allclose(gt2.boxes[0], [10.0, 10.0, 90.0, 40.0])
