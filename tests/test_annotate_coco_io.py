from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mata.annotate import coco_io
from mata.eval.dataset import DatasetLoader
from mata.training.datasets.coco_dataset import COCODetectionDataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def _build_coco_dataset(root: Path, image_count: int = 20) -> dict:
    dataset_images = root / "images"
    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")
    car_id = coco_io.add_category(coco, "car")

    for index in range(image_count):
        file_name = f"img_{index:03d}.jpg"
        _write_image(dataset_images / file_name, (index % 255, 20, 40))
        image_id = coco_io.add_image(coco, file_name, width=32, height=24)
        category_id = person_id if index % 2 == 0 else car_id
        coco_io.add_annotation(coco, image_id, [1, 2, 10, 12], category_id)

    return coco


def test_generate_yaml_config_matches_split_aware_format(tmp_path: Path) -> None:
    yaml_path = coco_io.generate_yaml_config(tmp_path, ["person", "car"])

    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert parsed == {
        "path": str(tmp_path.resolve()),
        "train": "train",
        "val": "val",
        "train_annotations": "annotations/instances_train.json",
        "val_annotations": "annotations/instances_val.json",
        "names": {0: "person", 1: "car"},
    }


def test_split_dataset_is_deterministic_and_annotations_follow_images(tmp_path: Path) -> None:
    coco = _build_coco_dataset(tmp_path, image_count=20)

    train_coco, val_coco = coco_io.split_dataset(coco, ratio=0.8, seed=42)
    train_coco_again, val_coco_again = coco_io.split_dataset(coco, ratio=0.8, seed=42)

    assert len(train_coco["images"]) == 16
    assert len(val_coco["images"]) == 4
    assert {image["id"] for image in train_coco["images"]} == {
        image["id"] for image in train_coco_again["images"]
    }
    assert {image["id"] for image in val_coco["images"]} == {
        image["id"] for image in val_coco_again["images"]
    }

    train_image_ids = {image["id"] for image in train_coco["images"]}
    val_image_ids = {image["id"] for image in val_coco["images"]}

    assert all(annotation["image_id"] in train_image_ids for annotation in train_coco["annotations"])
    assert all(annotation["image_id"] in val_image_ids for annotation in val_coco["annotations"])
    assert train_image_ids.isdisjoint(val_image_ids)


def test_export_dataset_writes_yaml_and_annotation_jsons(tmp_path: Path) -> None:
    """Export writes dataset.yaml + instances_train/val.json from pre-split images."""
    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")
    car_id = coco_io.add_category(coco, "car")

    # 8 train images, 2 val images — already in split dirs
    for i in range(8):
        fname = f"train/img_{i:03d}.jpg"
        _write_image(tmp_path / fname, (i * 20, 30, 40))
        img_id = coco_io.add_image(coco, fname, width=32, height=24)
        coco_io.add_annotation(coco, img_id, [1, 2, 10, 12], person_id if i % 2 == 0 else car_id)
    for i in range(2):
        fname = f"val/img_val_{i:03d}.jpg"
        _write_image(tmp_path / fname, (i * 50, 80, 40))
        img_id = coco_io.add_image(coco, fname, width=32, height=24)
        coco_io.add_annotation(coco, img_id, [3, 3, 12, 12], person_id)

    yaml_path, unassigned = coco_io.export_dataset(tmp_path, coco, ["person", "car"])

    assert unassigned == []
    assert yaml_path == tmp_path / "dataset.yaml"
    assert (tmp_path / "annotations" / "instances_train.json").is_file()
    assert (tmp_path / "annotations" / "instances_val.json").is_file()

    train_json = json.loads((tmp_path / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    val_json = json.loads((tmp_path / "annotations" / "instances_val.json").read_text(encoding="utf-8"))

    assert len(train_json["images"]) == 8
    assert len(val_json["images"]) == 2

    # Images must NOT have been moved — still in original locations
    assert (tmp_path / "train" / "img_000.jpg").is_file()
    assert (tmp_path / "val" / "img_val_000.jpg").is_file()

    loader = DatasetLoader.from_yaml(str(yaml_path), split="val")
    assert len(loader) == 2
    assert loader.names == {0: "person", 1: "car"}

    train_dataset = COCODetectionDataset(str(yaml_path), split="train")
    assert len(train_dataset) == 8
    _, target = train_dataset[0]
    assert "boxes" in target


def test_export_dataset_raises_when_no_split_dirs(tmp_path: Path) -> None:
    """export_dataset raises ValueError when no split dirs exist (use Redistribute first)."""
    coco = coco_io.create_empty_coco()
    coco_io.add_category(coco, "person")
    coco_io.add_image(coco, "flat.jpg", width=32, height=24)

    with pytest.raises(ValueError, match="No split structure found"):
        coco_io.export_dataset(tmp_path, coco, ["person"])


def test_export_dataset_preserves_existing_split(tmp_path: Path) -> None:
    """Export writes annotation JSONs from existing split dirs — images not copied."""
    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")

    # Pre-split: 3 images in train/, 1 image in val/
    for i in range(3):
        fname = f"train/img_train_{i:02d}.jpg"
        _write_image(tmp_path / fname, (i * 50, 20, 40))
        img_id = coco_io.add_image(coco, fname, width=32, height=24)
        coco_io.add_annotation(coco, img_id, [1, 2, 10, 12], person_id)

    val_fname = "val/img_val_00.jpg"
    _write_image(tmp_path / val_fname, (100, 200, 40))
    img_id_val = coco_io.add_image(coco, val_fname, width=32, height=24)
    coco_io.add_annotation(coco, img_id_val, [5, 5, 15, 15], person_id)

    yaml_path, unassigned = coco_io.export_dataset(tmp_path, coco, ["person"])

    assert unassigned == []
    assert yaml_path == tmp_path / "dataset.yaml"
    assert (tmp_path / "annotations" / "instances_train.json").is_file()
    assert (tmp_path / "annotations" / "instances_val.json").is_file()

    # Images must NOT be moved — still in their original split directories
    assert (tmp_path / "train" / "img_train_00.jpg").is_file()
    assert (tmp_path / "val" / "img_val_00.jpg").is_file()

    train_json = json.loads((tmp_path / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    val_json = json.loads((tmp_path / "annotations" / "instances_val.json").read_text(encoding="utf-8"))

    assert len(train_json["images"]) == 3
    assert len(val_json["images"]) == 1


def test_export_dataset_detects_split_from_filesystem(tmp_path: Path) -> None:
    """Images with flat file_names that live in split dirs are detected from FS."""
    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")

    # Flat file_names (no path prefix) but physically in train/ and val/ dirs
    for i in range(4):
        base = f"flat_{i:02d}.jpg"
        split = "train" if i < 3 else "val"
        _write_image(tmp_path / split / base, (i * 30, 30, 30))
        img_id = coco_io.add_image(coco, base, width=32, height=24)
        coco_io.add_annotation(coco, img_id, [1, 1, 8, 8], person_id)

    yaml_path, unassigned = coco_io.export_dataset(tmp_path, coco, ["person"])

    assert unassigned == []
    # Images must still be in their original locations
    assert (tmp_path / "train" / "flat_00.jpg").is_file()
    assert (tmp_path / "val" / "flat_03.jpg").is_file()

    train_json = json.loads((tmp_path / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    val_json = json.loads((tmp_path / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
    assert len(train_json["images"]) == 3
    assert len(val_json["images"]) == 1


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


def test_create_empty_coco_has_required_keys() -> None:
    """create_empty_coco() returns a dict with all required top-level keys."""
    coco = coco_io.create_empty_coco()
    assert set(coco.keys()) >= {"images", "annotations", "categories"}
    assert coco["images"] == []
    assert coco["annotations"] == []
    assert coco["categories"] == []


def test_create_empty_coco_with_initial_images() -> None:
    """create_empty_coco() accepts pre-built image list."""
    imgs = [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}]
    coco = coco_io.create_empty_coco(images=imgs)
    assert coco["images"] == imgs


# ---------------------------------------------------------------------------
# Annotation CRUD
# ---------------------------------------------------------------------------


def test_add_annotation_auto_id_starts_at_one() -> None:
    """add_annotation() returns id=1 for the first annotation."""
    coco = coco_io.create_empty_coco()
    coco_io.add_category(coco, "dog")
    img_id = coco_io.add_image(coco, "img.jpg", 100, 100)

    ann_id = coco_io.add_annotation(coco, img_id, [10, 20, 30, 40], 1)

    assert ann_id == 1
    assert coco["annotations"][0]["id"] == 1


def test_add_annotation_auto_id_increments() -> None:
    """Subsequent add_annotation() calls produce ascending IDs."""
    coco = coco_io.create_empty_coco()
    img_id = coco_io.add_image(coco, "img.jpg", 100, 100)
    coco_io.add_category(coco, "cat")

    id1 = coco_io.add_annotation(coco, img_id, [0, 0, 10, 10], 1)
    id2 = coco_io.add_annotation(coco, img_id, [5, 5, 15, 15], 1)

    assert id2 == id1 + 1


def test_add_remove_annotation_roundtrip() -> None:
    """add_annotation() then remove_annotation() leaves an empty list."""
    coco = coco_io.create_empty_coco()
    img_id = coco_io.add_image(coco, "img.jpg", 100, 100)
    coco_io.add_category(coco, "car")

    ann_id = coco_io.add_annotation(coco, img_id, [0, 0, 50, 50], 1)
    assert len(coco["annotations"]) == 1

    coco_io.remove_annotation(coco, ann_id)
    assert len(coco["annotations"]) == 0


def test_remove_annotation_raises_key_error_for_missing_id() -> None:
    """remove_annotation() raises KeyError when the ID does not exist."""
    coco = coco_io.create_empty_coco()

    with pytest.raises(KeyError):
        coco_io.remove_annotation(coco, 999)


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """save_annotations() then load_annotations() reproduces the original dict."""
    coco = coco_io.create_empty_coco()
    cat_id = coco_io.add_category(coco, "person")
    img_id = coco_io.add_image(coco, "photo.jpg", 640, 480)
    coco_io.add_annotation(coco, img_id, [10, 20, 100, 80], cat_id)

    json_path = tmp_path / "out.json"
    coco_io.save_annotations(coco, json_path)
    loaded = coco_io.load_annotations(json_path)

    assert loaded["images"] == coco["images"]
    assert loaded["annotations"] == coco["annotations"]
    assert loaded["categories"] == coco["categories"]


def test_load_annotations_raises_on_missing_file(tmp_path: Path) -> None:
    """load_annotations() raises FileNotFoundError for absent files."""
    with pytest.raises(FileNotFoundError):
        coco_io.load_annotations(tmp_path / "nonexistent.json")


def test_load_annotations_raises_on_missing_keys(tmp_path: Path) -> None:
    """load_annotations() raises ValueError when required keys are absent."""
    import json

    bad = {"images": []}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        coco_io.load_annotations(path)


def test_atomic_save_creates_no_temp_on_success(tmp_path: Path) -> None:
    """save_annotations() leaves no .tmp files after a successful write."""
    coco = coco_io.create_empty_coco()
    json_path = tmp_path / "ann.json"

    coco_io.save_annotations(coco, json_path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"
    assert json_path.is_file()


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def test_xyxy_to_xywh_conversion() -> None:
    """xyxy_to_xywh converts [x1,y1,x2,y2] → [x,y,w,h]."""
    result = coco_io.xyxy_to_xywh([10.0, 20.0, 60.0, 80.0])
    assert result == [10.0, 20.0, 50.0, 60.0]


def test_xyxy_to_xywh_roundtrip() -> None:
    """xyxy → xywh → xyxy roundtrip produces the original values."""
    original = [5.0, 15.0, 55.0, 75.0]
    intermediate = coco_io.xyxy_to_xywh(original)
    recovered = coco_io.xywh_to_xyxy(intermediate)
    assert recovered == pytest.approx(original)


def test_xywh_to_xyxy_conversion() -> None:
    """xywh_to_xyxy converts [x,y,w,h] → [x1,y1,x2,y2]."""
    result = coco_io.xywh_to_xyxy([10.0, 20.0, 50.0, 60.0])
    assert result == [10.0, 20.0, 60.0, 80.0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_coco_clean_dict_has_no_warnings() -> None:
    """validate_coco() returns no warnings for a well-formed COCO dict."""
    coco = coco_io.create_empty_coco()
    cat_id = coco_io.add_category(coco, "dog")
    img_id = coco_io.add_image(coco, "a.jpg", 32, 24)
    coco_io.add_annotation(coco, img_id, [0, 0, 10, 10], cat_id)

    warnings = coco_io.validate_coco(coco)
    assert warnings == []


def test_validate_coco_orphan_annotation() -> None:
    """validate_coco() warns about annotations referencing unknown image_ids."""
    coco = coco_io.create_empty_coco()
    # Add annotation referencing an image that doesn't exist
    coco["annotations"].append({
        "id": 1, "image_id": 9999, "category_id": 1,
        "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0, "segmentation": [],
    })
    coco["categories"].append({"id": 1, "name": "obj", "supercategory": "obj"})

    warnings = coco_io.validate_coco(coco)
    assert any("9999" in w for w in warnings)


def test_validate_coco_orphan_category_reference() -> None:
    """validate_coco() warns about annotations with unknown category_ids."""
    coco = coco_io.create_empty_coco()
    img_id = coco_io.add_image(coco, "img.jpg", 10, 10)
    coco["annotations"].append({
        "id": 1, "image_id": img_id, "category_id": 999,
        "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0, "segmentation": [],
    })

    warnings = coco_io.validate_coco(coco)
    assert any("category_id=999" in w for w in warnings)


def test_validate_coco_nonpositive_bbox_dimensions() -> None:
    """validate_coco() warns about annotations with w<=0 or h<=0."""
    coco = coco_io.create_empty_coco()
    cat_id = coco_io.add_category(coco, "obj")
    img_id = coco_io.add_image(coco, "img.jpg", 100, 100)
    coco["annotations"].append({
        "id": 1, "image_id": img_id, "category_id": cat_id,
        "bbox": [10, 10, 0, 20], "area": 0, "iscrowd": 0, "segmentation": [],
    })

    warnings = coco_io.validate_coco(coco)
    assert any("non-positive" in w for w in warnings)