from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mata.annotate.dataset_manager import DatasetManager


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(255, 0, 0)).save(path)


def _make_classification_dataset(root: Path) -> Path:
    dataset = root / "classify_mini"

    _write_image(dataset / "train" / "circle" / "001.jpg")
    _write_image(dataset / "train" / "circle" / "002.jpg")
    _write_image(dataset / "train" / "square" / "101.jpg")
    _write_image(dataset / "val" / "circle" / "201.jpg")
    _write_image(dataset / "val" / "square" / "202.jpg")

    (dataset / "train" / "triangle").mkdir(parents=True, exist_ok=True)
    (dataset / "val" / "triangle").mkdir(parents=True, exist_ok=True)

    return dataset


def test_list_classes_uses_train_root_counts(tmp_path: Path) -> None:
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    classes = manager.list_classes("classify_mini")

    assert classes == [
        {"name": "circle", "count": 2},
        {"name": "square", "count": 1},
        {"name": "triangle", "count": 0},
    ]


def test_reclassify_image_moves_within_existing_split(tmp_path: Path) -> None:
    dataset = _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    manager.reclassify_image("classify_mini", "001.jpg", "circle", "square")

    assert not (dataset / "train" / "circle" / "001.jpg").exists()
    assert (dataset / "train" / "square" / "001.jpg").exists()
    assert (dataset / "val" / "circle" / "201.jpg").exists()


def test_reclassify_image_rejects_invalid_target_class(tmp_path: Path) -> None:
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Invalid class name"):
        manager.reclassify_image("classify_mini", "001.jpg", "circle", "../../etc")


def test_reclassify_image_requires_existing_destination_class(tmp_path: Path) -> None:
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Destination class 'hexagon' does not exist"):
        manager.reclassify_image("classify_mini", "001.jpg", "circle", "hexagon")


def test_create_class_creates_directories_for_all_splits(tmp_path: Path) -> None:
    dataset = _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    manager.create_class("classify_mini", "hexagon")

    assert (dataset / "train" / "hexagon").is_dir()
    assert (dataset / "val" / "hexagon").is_dir()


def test_delete_class_rejects_non_empty_directory(tmp_path: Path) -> None:
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Class 'circle' is not empty"):
        manager.delete_class("classify_mini", "circle")


def test_delete_class_removes_empty_directories_across_splits(tmp_path: Path) -> None:
    dataset = _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    manager.delete_class("classify_mini", "triangle")

    assert not (dataset / "train" / "triangle").exists()
    assert not (dataset / "val" / "triangle").exists()


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------


def _make_coco_dataset(root: Path, name: str = "coco_mini") -> Path:
    import json

    dataset = root / name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "000001.jpg")

    coco = {
        "images": [{"id": 1, "file_name": "000001.jpg", "width": 12, "height": 12}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                          "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0, "segmentation": []}],
        "categories": [{"id": 1, "name": "cat", "supercategory": "cat"}],
    }
    (dataset / "annotations" / "instances.json").write_text(
        json.dumps(coco, indent=2), encoding="utf-8"
    )
    return dataset


def test_list_datasets_returns_known_datasets(tmp_path: Path) -> None:
    """list_datasets() includes datasets present in the data root."""
    _make_coco_dataset(tmp_path, "coco_mini")
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    names = {d["name"] for d in manager.list_datasets()}

    assert "coco_mini" in names
    assert "classify_mini" in names


def test_list_datasets_returns_empty_for_empty_root(tmp_path: Path) -> None:
    """list_datasets() returns [] when no subdirectories exist."""
    manager = DatasetManager(tmp_path)
    assert manager.list_datasets() == []


def test_list_datasets_result_has_expected_keys(tmp_path: Path) -> None:
    """Each list_datasets() entry has name, image_count, has_annotations, type."""
    _make_coco_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    entries = manager.list_datasets()
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry.keys()) >= {"name", "image_count", "has_annotations", "type"}


# ---------------------------------------------------------------------------
# Image listing
# ---------------------------------------------------------------------------


def test_list_images_correct_count(tmp_path: Path) -> None:
    """list_images() returns one entry per image file."""
    dataset = tmp_path / "myds"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _write_image(dataset / "images" / f"{i:03d}.jpg")

    manager = DatasetManager(tmp_path)
    result = manager.list_images("myds")
    assert result["total"] == 5
    assert len(result["images"]) == 5


def test_list_images_excludes_hidden_directories(tmp_path: Path) -> None:
    """Images under .hidden directories are excluded."""
    dataset = tmp_path / "myds"
    _write_image(dataset / "images" / "visible.jpg")
    _write_image(dataset / ".cache" / "hidden.jpg")

    manager = DatasetManager(tmp_path)
    result = manager.list_images("myds")
    filenames = [img["filename"] for img in result["images"]]

    assert any("visible" in f for f in filenames)
    assert not any("hidden" in f for f in filenames)


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


def test_serve_image_returns_bytes(tmp_path: Path) -> None:
    """serve_image() returns (bytes, content_type) for a valid image."""
    dataset = tmp_path / "myds"
    _write_image(dataset / "images" / "test.jpg")

    manager = DatasetManager(tmp_path)
    data, ct = manager.serve_image("myds", "test.jpg")

    assert isinstance(data, bytes)
    assert len(data) > 0
    assert "image" in ct


def test_serve_image_path_traversal_blocked(tmp_path: Path) -> None:
    """serve_image() raises ValueError on Unix path traversal attempts."""
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Path traversal blocked"):
        manager.serve_image("../../etc", "passwd")


def test_serve_image_windows_path_traversal(tmp_path: Path) -> None:
    """serve_image() raises ValueError on Windows-style path traversal."""
    manager = DatasetManager(tmp_path)

    with pytest.raises((ValueError, FileNotFoundError)):
        # Windows backslash traversal — must not escape data root
        manager.serve_image("myds", "..\\..\\etc\\passwd")


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------


def test_create_dataset_creates_directories(tmp_path: Path) -> None:
    """create_dataset() creates images/ and annotations/ subdirectories."""
    manager = DatasetManager(tmp_path)
    result = manager.create_dataset("new_project")

    assert (tmp_path / "new_project" / "images").is_dir()
    assert (tmp_path / "new_project" / "annotations").is_dir()
    assert result["created"] is True


def test_create_dataset_invalid_name_traversal(tmp_path: Path) -> None:
    """create_dataset() refuses names containing path-traversal components."""
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Invalid dataset name"):
        manager.create_dataset("../evil")


def test_create_dataset_special_chars_rejected(tmp_path: Path) -> None:
    """create_dataset() refuses names with spaces or special characters."""
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Invalid dataset name"):
        manager.create_dataset("a b c")


def test_create_dataset_duplicate_raises(tmp_path: Path) -> None:
    """create_dataset() raises ValueError when dataset already exists."""
    manager = DatasetManager(tmp_path)
    manager.create_dataset("my_ds")

    with pytest.raises(ValueError, match="already exists"):
        manager.create_dataset("my_ds")


# ---------------------------------------------------------------------------
# Dataset type detection
# ---------------------------------------------------------------------------


def test_detect_dataset_type_coco(tmp_path: Path) -> None:
    """detect_dataset_type() returns 'coco' for COCO-layout datasets."""
    _make_coco_dataset(tmp_path, "coco_ds")
    manager = DatasetManager(tmp_path)

    assert manager.detect_dataset_type("coco_ds") == "coco"


def test_detect_dataset_type_imagefolder(tmp_path: Path) -> None:
    """detect_dataset_type() returns 'classification' for class-subdir split layout."""
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    assert manager.detect_dataset_type("classify_mini") == "classification"


def test_detect_dataset_type_empty(tmp_path: Path) -> None:
    """detect_dataset_type() returns 'empty' for non-existent dataset."""
    manager = DatasetManager(tmp_path)
    assert manager.detect_dataset_type("nonexistent_ds") == "empty"


# ---------------------------------------------------------------------------
# Classification path traversal in reclassify
# ---------------------------------------------------------------------------


def test_reclassify_path_traversal_from_class(tmp_path: Path) -> None:
    """reclassify_image() raises ValueError when from_class is a traversal."""
    _make_classification_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    with pytest.raises(ValueError, match="Invalid class name"):
        manager.reclassify_image("classify_mini", "001.jpg", "../etc", "circle")


# ---------------------------------------------------------------------------
# Image listing — search filter (Task A4)
# ---------------------------------------------------------------------------


def _make_searchable_dataset(root: Path) -> Path:
    """Create a dataset with distinctly named images for search testing."""
    dataset = root / "searchds"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for name in ("face_001.jpg", "face_002.jpg", "card_001.jpg", "FACE_003.jpg", "other.png"):
        _write_image(dataset / "images" / name)
    return dataset


def test_list_images_search_filters_by_filename(tmp_path: Path) -> None:
    """search param returns only images whose filename contains the substring."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("searchds", search="face")

    filenames = [img["filename"] for img in result["images"]]
    assert all("face" in f.lower() for f in filenames)
    assert result["total"] == 3  # face_001, face_002, FACE_003
    assert len(result["images"]) == 3


def test_list_images_search_is_case_insensitive(tmp_path: Path) -> None:
    """search param matches regardless of case (FACE matches face files)."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result_lower = manager.list_images("searchds", search="face")
    result_upper = manager.list_images("searchds", search="FACE")
    result_mixed = manager.list_images("searchds", search="Face")

    assert result_lower["total"] == result_upper["total"] == result_mixed["total"] == 3


def test_list_images_search_total_reflects_filtered_count(tmp_path: Path) -> None:
    """total in response reflects filtered count, not total dataset size."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    all_result = manager.list_images("searchds")
    filtered_result = manager.list_images("searchds", search="card")

    assert all_result["total"] == 5
    assert filtered_result["total"] == 1
    assert filtered_result["images"][0]["filename"] == "card_001.jpg"


def test_list_images_search_empty_string_returns_all(tmp_path: Path) -> None:
    """Empty search string applies no filter — all images returned."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("searchds", search="")

    assert result["total"] == 5


def test_list_images_search_none_returns_all(tmp_path: Path) -> None:
    """search=None applies no filter."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("searchds", search=None)

    assert result["total"] == 5


def test_list_images_search_with_pagination(tmp_path: Path) -> None:
    """Search combines with pagination: total reflects filtered count, page slices it."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("searchds", search="face", page=1, per_page=2)

    assert result["total"] == 3
    assert result["total_pages"] == 2
    assert len(result["images"]) == 2


def test_list_images_search_no_matches_returns_empty(tmp_path: Path) -> None:
    """search with no matches returns empty images list with total=0."""
    _make_searchable_dataset(tmp_path)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("searchds", search="zzz_no_match")

    assert result["total"] == 0
    assert result["images"] == []


def test_list_images_search_applies_to_filename_not_path(tmp_path: Path) -> None:
    """search matches filename only, not intermediate path segments."""
    dataset = tmp_path / "pathds"
    # File under a 'face' subdirectory but filename itself does not contain 'face'
    _write_image(dataset / "face" / "dog.jpg")
    _write_image(dataset / "images" / "face_cat.jpg")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pathds", search="face")

    filenames = [img["filename"] for img in result["images"]]
    # face_cat.jpg matches; dog.jpg is under face/ dir but 'face' is not in its filename
    assert any("face_cat" in f for f in filenames)
    assert not any("dog" in f for f in filenames)


# ---------------------------------------------------------------------------
# Image listing — pagination (Task F1)
# ---------------------------------------------------------------------------


def _make_numbered_dataset(root: Path, count: int, ds_name: str) -> Path:
    """Create a flat image dataset with *count* sequentially named image files."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        _write_image(dataset / "images" / f"img_{i:03d}.jpg")
    return dataset


def test_list_images_paginated_first_page(tmp_path: Path) -> None:
    """list_images(page=1, per_page=3) returns first 3 of 9 images."""
    _make_numbered_dataset(tmp_path, 9, "pg_dm1")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm1", page=1, per_page=3)

    assert result["total"] == 9
    assert len(result["images"]) == 3
    assert result["page"] == 1
    assert result["per_page"] == 3
    assert result["total_pages"] == 3


def test_list_images_paginated_second_page(tmp_path: Path) -> None:
    """list_images(page=2, per_page=3) returns a different set from page 1."""
    _make_numbered_dataset(tmp_path, 9, "pg_dm2")
    manager = DatasetManager(tmp_path)

    p1 = manager.list_images("pg_dm2", page=1, per_page=3)
    p2 = manager.list_images("pg_dm2", page=2, per_page=3)

    p1_names = {img["filename"] for img in p1["images"]}
    p2_names = {img["filename"] for img in p2["images"]}
    assert p1_names.isdisjoint(p2_names)
    assert len(p2["images"]) == 3


def test_list_images_paginated_last_page_partial(tmp_path: Path) -> None:
    """Last page may have fewer items than per_page (7 images, per_page=3 → last has 1)."""
    _make_numbered_dataset(tmp_path, 7, "pg_dm3")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm3", page=3, per_page=3)

    assert len(result["images"]) == 1  # 7 - 6 = 1


def test_list_images_paginated_total_pages_correct(tmp_path: Path) -> None:
    """total_pages is ceil(total / per_page)."""
    _make_numbered_dataset(tmp_path, 10, "pg_dm4")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm4", page=1, per_page=3)

    assert result["total_pages"] == 4  # ceil(10/3)


def test_list_images_all_pages_cover_all_images(tmp_path: Path) -> None:
    """Retrieving all pages yields all images with no duplicates."""
    _make_numbered_dataset(tmp_path, 8, "pg_dm5")
    manager = DatasetManager(tmp_path)

    per_page = 3
    collected: list[str] = []
    for pg in range(1, 4):  # pages 1-3 cover 8 images at per_page=3
        result = manager.list_images("pg_dm5", page=pg, per_page=per_page)
        collected.extend(img["filename"] for img in result["images"])

    assert len(set(collected)) == 8


def test_list_images_default_no_page_returns_all_compat_mode(tmp_path: Path) -> None:
    """list_images() without page returns all images in a single page (backward compat)."""
    _make_numbered_dataset(tmp_path, 6, "pg_dm6")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm6")

    assert result["total"] == 6
    assert len(result["images"]) == 6
    assert result["total_pages"] == 1


def test_list_images_pagination_envelope_keys_present(tmp_path: Path) -> None:
    """Pagination envelope always includes images, total, page, per_page, total_pages."""
    _make_numbered_dataset(tmp_path, 3, "pg_dm7")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm7", page=1, per_page=2)

    for key in ("images", "total", "page", "per_page", "total_pages"):
        assert key in result, f"Missing pagination key: {key}"


def test_list_images_out_of_range_page_clamped(tmp_path: Path) -> None:
    """Page beyond total_pages is clamped to the last page (not empty)."""
    _make_numbered_dataset(tmp_path, 3, "pg_dm8")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm8", page=999, per_page=2)

    assert result["page"] == 2  # last page: ceil(3/2)=2
    assert len(result["images"]) >= 1


def test_list_images_page_zero_clamped_to_one(tmp_path: Path) -> None:
    """page=0 is normalized to page=1."""
    _make_numbered_dataset(tmp_path, 4, "pg_dm9")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm9", page=0, per_page=2)

    assert result["page"] == 1
    assert len(result["images"]) == 2


def test_list_images_per_page_zero_normalized_to_one(tmp_path: Path) -> None:
    """per_page=0 is normalized to 1 (floor of max(1, per_page))."""
    _make_numbered_dataset(tmp_path, 3, "pg_dm10")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("pg_dm10", page=1, per_page=0)

    assert result["per_page"] == 1
    assert len(result["images"]) == 1


# ---------------------------------------------------------------------------
# Image listing — split filter (Task F1)
# ---------------------------------------------------------------------------


def _make_split_dataset_dm(root: Path, ds_name: str = "dm_split") -> Path:
    """Create a dataset with train/val/test split directories."""
    dataset = root / ds_name
    _write_image(dataset / "train" / "t1.jpg")
    _write_image(dataset / "train" / "t2.jpg")
    _write_image(dataset / "val" / "v1.jpg")
    _write_image(dataset / "test" / "te1.jpg")
    return dataset


def test_list_images_split_train_filters(tmp_path: Path) -> None:
    """split='train' returns only the 2 train images."""
    _make_split_dataset_dm(tmp_path, "sp_dm1")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_dm1", split="train")

    assert result["total"] == 2
    for img in result["images"]:
        assert img["split"] == "train"


def test_list_images_split_val_filters(tmp_path: Path) -> None:
    """split='val' returns only the 1 val image."""
    _make_split_dataset_dm(tmp_path, "sp_dm2")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_dm2", split="val")

    assert result["total"] == 1
    assert result["images"][0]["split"] == "val"


def test_list_images_split_test_filters(tmp_path: Path) -> None:
    """split='test' returns only the 1 test image."""
    _make_split_dataset_dm(tmp_path, "sp_dm3")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_dm3", split="test")

    assert result["total"] == 1
    assert result["images"][0]["split"] == "test"


def test_list_images_split_field_correctly_detected(tmp_path: Path) -> None:
    """split field is correctly populated per image based on directory structure."""
    _make_split_dataset_dm(tmp_path, "sp_dm4")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_dm4")
    split_map = {img["filename"]: img["split"] for img in result["images"]}

    assert split_map["t1.jpg"] == "train"
    assert split_map["t2.jpg"] == "train"
    assert split_map["v1.jpg"] == "val"
    assert split_map["te1.jpg"] == "test"


def test_list_images_split_invalid_name_returns_empty(tmp_path: Path) -> None:
    """split='unknown' returns empty list because no images match."""
    _make_split_dataset_dm(tmp_path, "sp_dm5")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_dm5", split="unknown_split")

    assert result["total"] == 0
    assert result["images"] == []


def test_list_images_valid_directory_treated_as_val_split(tmp_path: Path) -> None:
    """Images in a 'valid/' directory are returned when split='val'."""
    dataset = tmp_path / "sp_valid"
    _write_image(dataset / "train" / "t1.jpg")
    _write_image(dataset / "valid" / "v1.jpg")
    _write_image(dataset / "valid" / "v2.jpg")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_valid", split="val")

    assert result["total"] == 2
    for img in result["images"]:
        assert img["split"] == "val"


def test_list_images_split_field_val_for_valid_directory(tmp_path: Path) -> None:
    """Images in 'valid/' have split='val' when listed without a split filter."""
    dataset = tmp_path / "sp_valid2"
    _write_image(dataset / "valid" / "v1.jpg")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sp_valid2")
    assert result["images"][0]["split"] == "val"


# ---------------------------------------------------------------------------
# Image listing — sort (Task F1)
# ---------------------------------------------------------------------------


def _make_sortable_dataset_dm(root: Path, ds_name: str = "sortdm") -> Path:
    """Create a dataset with three images named banana, apple, cherry."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "banana.jpg")
    _write_image(dataset / "images" / "apple.jpg")
    _write_image(dataset / "images" / "cherry.jpg")
    return dataset


def test_list_images_sort_name_asc(tmp_path: Path) -> None:
    """sort='name_asc' returns filenames in alphabetical ascending order."""
    _make_sortable_dataset_dm(tmp_path, "sr_dm1")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sr_dm1", sort="name_asc")

    filenames = [img["filename"] for img in result["images"]]
    assert filenames == sorted(filenames)


def test_list_images_sort_name_desc(tmp_path: Path) -> None:
    """sort='name_desc' returns filenames in alphabetical descending order."""
    _make_sortable_dataset_dm(tmp_path, "sr_dm2")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sr_dm2", sort="name_desc")

    filenames = [img["filename"] for img in result["images"]]
    assert filenames == sorted(filenames, reverse=True)


def test_list_images_sort_size(tmp_path: Path) -> None:
    """sort='size' returns images ordered by size_bytes descending (largest first)."""
    dataset = tmp_path / "sr_dm3"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    small = dataset / "images" / "small.jpg"
    big = dataset / "images" / "big.jpg"
    _write_image(small)
    _write_image(big)
    # Overwrite big.jpg with substantially more bytes so size differs reliably
    big.write_bytes(b"x" * 5000)
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sr_dm3", sort="size")

    sizes = [img["size_bytes"] for img in result["images"]]
    assert sizes == sorted(sizes, reverse=True)
    assert result["images"][0]["filename"] == "big.jpg"


def test_list_images_sort_newest_accepted(tmp_path: Path) -> None:
    """sort='newest' is a valid sort option and returns all images."""
    _make_sortable_dataset_dm(tmp_path, "sr_dm4")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sr_dm4", sort="newest")

    assert result["total"] == 3


def test_list_images_sort_oldest_accepted(tmp_path: Path) -> None:
    """sort='oldest' is a valid sort option and returns all images."""
    _make_sortable_dataset_dm(tmp_path, "sr_dm5")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("sr_dm5", sort="oldest")

    assert result["total"] == 3


def test_list_images_default_sort_is_name_asc(tmp_path: Path) -> None:
    """Default sort (no sort arg) is equivalent to name_asc."""
    _make_sortable_dataset_dm(tmp_path, "sr_dm6")
    manager = DatasetManager(tmp_path)

    result_default = manager.list_images("sr_dm6")
    result_asc = manager.list_images("sr_dm6", sort="name_asc")

    default_names = [img["filename"] for img in result_default["images"]]
    asc_names = [img["filename"] for img in result_asc["images"]]
    assert default_names == asc_names


# ---------------------------------------------------------------------------
# Image listing — annotation count (Task F1)
# ---------------------------------------------------------------------------


def _make_coco_for_ann_count(root: Path, ds_name: str) -> tuple[Path, dict]:
    """Create dataset with annotated.jpg (2 anns) and unannotated.jpg (0 anns)."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "annotated.jpg")
    _write_image(dataset / "images" / "unannotated.jpg")
    coco: dict = {
        "images": [{"id": 1, "file_name": "annotated.jpg", "width": 12, "height": 12}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
             "area": 25, "iscrowd": 0, "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3],
             "area": 9, "iscrowd": 0, "segmentation": []},
        ],
        "categories": [{"id": 1, "name": "cat", "supercategory": "cat"}],
    }
    ann_dir = dataset / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "instances.json").write_text(json.dumps(coco), encoding="utf-8")
    return dataset, coco


def test_list_images_annotation_count_zero_without_coco(tmp_path: Path) -> None:
    """annotation_count is 0 for all images when coco=None (no COCO provided)."""
    dataset = tmp_path / "an_dm1"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "img.jpg")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("an_dm1", coco=None)

    assert all(img["annotation_count"] == 0 for img in result["images"])


def test_list_images_annotation_count_from_coco(tmp_path: Path) -> None:
    """annotation_count is populated from the passed-in COCO dict."""
    dataset = tmp_path / "an_dm2"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "img.jpg")
    manager = DatasetManager(tmp_path)
    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 12, "height": 12}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
             "area": 25, "iscrowd": 0, "segmentation": []},
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }

    result = manager.list_images("an_dm2", coco=coco)

    img = next(i for i in result["images"] if i["filename"] == "img.jpg")
    assert img["annotation_count"] == 1


def test_list_images_annotation_count_multiple_anns_per_image(tmp_path: Path) -> None:
    """annotation_count correctly counts all annotations (2) for a single image."""
    _, coco = _make_coco_for_ann_count(tmp_path, "an_dm3")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("an_dm3", coco=coco)

    ann = next(i for i in result["images"] if i["filename"] == "annotated.jpg")
    assert ann["annotation_count"] == 2


def test_list_images_annotation_count_zero_for_unannotated(tmp_path: Path) -> None:
    """Images not referenced in the COCO file have annotation_count=0 (FS path)."""
    _, coco = _make_coco_for_ann_count(tmp_path, "an_dm4")
    manager = DatasetManager(tmp_path)

    # sort="newest" forces the filesystem scan path so images absent from the
    # COCO document are still returned (COCO fast path skips unknown files).
    result = manager.list_images("an_dm4", coco=coco, sort="newest")

    unann = next(i for i in result["images"] if i["filename"] == "unannotated.jpg")
    assert unann["annotation_count"] == 0


def test_list_images_annotated_filter_true_returns_only_annotated(tmp_path: Path) -> None:
    """annotated='true' returns only images with annotation_count > 0."""
    _, coco = _make_coco_for_ann_count(tmp_path, "an_dm5")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("an_dm5", annotated="true", coco=coco)

    assert result["total"] == 1
    assert result["images"][0]["filename"] == "annotated.jpg"


# ---------------------------------------------------------------------------
# _fast_image_count helpers
# ---------------------------------------------------------------------------

def _make_coco_annotations_dir(root: Path, name: str, image_count: int) -> Path:
    """Create root/name/annotations/instances.json with `image_count` COCO image entries."""
    dataset = root / name
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    coco_doc = {
        "images": [{"id": i, "file_name": f"img_{i}.jpg"} for i in range(image_count)],
        "annotations": [],
        "categories": [],
    }
    (dataset / "annotations" / "instances.json").write_text(
        json.dumps(coco_doc), encoding="utf-8"
    )
    return dataset


def test_fast_image_count_reads_coco_json(tmp_path: Path) -> None:
    """_fast_image_count returns len(coco['images']) from annotations/*.json."""
    from mata.annotate.dataset_manager import _fast_image_count

    dataset = _make_coco_annotations_dir(tmp_path, "fic_1", 7)
    assert _fast_image_count(dataset) == 7


def test_fast_image_count_does_not_require_physical_files(tmp_path: Path) -> None:
    """_fast_image_count returns count from JSON metadata without actual image files."""
    from mata.annotate.dataset_manager import _fast_image_count

    dataset = tmp_path / "fic_2"
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    coco_doc = {"images": [{"id": i} for i in range(5)], "annotations": [], "categories": []}
    (dataset / "annotations" / "instances.json").write_text(
        json.dumps(coco_doc), encoding="utf-8"
    )
    assert _fast_image_count(dataset) == 5


def test_fast_image_count_fallback_to_rglob(tmp_path: Path) -> None:
    """_fast_image_count falls back to rglob when no COCO JSON or YAML present."""
    from mata.annotate.dataset_manager import _fast_image_count

    dataset = tmp_path / "fic_3"
    dataset.mkdir()
    _write_image(dataset / "a.jpg")
    _write_image(dataset / "b.png")
    assert _fast_image_count(dataset) == 2


def test_fast_image_count_yaml_path(tmp_path: Path) -> None:
    """_fast_image_count reads count from annotation JSON referenced by dataset.yaml."""
    yaml = pytest.importorskip("yaml")
    from mata.annotate.dataset_manager import _fast_image_count

    dataset = tmp_path / "fic_4"
    # annotations/ dir is empty — strategy 1 won't fire
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    ann_doc = {"images": [{"id": i} for i in range(3)], "annotations": [], "categories": []}
    (dataset / "train_ann.json").write_text(json.dumps(ann_doc), encoding="utf-8")
    cfg = {"path": ".", "train": "images/train", "train_annotations": "train_ann.json"}
    (dataset / "dataset.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    assert _fast_image_count(dataset) == 3


def test_list_datasets_image_count_from_coco_json(tmp_path: Path) -> None:
    """list_datasets() returns image_count from COCO JSON metadata, not rglob."""
    _make_coco_annotations_dir(tmp_path, "lic_ds1", 4)
    manager = DatasetManager(tmp_path)

    datasets = manager.list_datasets()
    ds = next(d for d in datasets if d["name"] == "lic_ds1")

    # Count from JSON metadata (4), not from actual image files on disk (0)
    assert ds["image_count"] == 4


# ---------------------------------------------------------------------------
# list_images() COCO fast path (no filesystem scan)
# ---------------------------------------------------------------------------

def _make_coco_for_fast_path(root: Path, ds_name: str) -> tuple[Path, dict]:
    """Dataset with 3 physical images + a COCO doc referencing them."""
    dataset = root / ds_name
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "annotations").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "aaa.jpg")
    _write_image(dataset / "images" / "bbb.jpg")
    _write_image(dataset / "images" / "ccc.jpg")
    coco: dict = {
        "images": [
            {"id": 1, "file_name": "aaa.jpg", "width": 12, "height": 12},
            {"id": 2, "file_name": "bbb.jpg", "width": 24, "height": 24},
            # Subpath variant — base name "ccc.jpg" should still be returned
            {"id": 3, "file_name": "train/ccc.jpg", "width": 8, "height": 8},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0, "segmentation": [],
            },
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    (dataset / "annotations" / "instances.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )
    return dataset, coco


def test_list_images_coco_fast_path_returns_all_images(tmp_path: Path) -> None:
    """COCO fast path returns all images listed in the COCO document."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_1")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_1", coco=coco, sort="name_asc", page=1, per_page=50)

    filenames = [img["filename"] for img in result["images"]]
    assert "aaa.jpg" in filenames
    assert "bbb.jpg" in filenames
    assert "ccc.jpg" in filenames
    assert result["total"] == 3


def test_list_images_coco_fast_path_strips_subpath(tmp_path: Path) -> None:
    """COCO fast path uses only base filename (strips subpath prefix like 'train/')."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_2")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_2", coco=coco, sort="name_asc")
    filenames = [img["filename"] for img in result["images"]]

    assert "ccc.jpg" in filenames
    assert all("/" not in f for f in filenames)


def test_list_images_coco_fast_path_name_asc_sort(tmp_path: Path) -> None:
    """COCO fast path returns images in ascending alphabetical order."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_3")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_3", coco=coco, sort="name_asc")
    names = [img["filename"] for img in result["images"]]

    assert names == sorted(names)


def test_list_images_coco_fast_path_name_desc_sort(tmp_path: Path) -> None:
    """COCO fast path returns images in descending alphabetical order."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_4")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_4", coco=coco, sort="name_desc")
    names = [img["filename"] for img in result["images"]]

    assert names == sorted(names, reverse=True)


def test_list_images_coco_fast_path_search_filter(tmp_path: Path) -> None:
    """COCO fast path applies the search= substring filter correctly."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_5")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_5", coco=coco, search="aaa")

    assert result["total"] == 1
    assert result["images"][0]["filename"] == "aaa.jpg"


def test_list_images_coco_fast_path_annotated_filter(tmp_path: Path) -> None:
    """COCO fast path filters to annotated images only when annotated='true'."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_6")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_6", coco=coco, annotated="true")

    # Only "aaa.jpg" has 1 annotation in _make_coco_for_fast_path
    assert result["total"] == 1
    assert result["images"][0]["filename"] == "aaa.jpg"


def test_list_images_coco_fast_path_unannotated_filter(tmp_path: Path) -> None:
    """COCO fast path filters to unannotated images only when annotated='false'."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_7")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_7", coco=coco, annotated="false")

    filenames = [img["filename"] for img in result["images"]]
    assert result["total"] == 2
    assert "bbb.jpg" in filenames
    assert "ccc.jpg" in filenames
    assert "aaa.jpg" not in filenames


def test_list_images_sort_newest_bypasses_coco_fast_path(tmp_path: Path) -> None:
    """sort='newest' requires mtime — falls back to FS scan, not COCO fast path."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_8")
    manager = DatasetManager(tmp_path)

    # Physical files exist so fs scan returns them; should not raise
    result = manager.list_images("fp_8", coco=coco, sort="newest")

    assert result["total"] == 3


def test_list_images_sort_size_bypasses_coco_fast_path(tmp_path: Path) -> None:
    """sort='size' requires stat() — falls back to FS scan, not COCO fast path."""
    _, coco = _make_coco_for_fast_path(tmp_path, "fp_9")
    manager = DatasetManager(tmp_path)

    result = manager.list_images("fp_9", coco=coco, sort="size")

    assert result["total"] == 3


# ---------------------------------------------------------------------------
# serve_thumbnail() — corrupt / unidentifiable image fallback
# ---------------------------------------------------------------------------

def test_serve_thumbnail_corrupt_image_does_not_raise(tmp_path: Path) -> None:
    """serve_thumbnail returns raw bytes instead of raising PIL.UnidentifiedImageError."""
    dataset = tmp_path / "corrupt_ds"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    corrupt = dataset / "images" / "bad.jpg"
    corrupt.write_bytes(b"this is definitely not a valid image file")
    manager = DatasetManager(tmp_path)

    data, ct = manager.serve_thumbnail("corrupt_ds", "bad.jpg")

    assert isinstance(data, bytes)
    assert len(data) > 0


def test_serve_thumbnail_corrupt_image_returns_raw_bytes(tmp_path: Path) -> None:
    """serve_thumbnail fallback serves the original corrupt bytes unchanged."""
    raw_content = b"not-an-image-sentinel-value-12345"
    dataset = tmp_path / "corrupt_ds2"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "images" / "bad.jpg").write_bytes(raw_content)
    manager = DatasetManager(tmp_path)

    data, _ = manager.serve_thumbnail("corrupt_ds2", "bad.jpg")

    assert data == raw_content


def test_serve_thumbnail_valid_image_returns_thumbnail_bytes(tmp_path: Path) -> None:
    """serve_thumbnail returns JPEG thumbnail bytes for a valid image."""
    dataset = tmp_path / "thumb_ds"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "ok.jpg")
    manager = DatasetManager(tmp_path)

    data, ct = manager.serve_thumbnail("thumb_ds", "ok.jpg", max_size=32)

    assert isinstance(data, bytes)
    assert len(data) > 0
    assert ct in ("image/jpeg", "image/png")


# ---------------------------------------------------------------------------
# redistribute_splits — COCO file_name update
# ---------------------------------------------------------------------------


def test_redistribute_splits_updates_coco_file_names(tmp_path: Path) -> None:
    """redistribute_splits rewrites file_name in the COCO JSON after moving images."""
    from mata.annotate import coco_io

    ds = tmp_path / "myds"
    ann_dir = ds / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Two images at root (flat, no split yet)
    _write_image(ds / "a.jpg")
    _write_image(ds / "b.jpg")

    # COCO JSON with flat file_names
    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")
    img_a = coco_io.add_image(coco, "a.jpg", width=12, height=12)
    img_b = coco_io.add_image(coco, "b.jpg", width=12, height=12)
    coco_io.add_annotation(coco, img_a, [1, 1, 5, 5], person_id)
    coco_io.add_annotation(coco, img_b, [2, 2, 6, 6], person_id)
    coco_io.save_annotations(coco, ann_dir / "instances.json")

    manager = DatasetManager(tmp_path)
    result = manager.redistribute_splits("myds", train_pct=50, val_pct=50, test_pct=0, seed=0)

    assert result["total"] == 2
    assert result["moved"] == 2

    # COCO JSON must be updated — no flat file_names remain
    updated = json.loads((ann_dir / "instances.json").read_text(encoding="utf-8"))
    for img in updated["images"]:
        fname = img["file_name"]
        assert "/" in fname, f"Expected split-prefixed file_name, got: {fname!r}"
        split = fname.split("/")[0]
        assert split in ("train", "val"), f"Unexpected split: {split!r}"
        # Physical file must exist at the new location
        assert (ds / fname).is_file(), f"Image not found at {fname}"


def test_redistribute_splits_already_in_place_normalises_file_names(tmp_path: Path) -> None:
    """Images already in the right split dir are still normalised in COCO JSON."""
    from mata.annotate import coco_io

    ds = tmp_path / "ds2"
    ann_dir = ds / "annotations"

    _write_image(ds / "train" / "x.jpg")
    _write_image(ds / "val" / "y.jpg")

    coco = coco_io.create_empty_coco()
    person_id = coco_io.add_category(coco, "person")
    # Flat file_names even though images are physically in split dirs
    coco_io.add_image(coco, "x.jpg", width=12, height=12)
    coco_io.add_image(coco, "y.jpg", width=12, height=12)
    ann_dir.mkdir(parents=True, exist_ok=True)
    coco_io.save_annotations(coco, ann_dir / "instances.json")

    manager = DatasetManager(tmp_path)
    manager.redistribute_splits("ds2", train_pct=50, val_pct=50, test_pct=0, seed=0)

    updated = json.loads((ann_dir / "instances.json").read_text(encoding="utf-8"))
    file_names = {img["file_name"] for img in updated["images"]}
    assert all("/" in fn for fn in file_names), f"Flat file_names remain: {file_names}"