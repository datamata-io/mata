"""Tests for mata.training.datasets — all dataset classes, collators, and factory.

All tests use synthetic data generated in-memory (no external downloads).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from mata.core.exceptions import TrainingError
from mata.training.datasets import (
    COCODetectionDataset,
    COCOSegmentationDataset,
    DatasetFactory,
    ImageFolderDataset,
    VOCDetectionDataset,
    classification_collate_fn,
    detection_collate_fn,
    segmentation_collate_fn,
)


# ===========================================================================
# Helpers — synthetic data factories
# ===========================================================================


def _save_image(path: Path, width: int = 32, height: int = 32) -> None:
    """Save a small synthetic RGB image."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(str(path))


def _make_coco_json(
    num_images: int = 2,
    annotations_per_image: int = 2,
    include_crowd: bool = False,
    include_segmentation: bool = False,
    image_width: int = 32,
    image_height: int = 32,
) -> dict:
    """Build a minimal valid COCO JSON dict."""
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    images = [
        {"id": i + 1, "file_name": f"img{i + 1:04d}.jpg", "width": image_width, "height": image_height}
        for i in range(num_images)
    ]
    annotations = []
    ann_id = 1
    for img in images:
        for _ in range(annotations_per_image):
            ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": img["id"],
                "category_id": 1,
                # xywh — fits inside 32x32 image
                "bbox": [2.0, 4.0, 10.0, 8.0],
                "iscrowd": 0,
            }
            if include_segmentation:
                # Simple square polygon
                ann["segmentation"] = [[2.0, 4.0, 12.0, 4.0, 12.0, 12.0, 2.0, 12.0]]
            annotations.append(ann)
            ann_id += 1

    if include_crowd:
        # Add a crowd annotation for the first image
        annotations.append({
            "id": ann_id,
            "image_id": images[0]["id"],
            "category_id": 2,
            "bbox": [0.0, 0.0, 16.0, 16.0],
            "iscrowd": 1,
        })

    return {"images": images, "annotations": annotations, "categories": categories}


def _setup_coco_dir(
    tmp_path: Path,
    num_images: int = 2,
    annotations_per_image: int = 2,
    include_crowd: bool = False,
    include_segmentation: bool = False,
) -> tuple[Path, Path]:
    """Create a COCO-style directory with fake images and an annotation JSON.

    Returns: (images_dir, annotation_file)
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    coco_data = _make_coco_json(
        num_images=num_images,
        annotations_per_image=annotations_per_image,
        include_crowd=include_crowd,
        include_segmentation=include_segmentation,
    )

    for img_info in coco_data["images"]:
        _save_image(images_dir / img_info["file_name"])

    ann_path = ann_dir / "instances.json"
    ann_path.write_text(json.dumps(coco_data), encoding="utf-8")

    return images_dir, ann_path


def _make_coco_yaml(
    tmp_path: Path,
    split: str = "train",
    num_images: int = 2,
    annotations_per_image: int = 2,
) -> Path:
    """Create a COCO YAML config file + the referenced images/annotations."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    split_images = dataset_root / f"{split}2017"
    split_images.mkdir()
    ann_dir = dataset_root / "annotations"
    ann_dir.mkdir()

    coco_data = _make_coco_json(
        num_images=num_images,
        annotations_per_image=annotations_per_image,
    )
    for img_info in coco_data["images"]:
        _save_image(split_images / img_info["file_name"])

    ann_file = ann_dir / f"instances_{split}2017.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    yaml_path = tmp_path / "coco.yaml"
    yaml_content = (
        f"path: {dataset_root}\n"
        f"{split}: {split}2017\n"
        f"{split}_annotations: annotations/instances_{split}2017.json\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def _setup_voc_dir(
    tmp_path: Path,
    num_images: int = 3,
    include_difficult: bool = False,
    use_imageset_file: bool = True,
) -> Path:
    """Create a minimal Pascal VOC directory structure."""
    root = tmp_path / "VOC2012"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "ImageSets" / "Main").mkdir(parents=True)

    ids = [f"{i:06d}" for i in range(1, num_images + 1)]
    classes = ["cat", "dog"]

    for i, img_id in enumerate(ids):
        _save_image(root / "JPEGImages" / f"{img_id}.jpg")
        # Build XML
        ann = ET.Element("annotation")
        ET.SubElement(ann, "filename").text = f"{img_id}.jpg"
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = "32"
        ET.SubElement(size, "height").text = "32"
        ET.SubElement(size, "depth").text = "3"
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = classes[i % len(classes)]
        ET.SubElement(obj, "difficult").text = "1" if include_difficult else "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "2"
        ET.SubElement(bndbox, "ymin").text = "4"
        ET.SubElement(bndbox, "xmax").text = "14"
        ET.SubElement(bndbox, "ymax").text = "18"
        tree_out = ET.ElementTree(ann)
        tree_out.write(str(root / "Annotations" / f"{img_id}.xml"))

    if use_imageset_file:
        (root / "ImageSets" / "Main" / "trainval.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )

    return root


def _setup_imagefolder(tmp_path: Path, classes: list[str] | None = None) -> Path:
    """Create an ImageFolder directory with synthetic images."""
    root = tmp_path / "flowers"
    if classes is None:
        classes = ["daisy", "rose", "sunflower"]
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for j in range(2):
            _save_image(cls_dir / f"img_{j}.jpg")
    return root


# ===========================================================================
# COCODetectionDataset
# ===========================================================================


class TestCOCODetectionDataset:
    def test_explicit_paths_construction(self, tmp_path):
        """Dataset loads from explicit root + annotation_file paths."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, num_images=2)
        ds = COCODetectionDataset(
            root=str(images_dir), annotation_file=str(ann_path)
        )
        assert len(ds) == 2

    def test_len_matches_number_of_images(self, tmp_path):
        """__len__ matches the number of images in the COCO JSON."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, num_images=4)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        assert len(ds) == 4

    def test_getitem_returns_pil_image_and_target_dict(self, tmp_path):
        """__getitem__ returns (PIL.Image, dict) tuple."""
        images_dir, ann_path = _setup_coco_dir(tmp_path)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        image, target = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target

    def test_boxes_are_xyxy_format(self, tmp_path):
        """Boxes are in xyxy format (converted from COCO xywh)."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, annotations_per_image=1)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        boxes = target["boxes"]
        assert boxes.shape[-1] == 4
        # COCO bbox [2, 4, 10, 8] → xyxy [2, 4, 12, 12]
        assert float(boxes[0, 0]) == pytest.approx(2.0)
        assert float(boxes[0, 1]) == pytest.approx(4.0)
        assert float(boxes[0, 2]) == pytest.approx(12.0)  # x + w
        assert float(boxes[0, 3]) == pytest.approx(12.0)  # y + h
        # x2 > x1, y2 > y1
        assert (boxes[:, 2] > boxes[:, 0]).all()
        assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_labels_are_zero_indexed(self, tmp_path):
        """Labels are 0-indexed based on category order in JSON."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, annotations_per_image=1)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        # category_id=1 → 0-indexed label 0 (first category)
        assert target["labels"][0].item() == 0

    def test_zero_annotation_image_returns_empty_tensors(self, tmp_path):
        """Images with 0 annotations return empty tensors (not errors)."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _save_image(images_dir / "empty.jpg")

        coco_data = {
            "images": [{"id": 1, "file_name": "empty.jpg", "width": 32, "height": 32}],
            "annotations": [],  # no annotations
            "categories": [{"id": 1, "name": "cat"}],
        }
        ann_path = tmp_path / "ann.json"
        ann_path.write_text(json.dumps(coco_data), encoding="utf-8")

        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)

    def test_crowd_annotations_excluded(self, tmp_path):
        """Crowd annotations (iscrowd=1) are excluded from targets."""
        images_dir, ann_path = _setup_coco_dir(
            tmp_path, num_images=1, annotations_per_image=1, include_crowd=True
        )
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        # 1 non-crowd annotation + 1 crowd annotation → only 1 box returned
        assert len(target["boxes"]) == 1
        assert len(target["labels"]) == 1

    def test_class_names_returns_correct_mapping(self, tmp_path):
        """class_names property returns {0: 'cat', 1: 'dog', ...}."""
        images_dir, ann_path = _setup_coco_dir(tmp_path)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        names = ds.class_names
        assert isinstance(names, dict)
        assert names[0] == "cat"
        assert names[1] == "dog"

    def test_invalid_annotation_path_raises_error(self, tmp_path):
        """FileNotFoundError raised when annotation file does not exist."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            COCODetectionDataset(
                root=str(images_dir),
                annotation_file=str(tmp_path / "nonexistent.json"),
            )

    def test_invalid_images_dir_raises_error(self, tmp_path):
        """FileNotFoundError raised when images directory does not exist."""
        _, ann_path = _setup_coco_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            COCODetectionDataset(
                root=str(tmp_path / "nonexistent_dir"),
                annotation_file=str(ann_path),
            )

    def test_yaml_mode_train_split(self, tmp_path):
        """Dataset loads correctly from YAML config with train split."""
        yaml_path = _make_coco_yaml(tmp_path, split="train", num_images=3)
        ds = COCODetectionDataset(str(yaml_path), split="train")
        assert len(ds) == 3
        image, target = ds[0]
        assert isinstance(image, Image.Image)
        assert "boxes" in target

    def test_yaml_mode_uses_train_annotations_key(self, tmp_path):
        """YAML config 'train_annotations' key is used for split='train'."""
        yaml_path = _make_coco_yaml(tmp_path, split="train", num_images=2)
        ds = COCODetectionDataset(str(yaml_path), split="train")
        assert len(ds) == 2

    def test_transforms_applied_to_sample(self, tmp_path):
        """Transforms are called with (image, target) and result returned."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, num_images=1)

        def mock_transform(image, target):
            target["transformed"] = True
            return image, target

        ds = COCODetectionDataset(
            root=str(images_dir),
            annotation_file=str(ann_path),
            transforms=mock_transform,
        )
        _, target = ds[0]
        assert target.get("transformed") is True

    def test_boxes_tensor_dtype_float32(self, tmp_path):
        """Boxes tensor has dtype float32."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, annotations_per_image=1)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        assert target["boxes"].dtype == torch.float32

    def test_labels_tensor_dtype_long(self, tmp_path):
        """Labels tensor has dtype int64 (long)."""
        images_dir, ann_path = _setup_coco_dir(tmp_path, annotations_per_image=1)
        ds = COCODetectionDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        assert target["labels"].dtype == torch.long


# ===========================================================================
# COCOSegmentationDataset
# ===========================================================================


class TestCOCOSegmentationDataset:
    def test_returns_binary_masks_in_target(self, tmp_path):
        """Target dict includes 'masks' key with binary mask tensor."""
        images_dir, ann_path = _setup_coco_dir(
            tmp_path, num_images=1, annotations_per_image=2, include_segmentation=True
        )
        ds = COCOSegmentationDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        assert "masks" in target
        masks = target["masks"]
        assert isinstance(masks, torch.Tensor)
        # Binary: values are 0 or 1
        assert masks.max().item() <= 1

    def test_mask_shape_matches_n_h_w(self, tmp_path):
        """Masks tensor shape is (N, H, W) matching num annotations and image size."""
        images_dir, ann_path = _setup_coco_dir(
            tmp_path,
            num_images=1,
            annotations_per_image=2,
            include_segmentation=True,
        )
        ds = COCOSegmentationDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        masks = target["masks"]
        # 2 annotations → N=2, image is 32x32
        assert masks.shape[0] == 2
        assert masks.shape[1] == 32
        assert masks.shape[2] == 32

    def test_boxes_and_labels_also_present(self, tmp_path):
        """COCOSegmentationDataset still returns boxes and labels."""
        images_dir, ann_path = _setup_coco_dir(
            tmp_path, num_images=1, annotations_per_image=1, include_segmentation=True
        )
        ds = COCOSegmentationDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        assert "boxes" in target
        assert "labels" in target

    def test_no_annotations_returns_empty_mask_tensor(self, tmp_path):
        """Image with zero annotations returns empty Tensor[0, H, W] for masks."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _save_image(images_dir / "empty.jpg")

        coco_data = {
            "images": [{"id": 1, "file_name": "empty.jpg", "width": 32, "height": 32}],
            "annotations": [],
            "categories": [{"id": 1, "name": "cat"}],
        }
        ann_path = tmp_path / "ann.json"
        ann_path.write_text(json.dumps(coco_data), encoding="utf-8")

        ds = COCOSegmentationDataset(root=str(images_dir), annotation_file=str(ann_path))
        _, target = ds[0]
        masks = target["masks"]
        assert masks.shape == (0, 32, 32)


# ===========================================================================
# VOCDetectionDataset
# ===========================================================================


class TestVOCDetectionDataset:
    def test_loads_voc_directory_structure(self, tmp_path):
        """Dataset loads from a standard VOC directory layout."""
        root = _setup_voc_dir(tmp_path, num_images=3)
        ds = VOCDetectionDataset(str(root))
        assert len(ds) == 3

    def test_getitem_returns_image_and_target(self, tmp_path):
        """__getitem__ returns (PIL.Image, dict) with boxes and labels."""
        root = _setup_voc_dir(tmp_path, num_images=2)
        ds = VOCDetectionDataset(str(root))
        image, target = ds[0]
        assert isinstance(image, Image.Image)
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target

    def test_boxes_from_xyxy_xml_format(self, tmp_path):
        """Boxes parsed from XML xmin/ymin/xmax/ymax in xyxy order."""
        root = _setup_voc_dir(tmp_path, num_images=1)
        ds = VOCDetectionDataset(str(root))
        _, target = ds[0]
        boxes = target["boxes"]
        # Synthetic XML: xmin=2, ymin=4, xmax=14, ymax=18
        assert boxes.shape[-1] == 4
        assert float(boxes[0, 0]) == pytest.approx(2.0)
        assert float(boxes[0, 1]) == pytest.approx(4.0)
        assert float(boxes[0, 2]) == pytest.approx(14.0)
        assert float(boxes[0, 3]) == pytest.approx(18.0)

    def test_class_names_auto_discovered(self, tmp_path):
        """class_names auto-discovered from annotation files and sorted."""
        root = _setup_voc_dir(tmp_path, num_images=2)
        ds = VOCDetectionDataset(str(root))
        names = ds.class_names
        assert isinstance(names, dict)
        sorted_names = sorted(names.values())
        assert sorted_names == sorted(names.values())  # alphabetically sorted

    def test_missing_annotation_file_returns_empty_tensors(self, tmp_path):
        """Missing XML annotation file returns empty tensors (no crash)."""
        root = _setup_voc_dir(tmp_path, num_images=2)
        # Remove one annotation file
        ann_files = list((root / "Annotations").glob("*.xml"))
        ann_to_remove = ann_files[0]
        ann_to_remove.unlink()

        ds = VOCDetectionDataset(str(root))
        # Finding the index corresponding to the removed annotation
        removed_id = ann_to_remove.stem
        idx = ds._image_ids.index(removed_id)
        _, target = ds[idx]
        assert target["boxes"].shape[0] == 0
        assert target["labels"].shape[0] == 0

    def test_difficult_objects_skipped_by_default(self, tmp_path):
        """Objects with difficult=1 are excluded when skip_difficult=True (default)."""
        root = _setup_voc_dir(tmp_path, num_images=1, include_difficult=True)
        ds = VOCDetectionDataset(str(root), skip_difficult=True)
        _, target = ds[0]
        # All objects are difficult → empty tensors
        assert target["boxes"].shape[0] == 0

    def test_difficult_objects_included_when_disabled(self, tmp_path):
        """skip_difficult=False includes difficult objects."""
        root = _setup_voc_dir(tmp_path, num_images=1, include_difficult=True)
        ds = VOCDetectionDataset(str(root), skip_difficult=False)
        _, target = ds[0]
        assert target["boxes"].shape[0] == 1

    def test_fallback_when_no_imageset_file(self, tmp_path):
        """Falls back to scanning Annotations/*.xml when ImageSets file is absent."""
        root = _setup_voc_dir(tmp_path, num_images=3, use_imageset_file=False)
        ds = VOCDetectionDataset(str(root), image_set="trainval")
        assert len(ds) == 3

    def test_incorrect_root_raises_training_error(self, tmp_path):
        """Missing Annotations dir raises TrainingError."""
        with pytest.raises(TrainingError):
            VOCDetectionDataset(str(tmp_path / "nonexistent"))

    def test_boxes_tensor_shape(self, tmp_path):
        """Boxes tensor has shape (N, 4)."""
        root = _setup_voc_dir(tmp_path, num_images=1)
        ds = VOCDetectionDataset(str(root))
        _, target = ds[0]
        assert target["boxes"].ndim == 2
        assert target["boxes"].shape[1] == 4


# ===========================================================================
# ImageFolderDataset
# ===========================================================================


class TestImageFolderDataset:
    def test_auto_discovers_class_names_from_subdirs(self, tmp_path):
        """Class names are discovered from immediate subdirectory names."""
        root = _setup_imagefolder(tmp_path, classes=["cat", "dog"])
        ds = ImageFolderDataset(str(root))
        assert set(ds.class_names.values()) == {"cat", "dog"}

    def test_class_names_sorted_alphabetically(self, tmp_path):
        """Class names are sorted alphabetically for deterministic indices."""
        root = _setup_imagefolder(tmp_path, classes=["rose", "daisy", "sunflower"])
        ds = ImageFolderDataset(str(root))
        names = [ds.class_names[i] for i in sorted(ds.class_names)]
        assert names == sorted(names)
        # daisy=0, rose=1, sunflower=2
        assert ds.class_names[0] == "daisy"
        assert ds.class_names[1] == "rose"
        assert ds.class_names[2] == "sunflower"

    def test_getitem_returns_image_and_label_dict(self, tmp_path):
        """__getitem__ returns (PIL.Image, {"label": int}) tuple."""
        root = _setup_imagefolder(tmp_path, classes=["cat", "dog"])
        ds = ImageFolderDataset(str(root))
        image, target = ds[0]
        assert isinstance(image, Image.Image)
        assert "label" in target
        assert isinstance(target["label"], int)

    def test_filters_by_image_extensions(self, tmp_path):
        """Non-image files are not included in the dataset."""
        root = tmp_path / "root"
        cls_dir = root / "cat"
        cls_dir.mkdir(parents=True)
        _save_image(cls_dir / "valid.jpg")
        (cls_dir / "readme.txt").write_text("not an image")
        (cls_dir / "script.py").write_text("not an image")

        ds = ImageFolderDataset(str(root))
        assert len(ds) == 1  # only valid.jpg

    def test_skips_hidden_files_and_directories(self, tmp_path):
        """Files and directories starting with '.' are ignored."""
        root = tmp_path / "root"
        cls_dir = root / "cat"
        cls_dir.mkdir(parents=True)
        hidden_cls_dir = root / ".hidden_class"
        hidden_cls_dir.mkdir()
        _save_image(cls_dir / "visible.jpg")
        _save_image(cls_dir / ".hidden_img.jpg")
        _save_image(hidden_cls_dir / "img.jpg")

        ds = ImageFolderDataset(str(root))
        # Only 1 class visible; only 1 visible image in it
        assert len(ds.class_names) == 1
        assert len(ds) == 1

    def test_empty_root_directory_raises_training_error(self, tmp_path):
        """Empty directory (no class subdirs) raises TrainingError."""
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(TrainingError):
            ImageFolderDataset(str(root))

    def test_single_class_works(self, tmp_path):
        """A single class subdirectory is valid."""
        root = _setup_imagefolder(tmp_path, classes=["only_class"])
        ds = ImageFolderDataset(str(root))
        assert len(ds.class_names) == 1
        assert ds.class_names[0] == "only_class"

    def test_len_matches_total_images(self, tmp_path):
        """__len__ equals the total number of image files across all classes."""
        root = _setup_imagefolder(tmp_path, classes=["a", "b", "c"])
        ds = ImageFolderDataset(str(root))
        # Each class has 2 images → total = 6
        assert len(ds) == 6

    def test_transforms_applied(self, tmp_path):
        """Transforms callable is invoked on each sample."""
        root = _setup_imagefolder(tmp_path, classes=["cat"])

        def add_flag(image, target):
            target["flag"] = 99
            return image, target

        ds = ImageFolderDataset(str(root), transforms=add_flag)
        _, target = ds[0]
        assert target["flag"] == 99


# ===========================================================================
# Collators
# ===========================================================================


class TestCollators:
    def test_detection_collate_returns_lists(self):
        """detection_collate_fn returns (list[image], list[dict])."""
        img = torch.rand(3, 32, 32)
        batch = [
            (img, {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.long)}),
            (img, {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long)}),
        ]
        images, targets = detection_collate_fn(batch)
        assert isinstance(images, list)
        assert isinstance(targets, list)
        assert len(images) == 2
        assert len(targets) == 2

    def test_classification_collate_stacks_images_and_labels(self):
        """classification_collate_fn stacks images into [N, C, H, W] tensor."""
        batch = [
            (torch.rand(3, 32, 32), {"label": 0}),
            (torch.rand(3, 32, 32), {"label": 1}),
            (torch.rand(3, 32, 32), {"label": 0}),
        ]
        images, labels = classification_collate_fn(batch)
        assert isinstance(images, torch.Tensor)
        assert images.shape == (3, 3, 32, 32)
        assert isinstance(labels, torch.Tensor)
        assert labels.tolist() == [0, 1, 0]

    def test_empty_batch_detection_returns_empty_lists(self):
        """detection_collate_fn handles an empty batch gracefully."""
        images, targets = detection_collate_fn([])
        assert images == []
        assert targets == []

    def test_empty_batch_classification_returns_empty_tensors(self):
        """classification_collate_fn handles an empty batch gracefully."""
        images, labels = classification_collate_fn([])
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.numel() == 0

    def test_segmentation_collate_returns_lists(self):
        """segmentation_collate_fn returns (list, list) matching detection."""
        img = torch.rand(3, 32, 32)
        batch = [
            (img, {"boxes": torch.zeros(1, 4), "masks": torch.zeros(1, 32, 32)}),
        ]
        images, targets = segmentation_collate_fn(batch)
        assert isinstance(images, list)
        assert isinstance(targets, list)
        assert len(images) == 1

    def test_empty_batch_segmentation_returns_empty_lists(self):
        """segmentation_collate_fn handles an empty batch gracefully."""
        images, targets = segmentation_collate_fn([])
        assert images == []
        assert targets == []


# ===========================================================================
# DatasetFactory
# ===========================================================================


class TestDatasetFactory:
    def test_auto_detects_coco_from_yaml(self, tmp_path):
        """YAML file with 'train_annotations' key → COCODetectionDataset."""
        yaml_path = _make_coco_yaml(tmp_path, split="train", num_images=2)
        dataset, collate_fn = DatasetFactory.create("detect", str(yaml_path), split="train")
        assert isinstance(dataset, COCODetectionDataset)
        assert collate_fn is detection_collate_fn

    def test_auto_detects_imagefolder_from_directory(self, tmp_path):
        """Directory with only subdirectories → ImageFolderDataset (classify)."""
        root = _setup_imagefolder(tmp_path, classes=["cat", "dog"])
        dataset, collate_fn = DatasetFactory.create("classify", str(root))
        assert isinstance(dataset, ImageFolderDataset)
        assert collate_fn is classification_collate_fn

    def test_pass_through_pytorch_dataset(self, tmp_path):
        """An existing torch.utils.data.Dataset is passed through unchanged."""
        import torch.utils.data as tud

        class DummyDataset(tud.Dataset):
            def __getitem__(self, i):
                return i

            def __len__(self):
                return 5

        dummy = DummyDataset()
        dataset, collate_fn = DatasetFactory.create("detect", dummy)
        assert dataset is dummy

    def test_unknown_format_raises_training_error(self, tmp_path):
        """Unrecognizable directory format raises TrainingError."""
        # Directory with regular files at root → not COCO, not VOC, not ImageFolder
        root = tmp_path / "unknown"
        root.mkdir()
        _save_image(root / "img.jpg")
        (root / "data.txt").write_text("some text")

        with pytest.raises(TrainingError):
            DatasetFactory.create("detect", str(root))

    def test_auto_detects_voc_from_directory(self, tmp_path):
        """Directory with Annotations/*.xml → VOCDetectionDataset."""
        root = _setup_voc_dir(tmp_path, num_images=2)
        dataset, collate_fn = DatasetFactory.create("detect", str(root))
        assert isinstance(dataset, VOCDetectionDataset)
        assert collate_fn is detection_collate_fn

    def test_detect_returns_detection_collate(self, tmp_path):
        """Factory returns detection_collate_fn for task='detect'."""
        yaml_path = _make_coco_yaml(tmp_path, split="train", num_images=1)
        _, collate_fn = DatasetFactory.create("detect", str(yaml_path), split="train")
        assert collate_fn is detection_collate_fn

    def test_classify_returns_classification_collate(self, tmp_path):
        """Factory returns classification_collate_fn for task='classify'."""
        root = _setup_imagefolder(tmp_path, classes=["a", "b"])
        _, collate_fn = DatasetFactory.create("classify", str(root))
        assert collate_fn is classification_collate_fn

    def test_unsupported_task_raises_training_error(self, tmp_path):
        """Unsupported task string raises TrainingError."""
        with pytest.raises(TrainingError):
            DatasetFactory.create("track", "some_path")

    def test_nonexistent_path_raises_training_error(self, tmp_path):
        """Non-existent data path raises TrainingError."""
        with pytest.raises(TrainingError):
            DatasetFactory.create("detect", str(tmp_path / "nonexistent"))

    def test_coco_yaml_with_fallback_annotations_key(self, tmp_path):
        """YAML with generic 'annotations' key (backward-compat) loads for detect."""
        dataset_root = tmp_path / "dataset"
        images_dir = dataset_root / "images"
        images_dir.mkdir(parents=True)
        ann_dir = dataset_root / "annotations"
        ann_dir.mkdir()

        coco_data = _make_coco_json(num_images=2)
        for img in coco_data["images"]:
            _save_image(images_dir / img["file_name"])
        ann_file = ann_dir / "instances.json"
        ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

        yaml_content = (
            f"path: {dataset_root}\n"
            "train: images\n"
            "annotations: annotations/instances.json\n"
        )
        yaml_path = tmp_path / "coco_compat.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        dataset, _ = DatasetFactory.create("detect", str(yaml_path), split="train")
        assert isinstance(dataset, COCODetectionDataset)
        assert len(dataset) == 2
