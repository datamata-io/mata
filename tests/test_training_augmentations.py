"""Tests for mata.training.augmentations — Task E3.

Covers BasicDetectionAugmentation, BasicClassificationAugmentation,
BasicSegmentationAugmentation, AlbumentationsWrapper, and AugmentationFactory.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mata.training.augmentations import (
    AugmentationFactory,
    AlbumentationsWrapper,
    BasicClassificationAugmentation,
    BasicDetectionAugmentation,
    BasicSegmentationAugmentation,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_pil(h: int = 100, w: int = 100, fill: int = 128) -> Image.Image:
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    return Image.fromarray(arr)


def _empty_detect_target() -> dict:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.long),
    }


def _detect_target(boxes: list, labels: list | None = None) -> dict:
    t = torch.tensor(boxes, dtype=torch.float32)
    l = torch.zeros(len(boxes), dtype=torch.long) if labels is None else torch.tensor(labels, dtype=torch.long)
    return {"boxes": t, "labels": l}


def _seg_target(mask: torch.Tensor, boxes: torch.Tensor | None = None) -> dict:
    if boxes is None:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
    return {
        "boxes": boxes,
        "labels": torch.zeros(len(boxes), dtype=torch.long),
        "masks": mask,
    }


# ─── BasicDetectionAugmentation ─────────────────────────────────────────────

class TestBasicDetectionAugmentation:
    """Tests for the detection augmentation pipeline."""

    def test_returns_tuple(self):
        """Output is a (tensor, dict) tuple."""
        aug = BasicDetectionAugmentation(size=50, train=False)
        img, target = aug(_make_pil(50, 50), _empty_detect_target())
        assert isinstance(img, torch.Tensor)
        assert isinstance(target, dict)

    def test_image_output_is_tensor(self):
        """The transformed image is a torch.Tensor."""
        aug = BasicDetectionAugmentation(size=50, train=False)
        img_t, _ = aug(_make_pil(50, 50), _empty_detect_target())
        assert isinstance(img_t, torch.Tensor)

    def test_image_has_3_channels(self):
        """Transformed image has shape (3, H, W)."""
        aug = BasicDetectionAugmentation(size=50, train=False)
        img_t, _ = aug(_make_pil(50, 50), _empty_detect_target())
        assert img_t.ndim == 3
        assert img_t.shape[0] == 3

    def test_val_mode_fixed_size(self):
        """Val mode resizes to (size, size) regardless of input aspect ratio."""
        aug = BasicDetectionAugmentation(size=64, train=False)
        img_t, _ = aug(_make_pil(h=100, w=150), _empty_detect_target())
        assert img_t.shape == (3, 64, 64)

    def test_train_mode_preserves_aspect_ratio(self):
        """Train mode resizes the shorter edge to `size` (aspect ratio preserved)."""
        # Input: 50 tall × 100 wide — shorter edge is 50, longer is 100
        aug = BasicDetectionAugmentation(size=100, train=True)
        img_t, _ = aug(_make_pil(h=50, w=100), _empty_detect_target())
        # After resize(100): shorter edge 50 → 100, longer edge 100 → 200
        assert img_t.shape == (3, 100, 200)

    def test_normalization_applied(self):
        """ImageNet normalization pushes white-pixel values above 1.0."""
        aug = BasicDetectionAugmentation(size=32, train=False)
        # White image: all pixel values = 1.0 before normalization
        white_img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        img_t, _ = aug(white_img, _empty_detect_target())
        # After (1.0 - 0.485) / 0.229 ≈ 2.25 for R channel
        assert img_t.max().item() > 1.0

    def test_flip_mirrors_box_coordinates(self):
        """With flip_prob=1.0, box x-coordinates are mirrored horizontally."""
        aug = BasicDetectionAugmentation(
            size=100,
            flip_prob=1.0,
            jitter_brightness=0,
            jitter_contrast=0,
            jitter_saturation=0,
            jitter_hue=0,
            train=True,
        )
        # Square 100×100 image; box on the right side (x-center ≈ 80)
        img = _make_pil(100, 100)
        target = _detect_target([[70.0, 10.0, 90.0, 80.0]])
        _, out_target = aug(img, target)
        boxes = out_target["boxes"]
        assert boxes.shape == (1, 4), "Box count should not change after flip"
        center_x = (boxes[0, 0] + boxes[0, 2]) / 2.0
        # After flip the box should be on the left half
        assert center_x.item() < 50.0

    def test_boxes_remain_valid_after_transform(self):
        """x1 < x2 and y1 < y2 for all boxes after augmentation."""
        aug = BasicDetectionAugmentation(size=100, train=True)
        target = _detect_target([[10.0, 20.0, 60.0, 80.0], [50.0, 30.0, 90.0, 70.0]])
        for _ in range(5):
            img_t, out_t = aug(_make_pil(200, 200), target)
            boxes = out_t["boxes"]
            assert (boxes[:, 0] < boxes[:, 2]).all(), "x1 must be < x2"
            assert (boxes[:, 1] < boxes[:, 3]).all(), "y1 must be < y2"

    def test_empty_boxes_handled(self):
        """Augmentation with no boxes in the target produces no error."""
        aug = BasicDetectionAugmentation(size=50, train=True)
        img_t, out_t = aug(_make_pil(50, 50), _empty_detect_target())
        assert out_t["boxes"].shape == (0, 4)


# ─── BasicClassificationAugmentation ────────────────────────────────────────

class TestBasicClassificationAugmentation:
    """Tests for the classification augmentation pipeline."""

    def test_returns_tuple(self):
        """Output is a (tensor, dict) tuple."""
        aug = BasicClassificationAugmentation(size=224, train=False)
        result = aug(_make_pil(256, 256), {"label": 3})
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)

    def test_output_shape_is_correct(self):
        """Output tensor has shape (3, size, size)."""
        aug = BasicClassificationAugmentation(size=224, train=False)
        img_t, _ = aug(_make_pil(300, 300), {"label": 0})
        assert img_t.shape == (3, 224, 224)

    def test_normalization_applied(self):
        """ImageNet normalization is applied — white-image values exceed 1.0."""
        aug = BasicClassificationAugmentation(size=32, train=False)
        white = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
        img_t, _ = aug(white, {"label": 0})
        assert img_t.max().item() > 1.0

    def test_label_preserved_in_target(self):
        """Target dict is returned unmodified (label not dropped)."""
        aug = BasicClassificationAugmentation(size=32, train=False)
        _, out_t = aug(_make_pil(64, 64), {"label": 7})
        assert out_t["label"] == 7

    def test_val_mode_is_deterministic(self):
        """Val mode produces identical output for the same input."""
        aug = BasicClassificationAugmentation(size=64, train=False)
        img = _make_pil(128, 128, fill=100)
        t1, _ = aug(img, {"label": 0})
        t2, _ = aug(img, {"label": 0})
        assert torch.allclose(t1, t2)

    def test_train_mode_is_stochastic(self):
        """Training mode (random crop + jitter) produces varying outputs."""
        aug = BasicClassificationAugmentation(size=64, train=True)
        img = _make_pil(200, 200)
        outputs = [aug(img, {"label": 0})[0] for _ in range(8)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Random augmentations should produce different outputs"


# ─── BasicSegmentationAugmentation ──────────────────────────────────────────

class TestBasicSegmentationAugmentation:
    """Tests for the segmentation augmentation pipeline."""

    def test_returns_tuple(self):
        """Output is a (tensor, dict) tuple."""
        aug = BasicSegmentationAugmentation(size=50, train=False)
        mask = torch.zeros(1, 50, 50, dtype=torch.uint8)
        img_t, out_t = aug(_make_pil(50, 50), _seg_target(mask))
        assert isinstance(img_t, torch.Tensor)
        assert isinstance(out_t, dict)

    def test_mask_dimensions_match_image_after_resize(self):
        """After val-mode resize, mask spatial dims match the image."""
        aug = BasicSegmentationAugmentation(size=64, train=False)
        mask = torch.zeros(2, 100, 100, dtype=torch.uint8)
        img_t, out_t = aug(_make_pil(100, 100), _seg_target(mask))
        _, H, W = img_t.shape
        assert out_t["masks"].shape[-2] == H
        assert out_t["masks"].shape[-1] == W

    def test_binary_mask_values_preserved(self):
        """Mask values remain 0 or 1 (binary) after transforms."""
        aug = BasicSegmentationAugmentation(size=50, train=True)
        mask = torch.zeros(1, 50, 50, dtype=torch.uint8)
        mask[0, 10:30, 10:30] = 1
        _, out_t = aug(_make_pil(50, 50), _seg_target(mask))
        unique_vals = out_t["masks"].unique().tolist()
        for v in unique_vals:
            # Accept int (0/1) or float (0.0/1.0)
            assert v in (0, 1, 0.0, 1.0), f"Unexpected mask value: {v}"

    def test_image_and_mask_flip_consistently(self):
        """Horizontal flip is applied to both image and mask simultaneously."""
        aug = BasicSegmentationAugmentation(
            size=50,
            flip_prob=1.0,          # always flip
            jitter_brightness=0,
            jitter_contrast=0,
            jitter_saturation=0,
            jitter_hue=0,
            train=True,
        )
        # Mask region on the right side (columns 35–45)
        mask = torch.zeros(1, 50, 50, dtype=torch.uint8)
        mask[0, 10:40, 35:45] = 1
        _, out_t = aug(_make_pil(50, 50), _seg_target(mask))
        out_mask = out_t["masks"]
        # After flip, region should have moved to the left half
        left_sum = out_mask[0, :, :25].sum().item()
        right_sum = out_mask[0, :, 25:].sum().item()
        assert left_sum > right_sum, "Flipped mask region should be on the left"

    def test_empty_masks_handled(self):
        """Augmentation with zero-instance mask produces no error."""
        aug = BasicSegmentationAugmentation(size=50, train=True)
        mask = torch.zeros(0, 50, 50, dtype=torch.uint8)
        img_t, out_t = aug(_make_pil(50, 50), _seg_target(mask))
        assert isinstance(img_t, torch.Tensor)
        assert out_t["masks"].shape[0] == 0


# ─── AlbumentationsWrapper ───────────────────────────────────────────────────

class TestAlbumentationsWrapper:
    """Tests for the optional albumentations integration."""

    def test_raises_import_error_if_not_installed(self):
        """AlbumentationsWrapper raises ImportError when albumentations is absent."""
        # albumentations is not in the test environment
        with patch.dict("sys.modules", {"albumentations": None}):
            with pytest.raises(ImportError, match="albumentations"):
                AlbumentationsWrapper(MagicMock())

    def test_wraps_pipeline_and_returns_tensor(self):
        """Wrapper calls the underlying transform and converts output to tensor."""
        mock_albu = MagicMock()
        with patch.dict("sys.modules", {"albumentations": mock_albu}):
            mock_transform = MagicMock(return_value={
                "image": np.zeros((20, 20, 3), dtype=np.uint8),
                "bboxes": [],
                "class_labels": [],
            })
            wrapper = AlbumentationsWrapper(mock_transform)
            img_t, out_t = wrapper(_make_pil(20, 20), _empty_detect_target())

        assert isinstance(img_t, torch.Tensor)
        assert img_t.shape == (3, 20, 20)
        assert img_t.dtype == torch.float32

    def test_bbox_format_conversion_xyxy_to_pascal_voc(self):
        """Boxes are passed to albumentations in pascal_voc (xyxy) list format."""
        mock_albu = MagicMock()
        with patch.dict("sys.modules", {"albumentations": mock_albu}):
            mock_transform = MagicMock(return_value={
                "image": np.zeros((100, 100, 3), dtype=np.uint8),
                "bboxes": [[10.0, 20.0, 50.0, 60.0]],
                "class_labels": [1],
            })
            wrapper = AlbumentationsWrapper(mock_transform)
            img = _make_pil(100, 100)
            target = {
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
                "labels": torch.tensor([1]),
            }
            img_t, out_t = wrapper(img, target)

        # Verify boxes were passed as list-of-lists to the transform
        call_kwargs = mock_transform.call_args[1]
        assert "bboxes" in call_kwargs
        assert call_kwargs["bboxes"] == [[10.0, 20.0, 50.0, 60.0]]

        # Output boxes re-converted to tensor
        assert isinstance(out_t["boxes"], torch.Tensor)
        assert out_t["boxes"].shape == (1, 4)

    def test_mask_transforms_applied(self):
        """Masks are converted to numpy, passed to albumentations, and back to tensor."""
        mock_albu = MagicMock()
        out_mask_np = np.zeros((30, 30), dtype=np.uint8)
        out_mask_np[5:15, 5:15] = 1
        with patch.dict("sys.modules", {"albumentations": mock_albu}):
            mock_transform = MagicMock(return_value={
                "image": np.zeros((30, 30, 3), dtype=np.uint8),
                "bboxes": [],
                "class_labels": [],
                "masks": [out_mask_np],
            })
            wrapper = AlbumentationsWrapper(mock_transform)
            in_mask = torch.zeros(1, 30, 30, dtype=torch.uint8)
            in_mask[0, 5:15, 5:15] = 1
            target = {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros(0, dtype=torch.long),
                "masks": in_mask,
            }
            _, out_t = wrapper(_make_pil(30, 30), target)

        assert "masks" in out_t
        assert isinstance(out_t["masks"], torch.Tensor)
        assert out_t["masks"].shape == (1, 30, 30)


# ─── AugmentationFactory ─────────────────────────────────────────────────────

class TestAugmentationFactory:
    """Tests for the augmentation factory."""

    def test_detect_task_returns_basic_detection(self):
        """Factory creates BasicDetectionAugmentation for 'detect' task."""
        aug = AugmentationFactory.create("detect")
        assert isinstance(aug, BasicDetectionAugmentation)

    def test_classify_task_returns_basic_classification(self):
        """Factory creates BasicClassificationAugmentation for 'classify' task."""
        aug = AugmentationFactory.create("classify")
        assert isinstance(aug, BasicClassificationAugmentation)

    def test_segment_task_returns_basic_segmentation(self):
        """Factory creates BasicSegmentationAugmentation for 'segment' task."""
        aug = AugmentationFactory.create("segment")
        assert isinstance(aug, BasicSegmentationAugmentation)

    def test_custom_size_forwarded_to_augmentation(self):
        """Config 'size' key is forwarded to the underlying augmentation class."""
        aug = AugmentationFactory.create("detect", config={"size": 320})
        assert aug.size == 320

    def test_albumentations_type_requires_albumentations(self):
        """Requesting albumentations type raises ImportError if not installed."""
        config = {
            "type": "albumentations",
            "transform": MagicMock(),
        }
        with patch.dict("sys.modules", {"albumentations": None}):
            with pytest.raises(ImportError, match="albumentations"):
                AugmentationFactory.create("detect", config=config)

    def test_unknown_task_raises_value_error(self):
        """Unsupported task name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="detect"):
            AugmentationFactory.create("unknown_task")

    def test_val_mode_passed_to_augmentation(self):
        """train=False is forwarded — created augmentation is in val mode."""
        aug = AugmentationFactory.create("classify", train=False)
        assert aug.train is False

    def test_albumentations_config_returns_wrapper(self):
        """Config with type='albumentations' returns AlbumentationsWrapper."""
        mock_albu = MagicMock()
        config = {"type": "albumentations", "transform": MagicMock()}
        with patch.dict("sys.modules", {"albumentations": mock_albu}):
            aug = AugmentationFactory.create("detect", config=config)
        assert isinstance(aug, AlbumentationsWrapper)
