"""Unit tests for mata.eval.metrics.iou — box IoU, mask IoU, and helpers.

Covers Task A2 (box IoU) and Task A3 (mask IoU) acceptance criteria.
All tests use only numpy + synthetic data; no external model weights needed.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from mata.eval.metrics.iou import (
    COCO_IOU_THRESHOLDS,
    _decode_rle_fallback,
    _normalize_mask,
    _polygon_to_binary,
    box_iou,
    box_iou_batch,
    mask_iou,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def _binary_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return a (h, w) bool mask with a filled rectangle."""
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _rle_encode(binary: np.ndarray) -> dict:
    """Simple RLE encoder using pycocotools (skipped if not available)."""
    pytest.importorskip("pycocotools")
    from pycocotools import mask as coco_mask_api  # type: ignore

    fortran = np.asfortranarray(binary.astype(np.uint8))
    return coco_mask_api.encode(fortran)


# ===========================================================================
# COCO_IOU_THRESHOLDS
# ===========================================================================


class TestCocoIouThresholds:
    def test_length(self):
        assert len(COCO_IOU_THRESHOLDS) == 10

    def test_first_threshold(self):
        assert COCO_IOU_THRESHOLDS[0] == pytest.approx(0.50)

    def test_last_threshold(self):
        assert COCO_IOU_THRESHOLDS[-1] == pytest.approx(0.95)

    def test_step(self):
        for i in range(1, len(COCO_IOU_THRESHOLDS)):
            assert COCO_IOU_THRESHOLDS[i] - COCO_IOU_THRESHOLDS[i - 1] == pytest.approx(0.05)


# ===========================================================================
# box_iou
# ===========================================================================


class TestBoxIou:
    # ---- basic correctness ------------------------------------------------

    def test_identical_boxes_iou_equals_one(self):
        b = _make_box(0, 0, 10, 10)
        result = box_iou(b, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_zero(self):
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(20, 20, 30, 30)
        assert box_iou(b1, b2)[0, 0] == pytest.approx(0.0)

    def test_adjacent_boxes_iou_zero(self):
        """Boxes that share only an edge have zero intersection area."""
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(10, 0, 20, 10)
        assert box_iou(b1, b2)[0, 0] == pytest.approx(0.0)

    def test_partially_overlapping_boxes(self):
        # b1=[0,0,10,10] area=100, b2=[5,0,15,10] area=100, inter=[5,0,10,10] area=50
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(5, 0, 15, 10)
        iou = box_iou(b1, b2)[0, 0]
        # union = 150, inter = 50 → iou = 50/150 ≈ 0.3333
        assert iou == pytest.approx(50.0 / 150.0, abs=1e-5)

    def test_fully_nested_smaller_in_larger(self):
        # inner=[2,2,8,8] area=36, outer=[0,0,10,10] area=100
        inner = _make_box(2, 2, 8, 8)
        outer = _make_box(0, 0, 10, 10)
        iou = box_iou(inner, outer)[0, 0]
        assert iou == pytest.approx(36.0 / 100.0, abs=1e-5)

    def test_symmetry(self):
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(5, 5, 15, 15)
        assert box_iou(b1, b2)[0, 0] == pytest.approx(box_iou(b2, b1)[0, 0])

    # ---- output shape / dtype ---------------------------------------------

    def test_output_dtype_float32(self):
        b = _make_box(0, 0, 5, 5)
        result = box_iou(b, b)
        assert result.dtype == np.float32

    def test_output_shape_n_m(self):
        b1 = np.tile(np.array([0, 0, 10, 10], dtype=np.float32), (3, 1))
        b2 = np.tile(np.array([0, 0, 10, 10], dtype=np.float32), (5, 1))
        result = box_iou(b1, b2)
        assert result.shape == (3, 5)

    def test_vectorized_pairwise_correctness(self):
        """N=2, M=2 — check all four pairwise values."""
        b1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b2 = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
        result = box_iou(b1, b2)
        assert result[0, 0] == pytest.approx(1.0)  # identical
        assert result[1, 0] == pytest.approx(0.0)  # no overlap
        assert result[0, 1] == pytest.approx(25.0 / 100.0)  # 5×5 / 10×10
        assert result[1, 1] == pytest.approx(0.0)  # no overlap

    # ---- edge cases -------------------------------------------------------

    def test_empty_boxes1_returns_zero_rows(self):
        b1 = np.zeros((0, 4), dtype=np.float32)
        b2 = _make_box(0, 0, 10, 10)
        result = box_iou(b1, b2)
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    def test_empty_boxes2_returns_zero_cols(self):
        b1 = _make_box(0, 0, 10, 10)
        b2 = np.zeros((0, 4), dtype=np.float32)
        result = box_iou(b1, b2)
        assert result.shape == (1, 0)
        assert result.dtype == np.float32

    def test_both_empty_returns_empty_matrix(self):
        b1 = np.zeros((0, 4), dtype=np.float32)
        b2 = np.zeros((0, 4), dtype=np.float32)
        result = box_iou(b1, b2)
        assert result.shape == (0, 0)

    def test_zero_area_box_returns_zero(self):
        """A degenerate box (x1==x2) must not cause division errors."""
        b1 = _make_box(5, 5, 5, 10)  # zero width
        b2 = _make_box(0, 0, 10, 10)
        assert box_iou(b1, b2)[0, 0] == pytest.approx(0.0)

    def test_1d_input_treated_as_single_box(self):
        b1 = np.array([0, 0, 10, 10], dtype=np.float32)
        b2 = np.array([0, 0, 10, 10], dtype=np.float32)
        result = box_iou(b1, b2)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)


# ===========================================================================
# box_iou_batch
# ===========================================================================


class TestBoxIouBatch:
    def test_output_shape(self):
        b1 = np.tile(np.array([0, 0, 10, 10], dtype=np.float32), (3, 1))
        b2 = np.tile(np.array([0, 0, 10, 10], dtype=np.float32), (5, 1))
        result = box_iou_batch(b1, b2, COCO_IOU_THRESHOLDS)
        assert result.shape == (10, 3, 5)

    def test_output_dtype_bool(self):
        b = _make_box(0, 0, 10, 10)
        result = box_iou_batch(b, b, [0.5])
        assert result.dtype == bool

    def test_identical_boxes_all_thresholds_true(self):
        b = _make_box(0, 0, 10, 10)
        result = box_iou_batch(b, b, COCO_IOU_THRESHOLDS)
        assert result.all()

    def test_non_overlapping_all_thresholds_false(self):
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(50, 50, 60, 60)
        result = box_iou_batch(b1, b2, COCO_IOU_THRESHOLDS)
        assert not result.any()

    def test_threshold_boundary(self):
        """iou=0.5 → True at t=0.5, False at t=0.55."""
        b1 = _make_box(0, 0, 10, 10)
        b2 = _make_box(5, 0, 15, 10)  # iou = 1/3 ≈ 0.333
        result = box_iou_batch(b1, b2, [0.3, 0.34])
        assert result[0, 0, 0] is np.bool_(True)  # 0.333 ≥ 0.30
        assert result[1, 0, 0] is np.bool_(False)  # 0.333 < 0.34

    def test_empty_pred_boxes(self):
        b1 = np.zeros((0, 4), dtype=np.float32)
        b2 = _make_box(0, 0, 10, 10)
        result = box_iou_batch(b1, b2, COCO_IOU_THRESHOLDS)
        assert result.shape == (10, 0, 1)

    def test_single_threshold(self):
        b = _make_box(0, 0, 10, 10)
        result = box_iou_batch(b, b, [0.5])
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0]


# ===========================================================================
# _decode_rle_fallback
# ===========================================================================


class TestDecodeRleFallback:
    def test_simple_rectangle(self):
        h, w = 4, 4
        # Fortran-order (column-major): fill column 1-2, rows 1-2 (0-indexed)
        binary = np.zeros((h, w), dtype=bool)
        binary[1:3, 1:3] = True

        # Build uncompressed counts manually in Fortran order
        flat = binary.reshape(-1, order="F")
        counts = []
        flat[0]
        run = 0
        val = False
        for v in flat:
            if v == val:
                run += 1
            else:
                counts.append(run)
                run = 1
                val = v
        counts.append(run)

        rle = {"counts": counts, "size": [h, w]}
        result = _decode_rle_fallback(rle, h, w)
        np.testing.assert_array_equal(result, binary)

    def test_all_zeros(self):
        h, w = 3, 3
        rle = {"counts": [h * w], "size": [h, w]}
        result = _decode_rle_fallback(rle, h, w)
        assert result.shape == (h, w)
        assert not result.any()

    def test_all_ones(self):
        h, w = 3, 3
        rle = {"counts": [0, h * w], "size": [h, w]}
        result = _decode_rle_fallback(rle, h, w)
        assert result.shape == (h, w)
        assert result.all()


# ===========================================================================
# _normalize_mask
# ===========================================================================


class TestNormalizeMask:
    IMAGE_SHAPE = (20, 20)

    def test_binary_ndarray_passthrough(self):
        m = _binary_mask(20, 20, 5, 5, 15, 15)
        result = _normalize_mask(m, self.IMAGE_SHAPE)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, m)

    def test_binary_ndarray_non_bool_converted(self):
        m = np.zeros((20, 20), dtype=np.uint8)
        m[5:15, 5:15] = 1
        result = _normalize_mask(m, self.IMAGE_SHAPE)
        assert result.dtype == bool

    def test_binary_ndarray_wrong_shape_raises(self):
        m = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            _normalize_mask(m, self.IMAGE_SHAPE)

    def test_unrecognised_type_raises(self):
        with pytest.raises(NotImplementedError):
            _normalize_mask("not_a_mask", self.IMAGE_SHAPE)  # type: ignore

    def test_rle_mask_with_pycocotools(self):
        pytest.importorskip("pycocotools")
        binary = _binary_mask(20, 20, 2, 2, 10, 10)
        rle = _rle_encode(binary)
        result = _normalize_mask(rle, self.IMAGE_SHAPE)
        np.testing.assert_array_equal(result, binary)

    def test_rle_mask_fallback_without_pycocotools(self):
        """When pycocotools is unavailable the fallback decoder must work."""
        binary = np.zeros((20, 20), dtype=bool)
        binary[5:10, 5:10] = True
        flat = binary.reshape(-1, order="F")
        counts = []
        val = False
        run = 0
        for v in flat:
            if v == val:
                run += 1
            else:
                counts.append(run)
                run = 1
                val = v
        counts.append(run)
        rle = {"counts": counts, "size": [20, 20]}

        with patch.dict(sys.modules, {"pycocotools": None, "pycocotools.mask": None}):
            result = _normalize_mask(rle, self.IMAGE_SHAPE)
        np.testing.assert_array_equal(result, binary)

    def test_polygon_mask_single_poly(self):
        # Square [2,2,12,2,12,12,2,12]
        poly = [[2.0, 2.0, 12.0, 2.0, 12.0, 12.0, 2.0, 12.0]]
        result = _normalize_mask(poly, self.IMAGE_SHAPE)
        assert result.dtype == bool
        assert result[7, 7]  # centre is inside
        assert not result[0, 0]  # corner is outside


# ===========================================================================
# _polygon_to_binary
# ===========================================================================


class TestPolygonToBinary:
    def test_square_polygon(self):
        poly = [[2.0, 2.0, 8.0, 2.0, 8.0, 8.0, 2.0, 8.0]]
        result = _polygon_to_binary(poly, 10, 10)
        assert result.dtype == bool
        assert result[5, 5]  # interior
        assert not result[0, 0]  # exterior

    def test_empty_polygon_list(self):
        result = _polygon_to_binary([], 10, 10)
        assert result.shape == (10, 10)
        assert not result.any()

    def test_output_shape(self):
        poly = [[0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0]]
        result = _polygon_to_binary(poly, 8, 12)
        assert result.shape == (8, 12)


# ===========================================================================
# mask_iou
# ===========================================================================


class TestMaskIou:
    IMAGE_SHAPE = (20, 20)

    # ---- basic correctness ------------------------------------------------

    def test_identical_binary_masks_iou_one(self):
        m = _binary_mask(20, 20, 2, 2, 12, 12)
        result = mask_iou([m], [m], self.IMAGE_SHAPE)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_non_overlapping_binary_masks_zero(self):
        m1 = _binary_mask(20, 20, 0, 0, 5, 5)
        m2 = _binary_mask(20, 20, 10, 10, 15, 15)
        result = mask_iou([m1], [m2], self.IMAGE_SHAPE)
        assert result[0, 0] == pytest.approx(0.0)

    def test_partial_overlap_binary(self):
        # m1 covers columns 0-9, m2 covers columns 5-14 (same rows 0-19)
        m1 = _binary_mask(20, 20, 0, 0, 10, 20)  # 200 px
        m2 = _binary_mask(20, 20, 5, 0, 15, 20)  # 200 px, inter=100
        result = mask_iou([m1], [m2], self.IMAGE_SHAPE)
        expected = 100.0 / 300.0  # inter / union
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_nested_masks_iou(self):
        outer = _binary_mask(20, 20, 0, 0, 20, 20)  # 400 px
        inner = _binary_mask(20, 20, 5, 5, 15, 15)  # 100 px
        result = mask_iou([inner], [outer], self.IMAGE_SHAPE)
        assert result[0, 0] == pytest.approx(100.0 / 400.0)

    # ---- rle masks --------------------------------------------------------

    def test_rle_mask_same_as_binary_mask(self):
        pytest.importorskip("pycocotools")
        binary = _binary_mask(20, 20, 3, 3, 13, 13)
        rle = _rle_encode(binary)
        result_rle = mask_iou([rle], [rle], self.IMAGE_SHAPE)
        result_bin = mask_iou([binary], [binary], self.IMAGE_SHAPE)
        assert result_rle[0, 0] == pytest.approx(result_bin[0, 0], abs=1e-4)

    def test_rle_vs_binary_cross_format(self):
        """RLE mask1 vs binary mask2 gives same IoU as binary vs binary."""
        pytest.importorskip("pycocotools")
        m1 = _binary_mask(20, 20, 0, 0, 10, 10)
        m2 = _binary_mask(20, 20, 5, 5, 15, 15)
        rle1 = _rle_encode(m1)
        result_cross = mask_iou([rle1], [m2], self.IMAGE_SHAPE)
        result_bin = mask_iou([m1], [m2], self.IMAGE_SHAPE)
        assert result_cross[0, 0] == pytest.approx(result_bin[0, 0], abs=1e-4)

    # ---- polygon masks ----------------------------------------------------

    def test_polygon_mask_rasterizes_correctly(self):
        # Full 20×20 square polygon → same as all-True binary mask
        poly = [[0.0, 0.0, 20.0, 0.0, 20.0, 20.0, 0.0, 20.0]]
        full = np.ones((20, 20), dtype=bool)
        result = mask_iou([poly], [full], self.IMAGE_SHAPE)
        assert result[0, 0] == pytest.approx(1.0, abs=0.01)

    # ---- empty inputs -----------------------------------------------------

    def test_empty_masks1_returns_empty_matrix(self):
        m = _binary_mask(20, 20, 0, 0, 10, 10)
        result = mask_iou([], [m], self.IMAGE_SHAPE)
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    def test_empty_masks2_returns_empty_matrix(self):
        m = _binary_mask(20, 20, 0, 0, 10, 10)
        result = mask_iou([m], [], self.IMAGE_SHAPE)
        assert result.shape == (1, 0)
        assert result.dtype == np.float32

    def test_both_empty_returns_empty_matrix(self):
        result = mask_iou([], [], self.IMAGE_SHAPE)
        assert result.shape == (0, 0)

    # ---- output properties ------------------------------------------------

    def test_output_dtype_float32(self):
        m = _binary_mask(20, 20, 0, 0, 10, 10)
        result = mask_iou([m], [m], self.IMAGE_SHAPE)
        assert result.dtype == np.float32

    def test_output_shape_n_m(self):
        masks_a = [_binary_mask(20, 20, i, i, i + 5, i + 5) for i in range(3)]
        masks_b = [_binary_mask(20, 20, i, i, i + 5, i + 5) for i in range(4)]
        result = mask_iou(masks_a, masks_b, self.IMAGE_SHAPE)
        assert result.shape == (3, 4)

    def test_iou_values_in_0_1(self):
        m1 = _binary_mask(20, 20, 0, 0, 10, 10)
        m2 = _binary_mask(20, 20, 5, 5, 15, 15)
        result = mask_iou([m1], [m2], self.IMAGE_SHAPE)
        assert 0.0 <= result[0, 0] <= 1.0

    # ---- fallback without pycocotools -------------------------------------

    def test_fallback_without_pycocotools(self):
        """mask_iou must work correctly even when pycocotools is absent."""
        m1 = _binary_mask(20, 20, 0, 0, 10, 10)
        m2 = _binary_mask(20, 20, 5, 5, 15, 15)
        expected = mask_iou([m1], [m2], self.IMAGE_SHAPE)  # compute with any available backend

        with patch.dict(sys.modules, {"pycocotools": None, "pycocotools.mask": None}):
            # Re-import with mocked pycocotools
            import importlib

            import mata.eval.metrics.iou as iou_module

            importlib.reload(iou_module)
            result = iou_module.mask_iou([m1], [m2], self.IMAGE_SHAPE)

        assert result[0, 0] == pytest.approx(expected[0, 0], abs=1e-4)
