"""Tests for tile geometry and merge helpers."""

import numpy as np
import pytest

from mata.core.tiling import Tile, generate_tiles, merge_instances
from mata.core.types import Instance


def test_tiles_cover_a_large_image_without_exceeding_bounds():
    tiles = generate_tiles(2560, 1440, slice_size=(640, 640), overlap=0.2)
    assert len(tiles) == 15  # 5 columns x 3 rows
    assert {t.x1 for t in tiles} == {0, 512, 1024, 1536, 1920}
    assert {t.y1 for t in tiles} == {0, 512, 800}
    for tile in tiles:
        assert 0 <= tile.x1 < tile.x2 <= 2560
        assert 0 <= tile.y1 < tile.y2 <= 1440
    assert min(t.x1 for t in tiles) == 0
    assert max(t.x2 for t in tiles) == 2560
    assert min(t.y1 for t in tiles) == 0
    assert max(t.y2 for t in tiles) == 1440


def test_zero_overlap_uses_slice_stride():
    tiles = generate_tiles(2560, 1440, slice_size=640, overlap=0.0)
    assert len(tiles) == 12  # 4 columns x 3 rows
    assert {t.x1 for t in tiles} == {0, 640, 1280, 1920}
    assert {t.y1 for t in tiles} == {0, 640, 800}


def test_fractional_overlap_truncates_the_overlap_pixels():
    # 0.3 * 512 = 153.6; the stride is 512 - 153 = 359, not round(358.4).
    tiles = generate_tiles(1363, 1000, slice_size=(512, 512), overlap=0.3)
    assert {t.x1 for t in tiles} == {0, 359, 718, 851}


def test_small_image_is_one_tile():
    tiles = generate_tiles(500, 300, slice_size=640, overlap=0.2)
    assert tiles == [Tile(0, 0, 500, 300)]


def test_invalid_tile_parameters_raise():
    with pytest.raises(ValueError):
        generate_tiles(0, 100)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, slice_size=0)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, overlap=1.0)


def test_tile_offsets_a_local_box():
    tile = Tile(100, 200, 300, 400)
    assert tile.offset_box((5, 10, 50, 60)) == (105, 210, 150, 260)


def test_merge_drops_duplicate_same_class():
    instances = [
        Instance(bbox=(0, 0, 10, 10), score=0.9, label=0),
        Instance(bbox=(1, 1, 11, 11), score=0.8, label=0),
    ]
    merged = merge_instances(instances, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_keeps_same_box_with_different_classes():
    instances = [
        Instance(bbox=(0, 0, 10, 10), score=0.9, label=0),
        Instance(bbox=(0, 0, 10, 10), score=0.8, label=1),
    ]
    merged = merge_instances(instances, iou_threshold=0.5)
    assert len(merged) == 2


def test_merge_keeps_non_overlapping_boxes_and_caps():
    instances = [Instance(bbox=(i * 20, 0, i * 20 + 10, 10), score=0.5, label=0) for i in range(5)]
    assert len(merge_instances(instances, iou_threshold=0.5)) == 5
    assert len(merge_instances(instances, iou_threshold=0.5, max_det=2)) == 2


def test_merge_skips_instances_without_bbox():
    mask_only = Instance(bbox=None, mask=np.zeros((4, 4), dtype=bool), score=0.9, label=0)
    merged = merge_instances([mask_only], iou_threshold=0.5)
    assert merged == []
