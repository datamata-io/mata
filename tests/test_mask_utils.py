"""Tests for polygon mask conversion utilities."""

import numpy as np
import pytest

from mata.core.mask_utils import binary_mask_to_polygon, compute_polygon_area, polygon_to_bbox, polygon_to_binary_mask

# Skip all tests if OpenCV not available
cv2 = pytest.importorskip("cv2")


def test_binary_mask_to_polygon_rectangle():
    """Test converting rectangle binary mask to polygon."""
    # Create 10x10 mask with 4x4 rectangle
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True

    polygons = binary_mask_to_polygon(mask, tolerance=1.0, min_area=5)

    # Should get one polygon
    assert len(polygons) == 1

    # Polygon should have at least 4 points (8 coordinates)
    assert len(polygons[0]) >= 8

    # Verify it's a closed polygon
    assert len(polygons[0]) % 2 == 0


def test_binary_mask_to_polygon_multiple_regions():
    """Test converting mask with multiple disconnected regions."""
    mask = np.zeros((20, 20), dtype=bool)
    # Two separate rectangles
    mask[2:6, 2:6] = True
    mask[12:16, 12:16] = True

    polygons = binary_mask_to_polygon(mask, tolerance=1.0, min_area=5)

    # Should get two polygons
    assert len(polygons) == 2


def test_polygon_to_binary_mask():
    """Test converting polygon to binary mask."""
    # Square polygon: (10, 10) -> (50, 10) -> (50, 50) -> (10, 50)
    polygon = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]]

    mask = polygon_to_binary_mask(polygon, height=100, width=100)

    # Should be 100x100
    assert mask.shape == (100, 100)
    assert mask.dtype == bool

    # Should have filled region
    assert mask.sum() > 0

    # Center should be filled
    assert mask[30, 30]

    # Outside should be empty
    assert not mask[5, 5]


def test_compute_polygon_area():
    """Test polygon area computation."""
    # 10x10 square
    polygon = [0, 0, 10, 0, 10, 10, 0, 10]
    area = compute_polygon_area(polygon)

    # Should be 100 square pixels
    assert abs(area - 100.0) < 1.0


def test_polygon_to_bbox():
    """Test bounding box computation from polygon."""
    polygon = [10, 20, 50, 20, 50, 60, 10, 60]
    bbox = polygon_to_bbox(polygon)

    assert bbox == (10.0, 20.0, 50.0, 60.0)


def test_roundtrip_conversion():
    """Test binary -> polygon -> binary roundtrip."""
    # Create simple rectangle mask
    original_mask = np.zeros((50, 50), dtype=bool)
    original_mask[10:40, 10:40] = True

    # Convert to polygon
    polygons = binary_mask_to_polygon(original_mask, tolerance=1.0)

    # Convert back to mask
    reconstructed_mask = polygon_to_binary_mask(polygons, height=50, width=50)

    # Should be very similar (some edge pixels may differ due to polygon approximation)
    similarity = (original_mask == reconstructed_mask).sum() / original_mask.size
    assert similarity > 0.95  # At least 95% similar


def test_polygon_min_area_filter():
    """Test that small regions are filtered out."""
    mask = np.zeros((100, 100), dtype=bool)
    # Large region
    mask[10:40, 10:40] = True
    # Tiny region (only 2x2 pixels = 4 pixels)
    mask[80:82, 80:82] = True

    # Filter out regions < 10 pixels
    polygons = binary_mask_to_polygon(mask, tolerance=1.0, min_area=10)

    # Should only get the large region
    assert len(polygons) == 1


def test_polygon_tolerance_affects_complexity():
    """Test that tolerance affects polygon complexity."""
    # Create circle-like shape
    from math import cos, pi, sin

    mask = np.zeros((100, 100), dtype=bool)
    for angle in np.linspace(0, 2 * pi, 100):
        x = int(50 + 30 * cos(angle))
        y = int(50 + 30 * sin(angle))
        mask[y - 2 : y + 2, x - 2 : x + 2] = True

    # Low tolerance = more points
    polygons_precise = binary_mask_to_polygon(mask, tolerance=0.5)

    # High tolerance = fewer points
    polygons_simple = binary_mask_to_polygon(mask, tolerance=5.0)

    if polygons_precise and polygons_simple:
        # Simpler polygon should have fewer points
        assert len(polygons_simple[0]) <= len(polygons_precise[0])
