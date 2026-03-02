"""Utilities for mask format conversions.

This module provides functions to convert between different mask formats:
- RLE (Run-Length Encoding) from pycocotools
- Binary numpy arrays (H, W) boolean masks
- Polygon coordinates [x1, y1, x2, y2, ...] in COCO format
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    from pycocotools import mask as mask_utils_coco

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    mask_utils_coco = None  # type: ignore


def binary_mask_to_polygon(mask: np.ndarray, tolerance: float = 2.0, min_area: int = 10) -> list[list[float]]:
    """Convert binary mask to polygon coordinates.

    Uses OpenCV contour detection to extract polygon boundaries from binary masks.
    Returns multiple polygons if the mask contains disconnected regions.

    Args:
        mask: Binary numpy array of shape (H, W) with dtype bool or uint8
        tolerance: Polygon approximation tolerance (epsilon parameter for cv2.approxPolyDP)
            - Lower values: More precise polygons with more points
            - Higher values: Simpler polygons with fewer points
            - Recommended range: 1.0-3.0
        min_area: Minimum contour area in pixels to include
            - Filters out small noise regions
            - Recommended: 10-100 depending on image size

    Returns:
        List of polygons, where each polygon is [x1, y1, x2, y2, ...] in COCO format
        Empty list if no valid contours found

    Raises:
        ImportError: If OpenCV is not installed
        ValueError: If mask is not 2D or has invalid dtype

    Examples:
        >>> mask = np.array([[0, 0, 1], [0, 1, 1]], dtype=bool)
        >>> polygons = binary_mask_to_polygon(mask)
        >>> # Returns: [[2.0, 0.0, 2.0, 1.0, 1.0, 1.0, ...]]
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for polygon conversion. " "Install with: pip install opencv-python")

    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for polygon conversion")

    # Validate input
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D array, got shape {mask.shape}")

    # Convert to uint8 if needed
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    elif mask.dtype == np.uint8:
        mask_uint8 = mask.copy()
    else:
        raise ValueError(f"Mask dtype must be bool or uint8, got {mask.dtype}")

    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,  # Only external contours (ignore holes)
        cv2.CHAIN_APPROX_SIMPLE,  # Compress horizontal/vertical/diagonal segments
    )

    polygons = []
    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Approximate polygon to reduce points
        epsilon = tolerance
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Convert to COCO format: [x1, y1, x2, y2, ...]
        # contour shape: (N, 1, 2) -> flatten to (N*2,)
        polygon = approx.flatten().tolist()

        # Need at least 3 points (6 coordinates) for a valid polygon
        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def rle_to_polygon(rle: dict[str, Any], tolerance: float = 2.0, min_area: int = 10) -> list[list[float]]:
    """Convert RLE mask to polygon coordinates.

    First decodes RLE to binary mask, then extracts polygons.

    Args:
        rle: RLE dict with 'size' and 'counts' from pycocotools
        tolerance: Polygon approximation tolerance (see binary_mask_to_polygon)
        min_area: Minimum contour area in pixels

    Returns:
        List of polygons in COCO format

    Raises:
        ImportError: If pycocotools or OpenCV not installed

    Examples:
        >>> from pycocotools import mask as mask_utils
        >>> rle = mask_utils.encode(np.asfortranarray(binary_mask))
        >>> polygons = rle_to_polygon(rle)
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools is required for RLE conversion. " "Install with: pip install pycocotools")

    # Decode RLE to binary mask
    binary_mask = mask_utils_coco.decode(rle)

    # Convert binary to polygon
    return binary_mask_to_polygon(binary_mask, tolerance=tolerance, min_area=min_area)


def polygon_to_binary_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Convert polygon coordinates to binary mask.

    Args:
        polygons: List of polygons, each as [x1, y1, x2, y2, ...] in COCO format
        height: Output mask height
        width: Output mask width

    Returns:
        Binary numpy array of shape (height, width) with dtype bool

    Raises:
        ImportError: If OpenCV or numpy not installed

    Examples:
        >>> polygons = [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]]
        >>> mask = polygon_to_binary_mask(polygons, 100, 100)
        >>> # Returns 100x100 boolean array with filled rectangle
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for polygon to mask conversion")

    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for polygon to mask conversion")

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill each polygon
    for polygon in polygons:
        if len(polygon) < 6:  # Need at least 3 points
            warnings.warn(f"Polygon has only {len(polygon)//2} points, skipping (need >= 3)", UserWarning)
            continue

        # Reshape [x1, y1, x2, y2, ...] to [(x1, y1), (x2, y2), ...]
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [points], color=255)  # type: ignore[call-overload]

    return mask.astype(bool)


def compute_polygon_area(polygon: list[float]) -> float:
    """Compute area of a polygon using Shoelace formula.

    Args:
        polygon: Polygon coordinates [x1, y1, x2, y2, ...] in COCO format

    Returns:
        Area in pixels (float)

    Examples:
        >>> # Square from (0,0) to (10,10)
        >>> area = compute_polygon_area([0, 0, 10, 0, 10, 10, 0, 10])
        >>> # Returns: 100.0
    """
    if len(polygon) < 6:
        return 0.0

    # Reshape to (N, 2) array of (x, y) points
    points = np.array(polygon).reshape(-1, 2)

    # Shoelace formula: A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = points[:, 0]
    y = points[:, 1]

    area = 0.5 * abs(np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])

    return float(area)


def polygon_to_bbox(polygon: list[float]) -> tuple[float, float, float, float]:
    """Compute bounding box from polygon coordinates.

    Args:
        polygon: Polygon coordinates [x1, y1, x2, y2, ...] in COCO format

    Returns:
        Bounding box (x1, y1, x2, y2) in xyxy format

    Examples:
        >>> bbox = polygon_to_bbox([10, 20, 50, 20, 50, 60, 10, 60])
        >>> # Returns: (10.0, 20.0, 50.0, 60.0)
    """
    if len(polygon) < 2:
        return (0.0, 0.0, 0.0, 0.0)

    # Extract x and y coordinates
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)

    return (float(x1), float(y1), float(x2), float(y2))
