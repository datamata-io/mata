"""IoU computation utilities for MATA evaluation.

Implements box IoU (vectorised numpy) and mask IoU (multi-format).
These are the only functions that require numpy; no PyTorch or
pycocotools dependency is introduced here.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard COCO IoU sweep: 0.50 – 0.95 in steps of 0.05
COCO_IOU_THRESHOLDS: list[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Box IoU
# ---------------------------------------------------------------------------


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy bounding boxes.

    Args:
        boxes1: (N, 4) float array, xyxy format.
        boxes2: (M, 4) float array, xyxy format.

    Returns:
        (N, M) float32 IoU matrix.  Zero-area or non-overlapping boxes
        return 0.0.  Identical boxes return 1.0.
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    # Intersection
    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])  # (N, M)
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = (inter_w * inter_h).astype(np.float32)

    # Areas
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])).astype(np.float32)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])).astype(np.float32)
    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area

    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union_area > 0.0, inter_area / union_area, 0.0).astype(np.float32)
    return iou


def box_iou_batch(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresholds: list[float],
) -> np.ndarray:
    """Boolean match matrix for each IoU threshold.

    Args:
        pred_boxes: (N, 4) xyxy predicted boxes.
        gt_boxes:   (M, 4) xyxy ground-truth boxes.
        iou_thresholds: list of T float thresholds.

    Returns:
        (T, N, M) bool array — ``True`` where IoU ≥ threshold.
    """
    iou = box_iou(pred_boxes, gt_boxes)  # (N, M)
    thresholds = np.array(iou_thresholds, dtype=np.float32)[:, np.newaxis, np.newaxis]
    return iou[np.newaxis, :, :] >= thresholds  # (T, N, M)


# ---------------------------------------------------------------------------
# Mask IoU
# ---------------------------------------------------------------------------

MaskInput = dict | np.ndarray | list


def _normalize_mask(
    mask: MaskInput,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert any MATA mask format to a binary (H, W) bool array.

    Supported input formats:
    * **RLE dict** — ``{"counts": ..., "size": [H, W]}`` (pycocotools RLE)
    * **Binary ndarray** — shape ``(H, W)``, any dtype (truthy → True)
    * **Polygon list** — ``list[list[float]]`` in COCO format (x, y, x, y, …)

    Args:
        mask: Input mask in one of the three supported MATA formats.
        image_shape: ``(height, width)`` of the source image.

    Returns:
        Boolean (H, W) numpy array.

    Raises:
        NotImplementedError: If the mask format cannot be recognised.
    """
    h, w = image_shape

    # --- RLE dict ---
    if isinstance(mask, dict) and "counts" in mask:
        try:
            from pycocotools import mask as coco_mask_api  # type: ignore

            binary = coco_mask_api.decode(mask).astype(bool)
        except ImportError:
            # Fallback: manual uncompressed RLE decode
            binary = _decode_rle_fallback(mask, h, w)
        return binary

    # --- Binary ndarray ---
    if isinstance(mask, np.ndarray):
        if mask.shape != (h, w):
            raise ValueError(f"Mask array shape {mask.shape} does not match image_shape {image_shape}")
        return mask.astype(bool)

    # --- Polygon list (COCO flat format) ---
    if isinstance(mask, list):
        return _polygon_to_binary(mask, h, w)

    raise NotImplementedError(
        f"Unrecognised mask type: {type(mask).__name__}. " "Expected RLE dict, binary ndarray, or polygon list."
    )


def _decode_rle_fallback(rle: dict, h: int, w: int) -> np.ndarray:
    """Decode an uncompressed COCO RLE dict without pycocotools."""
    counts = rle["counts"]
    binary = np.zeros(h * w, dtype=bool)
    idx = 0
    val = False
    for c in counts:
        binary[idx : idx + c] = val
        idx += c
        val = not val
    return binary.reshape(h, w, order="F")


def _polygon_to_binary(polygons: list, h: int, w: int) -> np.ndarray:
    """Rasterise a list of polygon coordinate arrays into a binary mask."""
    try:
        import cv2  # type: ignore

        canvas = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(canvas, [pts.astype(np.int32)], 1)  # type: ignore[call-overload]
        return canvas.astype(bool)
    except ImportError:
        pass

    # PIL fallback
    from PIL import Image, ImageDraw  # type: ignore

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        draw.polygon([tuple(p) for p in pts], fill=1)
    return np.array(img, dtype=bool)


def mask_iou(
    masks1: list[MaskInput],
    masks2: list[MaskInput],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pairwise IoU between two lists of masks (any MATA format).

    Args:
        masks1: N masks (RLE dict, binary ndarray, or polygon list).
        masks2: M masks.
        image_shape: ``(height, width)`` of the source image.

    Returns:
        (N, M) float32 IoU matrix.
    """
    if len(masks1) == 0 or len(masks2) == 0:
        return np.zeros((len(masks1), len(masks2)), dtype=np.float32)

    # Try pycocotools fast path for RLE inputs
    if isinstance(masks1[0], dict) and isinstance(masks2[0], dict):
        try:
            from pycocotools import mask as coco_mask_api  # type: ignore

            iou_flat = coco_mask_api.iou(masks1, masks2, [0] * len(masks2))
            return np.asarray(iou_flat, dtype=np.float32)
        except ImportError:
            pass

    # General path: normalise all masks to binary arrays then compute IoU
    h, w = image_shape
    bin1 = np.stack([_normalize_mask(m, image_shape).reshape(-1) for m in masks1], axis=0).astype(
        np.float32
    )  # (N, H*W)
    bin2 = np.stack([_normalize_mask(m, image_shape).reshape(-1) for m in masks2], axis=0).astype(
        np.float32
    )  # (M, H*W)

    inter = bin1 @ bin2.T  # (N, M)
    area1 = bin1.sum(axis=1, keepdims=True)  # (N, 1)
    area2 = bin2.sum(axis=1, keepdims=True).T  # (1, M)
    union = area1 + area2 - inter
    iou = np.where(union > 0.0, inter / union, 0.0).astype(np.float32)
    return iou
