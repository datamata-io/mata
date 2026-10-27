"""Tile geometry and detection merging for sliced inference.

Sliced inference runs a detector over overlapping crops and maps the results
back to source-image coordinates. This module holds the pure geometry and
merge helpers so adapters, nodes and tests share one implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from mata.core.types import Instance


@dataclass(frozen=True)
class Tile:
    """A crop rectangle in source-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def offset_box(self, bbox: Sequence[float]) -> tuple[float, float, float, float]:
        """Translate a box from tile-local to source-image coordinates."""
        return (
            float(bbox[0]) + self.x1,
            float(bbox[1]) + self.y1,
            float(bbox[2]) + self.x1,
            float(bbox[3]) + self.y1,
        )


def _axis_positions(total: int, slice_len: int, overlap: float) -> list[int]:
    """Start offsets along one axis, shifted inward at the far edge."""
    if total <= slice_len:
        return [0]
    step = max(1, slice_len - int(slice_len * overlap))
    last = total - slice_len
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tiles(
    width: int,
    height: int,
    slice_size: tuple[int, int] | int = (640, 640),
    overlap: float = 0.2,
) -> list[Tile]:
    """Generate overlapping tile rectangles covering an image.

    Args:
        width: Source image width in pixels.
        height: Source image height in pixels.
        slice_size: ``(width, height)`` of a tile, or a single int for square tiles.
        overlap: Fractional overlap between neighbouring tiles, in ``[0, 1)``.

    Returns:
        Tiles in row-major order. Edge tiles are shifted inward so they stay
        inside the image; no padding is added.

    Raises:
        ValueError: If dimensions or parameters are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if isinstance(slice_size, int):
        slice_w = slice_h = slice_size
    else:
        slice_w, slice_h = slice_size
    if slice_w <= 0 or slice_h <= 0:
        raise ValueError(f"slice_size must be positive, got {slice_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    xs = _axis_positions(width, slice_w, overlap)
    ys = _axis_positions(height, slice_h, overlap)
    tiles = [
        Tile(x1=int(x), y1=int(y), x2=int(min(x + slice_w, width)), y2=int(min(y + slice_h, height)))
        for y in ys
        for x in xs
    ]
    return [t for t in tiles if t.width > 0 and t.height > 0]


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one ``xyxy`` box against an ``(N, 4)`` array."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def merge_instances(
    instances: Iterable[Instance],
    iou_threshold: float = 0.5,
    max_det: int | None = None,
    mode: str = "nms",
) -> list[Instance]:
    """Class-aware duplicate handling for detections collected from overlapping tiles.

    Args:
        instances: Detections already mapped to source-image coordinates.
        iou_threshold: IoU above which two same-class boxes are considered duplicates.
        max_det: Optional cap on the number of returned instances.
        mode: ``"nms"`` (default) suppresses the lower-scoring box; ``"nmm"``
            merges duplicates into a union box with the higher score.

    Returns:
        Kept instances sorted by descending score.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    if max_det is not None and max_det <= 0:
        raise ValueError(f"max_det must be positive or None, got {max_det}")
    if mode not in ("nms", "nmm"):
        raise ValueError(f"mode must be 'nms' or 'nmm', got {mode!r}")

    candidates = [inst for inst in instances if inst.bbox is not None]
    if not candidates:
        return []

    kept: list[Instance] = []
    for label in sorted({int(inst.label) for inst in candidates}):
        group = [inst for inst in candidates if int(inst.label) == label]
        group.sort(key=lambda inst: float(inst.score), reverse=True)
        if mode == "nms":
            kept.extend(_nms_group(group, iou_threshold))
        else:
            kept.extend(_nmm_group(group, iou_threshold))

    kept.sort(key=lambda inst: float(inst.score), reverse=True)
    return kept[:max_det] if max_det is not None else kept


def _nms_group(group: list[Instance], iou_threshold: float) -> list[Instance]:
    """Greedy NMS within one class, highest score first."""
    boxes = np.array([inst.bbox for inst in group], dtype=np.float32)
    remaining = list(range(len(group)))
    kept: list[Instance] = []
    while remaining:
        best = remaining[0]
        kept.append(group[best])
        if len(remaining) == 1:
            break
        rest = np.array(remaining[1:], dtype=np.int64)
        ious = _iou_one_to_many(boxes[best], boxes[rest])
        remaining = [int(rest[i]) for i in np.flatnonzero(ious <= iou_threshold)]
    return kept


def _nmm_group(group: list[Instance], iou_threshold: float) -> list[Instance]:
    """Greedy non-maximum merging within one class, highest score first.

    Each kept box absorbs the lower-scoring boxes whose IoU with it exceeds the
    threshold, and the result carries the highest score and the union box. The
    match set is computed once per kept box (vectorized), matching the greedy
    reference behaviour rather than growing the union transitively.
    """
    remaining = list(group)
    kept: list[Instance] = []
    while remaining:
        keep = remaining.pop(0)
        assert keep.bbox is not None
        box = np.array(keep.bbox, dtype=np.float32)
        if remaining:
            boxes = np.array([inst.bbox for inst in remaining], dtype=np.float32)
            ious = _iou_one_to_many(box, boxes)
            matched = ious > iou_threshold
            if matched.any():
                hits = boxes[matched]
                union = np.concatenate([box.reshape(1, 4), hits], axis=0)
                box = np.array(
                    [
                        union[:, 0].min(axis=0),
                        union[:, 1].min(axis=0),
                        union[:, 2].max(axis=0),
                        union[:, 3].max(axis=0),
                    ],
                    dtype=np.float32,
                )
                remaining = [inst for inst, hit in zip(remaining, matched) if not hit]
        kept.append(replace(keep, bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3]))))
    return kept
