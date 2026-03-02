"""ExpandBoxes node for data transformation.

Recomputes detection bounding boxes from segmentation masks so that
boxes tightly enclose the mask region, and returns an updated
Detections artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node
from mata.core.types import Instance

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


def _bbox_from_binary_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute tight xyxy bounding box from a binary mask.

    Args:
        mask: 2-D boolean/uint8 numpy array of shape (H, W).

    Returns:
        (x1, y1, x2, y2) tuple in absolute pixels, or *None* if the mask
        is entirely empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # xyxy format (x2/y2 are inclusive pixel indices → +1 for exclusive bound)
    return (float(x1), float(y1), float(x2 + 1), float(y2 + 1))


def _mask_to_binary(mask: Any) -> np.ndarray | None:
    """Best-effort conversion of an Instance mask to a binary numpy array.

    Supports:
    - numpy ndarray (binary or uint8)
    - RLE dict (requires pycocotools)
    - polygon list (requires mata.core.mask_utils)

    Returns:
        2-D boolean ndarray or *None* if conversion is not possible.
    """
    if mask is None:
        return None

    if isinstance(mask, np.ndarray):
        return mask.astype(bool)

    if isinstance(mask, dict):
        # RLE format
        try:
            from pycocotools import mask as mask_utils_coco

            decoded = mask_utils_coco.decode(mask)
            return decoded.astype(bool)
        except ImportError:
            return None

    if isinstance(mask, list):
        # Polygon format — need image size context which we don't always have
        try:
            from mata.core.mask_utils import polygon_to_binary_mask

            # Infer size from polygon extents (rough)
            xs = mask[0::2]
            ys = mask[1::2]
            h = int(max(ys)) + 1
            w = int(max(xs)) + 1
            return polygon_to_binary_mask(mask, h, w).astype(bool)
        except (ImportError, Exception):
            return None

    return None


class ExpandBoxes(Node):
    """Recompute detection bounding boxes from segmentation masks.

    For each instance, the node converts the associated mask to binary,
    computes the tightest enclosing bounding box, and returns a new
    Detections artifact with updated boxes.  Instances that have no mask
    or whose mask cannot be decoded retain their original bounding box.

    Instance IDs are preserved so that downstream nodes can continue to
    correlate the results.

    Args:
        src_dets: Name of the detections artifact (default: "dets").
        src_masks: Name of the masks artifact (default: "masks").
        out: Name of the output key (default: "expanded").
        name: Optional human-readable node name.

    Inputs:
        detections (Detections): Detections whose boxes should be updated.
        masks (Masks): Segmentation masks aligned to the detections by
            instance ID.

    Outputs:
        detections (Detections): Detections with updated bounding boxes.

    Example:
        ```python
        from mata.nodes import ExpandBoxes

        node = ExpandBoxes(src_dets="dets", src_masks="masks", out="tight_dets")
        ```
    """

    inputs: dict[str, type[Artifact]] = {"detections": Detections, "masks": Masks}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        src_dets: str = "dets",
        src_masks: str = "masks",
        out: str = "expanded",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.src_dets = src_dets
        self.src_masks = src_masks
        self.out = out

    def run(
        self,
        ctx: ExecutionContext,
        detections: Detections,
        masks: Masks,
    ) -> dict[str, Artifact]:
        """Recompute bounding boxes from masks.

        Mask-to-detection alignment is performed by matching ``instance_ids``.
        If a detection has no corresponding mask, its original bbox is kept.

        Args:
            ctx: Execution context.
            detections: Source detections.
            masks: Segmentation masks.

        Returns:
            Dict with a single key (``self.out``) mapping to the updated
            Detections artifact.
        """
        # Build a lookup: instance_id → mask Instance
        mask_lookup: dict[str, Instance] = {}
        for mask_inst, mask_id in zip(masks.instances, masks.instance_ids):
            mask_lookup[mask_id] = mask_inst

        updated_instances: list[Instance] = []
        updated_count = 0

        for inst, inst_id in zip(detections.instances, detections.instance_ids):
            mask_inst = mask_lookup.get(inst_id)
            if mask_inst is not None and mask_inst.mask is not None:
                binary = _mask_to_binary(mask_inst.mask)
                if binary is not None:
                    new_bbox = _bbox_from_binary_mask(binary)
                    if new_bbox is not None:
                        # Build updated Instance with new bbox (and attach mask)
                        updated = Instance(
                            bbox=new_bbox,
                            mask=inst.mask,  # keep original mask on detection
                            score=inst.score,
                            label=inst.label,
                            label_name=inst.label_name,
                            area=inst.area,
                            is_stuff=inst.is_stuff,
                            embedding=inst.embedding,
                            track_id=inst.track_id,
                            keypoints=inst.keypoints,
                        )
                        updated_instances.append(updated)
                        updated_count += 1
                        continue

            # Fallback — keep original instance
            updated_instances.append(inst)

        result = Detections(
            instances=updated_instances,
            instance_ids=list(detections.instance_ids),
            entities=list(detections.entities),
            entity_ids=list(detections.entity_ids),
            meta=detections.meta.copy(),
        )

        ctx.record_metric(self.name, "total_instances", len(detections.instances))
        ctx.record_metric(self.name, "updated_boxes", updated_count)

        return {self.out: result}

    def __repr__(self) -> str:
        return f"ExpandBoxes(name='{self.name}')"
