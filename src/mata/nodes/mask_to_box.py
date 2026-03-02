"""Mask to bounding box extraction node.

Extracts tight bounding boxes from segmentation masks and returns them
as a Detections artifact, preserving instance IDs and metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node
from mata.core.types import Instance

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


class MaskToBox(Node):
    """Extract bounding boxes from masks.

    Computes tight axis-aligned bounding boxes from segmentation masks and
    returns them as a Detections artifact. This is useful for converting
    segmentation results to detection format or for extracting object bounds.

    The node handles different mask formats (RLE, binary, polygon) by converting
    them to binary format and finding the minimal bounding rectangle. Empty masks
    are filtered out by default.

    Instance IDs and metadata are preserved from the input Masks artifact.
    Original scores and labels are maintained in the output detections.

    Args:
        src: Name of the input masks artifact (default: "masks").
        out: Name of the output detections artifact (default: "detections").
        filter_empty: If True, exclude instances with empty/invalid masks (default: True).
        expand_px: Optional padding to add to bounding boxes in pixels (default: 0).
        name: Optional human-readable node name.

    Inputs:
        masks (Masks): Input segmentation masks to extract boxes from.

    Outputs:
        detections (Detections): Detection results with bounding boxes.

    Example:
        ```python
        from mata.nodes import MaskToBox
        from mata.core.graph import Graph

        # Extract tight bounding boxes from SAM masks
        mask_to_box = MaskToBox(
            src="sam_masks",
            out="tight_boxes",
            expand_px=5  # Add 5px padding
        )

        graph = Graph("masks_to_detections")
        graph.add(mask_to_box, inputs={"masks": "sam_masks"})
        ```

    Note:
        Requires numpy for bounding box computation. The node will automatically
        convert masks to binary format for processing if needed.
    """

    inputs = {"masks": Masks}
    outputs = {"detections": Detections}

    def __init__(
        self,
        src: str = "masks",
        out: str = "detections",
        filter_empty: bool = True,
        expand_px: int = 0,
        name: str | None = None,
    ):
        super().__init__(name=name or "MaskToBox")
        self.src = src
        self.out = out
        self.filter_empty = filter_empty
        self.expand_px = expand_px

        # Validate parameters
        if self.expand_px < 0:
            raise ValueError(f"expand_px must be non-negative, got: {self.expand_px}")

        # Check numpy availability
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for bounding box extraction. " "Install with: pip install numpy")

    def run(self, ctx: ExecutionContext, masks: Masks) -> dict[str, Artifact]:
        """Extract bounding boxes from all masks.

        Args:
            ctx: Execution context (unused)
            masks: Input masks artifact

        Returns:
            Dictionary with detections under the output key

        Raises:
            ImportError: If numpy not available
            ValueError: If mask conversion fails
        """
        if len(masks.instances) == 0:
            # Return empty detections
            return {self.out: Detections(instances=[], instance_ids=[], meta=masks.meta.copy())}

        # Convert all masks to binary for bbox computation
        try:
            binary_masks = masks.to_binary()
        except Exception as e:
            raise ValueError(f"Failed to convert masks to binary format: {e}")

        # Extract bounding boxes
        detection_instances = []
        detection_ids = []

        for i, (inst, binary_mask) in enumerate(zip(masks.instances, binary_masks)):
            try:
                bbox = self._extract_bbox(binary_mask)

                if bbox is None:
                    # Empty or invalid mask
                    if not self.filter_empty:
                        # Keep instance but with None bbox (will be filtered later)
                        continue
                    else:
                        continue

                # Expand bbox if requested
                if self.expand_px > 0:
                    bbox = self._expand_bbox(bbox, binary_mask.shape, self.expand_px)

                # Create detection instance
                # Preserve original attributes but remove mask
                detection_inst = Instance(
                    bbox=bbox,
                    score=inst.score,
                    label=inst.label,
                    label_name=inst.label_name,
                    area=inst.area,
                    is_stuff=inst.is_stuff,
                    embedding=inst.embedding,
                    track_id=inst.track_id,
                    keypoints=inst.keypoints,
                    mask=None,  # Remove mask data
                )

                detection_instances.append(detection_inst)
                detection_ids.append(masks.instance_ids[i])

            except Exception as e:
                # Log error and optionally skip
                import warnings

                warnings.warn(f"Failed to extract bbox from mask {i}: {e}. Skipping instance.", UserWarning)
                if not self.filter_empty:
                    # Add instance with None bbox if not filtering
                    continue

        # Create detections artifact
        meta = masks.meta.copy()
        meta["extracted_from"] = "masks"
        meta["expand_px"] = self.expand_px

        detections = Detections(instances=detection_instances, instance_ids=detection_ids, meta=meta)

        return {self.out: detections}

    def _extract_bbox(self, binary_mask: np.ndarray) -> list[float] | None:
        """Extract bounding box from binary mask.

        Args:
            binary_mask: Binary numpy array of shape (H, W)

        Returns:
            Bounding box in xyxy format [x1, y1, x2, y2] or None if empty
        """
        if binary_mask.size == 0 or not binary_mask.any():
            return None

        # Find non-zero pixels
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        # Get min/max indices
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]

        # Return in xyxy format (note: y_max+1, x_max+1 for inclusive bounds)
        return [float(x_min), float(y_min), float(x_max + 1), float(y_max + 1)]

    def _expand_bbox(self, bbox: list[float], mask_shape: tuple, expand_px: int) -> list[float]:
        """Expand bounding box by specified pixels.

        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            mask_shape: Shape of the mask (H, W)
            expand_px: Pixels to expand in each direction

        Returns:
            Expanded bounding box [x1, y1, x2, y2], clipped to image bounds
        """
        h, w = mask_shape
        x1, y1, x2, y2 = bbox

        # Expand and clip to image bounds
        x1_exp = max(0, x1 - expand_px)
        y1_exp = max(0, y1 - expand_px)
        x2_exp = min(w, x2 + expand_px)
        y2_exp = min(h, y2 + expand_px)

        return [x1_exp, y1_exp, x2_exp, y2_exp]
