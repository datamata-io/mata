"""ExtractROIs node for data transformation.

Crops image regions from detections using bounding boxes, producing an
ROIs artifact that maps back to source detection instance IDs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image as PILImage

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.rois import ROIs
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class ExtractROIs(Node):
    """Crop image regions from detections using bounding boxes.

    For every instance in the input Detections that has a bounding box,
    this node extracts the corresponding image crop from the source image.
    The resulting ROIs artifact preserves the instance IDs so that downstream
    nodes can correlate crops back to their source detections.

    An optional *padding* parameter can expand the crop region by a fixed
    number of pixels (clamped to image bounds).

    Args:
        src_image: Name of the image artifact (default: "image").
        src_dets: Name of the detections artifact (default: "dets").
        out: Name of the output ROIs artifact key (default: "rois").
        padding: Number of pixels to expand each side of the box (default: 0).
        name: Optional human-readable node name.

    Inputs:
        image (Image): Source image to crop from.
        detections (Detections): Detections with bounding boxes.

    Outputs:
        rois (ROIs): Cropped image regions with source mapping.

    Example:
        ```python
        from mata.nodes import ExtractROIs

        node = ExtractROIs(src_image="image", src_dets="dets", out="rois", padding=5)
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image, "detections": Detections}
    outputs: dict[str, type[Artifact]] = {"rois": ROIs}

    def __init__(
        self,
        src_image: str = "image",
        src_dets: str = "dets",
        out: str = "rois",
        padding: int = 0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")
        self.src_image = src_image
        self.src_dets = src_dets
        self.out = out
        self.padding = padding

    def run(
        self,
        ctx: ExecutionContext,
        image: Image,
        detections: Detections,
    ) -> dict[str, Artifact]:
        """Crop image regions from detections.

        Instances without a bounding box are silently skipped.

        Args:
            ctx: Execution context.
            image: Source image to crop from.
            detections: Detections with bounding boxes.

        Returns:
            Dict with a single key (``self.out``) mapping to the ROIs artifact.
            Returns an empty ROIs artifact if there are no valid bounding boxes.
        """
        pil_image = image.to_pil()
        img_w, img_h = pil_image.size

        roi_images: list[PILImage.Image] = []
        instance_ids: list[str] = []
        source_boxes: list[tuple[int, int, int, int]] = []

        for inst, inst_id in zip(detections.instances, detections.instance_ids):
            if inst.bbox is None:
                continue

            x1, y1, x2, y2 = inst.bbox

            # Apply padding (clamped to image bounds)
            x1 = max(0, int(x1) - self.padding)
            y1 = max(0, int(y1) - self.padding)
            x2 = min(img_w, int(x2) + self.padding)
            y2 = min(img_h, int(y2) + self.padding)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_image.crop((x1, y1, x2, y2))
            roi_images.append(crop)
            instance_ids.append(inst_id)
            source_boxes.append((x1, y1, x2, y2))

        # Handle empty result gracefully
        if not roi_images:
            ctx.record_metric(self.name, "num_rois", 0)
            # Return minimal valid ROIs (empty lists, but matching lengths)
            return {
                self.out: ROIs(
                    roi_images=[],
                    instance_ids=[],
                    source_boxes=[],
                    meta={
                        "source_width": img_w,
                        "source_height": img_h,
                        "padding": self.padding,
                    },
                )
            }

        rois = ROIs(
            roi_images=roi_images,
            instance_ids=instance_ids,
            source_boxes=source_boxes,
            meta={
                "source_width": img_w,
                "source_height": img_h,
                "padding": self.padding,
            },
        )

        ctx.record_metric(self.name, "num_rois", len(roi_images))

        return {self.out: rois}

    def __repr__(self) -> str:
        return f"ExtractROIs(name='{self.name}', " f"padding={self.padding})"
