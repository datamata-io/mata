"""NMS node — non-maximum suppression analysis node.

Provides native non-maximum suppression filtering using torchvision.ops.nms
for removing redundant overlapping detections in graph workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class NMS(Node):
    """Non-maximum suppression on detections.

    Filters overlapping detections using IoU-based non-maximum suppression.
    Uses torchvision.ops.nms for efficient native implementation.
    Preserves instance order and metadata from original detections.

    Args:
        iou_threshold: IoU threshold for suppression (0.0-1.0). Default: 0.5.
        out: Key under which the output artifact is stored. Default: "nms_dets".
        detections_src: Key for input detections artifact. Default: "detections".

    Inputs:
        detections (Detections): Detection results to filter.

    Outputs:
        detections (Detections): Filtered detection results.

    Example:
        ```python
        from mata.nodes import NMS

        # Standard NMS filtering
        node = NMS(iou_threshold=0.5, out="filtered")
        result = node.run(ctx, detections=dets)
        filtered = result["filtered"]

        # Aggressive filtering
        node = NMS(iou_threshold=0.3, out="clean_dets")

        # In a graph pipeline
        Graph("detection_pipeline")
            .then(Detect(using="detector", out="dets"))
            .then(NMS(iou_threshold=0.5, out="filtered"))
            .then(Annotate(show_boxes=True, out="viz"))
        ```
    """

    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, iou_threshold: float = 0.5, out: str = "nms_dets", detections_src: str = "detections"):
        """Initialize NMS node with filtering settings.

        Args:
            iou_threshold: IoU threshold for suppression (0.0-1.0)
            out: Output artifact key
            detections_src: Input detections artifact key
        """
        super().__init__(name="NMS")
        self.iou_threshold = iou_threshold
        self.out = out
        self.detections_src = detections_src

        # Validate IoU threshold
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"Invalid iou_threshold '{iou_threshold}'. Must be between 0.0 and 1.0.")

    def run(self, ctx: ExecutionContext, detections: Detections) -> dict[str, Artifact]:
        """Execute NMS filtering on input detections.

        Args:
            ctx: Execution context
            detections: Detection results to filter

        Returns:
            Dictionary with filtered detections artifact
        """
        # Handle empty detections
        if len(detections.instances) == 0:
            return {self.out: detections}

        try:
            import torch
            from torchvision.ops import nms
        except ImportError as e:
            raise ImportError(
                "PyTorch and torchvision required for NMS node. " "Install with: pip install torch torchvision"
            ) from e

        # Extract boxes and scores from instances
        boxes = []
        scores = []

        for inst in detections.instances:
            if inst.bbox is not None:
                # Ensure bbox is in XYXY format (should already be from MATA)
                boxes.append(inst.bbox)
                scores.append(inst.score if inst.score is not None else 1.0)
            else:
                # Skip instances without bounding boxes
                continue

        # Handle case where no instances have valid bboxes
        if len(boxes) == 0:
            # Return empty detections but preserve metadata
            return {
                self.out: Detections(
                    instances=[],
                    instance_ids=[],
                    entities=detections.entities,
                    entity_ids=detections.entity_ids,
                    meta={
                        **detections.meta,
                        "nms_applied": True,
                        "nms_iou_threshold": self.iou_threshold,
                        "pre_nms_count": len(detections.instances),
                        "post_nms_count": 0,
                    },
                )
            }

        # Convert to tensors (XYXY format expected by torchvision.ops.nms)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Apply NMS
        keep_indices = nms(boxes_tensor, scores_tensor, self.iou_threshold)

        # Filter instances based on kept indices
        filtered_instances = [detections.instances[i] for i in keep_indices.tolist()]
        filtered_instance_ids = [detections.instance_ids[i] for i in keep_indices.tolist()]

        # Create new filtered detections
        filtered_detections = Detections(
            instances=filtered_instances,
            instance_ids=filtered_instance_ids,
            entities=detections.entities,  # Preserve entities (from VLM workflows)
            entity_ids=detections.entity_ids,
            meta={
                **detections.meta,
                "nms_applied": True,
                "nms_iou_threshold": self.iou_threshold,
                "pre_nms_count": len(detections.instances),
                "post_nms_count": len(filtered_instances),
            },
        )

        return {self.out: filtered_detections}
