"""Merge node — merge multiple artifacts by instance_id into VisionResult.

Aligns multiple artifact types (detections, masks, keypoints) by instance_id
and merges them into unified Instance objects within a VisionResult.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.keypoints import Keypoints
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node
from mata.core.types import Instance, VisionResult

if TYPE_CHECKING:
    import numpy as np

    from mata.core.graph.context import ExecutionContext


class Merge(Node):
    """Merge multiple artifacts by instance_id into VisionResult.

    The Merge node combines artifacts from different task nodes (detection,
    segmentation, pose estimation) by aligning them through instance_id matching.
    Each detection instance can be enhanced with masks, keypoints, and other
    modal data to create comprehensive multi-modal instances.

    The base artifacts must contain detections (bounding boxes). Optional
    artifacts (masks, keypoints) are merged in when available, matched by
    instance_id. This enables complex workflows like:
    - Detection + SAM segmentation
    - Detection + pose estimation
    - Detection + segmentation + pose estimation

    Args:
        dets: Detections artifact name in context (default: "dets")
        masks: Optional masks artifact name (default: "masks")
        keypoints: Optional keypoints artifact name (default: "keypoints")
        out: Output VisionResult name (default: "merged")

    Examples:
        >>> # Detection + segmentation merge
        >>> merge_node = Merge(
        ...     dets="filtered_dets",
        ...     masks="sam_masks",
        ...     out="detection_with_masks"
        ... )

        >>> # Detection + pose estimation
        >>> merge_node = Merge(
        ...     dets="person_dets",
        ...     keypoints="pose_kpts",
        ...     out="person_with_pose"
        ... )

        >>> # Full multi-modal merge
        >>> merge_node = Merge(
        ...     dets="dets",
        ...     masks="masks",
        ...     keypoints="kpts",
        ...     out="complete_instances"
        ... )

        >>> # Use in graph
        >>> from mata.core.graph.graph import Graph
        >>> graph = (Graph("detect_segment_pose")
        ...     .then(Detect(using="detr", out="dets"))
        ...     .then(PromptBoxes(using="sam", dets="dets", out="masks"))
        ...     .then(EstimatePose(using="hrnet", dets="dets", out="poses"))
        ...     .then(Merge(dets="dets", masks="masks", keypoints="poses")))
    """

    inputs = {
        "detections": Detections,
        "masks": Masks | None,  # type: ignore[dict-item]
        "keypoints": Keypoints | None,  # type: ignore[dict-item]
    }
    outputs = {"vision_result": VisionResult}  # type: ignore[dict-item]

    def __init__(self, dets: str = "dets", masks: str = "masks", keypoints: str = "keypoints", out: str = "merged"):
        """Initialize Merge node.

        Args:
            dets: Artifact name for base detections (required)
            masks: Artifact name for masks to merge (optional)
            keypoints: Artifact name for keypoints to merge (optional)
            out: Output artifact name for merged VisionResult
        """
        super().__init__(name="Merge")
        self.dets_src = dets
        self.masks_src = masks
        self.keypoints_src = keypoints
        self.output_name = out

    def run(
        self,
        ctx: ExecutionContext,
        detections: Detections,
        masks: Masks | None = None,
        keypoints: Keypoints | None = None,
    ) -> dict[str, Artifact]:
        """Merge artifacts by instance_id alignment.

        Args:
            ctx: Graph execution context for metrics recording
            detections: Base detections with bounding boxes (required)
            masks: Optional segmentation masks to merge
            keypoints: Optional keypoints to merge

        Returns:
            Dictionary mapping output_name to VisionResult with merged instances

        Note:
            - Instances without corresponding masks/keypoints keep original data
            - Only instance_id matches are merged; orphaned artifacts are ignored
            - Original instance data (bbox, score, label) is preserved
            - Meta information is merged from all input artifacts
        """
        start_time = time.time()

        # Build lookups for optional artifacts by instance_id
        mask_lookup: dict[str, Instance] = {}
        if masks is not None:
            for mask_inst, mask_id in zip(masks.instances, masks.instance_ids):
                mask_lookup[mask_id] = mask_inst

        keypoints_lookup: dict[str, np.ndarray] = {}
        if keypoints is not None:
            for kp_array, kp_id in zip(keypoints.keypoints, keypoints.instance_ids):
                keypoints_lookup[kp_id] = kp_array

        # Merge instances by aligning through instance_id
        merged_instances: list[Instance] = []
        merge_stats = {"masks_merged": 0, "keypoints_merged": 0, "total_instances": 0}

        for det_inst, instance_id in zip(detections.instances, detections.instance_ids):
            # Start with base detection instance
            merged_instance = det_inst

            # Merge mask data if available
            if instance_id in mask_lookup:
                mask_instance = mask_lookup[instance_id]
                if mask_instance.mask is not None:
                    merged_instance = replace(merged_instance, mask=mask_instance.mask)
                    merge_stats["masks_merged"] += 1

            # Merge keypoints data if available
            if instance_id in keypoints_lookup:
                kp_array = keypoints_lookup[instance_id]
                merged_instance = replace(merged_instance, keypoints=kp_array)
                merge_stats["keypoints_merged"] += 1

            merged_instances.append(merged_instance)
            merge_stats["total_instances"] += 1

        # Merge metadata from all input artifacts
        merged_meta = detections.meta.copy()
        if masks is not None:
            merged_meta.update({f"masks_{k}": v for k, v in masks.meta.items()})
        if keypoints is not None:
            merged_meta.update({f"keypoints_{k}": v for k, v in keypoints.meta.items()})

        # Create VisionResult with merged instances
        # Preserve entities if present in base detections
        vision_result = VisionResult(
            instances=merged_instances,
            entities=list(detections.entities) if hasattr(detections, "entities") else [],
            meta=merged_meta,
        )

        # Record merge metrics
        merge_time = (time.time() - start_time) * 1000  # Convert to ms
        ctx.record_metric(self.name, "merge_latency_ms", merge_time)
        ctx.record_metric(self.name, "total_instances", merge_stats["total_instances"])
        ctx.record_metric(self.name, "masks_merged", merge_stats["masks_merged"])
        ctx.record_metric(self.name, "keypoints_merged", merge_stats["keypoints_merged"])

        return {self.output_name: vision_result}  # type: ignore[dict-item]

    def _find_instance(self, artifact: Artifact, instance_id: str) -> Instance | None:
        """Find instance in artifact by instance_id.

        Args:
            artifact: Artifact to search (must have instances and instance_ids)
            instance_id: ID to search for

        Returns:
            Matching Instance or None if not found
        """
        if not hasattr(artifact, "instances") or not hasattr(artifact, "instance_ids"):
            return None

        try:
            index = artifact.instance_ids.index(instance_id)
            return artifact.instances[index]
        except (ValueError, IndexError):
            return None

    def __repr__(self) -> str:
        inputs = [f"dets={self.dets_src}"]
        if hasattr(self, "masks_src"):
            inputs.append(f"masks={self.masks_src}")
        if hasattr(self, "keypoints_src"):
            inputs.append(f"keypoints={self.keypoints_src}")
        inputs_str = ", ".join(inputs)
        return f"Merge({inputs_str}, out='{self.output_name}')"
