"""Graph artifact system for MATA.

This module provides the foundational artifact type system for the graph execution framework,
including base classes, type registry, and common artifact types.
"""

from __future__ import annotations

from mata.core.artifacts.base import Artifact, ArtifactTypeRegistry
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.converters import (
    align_instance_ids,
    artifact_to_classify_result,
    artifact_to_depth_result,
    auto_promote_vision_result,
    # ClassifyResult conversions (Task 1.6 - stubs for future)
    classify_result_to_artifact,
    # DepthResult conversions (Task 1.6 - stubs for future)
    depth_result_to_artifact,
    detections_to_vision_result,
    ensure_instance_ids,
    # Instance ID management (Task 1.6)
    generate_instance_ids,
    masks_to_vision_result,
    match_entity_to_instance,
    merge_entity_attributes,
    # Entity promotion utilities (Task 1.7)
    promote_entities_to_instances,
    # VisionResult conversions (Task 1.6)
    vision_result_to_detections,
    vision_result_to_masks,
)
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.keypoints import Keypoints
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.ocr_text import OCRText, TextBlock
from mata.core.artifacts.result import MultiResult
from mata.core.artifacts.rois import ROIs
from mata.core.artifacts.tracks import Track, Tracks

__all__ = [
    "Artifact",
    "ArtifactTypeRegistry",
    "Image",
    "Detections",
    "Masks",
    "Classifications",
    "DepthMap",
    "OCRText",
    "TextBlock",
    "Keypoints",
    "Track",
    "Tracks",
    "ROIs",
    "MultiResult",
    # Entity promotion utilities (Task 1.7)
    "promote_entities_to_instances",
    "match_entity_to_instance",
    "merge_entity_attributes",
    "auto_promote_vision_result",
    # VisionResult conversions (Task 1.6)
    "vision_result_to_detections",
    "vision_result_to_masks",
    "detections_to_vision_result",
    "masks_to_vision_result",
    # ClassifyResult conversions (Task 1.6)
    "classify_result_to_artifact",
    "artifact_to_classify_result",
    # DepthResult conversions (Task 1.6)
    "depth_result_to_artifact",
    "artifact_to_depth_result",
    # Instance ID management (Task 1.6)
    "generate_instance_ids",
    "ensure_instance_ids",
    "align_instance_ids",
]
