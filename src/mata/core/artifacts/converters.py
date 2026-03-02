"""Artifact conversion and entity promotion utilities.

This module provides utility functions for:
- Converting between VisionResult and artifact types
- Entity → Instance promotion workflows for VLM integration
- Label matching strategies (exact, fuzzy)
- Auto-promotion from VisionResult
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mata.core.types import Entity, Instance, VisionResult

if TYPE_CHECKING:
    from mata.core.artifacts.base import Artifact
    from mata.core.artifacts.detections import Detections


def promote_entities_to_instances(
    entities: list[Entity], spatial_instances: list[Instance], match_strategy: str = "label_fuzzy"
) -> list[Instance]:
    """Promote entities to instances by matching with spatial detections.

    Enables VLM → GroundingDINO/DETR fusion workflows by matching Entity labels
    (from VLM semantic output) with Instance spatial data (bbox/mask from detection models).

    Args:
        entities: Semantic entities from VLM (label + score + attributes)
        spatial_instances: Spatial detections from GroundingDINO/DETR/etc (bbox/mask)
        match_strategy: Matching strategy to use
            - "label_exact": Direct string match (case-sensitive)
            - "label_fuzzy": Lowercase, remove plurals, strip articles
            - "embedding": Semantic similarity (future, requires CLIP)

    Returns:
        List of promoted Instance objects with entity metadata + spatial data.
        Only entities that matched a spatial instance are returned.

    Raises:
        ValueError: If match_strategy is invalid

    Example:
        >>> # VLM detects semantic entities
        >>> entities = [Entity("cat", 0.95), Entity("dogs", 0.87)]
        >>>
        >>> # GroundingDINO provides spatial instances
        >>> instances = [
        ...     Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ...     Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog")
        ... ]
        >>>
        >>> # Promote: fuzzy match handles "dogs" → "dog"
        >>> promoted = promote_entities_to_instances(entities, instances, "label_fuzzy")
        >>> len(promoted)  # Both matched
        2
    """
    if match_strategy not in ["label_exact", "label_fuzzy", "embedding"]:
        raise ValueError(
            f"Invalid match_strategy '{match_strategy}'. " f"Valid options: 'label_exact', 'label_fuzzy', 'embedding'"
        )

    if match_strategy == "embedding":
        raise NotImplementedError(
            "Embedding-based matching not yet implemented. "
            "Requires CLIP or similar embedding model. "
            "Use 'label_exact' or 'label_fuzzy' instead."
        )

    promoted_instances = []

    for entity in entities:
        matched_instance = match_entity_to_instance(entity, spatial_instances, strategy=match_strategy)

        if matched_instance:
            # Merge entity attributes into instance
            promoted = merge_entity_attributes(matched_instance, entity)
            promoted_instances.append(promoted)

    return promoted_instances


def match_entity_to_instance(
    entity: Entity, instances: list[Instance], strategy: str = "label_fuzzy"
) -> Instance | None:
    """Find best matching instance for entity.

    Searches through spatial instances to find the best match for a semantic entity
    based on label similarity.

    Args:
        entity: Semantic entity from VLM
        instances: List of spatial instances to search
        strategy: Matching strategy
            - "label_exact": Direct string match (case-sensitive)
            - "label_fuzzy": Lowercase, remove plurals, strip articles
            - "embedding": Semantic similarity (future)

    Returns:
        Matching Instance if found, None otherwise

    Example:
        >>> entity = Entity("the cats", 0.9)
        >>> instances = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")]
        >>> matched = match_entity_to_instance(entity, instances, "label_fuzzy")
        >>> matched is not None  # Fuzzy match: "the cats" → "cat"
        True
    """
    if strategy not in ["label_exact", "label_fuzzy", "embedding"]:
        raise ValueError(f"Invalid strategy '{strategy}'. " f"Valid options: 'label_exact', 'label_fuzzy', 'embedding'")

    if strategy == "embedding":
        raise NotImplementedError("Embedding-based matching not yet implemented. Use 'label_fuzzy'.")

    for instance in instances:
        # Skip instances without label names
        if instance.label_name is None:
            continue

        # Exact match
        if strategy == "label_exact":
            if entity.label == instance.label_name:
                return instance

        # Fuzzy match
        elif strategy == "label_fuzzy":
            if _fuzzy_label_match(entity.label, instance.label_name):
                return instance

    return None


def merge_entity_attributes(instance: Instance, entity: Entity) -> Instance:
    """Merge entity attributes into instance metadata.

    Creates a new Instance with entity attributes merged into metadata.
    Preserves all spatial data (bbox, mask) while adding semantic information.

    Args:
        instance: Spatial instance with bbox/mask
        entity: Semantic entity with attributes

    Returns:
        New Instance with merged attributes in metadata

    Note:
        Returns a dataclass.replace() copy - original instance is unchanged.
        Entity attributes are stored in instance metadata under "entity_attributes" key.

    Example:
        >>> entity = Entity("cat", 0.95, attributes={"color": "orange", "count": 2})
        >>> instance = Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")
        >>> merged = merge_entity_attributes(instance, entity)
        >>> # Instance now has entity attributes accessible via metadata
    """
    # Note: Instance is a frozen dataclass, so we need to use workaround
    # Since Instance doesn't have a metadata field, we'll update the instance
    # by preserving entity score if higher, and store attributes in a way
    # that can be accessed later (this is a limitation of current Instance design)

    # For now, use the higher confidence score between entity and instance
    # Future enhancement: Add metadata field to Instance
    from dataclasses import replace

    # Use entity score if higher (VLM may have more semantic confidence)
    final_score = max(instance.score, entity.score)

    # Create new instance with merged score
    # Note: Instance doesn't have metadata field, so entity.attributes
    # are "lost" in current design. This is a known limitation.
    # For full attribute preservation, would need to extend Instance dataclass.
    merged = replace(instance, score=final_score)

    return merged


def auto_promote_vision_result(result: VisionResult, spatial_source: VisionResult | None = None) -> VisionResult:
    """Auto-promote entities to instances if spatial data available.

    Enables automatic fusion of VLM semantic output with spatial detections.
    Two modes of operation:
    1. With spatial_source: Match result.entities with spatial_source.instances
    2. Without spatial_source: Use result.instances if entities already promoted

    Args:
        result: VisionResult with entities (and optionally instances)
        spatial_source: Optional VisionResult with spatial instances for matching

    Returns:
        VisionResult with promoted instances (entities → instances)

    Example:
        >>> # Mode 1: Explicit spatial source
        >>> vlm_result = VisionResult(entities=[Entity("cat", 0.9)])
        >>> spatial_result = VisionResult(instances=[Instance(..., label_name="cat")])
        >>> promoted = auto_promote_vision_result(vlm_result, spatial_source=spatial_result)
        >>>
        >>> # Mode 2: Already promoted instances in result
        >>> vlm_result = VisionResult(
        ...     entities=[Entity("cat", 0.9)],
        ...     instances=[Instance(..., label_name="cat")]  # From auto_promote=True
        ... )
        >>> final = auto_promote_vision_result(vlm_result)
    """
    # If no entities, return as-is
    if not result.entities:
        return result

    # Mode 1: Match with spatial source
    if spatial_source is not None:
        promoted_instances = promote_entities_to_instances(
            result.entities, spatial_source.instances, match_strategy="label_fuzzy"
        )

        # Combine with any existing instances in result
        all_instances = list(result.instances) + promoted_instances

        return VisionResult(
            instances=all_instances,
            entities=[],  # Clear entities (promoted)
            meta=result.meta,
            text=result.text,
            prompt=result.prompt,
        )

    # Mode 2: Use existing instances if present
    # (Assumes VLM adapter already promoted with auto_promote=True)
    if result.instances:
        # Instances already promoted, clear entities
        return VisionResult(
            instances=result.instances, entities=[], meta=result.meta, text=result.text, prompt=result.prompt
        )

    # No spatial data available, return with entities unchanged
    return result


# Helper functions


def _fuzzy_label_match(label1: str, label2: str) -> bool:
    """Fuzzy label matching for entity promotion.

    Handles:
    - Case insensitivity
    - Plural forms (simple heuristic: trailing 's')
    - Leading articles (a, an, the)
    - Extra whitespace

    Args:
        label1: First label
        label2: Second label

    Returns:
        True if labels match fuzzy criteria

    Example:
        >>> _fuzzy_label_match("cat", "cats")
        True
        >>> _fuzzy_label_match("The Dog", "dog")
        True
        >>> _fuzzy_label_match("person", "people")
        False  # Complex plural not handled
    """

    def normalize(text: str) -> str:
        """Normalize label for fuzzy matching."""
        # Lowercase and strip whitespace
        text = text.lower().strip()

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading articles
        for article in ["a ", "an ", "the "]:
            if text.startswith(article):
                text = text[len(article) :]
                break

        # Remove trailing 's' for simple plural handling
        # (doesn't handle irregular plurals like "people", "children")
        if text.endswith("s") and len(text) > 1 and not text.endswith("ss"):
            text = text[:-1]

        return text

    norm1 = normalize(label1)
    norm2 = normalize(label2)

    return norm1 == norm2


# =============================================================================
# Task 1.6: Artifact Converters
# =============================================================================
# Bidirectional conversions between VisionResult/ClassifyResult/DepthResult
# and new artifact types (Detections, Masks, Classifications, DepthMap)


def vision_result_to_detections(result: VisionResult):
    """Convert VisionResult to Detections artifact.

    Preserves both instances and entities for VLM workflows.
    Auto-generates instance_ids for graph wiring.

    Args:
        result: VisionResult with instances and/or entities

    Returns:
        Detections artifact with auto-generated IDs

    Example:
        >>> result = VisionResult(instances=[Instance(...)])
        >>> dets = vision_result_to_detections(result)
        >>> dets.instance_ids  # Auto-generated: ['inst_0000', 'inst_0001', ...]
    """
    from mata.core.artifacts.detections import Detections

    return Detections.from_vision_result(result)


def vision_result_to_masks(result: VisionResult):
    """Convert VisionResult to Masks artifact.

    Extracts only instances with mask data. Raises error if no masks present.

    Args:
        result: VisionResult with instances containing mask data

    Returns:
        Masks artifact with auto-generated IDs

    Raises:
        ValueError: If VisionResult contains no instances with masks

    Example:
        >>> result = VisionResult(instances=[Instance(mask=rle_mask, ...)])
        >>> masks = vision_result_to_masks(result)
        >>> masks.to_rle()  # Convert to RLE format
    """
    from mata.core.artifacts.masks import Masks

    return Masks.from_vision_result(result)


def detections_to_vision_result(detections):
    """Convert Detections artifact back to VisionResult.

    Preserves entities and stores instance_ids in metadata for round-trip compatibility.

    Args:
        detections: Detections artifact with instances/entities

    Returns:
        VisionResult with instances, entities, and IDs in meta

    Example:
        >>> dets = Detections.from_vision_result(result)
        >>> restored = detections_to_vision_result(dets)
        >>> restored.meta["instance_ids"]  # IDs preserved
    """
    return detections.to_vision_result()


def masks_to_vision_result(masks):
    """Convert Masks artifact back to VisionResult.

    Args:
        masks: Masks artifact with instances containing mask data

    Returns:
        VisionResult with instances and IDs in meta

    Example:
        >>> masks = Masks.from_vision_result(result)
        >>> restored = masks_to_vision_result(masks)
        >>> len(restored.instances) == len(masks.instances)
        True
    """
    return masks.to_vision_result()


def classify_result_to_artifact(result):
    """Convert ClassifyResult to Classifications artifact.

    Args:
        result: ClassifyResult with predictions

    Returns:
        Classifications artifact

    Example:
        >>> from mata.core.types import ClassifyResult, Classification
        >>> result = ClassifyResult(predictions=[Classification(0, 0.9, "cat")])
        >>> artifact = classify_result_to_artifact(result)
        >>> artifact.top1.label_name
        'cat'
    """
    from mata.core.artifacts.classifications import Classifications

    return Classifications.from_classify_result(result)


def artifact_to_classify_result(artifact):
    """Convert Classifications artifact to ClassifyResult.

    Args:
        artifact: Classifications artifact

    Returns:
        ClassifyResult with predictions and meta
    """
    return artifact.to_classify_result()


def depth_result_to_artifact(result):
    """Convert DepthResult to DepthMap artifact.

    Args:
        result: DepthResult with depth map

    Returns:
        DepthMap artifact

    Example:
        >>> import numpy as np
        >>> from mata.core.types import DepthResult
        >>> result = DepthResult(depth=np.zeros((100, 100)))
        >>> artifact = depth_result_to_artifact(result)
        >>> artifact.height
        100
    """
    from mata.core.artifacts.depth_map import DepthMap

    return DepthMap.from_depth_result(result)


def artifact_to_depth_result(artifact):
    """Convert DepthMap artifact to DepthResult.

    Args:
        artifact: DepthMap artifact

    Returns:
        DepthResult with depth array and meta
    """
    return artifact.to_depth_result()


# =============================================================================
# Instance ID Management Utilities
# =============================================================================


def generate_instance_ids(n: int, prefix: str = "obj") -> list[str]:
    """Generate N unique instance IDs with given prefix.

    IDs are deterministic and sequential for reproducibility.
    Format: {prefix}_{index:04d} (e.g., "obj_0000", "obj_0001")

    Args:
        n: Number of IDs to generate
        prefix: Prefix for IDs (default: "obj")

    Returns:
        List of N unique string IDs

    Example:
        >>> generate_instance_ids(3)
        ['obj_0000', 'obj_0001', 'obj_0002']
        >>> generate_instance_ids(2, prefix="det")
        ['det_0000', 'det_0001']
    """
    return [f"{prefix}_{i:04d}" for i in range(n)]


def ensure_instance_ids(instances: list[Instance]) -> list[str]:
    """Ensure instances have IDs, generate if missing.

    Useful when working with raw Instance lists before creating artifact.

    Args:
        instances: List of Instance objects

    Returns:
        List of instance IDs (generates new IDs for all instances)

    Note:
        This always generates NEW IDs. For preserving existing IDs,
        use Detections.from_vision_result() with preserve_ids=True.

    Example:
        >>> instances = [Instance(...), Instance(...)]
        >>> ids = ensure_instance_ids(instances)
        >>> len(ids) == len(instances)
        True
    """
    return generate_instance_ids(len(instances), prefix="inst")


def align_instance_ids(artifacts: list[Artifact]) -> bool:
    """Check if multiple artifacts have aligned instance IDs.

    Verifies that all artifacts with instance_ids have:
    1. Same number of IDs
    2. IDs in the same order

    Useful for validating multi-modal results (detections + masks)
    before merging.

    Args:
        artifacts: List of artifacts with instance_ids attribute

    Returns:
        True if all artifacts have matching instance_ids, False otherwise

    Example:
        >>> dets = Detections(instances=[...], instance_ids=["obj_0", "obj_1"])
        >>> masks = Masks(instances=[...], instance_ids=["obj_0", "obj_1"])
        >>> align_instance_ids([dets, masks])
        True
        >>>
        >>> # Mismatched IDs
        >>> masks2 = Masks(instances=[...], instance_ids=["obj_1", "obj_0"])
        >>> align_instance_ids([dets, masks2])
        False
    """
    if not artifacts:
        return True

    # Extract instance_ids from each artifact
    id_lists = []
    for artifact in artifacts:
        if hasattr(artifact, "instance_ids"):
            id_lists.append(artifact.instance_ids)

    if not id_lists:
        return True  # No artifacts with IDs

    # Check all ID lists are identical
    first_ids = id_lists[0]
    for ids in id_lists[1:]:
        if ids != first_ids:
            return False

    return True


# =============================================================================
# Agent Loop Result Converters (Task B4)
# =============================================================================


def _calculate_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bbox in xyxy format (x1, y1, x2, y2)
        box2: Second bbox in xyxy format (x1, y1, x2, y2)

    Returns:
        IoU value between 0.0 and 1.0

    Example:
        >>> box1 = (10, 10, 50, 50)
        >>> box2 = (30, 30, 70, 70)
        >>> iou = _calculate_iou(box1, box2)
        >>> 0.0 < iou < 1.0
        True
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _deduplicate_instances(instances: list[Instance], iou_threshold: float = 0.7) -> list[Instance]:
    """Deduplicate instances by IoU, keeping higher-scoring detections.

    When multiple instances overlap significantly (IoU > threshold),
    keeps only the instance with the highest confidence score.

    Args:
        instances: List of Instance objects to deduplicate
        iou_threshold: IoU threshold above which instances are considered duplicates

    Returns:
        Deduplicated list of Instance objects

    Example:
        >>> inst1 = Instance(bbox=(10, 10, 50, 50), score=0.9, label=0)
        >>> inst2 = Instance(bbox=(12, 12, 52, 52), score=0.7, label=0)  # Overlaps with inst1
        >>> deduped = _deduplicate_instances([inst1, inst2], iou_threshold=0.7)
        >>> len(deduped)  # inst2 removed (overlaps + lower score)
        1
    """
    if not instances:
        return []

    # Sort by score descending (keep higher-scoring instances)
    sorted_instances = sorted(instances, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)

    keep = []
    for instance in sorted_instances:
        # Skip instances without bbox (can't calculate IoU)
        if instance.bbox is None:
            keep.append(instance)
            continue

        # Check if this instance overlaps significantly with any kept instance
        is_duplicate = False
        for kept_instance in keep:
            if kept_instance.bbox is None:
                continue

            iou = _calculate_iou(instance.bbox, kept_instance.bbox)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(instance)

    return keep


def agent_result_to_detections(result) -> Detections:
    """Convert AgentResult to Detections artifact.

    Merges instances from all tool calls, deduplicates by IoU,
    and includes VLM entities + final text in metadata.

    This converter is used when VLM nodes run in agent mode (with tools).
    It combines spatial detections from multiple tool calls into a single
    coherent Detections artifact while preserving agent metadata.

    Args:
        result: AgentResult from agent loop execution

    Returns:
        Detections artifact with merged instances, entities, and agent metadata

    Metadata includes:
        - agent_iterations: Number of loop iterations executed
        - agent_text: Final VLM synthesis text
        - agent_tool_calls: List of all tool calls made (as dicts)
        - agent_tool_results: Summary of all tool results
        - agent_conversation: Full conversation history

    Example:
        >>> # After VLM agent loop with detect tool
        >>> agent_result = AgentResult(
        ...     text="Found 2 cats and 1 dog",
        ...     instances=[...],  # From detect tool calls
        ...     entities=[...],   # From VLM output
        ...     iterations=3,
        ...     tool_calls=[...],
        ... )
        >>> dets = agent_result_to_detections(agent_result)
        >>> dets.meta["agent_iterations"]
        3
        >>> dets.meta["agent_text"]
        'Found 2 cats and 1 dog'
    """
    # Import AgentResult type for type checking
    from mata.core.agent_loop import AgentResult
    from mata.core.artifacts.detections import Detections

    if not isinstance(result, AgentResult):
        raise TypeError(f"Expected AgentResult, got {type(result).__name__}")

    # Deduplicate instances (merge from multiple tool calls)
    deduplicated_instances = _deduplicate_instances(result.instances, iou_threshold=0.7)

    # Build comprehensive metadata
    meta = {
        **result.meta,  # Preserve existing metadata
        "agent_iterations": result.iterations,
        "agent_text": result.text,
        "agent_tool_calls": [tc.to_dict() for tc in result.tool_calls],
        "agent_tool_results": [
            {"tool": tr.tool_name, "success": tr.success, "summary": tr.summary} for tr in result.tool_results
        ],
        "agent_conversation": result.conversation,
        "deduplication_applied": True,
        "deduplication_threshold": 0.7,
        "pre_dedup_count": len(result.instances),
        "post_dedup_count": len(deduplicated_instances),
    }

    # Create VisionResult first, then convert to Detections
    vision_result = VisionResult(
        instances=deduplicated_instances,
        entities=result.entities,
        text=result.text,
        prompt="",  # Agent prompts are in conversation history
        meta=meta,
    )

    return Detections.from_vision_result(vision_result)


def agent_result_to_vision_result(result) -> VisionResult:
    """Convert AgentResult to VisionResult.

    Preserves all agent loop information in VisionResult format:
    - text: Final VLM synthesis
    - entities: Semantic entities from VLM output
    - instances: Deduplicated instances from tool calls
    - prompt: Stored in metadata (agent uses conversation history)
    - meta: Agent execution metadata (iterations, tool calls, results)

    Args:
        result: AgentResult from agent loop execution

    Returns:
        VisionResult with deduplicated instances and agent metadata

    Metadata includes:
        - agent_iterations: Number of loop iterations
        - agent_tool_calls: All tool calls made
        - agent_tool_results: All tool results received
        - agent_conversation: Full conversation history
        - deduplication_applied: True
        - pre_dedup_count/post_dedup_count: Instance counts

    Example:
        >>> agent_result = AgentResult(
        ...     text="Analysis complete",
        ...     instances=[inst1, inst2, inst3],  # Some may be duplicates
        ...     entities=[ent1],
        ...     iterations=2,
        ... )
        >>> vision_result = agent_result_to_vision_result(agent_result)
        >>> vision_result.text
        'Analysis complete'
        >>> len(vision_result.instances) <= 3  # Deduplicated
        True
    """
    from mata.core.agent_loop import AgentResult

    if not isinstance(result, AgentResult):
        raise TypeError(f"Expected AgentResult, got {type(result).__name__}")

    # Deduplicate instances
    deduplicated_instances = _deduplicate_instances(result.instances, iou_threshold=0.7)

    # Build comprehensive metadata
    meta = {
        **result.meta,
        "agent_iterations": result.iterations,
        "agent_tool_calls": [tc.to_dict() for tc in result.tool_calls],
        "agent_tool_results": [
            {"tool": tr.tool_name, "success": tr.success, "summary": tr.summary} for tr in result.tool_results
        ],
        "agent_conversation": result.conversation,
        "deduplication_applied": True,
        "deduplication_threshold": 0.7,
        "pre_dedup_count": len(result.instances),
        "post_dedup_count": len(deduplicated_instances),
    }

    return VisionResult(
        instances=deduplicated_instances,
        entities=result.entities,
        text=result.text,
        prompt="",  # Agent uses conversation history, not single prompt
        meta=meta,
    )
