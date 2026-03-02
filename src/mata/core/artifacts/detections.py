"""Detections artifact for graph system.

Provides detection results container with support for both spatial detections
(Instance with bbox/mask) and semantic detections (Entity from VLM) for multi-modal workflows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mata.core.artifacts.base import Artifact
from mata.core.types import Entity, Instance, VisionResult


def _generate_id(prefix: str = "obj", index: int | None = None) -> str:
    """Generate a stable instance/entity ID.

    Args:
        prefix: Prefix for the ID (default: "obj")
        index: Optional index to use for deterministic IDs

    Returns:
        String ID in format "prefix_<index>" or "prefix_<uuid>"
    """
    if index is not None:
        return f"{prefix}_{index:04d}"
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _fuzzy_label_match(label1: str, label2: str) -> bool:
    """Fuzzy label matching for entity promotion.

    Handles:
    - Case insensitivity
    - Plural forms (simple heuristic)
    - Leading/trailing articles (a, an, the)

    Args:
        label1: First label
        label2: Second label

    Returns:
        True if labels match fuzzy criteria
    """

    # Normalize: lowercase, strip articles
    def normalize(text: str) -> str:
        text = text.lower().strip()
        # Remove leading articles
        for article in ["a ", "an ", "the "]:
            if text.startswith(article):
                text = text[len(article) :]
        # Remove trailing 's' for simple plural handling
        if text.endswith("s") and len(text) > 1:
            text = text[:-1]
        return text

    norm1 = normalize(label1)
    norm2 = normalize(label2)

    return norm1 == norm2


@dataclass(frozen=True)
class Detections(Artifact):
    """Detection results artifact with instance IDs for graph wiring.

    Supports both spatial detections (Instance with bbox/mask) and
    semantic detections (Entity with label/score only) for VLM workflows.

    This artifact wraps VisionResult to provide:
    - Stable instance/entity IDs for graph wiring
    - Filtering and transformation operations
    - Entity → Instance promotion for VLM fusion workflows
    - Immutable operations (all methods return new instances)

    Attributes:
        instances: List of spatial detections with bbox and/or mask
        instance_ids: Stable string identifiers for instances (auto-generated if missing)
        entities: List of semantic detections from VLM (label/score only)
        entity_ids: Stable string identifiers for entities (auto-generated if missing)
        meta: Optional metadata dictionary

    Examples:
        >>> # Create from VisionResult (detection)
        >>> vision_result = VisionResult(instances=[...])
        >>> dets = Detections.from_vision_result(vision_result)
        >>>
        >>> # Filter detections
        >>> filtered = dets.filter_by_score(0.5)
        >>> top3 = dets.top_k(3)
        >>> cats_only = dets.filter_by_label(["cat"])
        >>>
        >>> # VLM entity promotion workflow
        >>> vlm_result = VisionResult(entities=[Entity("cat", 0.9), Entity("dog", 0.8)])
        >>> vlm_dets = Detections.from_vision_result(vlm_result)
        >>> spatial_dets = Detections.from_vision_result(grounding_result)
        >>> promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_fuzzy")
        >>>
        >>> # Access properties
        >>> boxes = dets.boxes  # (N, 4) xyxy numpy array
        >>> scores = dets.scores  # (N,) numpy array
        >>> labels = dets.labels  # List[str]
    """

    instances: list[Instance] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-generate IDs if missing."""
        # Auto-generate instance_ids if missing
        if len(self.instances) > 0 and len(self.instance_ids) == 0:
            # Use object.__setattr__ since we're frozen
            object.__setattr__(self, "instance_ids", [_generate_id("inst", i) for i in range(len(self.instances))])

        # Auto-generate entity_ids if missing
        if len(self.entities) > 0 and len(self.entity_ids) == 0:
            object.__setattr__(self, "entity_ids", [_generate_id("ent", i) for i in range(len(self.entities))])

        # Validate lengths match
        if len(self.instances) != len(self.instance_ids):
            raise ValueError(
                f"instances and instance_ids length mismatch: " f"{len(self.instances)} vs {len(self.instance_ids)}"
            )

        if len(self.entities) != len(self.entity_ids):
            raise ValueError(
                f"entities and entity_ids length mismatch: " f"{len(self.entities)} vs {len(self.entity_ids)}"
            )

    @classmethod
    def from_vision_result(cls, result: VisionResult, preserve_ids: bool = False) -> Detections:
        """Convert VisionResult to Detections artifact.

        Preserves both instances AND entities from VisionResult for VLM support.

        Args:
            result: VisionResult with instances and/or entities
            preserve_ids: If True, attempt to preserve IDs from meta (advanced usage)

        Returns:
            Detections artifact with auto-generated IDs
        """
        # Auto-generate IDs (will be created in __post_init__)
        instance_ids = []
        entity_ids = []

        # Optionally preserve IDs from metadata
        if preserve_ids and "instance_ids" in result.meta:
            instance_ids = result.meta["instance_ids"]
        if preserve_ids and "entity_ids" in result.meta:
            entity_ids = result.meta["entity_ids"]

        return cls(
            instances=result.instances,
            instance_ids=instance_ids,
            entities=result.entities,
            entity_ids=entity_ids,
            meta=result.meta.copy() if result.meta else {},
        )

    def to_vision_result(self) -> VisionResult:
        """Convert Detections artifact back to VisionResult.

        Preserves entities field for backward compatibility with VLM workflows.
        Stores IDs in metadata for potential round-trip preservation.

        Returns:
            VisionResult with instances, entities, and IDs in meta
        """
        meta = self.meta.copy()
        meta["instance_ids"] = self.instance_ids
        meta["entity_ids"] = self.entity_ids

        return VisionResult(instances=self.instances, entities=self.entities, meta=meta)

    def promote_entities(self, spatial_source: Detections, match_strategy: str = "label_fuzzy") -> Detections:
        """Promote entities to instances using spatial data from another source.

        Enables VLM → GroundingDINO fusion workflows by matching Entity labels
        with Instance spatial data (bbox/mask).

        Args:
            spatial_source: Detections artifact with spatial instances (bbox/mask)
            match_strategy: Matching strategy
                - "label_exact": Exact string match (case-sensitive)
                - "label_fuzzy": Fuzzy matching (case-insensitive, handles plurals)

        Returns:
            New Detections with promoted instances (entities that matched)

        Example:
            >>> # VLM detects "cat" and "dog" as entities
            >>> vlm_result = VisionResult(entities=[Entity("cat", 0.9)])
            >>> vlm_dets = Detections.from_vision_result(vlm_result)
            >>>
            >>> # GroundingDINO provides spatial data
            >>> spatial_result = VisionResult(instances=[Instance(..., label_name="cat")])
            >>> spatial_dets = Detections.from_vision_result(spatial_result)
            >>>
            >>> # Promote: match VLM entity with GroundingDINO bbox
            >>> promoted = vlm_dets.promote_entities(spatial_dets, "label_fuzzy")
        """
        if match_strategy not in ["label_exact", "label_fuzzy"]:
            raise ValueError(f"Invalid match_strategy '{match_strategy}'. " f"Valid: 'label_exact', 'label_fuzzy'")

        promoted_instances = []
        promoted_ids = []

        # For each entity, find matching instance in spatial_source
        for entity, entity_id in zip(self.entities, self.entity_ids):
            matched_instance = None

            # Search for matching label in spatial instances
            for inst in spatial_source.instances:
                if inst.label_name is None:
                    continue

                # Exact match
                if match_strategy == "label_exact":
                    if entity.label == inst.label_name:
                        matched_instance = inst
                        break

                # Fuzzy match
                elif match_strategy == "label_fuzzy":
                    if _fuzzy_label_match(entity.label, inst.label_name):
                        matched_instance = inst
                        break

            # If matched, promote entity to instance
            if matched_instance:
                # Merge entity attributes into instance metadata
                promoted = entity.promote(
                    bbox=matched_instance.bbox,
                    mask=matched_instance.mask,
                    label_id=matched_instance.label,
                    area=matched_instance.area,
                    is_stuff=matched_instance.is_stuff,
                    embedding=matched_instance.embedding,
                )
                promoted_instances.append(promoted)
                promoted_ids.append(entity_id)  # Preserve entity ID

        return Detections(
            instances=promoted_instances,
            instance_ids=promoted_ids,
            entities=[],  # Promoted, no longer entities
            entity_ids=[],
            meta=self.meta.copy(),
        )

    def filter_by_score(self, threshold: float) -> Detections:
        """Filter instances and entities by confidence threshold.

        Args:
            threshold: Minimum confidence score [0.0, 1.0]

        Returns:
            New Detections with filtered instances/entities
        """
        # Filter instances
        filtered_instances = []
        filtered_inst_ids = []
        for inst, inst_id in zip(self.instances, self.instance_ids):
            if inst.score >= threshold:
                filtered_instances.append(inst)
                filtered_inst_ids.append(inst_id)

        # Filter entities
        filtered_entities = []
        filtered_ent_ids = []
        for ent, ent_id in zip(self.entities, self.entity_ids):
            if ent.score >= threshold:
                filtered_entities.append(ent)
                filtered_ent_ids.append(ent_id)

        return Detections(
            instances=filtered_instances,
            instance_ids=filtered_inst_ids,
            entities=filtered_entities,
            entity_ids=filtered_ent_ids,
            meta=self.meta.copy(),
        )

    def filter_by_label(self, labels: list[str], fuzzy: bool = False) -> Detections:
        """Filter instances and entities by label names.

        Args:
            labels: List of label names to keep
            fuzzy: If True, use fuzzy matching (case-insensitive, handles plurals)

        Returns:
            New Detections with filtered instances/entities
        """
        # Filter instances
        filtered_instances = []
        filtered_inst_ids = []
        for inst, inst_id in zip(self.instances, self.instance_ids):
            if inst.label_name is None:
                continue

            matched = False
            for label in labels:
                if fuzzy:
                    if _fuzzy_label_match(inst.label_name, label):
                        matched = True
                        break
                else:
                    if inst.label_name == label:
                        matched = True
                        break

            if matched:
                filtered_instances.append(inst)
                filtered_inst_ids.append(inst_id)

        # Filter entities
        filtered_entities = []
        filtered_ent_ids = []
        for ent, ent_id in zip(self.entities, self.entity_ids):
            matched = False
            for label in labels:
                if fuzzy:
                    if _fuzzy_label_match(ent.label, label):
                        matched = True
                        break
                else:
                    if ent.label == label:
                        matched = True
                        break

            if matched:
                filtered_entities.append(ent)
                filtered_ent_ids.append(ent_id)

        return Detections(
            instances=filtered_instances,
            instance_ids=filtered_inst_ids,
            entities=filtered_entities,
            entity_ids=filtered_ent_ids,
            meta=self.meta.copy(),
        )

    def top_k(self, k: int) -> Detections:
        """Keep top K instances/entities by score.

        Args:
            k: Number of top-scoring detections to keep

        Returns:
            New Detections with top K instances/entities
        """
        # Sort instances by score (descending)
        inst_pairs = list(zip(self.instances, self.instance_ids))
        inst_pairs_sorted = sorted(inst_pairs, key=lambda x: x[0].score, reverse=True)
        top_instances = [pair[0] for pair in inst_pairs_sorted[:k]]
        top_inst_ids = [pair[1] for pair in inst_pairs_sorted[:k]]

        # Sort entities by score (descending)
        ent_pairs = list(zip(self.entities, self.entity_ids))
        ent_pairs_sorted = sorted(ent_pairs, key=lambda x: x[0].score, reverse=True)
        top_entities = [pair[0] for pair in ent_pairs_sorted[:k]]
        top_ent_ids = [pair[1] for pair in ent_pairs_sorted[:k]]

        return Detections(
            instances=top_instances,
            instance_ids=top_inst_ids,
            entities=top_entities,
            entity_ids=top_ent_ids,
            meta=self.meta.copy(),
        )

    @property
    def boxes(self) -> np.ndarray:
        """Get bounding boxes as numpy array.

        Returns:
            (N, 4) numpy array in xyxy format, where N is number of instances
            Returns empty (0, 4) array if no instances with bboxes
        """
        bboxes = []
        for inst in self.instances:
            if inst.bbox is not None:
                bboxes.append(inst.bbox)

        if len(bboxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        return np.array(bboxes, dtype=np.float32)

    @property
    def scores(self) -> np.ndarray:
        """Get confidence scores as numpy array.

        Returns:
            (N,) numpy array of scores, where N is total instances + entities
        """
        all_scores = [inst.score for inst in self.instances]
        all_scores.extend([ent.score for ent in self.entities])

        if len(all_scores) == 0:
            return np.empty((0,), dtype=np.float32)

        return np.array(all_scores, dtype=np.float32)

    @property
    def labels(self) -> list[str]:
        """Get label names as list.

        Returns:
            List of label names from instances and entities
            Uses label_name for instances, label for entities
        """
        all_labels = []

        # Instance labels
        for inst in self.instances:
            if inst.label_name is not None:
                all_labels.append(inst.label_name)
            else:
                all_labels.append(f"class_{inst.label}")

        # Entity labels
        all_labels.extend([ent.label for ent in self.entities])

        return all_labels

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with instances, entities, IDs, and metadata
        """
        return {
            "instances": [inst.to_dict() for inst in self.instances],
            "instance_ids": self.instance_ids,
            "entities": [ent.to_dict() for ent in self.entities],
            "entity_ids": self.entity_ids,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detections:
        """Create from dictionary representation.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Detections artifact
        """
        # Reconstruct instances (use VisionResult's logic)
        instances = []
        for inst_data in data.get("instances", []):
            # Handle mask reconstruction
            mask = None
            mask_info = inst_data.get("mask")
            if mask_info:
                if mask_info.get("format") == "rle":
                    mask = mask_info["data"]
                elif mask_info.get("format") == "binary":
                    mask = np.array(mask_info["data"], dtype=mask_info.get("dtype", "bool"))
                elif mask_info.get("format") == "polygon":
                    mask = mask_info["data"]
                else:
                    mask = mask_info.get("data")

            # Handle embedding reconstruction
            embedding = None
            emb_info = inst_data.get("embedding")
            if emb_info:
                if isinstance(emb_info, dict):
                    embedding = np.array(emb_info["data"], dtype=emb_info.get("dtype", "float32"))
                else:
                    embedding = emb_info

            instances.append(
                Instance(
                    bbox=tuple(inst_data["bbox"]) if inst_data.get("bbox") else None,
                    mask=mask,
                    score=inst_data["score"],
                    label=inst_data["label"],
                    label_name=inst_data.get("label_name"),
                    area=inst_data.get("area"),
                    is_stuff=inst_data.get("is_stuff"),
                    embedding=embedding,
                    track_id=inst_data.get("track_id"),
                )
            )

        # Reconstruct entities
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]

        return cls(
            instances=instances,
            instance_ids=data.get("instance_ids", []),
            entities=entities,
            entity_ids=data.get("entity_ids", []),
            meta=data.get("meta", {}),
        )

    def validate(self) -> None:
        """Validate detection artifact data.

        Raises:
            ValueError: If validation fails
        """
        # Check ID lengths match
        if len(self.instances) != len(self.instance_ids):
            raise ValueError(
                f"instances/instance_ids length mismatch: " f"{len(self.instances)} vs {len(self.instance_ids)}"
            )

        if len(self.entities) != len(self.entity_ids):
            raise ValueError(f"entities/entity_ids length mismatch: " f"{len(self.entities)} vs {len(self.entity_ids)}")

        # Check scores are in valid range
        for inst in self.instances:
            if not (0.0 <= inst.score <= 1.0):
                raise ValueError(f"Instance score {inst.score} not in [0, 1]")

        for ent in self.entities:
            if not (0.0 <= ent.score <= 1.0):
                raise ValueError(f"Entity score {ent.score} not in [0, 1]")
