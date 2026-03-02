"""Filter node for data transformation.

Filters detections by confidence score and/or label names,
returning a new immutable Detections artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections, _fuzzy_label_match
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Filter(Node):
    """Filter detections by score threshold and/or label names.

    Applies one or more filtering criteria to a Detections artifact and
    returns a new Detections with only the matching instances/entities.

    All operations are immutable — the input artifact is never modified.
    Instance IDs are preserved through filtering so that downstream nodes
    can correlate filtered results with other artifacts.

    Args:
        src: Name of the input artifact to read from context (default: "dets").
        out: Name of the output artifact key (default: "filtered").
        score_gt: If set, keep only detections with score > this threshold.
        label_in: If set, keep only detections whose label is in this list.
        label_not_in: If set, exclude detections whose label is in this list.
        fuzzy: Use fuzzy label matching (case-insensitive, handles plurals).
        name: Optional human-readable node name.

    Inputs:
        detections (Detections): Detection results to filter.

    Outputs:
        detections (Detections): Filtered detection results.

    Example:
        ```python
        from mata.nodes import Filter
        from mata.core.artifacts.detections import Detections

        # Keep only high-confidence "person" detections
        node = Filter(
            src="dets",
            out="people",
            score_gt=0.7,
            label_in=["person"],
        )
        ```
    """

    inputs: dict[str, type[Artifact]] = {"detections": Detections}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        src: str = "dets",
        out: str = "filtered",
        score_gt: float | None = None,
        label_in: list[str] | None = None,
        label_not_in: list[str] | None = None,
        fuzzy: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.src = src
        self.out = out
        self.score_gt = score_gt
        self.label_in = label_in
        self.label_not_in = label_not_in
        self.fuzzy = fuzzy

    def run(self, ctx: ExecutionContext, detections: Detections) -> dict[str, Artifact]:
        """Apply filtering criteria to detections.

        Filtering is applied sequentially:
        1. ``score_gt`` — remove low-confidence detections
        2. ``label_in`` — keep only specified labels
        3. ``label_not_in`` — exclude specified labels

        Args:
            ctx: Execution context (unused by this node).
            detections: Input detections to filter.

        Returns:
            Dict with a single key (``self.out``) mapping to the filtered
            Detections artifact.
        """
        filtered = detections

        # 1. Filter by confidence score
        if self.score_gt is not None:
            filtered = filtered.filter_by_score(self.score_gt)

        # 2. Keep only specified labels
        if self.label_in is not None:
            filtered = filtered.filter_by_label(self.label_in, fuzzy=self.fuzzy)

        # 3. Exclude specified labels
        if self.label_not_in is not None:
            filtered = self._exclude_labels(filtered, self.label_not_in)

        # Record metrics
        ctx.record_metric(
            self.name,
            "input_count",
            len(detections.instances) + len(detections.entities),
        )
        ctx.record_metric(
            self.name,
            "output_count",
            len(filtered.instances) + len(filtered.entities),
        )

        return {self.out: filtered}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exclude_labels(self, detections: Detections, labels: list[str]) -> Detections:
        """Exclude detections whose label is in *labels*.

        This is the inverse of ``filter_by_label``.

        Args:
            detections: Source detections.
            labels: Labels to exclude.

        Returns:
            New Detections artifact without the excluded labels.
        """
        # Filter instances — keep those NOT matching any excluded label
        kept_instances = []
        kept_inst_ids = []
        for inst, inst_id in zip(detections.instances, detections.instance_ids):
            if inst.label_name is None:
                # No label → keep by default
                kept_instances.append(inst)
                kept_inst_ids.append(inst_id)
                continue

            excluded = False
            for label in labels:
                if self.fuzzy:
                    if _fuzzy_label_match(inst.label_name, label):
                        excluded = True
                        break
                else:
                    if inst.label_name == label:
                        excluded = True
                        break

            if not excluded:
                kept_instances.append(inst)
                kept_inst_ids.append(inst_id)

        # Filter entities
        kept_entities = []
        kept_ent_ids = []
        for ent, ent_id in zip(detections.entities, detections.entity_ids):
            excluded = False
            for label in labels:
                if self.fuzzy:
                    if _fuzzy_label_match(ent.label, label):
                        excluded = True
                        break
                else:
                    if ent.label == label:
                        excluded = True
                        break

            if not excluded:
                kept_entities.append(ent)
                kept_ent_ids.append(ent_id)

        return Detections(
            instances=kept_instances,
            instance_ids=kept_inst_ids,
            entities=kept_entities,
            entity_ids=kept_ent_ids,
            meta=detections.meta.copy(),
        )

    def __repr__(self) -> str:
        parts = [f"Filter(name='{self.name}'"]
        if self.score_gt is not None:
            parts.append(f"score_gt={self.score_gt}")
        if self.label_in is not None:
            parts.append(f"label_in={self.label_in}")
        if self.label_not_in is not None:
            parts.append(f"label_not_in={self.label_not_in}")
        return ", ".join(parts) + ")"
