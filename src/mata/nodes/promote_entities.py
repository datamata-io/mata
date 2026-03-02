"""PromoteEntities node — promote semantic entities to spatial instances.

Enables VLM -> GroundingDINO/DETR fusion workflows by matching Entity labels
(from VLM semantic output) with Instance spatial data (bbox/mask from spatial
detection models).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class PromoteEntities(Node):
    """Promote semantic entities to spatial instances.

    Takes two Detections artifacts — one from a VLM (containing entities)
    and one from a spatial detector like GroundingDINO or DETR (containing
    instances with bboxes/masks) — and promotes the VLM entities to full
    instances by matching labels.

    Args:
        entities_src: Artifact name for the VLM detections input
            (default ``"vlm_dets"``).
        spatial_src: Artifact name for the spatial detections input
            (default ``"dino_dets"``).
        match_strategy: Label matching strategy — ``"label_exact"`` for
            exact case-sensitive matching, or ``"label_fuzzy"`` for
            normalized matching (strips articles, plurals, casing)
            (default ``"label_fuzzy"``).
        out: Key under which the output artifact is stored
            (default ``"promoted"``).
        name: Optional human-readable node name.

    Inputs:
        entities (Detections): VLM detections containing entities.
        spatial (Detections): Spatial detections containing instances
            with bboxes/masks.

    Outputs:
        detections (Detections): Promoted detections where entities have
            been matched to spatial instances.

    Example:
        ```python
        from mata.nodes import PromoteEntities

        node = PromoteEntities(
            entities_src="vlm_dets",
            spatial_src="dino_dets",
            match_strategy="label_fuzzy",
        )
        result = node.run(ctx, entities=vlm_dets, spatial=dino_dets)
        promoted = result["promoted"]
        ```
    """

    inputs: dict[str, type[Artifact]] = {
        "entities": Detections,
        "spatial": Detections,
    }
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        entities_src: str = "vlm_dets",
        spatial_src: str = "dino_dets",
        match_strategy: str = "label_fuzzy",
        out: str = "promoted",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.entities_src = entities_src
        self.spatial_src = spatial_src
        self.match_strategy = match_strategy
        self.output_name = out

    def run(
        self,
        ctx: ExecutionContext,
        entities: Detections,
        spatial: Detections,
    ) -> dict[str, Artifact]:
        """Match entities to spatial detections and promote.

        Args:
            ctx: Execution context with providers and metrics.
            entities: Detections artifact from VLM (containing entities).
            spatial: Detections artifact from spatial detector (containing
                instances with bboxes/masks).

        Returns:
            Dict with a single key (``self.output_name``) mapping to a
            new Detections artifact with promoted instances.
        """
        from mata.core.artifacts.converters import (
            generate_instance_ids,
            promote_entities_to_instances,
        )

        start = time.time()

        # Promote entities using spatial instances
        promoted_instances = promote_entities_to_instances(
            entities.entities,
            spatial.instances,
            match_strategy=self.match_strategy,
        )

        latency_ms = (time.time() - start) * 1000

        # Create new Detections with promoted instances
        promoted = Detections(
            instances=promoted_instances,
            entities=[],  # All promoted to instances
            instance_ids=generate_instance_ids(len(promoted_instances)),
            entity_ids=[],
            meta={
                "match_strategy": self.match_strategy,
                "promoted_count": len(promoted_instances),
                "source_entity_count": len(entities.entities),
                "source_spatial_count": len(spatial.instances),
            },
        )

        # Record metrics
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_promoted", float(len(promoted_instances)))
        ctx.record_metric(self.name, "num_entities_input", float(len(entities.entities)))
        ctx.record_metric(self.name, "num_spatial_input", float(len(spatial.instances)))

        return {self.output_name: promoted}
