"""KeepBestMask node for deduplicating SAM multi-mask output.

When using SAM with box prompts, the model returns multiple mask candidates
per box (typically 3). This node keeps only the highest-scoring mask per detection.

.. deprecated::
    As of v1.5.2, ``PromptBoxes`` has a built-in ``keep_best=True`` parameter
    (enabled by default) that handles multi-mask deduplication automatically.
    A separate ``KeepBestMask`` node is no longer needed in most graphs.
    This node remains available for advanced pipelines that require standalone
    mask deduplication (e.g. after ``PromptPoints`` or custom mask sources).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class KeepBestMask(Node):
    """Keep only the best mask per group (deduplicates SAM multi-mask output).

    SAM returns multiple mask candidates per prompt (typically 3 per box).
    This node groups masks and keeps only the highest-scoring one from each group.

    .. deprecated::
        Prefer using ``PromptBoxes(keep_best=True)`` (the default) instead of
        a separate ``KeepBestMask`` step. This node is retained for standalone
        deduplication after ``PromptPoints`` or other mask sources.

    Args:
        src: Name of the input masks artifact (default: "masks")
        out: Name of the output masks artifact (default: "best_masks")
        group_size: Number of masks per group (default: 3 for SAM)
        name: Optional human-readable node name

    Inputs:
        masks (Masks): Masks with multiple candidates per object

    Outputs:
        masks (Masks): Deduplicated masks (one per object)

    Example:
        ```python
        # Preferred (v1.5.2+) — no separate KeepBestMask needed:
        from mata.nodes import Detect, Filter, PromptBoxes, Fuse

        graph = [
            Detect(using="detr", out="dets"),
            Filter(src="dets", score_gt=0.9, out="filtered"),
            PromptBoxes(using="sam", dets_src="filtered", out="masks"),  # keep_best=True by default
            Fuse(detections="filtered", masks="masks", out="final"),
        ]

        # Legacy usage (still supported):
        from mata.nodes import KeepBestMask
        graph = [
            ...
            PromptBoxes(using="sam", dets_src="filtered", keep_best=False, out="all_masks"),
            KeepBestMask(src="all_masks", out="masks"),
            ...
        ]
        ```
    """

    inputs: dict[str, type[Artifact]] = {"masks": Masks}
    outputs: dict[str, type[Artifact]] = {"masks": Masks}

    def __init__(
        self,
        src: str = "masks",
        out: str = "best_masks",
        group_size: int = 3,
        name: str | None = None,
    ):
        super().__init__(name=name or "KeepBestMask")
        self.src = src
        self.out = out
        self.group_size = group_size

    def run(self, ctx: ExecutionContext, masks: Masks) -> dict[str, Artifact]:
        """Keep only the best mask from each group.

        Assumes masks are ordered as: [obj1_mask1, obj1_mask2, obj1_mask3,
        obj2_mask1, obj2_mask2, obj2_mask3, ...] where each group of
        group_size masks belongs to the same object.

        Args:
            ctx: Execution context
            masks: Input masks with multiple candidates per object

        Returns:
            Dictionary with deduplicated masks
        """
        if len(masks.instances) == 0:
            # No masks to process
            return {self.out: masks}

        # Group masks and keep the best from each group
        best_instances = []
        best_ids = []

        num_groups = len(masks.instances) // self.group_size
        remainder = len(masks.instances) % self.group_size

        for group_idx in range(num_groups):
            start_idx = group_idx * self.group_size
            end_idx = start_idx + self.group_size

            # Get group of masks
            group_instances = masks.instances[start_idx:end_idx]
            group_ids = masks.instance_ids[start_idx:end_idx]

            # Find best mask in group (highest score)
            best_idx = max(range(len(group_instances)), key=lambda i: group_instances[i].score or 0.0)

            best_instances.append(group_instances[best_idx])
            best_ids.append(group_ids[best_idx])

        # Handle remainder masks (if not evenly divisible)
        if remainder > 0:
            start_idx = num_groups * self.group_size
            remaining_instances = masks.instances[start_idx:]
            remaining_ids = masks.instance_ids[start_idx:]

            # Find best among remaining
            best_idx = max(range(len(remaining_instances)), key=lambda i: remaining_instances[i].score or 0.0)
            best_instances.append(remaining_instances[best_idx])
            best_ids.append(remaining_ids[best_idx])

        # Create new Masks artifact
        deduplicated = Masks(
            instances=best_instances,
            instance_ids=best_ids,
            meta={**masks.meta, "deduplicated": True, "original_count": len(masks.instances)},
        )

        # Record metrics
        ctx.record_metric(self.name, "input_masks", len(masks.instances))
        ctx.record_metric(self.name, "output_masks", len(best_instances))
        ctx.record_metric(self.name, "group_size", self.group_size)

        return {self.out: deduplicated}

    def __repr__(self) -> str:
        return f"KeepBestMask(name='{self.name}', src='{self.src}', out='{self.out}', group_size={self.group_size})"
