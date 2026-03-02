"""PromptBoxes node for SAM prompt-based segmentation.

Segments regions using detection bounding boxes as prompts for SAM-style
segmentation models. Each detection box becomes a prompt, producing a
corresponding mask with aligned instance IDs.

Includes built-in multi-mask deduplication: SAM typically returns multiple
mask candidates per box prompt (e.g. 3). When ``keep_best=True`` (the
default), only the highest-scoring mask per prompt is kept automatically,
removing the need for a separate ``KeepBestMask`` node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node
from mata.core.types import VisionResult

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class PromptBoxes(Node):
    """Segment using detection bounding boxes as prompts (SAM).

    Takes an image and a set of detections, converts each detection's
    bounding box into a box prompt, and runs a segmentation provider
    (typically SAM) to produce instance masks.

    Instance IDs from the input detections are preserved on the output
    masks so that downstream nodes can align detections with their
    corresponding masks.

    **Multi-mask deduplication (keep_best):**

    SAM models return multiple mask candidates per prompt (typically 3),
    each scored by IoU confidence. By default (``keep_best=True``), this
    node automatically keeps only the highest-scoring mask per box prompt,
    producing exactly one mask per detection. Set ``keep_best=False`` to
    retain all mask candidates (useful for custom downstream selection).

    The ``group_size`` parameter controls how many consecutive masks form
    one group for deduplication. It defaults to ``3`` (SAM's standard
    output). When set to ``None``, it is auto-detected from the ratio
    ``output_masks / input_prompts``.

    Args:
        using: Name of the Segmenter provider to look up in the context
            (e.g. ``"sam"``, ``"sam3"``).
        image_src: Artifact name for the input image (default: ``"image"``).
        dets_src: Artifact name for the input detections (default: ``"dets"``).
        out: Key used for the output Masks artifact (default: ``"masks"``).
        keep_best: If ``True`` (default), automatically deduplicate SAM
            multi-mask output by keeping only the highest-scoring mask
            per box prompt. If ``False``, return all mask candidates.
        group_size: Number of mask candidates per prompt for grouping
            during deduplication (default: ``3``). Set to ``None`` for
            auto-detection based on output/input ratio. Ignored when
            ``keep_best=False``.
        name: Optional human-readable node name.
        **kwargs: Additional keyword arguments forwarded to the segmenter's
            ``segment()`` call (e.g. ``multimask_output``, ``threshold``).

    Inputs:
        image (Image): Source image to segment.
        detections (Detections): Detection results whose bounding boxes
            are used as box prompts.

    Outputs:
        masks (Masks): Segmentation masks aligned with input detections.

    Example:
        ```python
        from mata.nodes import Detect, Filter, PromptBoxes, Fuse

        # Default: keep_best=True — produces 1 mask per detection
        graph = [
            Detect(using="detr", out="dets"),
            Filter(src="dets", score_gt=0.9, out="filtered"),
            PromptBoxes(using="sam", dets_src="filtered", out="masks"),
            Fuse(detections="filtered", masks="masks", out="final"),
        ]

        # Explicit: keep all 3 candidates per box
        PromptBoxes(using="sam", keep_best=False, out="all_masks")
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image, "detections": Detections}
    outputs: dict[str, type[Artifact]] = {"masks": Masks}

    def __init__(
        self,
        using: str,
        image_src: str = "image",
        dets_src: str = "dets",
        out: str = "masks",
        keep_best: bool = True,
        group_size: int | None = 3,
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.image_src = image_src
        self.dets_src = dets_src
        self.output_name = out
        self.keep_best = keep_best
        self.group_size = group_size
        self.kwargs = kwargs

    def run(self, ctx: ExecutionContext, image: Image, detections: Detections) -> dict[str, Artifact]:
        """Segment using detection boxes as prompts.

        Steps:
        1. Retrieve the Segmenter provider from context.
        2. Extract bounding boxes from the input detections.
        3. Call the segmenter with ``box_prompts`` and ``mode="boxes"``.
        4. If ``keep_best`` is enabled and the segmenter returned more masks
           than prompts (SAM multi-mask output), deduplicate by keeping only
           the highest-scoring mask per group.
        5. Align output mask instance IDs with detection instance IDs.
        6. Record metrics.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.
            detections: Detections whose bboxes serve as prompts.

        Returns:
            Dict mapping ``self.output_name`` to the resulting Masks artifact.

        Raises:
            KeyError: If the segmentation provider is not found in context.
            ValueError: If detections contain no instances with bboxes.
        """
        segmenter = ctx.get_provider("segment", self.provider_name)

        # Extract bounding boxes from detections
        box_prompts = self._extract_box_prompts(detections)

        if not box_prompts:
            raise ValueError(
                f"PromptBoxes node '{self.name}': no bounding boxes found in "
                f"input detections. Ensure upstream detection node produces "
                f"instances with bbox data."
            )

        # Segment with box prompts
        masks = segmenter.segment(
            image,
            box_prompts=box_prompts,
            mode="boxes",
            **self.kwargs,
        )

        # Convert VisionResult to Masks artifact if needed
        if isinstance(masks, VisionResult):
            masks = Masks.from_vision_result(masks)

        # Deduplicate SAM multi-mask output if enabled
        num_raw_masks = len(masks.instances)
        if self.keep_best and num_raw_masks > len(box_prompts):
            masks = self._keep_best_per_group(masks, len(box_prompts))

        # Align instance IDs from detections to masks
        masks = self._align_instance_ids(masks, detections)

        # Record metrics
        ctx.record_metric(self.name, "num_box_prompts", float(len(box_prompts)))
        ctx.record_metric(self.name, "num_masks", float(len(masks.instances)))
        if self.keep_best and num_raw_masks > len(box_prompts):
            ctx.record_metric(self.name, "raw_masks_before_dedup", float(num_raw_masks))

        return {self.output_name: masks}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_box_prompts(detections: Detections) -> list[tuple]:
        """Extract bounding box prompts from detection instances.

        Only instances that have a non-None ``bbox`` are included.

        Args:
            detections: Source detections.

        Returns:
            List of bounding box tuples in xyxy format.
        """
        return [inst.bbox for inst in detections.instances if inst.bbox is not None]

    @staticmethod
    def _align_instance_ids(masks: Masks, detections: Detections) -> Masks:
        """Create new Masks with instance IDs aligned to detections.

        Uses the detection instance IDs so that masks and detections share
        the same ID namespace for downstream fusion/merging.

        If the number of masks differs from detections (e.g. the segmenter
        filtered some), only the overlapping prefix is aligned; extra masks
        keep their auto-generated IDs.

        Args:
            masks: Masks produced by the segmenter.
            detections: Original detections with instance IDs.

        Returns:
            New Masks artifact with aligned instance IDs.
        """
        # Only align IDs for instances that have bboxes (same order as prompts)
        det_ids = [
            inst_id for inst, inst_id in zip(detections.instances, detections.instance_ids) if inst.bbox is not None
        ]

        num_masks = len(masks.instances)
        num_det_ids = len(det_ids)

        if num_masks == num_det_ids:
            aligned_ids = list(det_ids)
        elif num_masks < num_det_ids:
            aligned_ids = det_ids[:num_masks]
        else:
            # More masks than det IDs — keep det IDs for first N, auto-gen rest
            aligned_ids = list(det_ids) + masks.instance_ids[num_det_ids:]

        return Masks(
            instances=masks.instances,
            instance_ids=aligned_ids,
            meta={**masks.meta, "prompt_type": "boxes"},
        )

    def _keep_best_per_group(self, masks: Masks, num_prompts: int) -> Masks:
        """Deduplicate multi-mask output by keeping the best mask per group.

        SAM returns multiple mask candidates per prompt (typically 3). This
        method groups consecutive masks and keeps only the highest-scoring
        one from each group.

        Args:
            masks: Masks artifact with multiple candidates per prompt.
            num_prompts: Number of box prompts sent to the segmenter.

        Returns:
            Deduplicated Masks artifact with one mask per prompt.
        """
        if len(masks.instances) == 0:
            return masks

        # Determine group size: use configured value or auto-detect
        if self.group_size is not None:
            group_size = self.group_size
        else:
            group_size = len(masks.instances) // num_prompts if num_prompts > 0 else 1

        if group_size <= 1:
            return masks

        best_instances = []
        best_ids = []

        num_groups = len(masks.instances) // group_size
        remainder = len(masks.instances) % group_size

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            group_instances = masks.instances[start_idx:end_idx]
            group_ids = masks.instance_ids[start_idx:end_idx]

            best_idx = max(
                range(len(group_instances)),
                key=lambda i: group_instances[i].score or 0.0,
            )
            best_instances.append(group_instances[best_idx])
            best_ids.append(group_ids[best_idx])

        # Handle remainder masks (if not evenly divisible)
        if remainder > 0:
            start_idx = num_groups * group_size
            remaining_instances = masks.instances[start_idx:]
            remaining_ids = masks.instance_ids[start_idx:]

            best_idx = max(
                range(len(remaining_instances)),
                key=lambda i: remaining_instances[i].score or 0.0,
            )
            best_instances.append(remaining_instances[best_idx])
            best_ids.append(remaining_ids[best_idx])

        return Masks(
            instances=best_instances,
            instance_ids=best_ids,
            meta={**masks.meta, "deduplicated": True, "original_count": len(masks.instances)},
        )

    def __repr__(self) -> str:
        keep_str = f", keep_best={self.keep_best}" if not self.keep_best else ""
        return (
            f"PromptBoxes(name='{self.name}', " f"using='{self.provider_name}', " f"out='{self.output_name}'{keep_str})"
        )
