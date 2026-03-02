"""PromptPoints node for SAM point-based segmentation.

Segments regions using explicit point prompts for SAM-style models.
Each point is specified as ``(x, y, label)`` where *label* is 1 for
foreground and 0 for background.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class PromptPoints(Node):
    """Segment using explicit point prompts.

    Accepts an image and a list of user-defined point prompts, then runs a
    segmentation provider (typically SAM) in ``"points"`` mode. Each prompt
    is a tuple ``(x, y, label)`` where *label* indicates foreground (1) or
    background (0).

    Point prompts are configured at construction time (not wired from the
    graph) because they typically come from user interaction rather than
    from another node's output.

    Args:
        using: Name of the Segmenter provider in the context
            (e.g. ``"sam"``, ``"sam3"``).
        points: List of point prompts as ``(x, y, label)`` tuples.
            - ``x, y``: Pixel coordinates in the image.
            - ``label``: 1 = foreground, 0 = background.
        image_src: Artifact name for the input image (default: ``"image"``).
        out: Key used for the output Masks artifact (default: ``"masks"``).
        name: Optional human-readable node name.
        **kwargs: Additional keyword arguments forwarded to the segmenter's
            ``segment()`` call (e.g. ``multimask_output``).

    Inputs:
        image (Image): Source image to segment.

    Outputs:
        masks (Masks): Segmentation masks produced from point prompts.

    Example:
        ```python
        from mata.nodes.prompt_points import PromptPoints

        # Foreground point at (150, 200), background at (10, 10)
        node = PromptPoints(
            using="sam",
            points=[(150, 200, 1), (10, 10, 0)],
            out="masks",
        )
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"masks": Masks}

    def __init__(
        self,
        using: str,
        points: list[tuple[int, int, int]] | None = None,
        image_src: str = "image",
        out: str = "masks",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.provider_name = using
        self.points = list(points) if points else []
        self.image_src = image_src
        self.output_name = out
        self.kwargs = kwargs

        self._validate_points(self.points)

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Segment using point prompts.

        Steps:
        1. Retrieve the Segmenter provider from context.
        2. Validate the configured point prompts.
        3. Call the segmenter with ``point_prompts`` and ``mode="points"``.
        4. Record metrics.

        Args:
            ctx: Execution context with providers and metrics.
            image: Input image artifact.

        Returns:
            Dict mapping ``self.output_name`` to the resulting Masks artifact.

        Raises:
            KeyError: If the segmentation provider is not found in context.
            ValueError: If no point prompts have been configured.
        """
        if not self.points:
            raise ValueError(
                f"PromptPoints node '{self.name}': no point prompts configured. "
                f"Pass points=[(x, y, label), ...] at construction time."
            )

        segmenter = ctx.get_provider("segment", self.provider_name)

        # Segment with point prompts
        masks = segmenter.segment(
            image,
            point_prompts=self.points,
            mode="points",
            **self.kwargs,
        )

        # Add prompt metadata
        masks = Masks(
            instances=masks.instances,
            instance_ids=masks.instance_ids,
            meta={**masks.meta, "prompt_type": "points", "num_points": len(self.points)},
        )

        # Record metrics
        ctx.record_metric(self.name, "num_point_prompts", float(len(self.points)))
        fg_count = sum(1 for _, _, label in self.points if label == 1)
        bg_count = len(self.points) - fg_count
        ctx.record_metric(self.name, "fg_points", float(fg_count))
        ctx.record_metric(self.name, "bg_points", float(bg_count))
        ctx.record_metric(self.name, "num_masks", float(len(masks.instances)))

        return {self.output_name: masks}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_points(points: list[tuple[int, int, int]]) -> None:
        """Validate point prompt format.

        Each point must be a 3-element tuple ``(x, y, label)`` where
        *label* is 0 (background) or 1 (foreground).

        Args:
            points: Point prompts to validate.

        Raises:
            ValueError: If any point has invalid format.
        """
        for i, pt in enumerate(points):
            if not isinstance(pt, (tuple, list)) or len(pt) != 3:
                raise ValueError(f"Point prompt at index {i} must be a (x, y, label) tuple, " f"got {pt!r}")

            x, y, label = pt
            if not isinstance(label, (int, float)) or int(label) not in (0, 1):
                raise ValueError(
                    f"Point prompt label at index {i} must be 0 (background) or " f"1 (foreground), got {label!r}"
                )

    def __repr__(self) -> str:
        return (
            f"PromptPoints(name='{self.name}', "
            f"using='{self.provider_name}', "
            f"points={len(self.points)}, "
            f"out='{self.output_name}')"
        )
