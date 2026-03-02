"""Annotate node — visualization node for rendering detections/masks onto images.

Provides native visualization capabilities using existing MATA visualization
backends (PIL/matplotlib) for graph workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Annotate(Node):
    """Render detections/masks onto image using native backends.

    Uses existing MATA visualization backends (PIL for speed, matplotlib for quality)
    to overlay bounding boxes, labels, masks, and scores onto input images.
    Automatically detects whether to render as detection or segmentation based on
    the presence of masks in the detections.

    Args:
        using: Visualization backend ("pil" or "matplotlib"). Default: "pil".
        show_boxes: Whether to draw bounding boxes. Default: True.
        show_labels: Whether to show class labels. Default: True.
        show_masks: Whether to show segmentation masks. Default: True.
        show_scores: Whether to show confidence scores. Default: True.
        show_track_ids: Prepend ``#<track_id>`` to labels and use per-track
            deterministic colors. Default: False.
        alpha: Mask transparency (0.0=invisible, 1.0=opaque). Default: 0.5.
        line_width: Bounding box line width. Default: 2.
        out: Key under which the output artifact is stored. Default: "annotated".
        image_src: Key for input image artifact. Default: "image".
        detections_src: Key for input detections artifact. Default: "detections".

    Inputs:
        image (Image): Input image artifact.
        detections (Detections): Detection results with optional masks.

    Outputs:
        annotated (Image): Image with overlaid visualizations.

    Example:
        ```python
        from mata.nodes import Annotate

        # Simple bbox + label overlay
        node = Annotate(show_boxes=True, show_labels=True, out="viz")
        result = node.run(ctx, image=img, detections=dets)
        annotated = result["viz"]

        # High-quality segmentation visualization
        node = Annotate(
            using="matplotlib",
            show_masks=True,
            alpha=0.6,
            line_width=3,
            out="hq_viz"
        )
        ```
    """

    inputs = {"image": Image, "detections": Detections}
    outputs = {"annotated": Image}

    def __init__(
        self,
        using: str = "pil",
        show_boxes: bool = True,
        show_labels: bool = True,
        show_masks: bool = True,
        show_scores: bool = True,
        show_track_ids: bool = False,
        alpha: float = 0.5,
        line_width: int = 2,
        out: str = "annotated",
        image_src: str = "image",
        detections_src: str = "detections",
        **kwargs,
    ):
        """Initialize Annotate node with visualization settings.

        Args:
            using: Visualization backend ("pil" or "matplotlib")
            show_boxes: Whether to draw bounding boxes
            show_labels: Whether to show class labels
            show_masks: Whether to show segmentation masks
            show_scores: Whether to show confidence scores
            show_track_ids: Prepend ``#<track_id>`` to labels and use per-track
                deterministic colors
            alpha: Mask transparency (0.0-1.0)
            line_width: Bounding box line width
            out: Output artifact key
            image_src: Input image artifact key
            detections_src: Input detections artifact key
            **kwargs: Additional visualization arguments
        """
        super().__init__(name="Annotate")
        self.using = using
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_masks = show_masks
        self.show_scores = show_scores
        self.show_track_ids = show_track_ids
        self.alpha = alpha
        self.line_width = line_width
        self.out = out
        self.image_src = image_src
        self.detections_src = detections_src
        self.kwargs = kwargs

        # Validate backend
        if self.using not in ["pil", "matplotlib"]:
            raise ValueError(f"Invalid backend '{self.using}'. Must be 'pil' or 'matplotlib'.")

    def run(self, ctx: ExecutionContext, image: Image, detections: Detections) -> dict[str, Artifact]:
        """Execute annotation rendering on input image and detections.

        Args:
            ctx: Execution context
            image: Input image artifact
            detections: Detection results to visualize

        Returns:
            Dictionary with annotated image artifact
        """
        from mata.visualization import visualize_segmentation

        # Convert detections to VisionResult format expected by visualization
        result = detections.to_vision_result()

        # Check if we have masks for segmentation visualization
        has_masks = any(inst.mask is not None for inst in result.instances if inst.mask is not None)

        # Use segmentation visualization which handles both detections and masks
        vis = visualize_segmentation(
            result=result,
            image=image.to_pil(),
            show_boxes=self.show_boxes,
            show_labels=self.show_labels,
            show_scores=self.show_scores,
            show_track_ids=self.show_track_ids,
            alpha=self.alpha if (has_masks and self.show_masks) else 0.0,  # No alpha for detection-only
            backend=self.using,
            **self.kwargs,
        )

        # Convert PIL/matplotlib result back to Image artifact
        if self.using == "pil":
            annotated_image = Image.from_pil(
                vis, timestamp_ms=image.timestamp_ms, frame_id=image.frame_id, source_path=image.source_path
            )
        else:  # matplotlib
            # For matplotlib, save to PIL and then convert
            import io

            from PIL import Image as PILImage

            buf = io.BytesIO()
            vis.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            pil_img = PILImage.open(buf)

            annotated_image = Image.from_pil(
                pil_img, timestamp_ms=image.timestamp_ms, frame_id=image.frame_id, source_path=image.source_path
            )

            # Close matplotlib figure to save memory
            import matplotlib.pyplot as plt

            plt.close(vis)

        return {self.out: annotated_image}
