"""TiledDetect node — sliced inference over overlapping crops.

Runs a detection provider on a grid of overlapping tiles and merges the
results back into source-image coordinates. Useful for high-resolution
imagery where small objects are lost when the whole frame is resized to the
model input.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.node import Node
from mata.core.tiling import Tile, generate_tiles, merge_instances

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class TiledDetect(Node):
    """Object detection over overlapping tiles, merged into source coordinates.

    The node slices the input image into a grid (edge tiles are shifted inward,
    never padded), runs the provider on every tile, offsets each box by the
    tile origin, and applies class-aware NMS across the seam duplicates.

    Args:
        using: Name of the detection provider registered in the context.
        out: Key under which the merged Detections artifact is stored.
        slice_size: ``(width, height)`` of a tile, or one int for square tiles.
        overlap: Fractional overlap between neighbouring tiles, in ``[0, 1)``.
        include_full_image: Also run the provider on the full image (adds one
            inference, catches objects larger than a tile). Ignored when the
            image already fits in a single tile.
        iou_threshold: NMS IoU used when merging tile detections.
        max_det: Optional cap on the number of returned detections.
        name: Optional human-readable node name.
        **predict_kwargs: Extra keyword arguments forwarded to ``predict()``.

    Inputs:
        image (Image): Source image artifact.

    Outputs:
        detections (Detections): Merged detections in source-image pixels.

    Example:
        ```python
        from mata.nodes import TiledDetect

        node = TiledDetect(
            using="detector",
            slice_size=(640, 640),
            overlap=0.2,
            include_full_image=True,
            out="dets",
        )
        ```
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"detections": Detections}

    def __init__(
        self,
        using: str,
        out: str = "dets",
        slice_size: tuple[int, int] | int = (640, 640),
        overlap: float = 0.2,
        include_full_image: bool = True,
        iou_threshold: float = 0.5,
        max_det: int | None = None,
        name: str | None = None,
        **predict_kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if max_det is not None and max_det <= 0:
            raise ValueError(f"max_det must be positive or None, got {max_det}")
        if isinstance(slice_size, int):
            slice_size = (slice_size, slice_size)
        if len(slice_size) != 2 or slice_size[0] <= 0 or slice_size[1] <= 0:
            raise ValueError(f"slice_size must be two positive ints, got {slice_size}")

        self.provider_name = using
        self.output_name = out
        self.slice_size = (int(slice_size[0]), int(slice_size[1]))
        self.overlap = float(overlap)
        self.include_full_image = include_full_image
        self.iou_threshold = float(iou_threshold)
        self.max_det = max_det
        self.kwargs = predict_kwargs

    def run(self, ctx: ExecutionContext, image: Image) -> dict[str, Artifact]:
        """Run the provider over every tile and merge the detections."""
        detector = ctx.get_provider("detect", self.provider_name)
        pil_image = image.to_pil()
        width, height = pil_image.size

        start = time.time()
        tiles = generate_tiles(width, height, self.slice_size, self.overlap)
        if self.include_full_image and len(tiles) > 1:
            tiles = [Tile(0, 0, width, height), *tiles]

        raw_instances = []
        for tile in tiles:
            crop = pil_image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
            result = detector.predict(Image.from_pil(crop), **self.kwargs)
            detections = result if isinstance(result, Detections) else Detections.from_vision_result(result)
            for inst in detections.instances:
                if inst.bbox is None:
                    continue
                raw_instances.append(replace(inst, bbox=tile.offset_box(inst.bbox)))

        merged = merge_instances(raw_instances, self.iou_threshold, self.max_det)
        latency_ms = (time.time() - start) * 1000

        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_tiles", float(len(tiles)))
        ctx.record_metric(self.name, "num_detections_raw", float(len(raw_instances)))
        ctx.record_metric(self.name, "num_detections", float(len(merged)))

        return {
            self.output_name: Detections(
                instances=merged,
                meta={
                    "source_size": [width, height],
                    "tiles": len(tiles),
                    "slice_size": list(self.slice_size),
                    "overlap": self.overlap,
                    "raw_detections": len(raw_instances),
                },
            )
        }

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = (
            f"TiledDetect(using='{self.provider_name}', out='{self.output_name}', "
            f"slice_size={self.slice_size}, overlap={self.overlap}"
        )
        if extra:
            parts += f", {extra}"
        return parts + ")"
