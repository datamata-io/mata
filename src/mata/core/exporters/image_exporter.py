"""Image overlay exporter for MATA result types.

Exports visual overlays with bounding boxes, masks, and labels.
Supports VisionResult, DetectResult, SegmentResult, ClassifyResult, and DepthResult.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mata.core.exceptions import InvalidInputError
from mata.core.logging import get_logger

if TYPE_CHECKING:
    from mata.core.types import (
        ClassifyResult,
        DepthResult,
        DetectResult,
        Instance,
        OCRResult,
        SegmentResult,
        VisionResult,
    )

logger = get_logger(__name__)


def export_image(
    result: VisionResult | DetectResult | SegmentResult | ClassifyResult | DepthResult,
    output_path: str | Path,
    image: str | Path | Image.Image | np.ndarray | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_scores: bool = True,
    show_masks: bool = True,
    show_track_ids: bool = False,
    alpha: float = 0.5,
    **kwargs,
) -> None:
    """Export result as image overlay.

    Creates visual representation by drawing bboxes/masks on original image.
    For ClassifyResult, generates a bar chart of predictions.

    Args:
        result: Result object to visualize
        output_path: Path to save output image
        image: Original image (path, PIL Image, or numpy array).
            If None, uses result.meta['input_path'] if available.
        show_boxes: Draw bounding boxes (default: True)
        show_labels: Draw class labels (default: True)
        show_scores: Show confidence scores (default: True)
        show_masks: Overlay segmentation masks (default: True)
        alpha: Mask overlay transparency [0.0, 1.0] (default: 0.5)
        **kwargs: Additional visualization parameters

    Raises:
        InvalidInputError: If image is None and not in result.meta
        ValueError: If result type has no image visualization
        IOError: If image save fails

    Examples:
        >>> # Auto-use stored input_path
        >>> result = mata.run("detect", "image.jpg")
        >>> export_image(result, "output.png")
        >>>
        >>> # Explicit image parameter
        >>> result = mata.run("detect", pil_image)
        >>> export_image(result, "output.png", image="image.jpg")
        >>>
        >>> # Customize overlay
        >>> export_image(
        ...     result,
        ...     "output.png",
        ...     show_boxes=True,
        ...     show_labels=False,
        ...     alpha=0.3
        ... )
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine result type and export accordingly
    result_type = type(result).__name__

    try:
        if result_type in ("VisionResult", "DetectResult", "SegmentResult"):
            _export_detection_segmentation_image(
                result,
                output_path,
                image,
                show_boxes,
                show_labels,
                show_scores,
                show_masks,
                alpha,
                show_track_ids=show_track_ids,
            )
        elif result_type == "ClassifyResult":
            _export_classification_image(result, output_path, image, **kwargs)
        elif result_type == "DepthResult":
            _export_depth_image(result, output_path, **kwargs)
        elif result_type == "OCRResult":
            export_ocr_image(result, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported result type for image export: {result_type}")
    except Exception as e:
        logger.error(f"Failed to write image to {output_path}: {e}")
        raise OSError(f"Failed to export image: {e}") from e


def _export_detection_segmentation_image(
    result: VisionResult | DetectResult | SegmentResult,
    output_path: Path,
    image: str | Path | Image.Image | np.ndarray | None,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    show_masks: bool,
    alpha: float,
    show_track_ids: bool = False,
) -> None:
    """Export detection/segmentation overlay image.

    Draws bounding boxes and/or masks on the original image.
    """
    # Resolve image path
    if image is None:
        if hasattr(result, "get_input_path"):
            image = result.get_input_path()
        elif hasattr(result, "meta") and result.meta is not None and "input_path" in result.meta:
            image = result.meta["input_path"]

        if image is None:
            raise InvalidInputError(
                "Image overlay requires original image. "
                "Provide via: export_image(result, output_path, image='path.jpg')"
            )

    # Load image
    pil_image = _load_image(image)

    # For segmentation with masks, use existing visualization
    if show_masks and hasattr(result, "masks") and len(result.masks) > 0:
        from mata.visualization import visualize_segmentation

        vis_image = visualize_segmentation(
            result,
            image=pil_image,
            alpha=alpha,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_scores=show_scores,
            backend="pil",
        )

        if vis_image is not None:
            vis_image.save(output_path)
            logger.info(f"Exported segmentation overlay to {output_path}")
            return

    # For detection-only, draw bounding boxes
    if show_boxes:
        # Get instances with bboxes
        if hasattr(result, "detections"):
            instances = result.detections
        elif hasattr(result, "instances"):
            instances = [inst for inst in result.instances if inst.bbox is not None]
        else:
            instances = []

        if len(instances) > 0:
            pil_image = _draw_bounding_boxes(
                pil_image,
                instances,
                show_labels=show_labels,
                show_scores=show_scores,
                show_track_ids=show_track_ids,
            )

    # Save output
    pil_image.save(output_path)
    logger.info(f"Exported detection overlay to {output_path}")


def _export_classification_image(
    result: ClassifyResult,
    output_path: Path,
    image: str | Path | Image.Image | np.ndarray | None,
    top_k: int = 5,
    **kwargs,
) -> None:
    """Export classification result as bar chart.

    Creates a horizontal bar chart showing top-k predictions.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib required for classification visualization. " "Install with: pip install matplotlib"
        )

    predictions = result.predictions[:top_k] if hasattr(result, "predictions") else []

    if len(predictions) == 0:
        logger.warning("No predictions to visualize")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(predictions) * 0.6)))

    labels = [p.label_name or f"class_{p.label}" for p in predictions]
    scores = [p.score for p in predictions]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, scores, align="center", alpha=0.8, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Highest score at top
    ax.set_xlabel("Confidence Score")
    ax.set_title(f"Top {len(predictions)} Classifications")
    ax.set_xlim(0, 1.0)

    # Add score text
    for i, (label, score) in enumerate(zip(labels, scores)):
        ax.text(score + 0.02, i, f"{score:.3f}", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Exported classification chart to {output_path}")


def _export_depth_image(result: DepthResult, output_path: Path, colormap: str | None = None, **kwargs) -> None:
    """Export depth result as a grayscale or colormap image.

    Args:
        result: DepthResult instance
        output_path: Output image path
        colormap: Optional matplotlib colormap name (e.g., "magma")
    """
    depth = result.normalized if result.normalized is not None else result.depth
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth, dtype=np.float32)

    depth_min = float(np.nanmin(depth))
    depth_max = float(np.nanmax(depth))
    if depth_max - depth_min > 1e-8:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)

    if colormap:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as cm

            cmap = cm.get_cmap(colormap)
            colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(colored)
        except ImportError:
            raise ImportError("Matplotlib required for colormap depth export. " "Install with: pip install matplotlib")
    else:
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        img = Image.fromarray(depth_uint8, mode="L")

    img.save(output_path)
    logger.info(f"Exported depth map to {output_path}")


def _load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load and convert image to PIL Image."""
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        raise InvalidInputError(f"Unsupported image type: {type(image).__name__}")
    return pil_image


def _get_track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic, visually distinct RGB color for a track ID.

    Uses MD5 hash of the track ID to derive a hue value, then converts from
    HSV (fixed high saturation and value) to RGB so that the output is always
    saturated and bright regardless of the track ID.

    Args:
        track_id: Integer track identifier.

    Returns:
        Tuple of (R, G, B) integers in [0, 255].
    """
    import colorsys
    import hashlib

    h = hashlib.md5(str(track_id).encode()).hexdigest()
    hue = int(h[:4], 16) / 65535.0  # Map 16-bit value to [0.0, 1.0]
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)  # High saturation and value
    return (int(r * 255), int(g * 255), int(b * 255))


class TrackTrailRenderer:
    """Maintains per-track position history and renders trajectory trails using PIL.

    Stores center positions per track ID across frames and draws fading
    polylines on annotated frames.  All rendering is PIL-only — no OpenCV
    dependency.

    Example usage::

        renderer = TrackTrailRenderer(trail_length=30)
        for result in mata.track("video.mp4", stream=True):
            pil_frame = ...  # your PIL image for this frame
            renderer.update(result.instances)
            pil_frame = renderer.draw_trails(pil_frame)
    """

    def __init__(self, trail_length: int = 30) -> None:
        """Initialise the renderer.

        Args:
            trail_length: Maximum number of historical center positions to
                keep per track ID (ring-buffer behaviour).
        """
        self._history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._last_seen: dict[int, int] = {}
        self._frame_count: int = 0
        self.trail_length: int = trail_length

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, instances: list[Instance]) -> None:  # type: ignore[name-defined]
        """Record center positions for the current frame's tracked instances.

        Appends the bounding-box center of each instance that has a
        ``track_id`` to its history deque, capping the history at
        *trail_length*.  Stale tracks (not seen for *trail_length* frames)
        are pruned automatically.

        Args:
            instances: List of :class:`~mata.core.types.Instance` objects
                from the current frame's :class:`~mata.core.types.VisionResult`.
        """
        self._frame_count += 1

        for inst in instances:
            if inst.track_id is None or inst.bbox is None:
                continue
            tid: int = inst.track_id
            self._last_seen[tid] = self._frame_count
            x1, y1, x2, y2 = inst.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            hist = self._history[tid]
            hist.append((cx, cy))
            # Cap to trail_length (ring-buffer)
            if len(hist) > self.trail_length:
                del hist[: len(hist) - self.trail_length]

        # Prune tracks not seen for trail_length frames
        stale = [tid for tid, last in self._last_seen.items() if self._frame_count - last > self.trail_length]
        for tid in stale:
            self._history.pop(tid, None)
            self._last_seen.pop(tid, None)

    def draw_trails(
        self,
        image: Image.Image,
        color_fn: Callable[[int], tuple] | None = None,
        thickness: int = 2,
        alpha: float = 0.7,
    ) -> Image.Image:
        """Draw trajectory polylines for all active tracks onto *image*.

        Uses PIL alpha compositing — no OpenCV required.  Each trail
        segment is drawn with a linearly increasing opacity so that
        recent positions appear solid and older positions fade out.

        Args:
            image: PIL Image on which to draw the trails.  The image is
                *not* modified in-place; a new image is returned.
            color_fn: Callable ``(track_id: int) -> (R, G, B)`` used to
                determine each track's color.  Defaults to
                :func:`_get_track_color`.
            thickness: Line width in pixels (default 2).
            alpha: Maximum opacity of the newest trail segment, in
                ``[0.0, 1.0]`` (default 0.7).  Older segments scale
                proportionally toward 0.

        Returns:
            A new PIL Image (RGB) with the trails composited on top of
            *image*.
        """
        if color_fn is None:
            color_fn = _get_track_color

        # Fast-path: nothing to draw
        has_trails = any(len(pts) >= 2 for pts in self._history.values())
        if not has_trails:
            return image

        # Create a fully transparent RGBA overlay canvas
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for tid, pts in self._history.items():
            n = len(pts)
            if n < 2:
                continue
            r, g, b = color_fn(tid)
            for i in range(1, n):
                # Linearly fade: index 1 → near-transparent, index n-1 → alpha
                fade = (i / n) * alpha
                a = int(fade * 255)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                draw.line([p1, p2], fill=(r, g, b, a), width=thickness)

        result = Image.alpha_composite(base, overlay)
        return result.convert("RGB")

    def reset(self) -> None:
        """Clear all track history and reset the frame counter."""
        self._history.clear()
        self._last_seen.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_track_ids(self) -> list[int]:
        """Return track IDs that have at least one recorded position."""
        return [tid for tid, pts in self._history.items() if pts]


def _draw_bounding_boxes(
    image: Image.Image,
    instances,
    show_labels: bool = True,
    show_scores: bool = True,
    show_track_ids: bool = False,
    box_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """Draw bounding boxes on image.

    Args:
        image: PIL Image to draw on
        instances: List of Instance objects with bbox
        show_labels: Show class labels
        show_scores: Show confidence scores
        show_track_ids: Prepend ``#<track_id>`` to labels and use per-track
            deterministic colors. Default False.
        box_width: Box line width in pixels
        font_size: Label font size

    Returns:
        Modified PIL Image
    """
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Color palette for different classes (used when show_track_ids=False)
    _class_colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "lime",
    ]

    for inst in instances:
        if inst.bbox is None:
            continue

        x1, y1, x2, y2 = inst.bbox

        # Choose color: per-track deterministic color when tracking, else class palette
        if show_track_ids and inst.track_id is not None:
            color = _get_track_color(inst.track_id)
        else:
            color = _class_colors[inst.label % len(_class_colors)]

        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        # Draw label
        if show_labels or show_scores or (show_track_ids and inst.track_id is not None):
            parts = []
            if show_track_ids and inst.track_id is not None:
                parts.append(f"#{inst.track_id}")
            if show_labels:
                parts.append(inst.label_name or f"class_{inst.label}")
            if show_scores:
                parts.append(f"{inst.score:.2f}")

            label_text = " ".join(parts)

            # Get text bounding box
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle for text
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)

            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    return image


def export_ocr_image(
    result: OCRResult,
    output_path: str | Path,
    source_image: str | Path | Image.Image | np.ndarray | None = None,
    box_color: tuple[int, int, int] = (0, 200, 100),
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 14,
    show_scores: bool = True,
    **kwargs,
) -> None:
    """Render OCR text regions as bounding boxes with labels on the source image.

    Uses PIL only (no matplotlib) for consistency with other exporters.

    Args:
        result: OCRResult to visualize.
        output_path: Save path (.png / .jpg).
        source_image: PIL Image, numpy array, or file path. If None, uses
            ``result.meta['input_path']`` when available; otherwise creates a
            blank 800×600 canvas.
        box_color: RGB colour for bounding boxes. Default: (0, 200, 100).
        text_color: RGB colour for label text. Default: (255, 255, 255).
        font_size: Font size for rendered labels.
        show_scores: Append confidence score to each label. Default: True.
        **kwargs: Ignored – accepted for forward-compatibility.

    Raises:
        OSError: If saving the image fails.

    Examples:
        >>> result = mata.run("ocr", "document.jpg")
        >>> result.save("overlay.png", source_image="document.jpg")

        >>> export_ocr_image(result, "overlay.png")          # blank canvas
        >>> export_ocr_image(result, "overlay.png", source_image=pil_img)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Resolve source image
    # ------------------------------------------------------------------
    if source_image is None:
        # Try to pull from result metadata (stored by mata.run / adapter)
        if hasattr(result, "meta") and "input_path" in result.meta:
            source_image = result.meta["input_path"]

    if source_image is None:
        canvas = Image.new("RGB", (800, 600), color=(240, 240, 240))
    elif isinstance(source_image, np.ndarray):
        canvas = Image.fromarray(source_image).convert("RGB")
    elif isinstance(source_image, Image.Image):
        canvas = source_image.convert("RGB").copy()
    else:
        canvas = Image.open(source_image).convert("RGB")

    canvas = canvas.copy()

    # ------------------------------------------------------------------
    # 2. Draw overlays
    # ------------------------------------------------------------------
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    for region in result.regions:
        if region.bbox is None:
            continue

        x1, y1, x2, y2 = (
            float(region.bbox[0]),
            float(region.bbox[1]),
            float(region.bbox[2]),
            float(region.bbox[3]),
        )

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Label text
        label = (region.text or "")[:40]
        if show_scores:
            label = f"{label} ({region.score:.2f})"

        # Position label just above the box; clamp so it stays on-canvas
        text_y = max(0.0, y1 - font_size - 4)
        text_bbox = draw.textbbox((x1, text_y), label, font=font)
        bg_x2 = text_bbox[2] + 2
        bg_y2 = text_bbox[3] + 2
        draw.rectangle([x1 - 1, text_y - 1, bg_x2, bg_y2], fill=box_color)
        draw.text((x1, text_y), label, fill=text_color, font=font)

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    try:
        canvas.save(output_path)
        logger.info(f"Exported OCR overlay to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save OCR overlay to {output_path}: {e}")
        raise OSError(f"Failed to export OCR image: {e}") from e
