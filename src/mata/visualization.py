"""Visualization utilities for segmentation masks."""

from __future__ import annotations

import warnings
from pathlib import Path

from mata.core.logging import get_logger
from mata.core.types import Instance, SegmentMask, SegmentResult, VisionResult

logger = get_logger(__name__)


def _get_track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic, visually distinct RGB color for a track ID.

    Uses MD5 to derive a hue, then converts from HSV (fixed high saturation
    and value) to RGB so colors are always bright and saturated.
    """
    import colorsys
    import hashlib

    h = hashlib.md5(str(track_id).encode()).hexdigest()
    hue = int(h[:4], 16) / 65535.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


# Check for optional dependencies
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available. Visualization features disabled.", ImportWarning, stacklevel=2)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL (Pillow) not available. Install with: pip install Pillow", ImportWarning, stacklevel=2)

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "matplotlib is not installed — matplotlib-based visualization unavailable. "
        "Install with: pip install mata[viz]",
        ImportWarning,
        stacklevel=2,
    )


# Color palette for consistent visualization (COCO-style colors)
COCO_COLORS = [
    (220, 20, 60),  # Crimson
    (119, 11, 32),  # Dark red
    (0, 0, 142),  # Blue
    (0, 0, 230),  # Bright blue
    (106, 0, 228),  # Purple
    (0, 60, 100),  # Teal
    (0, 80, 100),  # Dark teal
    (0, 0, 70),  # Navy
    (0, 0, 192),  # Royal blue
    (250, 170, 30),  # Orange
    (100, 170, 30),  # Yellow-green
    (220, 220, 0),  # Yellow
    (175, 116, 175),  # Pink
    (250, 0, 30),  # Red
    (165, 42, 42),  # Brown
    (255, 77, 255),  # Magenta
    (0, 226, 252),  # Cyan
    (182, 182, 255),  # Light purple
    (0, 82, 0),  # Dark green
    (120, 166, 157),  # Sage
]


def _ensure_dependencies() -> bool:
    """Ensure required dependencies are available."""
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for visualization. Install with: pip install numpy")
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for visualization. Install with: pip install Pillow")
    return True


def _get_color_for_label(label: int, num_classes: int = 80) -> tuple[int, int, int]:
    """Get consistent color for a label.

    Args:
        label: Class label ID
        num_classes: Total number of classes (for cycling colors)

    Returns:
        RGB tuple (0-255)
    """
    return COCO_COLORS[label % len(COCO_COLORS)]


def _rle_to_binary(rle_mask: dict) -> np.ndarray:
    """Convert RLE mask to binary numpy array.

    Args:
        rle_mask: RLE format dict with 'size' and 'counts'

    Returns:
        Binary numpy array of shape (H, W)
    """
    try:
        from pycocotools import mask as mask_utils

        # Decode RLE to binary mask
        binary_mask = mask_utils.decode(rle_mask)
        return binary_mask.astype(bool)
    except ImportError:
        # Fallback: Manual RLE decoding (simplified, may not work for all formats)
        warnings.warn(
            "pycocotools not available. RLE decoding may be incomplete. " "Install with: pip install pycocotools",
            UserWarning,
            stacklevel=2,
        )
        # For now, return zeros
        h, w = rle_mask["size"]
        return np.zeros((h, w), dtype=bool)


def _mask_to_binary(mask: dict | np.ndarray | list, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Convert any mask format to binary numpy array.

    Args:
        mask: Mask in RLE, binary, or polygon format
        image_size: (width, height) tuple - required for polygon masks

    Returns:
        Binary numpy array of shape (H, W)
    """
    if isinstance(mask, np.ndarray):
        # Already binary
        return mask.astype(bool)
    elif isinstance(mask, dict):
        # RLE format
        return _rle_to_binary(mask)
    elif isinstance(mask, list):
        # Polygon format - convert to binary mask
        if image_size is None:
            warnings.warn(
                "Polygon mask requires image_size but none provided. Using default 640x480.", UserWarning, stacklevel=2
            )
            width, height = 640, 480
        else:
            width, height = image_size

        try:
            from mata.core.mask_utils import polygon_to_binary_mask

            # polygon_to_binary_mask expects List[List[float]] but polygon can be just List[float]
            # Wrap single polygon in a list
            polygons = [mask] if mask and isinstance(mask[0], (int, float)) else mask
            binary_mask = polygon_to_binary_mask(polygons, height, width)
            return binary_mask.astype(bool)
        except ImportError as e:
            warnings.warn(
                f"Polygon mask conversion requires OpenCV: {e}. Returning empty mask.", UserWarning, stacklevel=2
            )
            return np.zeros((height, width), dtype=bool)
    else:
        raise ValueError(f"Unknown mask format: {type(mask)}")


def _compute_bbox_from_mask(binary_mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute bounding box from binary mask.

    Args:
        binary_mask: Binary mask array (H, W)

    Returns:
        Bounding box in xyxy format (x1, y1, x2, y2) or None if mask is empty
    """
    if not NUMPY_AVAILABLE:
        return None

    # Find non-zero pixels
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (float(x_min), float(y_min), float(x_max), float(y_max))


def visualize_segmentation(
    result: SegmentResult | VisionResult,
    image: str | Path | Image.Image | np.ndarray,
    alpha: float = 0.5,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_scores: bool = True,
    show_track_ids: bool = False,
    filter_threshold: float | None = None,
    output_path: str | Path | None = None,
    backend: str = "pil",
) -> Image.Image | plt.Figure | None:
    """Visualize segmentation masks overlaid on image.

    Supports both legacy SegmentResult and new VisionResult formats.

    Args:
        result: SegmentResult or VisionResult containing masks to visualize
        image: Input image as:
            - Path to image file (str or Path)
            - PIL Image object
            - Numpy array (H, W, 3) in RGB format
        alpha: Mask transparency (0.0=invisible, 1.0=opaque)
        show_boxes: Draw bounding boxes around masks
        show_labels: Show class labels
        show_scores: Show confidence scores
        filter_threshold: Only show masks above this score (overrides result filtering)
        output_path: Optional path to save visualization
        backend: Visualization backend:
            - "pil": Use Pillow (default, lightweight)
            - "matplotlib": Use matplotlib (more features)

    Returns:
        PIL Image if backend="pil", matplotlib Figure if backend="matplotlib"

    Raises:
        ImportError: If required dependencies not available
        ValueError: If invalid backend specified

    Example:
        >>> from mata import load
        >>> from mata.visualization import visualize_segmentation
        >>>
        >>> # Load model and run inference
        >>> model = load("segment", "facebook/mask2former-swin-tiny-coco-instance")
        >>> result = model.predict("image.jpg")
        >>>
        >>> # Visualize with default settings
        >>> vis = visualize_segmentation(result, "image.jpg")
        >>> vis.show()
        >>>
        >>> # Save visualization
        >>> visualize_segmentation(
        ...     result,
        ...     "image.jpg",
        ...     alpha=0.6,
        ...     output_path="output.png"
        ... )
        >>>
        >>> # Zero-shot detection with GroundingDINO→SAM pipeline
        >>> pipeline = load("pipeline",
        ...     detector_model_id="IDEA-Research/grounding-dino-tiny",
        ...     sam_model_id="facebook/sam-vit-base")
        >>> result = pipeline.predict("image.jpg", text_prompts="car . person")
        >>> vis = visualize_segmentation(result, "image.jpg")  # Shows bbox + mask
    """
    _ensure_dependencies()

    # Validate backend
    if backend not in ("pil", "matplotlib"):
        raise ValueError(f"Invalid backend '{backend}'. Choose 'pil' or 'matplotlib'")

    if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
        warnings.warn(
            "matplotlib not available. Falling back to PIL backend. " "Install with: pip install matplotlib",
            UserWarning,
            stacklevel=2,
        )
        backend = "pil"

    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # Convert numpy to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Invalid image type: {type(image)}")

    # Handle VisionResult vs SegmentResult
    if isinstance(result, VisionResult):
        # Filter instances by threshold if specified
        instances = result.instances
        if filter_threshold is not None:
            instances = [inst for inst in instances if inst.score >= filter_threshold]

        logger.info(f"Visualizing {len(instances)} instances (VisionResult) with {backend} backend (alpha={alpha})")

        # Delegate to backend-specific implementation
        if backend == "pil":
            return _visualize_instances_pil(
                instances,
                image,
                alpha,
                show_boxes,
                show_labels,
                show_scores,
                output_path,
                show_track_ids=show_track_ids,
            )
        else:
            return _visualize_instances_matplotlib(
                instances, image, alpha, show_boxes, show_labels, show_scores, output_path
            )
    else:
        # Legacy SegmentResult
        # Filter masks if threshold specified
        if filter_threshold is not None:
            masks = result.filter_by_score(filter_threshold).masks
        else:
            masks = result.masks

        logger.info(f"Visualizing {len(masks)} masks (SegmentResult) with {backend} backend (alpha={alpha})")

        # Delegate to backend-specific implementation
        if backend == "pil":
            return _visualize_pil(masks, image, alpha, show_boxes, show_labels, show_scores, output_path)
        else:
            return _visualize_matplotlib(masks, image, alpha, show_boxes, show_labels, show_scores, output_path)


def _visualize_pil(
    masks: list[SegmentMask],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
) -> Image.Image:
    """Visualize using Pillow backend (legacy SegmentResult)."""
    # Create copy for visualization
    vis_image = image.copy()
    overlay = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load font (fallback to default if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Draw each mask
    for i, mask in enumerate(masks):
        # Get color for this label
        color = _get_color_for_label(mask.label)
        rgba_color = (*color, int(255 * alpha))

        # Convert mask to binary
        binary_mask = _mask_to_binary(mask.mask, image_size=vis_image.size)

        # Create colored mask
        mask_image = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
        mask_array = np.array(mask_image)
        mask_array[binary_mask] = rgba_color
        mask_image = Image.fromarray(mask_array, mode="RGBA")

        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, mask_image)
        # Recreate draw object after composite (composite creates new image)
        draw = ImageDraw.Draw(overlay)

        # Determine bbox to use (from mask or computed)
        bbox_to_draw = mask.bbox
        if bbox_to_draw is None and show_boxes:
            # Compute bbox from mask if not provided
            bbox_to_draw = _compute_bbox_from_mask(binary_mask)
            if bbox_to_draw is not None:
                logger.debug(f"Computed bbox from mask for segment {i}: {bbox_to_draw}")

        # Draw bounding box
        if show_boxes and bbox_to_draw is not None:
            x1, y1, x2, y2 = bbox_to_draw
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label and score
        if (show_labels or show_scores) and bbox_to_draw is not None:
            x1, y1, x2, y2 = bbox_to_draw
            label_text = ""
            if show_labels:
                label_text = mask.label_name or f"Class {mask.label}"
            if show_scores:
                label_text += f" {mask.score:.2f}"

            # Draw text background
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill="white", font=font)

    # Composite overlay onto image
    vis_image = vis_image.convert("RGBA")
    vis_image = Image.alpha_composite(vis_image, overlay)
    vis_image = vis_image.convert("RGB")

    # Save if path provided
    if output_path is not None:
        vis_image.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

    return vis_image


def _visualize_instances_pil(
    instances: list[Instance],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
    show_track_ids: bool = False,
) -> Image.Image:
    """Visualize VisionResult instances using Pillow backend.

    Handles multi-modal instances (bbox-only, mask-only, or both).
    """
    # Create copy for visualization
    vis_image = image.copy()
    overlay = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    ImageDraw.Draw(vis_image)

    # Try to load font (fallback to default if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Draw each instance
    for i, instance in enumerate(instances):
        # Get color: deterministic per-track when tracking, else per-class palette
        if show_track_ids and instance.track_id is not None:
            color = _get_track_color(instance.track_id)
        else:
            color = _get_color_for_label(instance.label)
        rgba_color = (*color, int(255 * alpha))

        binary_mask = None  # Track binary mask for bbox computation

        # Draw mask if available
        if instance.mask is not None:
            # Convert mask to binary
            if hasattr(instance.mask, "to_binary"):
                binary_mask = instance.mask.to_binary()
            else:
                binary_mask = _mask_to_binary(instance.mask, image_size=vis_image.size)

            # Create colored mask
            mask_image = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
            mask_array = np.array(mask_image)
            mask_array[binary_mask] = rgba_color
            mask_image = Image.fromarray(mask_array, mode="RGBA")

            # Composite onto overlay
            overlay = Image.alpha_composite(overlay, mask_image)
            # Recreate draw object after composite (composite creates new image)
            draw_overlay = ImageDraw.Draw(overlay)

        # Determine bbox to use (from instance or computed from mask)
        bbox_to_draw = instance.bbox
        if bbox_to_draw is None and binary_mask is not None and show_boxes:
            # Compute bbox from mask if not provided
            bbox_to_draw = _compute_bbox_from_mask(binary_mask)
            if bbox_to_draw is not None:
                logger.debug(f"Computed bbox from mask for instance {i}: {bbox_to_draw}")

        # Draw bounding box
        if show_boxes and bbox_to_draw is not None:
            x1, y1, x2, y2 = bbox_to_draw
            draw_overlay.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label and score
        if (
            show_labels or show_scores or (show_track_ids and instance.track_id is not None)
        ) and bbox_to_draw is not None:
            x1, y1, x2, y2 = bbox_to_draw
            parts = []
            if show_track_ids and instance.track_id is not None:
                parts.append(f"#{instance.track_id}")
            if show_labels:
                parts.append(instance.label_name or f"Class {instance.label}")
            if show_scores:
                score_str = f"{instance.score:.2f}" if instance.score is not None else ""
                if score_str:
                    parts.append(score_str)
            label_text = " ".join(parts)

            # Draw text background
            bbox = draw_overlay.textbbox((x1, y1 - 20), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw_overlay.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw_overlay.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    # Composite overlay onto image
    vis_image = vis_image.convert("RGBA")
    vis_image = Image.alpha_composite(vis_image, overlay)
    vis_image = vis_image.convert("RGB")

    # Save if path provided
    if output_path is not None:
        vis_image.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

    return vis_image


def _visualize_pil(
    masks: list[SegmentMask],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
) -> Image.Image:
    """Visualize using Pillow backend."""
    # Create copy for visualization
    vis_image = image.copy()
    overlay = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load font (fallback to default if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Draw each mask
    for i, mask in enumerate(masks):
        # Get color for this label
        color = _get_color_for_label(mask.label)
        rgba_color = (*color, int(255 * alpha))

        # Convert mask to binary
        binary_mask = _mask_to_binary(mask.mask, image_size=vis_image.size)

        # Create colored mask
        mask_image = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
        mask_array = np.array(mask_image)
        mask_array[binary_mask] = rgba_color
        mask_image = Image.fromarray(mask_array, mode="RGBA")

        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, mask_image)

        # Draw bounding box
        if show_boxes and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label and score
        if (show_labels or show_scores) and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            label_text = ""
            if show_labels:
                label_text = mask.label_name or f"Class {mask.label}"
            if show_scores:
                label_text += f" {mask.score:.2f}"

            # Draw text background
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill="white", font=font)

    # Composite overlay onto image
    vis_image = vis_image.convert("RGBA")
    vis_image = Image.alpha_composite(vis_image, overlay)
    vis_image = vis_image.convert("RGB")

    # Save if path provided
    if output_path is not None:
        vis_image.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

    return vis_image


def _visualize_matplotlib(
    masks: list[SegmentMask],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
) -> plt.Figure:
    """Visualize using matplotlib backend (legacy SegmentResult)."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    # Create legend entries
    legend_handles = []

    # Draw each mask
    for i, mask in enumerate(masks):
        # Get color for this label
        color = _get_color_for_label(mask.label)
        color_normalized = tuple(c / 255.0 for c in color)

        # Convert mask to binary
        binary_mask = _mask_to_binary(mask.mask, image_size=image.size)

        # Create colored mask overlay
        colored_mask = np.zeros((*binary_mask.shape, 4))
        colored_mask[binary_mask] = (*color_normalized, alpha)

        # Overlay on image
        ax.imshow(colored_mask, interpolation="nearest")

        # Draw bounding box
        if show_boxes and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_normalized, facecolor="none"
            )
            ax.add_patch(rect)

        # Draw label and score
        if (show_labels or show_scores) and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            label_text = ""
            if show_labels:
                label_text = mask.label_name or f"Class {mask.label}"
            if show_scores:
                label_text += f" {mask.score:.2f}"

            ax.text(
                x1, y1 - 5, label_text, bbox=dict(facecolor=color_normalized, alpha=0.8), color="white", fontsize=10
            )

        # Add to legend
        if show_labels:
            label_name = mask.label_name or f"Class {mask.label}"
            patch = mpatches.Patch(color=color_normalized, label=label_name)
            legend_handles.append(patch)

    # Add legend
    if show_labels and legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)

    plt.tight_layout()

    # Save if path provided
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved visualization to {output_path}")

    return fig


def _visualize_instances_matplotlib(
    instances: list[Instance],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
) -> plt.Figure:
    """Visualize VisionResult instances using matplotlib backend.

    Handles multi-modal instances (bbox-only, mask-only, or both).
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    # Create legend entries
    legend_handles = []

    # Draw each instance
    for i, instance in enumerate(instances):
        # Get color for this label
        color = _get_color_for_label(instance.label)
        color_normalized = tuple(c / 255.0 for c in color)

        # Draw mask if available
        if instance.mask is not None:
            # Convert mask to binary
            if hasattr(instance.mask, "to_binary"):
                binary_mask = instance.mask.to_binary()
            else:
                binary_mask = _mask_to_binary(instance.mask, image_size=image.size)

            # Create colored mask overlay
            colored_mask = np.zeros((*binary_mask.shape, 4))
            colored_mask[binary_mask] = (*color_normalized, alpha)

            # Overlay on image
            ax.imshow(colored_mask, interpolation="nearest")

        # Draw bounding box
        if show_boxes and instance.bbox is not None:
            x1, y1, x2, y2 = instance.bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_normalized, facecolor="none"
            )
            ax.add_patch(rect)

        # Draw label and score
        if (show_labels or show_scores) and instance.bbox is not None:
            x1, y1, x2, y2 = instance.bbox
            label_text = ""
            if show_labels:
                label_text = instance.label_name or f"Class {instance.label}"
            if show_scores:
                score_str = f" {instance.score:.2f}" if instance.score is not None else ""
                label_text += score_str

            ax.text(
                x1, y1 - 5, label_text, bbox=dict(facecolor=color_normalized, alpha=0.8), color="white", fontsize=10
            )

        # Add to legend
        if show_labels:
            label_name = instance.label_name or f"Class {instance.label}"
            patch = mpatches.Patch(color=color_normalized, label=label_name)
            legend_handles.append(patch)

    # Add legend
    if show_labels and legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)

    plt.tight_layout()

    # Save if path provided
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved visualization to {output_path}")

    return fig


def _visualize_matplotlib(
    masks: list[SegmentMask],
    image: Image.Image,
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    output_path: str | Path | None,
) -> plt.Figure:
    """Visualize using matplotlib backend."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    # Create legend entries
    legend_handles = []

    # Draw each mask
    for i, mask in enumerate(masks):
        # Get color for this label
        color = _get_color_for_label(mask.label)
        color_normalized = tuple(c / 255.0 for c in color)

        # Convert mask to binary
        binary_mask = _mask_to_binary(mask.mask, image_size=image.size)

        # Create colored mask overlay
        colored_mask = np.zeros((*binary_mask.shape, 4))
        colored_mask[binary_mask] = (*color_normalized, alpha)

        # Overlay on image
        ax.imshow(colored_mask, interpolation="nearest")

        # Draw bounding box
        if show_boxes and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_normalized, facecolor="none"
            )
            ax.add_patch(rect)

        # Draw label and score
        if (show_labels or show_scores) and mask.bbox is not None:
            x1, y1, x2, y2 = mask.bbox
            label_text = ""
            if show_labels:
                label_text = mask.label_name or f"Class {mask.label}"
            if show_scores:
                label_text += f" {mask.score:.2f}"

            ax.text(
                x1, y1 - 5, label_text, bbox=dict(facecolor=color_normalized, alpha=0.8), color="white", fontsize=10
            )

        # Add to legend
        if show_labels:
            label_name = mask.label_name or f"Class {mask.label}"
            patch = mpatches.Patch(color=color_normalized, label=label_name)
            legend_handles.append(patch)

    # Add legend
    if show_labels and legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)

    plt.tight_layout()

    # Save if path provided
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved visualization to {output_path}")

    return fig


def create_panoptic_visualization(
    result: SegmentResult,
    image: str | Path | Image.Image | np.ndarray,
    output_path: str | Path | None = None,
    show_legend: bool = True,
) -> Image.Image:
    """Create panoptic segmentation visualization with distinct colors for instances.

    Unlike standard visualization which overlays semi-transparent masks, this creates
    a dense segmentation map where each pixel is assigned to exactly one class/instance.

    Args:
        result: SegmentResult in panoptic mode (with is_stuff field)
        image: Input image
        output_path: Optional path to save visualization
        show_legend: Show legend with class names

    Returns:
        PIL Image with panoptic segmentation

    Example:
        >>> result = model.predict("image.jpg", segment_mode="panoptic")
        >>> vis = create_panoptic_visualization(result, "image.jpg")
        >>> vis.show()
    """
    _ensure_dependencies()

    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")

    # Create segmentation map
    seg_map = np.zeros((*image.size[::-1], 3), dtype=np.uint8)

    # Draw stuff classes first (background)
    stuff_masks = result.get_stuff()
    for mask in stuff_masks:
        color = _get_color_for_label(mask.label)
        binary_mask = _mask_to_binary(mask.mask, image_size=image.size)
        seg_map[binary_mask] = color

    # Draw instance classes on top
    instance_masks = result.get_instances()
    for i, mask in enumerate(instance_masks):
        # Use unique color for each instance (combine label + instance ID)
        color = _get_color_for_label(mask.label * 100 + i)
        binary_mask = _mask_to_binary(mask.mask, image_size=image.size)
        seg_map[binary_mask] = color

    # Blend with original image
    seg_image = Image.fromarray(seg_map, mode="RGB")
    blended = Image.blend(image, seg_image, alpha=0.5)

    # Save if path provided
    if output_path is not None:
        blended.save(output_path)
        logger.info(f"Saved panoptic visualization to {output_path}")

    return blended
