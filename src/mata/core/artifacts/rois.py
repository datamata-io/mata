"""ROIs (Regions of Interest) artifact for graph system.

Provides cropped image regions extracted from detections for downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PILImage = None  # type: ignore

from mata.core.artifacts.base import Artifact


def _generate_roi_id(index: int) -> str:
    """Generate stable ROI instance ID."""
    return f"roi_{index:04d}"


@dataclass(frozen=True)
class ROIs(Artifact):
    """Regions of Interest artifact for cropped image regions.

    Contains cropped image patches extracted from detections, typically used for
    downstream processing (e.g., classification, pose estimation, re-identification).
    Each ROI maintains mapping to source detection via instance_ids.

    Attributes:
        roi_images: List of cropped images (PIL Image or numpy array)
        instance_ids: Stable string identifiers mapping to source detections
        source_boxes: Original bbox locations in source image (x1, y1, x2, y2)
        meta: Optional metadata dictionary (should include source image dimensions)

    Examples:
        >>> # Create ROIs from detection boxes
        >>> from PIL import Image
        >>>
        >>> source_img = Image.open("image.jpg")
        >>> boxes = [(10, 20, 100, 150), (200, 300, 280, 380)]
        >>>
        >>> # Crop regions
        >>> rois = [source_img.crop(box) for box in boxes]
        >>>
        >>> roi_artifact = ROIs(
        ...     roi_images=rois,
        ...     source_boxes=boxes,
        ...     meta={"source_width": 640, "source_height": 480}
        ... )
        >>>
        >>> # Access ROIs
        >>> for roi_id, roi_img in zip(roi_artifact.instance_ids, roi_artifact.roi_images):
        ...     # Process each ROI
        ...     roi_img.save(f"crop_{roi_id}.jpg")
    """

    roi_images: list[PILImage.Image | np.ndarray] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    source_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-generate IDs if missing."""
        # Auto-generate instance_ids if missing
        if len(self.roi_images) > 0 and len(self.instance_ids) == 0:
            object.__setattr__(self, "instance_ids", [_generate_roi_id(i) for i in range(len(self.roi_images))])

        # Validate lengths match
        if len(self.roi_images) != len(self.instance_ids):
            raise ValueError(
                f"roi_images and instance_ids length mismatch: " f"{len(self.roi_images)} vs {len(self.instance_ids)}"
            )

        if len(self.roi_images) != len(self.source_boxes):
            raise ValueError(
                f"roi_images and source_boxes length mismatch: " f"{len(self.roi_images)} vs {len(self.source_boxes)}"
            )

        # Validate ROI types
        for i, roi in enumerate(self.roi_images):
            if PIL_AVAILABLE and isinstance(roi, PILImage.Image):
                continue
            elif NUMPY_AVAILABLE and isinstance(roi, np.ndarray):
                if roi.ndim not in [2, 3]:
                    raise ValueError(f"ROI {i}: numpy array must be 2D or 3D, got shape {roi.shape}")
            else:
                raise TypeError(f"ROI {i} must be PIL Image or numpy array, got {type(roi)}")

        # Validate source boxes
        for i, box in enumerate(self.source_boxes):
            if len(box) != 4:
                raise ValueError(f"source_box {i} must have 4 values (x1, y1, x2, y2), got {len(box)}")
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"source_box {i} has invalid coordinates: ({x1}, {y1}, {x2}, {y2})")

    def get_roi_sizes(self) -> list[tuple[int, int]]:
        """Get dimensions of all ROIs as (width, height).

        Returns:
            List of (width, height) tuples
        """
        sizes = []
        for roi in self.roi_images:
            if PIL_AVAILABLE and isinstance(roi, PILImage.Image):
                sizes.append(roi.size)  # (width, height)
            elif NUMPY_AVAILABLE and isinstance(roi, np.ndarray):
                if roi.ndim == 2:
                    h, w = roi.shape
                else:  # 3D (H, W, C)
                    h, w = roi.shape[:2]
                sizes.append((w, h))
            else:
                sizes.append((0, 0))
        return sizes

    def get_roi_areas(self) -> list[int]:
        """Get areas of all ROIs in pixels.

        Returns:
            List of areas (width * height)
        """
        sizes = self.get_roi_sizes()
        return [w * h for w, h in sizes]

    def filter_by_size(self, min_width: int = 0, min_height: int = 0) -> ROIs:
        """Filter ROIs by minimum dimensions.

        Args:
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels

        Returns:
            New ROIs artifact with filtered regions
        """
        sizes = self.get_roi_sizes()

        filtered_rois = []
        filtered_ids = []
        filtered_boxes = []

        for roi, roi_id, box, (w, h) in zip(self.roi_images, self.instance_ids, self.source_boxes, sizes):
            if w >= min_width and h >= min_height:
                filtered_rois.append(roi)
                filtered_ids.append(roi_id)
                filtered_boxes.append(box)

        return ROIs(roi_images=filtered_rois, instance_ids=filtered_ids, source_boxes=filtered_boxes, meta=self.meta)

    def to_numpy_list(self) -> list[np.ndarray]:
        """Convert all ROIs to numpy arrays.

        Returns:
            List of numpy arrays (H, W) or (H, W, C)

        Raises:
            ImportError: If numpy not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for array conversion")

        arrays = []
        for roi in self.roi_images:
            if isinstance(roi, np.ndarray):
                arrays.append(roi)
            elif PIL_AVAILABLE and isinstance(roi, PILImage.Image):
                arrays.append(np.array(roi))
            else:
                raise TypeError(f"Cannot convert {type(roi)} to numpy array")

        return arrays

    def to_pil_list(self) -> list[PILImage.Image]:
        """Convert all ROIs to PIL Images.

        Returns:
            List of PIL Image objects

        Raises:
            ImportError: If PIL not available
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow required for PIL Image conversion")

        images = []
        for roi in self.roi_images:
            if isinstance(roi, PILImage.Image):
                images.append(roi)
            elif NUMPY_AVAILABLE and isinstance(roi, np.ndarray):
                # Convert numpy to PIL
                if roi.dtype == bool:
                    roi = (roi * 255).astype(np.uint8)
                elif roi.dtype in [np.float32, np.float64]:
                    # Assume normalized [0, 1]
                    roi = (roi * 255).astype(np.uint8)
                images.append(PILImage.fromarray(roi))
            else:
                raise TypeError(f"Cannot convert {type(roi)} to PIL Image")

        return images

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        ROI images are converted to lists for JSON serialization.
        """
        # Convert images to serializable format
        serialized_rois = []
        for roi in self.roi_images:
            if PIL_AVAILABLE and isinstance(roi, PILImage.Image):
                # Convert PIL to numpy then to list
                serialized_rois.append(
                    {"type": "pil", "mode": roi.mode, "size": roi.size, "data": np.array(roi).tolist()}
                )
            elif NUMPY_AVAILABLE and isinstance(roi, np.ndarray):
                serialized_rois.append(
                    {"type": "numpy", "shape": list(roi.shape), "dtype": str(roi.dtype), "data": roi.tolist()}
                )
            else:
                raise TypeError(f"Cannot serialize ROI type: {type(roi)}")

        return {
            "roi_images": serialized_rois,
            "instance_ids": self.instance_ids,
            "source_boxes": [list(box) for box in self.source_boxes],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ROIs:
        """Create from dictionary representation.

        Args:
            data: Dictionary with roi_images, instance_ids, source_boxes, and meta

        Returns:
            ROIs artifact
        """
        # Reconstruct ROI images
        roi_images = []
        for roi_data in data["roi_images"]:
            if roi_data["type"] == "numpy" and NUMPY_AVAILABLE:
                arr = np.array(roi_data["data"], dtype=roi_data["dtype"])
                roi_images.append(arr)
            elif roi_data["type"] == "pil" and PIL_AVAILABLE:
                arr = np.array(roi_data["data"], dtype=np.uint8)
                img = PILImage.fromarray(arr, mode=roi_data["mode"])
                roi_images.append(img)
            else:
                raise ImportError(f"Cannot reconstruct ROI type '{roi_data['type']}' - " "missing required libraries")

        source_boxes = [tuple(box) for box in data["source_boxes"]]

        return cls(
            roi_images=roi_images,
            instance_ids=data.get("instance_ids", []),
            source_boxes=source_boxes,
            meta=data.get("meta", {}),
        )

    def validate(self) -> None:
        """Validate ROIs artifact.

        Raises:
            ValueError: If validation fails
        """
        if len(self.roi_images) == 0:
            raise ValueError("ROIs artifact must contain at least one ROI")

        if len(self.roi_images) != len(self.source_boxes):
            raise ValueError("roi_images and source_boxes must have same length")

        # Validate source boxes
        for i, box in enumerate(self.source_boxes):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"source_box {i} has invalid coordinates")
