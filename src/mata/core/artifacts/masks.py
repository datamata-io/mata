"""Masks artifact for graph system.

Provides segmentation mask results with instance IDs and format conversions.
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

from mata.core.artifacts.base import Artifact
from mata.core.types import Instance, VisionResult

try:
    from mata.core.mask_utils import (
        binary_mask_to_polygon,
        polygon_to_binary_mask,
    )

    MASK_UTILS_AVAILABLE = True
except ImportError:
    MASK_UTILS_AVAILABLE = False

try:
    from pycocotools import mask as mask_utils_coco

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    mask_utils_coco = None  # type: ignore


def _generate_mask_id(index: int) -> str:
    """Generate stable mask instance ID."""
    return f"mask_{index:04d}"


def _rle_to_binary(rle_mask: dict, image_size: tuple | None = None) -> np.ndarray:
    """Convert RLE mask to binary numpy array.

    Args:
        rle_mask: RLE dict with 'size' and 'counts'
        image_size: Optional (H, W) tuple, uses rle['size'] if not provided

    Returns:
        Binary numpy array of shape (H, W)
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools required for RLE decoding. Install: pip install pycocotools")

    return mask_utils_coco.decode(rle_mask).astype(bool)


def _binary_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary numpy array to RLE format.

    Args:
        binary_mask: Binary array of shape (H, W)

    Returns:
        RLE dict with 'size' and 'counts'
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools required for RLE encoding. Install: pip install pycocotools")

    if not NUMPY_AVAILABLE:
        raise ImportError("numpy required for mask operations")

    # Ensure uint8 format for pycocotools
    mask_uint8 = binary_mask.astype(np.uint8)

    # Encode (requires Fortran order)
    rle = mask_utils_coco.encode(np.asfortranarray(mask_uint8))

    # Decode bytes to string for JSON serialization
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return rle


@dataclass(frozen=True)
class Masks(Artifact):
    """Segmentation masks artifact with instance IDs.

    Wraps Instance objects containing mask data (RLE/polygon/binary format)
    and provides conversions between formats. All masks are associated with
    stable instance IDs for graph wiring and cross-artifact alignment.

    Attributes:
        instances: List of Instance objects with mask data
        instance_ids: Stable string identifiers for instances
        meta: Optional metadata dictionary

    Examples:
        >>> # Create from VisionResult
        >>> result = VisionResult(instances=[...])  # Instances with masks
        >>> masks = Masks.from_vision_result(result)
        >>>
        >>> # Convert mask formats
        >>> rle_masks = masks.to_rle()  # List of RLE dicts
        >>> polygons = masks.to_polygons()  # List of polygon lists
        >>> binary_arrays = masks.to_binary()  # List of numpy arrays
        >>>
        >>> # Access data
        >>> for mask_id, inst in zip(masks.instance_ids, masks.instances):
        ...     print(f"Mask {mask_id}: label={inst.label_name}")
    """

    instances: list[Instance] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-generate IDs if missing."""
        # Auto-generate instance_ids if missing
        if len(self.instances) > 0 and len(self.instance_ids) == 0:
            object.__setattr__(self, "instance_ids", [_generate_mask_id(i) for i in range(len(self.instances))])

        # Validate lengths match
        if len(self.instances) != len(self.instance_ids):
            raise ValueError(
                f"instances and instance_ids length mismatch: " f"{len(self.instances)} vs {len(self.instance_ids)}"
            )

        # Validate all instances have masks
        for i, inst in enumerate(self.instances):
            if inst.mask is None:
                raise ValueError(f"Instance at index {i} has no mask data")

    @classmethod
    def from_vision_result(cls, result: VisionResult) -> Masks:
        """Convert VisionResult to Masks artifact.

        Extracts only instances that have mask data.

        Args:
            result: VisionResult with instances containing masks

        Returns:
            Masks artifact with auto-generated IDs

        Raises:
            ValueError: If no instances have masks
        """
        # Filter instances with masks
        masked_instances = [inst for inst in result.instances if inst.mask is not None]

        if not masked_instances:
            raise ValueError("VisionResult contains no instances with masks")

        # Preserve instance_ids from meta if available
        instance_ids = []
        if "instance_ids" in result.meta:
            # Filter IDs corresponding to masked instances
            all_ids = result.meta["instance_ids"]
            for i, inst in enumerate(result.instances):
                if inst.mask is not None and i < len(all_ids):
                    instance_ids.append(all_ids[i])

        return cls(
            instances=masked_instances,
            instance_ids=instance_ids,  # Will auto-generate if empty
            meta=result.meta.copy() if result.meta else {},
        )

    def to_vision_result(self) -> VisionResult:
        """Convert back to VisionResult.

        Returns:
            VisionResult with instances and IDs in meta
        """
        meta = self.meta.copy()
        meta["instance_ids"] = self.instance_ids

        return VisionResult(instances=self.instances, meta=meta)

    def to_rle(self) -> list[dict[str, Any]]:
        """Convert all masks to RLE format.

        Returns:
            List of RLE dicts with 'size' and 'counts'

        Raises:
            ImportError: If pycocotools not available
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools required for RLE conversion. Install: pip install pycocotools")

        rle_masks = []
        for inst in self.instances:
            if inst.is_rle():
                # Already RLE
                rle_masks.append(inst.mask)
            elif inst.is_binary():
                # Binary → RLE
                rle_masks.append(_binary_to_rle(inst.mask))
            elif inst.is_polygon():
                # Polygon → Binary → RLE
                if not MASK_UTILS_AVAILABLE:
                    raise ImportError("mask_utils required for polygon conversion")
                # Extract image size from meta or use default
                h, w = self.meta.get("image_height", 1024), self.meta.get("image_width", 1024)
                binary = polygon_to_binary_mask([inst.mask], h, w)  # type: ignore[list-item]
                rle_masks.append(_binary_to_rle(binary))
            else:
                raise ValueError(f"Unknown mask format for instance: {type(inst.mask)}")

        return rle_masks  # type: ignore[return-value]

    def to_polygons(self, tolerance: float = 2.0, min_area: int = 10) -> list[list[float]]:
        """Convert all masks to polygon format.

        Args:
            tolerance: Polygon approximation tolerance (epsilon for cv2.approxPolyDP)
            min_area: Minimum contour area in pixels

        Returns:
            List of polygon lists, where each polygon is [x1, y1, x2, y2, ...]

        Raises:
            ImportError: If opencv or mask_utils not available
        """
        if not MASK_UTILS_AVAILABLE:
            raise ImportError(
                "mask_utils (opencv) required for polygon conversion. " "Install: pip install opencv-python"
            )

        polygons = []
        for inst in self.instances:
            if inst.is_polygon():
                # Already polygon
                polygons.append(inst.mask)
            elif inst.is_binary():
                # Binary → Polygon
                poly_list = binary_mask_to_polygon(inst.mask, tolerance, min_area)
                # Return first polygon if multiple regions
                polygons.append(poly_list[0] if poly_list else [])
            elif inst.is_rle():
                # RLE → Binary → Polygon
                binary = _rle_to_binary(inst.mask)
                poly_list = binary_mask_to_polygon(binary, tolerance, min_area)
                polygons.append(poly_list[0] if poly_list else [])
            else:
                raise ValueError(f"Unknown mask format for instance: {type(inst.mask)}")

        return polygons  # type: ignore[return-value]

    def to_binary(self) -> list[np.ndarray]:
        """Convert all masks to binary numpy arrays.

        Returns:
            List of binary numpy arrays of shape (H, W) with dtype bool

        Raises:
            ImportError: If numpy or required conversion libraries not available
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for binary mask conversion")

        binary_masks = []
        for inst in self.instances:
            if inst.is_binary():
                # Already binary
                binary_masks.append(inst.mask)
            elif inst.is_rle():
                # RLE → Binary
                binary_masks.append(_rle_to_binary(inst.mask))
            elif inst.is_polygon():
                # Polygon → Binary
                if not MASK_UTILS_AVAILABLE:
                    raise ImportError("mask_utils (opencv) required for polygon conversion")
                # Extract image size from meta
                h, w = self.meta.get("image_height", 1024), self.meta.get("image_width", 1024)
                binary = polygon_to_binary_mask([inst.mask], h, w)  # type: ignore[list-item]
                binary_masks.append(binary)
            else:
                raise ValueError(f"Unknown mask format for instance: {type(inst.mask)}")

        return binary_masks  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "instances": [inst.to_dict() for inst in self.instances],
            "instance_ids": self.instance_ids,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Masks:
        """Create from dictionary representation.

        Args:
            data: Dictionary with instances, instance_ids, and meta

        Returns:
            Masks artifact
        """
        # Reconstruct instances from dict
        instances = []
        for inst_data in data["instances"]:
            # Reconstruct mask
            mask = None
            mask_info = inst_data.get("mask")
            if mask_info:
                if mask_info["format"] == "rle":
                    mask = mask_info["data"]
                elif mask_info["format"] == "binary" and NUMPY_AVAILABLE:
                    mask = np.array(mask_info["data"], dtype=bool)
                elif mask_info["format"] == "polygon":
                    mask = mask_info["data"]
                else:
                    mask = mask_info["data"]

            instances.append(
                Instance(
                    bbox=tuple(inst_data["bbox"]) if inst_data.get("bbox") else None,
                    mask=mask,
                    score=inst_data["score"],
                    label=inst_data["label"],
                    label_name=inst_data.get("label_name"),
                    area=inst_data.get("area"),
                    is_stuff=inst_data.get("is_stuff"),
                )
            )

        return cls(instances=instances, instance_ids=data.get("instance_ids", []), meta=data.get("meta", {}))

    def validate(self) -> None:
        """Validate masks artifact.

        Raises:
            ValueError: If validation fails
        """
        if not self.instances:
            raise ValueError("Masks artifact must contain at least one instance")

        for i, inst in enumerate(self.instances):
            if inst.mask is None:
                raise ValueError(f"Instance {i} has no mask data")
            if inst.score < 0 or inst.score > 1:
                raise ValueError(f"Instance {i} has invalid score: {inst.score}")
