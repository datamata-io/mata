"""Mask refinement node for morphological operations.

Applies morphological operations (close, open, dilate, erode) to mask artifacts
using OpenCV to improve mask quality and remove noise.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.masks import Masks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
    np = None  # type: ignore


class RefineMask(Node):
    """Apply morphological operations to masks.

    Performs morphological operations on binary mask data to refine mask shapes,
    remove noise, fill gaps, or separate connected components. All operations
    are applied in-place to binary representations and then converted back to
    the original mask format.

    The node processes each mask in the input Masks artifact independently,
    maintaining instance IDs and metadata preservation.

    Args:
        src: Name of the input masks artifact (default: "masks").
        out: Name of the output masks artifact (default: "masks_ref").
        method: Morphological operation type:
            - "morph_close": Fill small gaps and holes
            - "morph_open": Remove small noise and separate objects
            - "dilate": Expand mask boundaries
            - "erode": Shrink mask boundaries
        radius: Kernel radius for morphological operations (default: 3).
            Final kernel size will be (2*radius+1, 2*radius+1).
        name: Optional human-readable node name.

    Inputs:
        masks (Masks): Input segmentation masks to refine.

    Outputs:
        masks (Masks): Refined segmentation masks.

    Example:
        ```python
        from mata.nodes import RefineMask
        from mata.core.graph import Graph

        # Close small gaps in masks
        refine = RefineMask(
            src="raw_masks",
            out="clean_masks",
            method="morph_close",
            radius=5
        )

        graph = Graph("mask_cleanup")
        graph.add(refine, inputs={"masks": "raw_masks"})
        ```

    Note:
        Requires OpenCV (cv2) for morphological operations. The node will automatically
        convert between mask formats (RLE ↔ binary ↔ polygon) as needed while
        preserving the original format in the output.
    """

    inputs = {"masks": Masks}
    outputs = {"masks": Masks}

    def __init__(
        self,
        src: str = "masks",
        out: str = "masks_ref",
        method: str = "morph_close",
        radius: int = 3,
        name: str | None = None,
    ):
        super().__init__(name=name or f"RefineMask({method})")
        self.src = src
        self.out = out
        self.method = method
        self.radius = radius

        # Validate method
        valid_methods = {"morph_close", "morph_open", "dilate", "erode"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

        # Check OpenCV availability
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for mask refinement operations. " "Install with: pip install opencv-python"
            )

    def run(self, ctx: ExecutionContext, masks: Masks) -> dict[str, Artifact]:
        """Apply morphological operations to all masks.

        Args:
            ctx: Execution context (unused)
            masks: Input masks artifact

        Returns:
            Dictionary with refined masks under the output key

        Raises:
            ImportError: If OpenCV not available
            ValueError: If radius is invalid or mask format unsupported
        """
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got: {self.radius}")

        # Convert all masks to binary for processing
        try:
            binary_masks = masks.to_binary()
        except Exception as e:
            raise ValueError(f"Failed to convert masks to binary format: {e}")

        # Apply morphological operations
        refined_instances = []
        for i, (inst, binary_mask) in enumerate(zip(masks.instances, binary_masks)):
            try:
                # Apply morphological operation
                refined_binary = self._apply_morphology(binary_mask, self.method, self.radius)

                # Convert back to original format and update instance
                if inst.is_rle():
                    from mata.core.artifacts.masks import _binary_to_rle

                    refined_mask = _binary_to_rle(refined_binary)
                    refined_inst = replace(inst, mask=refined_mask)
                elif inst.is_binary():
                    refined_inst = replace(inst, mask=refined_binary)
                elif inst.is_polygon():
                    # Binary → Polygon conversion
                    from mata.core.mask_utils import binary_mask_to_polygon

                    polygons = binary_mask_to_polygon(refined_binary, tolerance=2.0, min_area=10)
                    refined_polygon = polygons[0] if polygons else []
                    refined_inst = replace(inst, mask=refined_polygon)
                else:
                    raise ValueError(f"Unsupported mask format: {type(inst.mask)}")

                refined_instances.append(refined_inst)

            except Exception as e:
                # Log error and preserve original instance
                import warnings

                warnings.warn(f"Failed to refine mask {i}: {e}. Preserving original mask.", UserWarning)
                refined_instances.append(inst)

        # Create refined Masks artifact
        refined_masks = Masks(
            instances=refined_instances, instance_ids=masks.instance_ids.copy(), meta=masks.meta.copy()
        )

        return {self.out: refined_masks}

    def _apply_morphology(self, mask: np.ndarray, method: str, radius: int) -> np.ndarray:
        """Apply morphological operation to binary mask.

        Args:
            mask: Binary numpy array of shape (H, W)
            method: Morphological operation type
            radius: Kernel radius

        Returns:
            Binary numpy array with applied operation
        """
        # Create elliptical kernel
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Ensure mask is uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8)

        # Apply operation
        if method == "morph_close":
            result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        elif method == "morph_open":
            result = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        elif method == "dilate":
            result = cv2.dilate(mask_uint8, kernel, iterations=1)
        elif method == "erode":
            result = cv2.erode(mask_uint8, kernel, iterations=1)
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Unsupported morphology method: {method}")

        # Convert back to boolean
        return result.astype(bool)
