"""DepthMap artifact for graph system.

Wraps DepthResult for typed graph wiring, providing immutable
depth estimation results with serialization and visualization helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMPY_AVAILABLE = False

from mata.core.artifacts.base import Artifact
from mata.core.types import DepthResult


@dataclass(frozen=True)
class DepthMap(Artifact):
    """Depth estimation artifact for graph wiring.

    Wraps a depth map (2-D float array) so depth output can
    participate in the strongly-typed graph system.

    Attributes:
        depth: Raw depth map as float array ``(H, W)``.
        normalized: Optional normalized depth map in ``[0, 1]``.
        meta: Arbitrary metadata (model info, timing, etc.).

    Example:
        ```python
        import numpy as np
        from mata.core.artifacts.depth_map import DepthMap

        dm = DepthMap(depth=np.random.rand(480, 640).astype(np.float32))
        dm.height, dm.width  # (480, 640)
        ```
    """

    depth: Any = None  # np.ndarray (H, W) — typed as Any to avoid frozen issues
    normalized: Any = None  # Optional np.ndarray (H, W)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_depth_result(cls, result: DepthResult) -> DepthMap:
        """Create from an existing DepthResult.

        Args:
            result: DepthResult from a depth adapter.

        Returns:
            DepthMap artifact.
        """
        return cls(
            depth=result.depth,
            normalized=result.normalized,
            meta=dict(result.meta) if result.meta else {},
        )

    def to_depth_result(self) -> DepthResult:
        """Convert back to DepthResult for adapter compatibility.

        Returns:
            DepthResult with depth array and meta.
        """
        return DepthResult(
            depth=self.depth,
            normalized=self.normalized,
            meta=dict(self.meta) if self.meta else {},
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        """Height of the depth map in pixels."""
        if self.depth is not None and NUMPY_AVAILABLE and isinstance(self.depth, np.ndarray):
            return int(self.depth.shape[0])
        return 0

    @property
    def width(self) -> int:
        """Width of the depth map in pixels."""
        if self.depth is not None and NUMPY_AVAILABLE and isinstance(self.depth, np.ndarray):
            return int(self.depth.shape[1])
        return 0

    @property
    def shape(self) -> tuple:
        """Shape of the depth map ``(H, W)``."""
        if self.depth is not None and NUMPY_AVAILABLE and isinstance(self.depth, np.ndarray):
            return tuple(self.depth.shape)
        return (0, 0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""

        def _ser(arr: Any) -> dict[str, Any] | None:
            if arr is None:
                return None
            if NUMPY_AVAILABLE and isinstance(arr, np.ndarray):
                return {
                    "data": arr.tolist(),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
            return {"data": arr}

        return {
            "depth": _ser(self.depth),
            "normalized": _ser(self.normalized),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthMap:
        """Deserialize from dictionary."""

        def _deser(info: Any) -> Any:
            if info is None:
                return None
            if not isinstance(info, dict) or "data" not in info:
                return info
            if NUMPY_AVAILABLE and "shape" in info:
                dtype = info.get("dtype", "float32")
                return np.array(info["data"], dtype=dtype).reshape(info["shape"])
            return info["data"]

        return cls(
            depth=_deser(data.get("depth")),
            normalized=_deser(data.get("normalized")),
            meta=data.get("meta", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    def validate(self) -> None:
        """Validate depth map structure."""
        if self.depth is None:
            raise ValueError("Depth map data is required (got None)")
        if NUMPY_AVAILABLE and isinstance(self.depth, np.ndarray):
            if self.depth.ndim != 2:
                raise ValueError(f"Depth map must be 2D array, got shape {self.depth.shape}")
        if self.normalized is not None and NUMPY_AVAILABLE and isinstance(self.normalized, np.ndarray):
            if self.normalized.ndim != 2:
                raise ValueError(f"Normalized depth must be 2D array, got shape {self.normalized.shape}")

    def __repr__(self) -> str:
        return f"DepthMap(shape={self.shape}, meta_keys={list(self.meta.keys())})"
