"""Keypoints artifact for graph system.

Provides keypoint detection results with instance IDs and skeleton information.
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


def _generate_keypoint_id(index: int) -> str:
    """Generate stable keypoint instance ID."""
    return f"kp_{index:04d}"


@dataclass(frozen=True)
class Keypoints(Artifact):
    """Keypoint detection artifact with instance IDs.

    Stores keypoint detections for pose estimation or other keypoint-based tasks.
    Each instance contains a set of keypoints with (x, y, score) format.

    Attributes:
        keypoints: List of numpy arrays of shape (num_keypoints, 3) with [x, y, score]
            - x, y: Pixel coordinates (float)
            - score: Keypoint visibility/confidence score [0.0, 1.0]
        instance_ids: Stable string identifiers for instances
        skeleton: Optional list of bone connections as (start_idx, end_idx) tuples
            - Defines which keypoints connect to form skeleton structure
            - Example: [(0, 1), (1, 2)] connects keypoint 0→1→2
        meta: Optional metadata dictionary

    Examples:
        >>> # Create keypoints for 2 people with 17 COCO keypoints each
        >>> kp1 = np.array([[100, 200, 0.9], [110, 220, 0.8], ...])  # (17, 3)
        >>> kp2 = np.array([[300, 250, 0.95], [310, 270, 0.85], ...])  # (17, 3)
        >>>
        >>> # COCO skeleton connections
        >>> skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), ...]  # Bone pairs
        >>>
        >>> keypoints = Keypoints(
        ...     keypoints=[kp1, kp2],
        ...     skeleton=skeleton,
        ...     meta={"dataset": "coco", "num_keypoints": 17}
        ... )
        >>>
        >>> # Access data
        >>> for kp_id, kp in zip(keypoints.instance_ids, keypoints.keypoints):
        ...     visible = kp[kp[:, 2] > 0.5]  # Filter by score
        ...     print(f"Instance {kp_id}: {len(visible)} visible keypoints")
    """

    keypoints: list[np.ndarray] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    skeleton: list[tuple[int, int]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-generate IDs if missing."""
        if not NUMPY_AVAILABLE:
            if len(self.keypoints) > 0:
                raise ImportError("numpy required for keypoint operations")

        # Auto-generate instance_ids if missing
        if len(self.keypoints) > 0 and len(self.instance_ids) == 0:
            object.__setattr__(self, "instance_ids", [_generate_keypoint_id(i) for i in range(len(self.keypoints))])

        # Validate lengths match
        if len(self.keypoints) != len(self.instance_ids):
            raise ValueError(
                f"keypoints and instance_ids length mismatch: " f"{len(self.keypoints)} vs {len(self.instance_ids)}"
            )

        # Validate keypoint array shapes
        for i, kp in enumerate(self.keypoints):
            if not isinstance(kp, np.ndarray):
                raise TypeError(f"Keypoint {i} must be numpy array, got {type(kp)}")

            if kp.ndim != 2:
                raise ValueError(f"Keypoint {i} must be 2D array (num_keypoints, 3), got shape {kp.shape}")

            if kp.shape[1] != 3:
                raise ValueError(f"Keypoint {i} must have 3 columns [x, y, score], got {kp.shape[1]} columns")

            # Validate scores in [0, 1]
            scores = kp[:, 2]
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError(
                    f"Keypoint {i} has scores outside [0, 1] range: " f"min={scores.min():.3f}, max={scores.max():.3f}"
                )

        # Validate skeleton if provided
        if self.skeleton is not None:
            num_kps = self.keypoints[0].shape[0] if self.keypoints else 0
            for i, (start, end) in enumerate(self.skeleton):
                if start < 0 or start >= num_kps:
                    raise ValueError(f"Skeleton bone {i}: start index {start} out of range [0, {num_kps})")
                if end < 0 or end >= num_kps:
                    raise ValueError(f"Skeleton bone {i}: end index {end} out of range [0, {num_kps})")

    def filter_by_visibility(self, threshold: float = 0.5) -> Keypoints:
        """Create new Keypoints with low-confidence keypoints zeroed out.

        Keypoints with score < threshold will have their scores set to 0.0
        (standard convention for invisible/occluded keypoints).

        Args:
            threshold: Minimum visibility score [0.0, 1.0]

        Returns:
            New Keypoints artifact with filtered keypoints
        """
        filtered_kps = []
        for kp in self.keypoints:
            kp_copy = kp.copy()
            mask = kp_copy[:, 2] < threshold
            kp_copy[mask, 2] = 0.0  # Set invisible keypoints to 0
            filtered_kps.append(kp_copy)

        return Keypoints(keypoints=filtered_kps, instance_ids=self.instance_ids, skeleton=self.skeleton, meta=self.meta)

    def get_visible_keypoints(self, instance_idx: int, threshold: float = 0.5) -> np.ndarray:
        """Get visible keypoints for a specific instance.

        Args:
            instance_idx: Index of instance
            threshold: Minimum visibility score

        Returns:
            Array of visible keypoints (num_visible, 3)
        """
        if instance_idx < 0 or instance_idx >= len(self.keypoints):
            raise IndexError(f"Instance index {instance_idx} out of range [0, {len(self.keypoints)})")

        kp = self.keypoints[instance_idx]
        return kp[kp[:, 2] >= threshold]

    def count_visible(self, threshold: float = 0.5) -> list[int]:
        """Count visible keypoints per instance.

        Args:
            threshold: Minimum visibility score

        Returns:
            List of counts (one per instance)
        """
        counts = []
        for kp in self.keypoints:
            count = np.sum(kp[:, 2] >= threshold)
            counts.append(int(count))
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "keypoints": [kp.tolist() for kp in self.keypoints],
            "instance_ids": self.instance_ids,
            "skeleton": self.skeleton,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keypoints:
        """Create from dictionary representation.

        Args:
            data: Dictionary with keypoints, instance_ids, skeleton, and meta

        Returns:
            Keypoints artifact
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for keypoint operations")

        keypoints = [np.array(kp, dtype=np.float32) for kp in data["keypoints"]]

        return cls(
            keypoints=keypoints,
            instance_ids=data.get("instance_ids", []),
            skeleton=data.get("skeleton"),
            meta=data.get("meta", {}),
        )

    def validate(self) -> None:
        """Validate keypoints artifact.

        Raises:
            ValueError: If validation fails
        """
        if not self.keypoints:
            raise ValueError("Keypoints artifact must contain at least one instance")

        # Check all instances have same number of keypoints
        num_kps = [kp.shape[0] for kp in self.keypoints]
        if len(set(num_kps)) > 1:
            raise ValueError(f"All instances must have same number of keypoints, got {num_kps}")

        # Validate scores
        for i, kp in enumerate(self.keypoints):
            scores = kp[:, 2]
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError(f"Instance {i} has invalid scores (must be in [0, 1])")
