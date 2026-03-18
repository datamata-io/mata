"""CrossMatches artifact for cross-camera re-identification results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class CrossMatch:
    """Single cross-camera match for a tracked object.

    Attributes:
        local_track_id: Track ID in the current camera.
        remote_camera_id: Camera ID where the match was found.
        remote_track_id: Track ID in the remote camera.
        similarity: Cosine similarity score [0.0, 1.0].
        remote_bbox: Optional bounding box in the remote camera (xyxy).

    Examples:
        >>> match = CrossMatch(
        ...     local_track_id=3,
        ...     remote_camera_id="cam-2",
        ...     remote_track_id=7,
        ...     similarity=0.92,
        ...     remote_bbox=(120.0, 80.0, 160.0, 200.0),
        ... )
    """

    local_track_id: int
    remote_camera_id: str
    remote_track_id: int
    similarity: float
    remote_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate CrossMatch data."""
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(
                f"similarity must be in [0.0, 1.0], got {self.similarity}"
            )
        if self.remote_bbox is not None:
            if len(self.remote_bbox) != 4:
                raise ValueError(
                    f"remote_bbox must have 4 values (x1, y1, x2, y2), "
                    f"got {len(self.remote_bbox)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "local_track_id": self.local_track_id,
            "remote_camera_id": self.remote_camera_id,
            "remote_track_id": self.remote_track_id,
            "similarity": self.similarity,
            "remote_bbox": list(self.remote_bbox) if self.remote_bbox else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossMatch:
        """Create from dictionary representation."""
        bbox = (
            tuple(data["remote_bbox"])  # type: ignore[arg-type]
            if data.get("remote_bbox")
            else None
        )
        return cls(
            local_track_id=data["local_track_id"],
            remote_camera_id=data["remote_camera_id"],
            remote_track_id=data["remote_track_id"],
            similarity=data["similarity"],
            remote_bbox=bbox,
        )


@dataclass(frozen=True)
class CrossMatches(Artifact):
    """Cross-camera re-identification results artifact.

    Carries cross-camera match results as a typed graph edge between the
    ReID node and downstream consumers (AnnotateRT, export, etc.).

    Attributes:
        matches: List of CrossMatch objects.
        camera_id: ID of the camera that produced these matches.
        meta: Optional metadata.

    Examples:
        >>> cm = CrossMatches(
        ...     matches=[
        ...         CrossMatch(
        ...             local_track_id=1,
        ...             remote_camera_id="cam-2",
        ...             remote_track_id=5,
        ...             similarity=0.88,
        ...         )
        ...     ],
        ...     camera_id="cam-1",
        ... )
        >>> len(cm)
        1
        >>> cm.has_cross_camera(1)
        True
        >>> cm.has_cross_camera(99)
        False
    """

    matches: list[CrossMatch] = field(default_factory=list)
    camera_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate CrossMatches data."""
        for match in self.matches:
            if not (0.0 <= match.similarity <= 1.0):
                raise ValueError(
                    f"All similarity scores must be in [0.0, 1.0], "
                    f"got {match.similarity} for local_track_id={match.local_track_id}"
                )

    def __len__(self) -> int:
        """Return count of matches."""
        return len(self.matches)

    def get_match(self, local_track_id: int) -> CrossMatch | None:
        """Return the first CrossMatch for the given local track ID, or None.

        Args:
            local_track_id: Local track ID to look up.

        Returns:
            Matching CrossMatch if found, None otherwise.
        """
        for match in self.matches:
            if match.local_track_id == local_track_id:
                return match
        return None

    def has_cross_camera(self, local_track_id: int) -> bool:
        """Return True if a cross-camera match exists for the given local track ID.

        Args:
            local_track_id: Local track ID to check.

        Returns:
            True if at least one match exists for this track ID.
        """
        return self.get_match(local_track_id) is not None

    @property
    def matched_track_ids(self) -> set[int]:
        """Return set of local track IDs that have cross-camera matches."""
        return {match.local_track_id for match in self.matches}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "matches": [m.to_dict() for m in self.matches],
            "camera_id": self.camera_id,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossMatches:
        """Create from dictionary representation."""
        matches = [CrossMatch.from_dict(m) for m in data.get("matches", [])]
        return cls(
            matches=matches,
            camera_id=data.get("camera_id", ""),
            meta=data.get("meta", {}),
        )
