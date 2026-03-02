"""Tracks artifact for graph system.

Provides object tracking results across video frames with track history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class Track:
    """Single tracked object across frames.

    Attributes:
        track_id: Unique tracking identifier (persistent across frames)
        bbox: Current bounding box in xyxy format
        score: Detection confidence score [0.0, 1.0]
        label: Class label string
        age: Number of frames this track has been tracked
        state: Track state - "active", "lost", or "terminated"
        history: Optional list of previous bboxes [(x1, y1, x2, y2), ...]
        label_id: Optional integer class label

    Examples:
        >>> track = Track(
        ...     track_id=5,
        ...     bbox=(100.0, 200.0, 150.0, 250.0),
        ...     score=0.95,
        ...     label="person",
        ...     age=10,
        ...     state="active"
        ... )
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    label: str
    age: int = 1
    state: str = "active"  # "active", "lost", "terminated"
    history: list[tuple[float, float, float, float]] | None = None
    label_id: int | None = None

    def __post_init__(self):
        """Validate track data."""
        if self.score < 0 or self.score > 1:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")

        if self.age < 1:
            raise ValueError(f"Age must be >= 1, got {self.age}")

        if self.state not in ["active", "lost", "terminated"]:
            raise ValueError(f"State must be 'active', 'lost', or 'terminated', got '{self.state}'")

        # Validate bbox format
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 values (x1, y1, x2, y2), got {len(self.bbox)}")

        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2})")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "score": self.score,
            "label": self.label,
            "age": self.age,
            "state": self.state,
            "history": [list(bbox) for bbox in self.history] if self.history else None,
            "label_id": self.label_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        """Create from dictionary representation."""
        history = None
        if data.get("history"):
            history = [tuple(bbox) for bbox in data["history"]]

        return cls(
            track_id=data["track_id"],
            bbox=tuple(data["bbox"]),
            score=data["score"],
            label=data["label"],
            age=data.get("age", 1),
            state=data.get("state", "active"),
            history=history,
            label_id=data.get("label_id"),
        )


@dataclass(frozen=True)
class Tracks(Artifact):
    """Tracking results artifact for temporal object tracking.

    Contains tracked objects across video frames with state management.
    Tracks can be active (currently tracked), lost (temporarily not detected),
    or terminated (permanently removed).

    Attributes:
        tracks: List of Track objects
        frame_id: Current frame identifier (timestamp, frame number, etc.)
        meta: Optional metadata dictionary

    Examples:
        >>> # Create tracking results for current frame
        >>> tracks = Tracks(
        ...     tracks=[
        ...         Track(track_id=1, bbox=(10, 20, 50, 60), score=0.9, label="car", age=5),
        ...         Track(track_id=2, bbox=(100, 150, 140, 190), score=0.85, label="person", age=3, state="lost")
        ...     ],
        ...     frame_id="frame_0042",
        ...     meta={"video": "traffic.mp4", "fps": 30}
        ... )
        >>>
        >>> # Filter tracks by state
        >>> active = tracks.get_active_tracks()
        >>> lost = tracks.get_lost_tracks()
        >>>
        >>> # Get specific track
        >>> track = tracks.get_track_by_id(1)
    """

    tracks: list[Track] = field(default_factory=list)
    frame_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tracks data."""
        if not self.frame_id:
            raise ValueError("frame_id must be provided")

        # Check for duplicate track IDs
        track_ids = [t.track_id for t in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            duplicates = [tid for tid in track_ids if track_ids.count(tid) > 1]
            raise ValueError(f"Duplicate track IDs found: {set(duplicates)}")

    def get_active_tracks(self) -> Tracks:
        """Get tracks in active state.

        Returns:
            New Tracks artifact containing only active tracks
        """
        active = [t for t in self.tracks if t.state == "active"]
        return Tracks(tracks=active, frame_id=self.frame_id, meta=self.meta)

    def get_lost_tracks(self) -> Tracks:
        """Get tracks in lost state.

        Returns:
            New Tracks artifact containing only lost tracks
        """
        lost = [t for t in self.tracks if t.state == "lost"]
        return Tracks(tracks=lost, frame_id=self.frame_id, meta=self.meta)

    def get_terminated_tracks(self) -> Tracks:
        """Get tracks in terminated state.

        Returns:
            New Tracks artifact containing only terminated tracks
        """
        terminated = [t for t in self.tracks if t.state == "terminated"]
        return Tracks(tracks=terminated, frame_id=self.frame_id, meta=self.meta)

    def get_track_by_id(self, track_id: int) -> Track | None:
        """Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def filter_by_label(self, labels: list[str]) -> Tracks:
        """Filter tracks by label.

        Args:
            labels: List of labels to keep

        Returns:
            New Tracks artifact with filtered tracks
        """
        filtered = [t for t in self.tracks if t.label in labels]
        return Tracks(tracks=filtered, frame_id=self.frame_id, meta=self.meta)

    def filter_by_score(self, threshold: float) -> Tracks:
        """Filter tracks by minimum score.

        Args:
            threshold: Minimum confidence score [0.0, 1.0]

        Returns:
            New Tracks artifact with filtered tracks
        """
        filtered = [t for t in self.tracks if t.score >= threshold]
        return Tracks(tracks=filtered, frame_id=self.frame_id, meta=self.meta)

    def filter_by_age(self, min_age: int) -> Tracks:
        """Filter tracks by minimum age.

        Args:
            min_age: Minimum track age (number of frames)

        Returns:
            New Tracks artifact with filtered tracks
        """
        filtered = [t for t in self.tracks if t.age >= min_age]
        return Tracks(tracks=filtered, frame_id=self.frame_id, meta=self.meta)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {"tracks": [t.to_dict() for t in self.tracks], "frame_id": self.frame_id, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tracks:
        """Create from dictionary representation.

        Args:
            data: Dictionary with tracks, frame_id, and meta

        Returns:
            Tracks artifact
        """
        tracks = [Track.from_dict(t) for t in data["tracks"]]

        return cls(tracks=tracks, frame_id=data["frame_id"], meta=data.get("meta", {}))

    def validate(self) -> None:
        """Validate tracks artifact.

        Raises:
            ValueError: If validation fails
        """
        if not self.frame_id:
            raise ValueError("frame_id must be provided")

        # Validate all tracks
        for i, track in enumerate(self.tracks):
            if track.score < 0 or track.score > 1:
                raise ValueError(f"Track {i} has invalid score: {track.score}")
            if track.age < 1:
                raise ValueError(f"Track {i} has invalid age: {track.age}")
