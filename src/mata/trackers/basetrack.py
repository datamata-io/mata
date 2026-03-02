"""Base track state management for multi-object tracking.

Provides TrackState enumeration and BaseTrack base class used by
both ByteTrack's STrack and BotSort's BOTrack.

Ported from Ultralytics tracker base (MIT-compatible).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------


class TrackState:
    """Object tracking state enumeration.

    States follow a strict lifecycle:
        New      → first seen this frame, not yet confirmed
        Tracked  → successfully associated with a detection
        Lost     → missed for 1+ frames, still held in memory
        Removed  → expired; will be dropped from the tracker pool
    """

    New: int = 0
    Tracked: int = 1
    Lost: int = 2
    Removed: int = 3


# ---------------------------------------------------------------------------
# BaseTrack
# ---------------------------------------------------------------------------


class BaseTrack:
    """Base class for object tracks, providing shared ID counter and lifecycle.

    All per-track state (Kalman filter, feature history, etc.) is managed
    by subclasses.  This class only provides:

    - A class-level auto-incrementing unique track ID.
    - Lifecycle state transitions (mark_lost / mark_removed).
    - Abstract interface that subclasses must implement.

    Attributes:
        _count (int): Class-level counter.  Shared across all instances
            and subclasses.  Reset with :meth:`reset_id` between sequences.
        track_id (int): Unique identifier assigned on :meth:`activate`.
        is_activated (bool): Whether the track has been confirmed.
        state (TrackState): Current lifecycle state.
        score (float): Detection confidence for the most recent observation.
        start_frame (int): Frame number when the track was first activated.
        frame_id (int): Frame number of the most recent update.
        time_since_update (int): Frames elapsed since the last detection match.
    """

    _count: int = 0

    def __init__(self) -> None:
        self.track_id: int = 0
        self.is_activated: bool = False
        self.state: int = TrackState.New

        self.score: float = 0.0
        self.start_frame: int = 0
        self.frame_id: int = 0
        self.time_since_update: int = 0

    # ------------------------------------------------------------------
    # Class-level ID management
    # ------------------------------------------------------------------

    @staticmethod
    def next_id() -> int:
        """Atomically increment and return the global track counter.

        Returns:
            int: New unique track ID (starts at 1).
        """
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to 0.

        Must be called between independent video sequences so that
        IDs within each sequence start from 1.
        """
        BaseTrack._count = 0

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def mark_lost(self) -> None:
        """Transition state to Lost (missed for one or more frames)."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Transition state to Removed (expired; will be purged)."""
        self.state = TrackState.Removed

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def activate(self, *args: object, **kwargs: object) -> None:
        """Activate the track for the first time.

        Subclasses must assign ``self.track_id``, initialise the Kalman
        filter, and set ``self.state = TrackState.Tracked``.
        """
        raise NotImplementedError

    def predict(self) -> None:
        """Advance the internal state model by one time step (prior step)."""
        raise NotImplementedError

    def update(self, *args: object, **kwargs: object) -> None:
        """Update the track with a new matched detection (posterior step)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def end_frame(self) -> int:
        """Frame ID of the most recent observation.

        Mirrors ``frame_id``; provided so consumer code can query the
        last frame without caring whether a track is active or lost.
        """
        return self.frame_id

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"id={self.track_id}, state={self.state}, "
            f"frame={self.frame_id}, score={self.score:.3f})"
        )
