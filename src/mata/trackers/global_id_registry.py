"""Cross-camera global identity registry.

Maps ``(camera_id, local_track_id)`` pairs to stable, monotonically-increasing
**global IDs** so that the same physical person tracked by different cameras
can be grouped under one identity label.

Typical usage with :class:`~mata.trackers.ReIDBridge`::

    from mata.trackers import GlobalIDRegistry, ReIDBridge

    registry = GlobalIDRegistry(ttl_frames=30)
    bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")

    # Inside your per-frame loop:
    for inst in result.instances:
        if inst.track_id is None or inst.embedding is None:
            continue
        matches = bridge.query(inst.embedding, exclude_camera="cam-1", top_k=1)
        if matches:
            gid = registry.resolve(
                "cam-1", inst.track_id,
                matches[0]["camera_id"], int(matches[0]["track_id"]),
            )
            if gid != -1:
                print(f"Global identity #{gid} seen in cam-1 AND {matches[0]['camera_id']}")

    # After all cameras have been processed for this frame:
    active_keys = {("cam-1", inst.track_id) for inst in result.instances
                   if inst.track_id is not None}
    registry.tick(frame_idx, active_keys)
"""

from __future__ import annotations

__all__ = ["GlobalIDRegistry"]


class GlobalIDRegistry:
    """Maps ``(camera_id, local_track_id)`` pairs to stable cross-camera global IDs.

    When *cam-1* track **#5** matches *cam-4* track **#3**, both keys receive the
    same monotonically increasing integer so crops can be grouped by identity.

    **TTL mechanism** — tracker algorithms (BotSORT / ByteTrack) reuse local
    track IDs when a person leaves the frame and a new one is detected.  Without
    expiry, the new person would inherit the old global ID.  :meth:`tick` is
    called once per inference round; any ``(cam, tid)`` key not seen for
    ``ttl_frames`` consecutive frames is evicted so the next appearance of that
    key starts a fresh global ID.

    Args:
        ttl_frames: Number of consecutive inference frames of absence before a
            ``(camera_id, track_id)`` → ``global_id`` mapping is evicted.
            Default is **30**.

    Attributes:
        ttl_frames: Currently configured TTL in frames.

    Example::

        registry = GlobalIDRegistry(ttl_frames=30)

        # cam-1 track #5 matched cam-4 track #3
        gid = registry.resolve("cam-1", 5, "cam-4", 3)
        assert gid >= 1  # a new global identity was assigned

        # Same pair again — returns the same ID
        assert registry.resolve("cam-1", 5, "cam-4", 3) == gid

        # Advance the frame clock, passing currently-active keys
        registry.tick(frame_idx=10, active_keys={("cam-1", 5), ("cam-4", 3)})
    """

    def __init__(self, ttl_frames: int = 30) -> None:
        self._map: dict[tuple[str, int], int] = {}
        self._last_seen: dict[tuple[str, int], int] = {}
        self._next_id: int = 1
        self.ttl_frames: int = ttl_frames

    # ------------------------------------------------------------------
    # Frame-level lifecycle
    # ------------------------------------------------------------------

    def tick(self, frame_idx: int, active_keys: set[tuple[str, int]]) -> None:
        """Record currently active tracks and evict stale mappings.

        Call once per inference frame **after** processing all cameras.

        Args:
            frame_idx: Current frame counter (monotonically increasing integer).
            active_keys: Set of ``(cam_id, local_track_id)`` pairs that are
                currently alive in this frame's results across *all* cameras.
        """
        for key in active_keys:
            self._last_seen[key] = frame_idx

        # Start the TTL clock for any keys in _map that have not yet been
        # reported as active.  This ensures keys that were resolved but never
        # confirmed active in a tick() call will still expire after ttl_frames
        # consecutive frames of absence.
        for key in self._map:
            if key not in self._last_seen:
                self._last_seen[key] = frame_idx

        expired = [k for k, last_f in self._last_seen.items() if frame_idx - last_f > self.ttl_frames]
        for k in expired:
            self._map.pop(k, None)
            del self._last_seen[k]

    # ------------------------------------------------------------------
    # ID resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        cam_id: str,
        local_tid: int,
        matched_cam_id: str,
        matched_tid: int,
    ) -> int:
        """Assign or look up the shared global ID for a cross-camera match pair.

        Given that *cam_id* / *local_tid* has been matched to *matched_cam_id* /
        *matched_tid* by a ReID query, this method ensures both keys share one
        stable global integer identity.

        **Return value semantics:**

        * **positive int** — the global ID assigned to both keys (new or existing).
        * **-1** — *conflict*: both keys already have *different* existing global
          IDs.  This signals a likely false-positive ReID match; the caller should
          skip saving a crop or taking action rather than merging two previously
          separate identities.

        Args:
            cam_id: Camera ID of the *local* track (e.g. ``"cam-1"``).
            local_tid: Local integer track ID within *cam_id*.
            matched_cam_id: Camera ID of the *matched* track.
            matched_tid: Local integer track ID within *matched_cam_id*.

        Returns:
            A positive global identity integer, or **-1** on ID conflict.
        """
        key_a = (cam_id, local_tid)
        key_b = (matched_cam_id, matched_tid)
        gid_a = self._map.get(key_a)
        gid_b = self._map.get(key_b)

        if gid_a is not None and gid_b is not None:
            if gid_a == gid_b:
                return gid_a  # already unified — nothing to do
            # Conflicting match: both keys belong to *different* known identities.
            return -1
        elif gid_a is not None:
            self._map[key_b] = gid_a
            return gid_a
        elif gid_b is not None:
            self._map[key_a] = gid_b
            return gid_b
        else:
            gid = self._next_id
            self._next_id += 1
            self._map[key_a] = gid
            self._map[key_b] = gid
            return gid

    def reset(self) -> None:
        """Clear all mappings and reset the global ID counter to 1.

        Use this when starting a new video sequence so that global IDs don't
        carry over stale state from a previous run.
        """
        self._map.clear()
        self._last_seen.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def num_global_ids(self) -> int:
        """Total number of unique global IDs ever assigned (not decremented on eviction)."""
        return self._next_id - 1

    @property
    def active_mappings(self) -> dict[tuple[str, int], int]:
        """Read-only snapshot of currently live ``(cam_id, track_id) → global_id`` mappings."""
        return dict(self._map)

    def __repr__(self) -> str:
        return (
            f"GlobalIDRegistry(ttl_frames={self.ttl_frames}, "
            f"num_global_ids={self.num_global_ids}, "
            f"active_keys={len(self._map)})"
        )
