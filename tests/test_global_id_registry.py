"""Unit tests for mata.trackers.GlobalIDRegistry (v1.9.2b2).

Covers:
- Basic ID assignment for a new matched pair
- Idempotent resolution (same pair returns same ID)
- Propagation: one key already mapped → partner inherits its ID
- ID conflict detection (returns -1 when both keys have different existing IDs)
- TTL eviction via tick()
- Active keys do NOT expire within TTL
- Keys expired after TTL_FRAMES+1 absence
- Reset() clears all state and counter
- num_global_ids reflects total assigned (not decremented on eviction)
- active_mappings snapshot
- repr/str
- Public import path: from mata.trackers import GlobalIDRegistry
"""

from __future__ import annotations

from mata.trackers import GlobalIDRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(ttl: int = 30) -> GlobalIDRegistry:
    return GlobalIDRegistry(ttl_frames=ttl)


# ---------------------------------------------------------------------------
# Basic resolve semantics
# ---------------------------------------------------------------------------


class TestResolveBasic:
    def test_new_pair_creates_positive_gid(self):
        reg = _make_registry()
        gid = reg.resolve("cam-1", 5, "cam-4", 3)
        assert gid >= 1

    def test_new_pair_increments_counter(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 1)
        assert reg.num_global_ids == 1
        reg.resolve("cam-1", 2, "cam-2", 2)
        assert reg.num_global_ids == 2

    def test_idempotent_same_pair(self):
        reg = _make_registry()
        gid_a = reg.resolve("cam-1", 5, "cam-4", 3)
        gid_b = reg.resolve("cam-1", 5, "cam-4", 3)
        assert gid_a == gid_b
        assert reg.num_global_ids == 1  # only one identity created

    def test_symmetric_resolve(self):
        """resolve(a, b) and resolve(b, a) must return the same gid."""
        reg = _make_registry()
        gid1 = reg.resolve("cam-1", 5, "cam-4", 3)
        gid2 = reg.resolve("cam-4", 3, "cam-1", 5)
        assert gid1 == gid2

    def test_distinct_pairs_get_distinct_ids(self):
        reg = _make_registry()
        gid1 = reg.resolve("cam-1", 1, "cam-2", 1)
        gid2 = reg.resolve("cam-1", 2, "cam-2", 2)
        assert gid1 != gid2
        assert reg.num_global_ids == 2

    def test_single_camera_pair(self):
        """Even same-camera matches should work (edge case)."""
        reg = _make_registry()
        gid = reg.resolve("cam-1", 1, "cam-1", 2)
        assert gid >= 1


# ---------------------------------------------------------------------------
# Propagation (one side already mapped)
# ---------------------------------------------------------------------------


class TestResolvePropagate:
    def test_first_key_already_mapped_propagates(self):
        reg = _make_registry()
        # cam-1#5 ↔ cam-4#3 assigned gid=1
        gid_orig = reg.resolve("cam-1", 5, "cam-4", 3)
        # Now cam-1#5 ↔ cam-2#9 — cam-1#5 already has gid=1, cam-2#9 is new
        gid_new = reg.resolve("cam-1", 5, "cam-2", 9)
        assert gid_new == gid_orig

    def test_second_key_already_mapped_propagates(self):
        reg = _make_registry()
        gid_orig = reg.resolve("cam-1", 5, "cam-4", 3)
        # cam-2#7 ↔ cam-4#3 — cam-4#3 already has gid, cam-2#7 is new
        gid_new = reg.resolve("cam-2", 7, "cam-4", 3)
        assert gid_new == gid_orig

    def test_propagation_does_not_increase_counter(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 1)
        count_before = reg.num_global_ids
        reg.resolve("cam-1", 1, "cam-3", 5)  # cam-1#1 already mapped
        assert reg.num_global_ids == count_before


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestResolveConflict:
    def test_conflict_returns_minus_one(self):
        reg = _make_registry()
        # Establish two independent identities
        reg.resolve("cam-1", 1, "cam-2", 10)  # gid=1
        reg.resolve("cam-3", 2, "cam-4", 20)  # gid=2
        # Now a (false-positive) match tries to unify cam-1#1 with cam-3#2
        result = reg.resolve("cam-1", 1, "cam-3", 2)
        assert result == -1

    def test_conflict_does_not_increase_counter(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 10)
        reg.resolve("cam-3", 2, "cam-4", 20)
        count_before = reg.num_global_ids
        reg.resolve("cam-1", 1, "cam-3", 2)
        assert reg.num_global_ids == count_before

    def test_conflict_does_not_merge_identities(self):
        reg = _make_registry()
        gid1 = reg.resolve("cam-1", 1, "cam-2", 10)
        gid2 = reg.resolve("cam-3", 2, "cam-4", 20)
        reg.resolve("cam-1", 1, "cam-3", 2)
        # Confirm original mappings unchanged
        assert reg._map[("cam-1", 1)] == gid1
        assert reg._map[("cam-3", 2)] == gid2


# ---------------------------------------------------------------------------
# TTL / tick()
# ---------------------------------------------------------------------------


class TestTick:
    def test_active_keys_not_evicted(self):
        reg = _make_registry(ttl=5)
        reg.resolve("cam-1", 1, "cam-2", 2)
        for f in range(10):
            reg.tick(f, {("cam-1", 1), ("cam-2", 2)})
        # Keys should still be present
        assert ("cam-1", 1) in reg._map
        assert ("cam-2", 2) in reg._map

    def test_inactive_keys_evicted_after_ttl(self):
        reg = _make_registry(ttl=5)
        reg.resolve("cam-1", 1, "cam-2", 2)
        # Tick without reporting the keys as active
        for f in range(7):  # 7 > ttl=5
            reg.tick(f, set())
        assert ("cam-1", 1) not in reg._map
        assert ("cam-2", 2) not in reg._map

    def test_eviction_does_not_lower_num_global_ids(self):
        """Counter is monotonic — eviction does not decrement it."""
        reg = _make_registry(ttl=2)
        reg.resolve("cam-1", 1, "cam-2", 2)
        reg.tick(0, set())
        reg.tick(1, set())
        reg.tick(3, set())  # past TTL
        assert reg.num_global_ids == 1  # still 1

    def test_re_entry_after_eviction_gets_new_gid(self):
        reg = _make_registry(ttl=3)
        gid_old = reg.resolve("cam-1", 1, "cam-2", 2)
        for f in range(5):
            reg.tick(f, set())
        # Re-enter: same local IDs get a fresh gid
        gid_new = reg.resolve("cam-1", 1, "cam-2", 2)
        assert gid_new != gid_old
        assert reg.num_global_ids == 2

    def test_tick_with_empty_active_set_only_evicts_expired(self):
        reg = _make_registry(ttl=5)
        reg.resolve("cam-1", 1, "cam-2", 2)
        reg.resolve("cam-3", 3, "cam-4", 4)
        # Refresh cam-1#1 and cam-2#2 but let cam-3/cam-4 go stale
        for f in range(7):
            reg.tick(f, {("cam-1", 1), ("cam-2", 2)})
        assert ("cam-1", 1) in reg._map
        assert ("cam-3", 3) not in reg._map

    def test_tick_exact_ttl_boundary(self):
        """Keys seen at frame 0 should survive tick at frame ttl, expire at ttl+1."""
        reg = _make_registry(ttl=5)
        reg.resolve("cam-1", 99, "cam-2", 99)
        reg.tick(0, {("cam-1", 99), ("cam-2", 99)})  # last_seen = 0
        reg.tick(5, set())  # frame 5 - 0 = 5 == ttl, NOT expired yet
        assert ("cam-1", 99) in reg._map
        reg.tick(6, set())  # 6 - 0 = 6 > ttl=5 → evicted
        assert ("cam-1", 99) not in reg._map


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_map(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 2)
        reg.reset()
        assert len(reg._map) == 0

    def test_reset_clears_last_seen(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 2)
        reg.tick(10, {("cam-1", 1)})
        reg.reset()
        assert len(reg._last_seen) == 0

    def test_reset_resets_counter(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 2)
        reg.resolve("cam-3", 3, "cam-4", 4)
        assert reg.num_global_ids == 2
        reg.reset()
        assert reg.num_global_ids == 0

    def test_reset_allows_reuse(self):
        reg = _make_registry()
        gid_before = reg.resolve("cam-1", 1, "cam-2", 2)
        reg.reset()
        gid_after = reg.resolve("cam-1", 1, "cam-2", 2)
        # After reset, counter restarts from 1
        assert gid_after == 1
        assert gid_before == 1  # both are 1 (first assigned in their run)


# ---------------------------------------------------------------------------
# active_mappings property
# ---------------------------------------------------------------------------


class TestActiveMappings:
    def test_returns_snapshot_not_reference(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 2)
        snapshot = reg.active_mappings
        # Modifying snapshot must not affect registry
        snapshot[("cam-9", 9)] = 999
        assert ("cam-9", 9) not in reg._map

    def test_snapshot_reflects_current_state(self):
        reg = _make_registry()
        reg.resolve("cam-1", 1, "cam-2", 2)
        snap = reg.active_mappings
        assert ("cam-1", 1) in snap
        assert ("cam-2", 2) in snap


# ---------------------------------------------------------------------------
# repr / str
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_key_info(self):
        reg = _make_registry(ttl=15)
        reg.resolve("cam-1", 1, "cam-2", 2)
        r = repr(reg)
        assert "ttl_frames=15" in r
        assert "num_global_ids=1" in r
        assert "active_keys=2" in r


# ---------------------------------------------------------------------------
# Public import
# ---------------------------------------------------------------------------


class TestPublicImport:
    def test_importable_from_mata_trackers(self):
        from mata.trackers import GlobalIDRegistry as ImportedCls  # noqa: F401

        assert ImportedCls is GlobalIDRegistry

    def test_in_all(self):
        import mata.trackers as pkg

        assert "GlobalIDRegistry" in pkg.__all__
