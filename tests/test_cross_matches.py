"""Tests for CrossMatches artifact and CrossMatch dataclass.

Covers serialization, validation, query helpers, and edge cases.
"""

from __future__ import annotations

import pytest

from mata.core.artifacts.cross_matches import CrossMatch, CrossMatches

# ---------------------------------------------------------------------------
# CrossMatch tests
# ---------------------------------------------------------------------------


class TestCrossMatch:
    """Unit tests for the CrossMatch dataclass."""

    def _make(self, **kwargs) -> CrossMatch:
        defaults = {
            "local_track_id": 1,
            "remote_camera_id": "cam-2",
            "remote_track_id": 7,
            "similarity": 0.88,
        }
        defaults.update(kwargs)
        return CrossMatch(**defaults)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_cross_match_to_dict(self):
        match = self._make(similarity=0.92, remote_bbox=(120.0, 80.0, 160.0, 200.0))
        d = match.to_dict()

        assert d["local_track_id"] == 1
        assert d["remote_camera_id"] == "cam-2"
        assert d["remote_track_id"] == 7
        assert d["similarity"] == 0.92
        assert d["remote_bbox"] == [120.0, 80.0, 160.0, 200.0]

    def test_cross_match_to_dict_no_bbox(self):
        match = self._make()
        d = match.to_dict()
        assert d["remote_bbox"] is None

    def test_cross_match_from_dict(self):
        data = {
            "local_track_id": 3,
            "remote_camera_id": "cam-3",
            "remote_track_id": 11,
            "similarity": 0.75,
            "remote_bbox": [10.0, 20.0, 50.0, 80.0],
        }
        match = CrossMatch.from_dict(data)

        assert match.local_track_id == 3
        assert match.remote_camera_id == "cam-3"
        assert match.remote_track_id == 11
        assert match.similarity == 0.75
        assert match.remote_bbox == (10.0, 20.0, 50.0, 80.0)

    def test_cross_match_from_dict_no_bbox(self):
        data = {
            "local_track_id": 1,
            "remote_camera_id": "cam-2",
            "remote_track_id": 7,
            "similarity": 0.5,
            "remote_bbox": None,
        }
        match = CrossMatch.from_dict(data)
        assert match.remote_bbox is None

    def test_cross_match_roundtrip(self):
        original = self._make(remote_bbox=(5.0, 10.0, 50.0, 90.0))
        restored = CrossMatch.from_dict(original.to_dict())

        assert restored.local_track_id == original.local_track_id
        assert restored.remote_camera_id == original.remote_camera_id
        assert restored.remote_track_id == original.remote_track_id
        assert restored.similarity == original.similarity
        assert restored.remote_bbox == original.remote_bbox

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_similarity_at_zero(self):
        match = self._make(similarity=0.0)
        assert match.similarity == 0.0

    def test_similarity_at_one(self):
        match = self._make(similarity=1.0)
        assert match.similarity == 1.0

    def test_similarity_too_low_raises(self):
        with pytest.raises(ValueError, match="similarity"):
            self._make(similarity=-0.01)

    def test_similarity_too_high_raises(self):
        with pytest.raises(ValueError, match="similarity"):
            self._make(similarity=1.01)

    def test_remote_bbox_wrong_length_raises(self):
        with pytest.raises(ValueError, match="remote_bbox"):
            CrossMatch(
                local_track_id=1,
                remote_camera_id="cam-2",
                remote_track_id=5,
                similarity=0.5,
                remote_bbox=(1.0, 2.0, 3.0),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# CrossMatches tests
# ---------------------------------------------------------------------------


class TestCrossMatches:
    """Unit tests for the CrossMatches artifact."""

    def _make_match(self, local_id: int, remote_id: int = 5, cam: str = "cam-2", sim: float = 0.85) -> CrossMatch:
        return CrossMatch(
            local_track_id=local_id,
            remote_camera_id=cam,
            remote_track_id=remote_id,
            similarity=sim,
        )

    # ------------------------------------------------------------------
    # Construction & defaults
    # ------------------------------------------------------------------

    def test_empty_cross_matches(self):
        cm = CrossMatches()
        assert cm.matches == []
        assert cm.camera_id == ""
        assert cm.meta == {}

    def test_empty_cross_matches_len(self):
        cm = CrossMatches()
        assert len(cm) == 0

    # ------------------------------------------------------------------
    # __len__
    # ------------------------------------------------------------------

    def test_len_single(self):
        cm = CrossMatches(matches=[self._make_match(1)], camera_id="cam-1")
        assert len(cm) == 1

    def test_len_multiple(self):
        cm = CrossMatches(
            matches=[self._make_match(1), self._make_match(2), self._make_match(3)],
            camera_id="cam-1",
        )
        assert len(cm) == 3

    # ------------------------------------------------------------------
    # get_match
    # ------------------------------------------------------------------

    def test_get_match_found(self):
        m = self._make_match(local_id=3)
        cm = CrossMatches(matches=[m], camera_id="cam-1")

        result = cm.get_match(3)
        assert result is not None
        assert result.local_track_id == 3

    def test_get_match_not_found(self):
        cm = CrossMatches(matches=[self._make_match(1)], camera_id="cam-1")
        assert cm.get_match(99) is None

    def test_get_match_returns_first_for_duplicate_local_ids(self):
        m1 = CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=5, similarity=0.9)
        m2 = CrossMatch(local_track_id=1, remote_camera_id="cam-3", remote_track_id=8, similarity=0.7)
        cm = CrossMatches(matches=[m1, m2], camera_id="cam-1")

        result = cm.get_match(1)
        assert result is not None
        assert result.remote_camera_id == "cam-2"

    # ------------------------------------------------------------------
    # has_cross_camera
    # ------------------------------------------------------------------

    def test_has_cross_camera_true(self):
        cm = CrossMatches(matches=[self._make_match(local_id=5)], camera_id="cam-1")
        assert cm.has_cross_camera(5) is True

    def test_has_cross_camera_false(self):
        cm = CrossMatches(matches=[self._make_match(local_id=5)], camera_id="cam-1")
        assert cm.has_cross_camera(99) is False

    def test_has_cross_camera_empty(self):
        cm = CrossMatches()
        assert cm.has_cross_camera(1) is False

    # ------------------------------------------------------------------
    # matched_track_ids
    # ------------------------------------------------------------------

    def test_matched_track_ids_empty(self):
        cm = CrossMatches()
        assert cm.matched_track_ids == set()

    def test_matched_track_ids(self):
        cm = CrossMatches(
            matches=[self._make_match(1), self._make_match(2), self._make_match(3)],
            camera_id="cam-1",
        )
        assert cm.matched_track_ids == {1, 2, 3}

    def test_matched_track_ids_deduplicated(self):
        m1 = CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=5, similarity=0.9)
        m2 = CrossMatch(local_track_id=1, remote_camera_id="cam-3", remote_track_id=8, similarity=0.7)
        cm = CrossMatches(matches=[m1, m2], camera_id="cam-1")
        assert cm.matched_track_ids == {1}

    # ------------------------------------------------------------------
    # to_dict / from_dict
    # ------------------------------------------------------------------

    def test_to_dict_keys(self):
        cm = CrossMatches(matches=[self._make_match(1)], camera_id="cam-1", meta={"fps": 30})
        d = cm.to_dict()
        assert "matches" in d
        assert "camera_id" in d
        assert "meta" in d

    def test_to_dict_roundtrip(self):
        original = CrossMatches(
            matches=[
                CrossMatch(
                    local_track_id=1,
                    remote_camera_id="cam-2",
                    remote_track_id=5,
                    similarity=0.88,
                    remote_bbox=(10.0, 20.0, 50.0, 80.0),
                ),
                CrossMatch(
                    local_track_id=2,
                    remote_camera_id="cam-3",
                    remote_track_id=9,
                    similarity=0.72,
                ),
            ],
            camera_id="cam-1",
            meta={"frame": 42},
        )
        restored = CrossMatches.from_dict(original.to_dict())

        assert len(restored) == len(original)
        assert restored.camera_id == original.camera_id
        assert restored.meta == original.meta
        assert restored.matches[0].local_track_id == 1
        assert restored.matches[0].remote_bbox == (10.0, 20.0, 50.0, 80.0)
        assert restored.matches[1].remote_bbox is None

    def test_from_dict_empty_matches(self):
        data = {"matches": [], "camera_id": "cam-1", "meta": {}}
        cm = CrossMatches.from_dict(data)
        assert len(cm) == 0
        assert cm.camera_id == "cam-1"

    def test_from_dict_missing_optional_fields(self):
        data = {"matches": []}
        cm = CrossMatches.from_dict(data)
        assert cm.camera_id == ""
        assert cm.meta == {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_similarity_validation_in_list(self):
        # CrossMatch itself rejects out-of-range similarity before CrossMatches is built
        with pytest.raises(ValueError, match="similarity"):
            CrossMatch(
                local_track_id=1,
                remote_camera_id="cam-2",
                remote_track_id=5,
                similarity=1.5,
            )

        with pytest.raises(ValueError, match="similarity"):
            CrossMatch(
                local_track_id=1,
                remote_camera_id="cam-2",
                remote_track_id=5,
                similarity=-0.1,
            )

    def test_boundary_similarities_accepted(self):
        cm = CrossMatches(
            matches=[
                CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=5, similarity=0.0),
                CrossMatch(local_track_id=2, remote_camera_id="cam-2", remote_track_id=6, similarity=1.0),
            ],
            camera_id="cam-1",
        )
        assert len(cm) == 2
