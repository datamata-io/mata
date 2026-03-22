"""Unit tests for Matches and MatchEntry artifacts.

All tests are pure Python / numpy — no model downloads required.
Run independently: pytest tests/test_matches_artifact.py -v
"""

from __future__ import annotations

import json

import pytest

from mata.core.artifacts.matches import MatchEntry, Matches


# ---------------------------------------------------------------------------
# TestMatchEntry
# ---------------------------------------------------------------------------


class TestMatchEntryCreation:
    def test_basic_creation(self):
        e = MatchEntry(instance_id="img_0", label="alice", similarity=0.9)
        assert e.instance_id == "img_0"
        assert e.label == "alice"
        assert e.similarity == 0.9

    def test_default_all_matches_empty(self):
        e = MatchEntry(instance_id="q", label="bob", similarity=0.5)
        assert e.all_matches == []

    def test_all_matches_provided(self):
        matches = [{"label": "alice", "similarity": 0.9, "index": 0}]
        e = MatchEntry(instance_id="q", label="alice", similarity=0.9, all_matches=matches)
        assert len(e.all_matches) == 1

    def test_frozen(self):
        e = MatchEntry(instance_id="q", label="alice", similarity=0.9)
        with pytest.raises((AttributeError, TypeError)):
            e.label = "bob"  # type: ignore[misc]

    def test_similarity_zero(self):
        e = MatchEntry(instance_id="q", label="unknown", similarity=0.0)
        assert e.similarity == 0.0

    def test_negative_similarity(self):
        e = MatchEntry(instance_id="q", label="x", similarity=-0.3)
        assert e.similarity == -0.3


class TestMatchEntryToDict:
    def test_to_dict_keys(self):
        e = MatchEntry(instance_id="id0", label="alice", similarity=0.85)
        d = e.to_dict()
        assert set(d.keys()) == {"instance_id", "label", "similarity", "all_matches"}

    def test_to_dict_values(self):
        e = MatchEntry(instance_id="id0", label="alice", similarity=0.85)
        d = e.to_dict()
        assert d["instance_id"] == "id0"
        assert d["label"] == "alice"
        assert abs(d["similarity"] - 0.85) < 1e-9

    def test_to_dict_all_matches_empty(self):
        e = MatchEntry(instance_id="x", label="y", similarity=0.0)
        assert e.to_dict()["all_matches"] == []

    def test_to_dict_all_matches_populated(self):
        raw = [{"label": "alice", "similarity": 0.9, "index": 0}]
        e = MatchEntry(instance_id="x", label="alice", similarity=0.9, all_matches=raw)
        d = e.to_dict()
        assert len(d["all_matches"]) == 1
        assert d["all_matches"][0]["label"] == "alice"


# ---------------------------------------------------------------------------
# TestMatches
# ---------------------------------------------------------------------------


def _make_entry(idx: int = 0, label: str = "alice", sim: float = 0.9) -> MatchEntry:
    return MatchEntry(instance_id=f"inst_{idx:04d}", label=label, similarity=sim)


def _make_matches(n: int = 3) -> Matches:
    entries = [_make_entry(i, f"person_{i}", 0.9 - i * 0.1) for i in range(n)]
    return Matches(entries=entries, meta={"model": "test"})


class TestMatchesCreation:
    def test_empty_matches(self):
        m = Matches(entries=[])
        assert len(m) == 0

    def test_with_entries(self):
        m = _make_matches(3)
        assert len(m) == 3

    def test_default_meta_empty(self):
        m = Matches(entries=[])
        assert isinstance(m.meta, dict)

    def test_meta_provided(self):
        m = Matches(entries=[], meta={"model": "clip", "top_k": 1})
        assert m.meta["model"] == "clip"

    def test_frozen(self):
        m = _make_matches(2)
        with pytest.raises((AttributeError, TypeError)):
            m.entries = []  # type: ignore[misc]


class TestMatchesIteration:
    def test_len(self):
        m = _make_matches(5)
        assert len(m) == 5

    def test_iter(self):
        m = _make_matches(3)
        items = list(m)
        assert len(items) == 3
        assert all(isinstance(e, MatchEntry) for e in items)

    def test_entries_order_preserved(self):
        labels = ["alice", "bob", "carol"]
        entries = [_make_entry(i, labels[i]) for i in range(3)]
        m = Matches(entries=entries)
        assert [e.label for e in m] == labels


class TestMatchesToDict:
    def test_to_dict_keys(self):
        m = _make_matches(2)
        d = m.to_dict()
        assert "entries" in d
        assert "meta" in d

    def test_to_dict_entries_count(self):
        m = _make_matches(3)
        d = m.to_dict()
        assert len(d["entries"]) == 3

    def test_to_dict_empty(self):
        m = Matches(entries=[], meta={})
        d = m.to_dict()
        assert d["entries"] == []

    def test_to_dict_meta_preserved(self):
        m = Matches(entries=[], meta={"model": "clip"})
        d = m.to_dict()
        assert d["meta"]["model"] == "clip"

    def test_to_dict_entries_are_dicts(self):
        m = _make_matches(2)
        d = m.to_dict()
        for entry_dict in d["entries"]:
            assert isinstance(entry_dict, dict)
            assert "instance_id" in entry_dict


class TestMatchesFromDict:
    def test_roundtrip(self):
        m = _make_matches(3)
        d = m.to_dict()
        m2 = Matches.from_dict(d)
        assert len(m2) == 3
        assert m2.entries[0].label == m.entries[0].label
        assert abs(m2.entries[0].similarity - m.entries[0].similarity) < 1e-9

    def test_empty_roundtrip(self):
        m = Matches(entries=[], meta={})
        m2 = Matches.from_dict(m.to_dict())
        assert len(m2) == 0

    def test_meta_roundtrip(self):
        m = Matches(entries=[], meta={"model": "onnx", "top_k": 5})
        m2 = Matches.from_dict(m.to_dict())
        assert m2.meta["model"] == "onnx"
        assert m2.meta["top_k"] == 5

    def test_from_dict_missing_entries_defaults_empty(self):
        m = Matches.from_dict({"meta": {}})
        assert len(m) == 0

    def test_from_dict_entry_all_matches_preserved(self):
        raw = [{"label": "alice", "similarity": 0.9, "index": 0}]
        d = {
            "entries": [
                {"instance_id": "q", "label": "alice", "similarity": 0.9, "all_matches": raw}
            ],
            "meta": {},
        }
        m = Matches.from_dict(d)
        assert m.entries[0].all_matches[0]["label"] == "alice"


class TestMatchesPublicAPI:
    def test_importable_from_mata_core_artifacts(self):
        from mata.core.artifacts import Matches, MatchEntry
        assert Matches is not None
        assert MatchEntry is not None

    def test_importable_from_mata(self):
        from mata import Matches, MatchEntry
        assert Matches is not None
        assert MatchEntry is not None

    def test_matches_in_mata_all(self):
        import mata
        assert "Matches" in mata.__all__
        assert "MatchEntry" in mata.__all__
