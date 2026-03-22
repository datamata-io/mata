"""Unit tests for Gallery and GalleryMatch.

All tests are pure Python / numpy — no model downloads required.
Run independently: pytest tests/test_gallery.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mata.recognition.gallery import Gallery, GalleryMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(dim: int = 64) -> np.ndarray:
    """Return a random L2-normalised vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_gallery(n: int = 3, dim: int = 64) -> tuple[Gallery, np.ndarray]:
    """Return a Gallery pre-loaded with n distinct embeddings."""
    g = Gallery()
    embeddings = []
    for i in range(n):
        v = _unit(dim)
        g.add(f"person_{i}", v)
        embeddings.append(v)
    return g, np.stack(embeddings)


# ---------------------------------------------------------------------------
# TestGalleryMatch
# ---------------------------------------------------------------------------

class TestGalleryMatch:
    def test_creation(self):
        m = GalleryMatch(label="alice", similarity=0.9, index=0)
        assert m.label == "alice"
        assert m.similarity == 0.9
        assert m.index == 0

    def test_frozen(self):
        m = GalleryMatch(label="alice", similarity=0.9, index=0)
        with pytest.raises((AttributeError, TypeError)):
            m.label = "bob"  # type: ignore[misc]

    def test_to_dict(self):
        m = GalleryMatch(label="alice", similarity=0.92, index=2)
        d = m.to_dict()
        assert d == {"label": "alice", "similarity": 0.92, "index": 2}

    def test_to_json(self):
        m = GalleryMatch(label="bob", similarity=0.75, index=1)
        parsed = json.loads(m.to_json())
        assert parsed["label"] == "bob"
        assert abs(parsed["similarity"] - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# TestGalleryAdd
# ---------------------------------------------------------------------------

class TestGalleryAdd:
    def test_add_returns_index_zero(self):
        g = Gallery()
        v = _unit(64)
        idx = g.add("alice", v)
        assert idx == 0

    def test_add_increments_index(self):
        g = Gallery()
        for i in range(5):
            idx = g.add(f"person_{i}", _unit(64))
            assert idx == i

    def test_size_after_add(self):
        g = Gallery()
        g.add("alice", _unit())
        g.add("bob", _unit())
        assert g.size == 2

    def test_add_normalizes_embedding(self):
        g = Gallery()
        v = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        g.add("X", v)
        stored = g._embeddings[0]
        np.testing.assert_allclose(np.linalg.norm(stored), 1.0, atol=1e-6)

    def test_add_1d_vector(self):
        g = Gallery()
        v = np.random.randn(128).astype(np.float32)
        g.add("alice", v)
        assert g.size == 1

    def test_add_2d_row_vector(self):
        g = Gallery()
        v = np.random.randn(1, 128).astype(np.float32)
        g.add("alice", v)
        assert g.size == 1


# ---------------------------------------------------------------------------
# TestGalleryAddMany
# ---------------------------------------------------------------------------

class TestGalleryAddMany:
    def test_add_many_returns_indices(self):
        g = Gallery()
        embeddings = np.random.randn(3, 64).astype(np.float32)
        indices = g.add_many(["a", "b", "c"], embeddings)
        assert indices == [0, 1, 2]

    def test_add_many_updates_size(self):
        g = Gallery()
        g.add_many(["a", "b"], np.random.randn(2, 64).astype(np.float32))
        assert g.size == 2

    def test_add_many_mismatched_labels_raises(self):
        g = Gallery()
        with pytest.raises(ValueError, match="labels length"):
            g.add_many(["a"], np.random.randn(2, 64).astype(np.float32))

    def test_add_many_single_1d_vector(self):
        g = Gallery()
        v = np.random.randn(64).astype(np.float32)
        indices = g.add_many(["solo"], v)
        assert indices == [0]

    def test_add_many_empty_labels_ok(self):
        g = Gallery()
        indices = g.add_many([], np.zeros((0, 64), dtype=np.float32))
        assert indices == []


# ---------------------------------------------------------------------------
# TestGallerySearch
# ---------------------------------------------------------------------------

class TestGallerySearch:
    def test_search_empty_gallery_returns_empty(self):
        g = Gallery()
        results = g.search(_unit())
        assert results == []

    def test_search_finds_exact_match(self):
        g = Gallery()
        v = _unit(64)
        g.add("alice", v)
        results = g.search(v, threshold=0.0)
        assert len(results) == 1
        assert results[0].label == "alice"
        assert abs(results[0].similarity - 1.0) < 1e-5

    def test_search_top_k_respected(self):
        g, _ = _make_gallery(10)
        q = _unit(64)
        results = g.search(q, top_k=3, threshold=-1.0)
        assert len(results) <= 3

    def test_search_threshold_filters(self):
        g = Gallery(similarity_thresh=0.99)
        v = _unit(64)
        g.add("alice", v)
        # All-zeros query: low similarity
        q = np.zeros(64, dtype=np.float32)
        results = g.search(q)
        assert results == []

    def test_search_sorted_by_similarity(self):
        g = Gallery(similarity_thresh=0.0)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        g.add("best", v1)
        g.add("other", v2)
        results = g.search(v1, top_k=2)
        if len(results) > 1:
            assert results[0].similarity >= results[1].similarity

    def test_search_returns_gallery_matches(self):
        g = Gallery()
        v = _unit(32)
        g.add("alice", v)
        results = g.search(v, threshold=0.0)
        assert all(isinstance(r, GalleryMatch) for r in results)

    def test_search_custom_threshold_overrides_default(self):
        g = Gallery(similarity_thresh=0.99)
        v = _unit(64)
        g.add("alice", v)
        # Search with low threshold: should return
        results = g.search(v, threshold=0.0)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# TestGallerySearchBatch
# ---------------------------------------------------------------------------

class TestGallerySearchBatch:
    def test_search_batch_same_len_as_queries(self):
        g, vecs = _make_gallery(5)
        queries = np.stack([_unit(64) for _ in range(3)])
        results = g.search_batch(queries, top_k=2, threshold=-1.0)
        assert len(results) == 3

    def test_search_batch_single_vector(self):
        g = Gallery()
        v = _unit(64)
        g.add("alpha", v)
        results = g.search_batch(v, top_k=1, threshold=0.0)
        assert len(results) == 1

    def test_search_batch_returns_lists_of_gallery_matches(self):
        g, _ = _make_gallery(3)
        queries = np.stack([_unit(64) for _ in range(2)])
        results = g.search_batch(queries, threshold=-1.0)
        for r in results:
            assert isinstance(r, list)
            assert all(isinstance(m, GalleryMatch) for m in r)


# ---------------------------------------------------------------------------
# TestGalleryRemove
# ---------------------------------------------------------------------------

class TestGalleryRemove:
    def test_remove_existing_label(self):
        g = Gallery()
        g.add("alice", _unit())
        removed = g.remove("alice")
        assert removed == 1
        assert g.size == 0

    def test_remove_nonexistent_label(self):
        g = Gallery()
        removed = g.remove("nobody")
        assert removed == 0

    def test_remove_multiple_same_label(self):
        g = Gallery()
        for _ in range(3):
            g.add("duplicate", _unit())
        removed = g.remove("duplicate")
        assert removed == 3
        assert g.size == 0

    def test_remove_does_not_affect_others(self):
        g = Gallery()
        g.add_many(["alice", "bob", "alice"], np.random.randn(3, 16).astype(np.float32))
        g.remove("alice")
        assert g.size == 1
        assert g._labels[0] == "bob"


# ---------------------------------------------------------------------------
# TestGalleryPersistence
# ---------------------------------------------------------------------------

class TestGalleryPersistence:
    def test_save_and_load(self):
        g = Gallery(similarity_thresh=0.6)
        v = _unit(32)
        g.add("alice", v)
        g.add("bob", _unit(32))
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        g.save(path)
        g2 = Gallery.load(path)
        assert g2.size == 2
        assert set(g2._labels) == {"alice", "bob"}
        np.testing.assert_allclose(g2._embeddings[0], g._embeddings[0], atol=1e-6)

    def test_save_load_preserves_threshold(self):
        g = Gallery(similarity_thresh=0.75)
        g.add("x", _unit())
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        g.save(path)
        g2 = Gallery.load(path)
        assert abs(g2._similarity_thresh - 0.75) < 1e-6

    def test_load_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            Gallery.load("/nonexistent/path/gallery.npz")

    def test_save_creates_parent_dir(self):
        g = Gallery()
        g.add("x", _unit())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "gallery.npz")
            g.save(path)
            assert Path(path).exists()


# ---------------------------------------------------------------------------
# TestGallerySerialization
# ---------------------------------------------------------------------------

class TestGallerySerialization:
    def test_to_dict_keys(self):
        g = Gallery()
        g.add("alice", _unit())
        d = g.to_dict()
        assert "embeddings" in d
        assert "labels" in d
        assert "similarity_thresh" in d

    def test_to_json_valid(self):
        g = Gallery()
        g.add("alice", _unit())
        s = g.to_json()
        parsed = json.loads(s)
        assert "labels" in parsed

    def test_from_dict_roundtrip(self):
        g = Gallery(similarity_thresh=0.65)
        g.add("alice", _unit(32))
        g.add("bob", _unit(32))
        g2 = Gallery.from_dict(g.to_dict())
        assert g2.size == g.size
        assert g2._labels == g._labels

    def test_from_json_roundtrip(self):
        g = Gallery()
        g.add("x", _unit(16))
        g2 = Gallery.from_json(g.to_json())
        assert g2.size == 1

    def test_empty_gallery_serialization(self):
        g = Gallery()
        g2 = Gallery.from_dict(g.to_dict())
        assert g2.size == 0


# ---------------------------------------------------------------------------
# TestGalleryPublicAPI
# ---------------------------------------------------------------------------

class TestGalleryPublicAPI:
    def test_importable_from_mata(self):
        from mata import Gallery
        assert Gallery is not None

    def test_importable_from_recognition(self):
        from mata.recognition import Gallery, GalleryMatch
        assert Gallery is not None
        assert GalleryMatch is not None

    def test_gallery_match_importable_from_mata(self):
        from mata import GalleryMatch
        assert GalleryMatch is not None

    def test_size_property_empty(self):
        g = Gallery()
        assert g.size == 0

    def test_size_property_after_add(self):
        g = Gallery()
        g.add("a", _unit())
        g.add("b", _unit())
        assert g.size == 2
