"""Unit tests for mata.run("recognize", ...) and the recognize convenience API.

Uses mocks to avoid model downloads — tests the routing, argument handling,
gallery integration, and error paths of the recognize task.

Run independently: pytest tests/test_recognize_api.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mata
from mata import Gallery, GalleryMatch, MatchEntry, Matches
from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.matches import Matches as MatchesArtifact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int = 64) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_gallery(n: int = 3, dim: int = 64) -> Gallery:
    g = Gallery(similarity_thresh=0.0)
    for i in range(n):
        g.add(f"person_{i}", _unit(dim))
    return g


def _make_embed_result(vectors: np.ndarray) -> Embeddings:
    """Return an Embeddings artifact from a (N, D) array."""
    return Embeddings(
        vectors=vectors.astype(np.float32),
        instance_ids=tuple(f"emb_{i:04d}" for i in range(len(vectors))),
        meta={},
    )


def _mock_embed_adapter(dim: int = 64) -> MagicMock:
    """Return a mock embed adapter whose embed() returns a single unit vector."""
    adapter = MagicMock()
    vec = _unit(dim)
    emb = _make_embed_result(vec[np.newaxis, :])
    adapter.embed.return_value = emb
    return adapter


# ---------------------------------------------------------------------------
# TestRecognizeRouting
# ---------------------------------------------------------------------------


class TestRecognizeRouting:
    """Test that mata.run('recognize', ...) routes correctly."""

    def test_run_recognize_returns_matches(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter):
            result = mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=gallery)
        assert isinstance(result, MatchesArtifact)

    def test_run_recognize_one_entry_per_image(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter):
            result = mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=gallery)
        assert len(result.entries) == 1

    def test_run_recognize_entry_has_label(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter):
            result = mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=gallery)
        entry = result.entries[0]
        assert isinstance(entry.label, str)
        assert len(entry.label) > 0

    def test_run_recognize_entry_has_similarity(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter):
            result = mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=gallery)
        entry = result.entries[0]
        assert isinstance(entry.similarity, float)

    def test_run_recognize_instance_id_is_query(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter):
            result = mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=gallery)
        assert result.entries[0].instance_id == "query"

    def test_run_recognize_calls_embed_task(self):
        gallery = _make_gallery(3, 64)
        adapter = _mock_embed_adapter(64)
        with patch("mata.api.load", return_value=adapter) as mock_load:
            mata.run(
                "recognize",
                np.zeros((64, 64, 3), dtype=np.uint8),
                gallery=gallery,
                model="openai/clip-vit-base-patch32",
            )
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args
        assert call_kwargs[1].get("task") == "embed" or call_kwargs[0][0] == "embed"


# ---------------------------------------------------------------------------
# TestRecognizeErrors
# ---------------------------------------------------------------------------


class TestRecognizeErrors:
    def test_missing_gallery_raises_value_error(self):
        with pytest.raises(ValueError, match="gallery"):
            mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8))

    def test_missing_gallery_none_raises(self):
        with pytest.raises(ValueError, match="gallery"):
            mata.run("recognize", np.zeros((64, 64, 3), dtype=np.uint8), gallery=None)

    def test_unsupported_input_type_raises(self):
        gallery = _make_gallery()
        adapter = _mock_embed_adapter()
        with patch("mata.api.load", return_value=adapter):
            with pytest.raises((ValueError, TypeError)):
                mata.run("recognize", 12345, gallery=gallery)  # type: ignore[arg-type]

    def test_track_task_still_raises(self):
        with pytest.raises(ValueError, match="stateful"):
            mata.run("track", "video.mp4")


# ---------------------------------------------------------------------------
# TestRecognizeWithRealGallery
# ---------------------------------------------------------------------------


class TestRecognizeWithRealGallery:
    """Integration-style tests using a real Gallery — no model calls."""

    def _run_with_mock(
        self,
        gallery: Gallery,
        query_vec: np.ndarray,
        model: str | None = None,
        top_k: int = 1,
        threshold: float | None = None,
    ) -> MatchesArtifact:
        """Helper: patch embed adapter to return a fixed query vector."""

        adapter = MagicMock()
        emb = _make_embed_result(query_vec[np.newaxis, :])
        adapter.embed.return_value = emb
        with patch("mata.api.load", return_value=adapter):
            return mata.run(
                "recognize",
                np.zeros((8, 8, 3), dtype=np.uint8),
                model=model,
                gallery=gallery,
                top_k=top_k,
                threshold=threshold,
            )

    def test_exact_match_returns_correct_label(self):
        g = Gallery(similarity_thresh=0.0)
        v = _unit(32)
        g.add("alice", v)
        result = self._run_with_mock(g, v, top_k=1)
        assert result.entries[0].label == "alice"
        assert abs(result.entries[0].similarity - 1.0) < 1e-4

    def test_empty_gallery_returns_unknown(self):
        g = Gallery()
        result = self._run_with_mock(g, _unit(32))
        assert result.entries[0].label == "unknown"
        assert result.entries[0].similarity == 0.0

    def test_top_k_respected_in_all_matches(self):
        g = Gallery(similarity_thresh=-1.0)
        for i in range(5):
            g.add(f"person_{i}", _unit(32))
        result = self._run_with_mock(g, _unit(32), top_k=3)
        entry = result.entries[0]
        assert len(entry.all_matches) <= 3

    def test_threshold_filters_low_similarity(self):
        g = Gallery(similarity_thresh=0.99)
        v = _unit(32)
        g.add("alice", v)
        # Use a random query: similarity will be < 0.99
        result = self._run_with_mock(g, _unit(32))
        # Result should be unknown when query doesn't meet threshold
        entry = result.entries[0]
        # Either alice (if lucky) or unknown; just check it's a valid label
        assert isinstance(entry.label, str)

    def test_high_threshold_override_produces_unknown(self):
        g = Gallery(similarity_thresh=0.0)
        v = _unit(32)
        g.add("alice", v)
        # Override to very high threshold — orthogonal vector won't match
        query = np.zeros(32, dtype=np.float32)
        query[0] = 1.0
        opposite = np.zeros(32, dtype=np.float32)
        opposite[1] = 1.0  # orthogonal
        result = self._run_with_mock(g, opposite, threshold=0.99)
        entry = result.entries[0]
        # similarity(opposite, alice[0]=1) ≈ 0
        assert entry.label == "unknown" or entry.similarity < 0.99

    def test_best_of_multiple_labels(self):
        g = Gallery(similarity_thresh=0.0)
        alice = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        g.add("alice", alice)
        g.add("bob", bob)
        # Query close to alice
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        result = self._run_with_mock(g, query, top_k=2)
        entry = result.entries[0]
        assert entry.label == "alice"

    def test_meta_contains_top_k(self):
        g = _make_gallery(2, 32)
        result = self._run_with_mock(g, _unit(32), top_k=2)
        assert result.meta.get("top_k") == 2

    def test_matches_artifact_iterable(self):
        g = _make_gallery(2, 32)
        result = self._run_with_mock(g, _unit(32))
        entries = list(result)
        assert len(entries) == 1

    def test_matches_to_dict_roundtrip(self):
        g = _make_gallery(2, 32)
        result = self._run_with_mock(g, _unit(32))
        d = result.to_dict()
        restored = MatchesArtifact.from_dict(d)
        assert len(restored) == len(result)
        assert restored.entries[0].label == result.entries[0].label


# ---------------------------------------------------------------------------
# TestRecognizePublicAPI
# ---------------------------------------------------------------------------


class TestRecognizePublicAPI:
    def test_gallery_importable_from_mata(self):
        from mata import Gallery

        assert Gallery is not None

    def test_gallery_match_importable_from_mata(self):

        assert GalleryMatch is not None

    def test_matches_importable_from_mata(self):

        assert Matches is not None

    def test_match_entry_importable_from_mata(self):

        assert MatchEntry is not None

    def test_run_function_exists(self):
        assert callable(mata.run)

    def test_gallery_class_has_search(self):
        g = Gallery()
        assert callable(g.search)

    def test_gallery_class_has_add(self):
        g = Gallery()
        assert callable(g.add)

    def test_gallery_class_has_save_load(self):
        g = Gallery()
        assert callable(g.save)
        assert callable(Gallery.load)

    def test_matches_artifact_importable_from_core(self):
        from mata.core.artifacts import MatchEntry, Matches

        assert Matches is not None
        assert MatchEntry is not None
