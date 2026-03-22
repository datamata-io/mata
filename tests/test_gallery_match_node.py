"""Unit tests for GalleryMatchNode graph node.

All tests use mocks — no real models required.
Run independently: pytest tests/test_gallery_match_node.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.nodes.gallery_match import GalleryMatchNode
from mata.recognition.gallery import Gallery, GalleryMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(dim: int = 32) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_embeddings_artifact(n: int = 3, dim: int = 32) -> MagicMock:
    """Return a mock Embeddings artifact with vectors + instance_ids."""
    from mata.core.artifacts.embeddings import Embeddings

    vectors = np.stack([_unit(dim) for _ in range(n)]).astype(np.float32)
    instance_ids = tuple(f"inst_{i:04d}" for i in range(n))
    artifact = MagicMock(spec=Embeddings)
    artifact.vectors = vectors
    artifact.instance_ids = instance_ids
    return artifact


def _make_gallery_with_persons(n: int = 3, dim: int = 32) -> Gallery:
    g = Gallery(similarity_thresh=0.0)  # low threshold: accept everything
    for i in range(n):
        g.add(f"person_{i}", _unit(dim))
    return g


# ---------------------------------------------------------------------------
# TestGalleryMatchNodeConstruction
# ---------------------------------------------------------------------------

class TestGalleryMatchNodeConstruction:
    def test_default_construction(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g)
        assert node._top_k == 1
        assert node._threshold is None
        assert node._src == "embeddings"
        assert node._out == "matches"

    def test_custom_src_out(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g, src="emb", out="id_matches")
        assert node._src == "emb"
        assert node._out == "id_matches"

    def test_custom_name(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g, name="MyNode")
        assert node.name == "MyNode"

    def test_default_name(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g)
        assert "GalleryMatchNode" in node.name


# ---------------------------------------------------------------------------
# TestGalleryMatchNodeInputsOutputs
# ---------------------------------------------------------------------------

class TestGalleryMatchNodeInputsOutputs:
    def test_inputs_contains_src_key(self):
        from mata.core.artifacts.embeddings import Embeddings

        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g)
        assert "embeddings" in node.inputs
        assert node.inputs["embeddings"] is Embeddings

    def test_outputs_contains_out_key(self):
        from mata.core.artifacts.matches import Matches

        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g)
        assert "matches" in node.outputs
        assert node.outputs["matches"] is Matches

    def test_custom_src_appears_in_inputs(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g, src="my_emb")
        assert "my_emb" in node.inputs

    def test_custom_out_appears_in_outputs(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g, out="my_matches")
        assert "my_matches" in node.outputs


# ---------------------------------------------------------------------------
# TestGalleryMatchNodeRun
# ---------------------------------------------------------------------------

class TestGalleryMatchNodeRun:
    def test_run_produces_matches_artifact(self):
        from mata.core.artifacts.matches import Matches

        g = _make_gallery_with_persons(3, 32)
        node = GalleryMatchNode(gallery=g, top_k=1)
        emb = _make_embeddings_artifact(3, 32)
        ctx = MagicMock()
        result = node.run(ctx, embeddings=emb)
        assert "matches" in result
        assert isinstance(result["matches"], Matches)

    def test_run_returns_one_entry_per_embedding(self):
        g = _make_gallery_with_persons(3, 32)
        node = GalleryMatchNode(gallery=g, top_k=1)
        emb = _make_embeddings_artifact(4, 32)  # 4 query vectors
        ctx = MagicMock()
        result = node.run(ctx, embeddings=emb)
        assert len(result["matches"]) == 4

    def test_run_preserves_instance_ids(self):
        g = _make_gallery_with_persons(3, 32)
        node = GalleryMatchNode(gallery=g, top_k=1)
        emb = _make_embeddings_artifact(2, 32)
        emb.instance_ids = ("id_a", "id_b")
        ctx = MagicMock()
        result = node.run(ctx, embeddings=emb)
        matches = result["matches"]
        ids = {e.instance_id for e in matches.entries}
        assert "id_a" in ids
        assert "id_b" in ids

    def test_run_missing_artifact_raises(self):
        g = _make_gallery_with_persons()
        node = GalleryMatchNode(gallery=g)
        ctx = MagicMock()
        with pytest.raises(ValueError, match="missing input artifact"):
            node.run(ctx)  # no embeddings kwarg

    def test_run_empty_gallery_produces_unknown(self):
        g = Gallery()  # empty
        node = GalleryMatchNode(gallery=g, top_k=1)
        emb = _make_embeddings_artifact(2, 32)
        ctx = MagicMock()
        result = node.run(ctx, embeddings=emb)
        for entry in result["matches"].entries:
            assert entry.label == "unknown"
            assert entry.similarity == 0.0

    def test_run_custom_src_key(self):
        from mata.core.artifacts.matches import Matches

        g = _make_gallery_with_persons(2, 32)
        node = GalleryMatchNode(gallery=g, src="feats", out="ids")
        emb = _make_embeddings_artifact(2, 32)
        ctx = MagicMock()
        result = node.run(ctx, feats=emb)
        assert "ids" in result
        assert isinstance(result["ids"], Matches)

    def test_run_entry_has_all_matches_list(self):
        g = _make_gallery_with_persons(3, 32)
        node = GalleryMatchNode(gallery=g, top_k=3)
        emb = _make_embeddings_artifact(1, 32)
        ctx = MagicMock()
        result = node.run(ctx, embeddings=emb)
        entry = result["matches"].entries[0]
        assert isinstance(entry.all_matches, list)


# ---------------------------------------------------------------------------
# TestGalleryMatchNodeImport
# ---------------------------------------------------------------------------

class TestGalleryMatchNodeImport:
    def test_importable_from_nodes(self):
        from mata.nodes import GalleryMatchNode
        assert GalleryMatchNode is not None

    def test_in_nodes_all(self):
        import mata.nodes as nodes
        assert "GalleryMatchNode" in nodes.__all__
