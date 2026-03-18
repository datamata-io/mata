"""Unit tests for the Embeddings artifact (Task D1).

Tests cover creation, validation, immutability, serialization, and edge cases.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mata.core.artifacts import Embeddings
from mata.core.artifacts.base import Artifact


class TestEmbeddingsArtifact:
    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def test_create_with_2d_array(self):
        vectors = np.random.randn(5, 512).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.vectors.shape == (5, 512)

    def test_create_with_1d_auto_reshape(self):
        vec = np.random.randn(256).astype(np.float32)
        embs = Embeddings(vectors=vec)
        assert embs.vectors.shape == (1, 256)

    def test_3d_input_raises_value_error(self):
        bad = np.random.randn(2, 3, 4).astype(np.float32)
        with pytest.raises(ValueError, match="2-D"):
            Embeddings(vectors=bad)

    # ------------------------------------------------------------------
    # instance_ids
    # ------------------------------------------------------------------

    def test_auto_generated_instance_ids(self):
        vectors = np.random.randn(3, 128).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.instance_ids == ("emb_0000", "emb_0001", "emb_0002")

    def test_custom_instance_ids(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        ids = ("track_1", "track_2")
        embs = Embeddings(vectors=vectors, instance_ids=ids)
        assert embs.instance_ids == ids

    def test_auto_generated_id_format(self):
        vectors = np.random.randn(10, 128).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.instance_ids[9] == "emb_0009"

    # ------------------------------------------------------------------
    # __len__ and __getitem__
    # ------------------------------------------------------------------

    def test_len_returns_num_vectors(self):
        vectors = np.random.randn(7, 512).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert len(embs) == 7

    def test_getitem_returns_vector(self):
        vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
        embs = Embeddings(vectors=vectors)
        np.testing.assert_array_equal(embs[0], vectors[0])
        np.testing.assert_array_equal(embs[1], vectors[1])

    def test_getitem_shape(self):
        vectors = np.random.randn(4, 256).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs[2].shape == (256,)

    # ------------------------------------------------------------------
    # embedding_dim
    # ------------------------------------------------------------------

    def test_embedding_dim_auto_set(self):
        vectors = np.random.randn(3, 384).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.embedding_dim == 384

    def test_embedding_dim_explicit_overrides_auto(self):
        vectors = np.random.randn(2, 512).astype(np.float32)
        embs = Embeddings(vectors=vectors, embedding_dim=512)
        assert embs.embedding_dim == 512

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def test_normalized_flag_default_true(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.normalized is True

    def test_normalized_flag_explicit_false(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors, normalized=False)
        assert embs.normalized is False

    def test_meta_default_empty(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert embs.meta == {}

    def test_meta_custom(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors, meta={"model": "clip"})
        assert embs.meta["model"] == "clip"

    # ------------------------------------------------------------------
    # Immutability
    # ------------------------------------------------------------------

    def test_frozen_immutable(self):
        vectors = np.random.randn(3, 128).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            embs.normalized = False  # type: ignore[misc]

    def test_frozen_cannot_set_vectors(self):
        vectors = np.random.randn(3, 128).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            embs.vectors = np.zeros((3, 128), dtype=np.float32)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Inheritance
    # ------------------------------------------------------------------

    def test_is_artifact_subclass(self):
        vectors = np.random.randn(2, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        assert isinstance(embs, Artifact)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_vectors(self):
        empty = np.empty((0, 128), dtype=np.float32)
        embs = Embeddings(vectors=empty)
        assert len(embs) == 0
        assert embs.instance_ids == ()
        # embedding_dim stays 0 for empty (N=0, but d=128; auto-set applies when d>0)
        # shape is preserved
        assert embs.vectors.shape == (0, 128)

    def test_single_vector(self):
        vec = np.random.randn(1, 512).astype(np.float32)
        embs = Embeddings(vectors=vec)
        assert len(embs) == 1
        assert embs.instance_ids == ("emb_0000",)

    def test_1d_to_2d_preserves_dim(self):
        vec = np.ones(128, dtype=np.float32)
        embs = Embeddings(vectors=vec)
        assert embs.vectors.shape == (1, 128)
        assert embs.embedding_dim == 128

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_to_dict_roundtrip(self):
        vectors = np.random.randn(3, 64).astype(np.float32)
        embs = Embeddings(vectors=vectors, meta={"src": "test"})
        d = embs.to_dict()
        restored = Embeddings.from_dict(d)
        assert len(restored) == 3
        assert restored.embedding_dim == 64
        assert restored.meta == {"src": "test"}
        np.testing.assert_allclose(restored.vectors, vectors, rtol=1e-5)

    def test_to_dict_contains_expected_keys(self):
        vectors = np.random.randn(2, 32).astype(np.float32)
        embs = Embeddings(vectors=vectors)
        d = embs.to_dict()
        assert set(d.keys()) == {"vectors", "instance_ids", "embedding_dim", "normalized", "meta"}

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def test_importable_from_artifacts(self):
        from mata.core.artifacts import Embeddings as Emb  # noqa: F401

        assert Emb is Embeddings

    def test_importable_from_embeddings_module(self):
        from mata.core.artifacts.embeddings import Embeddings as Emb  # noqa: F401

        assert Emb is Embeddings
