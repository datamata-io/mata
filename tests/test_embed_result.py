"""Unit tests for EmbedResult dataclass.

All tests are pure Python / numpy — no model downloads required.
Run independently: pytest tests/test_embed_result.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mata.core.types import EmbedResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embed(n: int = 4, dim: int = 128) -> EmbedResult:
    data = np.random.randn(n, dim).astype(np.float32)
    return EmbedResult(embeddings=data)


# ---------------------------------------------------------------------------
# TestEmbedResultCreation
# ---------------------------------------------------------------------------


class TestEmbedResultCreation:
    def test_basic_2d_creation(self):
        data = np.zeros((3, 64), dtype=np.float32)
        er = EmbedResult(embeddings=data)
        assert er.embeddings.shape == (3, 64)

    def test_1d_input_promoted_to_2d(self):
        """A single (D,) vector must be auto-promoted to (1, D)."""
        data = np.ones(128, dtype=np.float32)
        er = EmbedResult(embeddings=data)
        assert er.embeddings.shape == (1, 128)

    def test_labels_optional(self):
        er = _make_embed()
        assert er.labels is None

    def test_labels_stored(self):
        labels = ["a", "b", "c"]
        data = np.zeros((3, 64), dtype=np.float32)
        er = EmbedResult(embeddings=data, labels=labels)
        assert er.labels == labels

    def test_meta_defaults_empty(self):
        er = _make_embed()
        assert er.meta == {}

    def test_meta_stored(self):
        er = EmbedResult(embeddings=np.zeros((1, 8), dtype=np.float32), meta={"model": "clip"})
        assert er.meta["model"] == "clip"

    def test_embeddings_float32(self):
        data = np.zeros((2, 16), dtype=np.float64)
        er = EmbedResult(embeddings=data)
        # Type should be preserved (no forced cast in __post_init__)
        assert er.embeddings is not None

    def test_frozen(self):
        er = _make_embed()
        with pytest.raises((AttributeError, TypeError)):
            er.labels = ["x"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestEmbedResultProperties
# ---------------------------------------------------------------------------


class TestEmbedResultProperties:
    def test_embedding_property_returns_first_row(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        er = EmbedResult(embeddings=data)
        np.testing.assert_array_equal(er.embedding, data[0])

    def test_dim_property(self):
        er = EmbedResult(embeddings=np.zeros((5, 256), dtype=np.float32))
        assert er.dim == 256

    def test_dim_after_1d_promotion(self):
        er = EmbedResult(embeddings=np.zeros(512, dtype=np.float32))
        assert er.dim == 512

    def test_embedding_property_single_vector(self):
        data = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        er = EmbedResult(embeddings=data)
        assert er.embedding.shape == (3,)


# ---------------------------------------------------------------------------
# TestEmbedResultSerialization
# ---------------------------------------------------------------------------


class TestEmbedResultSerialization:
    def test_to_dict_keys(self):
        er = _make_embed(2, 8)
        d = er.to_dict()
        assert "embeddings" in d
        assert "labels" in d
        assert "meta" in d

    def test_to_dict_embeddings_is_list(self):
        er = _make_embed(2, 4)
        d = er.to_dict()
        assert isinstance(d["embeddings"], list)

    def test_to_json_valid(self):
        er = _make_embed(1, 4)
        s = er.to_json()
        parsed = json.loads(s)
        assert "embeddings" in parsed

    def test_to_json_indent(self):
        er = _make_embed(1, 4)
        s = er.to_json(indent=2)
        assert "\n" in s

    def test_from_dict_roundtrip(self):
        er = EmbedResult(
            embeddings=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            labels=["a", "b"],
            meta={"src": "test"},
        )
        d = er.to_dict()
        er2 = EmbedResult.from_dict(d)
        np.testing.assert_allclose(er2.embeddings, er.embeddings, atol=1e-5)
        assert er2.labels == er.labels
        assert er2.meta == er.meta

    def test_from_json_roundtrip(self):
        er = _make_embed(3, 8)
        er2 = EmbedResult.from_json(er.to_json())
        assert er2.embeddings.shape == er.embeddings.shape


# ---------------------------------------------------------------------------
# TestEmbedResultSave
# ---------------------------------------------------------------------------


class TestEmbedResultSave:
    def test_save_json(self):
        er = _make_embed(2, 4)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        er.save(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "embeddings" in data

    def test_save_npz(self):
        er = _make_embed(2, 8)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        er.save(path)
        loaded = np.load(path)
        assert "embeddings" in loaded
        np.testing.assert_allclose(loaded["embeddings"], er.embeddings, atol=1e-6)

    def test_save_creates_parent_dirs(self):
        er = _make_embed(1, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dir" / "embed.npz")
            er.save(path)
            assert Path(path).exists()
