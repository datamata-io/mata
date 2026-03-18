"""Unit tests for EmbedAdapter.

All tests use mocks — no real model downloads or GPU required.
Run independently: pytest tests/test_embed_adapter.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from mata.core.artifacts.image import Image
    from mata.core.artifacts.rois import ROIs

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoder(dim: int = 128) -> MagicMock:
    """Return a mock encoder that mimics ReIDAdapter behaviour."""
    encoder = MagicMock()
    encoder.embedding_dim = dim
    encoder.info.return_value = {"model_id": "mock/model", "device": "cpu"}
    encoder.predict.side_effect = lambda crops: (
        np.zeros((0, 0), dtype=np.float32) if len(crops) == 0 else np.ones((len(crops), dim), dtype=np.float32)
    )
    return encoder


def _make_image(h: int = 64, w: int = 64) -> Image:
    from mata.core.artifacts.image import Image

    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.from_numpy(arr)


def _make_rois(n: int = 3, h: int = 32, w: int = 32) -> ROIs:
    from mata.core.artifacts.rois import ROIs

    crops = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
    boxes = [(0, 0, w, h)] * n
    return ROIs(roi_images=crops, source_boxes=boxes)


# ---------------------------------------------------------------------------
# TestEmbedAdapterProtocol
# ---------------------------------------------------------------------------


class TestEmbedAdapterProtocol:
    def test_conforms_to_embedder_protocol(self):
        """EmbedAdapter must satisfy the Embedder runtime-checkable protocol."""
        from mata.adapters.embed_adapter import EmbedAdapter
        from mata.core.registry.protocols import Embedder

        adapter = EmbedAdapter(encoder=_make_encoder())
        assert isinstance(adapter, Embedder)

    def test_has_embed_method(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        assert callable(EmbedAdapter(_make_encoder()).embed)

    def test_has_embedding_dim_property(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        assert hasattr(EmbedAdapter(_make_encoder()), "embedding_dim")

    def test_has_info_method(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        assert callable(EmbedAdapter(_make_encoder()).info)


# ---------------------------------------------------------------------------
# TestEmbedAdapterImage
# ---------------------------------------------------------------------------


class TestEmbedAdapterImage:
    def test_embed_image_returns_array(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder(dim=256))
        result = adapter.embed(_make_image())
        assert isinstance(result, np.ndarray)

    def test_embed_image_shape(self):
        """Single image → (1, D) array."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 512
        adapter = EmbedAdapter(encoder=_make_encoder(dim=dim))
        result = adapter.embed(_make_image())
        assert result.shape == (1, dim)

    def test_image_converted_to_numpy(self):
        """encoder.predict should receive a list with one numpy array."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed(_make_image())

        encoder.predict.assert_called_once()
        call_args = encoder.predict.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], np.ndarray)

    def test_embed_image_dtype_float32(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder(dim=64)
        encoder.predict.return_value = np.ones((1, 64), dtype=np.float32)
        adapter = EmbedAdapter(encoder=encoder)
        result = adapter.embed(_make_image())
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestEmbedAdapterROIs
# ---------------------------------------------------------------------------


class TestEmbedAdapterROIs:
    def test_embed_rois_returns_array(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder())
        result = adapter.embed(_make_rois(n=3))
        assert isinstance(result, np.ndarray)

    def test_embed_rois_shape_matches_count(self):
        """ROIs with N regions → (N, D) output."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 128
        n = 4
        adapter = EmbedAdapter(encoder=_make_encoder(dim=dim))
        result = adapter.embed(_make_rois(n=n))
        assert result.shape == (n, dim)

    def test_rois_converted_to_numpy_list(self):
        """encoder.predict should receive a list of N numpy arrays."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        n = 5
        adapter.embed(_make_rois(n=n))

        call_args = encoder.predict.call_args[0][0]
        assert len(call_args) == n
        for crop in call_args:
            assert isinstance(crop, np.ndarray)

    def test_empty_rois_returns_empty(self):
        """Empty ROIs artifact → empty (0, 0) array without calling encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter
        from mata.core.artifacts.rois import ROIs

        encoder = _make_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        empty_rois = ROIs(roi_images=[], source_boxes=[])
        result = adapter.embed(empty_rois)

        assert result.shape[0] == 0
        encoder.predict.assert_not_called()

    def test_single_roi(self):
        """Single ROI → (1, D) array."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 64
        adapter = EmbedAdapter(encoder=_make_encoder(dim=dim))
        result = adapter.embed(_make_rois(n=1))
        assert result.shape == (1, dim)


# ---------------------------------------------------------------------------
# TestEmbedAdapterErrorHandling
# ---------------------------------------------------------------------------


class TestEmbedAdapterErrorHandling:
    def test_embed_invalid_type_raises_typeerror(self):
        """Non-Image/ROIs input must raise TypeError."""
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder())
        with pytest.raises(TypeError, match="EmbedAdapter.embed()"):
            adapter.embed("not_an_image")  # type: ignore[arg-type]

    def test_embed_none_raises_typeerror(self):
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder())
        with pytest.raises(TypeError):
            adapter.embed(None)  # type: ignore[arg-type]

    def test_encoder_error_propagated(self):
        """If encoder.predict raises, EmbedAdapter must propagate it."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        encoder.predict.side_effect = RuntimeError("model failure")
        adapter = EmbedAdapter(encoder=encoder)

        with pytest.raises(RuntimeError, match="model failure"):
            adapter.embed(_make_image())


# ---------------------------------------------------------------------------
# TestEmbedAdapterDelegation
# ---------------------------------------------------------------------------


class TestEmbedAdapterDelegation:
    def test_embedding_dim_delegates(self):
        """embedding_dim property must delegate to encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 768
        encoder = _make_encoder(dim=dim)
        adapter = EmbedAdapter(encoder=encoder)
        assert adapter.embedding_dim == dim

    def test_embedding_dim_none_before_predict(self):
        """encoder may return None before first predict; adapter must propagate."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        encoder.embedding_dim = None
        adapter = EmbedAdapter(encoder=encoder)
        assert adapter.embedding_dim is None

    def test_info_returns_embed_type(self):
        """info() must include 'type': 'embed'."""
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder())
        info = adapter.info()
        assert info.get("type") == "embed"

    def test_info_includes_base_encoder_fields(self):
        """info() must include fields from the underlying encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        encoder.info.return_value = {"model_id": "mock/encoder", "device": "cuda"}
        adapter = EmbedAdapter(encoder=encoder)
        info = adapter.info()

        assert info["model_id"] == "mock/encoder"
        assert info["device"] == "cuda"
        assert info["type"] == "embed"

    def test_encoder_predict_called_correctly(self):
        """Verify predict is called with list[ndarray], not individual arrays."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed(_make_rois(n=2))

        encoder.predict.assert_called_once()
        arg = encoder.predict.call_args[0][0]
        assert isinstance(arg, list)

    def test_embed_with_kwargs_forwarded(self):
        """Extra kwargs should reach encoder.predict (if encoder supports them)."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = MagicMock()
        encoder.predict.return_value = np.ones((1, 64), dtype=np.float32)
        adapter = EmbedAdapter(encoder=encoder)

        # Should not raise even when kwargs present
        adapter.embed(_make_image(), normalize=False)
        encoder.predict.assert_called_once()
