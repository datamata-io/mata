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
        """Non-Image/ROIs/list/str input must raise TypeError."""
        from mata.adapters.embed_adapter import EmbedAdapter

        adapter = EmbedAdapter(encoder=_make_encoder())
        with pytest.raises(TypeError, match="EmbedAdapter.embed()"):
            adapter.embed(42)  # type: ignore[arg-type]

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


# ---------------------------------------------------------------------------
# TestEmbedAdapterXCLIPDispatch
# ---------------------------------------------------------------------------


def _make_xclip_encoder(dim: int = 512) -> MagicMock:
    """Return a mock encoder that mimics XCLIPAdapter behaviour."""
    encoder = MagicMock()
    encoder.embedding_dim = dim
    encoder.info.return_value = {"model_id": "microsoft/xclip-base-patch32", "device": "cpu"}
    encoder.predict_video.return_value = np.ones((1, dim), dtype=np.float32)
    encoder.predict_text.return_value = np.ones((1, dim), dtype=np.float32)
    encoder.predict.side_effect = lambda crops: np.ones((len(crops), dim), dtype=np.float32)
    return encoder


def _make_non_xclip_encoder() -> MagicMock:
    """Return a mock encoder with no predict_video / predict_text (e.g. ViT or ONNX)."""
    encoder = MagicMock(spec=["embedding_dim", "info", "predict"])
    encoder.embedding_dim = 512
    encoder.info.return_value = {"model_id": "google/vit-base-patch16-224", "device": "cpu"}
    encoder.predict.return_value = np.ones((1, 512), dtype=np.float32)
    return encoder


class TestEmbedAdapterXCLIPDispatch:
    def test_embed_list_dispatches_to_predict_video(self):
        """embed(list) must call encoder.predict_video, not predict."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        adapter.embed([frame])

        encoder.predict_video.assert_called_once_with([frame])
        encoder.predict.assert_not_called()

    def test_embed_str_dispatches_to_predict_text(self):
        """embed(str) must call encoder.predict_text, not predict."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed("red truck")

        encoder.predict_text.assert_called_once_with("red truck")
        encoder.predict.assert_not_called()

    def test_embed_text_method_same_as_embed_str(self):
        """embed_text(q) must return the same result as embed(q)."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder(dim=512)
        adapter = EmbedAdapter(encoder=encoder)

        result_embed = adapter.embed("query text")
        result_method = adapter.embed_text("query text")

        np.testing.assert_array_equal(result_embed, result_method)

    def test_embed_list_no_predict_video_falls_back_to_mean_pool(self):
        """list input on a non-xclip encoder mean-pools individual frame embeddings.

        Image encoders like CLIP lack predict_video(); embed() falls back to
        calling predict() on each frame and averaging the results.
        """
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_non_xclip_encoder()  # has predict() but no predict_video()
        adapter = EmbedAdapter(encoder=encoder)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        result = adapter.embed([frame, frame, frame])

        assert result.shape == (1, 512)
        assert result.dtype == np.float32
        # L2-normalised
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)

    def test_embed_str_no_predict_text_raises(self):
        """str input on a non-xclip encoder must raise UnsupportedModelError."""
        from mata.adapters.embed_adapter import EmbedAdapter
        from mata.core.exceptions import UnsupportedModelError

        encoder = _make_non_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)

        with pytest.raises(UnsupportedModelError):
            adapter.embed("some query")

    def test_embed_list_non_xclip_calls_predict_not_predict_video(self):
        """list input on a non-xclip encoder must use predict(), not predict_video()."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_non_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        adapter.embed([frame])

        encoder.predict.assert_called_once_with([frame])

    def test_embed_str_error_message_mentions_xclip(self):
        """UnsupportedModelError for str input must name the xclip model."""
        from mata.adapters.embed_adapter import EmbedAdapter
        from mata.core.exceptions import UnsupportedModelError

        encoder = _make_non_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)

        with pytest.raises(UnsupportedModelError, match="microsoft/xclip-base-patch32"):
            adapter.embed("some query")

    def test_embed_image_unaffected_by_xclip_methods(self):
        """Normal Image input still routes to predict() on an xclip encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder(dim=512)
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed(_make_image())

        encoder.predict.assert_called_once()
        encoder.predict_video.assert_not_called()
        encoder.predict_text.assert_not_called()

    def test_embed_rois_unaffected_by_xclip_methods(self):
        """Normal ROIs input still routes to predict() on an xclip encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder(dim=512)
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed(_make_rois(n=2))

        encoder.predict.assert_called_once()
        encoder.predict_video.assert_not_called()
        encoder.predict_text.assert_not_called()

    def test_embed_list_empty_delegates_empty_to_encoder(self):
        """Empty list must be passed through to predict_video([])."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed([])

        encoder.predict_video.assert_called_once_with([])

    def test_embed_list_returns_ndarray(self):
        """embed(list) must return an np.ndarray."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder(dim=512)
        adapter = EmbedAdapter(encoder=encoder)
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = adapter.embed([frame])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 512)

    def test_embed_str_returns_ndarray(self):
        """embed(str) must return an np.ndarray."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_xclip_encoder(dim=512)
        adapter = EmbedAdapter(encoder=encoder)
        result = adapter.embed("fire hydrant")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 512)


# ---------------------------------------------------------------------------
# Helpers for Qwen3-VL tests
# ---------------------------------------------------------------------------


def _make_qwen3_encoder(dim: int = 2048) -> MagicMock:
    """Return a mock encoder mimicking Qwen3VLEmbeddingAdapter behaviour."""
    encoder = MagicMock()
    encoder.embedding_dim = dim
    # Use side_effect so each call returns a fresh dict — EmbedAdapter.info()
    # mutates the returned dict in-place, which would corrupt a shared return_value.
    encoder.info.side_effect = lambda: {
        "type": "qwen3_vl_embedding",
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
        "native_dim": dim,
        "embedding_dim": dim,
        "device": "cpu",
    }
    encoder.predict.side_effect = lambda crops: (
        np.zeros((0, 0), dtype=np.float32) if len(crops) == 0 else np.ones((len(crops), dim), dtype=np.float32)
    )
    encoder.predict_image.return_value = np.ones((1, dim), dtype=np.float32)
    encoder.predict_text.return_value = np.ones((1, dim), dtype=np.float32)
    encoder.predict_video.return_value = np.ones((1, dim), dtype=np.float32)
    encoder.predict_multimodal.return_value = np.ones((1, dim), dtype=np.float32)
    return encoder


# ---------------------------------------------------------------------------
# TestEmbedAdapterQwen3VLDispatch
# ---------------------------------------------------------------------------


class TestEmbedAdapterQwen3VLDispatch:
    def test_embed_image_delegates_to_predict(self):
        """Image → encoder.predict([np_image]) even when encoder has predict_video."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_qwen3_encoder(dim=2048)
        adapter = EmbedAdapter(encoder=encoder)
        adapter.embed(_make_image())

        encoder.predict.assert_called_once()
        encoder.predict_video.assert_not_called()

    def test_embed_adapter_wraps_qwen3_encoder(self):
        """EmbedAdapter(encoder=qwen3_mock) stores the encoder on _encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_qwen3_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        assert adapter._encoder is encoder

    def test_embedding_dim_from_qwen3_encoder(self):
        """embedding_dim delegates to Qwen3VL encoder's embedding_dim property."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 2048
        encoder = _make_qwen3_encoder(dim=dim)
        adapter = EmbedAdapter(encoder=encoder)
        assert adapter.embedding_dim == dim

    def test_info_wraps_qwen3_encoder_info(self):
        """info() merges Qwen3VL encoder fields; top-level type is 'embed'.

        The encoder's own info() reports type 'qwen3_vl_embedding', but
        EmbedAdapter.info() overrides type to 'embed' and merges the rest.
        """
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_qwen3_encoder(dim=2048)
        adapter = EmbedAdapter(encoder=encoder)
        info = adapter.info()

        # EmbedAdapter normalises type to "embed"
        assert info["type"] == "embed"
        # Qwen3VL-specific fields from the encoder are present in the merged dict
        assert info["model_id"] == "Qwen/Qwen3-VL-Embedding-2B"
        # Encoder's own info carries the "qwen3_vl_embedding" type identifier
        assert encoder.info()["type"] == "qwen3_vl_embedding"

    def test_embed_rois_with_qwen3_encoder(self):
        """ROIs dispatch calls encoder.predict(crops_list) on Qwen3 adapter."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 2048
        n = 3
        encoder = _make_qwen3_encoder(dim=dim)
        adapter = EmbedAdapter(encoder=encoder)
        result = adapter.embed(_make_rois(n=n))

        encoder.predict.assert_called_once()
        call_arg = encoder.predict.call_args[0][0]
        assert isinstance(call_arg, list)
        assert len(call_arg) == n
        assert result.shape == (n, dim)

    def test_embed_empty_rois_with_qwen3(self):
        """Empty ROIs returns (0, 0) array without calling encoder.predict."""
        from mata.adapters.embed_adapter import EmbedAdapter
        from mata.core.artifacts.rois import ROIs

        encoder = _make_qwen3_encoder()
        adapter = EmbedAdapter(encoder=encoder)
        empty_rois = ROIs(roi_images=[], source_boxes=[])
        result = adapter.embed(empty_rois)

        assert result.shape[0] == 0
        encoder.predict.assert_not_called()

    def test_embed_with_kwargs_forwarded_to_qwen3(self):
        """Extra kwargs passed to embed() do not raise with a Qwen3 encoder."""
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = _make_qwen3_encoder(dim=2048)
        adapter = EmbedAdapter(encoder=encoder)
        result = adapter.embed(_make_image(), normalize=True)

        assert isinstance(result, np.ndarray)
        encoder.predict.assert_called_once()

    def test_existing_clip_path_unchanged(self):
        """CLIP encoder (predict only, no predict_video) still routes via predict()."""
        from mata.adapters.embed_adapter import EmbedAdapter

        dim = 512
        clip_encoder = MagicMock(spec=["embedding_dim", "info", "predict"])
        clip_encoder.embedding_dim = dim
        clip_encoder.info.return_value = {
            "model_id": "openai/clip-vit-base-patch32",
            "device": "cpu",
        }
        clip_encoder.predict.return_value = np.ones((1, dim), dtype=np.float32)

        adapter = EmbedAdapter(encoder=clip_encoder)
        result = adapter.embed(_make_image())

        clip_encoder.predict.assert_called_once()
        assert result.shape == (1, dim)
