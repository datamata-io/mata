"""Unit tests for XCLIPAdapter.

All tests use mocks — no real model downloads or GPU required.
Run independently: pytest tests/test_xclip_adapter.py -v

⚠️ Patch target note (I7):
  AutoModel and AutoProcessor are lazy-imported inside __init__() (lazy), so
  they are never module-level names in xclip_adapter.py.
  Patch at the transformers source: "transformers.AutoModel" / "transformers.AutoProcessor".

⚠️ HuggingFace lazy-module note:
  transformers uses _LazyModule for deferred imports.  The very first access to
  transformers.AutoModel triggers real class loading, which means patch() may
  race against that loading on the first test.  We force-resolve both symbols at
  module import time (below) so all patches work reliably from test #1 onward.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Force HuggingFace lazy imports to resolve BEFORE any patch() calls so that
# patch("transformers.AutoModel") works reliably on the very first test.
import transformers as _transformers

_ = _transformers.AutoModel
_ = _transformers.AutoProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frames(n: int = 8, h: int = 224, w: int = 224) -> list[np.ndarray]:
    """Return a list of n BGR uint8 frames."""
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def _make_mock_adapter(n_frames: int = 8, dim: int = 512):
    """Create XCLIPAdapter bypassing __init__ entirely — no HuggingFace calls.

    Uses object.__new__ to skip __init__ and manually sets all instance
    attributes, guaranteeing zero network access and zero real model loading
    regardless of transformers lazy-import behaviour.
    """
    import torch as _torch

    from mata.adapters.xclip_adapter import XCLIPAdapter

    adapter = object.__new__(XCLIPAdapter)
    adapter.model_id = "microsoft/xclip-base-patch32"
    adapter.n_frames = n_frames
    adapter._embedding_dim = None
    adapter._device = _torch.device("cpu")
    # Dummy text inputs (normally populated in __init__; empty dict is fine
    # for unit tests because the model call is fully mocked).
    adapter._dummy_text_inputs = {}

    # Full forward pass (predict_video): model(**merged) → out.video_embeds
    mock_forward_out = MagicMock()
    mock_forward_out.video_embeds = _torch.ones(1, dim)

    # Text sub-model (predict_text): text_model(...) → pooler_output
    mock_text_out = MagicMock()
    mock_text_out.pooler_output = _torch.ones(1, dim)

    mock_model = MagicMock()
    mock_model.return_value = mock_forward_out  # model(**merged)
    mock_model.text_model.return_value = mock_text_out  # text_model(...)
    mock_model.text_projection.return_value = _torch.ones(1, dim)
    adapter._model = mock_model

    # Processor: images= → pixel_values dict; text= → input_ids/attention_mask
    def _proc_effect(*args, **kwargs):
        if "images" in kwargs:
            return {"pixel_values": _torch.zeros(1, n_frames, 3, 224, 224)}
        return {
            "input_ids": _torch.zeros(1, 4, dtype=_torch.long),
            "attention_mask": _torch.ones(1, 4, dtype=_torch.long),
        }

    mock_processor = MagicMock()
    mock_processor.side_effect = _proc_effect
    adapter._processor = mock_processor

    return adapter


# ---------------------------------------------------------------------------
# TestPredictVideoBasic  (8 tests)
# ---------------------------------------------------------------------------


class TestPredictVideoBasic:
    def test_returns_ndarray(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_video(_make_frames(8))
        assert isinstance(result, np.ndarray)

    def test_shape_is_1_by_512(self):
        adapter = _make_mock_adapter(dim=512)
        result = adapter.predict_video(_make_frames(8))
        assert result.shape == (1, 512)

    def test_dtype_is_float32(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_video(_make_frames(8))
        assert result.dtype == np.float32

    def test_output_is_l2_normalized(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_video(_make_frames(8))
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_dim_set_after_call(self):
        adapter = _make_mock_adapter(dim=512)
        assert adapter.embedding_dim is None  # before call
        adapter.predict_video(_make_frames(8))
        assert adapter.embedding_dim == 512

    def test_exact_n_frames_accepted(self):
        """Exactly n_frames frames should work without resampling error."""
        adapter = _make_mock_adapter(n_frames=8)
        result = adapter.predict_video(_make_frames(8))
        assert result.shape == (1, 512)

    def test_more_than_n_frames_resampled(self):
        """16 frames fed to an 8-frame adapter should still produce (1, 512)."""
        adapter = _make_mock_adapter(n_frames=8)
        result = adapter.predict_video(_make_frames(16))
        assert result.shape == (1, 512)

    def test_fewer_than_n_frames_repeated(self):
        """3 frames fed to an 8-frame adapter should be repeated to fill."""
        adapter = _make_mock_adapter(n_frames=8)
        result = adapter.predict_video(_make_frames(3))
        assert result.shape == (1, 512)

    def test_single_frame_repeated_to_n_frames(self):
        adapter = _make_mock_adapter(n_frames=8)
        result = adapter.predict_video(_make_frames(1))
        assert result.shape == (1, 512)


# ---------------------------------------------------------------------------
# TestPredictVideoEdgeCases  (4 tests)
# ---------------------------------------------------------------------------


class TestPredictVideoEdgeCases:
    def test_empty_frame_list_raises_value_error(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError):
            adapter.predict_video([])

    def test_bgr_to_rgb_conversion(self):
        """BGR→RGB flip should happen inside predict_video (frames are not mutated)."""
        adapter = _make_mock_adapter()
        # Use a frame with distinct R and B channels to test the flip
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, 0] = 10  # B
        frame[:, :, 2] = 200  # R
        original_b = frame[:, :, 0].copy()
        original_r = frame[:, :, 2].copy()

        adapter.predict_video([frame] * 8)

        # Original frame should NOT be mutated (in-place flip would change it)
        np.testing.assert_array_equal(frame[:, :, 0], original_b)
        np.testing.assert_array_equal(frame[:, :, 2], original_r)

    def test_large_frame_does_not_crash(self):
        """1080p frames should not raise (memory permitting with mocked model)."""
        adapter = _make_mock_adapter()
        large_frames = _make_frames(8, h=1080, w=1920)
        result = adapter.predict_video(large_frames)
        assert result.shape == (1, 512)

    def test_non_standard_n_frames_value(self):
        """n_frames != 8 should still produce correct shape."""
        adapter = _make_mock_adapter(n_frames=16)
        result = adapter.predict_video(_make_frames(16))
        assert result.shape == (1, 512)


# ---------------------------------------------------------------------------
# TestPredictTextBasic  (6 tests)
# ---------------------------------------------------------------------------


class TestPredictTextBasic:
    def test_returns_ndarray(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_text("a red truck on the highway")
        assert isinstance(result, np.ndarray)

    def test_shape_is_1_by_512(self):
        adapter = _make_mock_adapter(dim=512)
        result = adapter.predict_text("red truck")
        assert result.shape == (1, 512)

    def test_dtype_is_float32(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_text("red truck")
        assert result.dtype == np.float32

    def test_output_is_l2_normalized(self):
        adapter = _make_mock_adapter()
        result = adapter.predict_text("red truck")
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_dim_set_after_call(self):
        adapter = _make_mock_adapter(dim=512)
        assert adapter.embedding_dim is None
        adapter.predict_text("red truck")
        assert adapter.embedding_dim == 512

    def test_empty_string_accepted(self):
        """Empty string should not raise; result shape same as normal query."""
        adapter = _make_mock_adapter()
        result = adapter.predict_text("")
        assert result.shape == (1, 512)

    def test_long_string_accepted(self):
        adapter = _make_mock_adapter()
        long_text = "a " * 200
        result = adapter.predict_text(long_text)
        assert result.shape == (1, 512)


# ---------------------------------------------------------------------------
# TestVideoTextAlignment  (3 tests)
# ---------------------------------------------------------------------------


class TestVideoTextAlignment:
    def test_both_return_same_embedding_dim(self):
        adapter = _make_mock_adapter(dim=512)
        adapter.predict_video(_make_frames(8))
        video_dim = adapter.embedding_dim
        adapter.predict_text("red truck")
        text_dim = adapter.embedding_dim
        assert video_dim == text_dim == 512

    def test_both_norms_approximately_one(self):
        adapter = _make_mock_adapter()
        v_result = adapter.predict_video(_make_frames(8))
        t_result = adapter.predict_text("red truck")
        assert abs(np.linalg.norm(v_result[0]) - 1.0) < 1e-5
        assert abs(np.linalg.norm(t_result[0]) - 1.0) < 1e-5

    def test_shapes_match(self):
        adapter = _make_mock_adapter()
        v_result = adapter.predict_video(_make_frames(8))
        t_result = adapter.predict_text("red truck")
        assert v_result.shape == t_result.shape


# ---------------------------------------------------------------------------
# TestInfoDict  (4 tests)
# ---------------------------------------------------------------------------


class TestInfoDict:
    def test_info_contains_type_xclip(self):
        adapter = _make_mock_adapter()
        assert adapter.info()["type"] == "xclip"

    def test_info_contains_model_id(self):
        adapter = _make_mock_adapter()
        assert adapter.info()["model_id"] == "microsoft/xclip-base-patch32"

    def test_info_contains_n_frames(self):
        adapter = _make_mock_adapter(n_frames=16)
        assert adapter.info()["n_frames"] == 16

    def test_info_contains_device(self):
        adapter = _make_mock_adapter()
        assert "device" in adapter.info()

    def test_info_contains_embedding_dim(self):
        adapter = _make_mock_adapter()
        info_before = adapter.info()
        assert "embedding_dim" in info_before

        adapter.predict_video(_make_frames(8))
        info_after = adapter.info()
        assert info_after["embedding_dim"] == 512


# ---------------------------------------------------------------------------
# TestEmbeddingDimProperty  (2 tests)
# ---------------------------------------------------------------------------


class TestEmbeddingDimProperty:
    def test_none_before_first_call(self):
        adapter = _make_mock_adapter()
        assert adapter.embedding_dim is None

    def test_correct_value_after_predict_video(self):
        adapter = _make_mock_adapter(dim=512)
        adapter.predict_video(_make_frames(8))
        assert adapter.embedding_dim == 512

    def test_correct_value_after_predict_text(self):
        adapter = _make_mock_adapter(dim=512)
        adapter.predict_text("hello")
        assert adapter.embedding_dim == 512


# ---------------------------------------------------------------------------
# TestResampleFrames  (4 tests)
# ---------------------------------------------------------------------------


class TestResampleFrames:
    def _make_adapter(self, n_frames: int = 8) -> object:
        return _make_mock_adapter(n_frames=n_frames)

    def test_exact_match_returns_same_list(self):
        adapter = _make_mock_adapter(n_frames=8)
        frames = _make_frames(8)
        resampled = adapter._resample_frames(frames)
        assert len(resampled) == 8

    def test_upsample_short_list(self):
        """3 frames → 8-frame adapter should repeat to fill exactly 8."""
        adapter = _make_mock_adapter(n_frames=8)
        frames = _make_frames(3)
        resampled = adapter._resample_frames(frames)
        assert len(resampled) == 8

    def test_downsample_long_list(self):
        """32 frames → 8-frame adapter should subsample to exactly 8."""
        adapter = _make_mock_adapter(n_frames=8)
        frames = _make_frames(32)
        resampled = adapter._resample_frames(frames)
        assert len(resampled) == 8

    def test_single_frame_to_n_frames(self):
        """1 frame → 8-frame adapter should repeat to 8."""
        adapter = _make_mock_adapter(n_frames=8)
        frames = _make_frames(1)
        resampled = adapter._resample_frames(frames)
        assert len(resampled) == 8

    def test_empty_list_raises_value_error(self):
        adapter = _make_mock_adapter(n_frames=8)
        with pytest.raises(ValueError, match="empty"):
            adapter._resample_frames([])


# ---------------------------------------------------------------------------
# TestInitLoading  (4 tests)
# ---------------------------------------------------------------------------


class TestInitLoading:
    def test_automodel_from_pretrained_called_with_model_id(self):
        with (
            patch("transformers.AutoModel") as mock_model_cls,
            patch("transformers.AutoProcessor") as mock_proc_cls,
        ):
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_proc_cls.from_pretrained.return_value = MagicMock()

            from mata.adapters.xclip_adapter import XCLIPAdapter

            XCLIPAdapter("microsoft/xclip-base-patch32", device="cpu")
            mock_model_cls.from_pretrained.assert_called_once_with("microsoft/xclip-base-patch32")

    def test_autoprocessor_from_pretrained_called_with_model_id(self):
        with (
            patch("transformers.AutoModel") as mock_model_cls,
            patch("transformers.AutoProcessor") as mock_proc_cls,
        ):
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_proc_cls.from_pretrained.return_value = MagicMock()

            from mata.adapters.xclip_adapter import XCLIPAdapter

            XCLIPAdapter("microsoft/xclip-base-patch32", device="cpu")
            mock_proc_cls.from_pretrained.assert_called_once_with("microsoft/xclip-base-patch32")

    def test_model_eval_called(self):
        with (
            patch("transformers.AutoModel") as mock_model_cls,
            patch("transformers.AutoProcessor") as mock_proc_cls,
        ):
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = MagicMock()

            from mata.adapters.xclip_adapter import XCLIPAdapter

            XCLIPAdapter("microsoft/xclip-base-patch32", device="cpu")
            mock_model.eval.assert_called_once()

    def test_device_placement(self):
        with (
            patch("transformers.AutoModel") as mock_model_cls,
            patch("transformers.AutoProcessor") as mock_proc_cls,
        ):
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = MagicMock()

            from mata.adapters.xclip_adapter import XCLIPAdapter

            XCLIPAdapter("microsoft/xclip-base-patch32", device="cpu")
            mock_model.to.assert_called()

    def test_n_frames_stored(self):
        adapter = _make_mock_adapter(n_frames=16)
        assert adapter.n_frames == 16

    def test_model_id_stored(self):
        adapter = _make_mock_adapter()
        assert adapter.model_id == "microsoft/xclip-base-patch32"
