"""Unit tests for Qwen3VLEmbeddingAdapter.

All tests use mocks -- no real model downloads or GPU required.
Run independently: pytest tests/test_qwen3_vl_embedding_adapter.py -v

Patch target note:
  AutoModel and AutoProcessor are lazy-imported inside __init__() (lazy import
  pattern), so they are never module-level names in qwen3_vl_embedding_adapter.py.
  Patch at the transformers source: "transformers.AutoModel" / "transformers.AutoProcessor".

HuggingFace lazy-module note:
  transformers uses _LazyModule for deferred imports. We force-resolve both
  symbols at module import time so all patch() calls work reliably from test #1.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Force HuggingFace lazy imports to resolve BEFORE any patch() calls so that
# patch("transformers.AutoModel") works reliably on the very first test.
import transformers as _transformers

_ = _transformers.AutoModel
_ = _transformers.AutoProcessor

# Pre-import the adapter module (and class) so it is cached in sys.modules.
# The lazy import of AutoModel/AutoProcessor happens at __init__ call time,
# not at module import time, so patches will still intercept them correctly.
from mata.adapters.qwen3_vl_embedding_adapter import Qwen3VLEmbeddingAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
_DIM = 512  # embedding dimension used by all mocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bgr_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random BGR uint8 ndarray."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_mock_adapter(dim: int = _DIM, embed_dim: int | None = None) -> Qwen3VLEmbeddingAdapter:
    """Create Qwen3VLEmbeddingAdapter bypassing __init__ -- no HuggingFace calls.

    Uses object.__new__ to skip __init__ and manually sets all instance
    attributes so zero network access or model loading occurs.
    """
    import torch

    adapter = object.__new__(Qwen3VLEmbeddingAdapter)
    adapter.model_id = _MODEL_ID
    adapter._embed_dim = embed_dim
    adapter._native_dim = None
    adapter._embedding_dim = None
    adapter.fps = 1.0
    adapter.max_frames = 64
    adapter._device = torch.device("cpu")
    adapter._dtype = torch.float32

    # Build mock hidden_states output.
    # hidden_states is a list; last element: (1, seq_len, dim)
    hidden_tensor = torch.rand(1, 8, dim)  # (batch=1, seq=8, dim)
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = [hidden_tensor]

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs
    adapter._model = mock_model

    # Processor: apply_chat_template -> string, __call__ -> tensor dict
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "mock_prompt_text"
    mock_processor.return_value = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    adapter._processor = mock_processor

    return adapter


# ---------------------------------------------------------------------------
# TestInitialization (5 tests)
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_init_loads_model(self):
        """AutoModel.from_pretrained is called with the model_id."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor"),
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            Qwen3VLEmbeddingAdapter(_MODEL_ID, device="cpu")

        assert mock_auto_model.from_pretrained.called
        assert mock_auto_model.from_pretrained.call_args[0][0] == _MODEL_ID

    def test_init_processor_loaded(self):
        """AutoProcessor.from_pretrained is called with the model_id."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor") as mock_auto_processor,
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            Qwen3VLEmbeddingAdapter(_MODEL_ID, device="cpu")

        mock_auto_processor.from_pretrained.assert_called_once_with(_MODEL_ID)

    def test_init_auto_device_cpu(self):
        """device='auto' resolves to cpu when CUDA is not available."""
        import torch

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            adapter = Qwen3VLEmbeddingAdapter(_MODEL_ID, device="auto")

        assert adapter._device == torch.device("cpu")

    def test_init_dtype_bfloat16(self):
        """dtype='bfloat16' is resolved to torch.bfloat16."""
        import torch

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor"),
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            adapter = Qwen3VLEmbeddingAdapter(_MODEL_ID, device="cpu", dtype="bfloat16")

        assert adapter._dtype == torch.bfloat16

    def test_init_embed_dim_stored(self):
        """embed_dim kwarg is stored as _embed_dim."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor"),
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            adapter = Qwen3VLEmbeddingAdapter(_MODEL_ID, device="cpu", embed_dim=256)

        assert adapter._embed_dim == 256


# ---------------------------------------------------------------------------
# TestPredictImage (5 tests)
# ---------------------------------------------------------------------------


class TestPredictImage:
    def test_predict_image_shape(self):
        """predict_image returns (1, D) ndarray."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_image(_make_bgr_frame())
        assert result.shape == (1, _DIM)

    def test_predict_image_dtype_float32(self):
        """predict_image returns float32 array."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_image(_make_bgr_frame())
        assert result.dtype == np.float32

    def test_predict_image_l2_normalized(self):
        """predict_image output is L2-normalized (row norm ~= 1.0)."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_image(_make_bgr_frame())
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_bgr_to_rgb(self):
        """_to_pil converts BGR numpy array to RGB PIL Image."""
        from PIL import Image as PILImage

        adapter = _make_mock_adapter()
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # BGR: B=255 G=0 R=0
        pil = adapter._to_pil(bgr)
        assert isinstance(pil, PILImage.Image)
        rgb_arr = np.array(pil)
        # After BGR->RGB reversal: R=0, G=0, B=255
        assert rgb_arr[0, 0, 0] == 0  # R channel
        assert rgb_arr[0, 0, 2] == 255  # B channel

    def test_predict_image_pil_input(self):
        """predict_multimodal accepts PIL Image as image input."""
        from PIL import Image as PILImage

        adapter = _make_mock_adapter(dim=_DIM)
        pil_img = PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        result = adapter.predict_multimodal({"image": pil_img})
        assert result.shape == (1, _DIM)


# ---------------------------------------------------------------------------
# TestPredictText (4 tests)
# ---------------------------------------------------------------------------


class TestPredictText:
    def test_predict_text_shape(self):
        """predict_text returns (1, D) ndarray."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_text("a red truck")
        assert result.shape == (1, _DIM)

    def test_predict_text_dtype(self):
        """predict_text returns float32 array."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_text("hello world")
        assert result.dtype == np.float32

    def test_predict_text_normalized(self):
        """predict_text output is L2-normalized."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_text("query text")
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_predict_text_delegates_to_multimodal(self, monkeypatch):
        """predict_text calls predict_multimodal({'text': <text>})."""
        adapter = _make_mock_adapter()
        captured = []
        original = adapter.predict_multimodal

        def _spy(d):
            captured.append(d)
            return original(d)

        monkeypatch.setattr(adapter, "predict_multimodal", _spy)
        adapter.predict_text("hello")
        assert captured == [{"text": "hello"}]


# ---------------------------------------------------------------------------
# TestPredictVideo (4 tests)
# ---------------------------------------------------------------------------


class TestPredictVideo:
    def test_predict_video_shape(self):
        """predict_video returns (1, D) ndarray."""
        adapter = _make_mock_adapter(dim=_DIM)
        frames = [_make_bgr_frame() for _ in range(4)]
        result = adapter.predict_video(frames)
        assert result.shape == (1, _DIM)

    def test_predict_video_frame_sampling(self):
        """_sample_frames reduces to max_frames when given more frames."""
        adapter = _make_mock_adapter()
        adapter.max_frames = 4
        many_frames = [_make_bgr_frame() for _ in range(20)]
        sampled = adapter._sample_frames(many_frames)
        assert len(sampled) == 4

    def test_predict_video_empty_raises(self):
        """predict_video with empty frame list raises ValueError."""
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="Empty frame list"):
            adapter.predict_video([])

    def test_predict_video_delegates_to_multimodal(self, monkeypatch):
        """predict_video calls predict_multimodal({'video': frames})."""
        adapter = _make_mock_adapter()
        frames = [_make_bgr_frame()]
        captured = []
        original = adapter.predict_multimodal

        def _spy(d):
            captured.append(d)
            return original(d)

        monkeypatch.setattr(adapter, "predict_multimodal", _spy)
        adapter.predict_video(frames)
        assert len(captured) == 1
        assert "video" in captured[0]
        assert captured[0]["video"] is frames


# ---------------------------------------------------------------------------
# TestPredictMultimodal (5 tests)
# ---------------------------------------------------------------------------


class TestPredictMultimodal:
    def test_multimodal_text_image(self):
        """predict_multimodal handles text+image input."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_multimodal(
            {
                "text": "describe",
                "image": _make_bgr_frame(),
            }
        )
        assert result.shape == (1, _DIM)

    def test_multimodal_text_video(self):
        """predict_multimodal handles text+video input."""
        adapter = _make_mock_adapter(dim=_DIM)
        frames = [_make_bgr_frame() for _ in range(3)]
        result = adapter.predict_multimodal({"text": "describe", "video": frames})
        assert result.shape == (1, _DIM)

    def test_multimodal_all(self):
        """predict_multimodal handles text+image+video combined."""
        adapter = _make_mock_adapter(dim=_DIM)
        frames = [_make_bgr_frame() for _ in range(2)]
        result = adapter.predict_multimodal(
            {
                "text": "q",
                "image": _make_bgr_frame(),
                "video": frames,
            }
        )
        assert result.shape == (1, _DIM)

    def test_multimodal_empty_raises(self):
        """predict_multimodal with empty dict raises ValueError."""
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError, match="empty input"):
            adapter.predict_multimodal({})

    def test_multimodal_text_only(self):
        """predict_multimodal with text key only returns (1, D)."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict_multimodal({"text": "standalone text"})
        assert result.shape == (1, _DIM)


# ---------------------------------------------------------------------------
# TestPredictBatchCompat (3 tests)
# ---------------------------------------------------------------------------


class TestPredictBatchCompat:
    def test_predict_batch_shape(self):
        """predict(crops) returns (N, D) for N crops."""
        adapter = _make_mock_adapter(dim=_DIM)
        crops = [_make_bgr_frame() for _ in range(3)]
        result = adapter.predict(crops)
        assert result.shape == (3, _DIM)

    def test_predict_empty_crops(self):
        """predict([]) returns empty (0, 0) float32 array."""
        adapter = _make_mock_adapter()
        result = adapter.predict([])
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_predict_single_crop(self):
        """predict([crop]) returns (1, D) array."""
        adapter = _make_mock_adapter(dim=_DIM)
        result = adapter.predict([_make_bgr_frame()])
        assert result.shape == (1, _DIM)


# ---------------------------------------------------------------------------
# TestMatryoshkaTruncation (3 tests)
# ---------------------------------------------------------------------------


class TestMatryoshkaTruncation:
    def test_embed_dim_truncation(self):
        """embed_dim truncates output to the requested size."""
        adapter = _make_mock_adapter(dim=_DIM, embed_dim=64)
        result = adapter.predict_text("hello")
        assert result.shape == (1, 64)

    def test_embed_dim_renormalized(self):
        """After Matryoshka truncation, output is still L2-normalized."""
        adapter = _make_mock_adapter(dim=_DIM, embed_dim=64)
        result = adapter.predict_text("hello")
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_embed_dim_none_full(self):
        """embed_dim=None returns the full native dimensionality."""
        adapter = _make_mock_adapter(dim=_DIM, embed_dim=None)
        result = adapter.predict_text("hello")
        assert result.shape == (1, _DIM)


# ---------------------------------------------------------------------------
# TestEOSExtraction (2 tests)
# ---------------------------------------------------------------------------


class TestEOSExtraction:
    def test_eos_last_token(self):
        """The embedding comes from the last token position in hidden states."""
        import torch

        adapter = _make_mock_adapter(dim=_DIM)

        # Set all positions to zero except the last which is non-zero.
        # If EOS extraction is correct, result will be non-zero.
        hidden = torch.zeros(1, 8, _DIM)
        hidden[0, -1, :] = torch.ones(_DIM)  # only last token is non-zero

        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [hidden]
        adapter._model.return_value = mock_outputs

        result = adapter.predict_text("test")
        # Result derived from the non-zero last token
        assert np.linalg.norm(result[0]) > 0.5

    def test_eos_hidden_states_correct_layer(self):
        """Uses hidden_states[-1] (last layer), not an earlier layer."""
        import torch

        adapter = _make_mock_adapter(dim=_DIM)

        # Two layers: first all-zeros, last all-ones.
        # Correct extraction must use the last layer (all-ones) -> norm ~1.
        hidden_zero = torch.zeros(1, 4, _DIM)
        hidden_ones = torch.ones(1, 4, _DIM)

        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [hidden_zero, hidden_ones]
        adapter._model.return_value = mock_outputs

        result = adapter.predict_text("layer test")
        # Came from last-layer last-token (all-ones), normalized -> norm ~1.0
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TestQwenVlUtilsFallback (3 tests)
# ---------------------------------------------------------------------------


class TestQwenVlUtilsFallback:
    """Tests for the qwen-vl-utils optional dependency guard."""

    def test_without_qwen_vl_utils_no_crash(self):
        """_try_load_qwen_vl_utils sets AVAILABLE=False when package is missing."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        saved = (mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE)
        try:
            # None in sys.modules causes ImportError on `import qwen_vl_utils`
            with patch.dict(sys.modules, {"qwen_vl_utils": None}):
                mod._qwen_vl_utils = None
                mod._QWEN_VL_UTILS_AVAILABLE = None
                mod._try_load_qwen_vl_utils()
            assert mod._QWEN_VL_UTILS_AVAILABLE is False
        finally:
            mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE = saved

    def test_with_qwen_vl_utils_loaded(self):
        """_try_load_qwen_vl_utils sets AVAILABLE=True and returns the module."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        saved = (mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE)
        try:
            fake_utils = MagicMock()
            with patch.dict(sys.modules, {"qwen_vl_utils": fake_utils}):
                mod._qwen_vl_utils = None
                mod._QWEN_VL_UTILS_AVAILABLE = None
                result = mod._try_load_qwen_vl_utils()
            assert mod._QWEN_VL_UTILS_AVAILABLE is True
            assert result is fake_utils
        finally:
            mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE = saved

    def test_cached_import(self):
        """_try_load_qwen_vl_utils returns cached result on subsequent calls."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        saved = (mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE)
        try:
            # Pre-seed: already evaluated as unavailable
            mod._QWEN_VL_UTILS_AVAILABLE = False
            mod._qwen_vl_utils = None

            # Even with qwen_vl_utils now "importable", cached result wins
            fake_utils = MagicMock()
            with patch.dict(sys.modules, {"qwen_vl_utils": fake_utils}):
                result = mod._try_load_qwen_vl_utils()

            # Must return cached None (not fake_utils)
            assert result is None
            assert mod._QWEN_VL_UTILS_AVAILABLE is False
        finally:
            mod._qwen_vl_utils, mod._QWEN_VL_UTILS_AVAILABLE = saved


# ---------------------------------------------------------------------------
# TestInfo (3 tests)
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_contains_type(self):
        """`info()['type']` is 'qwen3_vl_embedding'."""
        adapter = _make_mock_adapter()
        assert adapter.info()["type"] == "qwen3_vl_embedding"

    def test_info_contains_model_id(self):
        """`info()['model_id']` matches the model_id used at construction."""
        adapter = _make_mock_adapter()
        assert adapter.info()["model_id"] == _MODEL_ID

    def test_info_qwen_vl_utils_flag(self):
        """`info()['qwen_vl_utils_available']` reflects module-level state."""
        import mata.adapters.qwen3_vl_embedding_adapter as mod

        adapter = _make_mock_adapter()
        saved = mod._QWEN_VL_UTILS_AVAILABLE
        try:
            mod._QWEN_VL_UTILS_AVAILABLE = True
            assert adapter.info()["qwen_vl_utils_available"] is True
            mod._QWEN_VL_UTILS_AVAILABLE = False
            assert adapter.info()["qwen_vl_utils_available"] is False
        finally:
            mod._QWEN_VL_UTILS_AVAILABLE = saved


# ---------------------------------------------------------------------------
# TestVideoFileExtraction (3 tests)
# ---------------------------------------------------------------------------


class TestVideoFileExtraction:
    def test_extract_frames_from_file(self):
        """_extract_frames_from_file reads frames via OpenCV VideoCapture."""
        adapter = _make_mock_adapter()
        adapter.fps = 30.0  # every frame at 30 fps source
        adapter.max_frames = 64

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # source_fps
        # Simulate 5 successful reads then EOF
        mock_cap.read.side_effect = [
            (True, frame),
            (True, frame),
            (True, frame),
            (True, frame),
            (True, frame),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            frames = adapter._extract_frames_from_file("fake_video.mp4")

        assert len(frames) == 5
        assert isinstance(frames[0], np.ndarray)

    def test_sample_frames_uniform(self):
        """_sample_frames returns exactly max_frames evenly-spaced frames."""
        adapter = _make_mock_adapter()
        adapter.max_frames = 4
        frames = [_make_bgr_frame() for _ in range(100)]
        sampled = adapter._sample_frames(frames)

        assert len(sampled) == 4
        # Expected indices: linspace(0, 99, 4, dtype=int) = [0, 33, 66, 99]
        expected_indices = np.linspace(0, 99, 4, dtype=int)
        for i, idx in enumerate(expected_indices):
            assert np.array_equal(sampled[i], frames[idx])

    def test_invalid_video_raises(self):
        """_extract_frames_from_file raises ValueError for unopenable video."""
        adapter = _make_mock_adapter()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(ValueError, match="Cannot open video file"):
                adapter._extract_frames_from_file("nonexistent.mp4")
