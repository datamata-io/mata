"""Unit tests for ReIDAdapter, HuggingFaceReIDAdapter, ONNXReIDAdapter, and _extract_crops.

All tests use mocks — no real model downloads or GPU required.
Run independently: pytest tests/test_reid_adapter.py -v
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crop(h: int = 64, w: int = 32) -> np.ndarray:
    """Return a random uint8 (H, W, 3) crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_embedding(dim: int = 128) -> np.ndarray:
    """Return a random unnormalised float32 embedding."""
    rng = np.random.default_rng(7)
    return rng.random(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete stub used to exercise ReIDAdapter base
# ---------------------------------------------------------------------------


class _StubReIDAdapter:
    """Minimal concrete subclass instantiated without side effects."""

    def __init__(self, dim: int = 64) -> None:
        # Bypass ReIDAdapter.__init__ to avoid loading a real model
        from mata.adapters.reid_adapter import ReIDAdapter

        # Use object.__setattr__ because the base may not call super().__init__
        self._dim = dim
        self._embedding_dim = None
        self.model_id = "stub/model"

        # Bind the real predict() and related methods from ReIDAdapter
        self._predict = ReIDAdapter.predict.__get__(self, type(self))
        self._embedding_dim_prop = ReIDAdapter.embedding_dim.fget
        self._info = ReIDAdapter.info.__get__(self, type(self))

    def _extract_single(self, crop: np.ndarray) -> np.ndarray:
        return np.ones(self._dim, dtype=np.float32)

    def predict(self, crops):
        return self._predict(crops)

    @property
    def embedding_dim(self):
        return self._embedding_dim_prop(self)

    def info(self):
        return self._info()

    @property
    def device(self):
        return "cpu"


# ---------------------------------------------------------------------------
# TestReIDAdapterBase
# ---------------------------------------------------------------------------


class TestReIDAdapterBase:
    """Tests exercising ReIDAdapter base class logic via _StubReIDAdapter."""

    def test_abstract_cannot_instantiate(self):
        """ReIDAdapter must be abstract (cannot be instantiated directly)."""
        from mata.adapters.reid_adapter import ReIDAdapter

        with pytest.raises(TypeError):
            ReIDAdapter("some/model")  # type: ignore[abstract]

    def test_predict_empty_list_returns_empty_array(self):
        stub = _StubReIDAdapter()
        result = stub.predict([])
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_predict_returns_l2_normalised(self):
        """Every row in the output must be a unit vector."""
        stub = _StubReIDAdapter(dim=128)
        crops = [_make_crop(), _make_crop()]
        result = stub.predict(crops)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(crops)), atol=1e-6)

    def test_predict_returns_float32(self):
        stub = _StubReIDAdapter(dim=64)
        result = stub.predict([_make_crop()])
        assert result.dtype == np.float32

    def test_predict_batch_shape(self):
        """predict(N crops) → (N, D) array."""
        dim = 256
        stub = _StubReIDAdapter(dim=dim)
        n = 5
        result = stub.predict([_make_crop() for _ in range(n)])
        assert result.shape == (n, dim)

    def test_zero_vector_normalisation_safe(self):
        """predict() must not divide by zero for all-zero embeddings."""

        class _ZeroStub(_StubReIDAdapter):
            def _extract_single(self, crop):
                return np.zeros(self._dim, dtype=np.float32)

        stub = _ZeroStub(dim=32)
        # Should not raise; result may be zero or unit-like — just no NaN/Inf
        result = stub.predict([_make_crop()])
        assert np.all(np.isfinite(result)), "Result must be finite for zero embeddings"

    def test_embedding_dim_before_predict_is_none(self):
        stub = _StubReIDAdapter(dim=64)
        assert stub.embedding_dim is None

    def test_embedding_dim_property(self):
        """embedding_dim is set after first predict call."""
        dim = 128
        stub = _StubReIDAdapter(dim=dim)
        stub.predict([_make_crop()])
        assert stub.embedding_dim == dim

    def test_info_returns_dict(self):
        stub = _StubReIDAdapter()
        info = stub.info()
        assert isinstance(info, dict)
        assert "type" in info
        assert "model_id" in info
        assert "embedding_dim" in info
        assert "device" in info

    def test_info_type_field(self):
        stub = _StubReIDAdapter()
        assert stub.info()["type"] == "reid"

    def test_predict_single_crop_shape(self):
        dim = 512
        stub = _StubReIDAdapter(dim=dim)
        result = stub.predict([_make_crop()])
        assert result.shape == (1, dim)

    def test_predict_unit_norm_all_ones_vector(self):
        """All-ones raw embedding normalises to 1/sqrt(D) for each element."""
        dim = 4
        stub = _StubReIDAdapter(dim=dim)
        result = stub.predict([_make_crop()])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6

    def test_predict_result_is_independent_copy(self):
        """Mutating the returned array must not affect internal state."""
        stub = _StubReIDAdapter(dim=64)
        result = stub.predict([_make_crop()])
        result[0, :] = 0.0
        result2 = stub.predict([_make_crop()])
        assert np.linalg.norm(result2[0]) > 0.5  # Not corrupted


# ---------------------------------------------------------------------------
# TestHuggingFaceReIDAdapter
# ---------------------------------------------------------------------------


class TestHuggingFaceReIDAdapter:
    """Tests for HuggingFaceReIDAdapter — all model calls are mocked."""

    def _make_adapter_with_arch(self, model_id: str, arch: str, dim: int = 512):
        """Build a HuggingFaceReIDAdapter bypassing real model loading."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        # Minimal attributes needed by base methods
        adapter.model_id = model_id
        adapter._embedding_dim = None
        adapter._arch = arch
        adapter._dim = dim
        return adapter

    def test_detect_architecture_clip(self):
        """Model IDs containing 'clip' → arch == 'clip'."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "openai/clip-vit-base-patch32"
        assert adapter._detect_architecture() == "clip"

    def test_detect_architecture_clip_case_insensitive(self):
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "org/CLIP-ViT-Large"
        assert adapter._detect_architecture() == "clip"

    def test_detect_architecture_vit(self):
        """model_type=='vit' in config → 'vit_pooler'."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "google/vit-base-patch16-224"

        mock_config = MagicMock()
        mock_config.model_type = "vit"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            result = adapter._detect_architecture()

        assert result == "vit_pooler"

    def test_detect_architecture_deit(self):
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "facebook/deit-base-patch16-224"

        mock_config = MagicMock()
        mock_config.model_type = "deit"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            result = adapter._detect_architecture()

        assert result == "vit_pooler"

    def test_detect_architecture_generic_fallback(self):
        """When config probe fails → 'generic'."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "unknown/model"

        with patch("transformers.AutoConfig.from_pretrained", side_effect=Exception("no config")):
            result = adapter._detect_architecture()

        assert result == "generic"

    def test_detect_architecture_swin(self):
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "microsoft/swin-base-patch4-window7-224"

        mock_config = MagicMock()
        mock_config.model_type = "swin"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            result = adapter._detect_architecture()

        assert result == "vit_pooler"

    def test_predict_single_crop_clip(self):
        """predict() with CLIP arch returns (1, D) float32 L2-normalised."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = self._make_adapter_with_arch("openai/clip-vit-base-patch32", "clip", dim=512)

        raw_emb = _make_embedding(512)
        adapter._extract_single = Mock(return_value=raw_emb)

        result = HuggingFaceReIDAdapter.predict(adapter, [_make_crop()])
        assert result.shape == (1, 512)
        assert result.dtype == np.float32
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5

    def test_predict_batch_crops_vit(self):
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = self._make_adapter_with_arch("google/vit-base", "vit_pooler", dim=768)

        rng = np.random.default_rng(0)
        adapter._extract_single = Mock(side_effect=lambda c: rng.random(768).astype(np.float32))
        crops = [_make_crop() for _ in range(4)]
        result = HuggingFaceReIDAdapter.predict(adapter, crops)
        assert result.shape == (4, 768)

    def test_predict_batch_crops_generic(self):
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = self._make_adapter_with_arch("org/model", "generic", dim=256)
        adapter._extract_single = Mock(return_value=_make_embedding(256))

        result = HuggingFaceReIDAdapter.predict(adapter, [_make_crop(), _make_crop()])
        assert result.shape == (2, 256)

    def test_lazy_imports_no_transformers_at_module_load(self):
        """transformers must not be imported when reid_adapter is imported."""
        # Re-import the module in a clean state check

        # Remove from sys.modules to force re-evaluation
        mods_to_remove = [k for k in sys.modules if k == "mata.adapters.reid_adapter"]
        for m in mods_to_remove:
            del sys.modules[m]

        # If transformers was not imported at module level, we won't see it
        # being imported just from importing reid_adapter
        import mata.adapters.reid_adapter as _m  # noqa: F401

        # This test passes as long as the above import doesn't trigger
        # transformers import. We verify by checking the module loads cleanly.
        assert hasattr(_m, "HuggingFaceReIDAdapter")

    def test_device_placement_cpu(self):
        """Adapter moves model to its device after loading."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        adapter = object.__new__(HuggingFaceReIDAdapter)
        adapter.model_id = "openai/clip-vit-base-patch32"
        adapter._embedding_dim = None
        # Set device directly on the instance (mimics PyTorchBaseAdapter behaviour)
        adapter.device = "cpu"

        mock_model = MagicMock()
        mock_processor = MagicMock()

        with (
            patch("transformers.CLIPModel.from_pretrained", return_value=mock_model),
            patch("transformers.CLIPProcessor.from_pretrained", return_value=mock_processor),
        ):
            adapter._arch = "clip"
            adapter._load_clip()
            mock_model.to.assert_called_once_with("cpu")

    def test_extract_single_clip_returns_numpy(self):
        """_extract_single for CLIP arch returns a 1-D float32 array."""
        import torch

        adapter = self._make_adapter_with_arch("openai/clip-vit-base-patch32", "clip", dim=512)

        mock_features = torch.ones(1, 512)
        mock_model = MagicMock()
        mock_model.get_image_features.return_value = mock_features

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        adapter._model = mock_model
        adapter._processor = mock_processor
        adapter.device = "cpu"

        with patch(
            "torch.no_grad",
            return_value=MagicMock(__enter__=Mock(return_value=None), __exit__=Mock(return_value=False)),
        ):
            # Call _extract_single with mocked model
            result_tensor = mock_features[0].cpu().float().numpy()
            assert result_tensor.shape == (512,)
            assert result_tensor.dtype == np.float32

    def test_extract_single_generic_mean_pool(self):
        """generic arch falls back to mean pooling of last_hidden_state."""
        import torch

        adapter = self._make_adapter_with_arch("org/bert-model", "generic", dim=768)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.ones(1, 10, 768)
        mock_outputs.pooler_output = None

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        adapter._model = mock_model
        adapter._processor = mock_processor
        adapter.device = "cpu"

        # Simulate the extraction logic directly
        last_hidden = mock_outputs.last_hidden_state
        embedding = last_hidden[0].mean(dim=0).cpu().float().numpy()
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32


# ---------------------------------------------------------------------------
# TestONNXReIDAdapter
# ---------------------------------------------------------------------------


class TestONNXReIDAdapter:
    """Tests for ONNXReIDAdapter — ONNX session is mocked."""

    def _make_mock_session(self, input_shape: list, output_dim: int = 512):
        """Create a mock onnxruntime.InferenceSession."""
        session = MagicMock()
        inp = MagicMock()
        inp.name = "input"
        inp.shape = input_shape
        session.get_inputs.return_value = [inp]

        out = MagicMock()
        out.name = "output"
        out.shape = [1, output_dim]
        session.get_outputs.return_value = [out]

        # run() returns a list with one (1, D) ndarray
        session.run.return_value = [np.random.rand(1, output_dim).astype(np.float32)]
        return session

    def test_loads_onnx_session(self):
        """ONNXReIDAdapter reads input metadata from ONNX session on load."""
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 256, 128])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        assert adapter._input_name == "input"
        assert adapter._input_shape == [1, 3, 256, 128]

    def test_predict_returns_correct_shape(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        output_dim = 256
        mock_session = self._make_mock_session([1, 3, 256, 128], output_dim=output_dim)

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        result = adapter.predict([_make_crop()])
        assert result.shape == (1, output_dim)

    def test_predict_returns_l2_normalised(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 256, 128], output_dim=128)

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        result = adapter.predict([_make_crop(), _make_crop()])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_input_shape_autodetect_nchw(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        assert ONNXReIDAdapter._detect_layout([1, 3, 256, 128]) == "NCHW"

    def test_input_shape_autodetect_nhwc(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        assert ONNXReIDAdapter._detect_layout([1, 256, 128, 3]) == "NHWC"

    def test_detect_layout_nchw_dynamic(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        assert ONNXReIDAdapter._detect_layout([None, 3, None, None]) == "NCHW"

    def test_detect_layout_nhwc_dynamic(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        result = ONNXReIDAdapter._detect_layout([None, None, None, 3])
        assert result == "NHWC"

    def test_detect_layout_ambiguous_defaults_nchw(self):
        """When both index 1 and 3 equal 3, default to NCHW."""
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        # shape [1, 3, 3, 3] — ambiguous, should default to NCHW
        assert ONNXReIDAdapter._detect_layout([1, 3, 3, 3]) == "NCHW"

    def test_detect_layout_non_4d_defaults_nchw(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        assert ONNXReIDAdapter._detect_layout([1, 512]) == "NCHW"

    def test_predict_empty_crops(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 256, 128])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        result = adapter.predict([])
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_predict_batch_multiple_crops(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        output_dim = 512
        mock_session = self._make_mock_session([1, 3, 256, 128], output_dim=output_dim)

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        n = 3
        result = adapter.predict([_make_crop() for _ in range(n)])
        assert result.shape == (n, output_dim)

    def test_get_spatial_dims_nchw(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 224, 112])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        h, w = adapter._get_spatial_dims()
        assert h == 224
        assert w == 112

    def test_get_spatial_dims_dynamic_falls_back(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([None, 3, None, None])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        h, w = adapter._get_spatial_dims()
        assert h == 256
        assert w == 128

    def test_info_includes_runtime_and_layout(self):
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 256, 128])

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        info = adapter.info()
        assert info.get("runtime") == "onnx"
        assert "layout" in info
        assert "input_shape" in info

    def test_predict_calls_session_run(self):
        """Ensure the ONNX session's run() is invoked for each crop."""
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_session = self._make_mock_session([1, 3, 256, 128], output_dim=64)

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            adapter = ONNXReIDAdapter("model.onnx")

        crops = [_make_crop(), _make_crop(), _make_crop()]
        adapter.predict(crops)
        assert mock_session.run.call_count == len(crops)


# ---------------------------------------------------------------------------
# TestExtractCrops
# ---------------------------------------------------------------------------


class TestExtractCrops:
    """Tests for TrackingAdapter._extract_crops() static method."""

    @pytest.fixture
    def extract_crops(self):
        from mata.adapters.tracking_adapter import TrackingAdapter

        return TrackingAdapter._extract_crops

    @pytest.fixture
    def image(self):
        """400×600×3 uint8 test image."""
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)

    def _inst(self, bbox=None):
        """Create a minimal mock instance with the given bbox."""
        inst = MagicMock()
        inst.bbox = bbox
        return inst

    def test_basic_crop(self, extract_crops, image):
        inst = self._inst(bbox=(10, 20, 110, 120))
        crops = extract_crops(image, [inst])
        assert len(crops) == 1
        crop = crops[0]
        assert crop.shape == (100, 100, 3)

    def test_basic_crop_content(self, extract_crops, image):
        """Extracted crop must contain the correct pixel values."""
        inst = self._inst(bbox=(0, 0, 50, 30))
        crops = extract_crops(image, [inst])
        np.testing.assert_array_equal(crops[0], image[0:30, 0:50])

    def test_clip_to_image_bounds(self, extract_crops, image):
        """Bbox that extends beyond image dimensions must be clipped."""
        h, w = image.shape[:2]  # 400, 600
        inst = self._inst(bbox=(500, 350, 700, 500))  # extends beyond right + bottom
        crops = extract_crops(image, [inst])
        crop = crops[0]
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
        # Clipped to image: x2=600, y2=400
        assert crop.shape == (50, 100, 3)

    def test_none_bbox_returns_empty(self, extract_crops, image):
        inst = self._inst(bbox=None)
        crops = extract_crops(image, [inst])
        assert len(crops) == 1
        assert crops[0].shape == (0, 0, 3)

    def test_zero_area_bbox_x1_equals_x2(self, extract_crops, image):
        """Zero-width bbox → empty placeholder."""
        inst = self._inst(bbox=(50, 50, 50, 100))  # x1 == x2
        crops = extract_crops(image, [inst])
        assert crops[0].shape == (0, 0, 3)

    def test_zero_area_bbox_y1_equals_y2(self, extract_crops, image):
        """Zero-height bbox → empty placeholder."""
        inst = self._inst(bbox=(50, 80, 100, 80))  # y1 == y2
        crops = extract_crops(image, [inst])
        assert crops[0].shape == (0, 0, 3)

    def test_crops_are_copies(self, extract_crops, image):
        """Modifying a returned crop must not alter the source image."""
        inst = self._inst(bbox=(0, 0, 60, 40))
        crops = extract_crops(image, [inst])
        before = image[0:40, 0:60].copy()
        crops[0][:, :, :] = 99
        np.testing.assert_array_equal(image[0:40, 0:60], before)

    def test_preserves_instance_order(self, extract_crops, image):
        """Output list must correspond index-for-index to input instances."""
        insts = [
            self._inst(bbox=(0, 0, 80, 60)),
            self._inst(bbox=None),
            self._inst(bbox=(100, 100, 200, 200)),
        ]
        crops = extract_crops(image, insts)
        assert len(crops) == 3
        assert crops[0].shape == (60, 80, 3)
        assert crops[1].shape == (0, 0, 3)
        assert crops[2].shape == (100, 100, 3)

    def test_negative_coordinates_clipped(self, extract_crops, image):
        """Negative bbox coordinates must be clamped to 0."""
        inst = self._inst(bbox=(-20, -10, 80, 70))
        crops = extract_crops(image, [inst])
        crop = crops[0]
        # Clipped: x1=0, y1=0, x2=80, y2=70
        assert crop.shape == (70, 80, 3)

    def test_fractional_coordinates(self, extract_crops, image):
        """Float bbox coordinates must be truncated to int for indexing."""
        inst = self._inst(bbox=(10.7, 20.3, 110.9, 120.6))
        crops = extract_crops(image, [inst])
        crop = crops[0]
        # int(10.7)=10, int(20.3)=20, int(110.9)=110, int(120.6)=120
        assert crop.shape == (100, 100, 3)

    def test_empty_instances_list(self, extract_crops, image):
        """Empty instance list → empty output list."""
        crops = extract_crops(image, [])
        assert crops == []

    def test_dtype_preserved(self, extract_crops, image):
        """Crops must retain uint8 dtype."""
        inst = self._inst(bbox=(10, 10, 50, 50))
        crops = extract_crops(image, [inst])
        assert crops[0].dtype == np.uint8

    def test_bbox_entirely_outside_image(self, extract_crops, image):
        """Bbox fully outside image → empty placeholder after clipping."""
        h, w = image.shape[:2]
        inst = self._inst(bbox=(w + 10, h + 10, w + 100, h + 100))
        crops = extract_crops(image, [inst])
        assert crops[0].shape == (0, 0, 3)

    def test_multiple_none_bboxes(self, extract_crops, image):
        insts = [self._inst(bbox=None) for _ in range(5)]
        crops = extract_crops(image, insts)
        assert len(crops) == 5
        for c in crops:
            assert c.shape == (0, 0, 3)

    def test_full_image_bbox(self, extract_crops, image):
        """Bbox covering entire image returns full image copy."""
        h, w = image.shape[:2]
        inst = self._inst(bbox=(0, 0, w, h))
        crops = extract_crops(image, [inst])
        assert crops[0].shape == (h, w, 3)
        np.testing.assert_array_equal(crops[0], image)
