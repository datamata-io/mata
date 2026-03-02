"""Tests for ONNX classification adapter."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata.core.exceptions import ModelLoadError
from mata.core.types import Classification, ClassifyResult


class TestONNXClassifyAdapter:
    """Test suite for ONNX classification adapter."""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create mock ONNX inference session."""
        session = MagicMock()

        # Mock input metadata
        input_meta = MagicMock()
        input_meta.name = "input"
        input_meta.shape = [1, 3, 224, 224]
        session.get_inputs.return_value = [input_meta]

        # Mock output metadata
        output_meta = MagicMock()
        output_meta.name = "output"
        output_meta.shape = [1, 1000]
        session.get_outputs.return_value = [output_meta]

        # Mock run method to return classification logits
        def mock_run(output_names, input_dict):
            # Return logits for 1000 ImageNet classes
            logits = np.random.randn(1, 1000).astype(np.float32)
            # Make first 5 logits higher to ensure predictable top_k
            logits[0, :5] = [10.0, 9.0, 8.0, 7.0, 6.0]
            return [logits]

        session.run.side_effect = mock_run

        return session

    @pytest.fixture
    def temp_onnx_file(self):
        """Create temporary ONNX model file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"mock_onnx_data")
            yield f.name
        Path(f.name).unlink()

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_adapter_initialization(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test ONNX classification adapter initialization."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        # Initialize adapter
        adapter = ONNXClassifyAdapter(model_path=temp_onnx_file, top_k=5, device="cpu")

        # Verify initialization
        assert adapter.model_path == Path(temp_onnx_file)
        assert adapter.top_k == 5
        mock_ort.InferenceSession.assert_called_once()

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_adapter_initialization_cuda(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test ONNX adapter initialization with CUDA provider."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        ONNXClassifyAdapter(model_path=temp_onnx_file, device="cuda")

        # Verify CUDA provider was requested
        call_args = mock_ort.InferenceSession.call_args
        providers = call_args[1]["providers"]
        assert "CUDAExecutionProvider" in providers

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_adapter_custom_id2label(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test ONNX adapter with custom label mapping."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        custom_labels = {0: "class_a", 1: "class_b", 2: "class_c"}
        adapter = ONNXClassifyAdapter(model_path=temp_onnx_file, id2label=custom_labels)

        assert adapter.id2label == custom_labels

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_adapter_initialization_file_not_found(self, mock_ensure_ort):
        """Test ONNX adapter with non-existent file."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        with pytest.raises(FileNotFoundError):
            ONNXClassifyAdapter(model_path="nonexistent.onnx")

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_adapter_initialization_load_error(self, mock_ensure_ort, temp_onnx_file):
        """Test ONNX adapter when model loading fails."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module with InferenceSession that raises error
        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = Exception("ONNX load failed")
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        with pytest.raises(ModelLoadError) as exc_info:
            ONNXClassifyAdapter(model_path=temp_onnx_file)

        assert "Failed to load ONNX classification model" in str(exc_info.value)

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    @patch("mata.adapters.base.Image")  # Patch Image in base.py where _load_image is defined
    def test_predict_basic(self, mock_image, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test basic classification prediction."""
        from PIL import Image

        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_img.size = (224, 224)
        mock_image.open.return_value = mock_img
        mock_image.BILINEAR = Image.BILINEAR  # Use real constant

        # Mock numpy conversion
        with patch("mata.adapters.onnx_classify_adapter.np") as mock_np:
            mock_np.array.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_np.transpose = np.transpose
            mock_np.expand_dims = np.expand_dims
            mock_np.ascontiguousarray = np.ascontiguousarray
            # Also pass through numpy math functions
            mock_np.exp = np.exp
            mock_np.max = np.max
            mock_np.sum = np.sum
            mock_np.argsort = np.argsort

            adapter = ONNXClassifyAdapter(model_path=temp_onnx_file, top_k=5)

            result = adapter.predict("test_image.jpg")

        # Verify result
        assert isinstance(result, ClassifyResult)
        assert len(result.predictions) == 5
        assert all(isinstance(pred, Classification) for pred in result.predictions)
        # Scores should be sorted descending
        scores = [pred.score for pred in result.predictions]
        assert scores == sorted(scores, reverse=True)

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    @patch("mata.adapters.base.Image")  # Patch Image in base.py where _load_image is defined
    def test_predict_with_runtime_top_k_override(self, mock_image, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test prediction with runtime top_k override."""
        from PIL import Image

        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.resize.return_value = mock_img
        mock_img.size = (224, 224)
        mock_image.open.return_value = mock_img
        mock_image.BILINEAR = Image.BILINEAR  # Use real constant

        with patch("mata.adapters.onnx_classify_adapter.np") as mock_np:
            mock_np.array.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_np.transpose = np.transpose
            mock_np.expand_dims = np.expand_dims
            mock_np.ascontiguousarray = np.ascontiguousarray
            # Also pass through numpy math functions
            mock_np.exp = np.exp
            mock_np.max = np.max
            mock_np.sum = np.sum
            mock_np.argsort = np.argsort

            adapter = ONNXClassifyAdapter(model_path=temp_onnx_file, top_k=5)  # Default

            # Override at runtime
            result = adapter.predict("test_image.jpg", top_k=10)

        assert len(result.predictions) == 10

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_predict_from_pil_image(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test prediction from PIL Image."""
        from PIL import Image

        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        adapter = ONNXClassifyAdapter(model_path=temp_onnx_file)

        # Create a real PIL image (small 10x10)
        img = Image.new("RGB", (10, 10))

        # No need to mock np functions - they work with real PIL images
        result = adapter.predict(img)

        assert isinstance(result, ClassifyResult)

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_predict_from_numpy_array(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test prediction from numpy array."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        adapter = ONNXClassifyAdapter(model_path=temp_onnx_file)

        # Create a small numpy array (10x10) - real array, not mocked
        img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        # No mocking needed - PIL and numpy work together
        result = adapter.predict(img_array)

        assert isinstance(result, ClassifyResult)

    @patch("mata.adapters.onnx_base._ensure_onnxruntime")
    def test_info_method(self, mock_ensure_ort, mock_onnx_session, temp_onnx_file):
        """Test adapter info() method."""
        from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

        # Setup mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_onnx_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ensure_ort.return_value = mock_ort

        adapter = ONNXClassifyAdapter(model_path=temp_onnx_file, top_k=10, device="cuda")

        info = adapter.info()

        assert info["name"] == "ONNXClassifyAdapter"
        assert info["task"] == "classify"
        assert info["top_k"] == 10
        assert info["backend"] == "onnxruntime"
        assert "model_path" in info
        assert "input_shape" in info
