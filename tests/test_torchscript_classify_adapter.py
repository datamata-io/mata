"""Tests for TorchScript classification adapter."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from mata.core.exceptions import ModelLoadError
from mata.core.types import Classification, ClassifyResult


class TestTorchScriptClassifyAdapter:
    """Test suite for TorchScript classification adapter."""

    @pytest.fixture
    def mock_torchscript_model(self):
        """Create mock TorchScript model."""
        model = MagicMock()
        model.eval.return_value = model

        # Mock forward pass
        def mock_forward(x):
            # Return logits for batch
            batch_size = x.shape[0]
            logits = torch.randn(batch_size, 1000)
            # Make first 5 logits higher
            logits[:, :5] = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
            return logits

        model.return_value = mock_forward
        model.__call__ = mock_forward

        return model

    @pytest.fixture
    def temp_torchscript_file(self):
        """Create temporary TorchScript model file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Save a simple traced model
            simple_model = torch.nn.Linear(10, 10)
            traced = torch.jit.trace(simple_model, torch.randn(1, 10))
            torch.jit.save(traced, f.name)
            yield f.name
        Path(f.name).unlink()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization(self, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test TorchScript classification adapter initialization."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        # Setup mock torch module
        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch
        mock_torch.jit.load.return_value = mock_torchscript_model

        # Create mock device object with .type attribute
        mock_device = MagicMock()
        mock_device.type = "cpu"
        mock_torch.device.return_value = mock_device

        # Initialize adapter
        adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file, top_k=5, device="cpu")

        # Verify initialization
        assert adapter.model_path == Path(temp_torchscript_file)
        assert adapter.top_k == 5
        assert adapter.input_size == 224
        mock_torch.jit.load.assert_called_once()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_cuda(self, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test TorchScript adapter initialization with CUDA."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch
        mock_torch.jit.load.return_value = mock_torchscript_model

        # Create mock CUDA device object
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_torch.device.return_value = mock_device
        mock_torch.cuda.is_available.return_value = True

        TorchScriptClassifyAdapter(model_path=temp_torchscript_file, device="cuda")

        # Verify CUDA device was used
        mock_torch.device.assert_called_with("cuda")

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_custom_id2label(self, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test TorchScript adapter with custom label mapping."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch
        mock_torch.jit.load.return_value = mock_torchscript_model

        # Create mock device object
        mock_device = MagicMock()
        mock_device.type = "cpu"
        mock_torch.device.return_value = mock_device

        custom_labels = {0: "class_a", 1: "class_b", 2: "class_c"}
        adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file, id2label=custom_labels)

        assert adapter.id2label == custom_labels

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_file_not_found(self, mock_torch_loader):
        """Test TorchScript adapter with non-existent file."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        with pytest.raises(FileNotFoundError):
            TorchScriptClassifyAdapter(model_path="nonexistent.pt")

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_load_error(self, mock_torch_loader, temp_torchscript_file):
        """Test TorchScript adapter when model loading fails."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        # Simulate TorchScript loading error
        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch
        mock_torch.jit.load.side_effect = Exception("TorchScript load failed")

        with pytest.raises(ModelLoadError) as exc_info:
            TorchScriptClassifyAdapter(model_path=temp_torchscript_file)

        assert "Failed to load TorchScript classification model" in str(exc_info.value)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.base.Image")
    def test_predict_basic(self, mock_image, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test basic classification prediction."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch

        # Setup model to return real logits tensor
        def mock_model_call(x):
            batch_size = x.shape[0] if hasattr(x, "shape") else 1
            logits = torch.randn(batch_size, 1000)
            # Make first 5 logits higher for predictable top-k results
            logits[:, :5] = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
            return logits

        mock_model = MagicMock()
        mock_model.side_effect = mock_model_call  # Use side_effect to call function
        mock_model.to.return_value = mock_model
        mock_torch.jit.load.return_value = mock_model

        # Use real torch device and functional operations
        mock_torch.device = torch.device
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.nn = torch.nn  # Use real torch.nn for functional.softmax
        mock_torch.topk = torch.topk

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock transforms
        with patch("mata.adapters.torchscript_classify_adapter.T") as mock_T:  # noqa: N806
            mock_transform = MagicMock()
            mock_transform.return_value = torch.randn(3, 224, 224)
            mock_T.Compose.return_value = mock_transform
            mock_T.Resize = MagicMock()
            mock_T.ToTensor = MagicMock()
            mock_T.Normalize = MagicMock()

            adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file, top_k=5)

            result = adapter.predict("test_image.jpg")

        # Verify result
        assert isinstance(result, ClassifyResult)
        assert len(result.predictions) == 5
        assert all(isinstance(pred, Classification) for pred in result.predictions)
        # Scores should be sorted descending
        scores = [pred.score for pred in result.predictions]
        assert scores == sorted(scores, reverse=True)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    @patch("mata.adapters.base.Image")
    def test_predict_with_runtime_top_k_override(
        self, mock_image, mock_torch_loader, mock_torchscript_model, temp_torchscript_file
    ):
        """Test prediction with runtime top_k override."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch

        # Setup model to return real logits tensor
        def mock_model_call(x):
            batch_size = x.shape[0] if hasattr(x, "shape") else 1
            logits = torch.randn(batch_size, 1000)
            logits[:, :10] = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
            return logits

        mock_model = MagicMock()
        mock_model.side_effect = mock_model_call
        mock_model.to.return_value = mock_model
        mock_torch.jit.load.return_value = mock_model

        # Use real torch device and functional operations
        mock_torch.device = torch.device
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.nn = torch.nn
        mock_torch.topk = torch.topk

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        with patch("mata.adapters.torchscript_classify_adapter.T") as mock_T:  # noqa: N806
            mock_transform = MagicMock()
            mock_transform.return_value = torch.randn(3, 224, 224)
            mock_T.Compose.return_value = mock_transform
            mock_T.Resize = MagicMock()
            mock_T.ToTensor = MagicMock()
            mock_T.Normalize = MagicMock()

            adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file, top_k=5)  # Default

            # Override at runtime
            result = adapter.predict("test_image.jpg", top_k=10)

        assert len(result.predictions) == 10

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_from_pil_image(self, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test prediction from PIL Image."""
        from PIL import Image

        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch

        # Setup model to return real logits tensor
        def mock_model_call(x):
            batch_size = x.shape[0] if hasattr(x, "shape") else 1
            logits = torch.randn(batch_size, 1000)
            logits[:, :5] = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
            return logits

        mock_model = MagicMock()
        mock_model.side_effect = mock_model_call
        mock_model.to.return_value = mock_model
        mock_torch.jit.load.return_value = mock_model

        # Use real torch device and functional operations
        mock_torch.device = torch.device
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.nn = torch.nn
        mock_torch.topk = torch.topk

        with patch("mata.adapters.torchscript_classify_adapter.T") as mock_T:  # noqa: N806
            mock_transform = MagicMock()
            mock_transform.return_value = torch.randn(3, 224, 224)
            mock_T.Compose.return_value = mock_transform
            mock_T.Resize = MagicMock()
            mock_T.ToTensor = MagicMock()
            mock_T.Normalize = MagicMock()

            adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file)

            # Create real PIL image instead of mock
            img = Image.new("RGB", (224, 224))
            result = adapter.predict(img)

        assert isinstance(result, ClassifyResult)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_info_method(self, mock_torch_loader, mock_torchscript_model, temp_torchscript_file):
        """Test adapter info() method."""
        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

        mock_torch = MagicMock()
        mock_torch_loader.return_value = mock_torch
        mock_torch.jit.load.return_value = mock_torchscript_model

        # Create mock CUDA device object
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_torch.device.return_value = mock_device
        mock_torch.cuda.is_available.return_value = True

        adapter = TorchScriptClassifyAdapter(model_path=temp_torchscript_file, top_k=10, device="cuda")

        info = adapter.info()

        assert info["name"] == "TorchScriptClassifyAdapter"
        assert info["task"] == "classify"
        assert info["top_k"] == 10
        assert info["backend"] == "torchscript"
        assert "model_path" in info
        assert "device" in info
