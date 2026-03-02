"""Unit tests for HuggingFaceDepthAdapter."""

from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


class TestHuggingFaceDepthAdapter:
    """Test HuggingFaceDepthAdapter initialization and methods."""

    @patch("mata.adapters.huggingface_depth_adapter._ensure_transformers")
    def test_adapter_initialization(self, mock_transformers):
        """Test adapter initialization."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value.eval.return_value = mock_model

        mock_transformers.return_value = {
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForDepthEstimation": Mock(from_pretrained=Mock(return_value=mock_model)),
        }

        from mata.adapters.huggingface_depth_adapter import HuggingFaceDepthAdapter

        adapter = HuggingFaceDepthAdapter(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            device="cpu",
            normalize=True,
        )

        assert adapter.model_id == "depth-anything/Depth-Anything-V2-Small-hf"
        assert adapter.normalize is True
        assert adapter.task == "depth"

    @patch("mata.adapters.huggingface_depth_adapter._ensure_transformers")
    def test_adapter_predict(self, mock_transformers):
        """Test depth prediction flow."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.to.return_value.eval.return_value = mock_model

        mock_transformers.return_value = {
            "AutoImageProcessor": Mock(from_pretrained=Mock(return_value=mock_processor)),
            "AutoModelForDepthEstimation": Mock(from_pretrained=Mock(return_value=mock_model)),
        }

        from mata.adapters.huggingface_depth_adapter import HuggingFaceDepthAdapter

        adapter = HuggingFaceDepthAdapter(
            model_id="depth-anything/Depth-Anything-V2-Small-hf",
            device="cpu",
            normalize=True,
        )

        mock_processor.return_value = {"pixel_values": adapter.torch.zeros(1, 3, 2, 2)}
        mock_model.return_value = Mock()

        predicted_depth = adapter.torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_processor.post_process_depth_estimation.return_value = [{"predicted_depth": predicted_depth}]

        image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
        result = adapter.predict(image)

        assert result.depth.shape == (2, 2)
        assert result.normalized is not None
        assert result.meta["task"] == "depth"
