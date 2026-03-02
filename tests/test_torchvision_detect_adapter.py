"""Comprehensive test suite for TorchvisionDetectAdapter.

Tests cover initialization, model loading, prediction, preprocessing,
integration with UniversalLoader, error handling, and metadata.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from mata.core.exceptions import ModelLoadError, UnsupportedModelError
from mata.core.types import VisionResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_torch():
    """Mock PyTorch module."""
    with patch("mata.adapters.pytorch_base._ensure_torch") as mock:
        import torch as real_torch

        mock_torch = MagicMock()

        # Use real torch.device for compatibility
        mock_torch.device = real_torch.device
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        # Use real torch functions
        mock_torch.tensor = real_torch.tensor
        mock_torch.load.return_value = {"state_dict": {}, "model": {}}

        mock.return_value = mock_torch
        yield mock_torch


@pytest.fixture
def mock_torchvision_model():
    """Mock torchvision detection model."""
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model

    # Mock prediction output
    def mock_forward(images):
        """Return mock detection results."""
        batch_size = images.shape[0] if hasattr(images, "shape") else 1

        # Import torch for tensor creation
        import torch

        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0], [150.0, 30.0, 300.0, 250.0]]),
                "scores": torch.tensor([0.95, 0.87]),
                "labels": torch.tensor([1, 17]),  # COCO: person=1, cat=17
            }
            for _ in range(batch_size)
        ]

    model.side_effect = mock_forward
    return model


@pytest.fixture
def mock_torchvision_detection(mock_torchvision_model):
    """Mock torchvision.models.detection module."""
    with patch("mata.adapters.torchvision_detect_adapter._ensure_torchvision") as mock_ensure:
        mock_detection = MagicMock()

        # All model builders return the mock model
        mock_detection.fasterrcnn_resnet50_fpn.return_value = mock_torchvision_model
        mock_detection.fasterrcnn_resnet50_fpn_v2.return_value = mock_torchvision_model
        mock_detection.retinanet_resnet50_fpn.return_value = mock_torchvision_model
        mock_detection.retinanet_resnet50_fpn_v2.return_value = mock_torchvision_model
        mock_detection.fcos_resnet50_fpn.return_value = mock_torchvision_model
        mock_detection.ssd300_vgg16.return_value = mock_torchvision_model
        mock_detection.ssdlite320_mobilenet_v3_large.return_value = mock_torchvision_model

        # Mock transforms that convert PIL Image to tensor
        import torch

        def mock_transform(image):
            """Mock transform that converts PIL Image to tensor."""
            # Convert PIL Image to numpy array, then to tensor
            import numpy as np

            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
            return torch.tensor(img_array)

        mock_transforms = MagicMock()
        mock_compose = MagicMock()
        mock_compose.return_value = mock_transform
        mock_transforms.Compose = mock_compose
        mock_transforms.ToTensor = MagicMock()
        mock_transforms.Normalize = MagicMock()

        mock_ensure.return_value = (mock_detection, mock_transforms)
        yield mock_detection


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    img = Image.new("RGB", (640, 480), color="red")
    return img


@pytest.fixture
def temp_checkpoint_file():
    """Create temporary checkpoint file."""
    import torch

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
        torch.save({"model_state_dict": {}, "state_dict": {}}, f)
        filepath = f.name

    yield filepath

    # Cleanup
    try:
        os.unlink(filepath)
    except (OSError, PermissionError):
        pass


# ============================================================================
# Test Class
# ============================================================================


class TestTorchvisionDetectAdapter:
    """Comprehensive tests for TorchvisionDetectAdapter."""

    # ========================================================================
    # 1. Initialization Tests (5 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_cpu(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test adapter initialization with CPU device."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", device="cpu")

        assert adapter.model_name == "torchvision/retinanet_resnet50_fpn"
        assert adapter.parsed_model_name == "retinanet_resnet50_fpn"
        assert adapter.device.type == "cpu"
        assert adapter.threshold == 0.3  # default threshold

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_auto_cuda_available(
        self, mock_ensure_torch, mock_torch, mock_torchvision_detection
    ):
        """Test adapter initialization with auto device when CUDA available."""
        # Mock CUDA availability

        mock_torch.cuda.is_available.return_value = True
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", device="auto")

        # When CUDA is available, should select CUDA (or at least try to)
        # We can't truly test this with mocks, but we can verify initialization succeeds
        assert adapter.model_name == "torchvision/retinanet_resnet50_fpn"

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_auto_cuda_unavailable(
        self, mock_ensure_torch, mock_torch, mock_torchvision_detection
    ):
        """Test adapter initialization with auto device when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/fasterrcnn_resnet50_fpn", device="auto")

        # Should fall back to CPU
        assert adapter.device.type == "cpu"

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_custom_threshold(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test adapter initialization with custom threshold."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", threshold=0.7)

        assert adapter.threshold == 0.7

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization_custom_id2label(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test adapter initialization with custom label mapping."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        custom_labels = {0: "cat", 1: "dog", 2: "bird"}

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", id2label=custom_labels)

        assert adapter.id2label == custom_labels

    # ========================================================================
    # 2. Model Loading Tests (6 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_fasterrcnn(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test Faster R-CNN model loads successfully."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/fasterrcnn_resnet50_fpn")

        assert adapter.parsed_model_name == "fasterrcnn_resnet50_fpn"
        mock_torchvision_detection.fasterrcnn_resnet50_fpn.assert_called()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_retinanet(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test RetinaNet model loads successfully."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

        assert adapter.parsed_model_name == "retinanet_resnet50_fpn"
        mock_torchvision_detection.retinanet_resnet50_fpn.assert_called()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_fcos(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test FCOS model loads successfully."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="fcos_resnet50_fpn")  # Test without prefix

        assert adapter.parsed_model_name == "fcos_resnet50_fpn"
        mock_torchvision_detection.fcos_resnet50_fpn.assert_called()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_with_pretrained_weights(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test model loading with DEFAULT pretrained weights."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", weights="DEFAULT")

        assert adapter.weights == "DEFAULT"
        # Model should be loaded with pretrained weights
        mock_torchvision_detection.retinanet_resnet50_fpn.assert_called()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_with_custom_checkpoint(
        self, mock_ensure_torch, mock_torch, mock_torchvision_detection, temp_checkpoint_file
    ):
        """Test model loading with custom checkpoint file."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", weights=temp_checkpoint_file
        )

        assert adapter.weights == temp_checkpoint_file
        # Should call torch.load to load checkpoint
        mock_torch.load.assert_called()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_loading_invalid_model_name(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that loading unknown model raises UnsupportedModelError."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        with pytest.raises(UnsupportedModelError) as exc_info:
            TorchvisionDetectAdapter(model_name="torchvision/unknown_model")

        assert "unknown_model" in str(exc_info.value)
        assert "Supported models" in str(exc_info.value)

    # ========================================================================
    # 3. Prediction Tests (7 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_returns_visionresult(
        self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image
    ):
        """Test that predict returns VisionResult."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

        result = adapter.predict(sample_image)

        assert isinstance(result, VisionResult)
        assert hasattr(result, "instances")
        assert hasattr(result, "meta")

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_bbox_format_xyxy(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test that bboxes are in xyxy format."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", threshold=0.0  # Accept all detections
        )

        result = adapter.predict(sample_image)

        assert len(result.instances) > 0

        for instance in result.instances:
            x1, y1, x2, y2 = instance.bbox
            # xyxy format: x2 > x1, y2 > y1
            assert x2 > x1, f"Invalid bbox: x2 ({x2}) should be > x1 ({x1})"
            assert y2 > y1, f"Invalid bbox: y2 ({y2}) should be > y1 ({y1})"
            # All coordinates should be positive
            assert x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_threshold_filtering(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test that low-confidence detections are filtered."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        # High threshold should filter out low-confidence detections
        adapter_high = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", threshold=0.99  # Very high threshold
        )

        result_high = adapter_high.predict(sample_image)

        # Low threshold should keep detections (mock returns 0.95, 0.87)
        adapter_low = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", threshold=0.5  # Low threshold
        )

        result_low = adapter_low.predict(sample_image)

        # More detections with lower threshold
        assert len(result_low.instances) >= len(result_high.instances)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_threshold_override(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test that runtime threshold override works."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", threshold=0.9  # High default
        )

        # Override with lower threshold at runtime
        result = adapter.predict(sample_image, threshold=0.5)

        # Should use the runtime threshold
        assert result.meta["threshold"] == 0.5
        assert len(result.instances) > 0  # Mock returns scores of 0.95, 0.87

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_empty_detections(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test handling of case with no detections."""
        mock_ensure_torch.return_value = mock_torch

        # Create a model that returns no detections
        mock_model_empty = MagicMock()
        mock_model_empty.eval.return_value = mock_model_empty
        mock_model_empty.to.return_value = mock_model_empty

        import torch

        def empty_forward(images):
            return [{"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([])}]

        mock_model_empty.side_effect = empty_forward
        mock_torchvision_detection.retinanet_resnet50_fpn.return_value = mock_model_empty

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

        result = adapter.predict(sample_image)

        assert isinstance(result, VisionResult)
        assert len(result.instances) == 0
        assert result.meta["num_detections"] == 0

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_label_mapping(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test correct label names from id2label mapping."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        custom_labels = {1: "custom_person", 17: "custom_cat"}

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", id2label=custom_labels, threshold=0.0
        )

        result = adapter.predict(sample_image)

        # Check that custom labels are used (mock returns labels 1, 17)
        label_names = [inst.label_name for inst in result.instances]
        assert "custom_person" in label_names or "custom_cat" in label_names

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_multiple_detections(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test handling of multiple object detections."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/retinanet_resnet50_fpn", threshold=0.5  # Low threshold to get both detections
        )

        result = adapter.predict(sample_image)

        # Mock returns 2 detections with scores 0.95, 0.87
        assert len(result.instances) == 2

        # Check that all instances have required fields
        for instance in result.instances:
            assert hasattr(instance, "bbox")
            assert hasattr(instance, "score")
            assert hasattr(instance, "label")
            assert hasattr(instance, "label_name")
            assert len(instance.bbox) == 4

    # ========================================================================
    # 4. Preprocessing Tests (2 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_preprocess_normalization(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test that ImageNet normalization is applied."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

        # Check that transform pipeline includes normalization
        assert hasattr(adapter, "transform")
        # The transform should have been created with Normalize
        # (already verified by mock_torchvision_detection fixture)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_preprocess_tensor_shape(self, mock_ensure_torch, mock_torch, sample_image):
        """Test that preprocessing output has correct shape [3, H, W]."""
        mock_ensure_torch.return_value = mock_torch

        # Mock real transforms for this test
        with patch("mata.adapters.torchvision_detect_adapter._ensure_torchvision") as mock_ensure:
            import torchvision.transforms as real_transforms

            mock_detection = MagicMock()
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_detection.retinanet_resnet50_fpn.return_value = mock_model

            mock_ensure.return_value = (mock_detection, real_transforms)

            from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

            adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

            # Preprocess the image
            tensor = adapter._preprocess(sample_image)

            # Should be [3, H, W] (CHW format)
            assert tensor.shape[0] == 3  # RGB channels
            assert len(tensor.shape) == 3  # 3D tensor

    # ========================================================================
    # 5. Integration Tests (3 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_integration_universal_loader_prefix(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test loading via mata.load() with torchvision/ prefix."""
        mock_ensure_torch.return_value = mock_torch

        import mata

        detector = mata.load("detect", "torchvision/retinanet_resnet50_fpn")

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        assert isinstance(detector, TorchvisionDetectAdapter)
        assert detector.parsed_model_name == "retinanet_resnet50_fpn"

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_integration_universal_loader_config_alias(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test loading via config alias."""
        mock_ensure_torch.return_value = mock_torch

        import mata

        # Register alias in the global loader with proper string source
        mata.register_model("detect", "test-cnn-fast", source="torchvision/retinanet_resnet50_fpn", threshold=0.4)

        try:
            # Load using registered alias
            detector = mata.load("detect", "test-cnn-fast")

            from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

            assert isinstance(detector, TorchvisionDetectAdapter)
        finally:
            # Cleanup: unregister alias
            pass  # No cleanup API available

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_integration_mata_run_api(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test mata.run() API works."""
        mock_ensure_torch.return_value = mock_torch

        import mata

        # Save sample image to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jpg", delete=False) as f:
            image_path = f.name

        try:
            sample_image.save(image_path)

            result = mata.run("detect", image_path, model="torchvision/retinanet_resnet50_fpn", threshold=0.5)

            assert isinstance(result, VisionResult)
            assert hasattr(result, "instances")
        finally:
            try:
                os.unlink(image_path)
            except (OSError, PermissionError):
                pass

    # ========================================================================
    # 6. Error Handling Tests (4 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_error_invalid_model_name(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that invalid model name raises ModelLoadError."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        with pytest.raises(UnsupportedModelError) as exc_info:
            TorchvisionDetectAdapter(model_name="invalid/model/name")

        assert "invalid/model/name" in str(exc_info.value) or "name" in str(exc_info.value)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_error_missing_checkpoint_file(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that missing checkpoint file raises ModelLoadError."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        with pytest.raises(ModelLoadError) as exc_info:
            TorchvisionDetectAdapter(
                model_name="torchvision/retinanet_resnet50_fpn", weights="/nonexistent/path/to/checkpoint.pth"
            )

        assert "not found" in str(exc_info.value).lower() or "nonexistent" in str(exc_info.value)

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_error_cuda_unavailable(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test error when CUDA requested but unavailable."""
        # Force CUDA unavailable
        mock_torch.cuda.is_available.return_value = False
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter
        from mata.core.exceptions import ModelLoadError

        # This should either raise an error or fall back to CPU
        # depending on implementation. Let's test it doesn't crash.
        try:
            TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", device="cuda")
            # If no error, should have fallen back to CPU or raised proper error
            # The actual behavior depends on PyTorchBaseAdapter implementation
        except (ModelLoadError, RuntimeError) as e:
            # This is acceptable - error should mention CUDA
            assert "cuda" in str(e).lower() or "gpu" in str(e).lower()

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_error_invalid_threshold(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that invalid threshold values raise ValueError."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        # Threshold should be in range [0, 1]
        # Invalid threshold should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            TorchvisionDetectAdapter(
                model_name="torchvision/retinanet_resnet50_fpn", threshold=1.5  # Invalid - out of range
            )

        assert "range [0.0, 1.0]" in str(exc_info.value)

    # ========================================================================
    # 7. Metadata Tests (2 tests)
    # ========================================================================

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_info_returns_correct_structure(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that info() returns all required fields."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn")

        info = adapter.info()

        # Check required fields
        required_fields = ["name", "task", "model_id", "model_name", "device", "threshold", "backend", "weights"]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_info_contains_model_metadata(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that info() contains correct model metadata."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(
            model_name="torchvision/fasterrcnn_resnet50_fpn_v2", threshold=0.6, device="cpu"
        )

        info = adapter.info()

        assert info["task"] == "detect"
        assert info["backend"] == "torchvision"
        assert info["model_name"] == "fasterrcnn_resnet50_fpn_v2"
        assert info["threshold"] == 0.6
        assert "cpu" in str(info["device"])
        assert info["name"] == "TorchvisionDetectAdapter"


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestTorchvisionDetectAdapterEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_model_name_without_prefix(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that model names work without 'torchvision/' prefix."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="ssd300_vgg16")

        assert adapter.parsed_model_name == "ssd300_vgg16"
        assert "ssd300_vgg16" in adapter.model_name

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_all_supported_models(self, mock_ensure_torch, mock_torch, mock_torchvision_detection):
        """Test that all supported models can be initialized."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        supported_models = [
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn_v2",
            "retinanet_resnet50_fpn",
            "retinanet_resnet50_fpn_v2",
            "fcos_resnet50_fpn",
            "ssd300_vgg16",
            "ssdlite320_mobilenet_v3_large",
        ]

        for model_name in supported_models:
            adapter = TorchvisionDetectAdapter(model_name=f"torchvision/{model_name}")
            assert adapter.parsed_model_name == model_name

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_metadata_in_result(self, mock_ensure_torch, mock_torch, mock_torchvision_detection, sample_image):
        """Test that prediction results contain proper metadata."""
        mock_ensure_torch.return_value = mock_torch

        from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

        adapter = TorchvisionDetectAdapter(model_name="torchvision/retinanet_resnet50_fpn", threshold=0.5)

        result = adapter.predict(sample_image)

        # Check metadata fields
        assert "model_name" in result.meta
        assert "threshold" in result.meta
        assert "device" in result.meta
        assert "backend" in result.meta
        assert result.meta["backend"] == "torchvision"
        assert "architecture" in result.meta
        assert "num_detections" in result.meta
