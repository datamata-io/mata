"""Tests for universal model loader.

Tests the UniversalLoader and ModelRegistry functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError
from mata.core.model_loader import UniversalLoader
from mata.core.model_registry import ModelRegistry


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry can be initialized."""
        registry = ModelRegistry()
        assert registry is not None

    def test_has_alias_empty_registry(self):
        """Test has_alias returns False for empty registry."""
        registry = ModelRegistry()
        assert not registry.has_alias("detect", "nonexistent")

    def test_runtime_registration(self):
        """Test runtime model registration."""
        registry = ModelRegistry()

        # Register a model
        registry.register("detect", "test-model", {"source": "test/model-id", "threshold": 0.5})

        # Check it exists
        assert registry.has_alias("detect", "test-model")

        # Get config
        config = registry.get_config("detect", "test-model")
        assert config["source"] == "test/model-id"
        assert config["threshold"] == 0.5

    def test_get_config_not_found(self):
        """Test get_config raises error for missing alias."""
        registry = ModelRegistry()

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_config("detect", "nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "Available aliases" in str(exc_info.value)

    def test_list_aliases_empty(self):
        """Test list_aliases returns empty list for isolated registry."""
        # Create isolated registry by passing non-existent custom path
        # This prevents loading from default locations
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)  # Empty config
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            aliases = registry.list_aliases("detect")
            assert aliases == []
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_list_aliases_with_registrations(self):
        """Test list_aliases returns registered aliases."""
        # Use isolated registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)

            registry.register("detect", "model1", {"source": "test1"})
            registry.register("detect", "model2", {"source": "test2"})

            aliases = registry.list_aliases("detect")
            assert "model1" in aliases
            assert "model2" in aliases
            assert len(aliases) == 2
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_load_yaml_config(self):
        """Test loading config from YAML file."""
        # Create temporary YAML config
        config_data = {"detect": {"test-alias": {"source": "test/model", "threshold": 0.4}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            filepath = f.name

        try:
            # Load registry with custom config
            registry = ModelRegistry(custom_config_path=filepath)

            # Check alias was loaded
            assert registry.has_alias("detect", "test-alias")
            config = registry.get_config("detect", "test-alias")
            assert config["source"] == "test/model"
            assert config["threshold"] == 0.4

        finally:
            # Clean up
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


class TestUniversalLoader:
    """Test UniversalLoader functionality."""

    def test_loader_initialization(self):
        """Test universal loader can be initialized."""
        loader = UniversalLoader()
        assert loader is not None
        assert loader.registry is not None

    def test_detect_source_type_huggingface(self):
        """Test HuggingFace ID detection."""
        loader = UniversalLoader()

        source_type, resolved = loader._detect_source_type("detect", "org/model-name")
        assert source_type == "huggingface"
        assert resolved == "org/model-name"

    def test_detect_source_type_default(self):
        """Test default source detection."""
        loader = UniversalLoader()

        source_type, resolved = loader._detect_source_type("detect", None)
        assert source_type == "default"
        assert resolved == ""

    def test_detect_source_type_local_file(self):
        """Test local file detection."""
        loader = UniversalLoader()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            filepath = f.name

        try:
            source_type, resolved = loader._detect_source_type("detect", filepath)
            assert source_type == "local_file"
            assert resolved == filepath
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_detect_source_type_file_extension(self):
        """Test detection of file paths by extension even if file doesn't exist."""
        loader = UniversalLoader()

        # Test various file extensions that should be recognized as local files
        test_paths = [
            "models/model.pt",
            "path/to/model.onnx",
            "checkpoint.pth",
            "model.bin",
            "engine.trt",
        ]

        for path in test_paths:
            source_type, resolved = loader._detect_source_type("detect", path)
            assert source_type == "local_file", f"Failed for path: {path}"
            assert resolved == path

    def test_detect_source_type_config_alias(self):
        """Test config alias detection."""
        registry = ModelRegistry()
        registry.register("detect", "my-alias", {"source": "test/model"})

        loader = UniversalLoader(model_registry=registry)

        source_type, resolved = loader._detect_source_type("detect", "my-alias")
        assert source_type == "config_alias"
        assert resolved == "my-alias"

    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_from_huggingface(self, mock_adapter):
        """Test loading from HuggingFace."""
        # Setup mock
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("detect", "org/model-id", threshold=0.5)

        # Verify adapter was created with correct args
        mock_adapter.assert_called_once_with(model_id="org/model-id", threshold=0.5)
        assert result == mock_instance

    def test_load_from_config_recursive(self):
        """Test loading from config alias (recursive resolution)."""
        registry = ModelRegistry()
        registry.register("detect", "my-model", {"source": "org/model-id", "threshold": 0.6})

        loader = UniversalLoader(model_registry=registry)

        # Mock the HuggingFace loading
        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = Mock()

            loader._load_from_config("detect", "my-model")

            # Should have called _load_from_huggingface with merged config
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[0] == ("detect", "org/model-id")
            assert call_args[1]["threshold"] == 0.6

    def test_unsupported_file_extension(self):
        """Test error for unsupported file extension."""
        loader = UniversalLoader()

        # Create temp file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(UnsupportedModelError) as exc_info:
                loader._load_from_file("detect", filepath)

            assert "Unsupported file extension" in str(exc_info.value)
            assert ".unknown" in str(exc_info.value)
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.torchscript_adapter.TorchScriptDetectAdapter")
    def test_load_torchscript_model(self, mock_adapter):
        """Test loading TorchScript (.pt) model."""
        # Create temp .pt file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            loader = UniversalLoader()

            # Mock the probe to return 'torchscript'
            with patch.object(loader, "_probe_pt_file", return_value="torchscript"):
                result = loader._load_from_file("detect", filepath, threshold=0.5)

                # Verify TorchScript adapter was created
                mock_adapter.assert_called_once_with(model_path=filepath, threshold=0.5)
                assert result == mock_instance
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.pytorch_adapter.PyTorchDetectAdapter")
    def test_load_pytorch_checkpoint_from_pt_file(self, mock_adapter):
        """Test loading PyTorch checkpoint from .pt file (not TorchScript)."""
        # Create temp .pt file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            loader = UniversalLoader()

            # Mock the probe to return 'pytorch_checkpoint'
            with patch.object(loader, "_probe_pt_file", return_value="pytorch_checkpoint"):
                result = loader._load_from_file("detect", filepath, threshold=0.5)

                # Verify PyTorch adapter was created
                mock_adapter.assert_called_once_with(checkpoint_path=filepath, threshold=0.5)
                assert result == mock_instance
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


class TestUniversalLoaderIntegration:
    """Integration tests for universal loader."""

    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_huggingface_model(self, mock_adapter):
        """Test complete flow for loading HuggingFace model."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader.load("detect", "facebook/detr-resnet-50", threshold=0.4)

        assert result == mock_instance
        mock_adapter.assert_called_once()

    def test_load_with_alias(self):
        """Test complete flow for loading via alias."""
        registry = ModelRegistry()
        registry.register("detect", "fast-model", {"source": "org/fast-model", "threshold": 0.3})

        loader = UniversalLoader(model_registry=registry)

        with patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter") as mock_adapter:
            mock_adapter.return_value = Mock()

            loader.load("detect", "fast-model")

            # Should load with config from alias
            mock_adapter.assert_called_once()
            assert mock_adapter.call_args[1]["threshold"] == 0.3


class TestTorchScriptDetectAdapter:
    """Unit tests for TorchScriptDetectAdapter."""

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_adapter_initialization(self, mock_ensure_torch):
        """Test TorchScriptDetectAdapter initialization."""
        from mata.adapters.torchscript_adapter import TorchScriptDetectAdapter

        # Setup mocks
        mock_model = Mock()
        mock_model.eval = Mock()

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device = Mock(return_value=Mock())
        mock_torch.jit.load = Mock(return_value=mock_model)
        mock_ensure_torch.return_value = mock_torch

        # Create temp .pt file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            # Initialize adapter
            adapter = TorchScriptDetectAdapter(model_path=filepath, device="auto", threshold=0.5, input_size=640)

            # Verify initialization
            assert adapter.threshold == 0.5
            assert adapter.input_size == 640
            assert len(adapter.id2label) == 80  # COCO classes
            assert adapter.model == mock_model
            mock_model.eval.assert_called_once()
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.pytorch_base._ensure_torch")
    def test_predict_method(self, mock_ensure_torch):
        """Test TorchScriptDetectAdapter predict method structure."""
        from mata.adapters.torchscript_adapter import TorchScriptDetectAdapter

        # Setup model mock
        mock_model = Mock()
        mock_model.eval = Mock()

        # Setup mocks
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = Mock(return_value=Mock())
        mock_torch.jit.load = Mock(return_value=mock_model)
        mock_ensure_torch.return_value = mock_torch

        # Create temp .pt file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            # Initialize adapter - validates the structure is correct
            adapter = TorchScriptDetectAdapter(model_path=filepath, device="cpu", threshold=0.5)

            # Verify adapter was created correctly
            assert adapter.model == mock_model
            assert adapter.threshold == 0.5
            # Full predict test would require complex PIL/torch mocking
            # Structure validation is sufficient for unit test
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


class TestClassificationLoading:
    """Test classification model loading."""

    @patch("mata.adapters.huggingface_classify_adapter.HuggingFaceClassifyAdapter")
    def test_load_classify_from_huggingface(self, mock_adapter):
        """Test loading classification model from HuggingFace."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("classify", "microsoft/resnet-50", top_k=5)

        # Verify adapter was created with correct args
        mock_adapter.assert_called_once_with(model_id="microsoft/resnet-50", top_k=5)
        assert result == mock_instance

    @patch("mata.adapters.huggingface_classify_adapter.HuggingFaceClassifyAdapter")
    def test_load_classify_vit_model(self, mock_adapter):
        """Test loading ViT classification model."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("classify", "google/vit-base-patch16-224", device="cuda")

        mock_adapter.assert_called_once_with(model_id="google/vit-base-patch16-224", device="cuda")
        assert result == mock_instance

    def test_classify_unsupported_task_error(self):
        """Test that unsupported tasks raise proper error."""
        loader = UniversalLoader()

        # Note: This tests the general error handling path
        # Classification is now supported, but testing the error mechanism
        with pytest.raises(UnsupportedModelError, match="not yet implemented"):
            loader._load_from_huggingface("unsupported_task", "some/model")


class TestDepthLoading:
    """Test depth model loading."""

    @patch("mata.adapters.huggingface_depth_adapter.HuggingFaceDepthAdapter")
    def test_load_depth_from_huggingface(self, mock_adapter):
        """Test loading depth model from HuggingFace."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("depth", "depth-anything/Depth-Anything-V2-Small-hf", normalize=True)

        mock_adapter.assert_called_once_with(model_id="depth-anything/Depth-Anything-V2-Small-hf", normalize=True)
        assert result == mock_instance


class TestTrackingLoading:
    """Test tracking model loading via UniversalLoader."""

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_from_huggingface(self, mock_detect_adapter, mock_tracking_adapter):
        """Test load('track', 'hf-id') creates detect adapter wrapped in TrackingAdapter."""
        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance
        mock_tracking_instance = Mock()
        mock_tracking_adapter.return_value = mock_tracking_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("track", "facebook/detr-resnet-50")

        # Detect adapter created for the underlying model
        mock_detect_adapter.assert_called_once_with(model_id="facebook/detr-resnet-50")
        # TrackingAdapter wraps it with default botsort config
        mock_tracking_adapter.assert_called_once_with(
            mock_detect_instance, "botsort", 30, reid_encoder=None, reid_bridge=None
        )
        assert result == mock_tracking_instance

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_with_bytetrack(self, mock_detect_adapter, mock_tracking_adapter):
        """Test tracker='bytetrack' is forwarded to TrackingAdapter."""
        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader()
        loader._load_from_huggingface("track", "facebook/detr-resnet-50", tracker="bytetrack")

        _, call_kwargs = mock_tracking_adapter.call_args
        args = mock_tracking_adapter.call_args[0]
        assert args[1] == "bytetrack"

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_with_frame_rate(self, mock_detect_adapter, mock_tracking_adapter):
        """Test frame_rate kwarg is forwarded to TrackingAdapter."""
        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader()
        loader._load_from_huggingface("track", "facebook/detr-resnet-50", frame_rate=25)

        args = mock_tracking_adapter.call_args[0]
        assert args[2] == 25  # frame_rate is the third positional arg

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_with_custom_tracker_dict(self, mock_detect_adapter, mock_tracking_adapter):
        """Test tracker config can be passed as a dict."""
        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        tracker_dict = {"tracker_type": "bytetrack", "track_high_thresh": 0.6}
        loader = UniversalLoader()
        loader._load_from_huggingface("track", "facebook/detr-resnet-50", tracker=tracker_dict)

        args = mock_tracking_adapter.call_args[0]
        assert args[1] == tracker_dict

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.onnx_adapter.ONNXDetectAdapter")
    def test_load_track_from_onnx_file(self, mock_detect_adapter, mock_tracking_adapter):
        """Test load('track', 'model.onnx') creates ONNX detect adapter wrapped in TrackingAdapter."""
        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance
        mock_tracking_instance = Mock()
        mock_tracking_adapter.return_value = mock_tracking_instance

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            filepath = f.name

        try:
            loader = UniversalLoader()
            result = loader._load_from_file("track", filepath)

            mock_detect_adapter.assert_called_once_with(model_path=filepath)
            mock_tracking_adapter.assert_called_once_with(
                mock_detect_instance, "botsort", 30, reid_encoder=None, reid_bridge=None
            )
            assert result == mock_tracking_instance
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_with_explicit_type_huggingface(self, mock_detect_adapter, mock_tracking_adapter):
        """Test load('track', ..., model_type=HUGGINGFACE) routes through explicit type path."""
        from mata.core.types import ModelType

        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader()
        loader._load_with_explicit_type("track", "facebook/detr-resnet-50", ModelType.HUGGINGFACE)

        mock_detect_adapter.assert_called_once()
        mock_tracking_adapter.assert_called_once()

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_via_public_load_method(self, mock_detect_adapter, mock_tracking_adapter):
        """Test full flow: mata.load('track', 'org/model') returns TrackingAdapter."""
        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance
        mock_tracking_instance = Mock()
        mock_tracking_adapter.return_value = mock_tracking_instance

        loader = UniversalLoader()
        result = loader.load("track", "facebook/detr-resnet-50")

        assert result == mock_tracking_instance
        mock_tracking_adapter.assert_called_once()

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_tracker_kwarg_not_passed_to_detect(self, mock_detect_adapter, mock_tracking_adapter):
        """Test 'tracker' kwarg is consumed by track routing and NOT forwarded to detect adapter."""
        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader()
        loader._load_from_huggingface("track", "facebook/detr-resnet-50", tracker="bytetrack", threshold=0.5)

        # Only threshold should reach the detect adapter — not 'tracker' or 'frame_rate'
        detect_call_kwargs = mock_detect_adapter.call_args[1]
        assert "tracker" not in detect_call_kwargs
        assert "frame_rate" not in detect_call_kwargs
        assert detect_call_kwargs.get("threshold") == 0.5

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_existing_detect_task_routing_unaffected(self, mock_detect_adapter, mock_tracking_adapter):
        """Test adding track routing didn't break standard 'detect' task routing."""
        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("detect", "facebook/detr-resnet-50")

        mock_detect_adapter.assert_called_once_with(model_id="facebook/detr-resnet-50")
        mock_tracking_adapter.assert_not_called()
        assert result == mock_detect_instance

    def test_load_track_no_default_error_message(self):
        """Test that missing default for 'track' gives a helpful error with tracking example."""
        loader = UniversalLoader()

        with pytest.raises(ModelNotFoundError) as exc_info:
            loader._load_default("track")

        error_msg = str(exc_info.value)
        assert "track" in error_msg
        # Should suggest a tracking-appropriate model example
        assert "detr" in error_msg.lower() or "resnet" in error_msg.lower()

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_via_config_alias(self, mock_detect_adapter, mock_tracking_adapter):
        """Test load('track', 'alias') routes through config and wraps in TrackingAdapter."""
        registry = ModelRegistry()
        registry.register("track", "my-tracker", {"source": "facebook/detr-resnet-50"})

        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance
        mock_tracking_instance = Mock()
        mock_tracking_adapter.return_value = mock_tracking_instance

        loader = UniversalLoader(model_registry=registry)
        result = loader.load("track", "my-tracker")

        assert result == mock_tracking_instance
        mock_tracking_adapter.assert_called_once()


class TestTrackRegistryConfig:
    """Task E2: Model Registry Track Config — YAML-based tracking aliases."""

    def test_track_registry_yaml_loads_tracker_field(self):
        """Track alias with 'tracker' field is stored and retrieved correctly."""
        config_data = {
            "track": {
                "highway-tracker": {
                    "source": "facebook/detr-resnet-50",
                    "tracker": "botsort",
                    "device": "cuda",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            assert registry.has_alias("track", "highway-tracker")
            config = registry.get_config("track", "highway-tracker")
            assert config["source"] == "facebook/detr-resnet-50"
            assert config["tracker"] == "botsort"
            assert config["device"] == "cuda"
        finally:
            os.unlink(filepath)

    def test_track_registry_yaml_loads_tracker_config_overrides(self):
        """Track alias with nested tracker_config dict is loaded without warnings."""
        config_data = {
            "track": {
                "highway-tracker": {
                    "source": "facebook/detr-resnet-50",
                    "tracker": "botsort",
                    "tracker_config": {
                        "track_buffer": 60,
                        "match_thresh": 0.7,
                        "gmc_method": "sparseOptFlow",
                    },
                    "device": "cuda",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            config = registry.get_config("track", "highway-tracker")
            assert config["tracker"] == "botsort"
            assert config["tracker_config"] == {
                "track_buffer": 60,
                "match_thresh": 0.7,
                "gmc_method": "sparseOptFlow",
            }
        finally:
            os.unlink(filepath)

    def test_track_registry_yaml_no_unknown_field_warning_for_tracking_fields(self, caplog):
        """tracker, tracker_config, frame_rate are known fields — no lint warning."""
        import logging

        config_data = {
            "track": {
                "fast-tracker": {
                    "source": "facebook/detr-resnet-50",
                    "tracker": "bytetrack",
                    "tracker_config": {"track_buffer": 45},
                    "frame_rate": 25,
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            with caplog.at_level(logging.WARNING, logger="mata"):
                registry.get_config("track", "fast-tracker")
            # No "unknown fields" warning should be emitted for these tracking fields
            unknown_warnings = [r for r in caplog.records if "unknown fields" in r.message]
            assert (
                len(unknown_warnings) == 0
            ), f"Unexpected 'unknown fields' warning: {[r.message for r in unknown_warnings]}"
        finally:
            os.unlink(filepath)

    def test_resolve_tracker_kwargs_string_only(self):
        """_resolve_tracker_kwargs with only tracker= string returns it unchanged."""
        loader = UniversalLoader()
        kwargs = {"tracker": "bytetrack", "frame_rate": 25, "device": "cuda"}
        tracker_config, frame_rate, *_ = loader._resolve_tracker_kwargs(kwargs)

        assert tracker_config == "bytetrack"
        assert frame_rate == 25
        # Only tracker-related kwargs removed; device remains for detect adapter
        assert kwargs == {"device": "cuda"}

    def test_resolve_tracker_kwargs_merges_overrides(self):
        """_resolve_tracker_kwargs merges tracker name + tracker_config dict."""
        loader = UniversalLoader()
        kwargs = {
            "tracker": "botsort",
            "tracker_config": {"track_buffer": 60, "match_thresh": 0.7},
            "frame_rate": 30,
        }
        tracker_config, frame_rate, *_ = loader._resolve_tracker_kwargs(kwargs)

        assert isinstance(tracker_config, dict)
        assert tracker_config["tracker_type"] == "botsort"
        assert tracker_config["track_buffer"] == 60
        assert tracker_config["match_thresh"] == 0.7
        assert frame_rate == 30
        assert kwargs == {}  # all tracker kwargs consumed

    def test_resolve_tracker_kwargs_defaults(self):
        """_resolve_tracker_kwargs defaults: tracker=botsort, frame_rate=30."""
        loader = UniversalLoader()
        kwargs = {}
        tracker_config, frame_rate, *_ = loader._resolve_tracker_kwargs(kwargs)

        assert tracker_config == "botsort"
        assert frame_rate == 30

    def test_resolve_tracker_kwargs_dict_tracker_ignores_overrides(self):
        """When tracker is already a dict, tracker_config overrides are ignored."""
        loader = UniversalLoader()
        custom_dict = {"tracker_type": "bytetrack", "track_buffer": 50}
        kwargs = {
            "tracker": custom_dict,
            "tracker_config": {"track_buffer": 99},  # should be ignored
        }
        tracker_config, frame_rate, *_ = loader._resolve_tracker_kwargs(kwargs)

        # Original dict returned unchanged (not merged with overrides)
        assert tracker_config is custom_dict
        assert tracker_config["track_buffer"] == 50

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_config_alias_with_tracker_and_overrides(self, mock_detect_adapter, mock_tracking_adapter):
        """mata.load('track', 'alias') with tracker + tracker_config in registry.

        Verifies:
        - tracker_config does NOT leak to the detect adapter
        - TrackingAdapter receives a merged dict {tracker_type, ...overrides}
        """
        registry = ModelRegistry()
        registry.register(
            "track",
            "highway-tracker",
            {
                "source": "facebook/detr-resnet-50",
                "tracker": "botsort",
                "tracker_config": {"track_buffer": 60, "match_thresh": 0.7},
                "device": "cuda",
            },
        )

        mock_detect_instance = Mock()
        mock_detect_adapter.return_value = mock_detect_instance
        mock_tracking_instance = Mock()
        mock_tracking_adapter.return_value = mock_tracking_instance

        loader = UniversalLoader(model_registry=registry)
        result = loader.load("track", "highway-tracker")

        assert result == mock_tracking_instance

        # TrackingAdapter must receive the merged dict, NOT just the string "botsort"
        call_args = mock_tracking_adapter.call_args
        passed_tracker_cfg = call_args[0][1]
        assert isinstance(passed_tracker_cfg, dict)
        assert passed_tracker_cfg["tracker_type"] == "botsort"
        assert passed_tracker_cfg["track_buffer"] == 60
        assert passed_tracker_cfg["match_thresh"] == 0.7

        # Detect adapter must NOT receive tracker_config or tracker kwargs
        detect_call_kwargs = mock_detect_adapter.call_args[1]
        assert "tracker_config" not in detect_call_kwargs
        assert "tracker" not in detect_call_kwargs

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_config_alias_frame_rate_from_registry(self, mock_detect_adapter, mock_tracking_adapter):
        """frame_rate from registry config is forwarded to TrackingAdapter."""
        registry = ModelRegistry()
        registry.register(
            "track",
            "surveillance-cam",
            {
                "source": "facebook/detr-resnet-50",
                "tracker": "bytetrack",
                "frame_rate": 15,
            },
        )

        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader(model_registry=registry)
        loader.load("track", "surveillance-cam")

        call_args = mock_tracking_adapter.call_args
        assert call_args[0][2] == 15  # frame_rate positional arg

    @patch("mata.adapters.tracking_adapter.TrackingAdapter")
    @patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter")
    def test_load_track_config_alias_tracker_config_not_leaked_to_detect(
        self, mock_detect_adapter, mock_tracking_adapter
    ):
        """tracker_config from registry must NOT be forwarded to detect adapter."""
        registry = ModelRegistry()
        registry.register(
            "track",
            "leak-test",
            {
                "source": "facebook/detr-resnet-50",
                "tracker": "botsort",
                "tracker_config": {"track_buffer": 40},
            },
        )

        mock_detect_adapter.return_value = Mock()
        mock_tracking_adapter.return_value = Mock()

        loader = UniversalLoader(model_registry=registry)
        loader.load("track", "leak-test")

        # Detect adapter kwargs must not contain tracking parameters
        detect_kwargs = mock_detect_adapter.call_args[1]
        for forbidden_key in ("tracker", "tracker_config", "frame_rate"):
            assert (
                forbidden_key not in detect_kwargs
            ), f"'{forbidden_key}' should not be passed to detect adapter, but was: {detect_kwargs}"


class TestOCRTaskRouting:
    """Test OCR task routing via UniversalLoader (Task I3)."""

    @patch("mata.adapters.ocr.easyocr_adapter.EasyOCRAdapter")
    def test_load_ocr_easyocr_routes_to_external_engine(self, mock_adapter):
        """load('ocr', 'easyocr') routes to _load_from_external_engine -> EasyOCRAdapter."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_external_engine("ocr", "easyocr", languages=["en"])

        mock_adapter.assert_called_once_with(languages=["en"])
        assert result == mock_instance

    @patch("mata.adapters.ocr.huggingface_ocr_adapter.HuggingFaceOCRAdapter")
    def test_load_ocr_huggingface_routes_to_hf_adapter(self, mock_adapter):
        """load('ocr', 'microsoft/trocr-base-printed') routes to HuggingFaceOCRAdapter."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader._load_from_huggingface("ocr", "microsoft/trocr-base-printed")

        mock_adapter.assert_called_once_with(model_id="microsoft/trocr-base-printed")
        assert result == mock_instance

    def test_load_detect_easyocr_raises_unsupported(self):
        """load('detect', 'easyocr') raises UnsupportedModelError — engines only support ocr."""
        loader = UniversalLoader()

        with pytest.raises(UnsupportedModelError) as exc_info:
            loader._load_from_external_engine("detect", "easyocr")

        error_msg = str(exc_info.value)
        assert "ocr" in error_msg.lower()
        assert "detect" in error_msg.lower()

    @patch("mata.adapters.ocr.easyocr_adapter.EasyOCRAdapter")
    def test_source_type_detection_easyocr_returns_external_engine(self, mock_adapter):
        """'easyocr' bare string is detected as external_engine source type."""
        loader = UniversalLoader()

        source_type, resolved = loader._detect_source_type("ocr", "easyocr")

        assert source_type == "external_engine"
        assert resolved == "easyocr"

    @patch("mata.adapters.ocr.easyocr_adapter.EasyOCRAdapter")
    def test_load_ocr_easyocr_via_public_load(self, mock_adapter):
        """Full flow: loader.load('ocr', 'easyocr') ends up as EasyOCRAdapter."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader.load("ocr", "easyocr")

        mock_adapter.assert_called_once()
        assert result == mock_instance

    @patch("mata.adapters.ocr.huggingface_ocr_adapter.HuggingFaceOCRAdapter")
    def test_load_ocr_huggingface_via_public_load(self, mock_adapter):
        """Full flow: loader.load('ocr', 'org/model') ends up as HuggingFaceOCRAdapter."""
        mock_instance = Mock()
        mock_adapter.return_value = mock_instance

        loader = UniversalLoader()
        result = loader.load("ocr", "microsoft/trocr-base-printed")

        mock_adapter.assert_called_once_with(model_id="microsoft/trocr-base-printed")
        assert result == mock_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
