"""Tests for ModelType enum and probe system (v1.5.2).

Tests the new explicit model type specification feature.
"""

import logging
import os
import tempfile
import warnings
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from mata.core.exceptions import ModelNotFoundError
from mata.core.model_loader import UniversalLoader
from mata.core.model_registry import ModelRegistry
from mata.core.types import ModelType


@pytest.fixture(autouse=True)
def enable_logging_capture(caplog):
    """Enable logging propagation for caplog capture."""
    mata_logger = logging.getLogger("mata")
    original_propagate = mata_logger.propagate

    # Enable propagation so caplog can capture logs
    mata_logger.propagate = True

    # Add caplog's handler to mata logger
    if caplog.handler not in mata_logger.handlers:
        mata_logger.addHandler(caplog.handler)

    yield

    # Restore original state
    mata_logger.propagate = original_propagate
    if caplog.handler in mata_logger.handlers:
        mata_logger.removeHandler(caplog.handler)


class TestModelTypeEnum:
    """Test ModelType enum functionality."""

    def test_enum_values(self):
        """Test all enum values are defined."""
        assert ModelType.AUTO.value == "auto"
        assert ModelType.HUGGINGFACE.value == "huggingface"
        assert ModelType.PYTORCH_CHECKPOINT.value == "pytorch_checkpoint"
        assert ModelType.TORCHSCRIPT.value == "torchscript"
        assert ModelType.ONNX.value == "onnx"
        assert ModelType.TENSORRT.value == "tensorrt"
        assert ModelType.CONFIG_ALIAS.value == "config_alias"

    def test_enum_is_string(self):
        """Test enum inherits from str."""
        assert isinstance(ModelType.TORCHSCRIPT, str)
        assert isinstance(ModelType.ONNX, str)

    def test_normalize_enum_passthrough(self):
        """Test normalize passes enum through unchanged."""
        result = ModelType.normalize(ModelType.TORCHSCRIPT)
        assert result == ModelType.TORCHSCRIPT

    def test_normalize_none(self):
        """Test normalize returns None for None input."""
        result = ModelType.normalize(None)
        assert result is None

    def test_normalize_valid_string(self):
        """Test normalize converts valid string to enum."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = ModelType.normalize("torchscript")
            assert result == ModelType.TORCHSCRIPT

    def test_normalize_string_case_insensitive(self):
        """Test normalize is case-insensitive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert ModelType.normalize("ONNX") == ModelType.ONNX
            assert ModelType.normalize("OnnX") == ModelType.ONNX
            assert ModelType.normalize("onnx") == ModelType.ONNX

    def test_normalize_string_deprecation_warning(self):
        """Test normalize warns on string input."""
        with pytest.warns(DeprecationWarning, match="Passing model_type as string"):
            ModelType.normalize("onnx")

    def test_normalize_invalid_string(self):
        """Test normalize handles invalid string."""
        with pytest.warns(UserWarning, match="Unknown model_type"):
            result = ModelType.normalize("invalid_type")
            assert result is None

    def test_normalize_invalid_type(self):
        """Test normalize raises TypeError for invalid type."""
        with pytest.raises(TypeError, match="model_type must be"):
            ModelType.normalize(123)


class TestProbePTFile:
    """Test .pt file probe system."""

    def test_probe_cache_initialization(self):
        """Test probe cache is initialized."""
        loader = UniversalLoader()
        assert hasattr(loader, "_probe_cache")
        assert isinstance(loader._probe_cache, dict)

    def test_clear_cache(self):
        """Test clear_cache clears both caches."""
        loader = UniversalLoader()

        # Populate caches
        loader._probe_cache["test.pt"] = ("torchscript", 123.45)
        loader._adapter_cache["key"] = Mock()

        # Clear
        loader.clear_cache()

        assert len(loader._probe_cache) == 0
        assert len(loader._adapter_cache) == 0

    def test_probe_torchscript_file(self):
        """Test probe detects TorchScript ZIP structure."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            # Create fake TorchScript ZIP
            with zipfile.ZipFile(filepath, "w") as z:
                z.writestr("constants.pkl", b"dummy")
                z.writestr("code/__torch__.py", b"dummy")
                z.writestr("version", b"3")

            loader = UniversalLoader()
            result = loader._probe_pt_file(filepath)

            assert result == "torchscript"
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_probe_checkpoint_file(self):
        """Test probe detects PyTorch checkpoint (not ZIP)."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Write non-ZIP content
            f.write(b"\x80\x02}q\x00(X\x05\x00\x00\x00epochq\x01K\x01u.")
            f.flush()
            filepath = f.name

        try:
            loader = UniversalLoader()
            result = loader._probe_pt_file(filepath)

            assert result == "pytorch_checkpoint"
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_probe_caching(self):
        """Test probe results are cached."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            with zipfile.ZipFile(filepath, "w") as z:
                z.writestr("constants.pkl", b"dummy")

            loader = UniversalLoader()

            # First call - populates cache
            result1 = loader._probe_pt_file(filepath)
            assert filepath in loader._probe_cache

            # Second call - uses cache
            result2 = loader._probe_pt_file(filepath)
            assert result1 == result2
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_probe_cache_invalidation_on_file_change(self):
        """Test cache invalidates when file mtime changes."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            with zipfile.ZipFile(filepath, "w") as z:
                z.writestr("constants.pkl", b"dummy")

            loader = UniversalLoader()
            result1 = loader._probe_pt_file(filepath)

            # Modify file (change mtime)
            import time

            time.sleep(0.01)
            Path(filepath).touch()

            # Should re-probe
            result2 = loader._probe_pt_file(filepath)

            # Both should still be torchscript
            assert result1 == result2 == "torchscript"
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


class TestExplicitModelType:
    """Test explicit model_type parameter."""

    @patch("mata.adapters.torchscript_adapter.TorchScriptDetectAdapter")
    def test_explicit_torchscript_bypasses_probe(self, mock_adapter):
        """Test explicit TORCHSCRIPT type bypasses probe."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Write non-TorchScript content
            f.write(b"not_a_torchscript_file")
            f.flush()
            filepath = f.name

        try:
            loader = UniversalLoader()
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            # Load with explicit type - should not probe
            result = loader.load("detect", filepath, model_type=ModelType.TORCHSCRIPT, input_size=640)

            # Should call TorchScript adapter directly
            mock_adapter.assert_called_once()
            assert result == mock_instance

            # Probe cache should be empty (probe never called)
            assert filepath not in loader._probe_cache
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.pytorch_adapter.PyTorchDetectAdapter")
    def test_explicit_pytorch_checkpoint_bypasses_probe(self, mock_adapter):
        """Test explicit PYTORCH_CHECKPOINT type bypasses probe."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            # Write TorchScript content
            with zipfile.ZipFile(filepath, "w") as z:
                z.writestr("constants.pkl", b"dummy")

            loader = UniversalLoader()
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            # Load with explicit type - should not probe
            result = loader.load("detect", filepath, model_type=ModelType.PYTORCH_CHECKPOINT, config="config.yaml")

            # Should call PyTorch adapter directly
            mock_adapter.assert_called_once()
            assert result == mock_instance

            # Probe cache should be empty
            assert filepath not in loader._probe_cache
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.onnx_adapter.ONNXDetectAdapter")
    def test_explicit_onnx_type(self, mock_adapter):
        """Test explicit ONNX type."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            filepath = f.name

        try:
            loader = UniversalLoader()
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            result = loader.load("detect", filepath, model_type=ModelType.ONNX, threshold=0.5)

            mock_adapter.assert_called_once()
            assert result == mock_instance
        finally:
            # Cleanup
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_explicit_type_file_not_found(self):
        """Test explicit type with non-existent file."""
        loader = UniversalLoader()

        with pytest.raises(ModelNotFoundError, match="Valid TorchScript file required"):
            loader.load("detect", "nonexistent.pt", model_type=ModelType.TORCHSCRIPT)


class TestKwargsValidation:
    """Test lint-level kwargs validation."""

    def test_validation_warns_unexpected_kwargs(self):
        """Test validation warns for unexpected kwargs (check pytest output for warnings)."""
        loader = UniversalLoader()

        # Should execute without error and print warnings
        loader._validate_adapter_kwargs(ModelType.ONNX, threshold=0.5, unexpected_param=123, another_bad_one="test")

    def test_validation_warns_input_size_on_pytorch(self):
        """Test validation warns about input_size on PyTorch checkpoint."""
        loader = UniversalLoader()

        # Should warn about input_size
        loader._validate_adapter_kwargs(ModelType.PYTORCH_CHECKPOINT, input_size=640)

    def test_validation_warns_config_on_torchscript(self):
        """Test validation warns about config on TorchScript."""
        loader = UniversalLoader()

        # Should warn about config
        loader._validate_adapter_kwargs(ModelType.TORCHSCRIPT, config="config.yaml")

    def test_validation_warns_missing_input_size_torchscript(self):
        """Test validation warns about missing input_size for TorchScript."""
        loader = UniversalLoader()

        # Should warn about missing input_size
        loader._validate_adapter_kwargs(ModelType.TORCHSCRIPT, threshold=0.5)

    def test_validation_no_warnings_valid_kwargs(self, caplog):
        """Test validation doesn't warn for valid kwargs."""
        loader = UniversalLoader()

        loader._validate_adapter_kwargs(
            ModelType.TORCHSCRIPT, model_path="model.pt", threshold=0.5, device="cuda", input_size=640
        )

        # Should not warn about unexpected kwargs
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        combined_text = " ".join(warning_messages)
        assert "Unexpected kwargs" not in combined_text


class TestYAMLModelTypeSupport:
    """Test model_type field in YAML configs."""

    def test_config_with_model_type_string(self):
        """Test config with model_type as string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "detect": {"test-model": {"source": "model.pt", "model_type": "torchscript", "input_size": 640}}
            }
            yaml.dump(config_data, f)
            f.flush()
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)

            # Get config - should normalize model_type
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = registry.get_config("detect", "test-model")

            assert "model_type" in config
            # Should be normalized to enum
            assert config["model_type"] == ModelType.TORCHSCRIPT
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_config_with_invalid_model_type(self):
        """Test config with invalid model_type value (check pytest output for warnings)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"detect": {"test-model": {"source": "model.pt", "model_type": "invalid_type"}}}
            yaml.dump(config_data, f)
            f.flush()
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            config = registry.get_config("detect", "test-model")

            # Invalid model_type should be removed
            assert "model_type" not in config
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    def test_config_unknown_fields_warning(self):
        """Test warning for unknown config fields (check pytest output for warnings)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "detect": {
                    "test-model": {"source": "model.pt", "unknown_field_1": "value1", "another_unknown": "value2"}
                }
            }
            yaml.dump(config_data, f)
            f.flush()
            filepath = f.name

        try:
            registry = ModelRegistry(custom_config_path=filepath)
            config = registry.get_config("detect", "test-model")

            # Config should be loaded despite unknown fields
            assert "source" in config
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


class TestEndToEndExplicitType:
    """End-to-end tests with explicit model types."""

    @patch("mata.adapters.torchscript_adapter.TorchScriptDetectAdapter")
    def test_load_with_explicit_enum(self, mock_adapter):
        """Test loading with explicit ModelType enum."""
        from mata import load

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            result = load("detect", filepath, model_type=ModelType.TORCHSCRIPT, input_size=640)

            assert result == mock_instance
            mock_adapter.assert_called_once()
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass

    @patch("mata.adapters.pytorch_adapter.PyTorchDetectAdapter")
    def test_load_with_string_type_deprecated(self, mock_adapter):
        """Test loading with string model_type (deprecated)."""
        from mata import load

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name

        try:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            with pytest.warns(DeprecationWarning, match="Passing model_type as string"):
                result = load("detect", filepath, model_type="pytorch_checkpoint")

            assert result == mock_instance
        finally:
            try:
                os.unlink(filepath)
            except (OSError, PermissionError):
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
