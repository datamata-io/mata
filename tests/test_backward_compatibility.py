"""Comprehensive backward compatibility tests for MATA public APIs.

This test suite ensures that all existing mata.load() and mata.run() API signatures,
return types, error behaviors, and expected functionality remain unchanged.
"""

import json
from unittest.mock import Mock, patch

import pytest

import mata
from mata.core import (
    ClassifyResult,
    DepthResult,
    Detection,
    DetectResult,
    Instance,
    VisionResult,
)
from mata.core.exceptions import (
    InvalidInputError,
    MATAError,
    ModelLoadError,
    ModelNotFoundError,
    TaskNotSupportedError,
)
from mata.core.types import ModelType


class TestPublicAPISignatures:
    """Test that public API function signatures are unchanged."""

    def test_mata_load_signature(self):
        """Test load() signature matches expected parameters."""
        import inspect

        sig = inspect.signature(mata.load)
        params = list(sig.parameters.keys())

        # Check required and optional parameters
        assert "task" in params
        assert "model" in params
        assert "model_type" in params

        # Check parameter defaults
        assert sig.parameters["model"].default is None
        assert sig.parameters["model_type"].default is None

        # Check kwargs
        assert sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD

    def test_mata_run_signature(self):
        """Test run() signature matches expected parameters."""
        import inspect

        sig = inspect.signature(mata.run)
        params = list(sig.parameters.keys())

        # Check required and optional parameters
        assert "task" in params
        assert "input" in params
        assert "model" in params
        assert "model_type" in params

        # Check parameter defaults
        assert sig.parameters["model"].default is None
        assert sig.parameters["model_type"].default is None

        # Check kwargs
        assert sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD

    def test_mata_list_models_signature(self):
        """Test list_models() signature is unchanged."""
        import inspect

        sig = inspect.signature(mata.list_models)
        params = sig.parameters

        assert "task" in params
        assert "limit" in params
        assert "sort" in params
        assert params["task"].default is None
        assert params["limit"].default == 20
        assert params["sort"].default == "downloads"

    def test_mata_get_model_info_signature(self):
        """Test get_model_info() signature is unchanged."""
        import inspect

        sig = inspect.signature(mata.get_model_info)
        params = list(sig.parameters.keys())

        assert params == ["model_id"]
        assert sig.parameters["model_id"].default == inspect.Parameter.empty

    def test_mata_register_model_signature(self):
        """Test register_model() signature is unchanged."""
        import inspect

        sig = inspect.signature(mata.register_model)
        params = list(sig.parameters.keys())

        assert "task" in params
        assert "alias" in params
        assert "source" in params
        assert "config" in params

        # Check kwargs
        assert sig.parameters["config"].kind == inspect.Parameter.VAR_KEYWORD

    def test_mata_infer_signature(self):
        """Test infer() signature is unchanged."""
        import inspect

        sig = inspect.signature(mata.infer)
        params = list(sig.parameters.keys())

        assert "image" in params
        assert "graph" in params
        assert "providers" in params
        assert "scheduler" in params
        assert "device" in params
        assert "kwargs" in params


class TestPublicAPIExports:
    """Test that all expected exports are available."""

    def test_api_functions_exported(self):
        """Test all API functions are exported from mata module."""
        assert hasattr(mata, "load")
        assert hasattr(mata, "run")
        assert hasattr(mata, "infer")
        assert hasattr(mata, "list_models")
        assert hasattr(mata, "get_model_info")
        assert hasattr(mata, "register_model")

    def test_result_types_exported(self):
        """Test all result types are exported from mata module."""
        # Current result types
        assert hasattr(mata, "DetectResult")
        assert hasattr(mata, "SegmentResult")
        assert hasattr(mata, "ClassifyResult")
        assert hasattr(mata, "DepthResult")
        assert hasattr(mata, "TrackResult")
        assert hasattr(mata, "VisionResult")

        # Component types
        assert hasattr(mata, "Detection")
        assert hasattr(mata, "SegmentMask")
        assert hasattr(mata, "Track")
        assert hasattr(mata, "Instance")
        assert hasattr(mata, "Entity")

    def test_exceptions_exported(self):
        """Test all exception types are exported from mata module."""
        assert hasattr(mata, "MATAError")
        assert hasattr(mata, "TaskNotSupportedError")
        assert hasattr(mata, "InvalidInputError")
        assert hasattr(mata, "ModelLoadError")

    def test_version_exported(self):
        """Test version is exported."""
        assert hasattr(mata, "__version__")
        assert isinstance(mata.__version__, str)


class TestLoadAPICompat:
    """Test mata.load() API backward compatibility."""

    @patch("mata.core.model_loader.UniversalLoader._load_from_huggingface")
    def test_load_huggingface_pattern(self, mock_load):
        """Test loading HuggingFace models unchanged."""
        mock_adapter = Mock()
        mock_load.return_value = mock_adapter

        # Basic load
        result = mata.load("detect", "facebook/detr-resnet-50")
        assert result is mock_adapter
        mock_load.assert_called_once()

        # With kwargs
        mock_load.reset_mock()
        mata.load("detect", "facebook/detr-resnet-50", threshold=0.6, device="cpu")
        mock_load.assert_called_once()

    @patch("mata.core.model_loader.UniversalLoader._load_from_file")
    def test_load_file_pattern(self, mock_load):
        """Test loading local files unchanged."""
        mock_adapter = Mock()
        mock_load.return_value = mock_adapter

        result = mata.load("detect", "model.onnx")
        assert result is mock_adapter
        mock_load.assert_called_once()

    def test_load_unsupported_task_raises(self):
        """Test that unsupported task raises appropriate error."""
        # Note: UniversalLoader doesn't validate task names upfront,
        # so invalid tasks result in ModelNotFoundError, not TaskNotSupportedError
        with pytest.raises(ModelNotFoundError):
            mata.load("unsupported_task", "some_model")

    def test_load_no_model_raises_helpful_error(self):
        """Test that load() with no model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError, match="No default model configured"):
            mata.load("detect")

    def test_load_model_type_parameter(self):
        """Test model_type parameter still works."""
        with patch("mata.core.model_loader.UniversalLoader.load") as mock_load:
            mock_load.return_value = Mock()

            # Test string values (deprecated but supported)
            mata.load("detect", "model.onnx", model_type="onnx")
            mock_load.assert_called_with(task="detect", source="model.onnx", model_type="onnx")

            # Test enum values
            mock_load.reset_mock()
            mata.load("detect", "model.onnx", model_type=ModelType.ONNX)
            mock_load.assert_called_with(task="detect", source="model.onnx", model_type=ModelType.ONNX)


class TestRunAPICompat:
    """Test mata.run() API backward compatibility."""

    @patch("mata.core.model_loader.UniversalLoader.load")
    def test_run_basic_pattern(self, mock_universal_load):
        """Test basic run() pattern unchanged."""
        # Mock adapter
        mock_adapter = Mock()
        mock_result = DetectResult(
            detections=[Detection(bbox=(10, 20, 100, 200), score=0.95, label=1, label_name="cat")],
            meta={"model": "mock"},
        )
        mock_adapter.predict.return_value = mock_result
        mock_universal_load.return_value = mock_adapter

        # Basic run
        result = mata.run("detect", "image.jpg")

        assert isinstance(result, DetectResult)
        assert len(result.detections) == 1
        assert result.detections[0].score == 0.95

        mock_universal_load.assert_called_once_with(task="detect", source=None, model_type=None)
        mock_adapter.predict.assert_called_once()

    @patch("mata.core.model_loader.UniversalLoader.load")
    def test_run_with_model_and_kwargs(self, mock_universal_load):
        """Test run() with model and kwargs unchanged."""
        mock_adapter = Mock()
        mock_universal_load.return_value = mock_adapter

        mata.run("detect", "image.jpg", model="model.onnx", threshold=0.7, device="gpu")

        # Should pass model and model_type to universal loader
        mock_universal_load.assert_called_once_with(
            task="detect", source="model.onnx", model_type=None, threshold=0.7, device="gpu"
        )

        # Should pass kwargs to predict()
        mock_adapter.predict.assert_called_once_with("image.jpg", threshold=0.7, device="gpu")

    def test_run_track_raises_error(self):
        """Test that run() with track task raises helpful ValueError."""
        with pytest.raises(ValueError, match="stateful"):
            mata.run("track", "image.jpg")

    def test_run_unsupported_task_raises(self):
        """Test that run() with unsupported task raises appropriate error."""
        # Note: Like load(), run() doesn't validate task names upfront
        with pytest.raises(ModelNotFoundError):
            mata.run("invalid_task", "image.jpg")

    @patch("mata.core.model_loader.UniversalLoader.load")
    def test_run_returns_correct_types(self, mock_universal_load):
        """Test that run() returns expected result types for each task."""
        # Test detect
        mock_adapter = Mock()
        mock_detect_result = DetectResult(detections=[], meta={})
        mock_adapter.predict.return_value = mock_detect_result
        mock_universal_load.return_value = mock_adapter

        result = mata.run("detect", "image.jpg")
        assert isinstance(result, DetectResult)

        # Test classify
        mock_classify_result = ClassifyResult(predictions=[], meta={})
        mock_adapter.predict.return_value = mock_classify_result

        result = mata.run("classify", "image.jpg")
        assert isinstance(result, ClassifyResult)

        # Test depth
        mock_depth_result = DepthResult(depth=[[1, 2], [3, 4]], meta={})
        mock_adapter.predict.return_value = mock_depth_result

        result = mata.run("depth", "image.jpg")
        assert isinstance(result, DepthResult)


class TestResultTypeCompat:
    """Test result type backward compatibility."""

    def test_detect_result_attributes(self):
        """Test DetectResult has expected attributes and methods."""
        det = Detection(bbox=(0, 0, 10, 10), score=0.9, label=0, label_name="test")
        result = DetectResult(detections=[det], meta={"test": "data"})

        # Attributes
        assert hasattr(result, "detections")
        assert hasattr(result, "meta")
        assert len(result.detections) == 1
        assert result.meta["test"] == "data"

        # Methods
        assert hasattr(result, "to_dict")
        assert hasattr(result, "to_json")
        assert hasattr(result, "from_dict")
        assert hasattr(result, "from_json")
        assert hasattr(result, "save")

        # Test serialization roundtrip
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "detections" in data

        json_str = result.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "detections" in parsed

    def test_vision_result_attributes(self):
        """Test VisionResult has expected attributes and methods."""
        instance = Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, label_name="test")
        result = VisionResult(instances=[instance], meta={"test": "data"})

        # Attributes
        assert hasattr(result, "instances")
        assert hasattr(result, "meta")
        assert hasattr(result, "text")
        assert hasattr(result, "prompt")
        assert hasattr(result, "entities")

        # Properties
        assert hasattr(result, "detections")  # Property that filters instances
        assert hasattr(result, "masks")  # Property that filters instances

        # Methods
        assert hasattr(result, "to_dict")
        assert hasattr(result, "to_json")
        assert hasattr(result, "from_dict")
        assert hasattr(result, "from_json")
        assert hasattr(result, "save")
        assert hasattr(result, "filter_by_score")
        assert hasattr(result, "get_instances")
        assert hasattr(result, "get_stuff")

        # Test property access
        detections = result.detections
        assert isinstance(detections, list)
        assert len(detections) == 1  # Has bbox

    def test_classify_result_attributes(self):
        """Test ClassifyResult has expected attributes and methods."""
        from mata.core.types import Classification

        classif = Classification(label=0, score=0.9, label_name="cat")
        result = ClassifyResult(predictions=[classif], meta={})

        # Attributes
        assert hasattr(result, "predictions")
        assert hasattr(result, "meta")

        # Methods
        assert hasattr(result, "to_dict")
        assert hasattr(result, "to_json")
        assert hasattr(result, "get_top1")
        assert hasattr(result, "filter_by_score")

        # Test top1 property
        top1 = result.get_top1()
        assert top1 is not None
        assert top1.score == 0.9

    def test_depth_result_attributes(self):
        """Test DepthResult has expected attributes and methods."""
        import numpy as np

        depth_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = DepthResult(depth=depth_data, meta={})

        # Attributes
        assert hasattr(result, "depth")
        assert hasattr(result, "normalized")
        assert hasattr(result, "meta")

        # Methods
        assert hasattr(result, "to_dict")
        assert hasattr(result, "to_json")
        assert hasattr(result, "save")

        # Test depth data
        assert isinstance(result.depth, np.ndarray)
        assert result.depth.shape == (2, 2)


class TestErrorHandlingCompat:
    """Test error handling backward compatibility."""

    def test_exception_hierarchy(self):
        """Test exception hierarchy is preserved."""
        # All MATA exceptions inherit from MATAError
        assert issubclass(TaskNotSupportedError, MATAError)
        assert issubclass(InvalidInputError, MATAError)
        assert issubclass(ModelLoadError, MATAError)

        # All MATA exceptions inherit from Exception
        assert issubclass(MATAError, Exception)

    def test_task_not_supported_error(self):
        """Test TaskNotSupportedError constructor and message."""
        exc = TaskNotSupportedError("invalid_task", ["detect", "classify"])
        assert "invalid_task" in str(exc)
        assert "detect" in str(exc)
        assert "classify" in str(exc)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError constructor and message."""
        exc = ModelNotFoundError("Model 'test' not found")
        assert "test" in str(exc)
        assert "not found" in str(exc)

    def test_invalid_input_error(self):
        """Test InvalidInputError constructor and attributes."""
        exc = InvalidInputError("Bad input", input_value="test")
        assert "Bad input" in str(exc)
        # input_value attribute is available but not part of message


class TestConfigSystemCompat:
    """Test configuration system backward compatibility."""

    def test_register_model_runtime(self):
        """Test register_model() works at runtime."""
        # This should not raise an error
        mata.register_model("detect", "test-alias", "test-model.onnx", threshold=0.5)

        # Verify it was registered (won't test loading since no real model)
        # Just check the call succeeds
        assert True  # If we get here, registration succeeded

    def test_list_models_api_available(self):
        """Test list_models() API is available."""
        # Should not raise NameError even if huggingface_hub not available
        try:
            result = mata.list_models("detect", limit=1)
            # If successful, should return list
            assert isinstance(result, list)
        except ImportError:
            # Expected if huggingface_hub not installed
            assert True
        except Exception as e:
            # Network errors, etc. are OK for this test
            assert "huggingface_hub" in str(e) or "network" in str(e).lower()

    def test_get_model_info_api_available(self):
        """Test get_model_info() API is available."""
        try:
            result = mata.get_model_info("facebook/detr-resnet-50")
            # If successful, should return dict
            assert isinstance(result, dict)
            assert "id" in result
        except ImportError:
            # Expected if huggingface_hub not installed
            assert True
        except Exception as e:
            # Network errors, model not found, etc. are OK for this test
            assert "huggingface_hub" in str(e) or any(
                word in str(e).lower() for word in ["network", "not found", "request"]
            )


class TestNewInferAPICompat:
    """Test that new infer() API doesn't break old patterns."""

    def test_infer_exists_but_not_required(self):
        """Test that infer() exists but old APIs still work."""
        # infer should exist
        assert hasattr(mata, "infer")
        assert callable(mata.infer)

        # But load() and run() should still work independently
        with patch("mata.core.model_loader.UniversalLoader._load_from_file") as mock_load:
            mock_adapter = Mock()
            mock_load.return_value = mock_adapter

            # Old pattern still works
            adapter = mata.load("detect", "model.onnx")
            assert adapter is mock_adapter

    @patch("mata.core.model_loader.UniversalLoader.load")
    def test_old_and_new_apis_coexist(self, mock_universal_load):
        """Test that old mata.run() and new mata.infer() can coexist."""
        mock_adapter = Mock()
        mock_result = DetectResult(detections=[], meta={})
        mock_adapter.predict.return_value = mock_result
        mock_universal_load.return_value = mock_adapter

        # Old API still works
        result1 = mata.run("detect", "test.jpg", model="test.onnx")
        assert isinstance(result1, DetectResult)

        # New API exists (but we won't test full functionality here)
        assert hasattr(mata, "infer")


class TestRegressionTests:
    """Regression tests for specific compatibility concerns."""

    def test_load_return_type_unchanged(self):
        """Test load() returns adapter objects with expected interface."""
        with patch("mata.core.model_loader.UniversalLoader.load") as mock_load:
            mock_adapter = Mock()
            mock_adapter.predict = Mock()
            mock_adapter.info = Mock(return_value={"name": "test"})
            mock_load.return_value = mock_adapter

            adapter = mata.load("detect", "test-model")

            # Should have predict method
            assert hasattr(adapter, "predict")
            assert callable(adapter.predict)

            # Should have info method (common across adapters)
            if hasattr(adapter, "info"):
                assert callable(adapter.info)

    @patch("mata.core.model_loader.UniversalLoader.load")
    def test_run_kwargs_passthrough(self, mock_universal_load):
        """Test that run() properly passes through kwargs."""
        mock_adapter = Mock()
        mock_universal_load.return_value = mock_adapter

        # Test kwargs passthrough
        mata.run("detect", "image.jpg", model="test.onnx", threshold=0.7, device="cpu")

        # Verify universal loader was called correctly
        mock_universal_load.assert_called_once()
        call_kwargs = mock_universal_load.call_args[1]
        assert "threshold" in call_kwargs
        assert call_kwargs["threshold"] == 0.7

        # Verify predict was called
        mock_adapter.predict.assert_called_once()

    def test_task_validation_unchanged(self):
        """Test that task validation behavior is unchanged."""
        # Valid tasks should not raise TaskNotSupportedError
        # (may raise ModelNotFoundError due to missing defaults)
        valid_tasks = ["detect", "classify", "segment", "depth", "vlm"]
        for task in valid_tasks:
            try:
                # This will raise ModelNotFoundError (no default models)
                mata.load(task, "fake-model")
            except ModelNotFoundError:
                # Expected - no model found
                pass
            except TaskNotSupportedError:
                pytest.fail(f"Task '{task}' should be supported")
            except Exception:
                # Other errors are expected without real models
                pass

        # Invalid tasks should also raise ModelNotFoundError
        # (UniversalLoader doesn't validate task names upfront)
        invalid_tasks = ["invalid", "unknown", "fake_task"]
        for task in invalid_tasks:
            try:
                mata.load(task, "fake-model")
                pytest.fail(f"Expected ModelNotFoundError for invalid task '{task}'")
            except ModelNotFoundError:
                # Expected behavior
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception for task '{task}': {e}")


if __name__ == "__main__":
    # Running this file directly runs the backward compatibility tests
    pytest.main([__file__, "-v"])
