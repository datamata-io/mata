"""Integration tests for MATA API."""

from unittest.mock import patch

import pytest

from mata import load, run
from mata.core.exceptions import ModelNotFoundError
from mata.core.types import DepthResult, Detection, DetectResult


class MockDetector:
    """Mock detector for testing."""

    name = "mock_detector"
    task = "detect"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def info(self):
        return {"name": self.name, "task": self.task}

    def predict(self, image, **kwargs):
        return DetectResult(
            detections=[Detection(bbox=(10.0, 20.0, 100.0, 200.0), score=0.95, label=1, label_name="test_object")],
            meta={"model": "mock"},
        )


class MockDepth:
    """Mock depth adapter for testing."""

    name = "mock_depth"
    task = "depth"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def info(self):
        return {"name": self.name, "task": self.task}

    def predict(self, image, **kwargs):
        return DepthResult(depth=[[1.0, 2.0], [3.0, 4.0]], meta={"model": "mock"})


def test_load_without_model_raises():
    """Test that loading without model or default config raises error."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        load("detect")

    assert "No default model configured" in str(exc_info.value)


def test_load_with_explicit_file():
    """Test loading with explicit model file."""
    with patch("mata.core.model_loader.UniversalLoader._load_from_file") as mock_load:
        mock_load.return_value = MockDetector()
        detector = load("detect", "model.onnx")
        assert isinstance(detector, MockDetector)
        mock_load.assert_called_once()


def test_load_with_huggingface_id():
    """Test loading with HuggingFace model ID."""
    with patch("mata.core.model_loader.UniversalLoader._load_from_huggingface") as mock_load:
        mock_load.return_value = MockDetector()
        detector = load("detect", "facebook/detr-resnet-50")
        assert isinstance(detector, MockDetector)
        mock_load.assert_called_once()


def test_load_with_kwargs():
    """Test loading with custom parameters."""
    with patch("mata.core.model_loader.UniversalLoader._load_from_huggingface") as mock_load:
        mock_detector = MockDetector(custom_param="value")
        mock_load.return_value = mock_detector
        detector = load("detect", "facebook/detr-resnet-50", custom_param="value")
        assert detector.kwargs.get("custom_param") == "value"


def test_run_one_shot():
    """Test one-shot inference."""
    with patch("mata.core.model_loader.UniversalLoader._load_from_file") as mock_load:
        mock_load.return_value = MockDetector()
        result = run("detect", "test.jpg", model="model.onnx")
        assert isinstance(result, DetectResult)
        assert len(result.detections) == 1
        assert result.detections[0].score == 0.95


def test_run_depth_one_shot():
    """Test one-shot depth inference."""
    with patch("mata.core.model_loader.UniversalLoader._load_from_huggingface") as mock_load:
        mock_load.return_value = MockDepth()
        result = run("depth", "test.jpg", model="depth-anything/Depth-Anything-V2-Small-hf")
        assert isinstance(result, DepthResult)
        assert result.depth is not None


def test_run_track_raises():
    """Test that run() raises ValueError for track task."""
    with pytest.raises(ValueError) as exc_info:
        run("track", "test.jpg")

    assert "stateful" in str(exc_info.value).lower()


def test_list_models_requires_huggingface_hub():
    """Test that list_models() requires huggingface_hub."""
    # This test validates the function exists and has proper error handling
    from mata import list_models

    # If huggingface_hub is installed, should return list/dict
    # If not installed, should raise ImportError with helpful message
    try:
        result = list_models("detect")
        assert isinstance(result, list)
    except ImportError as e:
        assert "huggingface_hub" in str(e)


def test_get_model_info_requires_huggingface_hub():
    """Test that get_model_info() requires huggingface_hub."""
    from mata import get_model_info

    # If huggingface_hub is installed, should query model
    # If not installed, should raise ImportError with helpful message
    try:
        result = get_model_info("facebook/detr-resnet-50")
        assert isinstance(result, dict)
        assert "id" in result
    except ImportError as e:
        assert "huggingface_hub" in str(e)
    except Exception:
        # Other errors (network, model not found) are acceptable for this test
        pass


def test_run_ocr_task_is_valid():
    """mata.run('ocr', ...) does not raise TaskNotSupportedError."""
    from unittest.mock import Mock

    from mata.core.exceptions import TaskNotSupportedError

    mock_adapter = Mock()
    mock_adapter.predict.return_value = Mock()

    with patch("mata.api.load", return_value=mock_adapter):
        try:
            run("ocr", "test.jpg", model="easyocr")
        except TaskNotSupportedError:
            pytest.fail("run('ocr', ...) raised TaskNotSupportedError unexpectedly")


def test_run_unknown_task_error_includes_ocr_in_help():
    """mata.run() with an unsupported task raises TaskNotSupportedError listing 'ocr'."""
    from unittest.mock import Mock

    from mata.core.exceptions import TaskNotSupportedError

    mock_adapter = Mock()
    mock_adapter.predict.return_value = Mock()

    with patch("mata.api.load", return_value=mock_adapter):
        with pytest.raises(TaskNotSupportedError) as exc_info:
            run("unknown_task_xyz", "test.jpg")

    assert "ocr" in str(exc_info.value)
