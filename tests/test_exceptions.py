"""Tests for exception handling."""

import pytest

from mata.core.exceptions import (
    InvalidInputError,
    MATAError,
    ModelLoadError,
    TaskNotSupportedError,
)


def test_mata_error():
    """Test base MATAError."""
    with pytest.raises(MATAError) as exc_info:
        raise MATAError("Test error")

    assert "Test error" in str(exc_info.value)


def test_task_not_supported_error():
    """Test TaskNotSupportedError formatting."""
    error = TaskNotSupportedError(task="unsupported", supported=["detect", "segment"])

    assert error.task == "unsupported"
    assert "unsupported" in str(error)
    assert "detect" in str(error)


def test_invalid_input_error():
    """Test InvalidInputError."""
    error = InvalidInputError("File not found", input_value="missing.jpg")

    assert error.input_value == "missing.jpg"
    assert "File not found" in str(error)


def test_model_load_error():
    """Test ModelLoadError formatting."""
    error = ModelLoadError(model_id="test/model", reason="Network timeout")

    assert error.model_id == "test/model"
    assert error.reason == "Network timeout"
    assert "test/model" in str(error)
    assert "Network timeout" in str(error)
    assert "network connection" in str(error).lower()
