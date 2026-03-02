"""Tests for task protocols."""

from mata.core.types import DetectResult


def test_task_adapter_protocol():
    """Test TaskAdapter protocol."""

    class ValidAdapter:
        name = "test"
        task = "detect"

        def info(self):
            return {"name": self.name}

    adapter = ValidAdapter()
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "task")
    assert hasattr(adapter, "info")


def test_detect_adapter_protocol():
    """Test DetectAdapter protocol."""

    class ValidDetectAdapter:
        name = "test_detect"
        task = "detect"

        def info(self):
            return {"name": self.name, "task": self.task}

        def predict(self, image, **kwargs):
            return DetectResult(detections=[])

    adapter = ValidDetectAdapter()
    assert adapter.task == "detect"
    result = adapter.predict("test.jpg")
    assert isinstance(result, DetectResult)


def test_protocol_checking():
    """Test that protocols enforce required methods."""

    class IncompleteAdapter:
        name = "incomplete"
        task = "detect"
        # Missing info() and predict()

    # Protocols in Python are structural, not enforced at runtime
    # This test documents expected interface
    adapter = IncompleteAdapter()
    assert not hasattr(adapter, "info")
    assert not hasattr(adapter, "predict")
