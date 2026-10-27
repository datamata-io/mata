"""End-to-end test against the committed tiny YOLO-layout ONNX fixture."""

import json
from pathlib import Path

import pytest
from PIL import Image

from mata.adapters.onnx_adapter import ONNXDetectAdapter

FIXTURES = Path(__file__).parent / "fixtures"


def test_tiny_yolo_fixture_matches_golden():
    pytest.importorskip("onnxruntime")
    model = FIXTURES / "tiny_yolo.onnx"
    expected = json.loads((FIXTURES / "tiny_yolo_expected.json").read_text())

    adapter = ONNXDetectAdapter(str(model), device="cpu", threshold=0.3, id2label={0: "a", 1: "b"})
    result = adapter.predict(Image.new("RGB", (64, 64), (10, 20, 30)))

    assert len(result.instances) == 1
    detection = expected["detections"][0]
    assert result.instances[0].bbox == pytest.approx(detection["bbox"], abs=1e-3)
    assert result.instances[0].score == pytest.approx(detection["score"], abs=1e-5)
    assert result.instances[0].label == detection["label"]
