"""Parity tests for the ndarray fast path in YOLO preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mata.adapters.onnx_adapter import ONNXDetectAdapter
from mata.core.exceptions import InvalidInputError

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_yolo.onnx"


def _adapter(channel_order: str = "rgb") -> ONNXDetectAdapter:
    return ONNXDetectAdapter(str(FIXTURE), device="cpu", max_det=None, channel_order=channel_order)


def test_array_and_pil_tensors_are_identical():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    adapter = _adapter()
    from_array, tf_array = adapter._preprocess_yolo_letterbox(array)
    from_pil, tf_pil = adapter._preprocess_yolo_letterbox(Image.fromarray(array))
    assert np.array_equal(from_array, from_pil)
    assert tf_array == tf_pil


def test_array_predict_matches_pil_predict():
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    adapter = _adapter()
    from_array = adapter.predict(array)
    from_pil = adapter.predict(Image.fromarray(array))
    assert len(from_array.instances) == len(from_pil.instances)
    for a, b in zip(from_array.instances, from_pil.instances):
        assert a.bbox == pytest.approx(b.bbox)
        assert a.score == pytest.approx(b.score)
        assert a.label == b.label


def test_bgr_array_matches_rgb_array():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()
    from_rgb = _adapter("rgb").predict(rgb)
    from_bgr = _adapter("bgr").predict(bgr)
    assert len(from_rgb.instances) == len(from_bgr.instances)
    for a, b in zip(from_rgb.instances, from_bgr.instances):
        assert a.bbox == pytest.approx(b.bbox)


def test_ndarray_predict_skips_the_pil_loader(monkeypatch):
    adapter = _adapter()

    def fail(_image):
        raise AssertionError("ndarray fast path should not call _load_image")

    monkeypatch.setattr(adapter, "_load_image", fail)
    result = adapter.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(result.instances) == 1


def test_invalid_array_shape_raises():
    with pytest.raises(InvalidInputError):
        _adapter().predict(np.zeros((10, 10, 5), dtype=np.uint8))
