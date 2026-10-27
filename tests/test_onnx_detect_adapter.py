"""Tests for the ONNX detection adapter (YOLO-family path)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from mata.adapters.onnx_adapter import ONNXDetectAdapter
from mata.core.exceptions import ConfigurationError, ModelLoadError

NAMES = {0: "person", 1: "vehicle"}


def _session(output: np.ndarray, *, providers=("CPUExecutionProvider",), capture=None):
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]
    session.get_outputs.return_value = [SimpleNamespace(name="output0", shape=list(output.shape))]
    session.get_providers.return_value = list(providers)

    def run(_names, feed):
        if capture is not None:
            capture["tensor"] = feed["images"]
        return [output]

    session.run.side_effect = run
    return session


@pytest.fixture
def make_adapter(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")

    def factory(session, **kwargs):
        available = kwargs.pop("available", ["CPUExecutionProvider"])
        ort = MagicMock()
        ort.InferenceSession.return_value = session
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC = 1
        ort.SessionOptions.return_value = MagicMock()
        ort.get_available_providers.return_value = list(available)
        monkeypatch.setattr("mata.adapters.onnx_base._ensure_onnxruntime", lambda: ort)
        factory.ort = ort
        return ONNXDetectAdapter(str(model), id2label=NAMES, **kwargs)

    return factory


def test_letterbox_tensor_and_padding(make_adapter):
    capture = {}
    out = np.zeros((1, 6, 10), np.float32)
    adapter = make_adapter(_session(out, capture=capture))
    adapter.predict(Image.new("RGB", (1280, 720), (255, 0, 0)))
    tensor = capture["tensor"]
    assert tensor.shape == (1, 3, 640, 640)
    assert np.allclose(tensor[0, :, 0, 0], 114 / 255)
    assert np.allclose(tensor[0, :, 140, 320], (1.0, 0.0, 0.0))


def test_letterbox_inverse_mapping(make_adapter):
    out = np.zeros((1, 6, 10), np.float32)
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = 320, 320, 100, 100
    out[0, 4, 0] = 0.9
    adapter = make_adapter(_session(out))
    result = adapter.predict(Image.new("RGB", (1280, 720)))
    assert result.instances[0].bbox == pytest.approx((540, 260, 740, 460), abs=1e-4)


def test_max_det_caps_detections(make_adapter):
    out = np.zeros((1, 6, 350), np.float32)
    out[0, 0, :] = 2 + np.arange(350) * 1.8
    out[0, 1, :] = 320
    out[0, 2, :] = 1.5
    out[0, 3, :] = 1.5
    out[0, 4, :] = 0.9
    adapter = make_adapter(_session(out), max_det=300)
    result = adapter.predict(Image.new("RGB", (640, 640)))
    assert len(result.instances) == 300


def test_bgr_input_matches_rgb(make_adapter):
    out = np.zeros((1, 6, 10), np.float32)
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[..., 0] = 255
    bgr = rgb[..., ::-1].copy()
    cap_rgb, cap_bgr = {}, {}
    make_adapter(_session(out, capture=cap_rgb)).predict(rgb)
    make_adapter(_session(out, capture=cap_bgr), channel_order="bgr").predict(bgr)
    assert np.array_equal(cap_rgb["tensor"], cap_bgr["tensor"])


def test_cuda_request_without_provider_raises(make_adapter):
    out = np.zeros((1, 6, 10), np.float32)
    with pytest.raises(ConfigurationError):
        make_adapter(_session(out), device="cuda", available=["CPUExecutionProvider"])


def test_session_fallback_to_cpu_raises(make_adapter):
    out = np.zeros((1, 6, 10), np.float32)
    session = _session(out, providers=("CPUExecutionProvider",))
    with pytest.raises(ModelLoadError):
        make_adapter(
            session,
            device="cuda",
            available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


def test_session_options_reach_ort(make_adapter):
    out = np.zeros((1, 6, 10), np.float32)
    adapter = make_adapter(_session(out), session_options={"intra_op_num_threads": 6})
    options = adapter._make_session_options()
    assert options.intra_op_num_threads == 6


def test_predict_logs_at_debug_only(make_adapter):
    import logging

    from mata.core.logging import get_logger

    out = np.zeros((1, 6, 10), np.float32)
    adapter = make_adapter(_session(out))

    logger = get_logger()  # MATA uses one global logger with propagate=False
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        adapter.predict(Image.new("RGB", (640, 640)))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
