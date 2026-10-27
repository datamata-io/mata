"""Tests for ONNX detection batch inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mata.adapters.onnx_adapter import ONNXDetectAdapter

FIXTURES = Path(__file__).parent / "fixtures"
DYNAMIC = FIXTURES / "tiny_yolo_batch.onnx"
STATIC = FIXTURES / "tiny_yolo.onnx"


def _adapter(path: Path = DYNAMIC) -> ONNXDetectAdapter:
    return ONNXDetectAdapter(str(path), device="cpu", max_det=None)


def _images(count: int = 2) -> list[np.ndarray]:
    return [np.full((64, 64, 3), value, dtype=np.uint8) for value in range(count)]


def test_dynamic_model_supports_batch() -> None:
    assert _adapter().supports_batch is True


def test_static_model_reports_no_batch_support() -> None:
    assert _adapter(STATIC).supports_batch is False


def test_batch_matches_single_image_results() -> None:
    adapter = _adapter()
    images = _images(2)
    batched = adapter.predict_batch(images, batch_size=2)
    singles = [adapter.predict(image) for image in images]

    assert len(batched) == 2
    for batch_result, single_result in zip(batched, singles):
        assert len(batch_result.instances) == len(single_result.instances)
        for batch_inst, single_inst in zip(batch_result.instances, single_result.instances):
            assert batch_inst.bbox == pytest.approx(single_inst.bbox)
            assert batch_inst.score == pytest.approx(single_inst.score)
            assert batch_inst.label == single_inst.label


def test_batch_size_chunks_session_calls() -> None:
    adapter = _adapter()

    class CountingSession:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return self.inner.run(*args, **kwargs)

    counting = CountingSession(adapter.session)
    adapter.session = counting  # type: ignore[assignment]
    results = adapter.predict_batch(_images(5), batch_size=2)
    assert len(results) == 5
    assert counting.calls == 3  # 2 + 2 + 1


def test_static_model_falls_back_to_sequential() -> None:
    adapter = _adapter(STATIC)
    results = adapter.predict_batch(_images(2), batch_size=2)
    assert len(results) == 2
    assert all(len(result.instances) == 1 for result in results)


def test_empty_input_returns_empty_list() -> None:
    assert _adapter().predict_batch([]) == []


def test_invalid_batch_size_raises() -> None:
    with pytest.raises(ValueError):
        _adapter().predict_batch(_images(2), batch_size=0)
