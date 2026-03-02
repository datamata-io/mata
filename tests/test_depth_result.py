"""Tests for DepthResult type."""

import numpy as np

from mata.core.types import DepthResult


def test_depth_result_serialization_roundtrip():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalized = (depth - depth.min()) / (depth.max() - depth.min())
    result = DepthResult(depth=depth, normalized=normalized, meta={"model": "test"})

    data = result.to_dict()
    restored = DepthResult.from_dict(data)

    assert np.allclose(restored.depth, depth)
    assert restored.normalized is not None
    assert np.allclose(restored.normalized, normalized)
    assert restored.meta["model"] == "test"


def test_depth_result_save_json(tmp_path):
    depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    result = DepthResult(depth=depth, meta={"model": "test"})

    output_path = tmp_path / "depth.json"
    result.save(output_path)

    assert output_path.exists()
