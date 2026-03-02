"""Tests for Graph.run() convenience method.

Tests that Graph.run() correctly delegates to mata.infer() and supports
all expected calling patterns: fluent chaining, separate build-and-run,
scheduler forwarding, device forwarding, and error propagation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.result import MultiResult
from mata.core.graph import Graph, ParallelScheduler
from mata.core.types import Instance, VisionResult
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse
from mata.nodes.topk import TopK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 200, height: int = 100) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(128, 128, 128))


def _make_numpy_image(width: int = 200, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_temp_image(suffix: str = ".png") -> str:
    img = _make_pil_image()
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


class _MockDetectAdapter:
    """Mock detection adapter returning fixed detections."""

    model_id = "mock/detector"

    def predict(self, image, **kwargs):
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.82, label=1, label_name="dog"),
        ]
        return VisionResult(instances=instances, meta={"model": "mock"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pil_image():
    return _make_pil_image()


@pytest.fixture
def numpy_image():
    return _make_numpy_image()


@pytest.fixture
def temp_image_path():
    path = _make_temp_image()
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_detector():
    return _MockDetectAdapter()


# ===========================================================================
# Test: Graph.run() Delegation
# ===========================================================================


class TestGraphRunDelegation:
    """Verify Graph.run() delegates to mata.infer() correctly."""

    def test_run_delegates_to_infer(self, pil_image, mock_detector):
        """Graph.run() should call mata.api.infer with correct args."""
        graph = Graph("test").then(Detect(using="detector", out="dets"))

        with patch("mata.api.infer") as mock_infer:
            mock_infer.return_value = MagicMock(spec=MultiResult)
            graph.run(pil_image, providers={"detector": mock_detector})

            mock_infer.assert_called_once_with(
                image=pil_image,
                graph=graph,
                providers={"detector": mock_detector},
                scheduler=None,
                device="auto",
            )

    def test_run_forwards_scheduler(self, pil_image, mock_detector):
        """Scheduler kwarg should be forwarded to infer()."""
        graph = Graph("test").then(Detect(using="detector", out="dets"))
        scheduler = ParallelScheduler()

        with patch("mata.api.infer") as mock_infer:
            mock_infer.return_value = MagicMock(spec=MultiResult)
            graph.run(pil_image, providers={"detector": mock_detector}, scheduler=scheduler)

            _, kwargs = mock_infer.call_args
            assert kwargs["scheduler"] is scheduler

    def test_run_forwards_device(self, pil_image, mock_detector):
        """Device kwarg should be forwarded to infer()."""
        graph = Graph("test").then(Detect(using="detector", out="dets"))

        with patch("mata.api.infer") as mock_infer:
            mock_infer.return_value = MagicMock(spec=MultiResult)
            graph.run(pil_image, providers={"detector": mock_detector}, device="cpu")

            _, kwargs = mock_infer.call_args
            assert kwargs["device"] == "cpu"

    def test_run_returns_infer_result(self, pil_image, mock_detector):
        """Graph.run() should return whatever infer() returns."""
        graph = Graph("test").then(Detect(using="detector", out="dets"))

        sentinel = MagicMock(spec=MultiResult)
        with patch("mata.api.infer", return_value=sentinel):
            result = graph.run(pil_image, providers={"detector": mock_detector})
            assert result is sentinel


# ===========================================================================
# Test: Graph.run() End-to-End
# ===========================================================================


class TestGraphRunEndToEnd:
    """Integration tests using real mata.infer() execution."""

    def test_run_with_pil_image(self, pil_image, mock_detector):
        """Run graph with PIL image."""
        graph = Graph("e2e").then(Detect(using="detector", out="dets"))
        result = graph.run(pil_image, providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)

    def test_run_with_numpy_image(self, numpy_image, mock_detector):
        """Run graph with numpy array."""
        graph = Graph("e2e").then(Detect(using="detector", out="dets"))
        result = graph.run(numpy_image, providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)

    def test_run_with_file_path(self, temp_image_path, mock_detector):
        """Run graph with string file path."""
        graph = Graph("e2e").then(Detect(using="detector", out="dets"))
        result = graph.run(temp_image_path, providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)

    def test_run_with_pathlib_path(self, temp_image_path, mock_detector):
        """Run graph with pathlib.Path."""
        graph = Graph("e2e").then(Detect(using="detector", out="dets"))
        result = graph.run(Path(temp_image_path), providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)

    def test_run_multi_node_pipeline(self, pil_image, mock_detector):
        """Run a multi-node sequential graph."""
        graph = (
            Graph("pipeline")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.3, out="filtered"))
            .then(Fuse(dets="filtered", out="final"))
        )
        result = graph.run(pil_image, providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)

    def test_run_with_topk(self, pil_image, mock_detector):
        """Run the exact example from the proposal: top5 pipeline."""
        graph = (
            Graph("top5_detections")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.3, out="filtered"))
            .then(TopK(k=5, src="filtered", out="top5"))
            .then(Fuse(dets="top5", out="final"))
        )
        result = graph.run(pil_image, providers={"detector": mock_detector})
        assert isinstance(result, MultiResult)


# ===========================================================================
# Test: Fluent Chained Execution
# ===========================================================================


class TestFluentChainedExecution:
    """Test the single-expression fluent pattern."""

    def test_fluent_chain_returns_multiresult(self, pil_image, mock_detector):
        """Single-expression build-and-run should return MultiResult."""
        result = (
            Graph("fluent")
            .then(Detect(using="detector", out="dets"))
            .then(Fuse(dets="dets", out="final"))
            .run(pil_image, providers={"detector": mock_detector})
        )
        assert isinstance(result, MultiResult)

    def test_fluent_chain_with_filter(self, pil_image, mock_detector):
        """Fluent chain with filter step."""
        result = (
            Graph("fluent_filter")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.5, out="filtered"))
            .then(Fuse(dets="filtered", out="final"))
            .run(pil_image, providers={"detector": mock_detector})
        )
        assert isinstance(result, MultiResult)


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestGraphRunErrors:
    """Test that errors from infer() propagate through run()."""

    def test_run_invalid_image_type_raises(self, mock_detector):
        """Invalid image type should raise ValueError."""
        graph = Graph("err").then(Detect(using="detector", out="dets"))
        with pytest.raises(ValueError, match="Unsupported image type"):
            graph.run(12345, providers={"detector": mock_detector})

    def test_run_empty_graph_raises(self, pil_image, mock_detector):
        """Empty graph should raise during compilation."""
        graph = Graph("empty")
        with pytest.raises(Exception):
            graph.run(pil_image, providers={"detector": mock_detector})


# ===========================================================================
# Test: Scheduler Integration
# ===========================================================================


class TestGraphRunScheduler:
    """Test scheduler parameter forwarding."""

    def test_run_with_sync_scheduler(self, pil_image, mock_detector):
        """Explicit SyncScheduler should work."""
        from mata.core.graph import SyncScheduler

        graph = Graph("sync").then(Detect(using="detector", out="dets"))
        result = graph.run(
            pil_image,
            providers={"detector": mock_detector},
            scheduler=SyncScheduler(),
        )
        assert isinstance(result, MultiResult)

    def test_run_with_parallel_scheduler(self, pil_image, mock_detector):
        """ParallelScheduler should work."""
        graph = Graph("parallel").then(Detect(using="detector", out="dets"))
        result = graph.run(
            pil_image,
            providers={"detector": mock_detector},
            scheduler=ParallelScheduler(),
        )
        assert isinstance(result, MultiResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
