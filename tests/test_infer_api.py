"""Unit tests for Task 7.1: mata.infer() API.

Tests cover:
- All image input formats (str, Path, PIL, numpy)
- Graph and list[Node] inputs
- Provider resolution (flat and nested dicts)
- Device selection
- Scheduler integration (SyncScheduler, ParallelScheduler)
- Backward compatibility (load/run APIs unchanged)
- Error handling (invalid image, empty graph, bad providers)
- Integration with full graph execution
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.result import MultiResult
from mata.core.graph import Graph, ParallelScheduler, SyncScheduler
from mata.core.types import Instance, VisionResult
from mata.nodes.detect import Detect
from mata.nodes.filter import Filter
from mata.nodes.fuse import Fuse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 200, height: int = 100) -> PILImage.Image:
    """Create a dummy PIL image."""
    return PILImage.new("RGB", (width, height), color=(128, 128, 128))


def _make_numpy_image(width: int = 200, height: int = 100) -> np.ndarray:
    """Create a dummy numpy image (HWC, uint8, RGB)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_temp_image(suffix: str = ".png") -> str:
    """Create a temporary image file and return the path."""
    img = _make_pil_image()
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


def _make_sample_detections() -> Detections:
    """Create a Detections artifact for mock return values."""
    instances = [
        Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
        Instance(bbox=(200, 50, 300, 200), score=0.82, label=1, label_name="dog"),
    ]
    vr = VisionResult(instances=instances, meta={"model": "mock"})
    return Detections.from_vision_result(vr)


class _MockDetectAdapter:
    """Mock detection adapter that returns fixed Detections."""

    model_id = "mock/detector"

    def predict(self, image, **kwargs):
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.95, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.82, label=1, label_name="dog"),
        ]
        return VisionResult(instances=instances, meta={"model": "mock"})


class _MockClassifyAdapter:
    """Mock classification adapter."""

    model_id = "mock/classifier"

    def predict(self, image, **kwargs):
        from mata.core.types import Classification, ClassifyResult

        return ClassifyResult(
            classifications=[Classification(label=0, label_name="cat", score=0.95)],
            meta={},
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_image_path():
    """Provide a temporary image file path, cleaned up after test."""
    path = _make_temp_image()
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def pil_image():
    return _make_pil_image()


@pytest.fixture
def numpy_image():
    return _make_numpy_image()


@pytest.fixture
def mock_detector():
    return _MockDetectAdapter()


@pytest.fixture
def mock_classifier():
    return _MockClassifyAdapter()


# ===========================================================================
# Test: Image Input Formats
# ===========================================================================


class TestInferImageInputs:
    """Test that infer() accepts all documented image formats."""

    def test_infer_with_str_path(self, temp_image_path, mock_detector):
        """String path to image."""
        from mata.api import infer

        result = infer(
            image=temp_image_path,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_with_pathlib_path(self, temp_image_path, mock_detector):
        """pathlib.Path to image."""
        from mata.api import infer

        result = infer(
            image=Path(temp_image_path),
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_with_pil_image(self, pil_image, mock_detector):
        """PIL.Image.Image input."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_with_numpy_array(self, numpy_image, mock_detector):
        """np.ndarray input."""
        from mata.api import infer

        result = infer(
            image=numpy_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_rejects_unsupported_image_type(self, mock_detector):
        """Passing an unsupported type should raise ValueError."""
        from mata.api import infer

        with pytest.raises(ValueError, match="Unsupported image type"):
            infer(
                image=12345,
                graph=[Detect(using="detector", out="dets")],
                providers={"detector": mock_detector},
            )

    def test_infer_rejects_list_as_image(self, mock_detector):
        """A list is not a valid image type."""
        from mata.api import infer

        with pytest.raises(ValueError, match="Unsupported image type"):
            infer(
                image=[1, 2, 3],
                graph=[Detect(using="detector", out="dets")],
                providers={"detector": mock_detector},
            )


# ===========================================================================
# Test: Graph Input Formats
# ===========================================================================


class TestInferGraphInputs:
    """Test that infer() accepts Graph objects and node lists."""

    def test_infer_with_node_list(self, pil_image, mock_detector):
        """A plain list of Node instances."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_with_graph_object(self, pil_image, mock_detector):
        """A pre-built Graph object."""
        from mata.api import infer

        g = Graph("test_graph").then(Detect(using="detector", out="dets"))
        result = infer(
            image=pil_image,
            graph=g,
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_infer_empty_node_list_raises(self, pil_image, mock_detector):
        """Empty list of nodes should raise ValueError."""
        from mata.api import infer

        with pytest.raises(ValueError, match="Node list cannot be empty"):
            infer(image=pil_image, graph=[], providers={"detector": mock_detector})

    def test_infer_non_node_in_list_raises(self, pil_image, mock_detector):
        """Non-Node objects in list should raise ValueError."""
        from mata.api import infer

        with pytest.raises(ValueError, match="Expected Node instance"):
            infer(
                image=pil_image,
                graph=["not_a_node"],
                providers={"detector": mock_detector},
            )

    def test_infer_unsupported_graph_type_raises(self, pil_image, mock_detector):
        """A string is not a valid graph."""
        from mata.api import infer

        with pytest.raises(ValueError, match="Unsupported graph type"):
            infer(
                image=pil_image,
                graph="not_a_graph",
                providers={"detector": mock_detector},
            )


# ===========================================================================
# Test: Provider Resolution
# ===========================================================================


class TestInferProviders:
    """Test flat and nested provider dicts."""

    def test_flat_providers_single(self, pil_image, mock_detector):
        """Flat dict with one provider."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)

    def test_nested_providers(self, pil_image, mock_detector):
        """Nested dict passed through directly."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detect": {"detector": mock_detector}},
        )
        assert isinstance(result, MultiResult)

    def test_empty_providers(self, pil_image):
        """Empty providers dict with a no-provider graph should still work
        if no node actually requires one. We use a node that does require one,
        so this should fail at runtime."""
        from mata.api import infer

        # Detect requires a provider, but providers is empty → should fail
        # (the exact error depends on compilation / runtime behaviour)
        with pytest.raises(Exception):
            infer(
                image=pil_image,
                graph=[Detect(using="detector", out="dets")],
                providers={},
            )

    def test_provider_name_matches_using(self, pil_image, mock_detector):
        """Provider key must match the 'using' kwarg on the node."""
        from mata.api import infer

        # "my_det" doesn't match "detector" → should fail at runtime
        with pytest.raises(Exception):
            infer(
                image=pil_image,
                graph=[Detect(using="detector", out="dets")],
                providers={"my_det": mock_detector},
            )


# ===========================================================================
# Test: Device Selection
# ===========================================================================


class TestInferDeviceSelection:
    """Test device kwarg routing."""

    def test_device_cpu(self, pil_image, mock_detector):
        """Explicit CPU device."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
            device="cpu",
        )
        assert isinstance(result, MultiResult)

    def test_device_auto(self, pil_image, mock_detector):
        """Auto-detect device (default)."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
            device="auto",
        )
        assert isinstance(result, MultiResult)


# ===========================================================================
# Test: Scheduler Integration
# ===========================================================================


class TestInferScheduler:
    """Test passing different schedulers."""

    def test_default_sync_scheduler(self, pil_image, mock_detector):
        """Default (None) uses SyncScheduler."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
            scheduler=None,
        )
        assert isinstance(result, MultiResult)

    def test_explicit_sync_scheduler(self, pil_image, mock_detector):
        """Explicitly pass SyncScheduler."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
            scheduler=SyncScheduler(),
        )
        assert isinstance(result, MultiResult)

    def test_parallel_scheduler(self, pil_image, mock_detector):
        """Use ParallelScheduler."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
            scheduler=ParallelScheduler(),
        )
        assert isinstance(result, MultiResult)


# ===========================================================================
# Test: Result Access
# ===========================================================================


class TestInferResultAccess:
    """Verify the MultiResult can be accessed correctly."""

    def test_result_has_channels(self, pil_image, mock_detector):
        """Output channels are populated."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert result.has_channel("dets")

    def test_result_attribute_access(self, pil_image, mock_detector):
        """Channels accessible as attributes."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        dets = result.dets
        assert isinstance(dets, Detections)

    def test_result_has_metrics(self, pil_image, mock_detector):
        """Metrics dict is populated."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert "total_time_ms" in result.metrics

    def test_result_has_provenance(self, pil_image, mock_detector):
        """Provenance metadata is present."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[Detect(using="detector", out="dets")],
            providers={"detector": mock_detector},
        )
        assert "graph_name" in result.provenance
        assert "timestamp" in result.provenance


# ===========================================================================
# Test: Multi-Node Graphs
# ===========================================================================


class TestInferMultiNodeGraph:
    """Integration-level tests with more complex graphs."""

    def test_detect_then_filter(self, pil_image, mock_detector):
        """Two-node sequential graph: Detect → Filter."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[
                Detect(using="detector", out="dets"),
                Filter(src="dets", score_gt=0.9, out="filtered"),
            ],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)
        assert result.has_channel("filtered")

    def test_detect_filter_fuse(self, pil_image, mock_detector):
        """Three-node graph: Detect → Filter → Fuse."""
        from mata.api import infer

        result = infer(
            image=pil_image,
            graph=[
                Detect(using="detector", out="dets"),
                Filter(src="dets", score_gt=0.5, out="filtered"),
                Fuse(dets="filtered", out="final"),
            ],
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)
        assert result.has_channel("final")

    def test_graph_object_multi_node(self, pil_image, mock_detector):
        """Pre-built Graph with chained .then() calls."""
        from mata.api import infer

        g = (
            Graph("pipeline")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.5, out="filtered"))
        )
        result = infer(
            image=pil_image,
            graph=g,
            providers={"detector": mock_detector},
        )
        assert isinstance(result, MultiResult)
        assert result.has_channel("filtered")


# ===========================================================================
# Test: Backward Compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure existing load() and run() APIs are unaffected."""

    def test_load_still_works(self):
        """load() function is still importable and callable."""
        import mata

        assert hasattr(mata, "load")
        assert callable(mata.load)

    def test_run_still_works(self):
        """run() function is still importable and callable."""
        import mata

        assert hasattr(mata, "run")
        assert callable(mata.run)

    def test_infer_is_exported(self):
        """infer() is accessible from the mata namespace."""
        import mata

        assert hasattr(mata, "infer")
        assert callable(mata.infer)

    def test_list_models_still_works(self):
        """list_models() is still importable."""
        import mata

        assert hasattr(mata, "list_models")

    def test_register_model_still_works(self):
        """register_model() is still importable."""
        import mata

        assert hasattr(mata, "register_model")


# ===========================================================================
# Test: Provider Normalization Helper
# ===========================================================================


class TestNormalizeProviders:
    """Unit tests for _normalize_providers helper."""

    def test_returns_empty_for_empty_providers(self):
        from mata.api import _normalize_providers

        g = Graph()
        g._nodes = []
        flat, nested = _normalize_providers({}, g)
        assert flat == {}
        assert nested == {}

    def test_passthrough_nested_dict(self, mock_detector):
        from mata.api import _normalize_providers

        g = Graph()
        g._nodes = []
        input_nested = {"detect": {"detr": mock_detector}}
        flat, nested = _normalize_providers(input_nested, g)
        assert nested == input_nested
        # Flat should contain the inner entries
        assert "detr" in flat
        assert flat["detr"] is mock_detector

    def test_flat_dict_auto_detect_capability(self, mock_detector):
        from mata.api import _normalize_providers

        g = Graph().then(Detect(using="detector", out="dets"))
        flat, nested = _normalize_providers({"detector": mock_detector}, g)

        assert "detect" in nested
        assert "detector" in nested["detect"]
        assert nested["detect"]["detector"] is mock_detector
        # Flat should pass through as-is
        assert flat["detector"] is mock_detector

    def test_flat_dict_fallback_infer_capability(self):
        """When the adapter class name contains a capability hint."""
        from mata.api import _normalize_providers

        # Use mock adapter named with 'detect' in class name
        adapter = _MockDetectAdapter()
        g = Graph()
        g._nodes = []
        _flat, nested = _normalize_providers({"my_adapter": adapter}, g)

        # Should infer "detect" from class name
        assert "detect" in nested


# ===========================================================================
# Test: _infer_capability helper
# ===========================================================================


class TestInferCapabilityHelper:
    """Test the _infer_capability fallback logic."""

    def test_detect_in_name(self):
        from mata.api import _infer_capability

        class FakeDetectAdapter:
            pass

        assert _infer_capability(FakeDetectAdapter()) == "detect"

    def test_classify_in_name(self):
        from mata.api import _infer_capability

        class FakeClassifyAdapter:
            pass

        assert _infer_capability(FakeClassifyAdapter()) == "classify"

    def test_segment_in_name(self):
        from mata.api import _infer_capability

        class FakeSegmentAdapter:
            pass

        assert _infer_capability(FakeSegmentAdapter()) == "segment"

    def test_depth_in_name(self):
        from mata.api import _infer_capability

        class FakeDepthAdapter:
            pass

        assert _infer_capability(FakeDepthAdapter()) == "depth"

    def test_sam_in_name(self):
        from mata.api import _infer_capability

        class FakeSAMAdapter:
            pass

        assert _infer_capability(FakeSAMAdapter()) == "segment"

    def test_vlm_in_name(self):
        from mata.api import _infer_capability

        class FakeVLMAdapter:
            pass

        assert _infer_capability(FakeVLMAdapter()) == "vlm"

    def test_unknown_returns_none(self):
        from mata.api import _infer_capability

        class FooBarBaz:
            pass

        assert _infer_capability(FooBarBaz()) is None


# ===========================================================================
# Test: _build_capability_map helper
# ===========================================================================


class TestBuildCapabilityMap:
    """Verify the node-to-capability mapping."""

    def test_all_expected_nodes_present(self):
        from mata.api import _build_capability_map

        cap_map = _build_capability_map()
        assert cap_map["Detect"] == "detect"
        assert cap_map["Classify"] == "classify"
        assert cap_map["SegmentImage"] == "segment"
        assert cap_map["EstimateDepth"] == "depth"
        assert cap_map["Track"] == "track"
        assert cap_map["PromptBoxes"] == "segment"
        assert cap_map["VLMDescribe"] == "vlm"
        assert cap_map["VLMDetect"] == "vlm"
        assert cap_map["VLMQuery"] == "vlm"
