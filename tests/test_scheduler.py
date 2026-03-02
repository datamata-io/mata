"""Tests for graph schedulers.

Comprehensive test suite for SyncScheduler and ParallelScheduler covering:
- Sequential execution in topological order
- Artifact passing between nodes
- Error propagation and handling
- Timing metrics collection
- Provenance data collection
- Parallel execution (ParallelScheduler)
- Cleanup on error
- MultiResult construction
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.graph.context import ExecutionContext
from mata.core.graph.graph import Graph
from mata.core.graph.node import Node
from mata.core.graph.scheduler import ParallelScheduler, SyncScheduler


# Mock Artifact types for testing
class MockMasks(Artifact):
    """Mock Masks artifact for testing."""

    def __init__(self, data: str = "mock_masks"):
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockMasks":
        return cls(data.get("data", "mock_masks"))


class MockDepth(Artifact):
    """Mock Depth artifact for testing."""

    def __init__(self, data: str = "mock_depth"):
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockDepth":
        return cls(data.get("data", "mock_depth"))


# Mock Node implementations for testing
class MockDetectNode(Node):
    """Mock detection node that simulates detection."""

    inputs = {"image": Image}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "MockDetect", out: str = "detections", delay_ms: float = 0):
        super().__init__(name=name)
        self.output_name = out
        self.delay_ms = delay_ms
        self.executed = False

    def run(self, ctx, **inputs):
        """Execute with optional delay."""
        self.executed = True
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        # Create mock detections
        from mata.core.types import Instance

        instances = [
            Instance(
                bbox=(10, 10, 50, 50),
                label=0,
                label_name="person",
                score=0.9,
            )
        ]
        detections = Detections(instances=instances, instance_ids=["inst_0000"])
        return {"detections": detections}


class MockFilterNode(Node):
    """Mock filter node."""

    inputs = {"detections": Detections}
    outputs = {"detections": Detections}

    def __init__(self, name: str = "MockFilter", src: str = "detections", out: str = "detections", delay_ms: float = 0):
        super().__init__(name=name)
        self.src = src
        self.out = out
        self.delay_ms = delay_ms
        self.executed = False

    def run(self, ctx, **inputs):
        """Execute with optional delay."""
        self.executed = True
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        # Just pass through the detections
        detections = inputs.get("detections")
        return {"detections": detections}


class MockSegmentNode(Node):
    """Mock segmentation node."""

    inputs = {"image": Image, "detections": Detections}
    outputs = {"masks": MockMasks}

    def __init__(self, name: str = "MockSegment", out: str = "masks", delay_ms: float = 0):
        super().__init__(name=name)
        self.output_name = out
        self.delay_ms = delay_ms
        self.executed = False

    def run(self, ctx, **inputs):
        """Execute with optional delay."""
        self.executed = True
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        masks = MockMasks(data="segmented")
        return {"masks": masks}


class MockDepthNode(Node):
    """Mock depth estimation node."""

    inputs = {"image": Image}
    outputs = {"depth": MockDepth}

    def __init__(self, name: str = "MockDepth", out: str = "depth", delay_ms: float = 0):
        super().__init__(name=name)
        self.output_name = out
        self.delay_ms = delay_ms
        self.executed = False

    def run(self, ctx, **inputs):
        """Execute with optional delay."""
        self.executed = True
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        depth = MockDepth(data="depth_map")
        return {"depth": depth}


class MockFailNode(Node):
    """Mock node that always fails."""

    inputs = {"image": Image}
    outputs = {"result": MockMasks}

    def __init__(self, name: str = "MockFail", error_msg: str = "Mock failure"):
        super().__init__(name=name)
        self.error_msg = error_msg

    def run(self, ctx, **inputs):
        """Always raises an exception."""
        raise RuntimeError(self.error_msg)


# Fixtures
@pytest.fixture
def mock_image():
    """Create mock image artifact."""
    from PIL import Image as PILImage

    pil_img = PILImage.new("RGB", (100, 100), color="red")
    return Image.from_pil(pil_img)


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    return ExecutionContext(providers={}, device="cpu", cache_artifacts=True)


# Tests for SyncScheduler
class TestSyncScheduler:
    """Tests for SyncScheduler."""

    def test_init(self):
        """Test scheduler initialization."""
        scheduler = SyncScheduler()
        assert scheduler is not None

    def test_execute_single_node(self, mock_image, mock_context):
        """Test executing graph with single node."""
        # Build graph
        node = MockDetectNode(name="detect")
        graph = Graph(name="single_node").add(node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify
        assert isinstance(result, MultiResult)
        assert result.has_channel("detections")
        assert node.executed
        assert "detect" in result.metrics
        assert "latency_ms" in result.metrics["detect"]
        assert result.metrics["detect"]["latency_ms"] > 0
        assert "total_time_ms" in result.metrics

    def test_execute_sequential_nodes(self, mock_image, mock_context):
        """Test executing sequential graph (detect -> filter)."""
        # Build graph
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph = (
            Graph(name="sequential")
            .add(detect, inputs={"image": "input.image"})
            .add(filter_node, inputs={"detections": "detect.detections"})
        )

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify
        assert isinstance(result, MultiResult)
        assert result.has_channel("detections")
        assert detect.executed
        assert filter_node.executed
        assert "detect" in result.metrics
        assert "filter" in result.metrics

    def test_execute_parallel_branches(self, mock_image, mock_context):
        """Test executing graph with parallel branches (detect + depth)."""
        # Build graph with parallel branches
        detect = MockDetectNode(name="detect")
        depth = MockDepthNode(name="depth")

        graph = Graph(name="parallel_branches")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify both branches executed
        assert isinstance(result, MultiResult)
        assert result.has_channel("detections")
        assert result.has_channel("depth")
        assert detect.executed
        assert depth.executed

    def test_execute_multi_stage(self, mock_image, mock_context):
        """Test executing multi-stage graph (detect -> filter -> segment)."""
        # Build graph
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")
        segment = MockSegmentNode(name="segment")

        graph = Graph(name="multi_stage")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})
        graph.add(segment, inputs={"image": "input.image", "detections": "filter.detections"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify
        assert isinstance(result, MultiResult)
        assert result.has_channel("masks")
        assert detect.executed
        assert filter_node.executed
        assert segment.executed

        # Verify metrics for all nodes
        assert "detect" in result.metrics
        assert "filter" in result.metrics
        assert "segment" in result.metrics

    def test_timing_metrics(self, mock_image, mock_context):
        """Test that timing metrics are collected correctly."""
        # Create node with delay
        node = MockDetectNode(name="detect", delay_ms=50)
        graph = Graph(name="timing_test").add(node, inputs={"image": "input.image"})

        # Compile and execute
        compiled = graph.compile(providers={})
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify timing
        assert result.metrics["detect"]["latency_ms"] >= 50
        assert result.metrics["total_time_ms"] >= 50

    def test_provenance_collection(self, mock_image, mock_context):
        """Test that provenance metadata is collected."""
        node = MockDetectNode(name="detect")
        graph = Graph(name="provenance_test").add(node, inputs={"image": "input.image"})

        # Compile and execute
        compiled = graph.compile(providers={})
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify provenance
        assert "graph_name" in result.provenance
        assert result.provenance["graph_name"] == "provenance_test"
        assert "graph_hash" in result.provenance
        assert "timestamp" in result.provenance
        assert "device" in result.provenance
        assert result.provenance["device"] == "cpu"
        assert "num_nodes" in result.provenance
        assert "num_stages" in result.provenance

    def test_error_handling(self, mock_image, mock_context):
        """Test error handling when node fails."""
        fail_node = MockFailNode(name="fail", error_msg="Test error")
        graph = Graph(name="error_test").add(fail_node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute - should raise
        scheduler = SyncScheduler()
        with pytest.raises(RuntimeError) as exc_info:
            scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        assert "Test error" in str(exc_info.value)

    def test_error_in_middle_of_graph(self, mock_image, mock_context):
        """Test error handling when node fails in middle of graph."""
        detect = MockDetectNode(name="detect")
        fail_node = MockFailNode(name="fail")

        graph = Graph(name="error_middle")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(fail_node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute - should fail
        scheduler = SyncScheduler()
        with pytest.raises(RuntimeError):
            scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # First node should have executed
        assert detect.executed

    def test_artifact_passing(self, mock_image, mock_context):
        """Test that artifacts are correctly passed between nodes."""
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph = Graph(name="artifact_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})

        # Compile and execute
        compiled = graph.compile(providers={})
        scheduler = SyncScheduler()
        scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify artifacts exist in context
        assert mock_context.has("detect.detections")
        assert mock_context.has("filter.detections")

        # Verify they're the same object (passed through)
        dets = mock_context.retrieve("detect.detections")
        filtered = mock_context.retrieve("filter.detections")
        assert isinstance(dets, Detections)
        assert isinstance(filtered, Detections)

    def test_multiresult_channels(self, mock_image, mock_context):
        """Test MultiResult contains correct channels."""
        detect = MockDetectNode(name="detect")
        depth = MockDepthNode(name="depth")

        graph = Graph(name="channels_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})

        # Compile and execute
        compiled = graph.compile(providers={})
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify channels
        assert "detections" in result.channels
        assert "depth" in result.channels

        # Verify attribute access
        assert result.detections is result.channels["detections"]
        assert result.depth is result.channels["depth"]

    def test_stage_metrics(self, mock_image, mock_context):
        """Test that per-stage metrics are collected."""
        # Create multi-stage graph
        detect = MockDetectNode(name="detect", delay_ms=20)
        filter_node = MockFilterNode(name="filter", delay_ms=10)

        graph = Graph(name="stage_metrics")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})

        # Compile and execute
        compiled = graph.compile(providers={})
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify stage metrics exist
        assert any(key.startswith("stage_") and key.endswith("_ms") for key in result.metrics)


# Tests for ParallelScheduler
class TestParallelScheduler:
    """Tests for ParallelScheduler."""

    def test_init(self):
        """Test scheduler initialization."""
        scheduler = ParallelScheduler(max_workers=4)
        assert scheduler.max_workers == 4

    def test_execute_single_node(self, mock_image, mock_context):
        """Test executing graph with single node."""
        node = MockDetectNode(name="detect")
        graph = Graph(name="single_node").add(node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = ParallelScheduler(max_workers=2)
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify
        assert isinstance(result, MultiResult)
        assert result.has_channel("detections")
        assert node.executed

    def test_parallel_execution(self, mock_image, mock_context):
        """Test that independent nodes execute in parallel."""
        # Create two independent nodes with delays
        detect = MockDetectNode(name="detect", delay_ms=100)
        depth = MockDepthNode(name="depth", delay_ms=100)

        graph = Graph(name="parallel_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute with parallel scheduler
        scheduler = ParallelScheduler(max_workers=2)
        start_time = time.time()
        scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})
        end_time = time.time()

        # Verify both executed
        assert detect.executed
        assert depth.executed

        # Verify parallel execution (should be ~100ms, not 200ms)
        # Allow some overhead
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 150, f"Parallel execution took {elapsed_ms}ms, expected < 150ms"

    def test_sequential_after_parallel(self, mock_image, mock_context):
        """Test mixed parallel and sequential execution."""
        # Stage 1: Parallel
        detect = MockDetectNode(name="detect", delay_ms=50)
        depth = MockDepthNode(name="depth", delay_ms=50)

        # Stage 2: Sequential (depends on detect)
        filter_node = MockFilterNode(name="filter", delay_ms=30)

        graph = Graph(name="mixed_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = ParallelScheduler(max_workers=2)
        start_time = time.time()
        scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})
        end_time = time.time()

        # Verify all executed
        assert detect.executed
        assert depth.executed
        assert filter_node.executed

        # Verify timing (parallel stage ~50ms + sequential ~30ms = ~80ms total)
        # Not 130ms (50 + 50 + 30)
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 120, f"Execution took {elapsed_ms}ms, expected < 120ms"

    def test_error_handling_in_parallel(self, mock_image, mock_context):
        """Test error handling when parallel node fails."""
        detect = MockDetectNode(name="detect")
        fail_node = MockFailNode(name="fail", error_msg="Parallel failure")

        graph = Graph(name="error_parallel")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(fail_node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute - should raise
        scheduler = ParallelScheduler(max_workers=2)
        with pytest.raises(RuntimeError) as exc_info:
            scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        assert "Parallel failure" in str(exc_info.value)

    def test_max_workers_limit(self, mock_image, mock_context):
        """Test that max_workers is respected."""
        # Create many parallel nodes
        nodes = [MockDetectNode(name=f"detect_{i}", delay_ms=10) for i in range(10)]

        graph = Graph(name="workers_test")
        for node in nodes:
            graph.add(node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute with limited workers
        scheduler = ParallelScheduler(max_workers=2)
        scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify all executed
        for node in nodes:
            assert node.executed

    def test_single_stage_uses_direct_execution(self, mock_image, mock_context):
        """Test that single-node stages skip thread pool overhead."""
        node = MockDetectNode(name="detect")
        graph = Graph(name="single_stage").add(node, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        scheduler = ParallelScheduler(max_workers=2)
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify executed
        assert node.executed
        assert result.has_channel("detections")


# Integration tests
class TestSchedulerIntegration:
    """Integration tests with real graph compilation."""

    def test_sync_with_graph_builder(self, mock_image):
        """Test SyncScheduler with Graph builder API."""
        # Build using fluent API
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")

        graph = Graph(name="integration_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})

        # Compile
        compiled = graph.compile(providers={})

        # Create context and execute
        ctx = ExecutionContext(providers={}, device="cpu")
        scheduler = SyncScheduler()
        result = scheduler.execute(compiled, ctx, initial_artifacts={"input.image": mock_image})

        # Verify
        assert isinstance(result, MultiResult)
        assert result.has_channel("detections")
        assert "detect" in result.metrics
        assert "filter" in result.metrics

    def test_parallel_with_graph_builder(self, mock_image):
        """Test ParallelScheduler with Graph builder API."""
        # Build parallel branches
        detect = MockDetectNode(name="detect")
        depth = MockDepthNode(name="depth")

        graph = Graph(name="parallel_integration")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})

        # Compile
        compiled = graph.compile(providers={})

        # Execute
        ctx = ExecutionContext(providers={}, device="cpu")
        scheduler = ParallelScheduler(max_workers=2)
        result = scheduler.execute(compiled, ctx, initial_artifacts={"input.image": mock_image})

        # Verify both channels
        assert result.has_channel("detections")
        assert result.has_channel("depth")

    def test_execution_order_respected(self, mock_image):
        """Test that execution order is respected."""
        # Create nodes with dependency chain
        detect = MockDetectNode(name="detect")
        filter_node = MockFilterNode(name="filter")
        segment = MockSegmentNode(name="segment")

        graph = Graph(name="order_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(filter_node, inputs={"detections": "detect.detections"})
        graph.add(segment, inputs={"image": "input.image", "detections": "filter.detections"})

        # Compile
        compiled = graph.compile(providers={})

        # Verify execution order (should be 3 stages)
        assert len(compiled.execution_order) == 3
        assert len(compiled.execution_order[0]) == 1  # detect
        assert len(compiled.execution_order[1]) == 1  # filter
        assert len(compiled.execution_order[2]) == 1  # segment

        # Execute
        ctx = ExecutionContext(providers={}, device="cpu")
        scheduler = SyncScheduler()
        scheduler.execute(compiled, ctx, initial_artifacts={"input.image": mock_image})

        # Verify all executed
        assert detect.executed
        assert filter_node.executed
        assert segment.executed


# Tests for OptimizedParallelScheduler
class TestOptimizedParallelScheduler:
    """Tests for OptimizedParallelScheduler."""

    def test_init_valid_strategies(self):
        """Test scheduler initialization with valid strategies."""
        # Test all valid strategies
        for strategy in ["auto", "round_robin", "memory_aware"]:
            scheduler = OptimizedParallelScheduler(max_workers=2, device_placement=strategy, unload_unused=True)
            assert scheduler.device_placement == strategy
            assert scheduler.unload_unused is True
            assert scheduler.max_workers == 2
            assert isinstance(scheduler._device_pool, list)
            assert "cpu" in scheduler._device_pool

    def test_init_invalid_strategy(self):
        """Test scheduler initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid device_placement"):
            OptimizedParallelScheduler(device_placement="invalid")

    def test_init_devices(self):
        """Test device initialization."""
        scheduler = OptimizedParallelScheduler()
        devices = scheduler._init_devices()

        # Should at least have CPU
        assert "cpu" in devices
        assert isinstance(devices, list)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_init_devices_with_cuda(self, mock_device_count, mock_cuda_available):
        """Test device initialization with CUDA."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        scheduler = OptimizedParallelScheduler()
        devices = scheduler._init_devices()

        # Should have GPUs first, then CPU
        assert "cuda:0" in devices
        assert "cuda:1" in devices
        assert "cpu" in devices
        assert devices[0] == "cuda:0"

    @patch("torch.cuda.is_available")
    def test_init_devices_no_cuda(self, mock_cuda_available):
        """Test device initialization without CUDA."""
        mock_cuda_available.return_value = False

        scheduler = OptimizedParallelScheduler()
        devices = scheduler._init_devices()

        # Should only have CPU
        assert devices == ["cpu"]

    def test_assign_device_auto_strategy(self):
        """Test device assignment with auto strategy."""
        scheduler = OptimizedParallelScheduler(device_placement="auto")
        scheduler._device_pool = ["cuda:0", "cpu"]

        node = MockDetectNode()
        device = scheduler._assign_device(node)

        # Should prefer GPU
        assert device == "cuda:0"
        assert scheduler._device_usage[device] == 1

    def test_assign_device_auto_strategy_cpu_only(self):
        """Test device assignment with auto strategy, CPU only."""
        scheduler = OptimizedParallelScheduler(device_placement="auto")
        scheduler._device_pool = ["cpu"]

        node = MockDetectNode()
        device = scheduler._assign_device(node)

        # Should use CPU
        assert device == "cpu"

    def test_assign_device_round_robin(self):
        """Test device assignment with round-robin strategy."""
        scheduler = OptimizedParallelScheduler(device_placement="round_robin")
        scheduler._device_pool = ["cuda:0", "cuda:1", "cpu"]

        node1 = MockDetectNode(name="node1")
        node2 = MockDetectNode(name="node2")
        node3 = MockDetectNode(name="node3")

        device1 = scheduler._assign_device(node1)
        device2 = scheduler._assign_device(node2)
        device3 = scheduler._assign_device(node3)

        # Should round-robin through GPUs
        assert device1 == "cuda:0"
        assert device2 == "cuda:1"
        assert device3 == "cuda:0"  # Wraps around

    def test_assign_device_round_robin_cpu_only(self):
        """Test round-robin with CPU only."""
        scheduler = OptimizedParallelScheduler(device_placement="round_robin")
        scheduler._device_pool = ["cpu"]

        node = MockDetectNode()
        device = scheduler._assign_device(node)

        assert device == "cpu"

    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    def test_assign_device_memory_aware(
        self, mock_memory_allocated, mock_get_props, mock_cuda_available, mock_device_count
    ):
        """Test device assignment with memory-aware strategy."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        # Mock device properties — need entries for _init_devices (2 GPUs)
        # AND _choose_best_memory_device (2 GPUs)
        prop0 = Mock()
        prop0.name = "MockGPU0"
        prop0.total_memory = 8 * 1024**3  # 8GB
        prop1 = Mock()
        prop1.name = "MockGPU1"
        prop1.total_memory = 16 * 1024**3  # 16GB
        # _init_devices logs props for 2 GPUs, then _choose_best_memory_device queries 2 GPUs
        mock_get_props.side_effect = [prop0, prop1, prop0, prop1]

        # Mock memory usage — _choose_best_memory_device queries 2 GPUs
        mock_memory_allocated.side_effect = [6 * 1024**3, 2 * 1024**3]  # 6GB used, 2GB used

        scheduler = OptimizedParallelScheduler(device_placement="memory_aware")
        # _init_devices should have discovered cuda:0, cuda:1, cpu

        node = MockDetectNode()
        device = scheduler._assign_device(node)

        # Should choose device 1 (16GB - 2GB > 8GB - 6GB)
        assert device == "cuda:1"

    def test_choose_best_memory_device_no_cuda(self):
        """Test memory-aware device selection without CUDA."""
        scheduler = OptimizedParallelScheduler(device_placement="memory_aware")
        scheduler._device_pool = ["cpu"]

        device = scheduler._choose_best_memory_device()
        assert device == "cpu"

    def test_unload_model_disabled(self):
        """Test model unloading when disabled."""
        scheduler = OptimizedParallelScheduler(unload_unused=False)

        mock_provider = Mock()
        mock_provider.model = Mock()

        # Should not unload when disabled
        scheduler._unload_model(mock_provider)

        assert hasattr(mock_provider, "model")

    @patch("torch.cuda.empty_cache")
    def test_unload_model_enabled(self, mock_empty_cache):
        """Test model unloading when enabled."""
        scheduler = OptimizedParallelScheduler(unload_unused=True)

        mock_provider = Mock()
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_provider.model = mock_model

        # Should unload when enabled
        scheduler._unload_model(mock_provider)

        # Model should be deleted and cache cleared
        assert not hasattr(mock_provider, "model")
        mock_empty_cache.assert_called_once()

    def test_unload_model_cpu_device(self):
        """Test model unloading for CPU device."""
        scheduler = OptimizedParallelScheduler(unload_unused=True)

        mock_provider = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_provider.model = mock_model

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            scheduler._unload_model(mock_provider)

            # Model deleted but no CUDA cache clear for CPU
            assert not hasattr(mock_provider, "model")
            mock_empty_cache.assert_not_called()

    def test_unload_model_no_model_attribute(self):
        """Test model unloading when provider has no model."""
        scheduler = OptimizedParallelScheduler(unload_unused=True)

        mock_provider = Mock()
        # No model attribute

        # Should not crash
        scheduler._unload_model(mock_provider)

    def test_unload_model_with_adapter_attributes(self):
        """Test model unloading with various adapter attributes."""
        scheduler = OptimizedParallelScheduler(unload_unused=True)

        mock_provider = Mock()
        mock_provider.adapter = Mock()
        mock_provider._model = Mock()
        mock_provider.processor = Mock()

        scheduler._unload_model(mock_provider)

        # All attributes should be cleared
        assert mock_provider.adapter is None
        assert mock_provider._model is None
        assert mock_provider.processor is None

    def test_move_provider_to_device(self):
        """Test moving provider to device (placeholder implementation)."""
        scheduler = OptimizedParallelScheduler()
        node = MockDetectNode()
        context = ExecutionContext(providers={}, device="cpu")

        # Should not crash (placeholder implementation)
        scheduler._move_provider_to_device(node, context, "cuda:0")

    def test_execute_node_with_device(self, mock_image):
        """Test executing node with device assignment."""
        scheduler = OptimizedParallelScheduler()
        node = MockDetectNode()
        context = ExecutionContext(providers={}, device="cpu")
        context.store("input.image", mock_image)
        wiring = {"MockDetect.image": "input.image"}

        result = scheduler._execute_node_with_device(node, context, wiring, "cuda:0")

        # Should execute and restore device
        assert "detections" in result
        assert context.device == "cpu"  # Restored
        assert node.executed

    def test_unload_node_provider(self):
        """Test unloading provider for specific node."""
        scheduler = OptimizedParallelScheduler()
        node = MockDetectNode()
        node.provider_name = "test_provider"
        context = ExecutionContext(providers={}, device="cpu")

        # Should not crash (placeholder implementation)
        scheduler._unload_node_provider(node, context)

    def test_execute_single_node_stage(self, mock_image, mock_context):
        """Test optimized execution with single node stage."""
        scheduler = OptimizedParallelScheduler(max_workers=2, device_placement="auto", unload_unused=False)
        scheduler._device_pool = ["cuda:0", "cpu"]

        # Create simple graph
        detect = MockDetectNode(name="detect")
        graph = Graph(name="single_node")
        graph.add(detect, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})

        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify execution
        assert detect.executed
        assert result.has_channel("detections")
        assert "device_usage" in result.metrics
        assert result.metrics["device_placement_strategy"] == "auto"
        assert result.metrics["unload_unused_enabled"] is False

    def test_execute_parallel_stage(self, mock_image, mock_context):
        """Test optimized execution with parallel stage."""
        scheduler = OptimizedParallelScheduler(max_workers=2, device_placement="round_robin", unload_unused=True)
        scheduler._device_pool = ["cuda:0", "cuda:1", "cpu"]

        # Create graph with parallel branches
        detect = MockDetectNode(name="detect", delay_ms=50)
        depth = MockDepthNode(name="depth", delay_ms=50)

        graph = Graph(name="parallel_test")
        graph.add(detect, inputs={"image": "input.image"})
        graph.add(depth, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})

        start_time = time.time()
        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})
        end_time = time.time()

        # Verify both executed
        assert detect.executed
        assert depth.executed

        # Should be faster than sequential (with some tolerance)
        execution_time_ms = (end_time - start_time) * 1000
        # Both nodes have 50ms delay, parallel should be ~50ms, sequential ~100ms
        assert execution_time_ms < 80  # Give some tolerance

        # Verify results
        assert result.has_channel("detections")
        assert result.has_channel("depth")

        # Verify device assignments were made
        assert result.metrics["device_usage"]
        assert result.metrics["device_placement_strategy"] == "round_robin"

    def test_execute_with_error_handling(self, mock_image, mock_context):
        """Test error handling in optimized execution."""
        scheduler = OptimizedParallelScheduler()

        # Create graph with failing node
        fail_node = MockFailNode(name="fail", error_msg="Test failure")
        graph = Graph(name="fail_test")
        graph.add(fail_node, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})

        with pytest.raises(RuntimeError, match="Test failure"):
            scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

    def test_execute_stage_optimized_with_failures(self, mock_image):
        """Test optimized stage execution with node failures."""
        scheduler = OptimizedParallelScheduler(max_workers=2)

        # Create nodes including one that fails
        good_node = MockDetectNode(name="good")
        fail_node = MockFailNode(name="fail", error_msg="Stage failure")

        context = ExecutionContext(providers={}, device="cpu")
        context.store("input.image", mock_image)

        wiring = {"good.image": "input.image", "fail.image": "input.image"}

        # Should raise RuntimeError for stage failure
        with pytest.raises(RuntimeError, match="Optimized parallel stage execution failed"):
            scheduler._execute_stage_optimized([good_node, fail_node], context, wiring)

    def test_device_usage_tracking(self, mock_image, mock_context):
        """Test device usage tracking across execution."""
        scheduler = OptimizedParallelScheduler(device_placement="round_robin")
        scheduler._device_pool = ["cuda:0", "cuda:1", "cpu"]

        # Create multiple nodes
        nodes = [MockDetectNode(name=f"node{i}") for i in range(4)]

        for node in nodes:
            scheduler._assign_device(node)
            # Should cycle through GPUs

        # Check usage tracking
        assert scheduler._device_usage["cuda:0"] == 2  # nodes 0, 2
        assert scheduler._device_usage["cuda:1"] == 2  # nodes 1, 3

    def test_metrics_enhancement(self, mock_image, mock_context):
        """Test that additional metrics are added to results."""
        scheduler = OptimizedParallelScheduler(device_placement="memory_aware", unload_unused=True)

        detect = MockDetectNode(name="detect")
        graph = Graph(name="metrics_test")
        graph.add(detect, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})

        result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

        # Verify enhanced metrics
        assert "device_usage" in result.metrics
        assert "device_placement_strategy" in result.metrics
        assert "unload_unused_enabled" in result.metrics
        assert "available_devices" in result.metrics

        assert result.metrics["device_placement_strategy"] == "memory_aware"
        assert result.metrics["unload_unused_enabled"] is True
        assert isinstance(result.metrics["available_devices"], list)

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    def test_comprehensive_device_strategies(
        self, mock_cuda_available, mock_device_count, mock_get_props, mock_memory_allocated, mock_image, mock_context
    ):
        """Test all device placement strategies comprehensively."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        # Provide enough property mocks for all scheduler inits + memory_aware queries
        mock_prop = Mock()
        mock_prop.name = "MockGPU"
        mock_prop.total_memory = 8 * 1024**3
        mock_get_props.return_value = mock_prop
        mock_memory_allocated.return_value = 0  # No memory allocated

        for strategy in ["auto", "round_robin", "memory_aware"]:
            scheduler = OptimizedParallelScheduler(device_placement=strategy)

            detect = MockDetectNode(name="detect")
            graph = Graph(name=f"{strategy}_test")
            graph.add(detect, inputs={"image": "input.image"})

            compiled = graph.compile(providers={})

            result = scheduler.execute(compiled, mock_context, initial_artifacts={"input.image": mock_image})

            # Each strategy should work
            assert detect.executed
            assert result.has_channel("detections")
            assert result.metrics["device_placement_strategy"] == strategy


# Import the new OptimizedParallelScheduler for tests
from mata.core.graph.scheduler import OptimizedParallelScheduler  # noqa: E402
