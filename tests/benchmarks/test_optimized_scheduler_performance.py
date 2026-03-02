"""Performance benchmarks for OptimizedParallelScheduler.

Benchmarks comparing optimized scheduler against standard schedulers for:
- Sequential vs parallel execution performance
- Device placement strategies
- Memory usage with model unloading
- Multi-GPU distribution
"""

import time
from unittest.mock import Mock, patch

import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph.context import ExecutionContext
from mata.core.graph.graph import Graph
from mata.core.graph.node import Node
from mata.core.graph.scheduler import OptimizedParallelScheduler, ParallelScheduler, SyncScheduler


class MockComputeNode(Node):
    """Mock node with configurable compute time."""

    inputs = {"image": Image}
    outputs = {"result": Detections}

    def __init__(self, name: str, compute_ms: float = 100):
        super().__init__(name=name)
        self.compute_ms = compute_ms
        self.executed = False
        self.device_used = None

    def run(self, ctx, **inputs):
        """Simulate compute work."""
        self.executed = True
        self.device_used = getattr(ctx, "device", "unknown")

        # Simulate compute time
        time.sleep(self.compute_ms / 1000.0)

        # Create mock result
        from mata.core.types import Instance

        instances = [Instance(bbox=(0, 0, 10, 10), label=0, label_name="test", score=0.9)]
        result = Detections(instances=instances, instance_ids=["test_0000"])

        return {"result": result}


@pytest.fixture
def mock_image():
    """Create mock image for benchmarks."""
    from PIL import Image as PILImage

    pil_img = PILImage.new("RGB", (224, 224), color="blue")
    return Image.from_pil(pil_img)


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    return ExecutionContext(providers={}, device="cpu")


class TestOptimizedSchedulerBenchmarks:
    """Performance benchmark tests for OptimizedParallelScheduler."""

    def test_sequential_vs_parallel_performance(self, mock_image, mock_context):
        """Benchmark sequential vs parallel execution."""
        # Create nodes with compute time
        nodes = [MockComputeNode(f"node_{i}", compute_ms=50) for i in range(4)]

        # Build parallel graph
        graph = Graph(name="parallel_benchmark")
        for node in nodes:
            graph.add(node, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})
        initial_artifacts = {"input.image": mock_image}

        # Benchmark SyncScheduler (sequential)
        sync_scheduler = SyncScheduler()
        start_time = time.time()
        sync_result = sync_scheduler.execute(compiled, mock_context, initial_artifacts)
        sync_time = time.time() - start_time

        # Reset nodes
        for node in nodes:
            node.executed = False

        # Benchmark OptimizedParallelScheduler
        opt_scheduler = OptimizedParallelScheduler(max_workers=4)
        start_time = time.time()
        opt_result = opt_scheduler.execute(compiled, mock_context, initial_artifacts)
        opt_time = time.time() - start_time

        # Verify both complete successfully
        assert all(node.executed for node in nodes)
        assert sync_result.metrics["total_time_ms"] > 0
        assert opt_result.metrics["total_time_ms"] > 0

        # Parallel should be significantly faster for independent nodes
        # 4 nodes * 50ms = ~200ms sequential vs ~50ms parallel
        speedup = sync_time / opt_time
        print(f"Sequential: {sync_time:.3f}s, Parallel: {opt_time:.3f}s, Speedup: {speedup:.2f}x")

        # Should get some speedup (at least 1.5x with 4 parallel nodes)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"

    def test_device_placement_performance(self, mock_image, mock_context):
        """Benchmark different device placement strategies."""
        strategies = ["auto", "round_robin", "memory_aware"]

        # Create nodes
        nodes = [MockComputeNode(f"node_{i}", compute_ms=30) for i in range(3)]

        # Build graph
        graph = Graph(name="device_benchmark")
        for node in nodes:
            graph.add(node, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})
        initial_artifacts = {"input.image": mock_image}

        execution_times = {}

        for strategy in strategies:
            # Reset nodes
            for node in nodes:
                node.executed = False
                node.device_used = None

            scheduler = OptimizedParallelScheduler(max_workers=3, device_placement=strategy, unload_unused=False)

            start_time = time.time()
            result = scheduler.execute(compiled, mock_context, initial_artifacts)
            execution_time = time.time() - start_time

            execution_times[strategy] = execution_time

            # Verify execution
            assert all(node.executed for node in nodes)
            assert result.metrics["device_placement_strategy"] == strategy

            print(f"Strategy '{strategy}': {execution_time:.3f}s")

        # All strategies should complete in reasonable time
        for strategy, exec_time in execution_times.items():
            assert exec_time < 1.0, f"Strategy '{strategy}' took too long: {exec_time:.3f}s"

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_multi_gpu_distribution_benchmark(self, mock_device_count, mock_cuda_available, mock_image, mock_context):
        """Benchmark multi-GPU distribution."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        # Create multiple compute nodes
        nodes = [MockComputeNode(f"gpu_node_{i}", compute_ms=40) for i in range(6)]

        graph = Graph(name="multi_gpu_benchmark")
        for node in nodes:
            graph.add(node, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})
        initial_artifacts = {"input.image": mock_image}

        # Benchmark round-robin distribution
        scheduler = OptimizedParallelScheduler(max_workers=6, device_placement="round_robin", unload_unused=False)

        start_time = time.time()
        result = scheduler.execute(compiled, mock_context, initial_artifacts)
        execution_time = time.time() - start_time

        # Verify execution
        assert all(node.executed for node in nodes)
        assert len(result.metrics["device_usage"]) > 0

        print(f"Multi-GPU execution: {execution_time:.3f}s")
        print(f"Device usage: {result.metrics['device_usage']}")

        # Should complete efficiently
        assert execution_time < 1.0

    def test_memory_optimization_benchmark(self, mock_image, mock_context):
        """Benchmark memory optimization with model unloading."""

        class MockProviderNode(MockComputeNode):
            """Node that simulates having a provider with model."""

            def __init__(self, name: str):
                super().__init__(name, compute_ms=20)
                self.provider_name = f"provider_{name}"
                self.model_unloaded = False

                # Simulate provider with model
                self.mock_provider = Mock()
                self.mock_provider.model = Mock()

        # Create nodes
        nodes = [MockProviderNode(f"model_node_{i}") for i in range(4)]

        graph = Graph(name="memory_benchmark")
        for node in nodes:
            graph.add(node, inputs={"image": "input.image"})

        # Provide mock providers matching the nodes' provider_name attributes
        providers = {node.provider_name: node.mock_provider for node in nodes}
        compiled = graph.compile(providers=providers)
        initial_artifacts = {"input.image": mock_image}

        # Test with unloading enabled
        scheduler_unload = OptimizedParallelScheduler(max_workers=4, unload_unused=True)

        start_time = time.time()
        result_unload = scheduler_unload.execute(compiled, mock_context, initial_artifacts)
        time_with_unload = time.time() - start_time

        # Reset nodes
        for node in nodes:
            node.executed = False

        # Test with unloading disabled
        scheduler_keep = OptimizedParallelScheduler(max_workers=4, unload_unused=False)

        start_time = time.time()
        result_keep = scheduler_keep.execute(compiled, mock_context, initial_artifacts)
        time_without_unload = time.time() - start_time

        # Both should complete successfully
        assert all(node.executed for node in nodes)
        assert result_unload.metrics["unload_unused_enabled"] is True
        assert result_keep.metrics["unload_unused_enabled"] is False

        print(f"With unloading: {time_with_unload:.3f}s")
        print(f"Without unloading: {time_without_unload:.3f}s")

        # Both should complete in reasonable time
        assert time_with_unload < 1.0
        assert time_without_unload < 1.0

    def test_scalability_benchmark(self, mock_image, mock_context):
        """Benchmark scalability with increasing number of nodes."""
        node_counts = [2, 4, 8]
        execution_times = {}

        for count in node_counts:
            # Create nodes
            nodes = [MockComputeNode(f"scale_node_{i}", compute_ms=25) for i in range(count)]

            graph = Graph(name=f"scale_benchmark_{count}")
            for node in nodes:
                graph.add(node, inputs={"image": "input.image"})

            compiled = graph.compile(providers={})
            initial_artifacts = {"input.image": mock_image}

            scheduler = OptimizedParallelScheduler(max_workers=min(count, 8), device_placement="auto")  # Cap workers

            start_time = time.time()
            result = scheduler.execute(compiled, mock_context, initial_artifacts)
            execution_time = time.time() - start_time

            execution_times[count] = execution_time

            # Verify execution
            assert all(node.executed for node in nodes)
            # All nodes produce unqualified output "result" (last writer wins),
            # so channels count is 1, but all nodes did execute
            assert len(result.channels) >= 1

            print(f"Nodes: {count}, Time: {execution_time:.3f}s")

        # Execution time shouldn't increase drastically with more parallel work
        # (since they run in parallel)
        max_time = max(execution_times.values())
        assert max_time < 1.0, f"Scalability failed: max time {max_time:.3f}s"

    def test_overhead_comparison(self, mock_image, mock_context):
        """Compare overhead of optimization features."""
        # Single fast node to measure scheduler overhead
        node = MockComputeNode("overhead_test", compute_ms=10)

        graph = Graph(name="overhead_benchmark")
        graph.add(node, inputs={"image": "input.image"})

        compiled = graph.compile(providers={})
        initial_artifacts = {"input.image": mock_image}

        # Test different schedulers
        schedulers = {
            "sync": SyncScheduler(),
            "parallel": ParallelScheduler(max_workers=1),
            "optimized_minimal": OptimizedParallelScheduler(
                max_workers=1, device_placement="auto", unload_unused=False
            ),
            "optimized_full": OptimizedParallelScheduler(
                max_workers=1, device_placement="memory_aware", unload_unused=True
            ),
        }

        results = {}

        for name, scheduler in schedulers.items():
            node.executed = False  # Reset

            start_time = time.time()
            scheduler.execute(compiled, mock_context, initial_artifacts)
            execution_time = time.time() - start_time

            results[name] = execution_time
            assert node.executed

            print(f"Scheduler '{name}': {execution_time:.3f}s")

        # Overhead should be minimal - all should complete quickly
        for name, exec_time in results.items():
            assert exec_time < 0.5, f"Scheduler '{name}' has excessive overhead: {exec_time:.3f}s"

        # Optimized scheduler shouldn't be significantly slower than basic scheduler
        overhead_ratio = results["optimized_full"] / results["sync"]
        assert overhead_ratio < 2.0, f"Optimization overhead too high: {overhead_ratio:.2f}x"


if __name__ == "__main__":
    # Run benchmarks if called directly
    pytest.main([__file__, "-v", "-s"])
