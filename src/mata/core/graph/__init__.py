"""Graph system for MATA - task orchestration and execution.

This module provides the graph system components for building and executing
multi-task computer vision workflows.
"""

# Re-export observability for convenience
from mata.core.observability import ExecutionTracer, MetricsCollector, ProvenanceTracker, Span

from .conditionals import CountAbove, HasLabel, If, Pass, ScoreAbove, count_above, has_label, score_above
from .context import ExecutionContext
from .dsl import NodePipe, bind, out, parallel_tasks, pipeline, sequential
from .graph import CompiledGraph, Graph
from .node import Node
from .scheduler import OptimizedParallelScheduler, ParallelScheduler, SyncScheduler
from .temporal import (
    FramePolicy,
    FramePolicyEveryN,
    FramePolicyLatest,
    FramePolicyQueue,
    VideoProcessor,
    Window,
)
from .validator import GraphValidator, ValidationResult

__all__ = [
    "Node",
    "ExecutionContext",
    "GraphValidator",
    "ValidationResult",
    "Graph",
    "CompiledGraph",
    # Schedulers
    "SyncScheduler",
    "ParallelScheduler",
    "OptimizedParallelScheduler",
    # DSL helpers
    "NodePipe",
    "out",
    "bind",
    "sequential",
    "parallel_tasks",
    "pipeline",
    # Conditional execution
    "If",
    "Pass",
    "HasLabel",
    "CountAbove",
    "ScoreAbove",
    "has_label",
    "count_above",
    "score_above",
    # Temporal / Video
    "FramePolicy",
    "FramePolicyEveryN",
    "FramePolicyLatest",
    "FramePolicyQueue",
    "VideoProcessor",
    "Window",
    # Observability
    "MetricsCollector",
    "ExecutionTracer",
    "Span",
    "ProvenanceTracker",
]
