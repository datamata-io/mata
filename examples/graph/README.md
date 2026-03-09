# MATA Graph System Examples

Examples demonstrating the MATA v1.6 multi-task graph system.

## Quick Start

```bash
# Install MATA with dev dependencies
pip install -e ".[dev]"

# Run any example
python examples/graph/simple_pipeline.py
```

## Core Examples (6)

These examples demonstrate the fundamental graph system capabilities:

| Example                                           | Description                                  | Key Features                                                       |
| ------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| ✅ [simple_pipeline.py](simple_pipeline.py)       | Detection > Filter > Segmentation > Fuse     | `mata.infer()`, `Graph.then()`, basic pipeline                     |
| [parallel_tasks.py](parallel_tasks.py)            | Parallel detection + classification + depth  | `Graph.parallel()`, `ParallelScheduler`, speedup                   |
| [video_tracking.py](video_tracking.py)            | Video processing with object tracking        | `VideoProcessor`, `Track`, frame policies                          |
| [vlm_workflows.py](vlm_workflows.py)              | VLM grounded detection & scene understanding | `VLMDetect`, `PromoteEntities`, VLM presets                        |
| [presets_demo.py](presets_demo.py)                | Using pre-built graph presets                | `grounding_dino_sam()`, `full_scene_analysis()`                    |
| [valkey_pipeline.py](valkey_pipeline.py)          | Valkey/Redis result storage & Pub/Sub        | `ValkeyStore`, `ValkeyLoad`, `publish_valkey`, rolling stream keys |

> **Advanced patterns** (custom nodes, conditional logic, provider integration) are documented with
> full context in the [Graph Cookbook](../../docs/GRAPH_COOKBOOK.md).

## Real-World Scenario Examples

The `scenarios/` subdirectory contains 20 industry-specific examples demonstrating production-ready workflows:

| Industry               | Scenarios                                                     | Example Scripts                              |
| ---------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **Manufacturing**      | Defect detection, assembly verification, component inspection | [scenarios/manufacturing\_\*.py](scenarios/) |
| **Retail**             | Shelf analysis, product search, stock assessment              | [scenarios/retail\_\*.py](scenarios/)        |
| **Autonomous Driving** | Distance estimation, scene analysis, traffic tracking         | [scenarios/driving\_\*.py](scenarios/)       |
| **Security**           | Crowd monitoring, suspicious object detection                 | [scenarios/security\_\*.py](scenarios/)      |
| **Agriculture**        | Disease classification, aerial crop analysis, pest detection  | [scenarios/agriculture\_\*.py](scenarios/)   |
| **Healthcare**         | ROI segmentation, report generation, pathology triage         | [scenarios/medical\_\*.py](scenarios/)       |

Each scenario includes:

- Comprehensive problem statement and use case documentation
- Mock mode (no model downloads) and real mode (actual inference)
- Production-ready graph configurations with industry-specific parameters
- Clear model requirements and hardware recommendations

See [Real-World Scenarios Guide](../../docs/REAL_WORLD_SCENARIOS.md) for detailed documentation.

## Running Without Real Models

All examples include a **mock mode** that runs with synthetic data, so you can explore
the graph API without downloading large models. Each example also shows the **real mode**
code that uses actual HuggingFace models.

```python
# Mock mode (default) — runs immediately, no models needed
python examples/graph/simple_pipeline.py

# Real mode — requires model downloads
python examples/graph/simple_pipeline.py --real
```

## Architecture Overview

```
Image input
    ↓
Graph (sequence of Nodes)
    ↓
mata.infer(image, graph, providers)
    ↓
Scheduler (Sync or Parallel)
    ↓
MultiResult (channels: detections, masks, depth, ...)
```

## Learn More

- [Graph System Guide](../../docs/GRAPH_SYSTEM_GUIDE.md)
- [API Reference](../../docs/GRAPH_API_REFERENCE.md)
- [Cookbook](../../docs/GRAPH_COOKBOOK.md) — advanced patterns: custom nodes, conditional logic, provider integration
