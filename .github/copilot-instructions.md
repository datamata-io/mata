# MATA — AI Agent Instructions

## Architecture Overview

MATA is a **task-centric, model-agnostic** computer vision framework with a llama.cpp-inspired universal loader. As of v1.9.0, it features a unified adapter system supporting multiple tasks and runtimes, plus a fully vendored ByteTrack/BotSort tracking system and an OCR evaluation pipeline.

**Universal Loading (v1.5.2+):**

```python
mata.load("detect", "facebook/detr-resnet-50")  # HuggingFace ID
mata.load("classify", "./model.onnx")           # Local ONNX file
mata.load("segment", "fast-model")              # Config alias
mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
mata.load("track", "facebook/detr-resnet-50", tracker="botsort")  # 🆕 v1.8.0
```

**Object Tracking (v1.8.0):**

```python
# One-liner video/stream tracking
results = mata.track("video.mp4",
    model="facebook/detr-resnet-50",
    tracker="botsort",   # or "bytetrack"
    conf=0.3, save=True, show_track_ids=True)

# Stream mode (constant memory)
for result in mata.track("rtsp://cam/stream",
                          model="...", stream=True):
    ...

# Persistent per-frame tracking
tracker = mata.load("track", "facebook/detr-resnet-50", tracker="bytetrack")
result = tracker.update(frame, persist=True)  # YOLO-like pattern
```

**Zero-Shot Capabilities:**

```python
mata.run("classify", "image.jpg", model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog", "bird"])
mata.run("detect", "image.jpg", model="IDEA-Research/grounding-dino-tiny",
    text_prompts="cat . dog . person")
mata.run("segment", "image.jpg", model="facebook/sam3",
    text_prompts="cat")
```

### Core Architecture Layers

```
User API (mata.load/run/track/infer/val)
    ↓
UniversalLoader (5-strategy auto-detection)
    ↓
Task Adapters (HuggingFace/ONNX/TorchScript/PyTorch)
    ↓
VisionResult (Unified result: bbox + mask + track_id + embedding)
    ↓
Runtime (PyTorch/ONNX Runtime/TorchScript)
    ↓           ↓              ↓
Export System  Tracking Layer  Evaluation Layer (v1.8.1+)
(JSON/CSV/     (v1.8.0)        ↓
Image/Crops)   ↓               Validator (detect/segment/classify/depth/ocr)
               TrackingAdapter ↓
               ↓               Metrics (DetMetrics/SegMetrics/ClassifyMetrics/
               Vendored Trackers         DepthMetrics/OCRMetrics)  ← v1.9.0
               (BYTETracker/           ↓
               BOTSORT)              Printer + DatasetLoader
               ↓ (no external dep)   (COCO/COCO-Text JSON)
               KalmanFilter + IoU
               matching + GMC
```

**Key Design Pattern:** Task contracts over model specifics - all adapters implement the same `predict()` interface returning task-specific results (VisionResult for detect/segment, ClassifyResult, DepthResult).

### VLM Tool-Calling Agent System (v1.7.0)

**NEW in v1.7.0:** VLM nodes can now operate in **agent mode**, where they iteratively call specialized tools (detect, classify, segment, depth, zoom, crop) to gather information before providing final answers.

**Usage:**

```python
from mata.nodes import VLMQuery

# Agent mode - VLM can call tools iteratively
node = VLMQuery(
    using="qwen3-vl",
    prompt="Analyze this medical image.",
    tools=["detect", "classify", "zoom"],  # ← Agent mode enabled
    max_iterations=5,
    on_error="retry",
)
result = mata.infer(graph, image="...", providers={...})
```

**Architecture:**

```
VLM Nodes (VLMDetect/Query/Describe)
    ↓ (agent mode: tools=[...])
AgentLoop (iterative execution)
    ↓ (multi-turn conversation)
ToolRegistry (resolve tool names to providers)
    ↓ (dispatch)
Built-in Tools (zoom, crop) + Provider Tools (detect, classify, etc.)
```

**Key Components:**

- `agent_loop.py`: Core VLM ↔ tool iteration loop with retry logic
- `tool_schema.py`: Schema definitions (ToolSchema, ToolCall, ToolResult)
- `tool_registry.py`: Tool resolution and execution dispatch
- `tool_prompts.py`: System prompt generation for VLMs
- `parsers.py`: Tool call parsing (fenced blocks, XML, raw JSON)
- `image_tools.py`: Built-in zoom/crop tools

**Design Principles:**

- **Backward compatible**: VLM nodes without `tools=` work identically to before
- **DAG preserved**: Agent loop runs inside VLM node, not as graph scheduler
- **Provider-based tools**: Tool names reference provider dict keys
- **Safety caps**: `max_iterations` prevents runaway execution
- **Error modes**: retry (default), skip, fail
- **Observability**: Full tracing and metrics integration

**Testing:** 336 comprehensive tests in `test_tool_schema.py`, `test_tool_registry.py`, `test_agent_loop.py`, `test_tool_prompts.py`, `test_tool_call_parser.py`, etc. All passing with zero regressions.

**Documentation:** See `docs/VLM_TOOL_CALLING_SUMMARY.md` for complete architecture details, design decisions, limitations, and future roadmap.

## Development Workflows

### Running Tests

```bash
# All tests (4307+ total, all must pass)
pytest tests/ -v

# Task-specific test suites
pytest tests/test_classify_adapter.py -v   # Classification (35+ tests)
pytest tests/test_clip_adapter.py -v       # CLIP zero-shot (46 tests)
pytest tests/test_segment_adapter.py -v    # Segmentation (30+ tests)
pytest tests/test_sam_adapter.py -v        # SAM (19 tests)
pytest tests/test_depth_adapter.py -v      # Depth (10+ tests)
pytest tests/test_universal_loader.py -v   # Universal loader (17 tests)

# Object Tracking test suites (v1.8.0)
pytest tests/test_trackers/ -v             # Vendored trackers (354 tests)
pytest tests/test_tracking_adapter.py -v   # TrackingAdapter (73 tests)
pytest tests/test_track_api.py -v          # mata.track() public API (62 tests)
pytest tests/test_tracking_visualization.py -v # Visualization/export (103 tests)
pytest tests/test_video_io.py -v           # Video I/O utilities (56 tests)
pytest tests/test_track_node.py -v         # Track graph node (39 tests)

# VLM tool-calling test suites (v1.7.0)
pytest tests/test_tool_schema.py -v        # Tool schema (33 tests)
pytest tests/test_tool_registry.py -v      # Tool registry (44 tests)
pytest tests/test_agent_loop.py -v         # Agent loop (51 tests)
pytest tests/test_tool_prompts.py -v       # Tool prompts (18 tests)
pytest tests/test_tool_call_parser.py -v   # Tool call parser (51 tests)
pytest tests/test_image_tools.py -v        # Built-in tools (37 tests)
pytest tests/test_vlm_tool_calling.py -v   # Integration (12 tests)

# OCR evaluation test suite (v1.9.0)
pytest tests/test_eval_ocr.py -v           # OCR evaluation (71 tests)

# With coverage (target: >80%)
pip install pytest-cov
pytest --cov=mata --cov-report=html
```

### Testing New Task Adapters

When implementing new task adapters:

1. Inherit from appropriate base adapter in `adapters/base/`
2. Implement `predict()` returning task-specific result type (VisionResult, ClassifyResult, DepthResult)
3. Add comprehensive tests in `tests/test_<task>_adapter.py`
4. Test all supported runtimes (PyTorch, ONNX, TorchScript where applicable)
5. Use `NotImplementedError` with helpful messages for incomplete features

### Configuration Testing

```python
# Test config loading from both locations
import tempfile
from mata.core.model_registry import ModelRegistry

registry = ModelRegistry(
    user_config_path="~/.mata/models.yaml",
    project_config_path=".mata/models.yaml"
)
```

## Code Conventions

### Import Patterns

- **Lazy imports for adapters:** Avoid circular deps, speed up startup

  ```python
  def _load_from_huggingface(self, task, model_id, **kwargs):
      from mata.adapters.huggingface_adapter import HuggingFaceDetectAdapter
  ```

- **Type hints required:** Use `from __future__ import annotations` for forward refs

  ```python
  from __future__ import annotations
  from typing import Optional, Any

  def load(self, task: str, source: Optional[str] = None) -> Any:
  ```

### Result Type Patterns

All task results follow unified patterns:

**VisionResult (Detection & Segmentation):**

```python
@dataclass
class VisionResult:
    instances: list[Instance]  # Multi-modal: bbox + mask + embedding
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str: ...
    def to_dict(self) -> dict: ...
    def save(self, output_path: str, **kwargs): ...  # Auto-format detection

    # Segmentation-specific
    def get_instances(self) -> list[Instance]: ...
    def get_stuff(self) -> list[Instance]: ...
```

**ClassifyResult:**

```python
@dataclass
class ClassifyResult:
    classifications: list[Classification]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def top1(self) -> Classification: ...
    @property
    def top5(self) -> list[Classification]: ...

    def to_json(self) -> str: ...
    def save(self, output_path: str, **kwargs): ...
```

**DepthResult:**

```python
@dataclass
class DepthResult:
    depth_map: np.ndarray  # (H, W) float array
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, output_path: str, colormap: str = "magma"): ...
```

**Type Aliases (Backward Compatibility):**

- `DetectResult = VisionResult` # For detection tasks
- `SegmentResult = VisionResult` # For segmentation tasks

**Coordinate/Format Standards:**

- Bboxes: Always `xyxy` format (absolute pixel coords)
- Masks: RLE encoding (default), binary arrays, or polygon coordinates
- Depth: Float arrays normalized to [0, 1] or raw values

### Error Handling Strategy

```python
# Use specific exceptions from mata.core.exceptions
from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError

# Provide actionable error messages
raise ModelNotFoundError(
    f"Model '{source}' not found. "
    f"Check config: ~/.mata/models.yaml or .mata/models.yaml. "
    f"Available aliases: {self.registry.list_aliases(task)}"
)
```

### Logging Patterns

```python
from mata.core.logging import get_logger
logger = get_logger(__name__)

logger.info(f"Loading {task} model from {source_type}: {resolved_source}")
logger.warning(f"Using legacy plugin '{name}'. Deprecated in 2.0.")
logger.debug(f"Detected architecture: {arch}")
```

## UniversalLoader Detection Chain

**Critical:** Understand the 5-strategy detection order in `model_loader.py`:

```python
def _detect_source_type(self, task: str, source: Optional[str]) -> tuple[str, str]:
    """
    1. None → "default" (registry.get_default())
    2. Config alias (registry.has_alias()) → "config_alias"
    3. Local file (os.path.exists()) → "local_file" (.onnx/.pth/.pt/.bin/.engine)
    4. Contains '/' → "huggingface" (org/model pattern)
    5. Otherwise → "config_alias" (will raise ModelNotFoundError)
    """
```

**When adding new detection strategies:** Insert at appropriate priority, update tests.

## Configuration System

### YAML Config Structure

```yaml
models:
  detect:
    rtdetr-r18:
      source: "facebook/detr-resnet-50" # Can be HF ID, local path, or another alias
      threshold: 0.5
      device: "cuda"
      # Optional: architecture-specific params

    production-model:
      source: "rtdetr-r18" # Alias chaining supported
      threshold: 0.7

  # 🆕 v1.8.0: Tracking task support
  track:
    highway-cam:
      source: "facebook/detr-resnet-50"
      tracker: botsort # or bytetrack
      frame_rate: 30
      tracker_config:
        track_high_thresh: 0.6
        track_buffer: 60
```

### Config Precedence (Highest to Lowest)

1. Runtime kwargs: `mata.load("detect", "model", threshold=0.9)`
2. Project-local: `.mata/models.yaml`
3. User-global: `~/.mata/models.yaml`
4. Runtime registration: `mata.register_model()`
5. Adapter defaults

## Backward Compatibility Rules

**NOTE:** Plugin system was removed in v1.5.2. All migration is complete.

- All result types maintain backward compatibility via type aliases
- Old test coverage must remain stable
- Breaking changes will only occur in v2.0

## Common Pitfalls

1. **Circular imports:** Use lazy imports in loaders/adapters
2. **Config file not found:** Both paths are optional, handle gracefully
3. **Type hints with forward refs:** Use `from __future__ import annotations`
4. **Coordinate systems:** RT-DETR returns xyxy, ensure consistency in adapters
5. **Test isolation:** Use `registry._discovered = True` to skip auto-discovery in tests

## External Dependencies

- **HuggingFace Transformers:** For RT-DETR/DINO/DETR models
- **PyTorch:** Runtime execution (required)
- **ONNX Runtime:** Optional, for `.onnx` models
- **PyYAML:** Config file parsing (added in v1.5)

## Quick Reference Commands

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run single test file
pytest tests/test_universal_loader.py -v

# Test with specific Python
python3.10 -m pytest tests/

# Format code
black src/ tests/

# Type check
mypy src/mata/

# Check deprecation warnings
pytest tests/ -W default::DeprecationWarning
```

## When Contributing

After making changes:

1. Update docstrings in modified code
2. Add/update tests in `tests/test_*.py`
3. Update `README.md` for user-facing changes
4. Run `pytest tests/ -v` to ensure all tests pass
