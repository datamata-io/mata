# Changelog

All notable changes to MATA are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.9.1] - 2026-03-08

### Changed

- Refactored graph flow notation from `→` to `>` in all examples, scripts, and documentation for consistency with the DSL operator syntax
- Updated expected output structure descriptions in examples and docs to match the new `>` notation

### Added

- `ToolRegistry` now requires `text_prompts` for zero-shot providers (GroundingDINO, OWL-ViT, CLIP) and raises `ValueError` when they are missing
- Improved tool schema generation: zero-shot providers automatically include a `text_prompts` parameter in their generated `ToolSchema`
- Tests for zero-shot provider detection and `text_prompts` schema requirement in `test_tool_registry.py`

### Fixed

- SAM adapter: minor issue where prompt-less calls could silently produce empty masks instead of raising a clear error
- Video tracking examples: corrected frame iteration and output path handling in `examples/track/`

---

## [1.9.0] - 2026-03-02

### Added

**OCR / Text Extraction**

- `mata.run("ocr", image)` and `mata.load("ocr", backend)` API for text extraction
- Four OCR backends: EasyOCR, PaddleOCR, Tesseract, and HuggingFace (TrOCR + GOT-OCR2)
- `OCRResult` and `TextRegion` result types with bbox, text, confidence
- `BaseOCRTask` in `tasks/base.py` defining the `predict() → OCRResult` contract
- `EasyOCRAdapter`, `PaddleOCRAdapter`, `TesseractAdapter`, `HuggingFaceOCRAdapter` implementations
- `HuggingFaceOCRAdapter` auto-detects model family: TrOCR (VisionEncoderDecoderModel) vs GOT-OCR2 (AutoModelForCausalLM)
- Optional dependency groups: `[ocr]` (EasyOCR), `[ocr-paddle]` (PaddleOCR), `[ocr-tesseract]` (Tesseract), `[ocr-all]`

**OCR Evaluation**

- `OCRMetrics` — CER, WER, precision, recall, F1, NED for OCR evaluation
- COCO-Text dataset format support for OCR ground truth
- `mata.val("ocr", model, data=...)` validation API via `Validator`
- OCR results printed in YOLO-style console table via `Printer`
- 71 OCR evaluation tests (`tests/test_eval_ocr.py`)

**Public Release**

- Initial open-source release of MATA on GitHub
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- GitHub issue templates (bug report, feature request) and PR template
- Restructured documentation: renamed `artifacts` directory to `docs`
- Version bumped from `1.8.1` to `1.9.0`

---

## [1.8.1] - 2026-03-01

### Added

- `Validator._build_label_remap()` — maps predicted label IDs to ground-truth label IDs by class-name matching, supporting both raw COCO category-ID models (1-indexed, non-contiguous) and 0-indexed contiguous models
- Metrics JSON export in `Validator`: evaluation results (precision, recall, AP per class) are now saved to a `.json` file alongside existing CSV/plot outputs
- COCO validation dataset YAML config (`examples/configs/coco.yaml`) for `mata.val("detect", ..., data="coco.yaml")`
- DIODE validation dataset YAML config for `mata.val("depth", ..., data="diode.yaml")`
- ImageNet validation dataset YAML config for `mata.val("classify", ..., data="imagenet.yaml")`
- Depth metrics in dataset configuration and annotation generation scripts
- `docs/VALIDATION_GUIDE.md` — comprehensive guide covering evaluation, dataset formats, supported tasks, and metrics

### Fixed

- `Validator` prediction label remapping now uses `_build_label_remap()` consistently, correcting silent label-ID mismatches when evaluating COCO-pretrained models against 0-indexed ground truth

### Changed

- CI: removed `environment: release` gate from the `publish` job in `.github/workflows/publish.yml` — unblocks automated PyPI releases on version tag push
- Test file references updated from `TASK_VALIDATION_METRICS.md` to `VALIDATION_GUIDE.md` for consistency with the new documentation

---

## [1.8.0] - 2026-02-20

### Added

**Object Tracking (`mata.track()`)**

- `mata.track()` one-liner API for video and stream tracking
- `TrackingAdapter` composing any detection adapter with a stateful tracker
- Vendored ByteTrack algorithm (zero dependency on ultralytics/boxmot)
- Vendored BotSort algorithm with Kalman filter and GMC (global motion compensation)
- `mata.load("track", model, tracker="bytetrack"|"botsort")` for persistent trackers
- `persist=True` mode for stateful frame-by-frame tracking
- Stream mode (`stream=True`) for constant-memory video processing
- `Track` and `TrackResult` result types with track ID, bbox, score, label
- `ByteTrackWrapper`, `BotSortWrapper`, `SimpleIOUTracker` graph nodes
- Track trail rendering and track ID overlay visualization
- JSON and CSV export for tracking results
- `iter_frames()`, `VideoWriter`, and `detect_source_type()` video I/O utilities
- TrackerConfig dataclass with built-in YAML defaults for bytetrack/botsort
- 687 new tracking tests (354 vendored trackers + 73 adapter + 62 API + 103 visualization + 56 video I/O + 39 node)

**Validation Metrics (`mata.val()`)**

- `mata.val(task, model, data, ...)` YOLO-style validation API
- `DetMetrics`: `box.map`, `box.map50`, `box.map75`, `box.maps`, `box.mp`, `box.mr`, `speed`, `confusion_matrix`
- `SegmentMetrics`: extends `DetMetrics` with `.seg` namespace for mask AP
- `ClassifyMetrics`: `top1`, `top5`, `fitness`, `confusion_matrix`
- `DepthMetrics`: `abs_rel`, `sq_rel`, `rmse`, `log_rmse`, `delta_1`, `delta_2`, `delta_3`
- 101-point COCO AP implementation matching `pycocotools.COCOeval` within 0.01
- Dual validation mode: dataset-driven (YAML → COCO JSON) and standalone (predictions vs GT)
- COCO JSON dataset ingestion with xywh→xyxy conversion and 1-indexed category normalization
- Mask IoU supporting all three MATA mask formats (RLE dict, binary array, polygon)
- Detection and classification confusion matrices with `plot()` → PNG
- PR/F1/P/R curve plots (matplotlib, YOLO style) with per-class and mean lines
- YOLO-style per-class console table output
- Depth `.npy` ground-truth support
- 678 new eval tests; 81% coverage on `src/mata/eval/`

### Fixed

- Hardcoded `"mata-1.5.2"` version string in `nodes/fuse.py` — now reads dynamically from `importlib.metadata`
- `class RuntimeError(MATAError)` shadowed Python built-in — renamed to `MATARuntimeError` (alias kept for backward compatibility)
- Dead links in `README.md` (migration guide, basic detection example, troubleshooting)
- Deprecated `.detections` API in `QUICKSTART.md` — updated to `.instances` throughout
- `pyproject.toml` misconfiguration where `dependencies` array was under `[project.urls]`

### Removed

- `PluginNotFoundError` from public API (plugin system removed in v1.5.2; `ModelNotFoundError` is the replacement)
- Plugin-related messaging from `verify_install.py`

---

## [1.7.0] - 2026-02-16

### Added

- VLM tool-calling agent system: VLM nodes can now call vision tools iteratively before answering
- `tools=[...]` parameter on `VLMQuery`, `VLMDetect`, `VLMDescribe` nodes to enable agent mode
- `AgentLoop` — iterative VLM ↔ tool execution loop with configurable `max_iterations`
- `ToolRegistry` — resolves tool names to provider dict entries or built-in tools
- `ToolSchema`, `ToolCall`, `ToolResult` dataclasses for typed tool interchange
- Built-in `zoom` and `crop` image tools (no provider needed)
- System prompt generation for VLMs describing available tools
- Tool call parsing supporting fenced blocks, XML tags, and raw JSON formats
- `on_error` modes: `retry` (default), `skip`, `fail`
- Full tracing and metrics integration for agent loop iterations
- 253 new VLM tool-calling tests across 7 test files

---

## [1.6.0] - 2026-02-12

### Added

- Graph execution system: DAG-based multi-task vision pipelines
- `mata.infer(graph, image, providers={...})` API for graph execution
- 23 built-in node types (detect, classify, segment, depth, VLM, fuse, conditionals, etc.)
- 10+ industry preset graphs (medical, surveillance, retail, manufacturing, autonomous driving)
- DSL for graph construction with operator overloading
- Parallel node execution for independent branches
- Observability: tracing, metrics, spans, execution context
- `ConditionalNode` for branch selection based on upstream results
- `FuseNode` for merging multi-branch results
- VLM nodes: `VLMDetect`, `VLMQuery`, `VLMDescribe` with HuggingFace VLM support
- SAM3 zero-shot segmentation with text prompt support
- Full segmentation support: instance and panoptic (Mask2Former, MaskFormer, OneFormer)
- Three mask formats: RLE, binary numpy arrays, polygon coordinates
- Dual visualization backends: PIL (default) and matplotlib
- 30 segmentation tests (100% pass rate)

---

## [1.5.2] - 2026-02-02

### Removed

- Legacy plugin system — `PluginNotFoundError`, plugin auto-discovery, plugin registration
- Plugin entry-point scanning at import time
- `plugin_name` parameter from `mata.load()`

### Added

- `ModelType` enum for unambiguous `.pt` file disambiguation (TorchScript vs PyTorch checkpoint)

### Fixed

- `.pt` files no longer misidentified — `model_type=ModelType.TORCHSCRIPT` forces TorchScript loading

---

## [1.5.1] - 2026-01-31

### Added

- TorchScript model support: load and run inference with `.pt` TorchScript models
- `TorchScriptDetectAdapter` for JIT-compiled detection models
- Auto-detection of TorchScript vs PyTorch checkpoint format for `.pt` files
- RT-DETRv4 TorchScript variants (s/m/l/x) support
- Config aliases for `rtv4_s/m/l/x` in `examples/models.yaml`
- `inference_torchscript.py` complete MATA-integrated example

---

## [1.5.0] - 2026-01-30

### Added

- Universal model loading (`mata.load`) — llama.cpp-inspired 5-strategy auto-detection:
  1. `None` → default model for task
  2. Config alias → `.mata/models.yaml` / `~/.mata/models.yaml`
  3. Local file → extension detection (`.onnx`, `.pth`, `.pt`, `.bin`, `.engine`)
  4. Contains `/` → HuggingFace Hub ID
  5. Fallback → `ModelNotFoundError` with helpful message
- `UniversalLoader` in `src/mata/core/model_loader.py`
- `ModelRegistry` with two-tier YAML config (project-local overrides user-global)
- `mata.register_model()` for runtime model registration without config files
- `HuggingFaceDetectAdapter` supporting DETR, RT-DETR, DINO, Conditional DETR, YOLOS
- `HuggingFaceZeroShotDetectAdapter` for GroundingDINO and OWL-ViT
- `TorchvisionDetectAdapter` for RetinaNet, Faster R-CNN, FCOS, SSD
- `ONNXDetectAdapter` for generic ONNX Runtime models
- `HuggingFaceClassifyAdapter`, `ONNXClassifyAdapter`, `TorchScriptClassifyAdapter`
- `CLIPAdapter` for zero-shot image classification via text prompts
- `HuggingFaceSegmentAdapter` (Mask2Former foundation)
- `SAMAdapter` / `SAM3Adapter` for zero-shot segmentation
- `DepthAnythingAdapter` / `DepthAnythingV2Adapter`
- `VisionResult` unified result type with `instances: list[Instance]` (bbox xyxy, mask, score, label, embedding)
- `ClassifyResult` with `.top1`, `.top5` accessors
- `DepthResult` with `depth_map: np.ndarray`
- `DetectResult = VisionResult` and `SegmentResult = VisionResult` type aliases
- Export system: JSON, CSV, image overlay, crops (`.save(path)` with auto-format detection)
- **pyyaml** added as required dependency for config file support
- Zero-shot API: `mata.run(task, image, model=..., text_prompts=[...])`

### Changed

- `mata.load()` signature: `load(task, source=None, **kwargs)` replaces plugin-name parameter

### Deprecated

- Plugin-based model registration (removed in v1.5.2)

---

_For older history, see git log._
