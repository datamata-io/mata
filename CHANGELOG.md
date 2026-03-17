# Changelog

All notable changes to MATA are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [2.0.0-beta] - 2026-03-17

### Added

**Training API**

- `mata.train(task, model, data, **kwargs)` — train or fine-tune detection, classification, and segmentation models from scratch or from a pretrained checkpoint
- `mata.finetune(task, model, data, **kwargs)` — fine-tuning shortcut with sensible defaults (`lr=1e-5`, `epochs=5`, `freeze_backbone=True`, `batch_size=16`)
- `from mata import train, finetune` — both exported at package level; lazy import keeps startup time unchanged
- `TrainingConfig` dataclass — centralised hyperparameter container with `validate()` and `from_yaml()` support; covers task, model, data paths, optimiser, scheduler, AMP, augmentation, checkpoint, and early-stopping settings
- `TrainingResult` dataclass — returned by all training calls; exposes `best_checkpoint`, `last_checkpoint`, `epochs_completed`, `history` (dict of metric lists), and `plot_loss()` helper
- `TrainingError` exception class added to `mata.core.exceptions` hierarchy

**Dataset Support**

- `COCODetectionDataset` — loads COCO-format JSON (train or val split); xyxy box conversion; crowd exclusion; YAML data-config support (`train:` split key)
- `COCOSegmentationDataset` — COCO instance masks as `(N, H, W)` binary tensors alongside detection targets
- `VOCDetectionDataset` — VOC 2007/2012 XML annotations; auto-discovers class names; optional `skip_difficult` flag
- `ImageFolderDataset` — classifies images from subdirectory structure; alphabetically sorted class names; hidden file/dir skip
- `detection_collate_fn` / `classification_collate_fn` — batch collators for the PyTorch DataLoader
- `DatasetFactory.build()` — auto-detects dataset format (COCO YAML, ImageFolder directory, or pass-through PyTorch `Dataset`) and returns the correct `(dataset, collate_fn)` pair

**Data Augmentation**

- `BasicDetectionAugmentation` — random horizontal flip with bbox mirroring, resize, colour jitter; torchvision-only (no extra dependencies)
- `BasicClassificationAugmentation` — train/val modes (random crop vs centre crop), ImageNet normalisation
- `BasicSegmentationAugmentation` — paired image + mask transforms; binary mask value preservation guaranteed
- `AlbumentationsWrapper` — wraps any albumentations pipeline with automatic xyxy ↔ pascal_voc bbox conversion; graceful `ImportError` when albumentations is not installed
- `AugmentationFactory.build()` — returns the correct augmentation for a task; routes custom dicts to the albumentations wrapper

**HuggingFace Training Engine**

- `HFTrainingEngine` — wraps HuggingFace `Trainer` for detection, classification, and segmentation tasks
- `_load_model_for_training()` — loads model in train mode with gradients enabled; no `.eval()` called
- `_build_training_args()` — maps `TrainingConfig` fields directly to `TrainingArguments`
- `_freeze_backbone()` / `_freeze_layers()` — selective parameter freezing by named-module prefix
- `_HistoryCallback` — HF `TrainerCallback` that captures per-epoch `train_loss` and validation metrics into `TrainingResult.history`

**Torchvision Training Engine**

- `TorchTrainingEngine` — custom training loop for `torchvision/` detection models (Faster R-CNN, RetinaNet, FCOS, SSD)
- `_modify_head()` — replaces the classification head for the target `num_classes`; supports FPN, anchor-free, and SSD architectures
- `_freeze_backbone()` — freezes backbone parameters, keeps the detection head trainable
- `_build_optimizer()` — AdamW (default), SGD, Adam selection
- `_build_scheduler()` — cosine, linear, step, or none schedulers
- AMP (`torch.cuda.amp`) enabled automatically on CUDA; disabled on CPU

**Checkpoint Manager**

- `CheckpointManager.save()` — writes `model_state.pth`, `optimizer_state.pth`, `training_state.json`, and `config.json` to a numbered epoch directory
- `CheckpointManager.load()` — restores all states with `weights_only=True` (security); raises `TrainingError` on corrupt or missing files
- `CheckpointManager.export_for_inference()` — HF models: calls `save_pretrained()`; torchvision models: writes `model.pth` + `metadata.json`; output directory is immediately loadable via `mata.load()`
- `CheckpointManager.list_checkpoints()` — returns sorted list of valid checkpoint directories

**Training Callbacks**

- `ValidationCallback` — fires `mata.val()` at configurable `val_every` intervals; restores model to `.train()` mode even on failure; returns `dict[str, float]` metrics
- `LoggingCallback` — prints a YOLO-style formatted table to console when `verbose=True`; writes to `{save_dir}/training.log` when `save_dir` is set; header printed once per run
- `EarlyStoppingCallback` — `mode="max"` (mAP, accuracy) or `mode="min"` (loss); `patience=0` disables; returns `True` to stop training

**UniversalLoader Checkpoint Support (`mata.load()` extension)**

- `_is_checkpoint_dir()` — detects a MATA training checkpoint directory by presence of `config.json` + `model_state.pth` / `model.safetensors`
- `_detect_source_type()` extended — new `"trained_checkpoint"` source type inserted after `"local_file"` check and before file-extension check (highest-priority local detection)
- `_load_from_checkpoint()` — reads `engine` field from `config.json`; routes HF checkpoints to `_load_from_huggingface(checkpoint_dir)` (uses `from_pretrained()`); routes torchvision checkpoints to `_load_from_torchvision()` + `torch.load(weights_only=True)` state-dict restore

**Optional Dependency Group**

- `pip install datamata[training]` — installs `albumentations>=1.3.0` and `tqdm>=4.65.0`; added to `all` and `dev` extras in `pyproject.toml`

**Documentation**

- `docs/TRAINING_GUIDE.md` — comprehensive guide: quickstart, dataset formats, data augmentation, full API reference, fine-tuning guide (backbone freezing, head replacement), checkpoint management, evaluation integration, HuggingFace vs Torchvision engine comparison, reload-and-deploy example, troubleshooting/FAQ
- `examples/train/finetune_detection.py` — fine-tune DETR on COCO dataset → evaluate → export → reload → predict
- `examples/train/finetune_classification.py` — fine-tune ResNet-50 on ImageFolder → evaluate → export → classify
- `examples/train/finetune_segmentation.py` — fine-tune Mask2Former on COCO segmentation → evaluate → export
- `examples/train/torchvision_finetune.py` — Faster R-CNN custom loop → train → export → reload → predict
- `examples/configs/training_detect.yaml` — annotated detection training config with all hyperparameters
- `examples/configs/training_classify.yaml` — annotated classification training config with all hyperparameters
- `QUICK_REFERENCE.md` — new "Training & Fine-Tuning (v2.0)" section added: quick-start snippets, supported/unsupported task table, key parameters table, fine-tuning defaults, checkpoint management one-liner, and supported model references
- README.md — Training & Fine-Tuning quick-start section added; Roadmap updated to reflect v2.0 completion; test count updated to 4,500+; Documentation section updated

### Changed

- `src/mata/api.py` — `train()` and `finetune()` appended after `val()`; lazy import of `mata.training` preserves startup time
- `src/mata/__init__.py` — `train` and `finetune` added to import line and `__all__`
- `src/mata/core/model_loader.py` — `_detect_source_type()` extended with checkpoint directory detection; `load()` dispatch chain updated
- `pyproject.toml` — `[training]` optional dependency group added; `all` and `dev` extras updated

### Tests

- 557 new tests across 10 test files: `test_training_config.py` (77), `test_training_datasets.py` (54), `test_training_augmentations.py` (32), `test_training_checkpoint.py` (52), `test_hf_trainer.py` (69+), `test_torch_trainer.py` (59+), `test_train_api.py` (55), `test_training_callbacks.py` (31), `test_training_integration.py` (19 slow-marked), `test_training_result.py` (new, 100% coverage for `result.py`)
- 414 tests from initial training implementation; 15 added in QA (unsupported-task error paths, Task B1); 128 added in QA (coverage gap tests for `result.py`, `trainer.py`, `hf_trainer.py`, `torch_trainer.py`, Task B3)
- All 4,307 pre-existing tests continue to pass with zero regressions; full fast suite: 4,950 passed

---

## [1.9.2] Beta Release - 2026-03-09

### Changed

**PyPI Distribution Rename**

- PyPI distribution name changed from `mata` to `datamata`; users now run `pip install datamata` to install
- `import mata` is unchanged — all existing code continues to work without modification
- Follows the PIL/Pillow precedent: distribution name and import name differ intentionally

### Added

**Valkey / Redis Graph Pipeline Storage**

- `export_valkey(result, url, key, ttl, serializer)` — serializes any MATA result type to a Valkey or Redis key with optional TTL; supports `json` (default) and `msgpack` serializers
- `load_valkey(url, key, result_type="auto")` — deserializes a stored result back to the original type; auto-detects `VisionResult`, `ClassifyResult`, `DepthResult`, and `OCRResult` from stored payload structure
- `publish_valkey(result, url, channel, serializer)` — fire-and-forget Pub/Sub broadcast; returns subscriber count
- `_parse_valkey_uri()` helper supporting `valkey://host:port/key`, `valkey://host:port/db/key`, and `redis://user:pass@host:port/db/key` formats
- `ValkeyStore` graph sink node — pass-through sink that writes an artifact to Valkey during graph execution; supports `{node}` and `{timestamp}` key template placeholders, TTL, and serializer selection
- `ValkeyLoad` graph source node — source node with `inputs={}` that loads a stored result from Valkey and injects it into the graph as a typed artifact
- `valkey://` and `redis://` URI scheme dispatch added to all six `result.save()` methods (`VisionResult`, `DetectResult`, `SegmentResult`, `ClassifyResult`, `DepthResult`, `OCRResult`) — existing file-based paths are fully unaffected
- `ModelRegistry.get_valkey_connection(name="default")` — reads named Valkey connection profiles from the `storage.valkey` section of `.mata/models.yaml` or `~/.mata/models.yaml`; resolves `password_env` from environment variables; raises `ModelNotFoundError` for unknown connection names
- YAML `storage.valkey.<name>` config schema with `url`, `db`, `ttl`, `password_env`, and `tls` fields
- Optional dependency groups: `pip install datamata[valkey]` → `valkey>=6.0.0`; `pip install datamata[redis]` → `redis>=5.0.0`; both added to the `dev` extras group
- `export_valkey`, `load_valkey`, and `publish_valkey` exported from `mata.core.exporters`
- `ValkeyStore` and `ValkeyLoad` exported from `mata.nodes`
- 89 new tests: 42 exporter tests (`test_valkey_exporter.py`), 33 graph node tests (`test_valkey_nodes.py`), 14 config and pub/sub tests (`test_valkey_config.py`)

**Documentation**

- `docs/VALKEY_GUIDE.md` — full integration guide covering installation, basic usage, graph pipeline integration, YAML configuration, streaming patterns, Pub/Sub architecture, security (TLS, `password_env`, SSRF prevention, key sanitization), performance tuning (serializer choice, TTL strategies, connection pooling, async patterns), and top-5 troubleshooting issues
- `docs/GRAPH_API_REFERENCE.md` — new "Storage Nodes" section with full parameter tables for `ValkeyStore` and `ValkeyLoad`
- `README.md` — Valkey added to Key Features list and Optional Dependencies table
- `QUICKSTART.md` — new "Valkey / Redis Result Storage" section with annotated code examples
- `QUICK_REFERENCE.md` — new "Valkey/Redis Storage Quick Reference (v1.9)" section with cheatsheet

**Appearance-Based ReID Tracking**

- `mata.track(..., reid_model="org/model")` — activate appearance-based re-identification for BotSort by supplying any HuggingFace image encoder ID or local `.onnx` path
- `ReIDAdapter` — abstract base class for appearance feature extractors; L2-normalised embedding output; lazy-loaded to keep startup cost zero when ReID is unused
- `HuggingFaceReIDAdapter` — ViT / CLIP / AutoModel architecture auto-detection (CLIP image encoder, ViT/DeiT/Swin/BEiT pooler output, generic AutoModel mean-pooling); all `transformers` imports lazy
- `ONNXReIDAdapter` — ONNX Runtime ReID extractor; auto-detects NCHW/NHWC input layout from model metadata; supports CPU and CUDA execution providers
- `TrackingAdapter.update()` now extracts detection crops, batch-encodes them through the ReID encoder, and injects embeddings into `BOTSORT` — activating the appearance distance branch in `get_dists()`; `Instance.embedding` populated in output `VisionResult`
- `mata.track()` extended with `reid_model: str | None` and `with_reid: bool = False` kwargs; `with_reid=True` without `reid_model` raises `ValueError`
- Config alias support: `reid_model` and `with_reid` keys can be declared in `.mata/models.yaml` under a `track:` alias; runtime kwargs always take precedence
- `ReIDBridge` — cross-camera appearance store backed by Valkey/Redis; publishes L2-normalised embeddings keyed by `reid:{camera_id}:{track_id}`; `query()` returns nearest matches above cosine-similarity threshold from other cameras; uses `scan_iter` (production-safe, non-blocking); TTL-based auto-eviction; `msgpack` binary serialisation
- `TrackingAdapter.__init__()` extended with `reid_bridge: ReIDBridge | None`; after each `update()` confirmed tracks with embeddings are published automatically; `ConnectionError` caught and logged, never raised
- `mata.track()` / `mata.load("track", ...)` extended with `reid_bridge` kwarg; forwarded to `TrackingAdapter`
- `ReIDAdapter`, `HuggingFaceReIDAdapter`, `ONNXReIDAdapter` exported from `mata.adapters`
- `ReIDBridge` exported from `mata.trackers`
- `src/mata/trackers/configs/botsort.yaml` — commented `reid_model` / `with_reid` documentation block added (v1.9.2+)
- 80+ new tests: `test_reid_adapter.py` (ReID adapter unit tests), `test_tracking_reid.py` (TrackingAdapter + API integration), `test_reid_bridge.py` (cross-camera bridge)
- `examples/track/reid_tracking.py` — basic single-camera ReID tracking example script
- `examples/track/cross_camera_reid.py` — cross-camera ReID via Valkey example script

**Documentation**

- `docs/VALKEY_GUIDE.md` — full integration guide covering installation, basic usage, graph pipeline integration, YAML configuration, streaming patterns, Pub/Sub architecture, security (TLS, `password_env`, SSRF prevention, key sanitization), performance tuning (serializer choice, TTL strategies, connection pooling, async patterns), and top-5 troubleshooting issues
- `docs/GRAPH_API_REFERENCE.md` — new "Storage Nodes" section with full parameter tables for `ValkeyStore` and `ValkeyLoad`
- `README.md` — Valkey added to Key Features list and Optional Dependencies table; ReID tracking section added with single-camera and cross-camera usage examples
- `QUICKSTART.md` — new "Valkey / Redis Result Storage" section with annotated code examples
- `QUICK_REFERENCE.md` — new "Valkey/Redis Storage Quick Reference (v1.9)" section with cheatsheet
- `docs/VALIDATION_GUIDE.md` — ReID tracking validation notes added

### Changed

- `mata.nodes.__all__` extended with `ValkeyStore` and `ValkeyLoad`
- `mata.core.exporters.__init__` extended with `export_valkey`, `load_valkey`, `publish_valkey`
- `mata.track()` signature extended with `reid_model`, `with_reid`, `reid_bridge` kwargs (backward-compatible defaults)
- `TrackingAdapter.__init__()` extended with `reid_encoder`, `reid_bridge` kwargs (both default to `None`; zero overhead when unused)
- `BOTSORT.get_dists()` appearance-distance branch now reachable when `encoder` is set via `reid_encoder`
- ByteTrack vs BotSort ReID comparison table in `README.md` updated to reflect v1.9.2 BotSort support

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
