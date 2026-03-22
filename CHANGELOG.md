# Changelog

All notable changes to MATA are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.9.5] - 2026-03-22

### Added

**First-Class CLI — `mata` command-line interface**

- `mata run <task> <image> [--model] [--conf] [--text] [--prompt] [--save] [--json]` — one-shot inference with Ultralytics-parity DX
- `mata recognize <image> --gallery <file.npz> [--model] [--top-k] [--threshold] [--json]` — gallery-based identity matching from the command line
- `mata track <video> [--model] [--tracker] [--conf] [--iou] [--reid-model] [--save] [--show]` — multi-object tracking
- `mata val <task> --data <yaml> [--model] [--conf] [--iou] [--plots] [--json]` — dataset evaluation
- `mata export <task> <model> [--format] [--quantize]` — stub: full support in v2.0
- `mata --version` — version display
- `mata <cmd> --help` — per-command help text
- `[project.scripts] mata = "mata.cli:main"` entrypoint in `pyproject.toml` (was already registered; now with `recognize` routing)

**Gallery Matching / Recognition — `mata.run("recognize", ...)`**

- `mata.run("recognize", image, gallery=gallery)` — convenience API: embeds the image, runs cosine similarity search against a `Gallery`, returns a `Matches` artifact
- `gallery` kwarg (required): pre-populated `Gallery` instance
- `top_k` kwarg (default: 1): maximum number of top matches to return
- `threshold` kwarg (optional): minimum cosine similarity (overrides gallery default)
- `Matches` and `MatchEntry` artifacts are now exported from `mata` top-level and from `mata.core.artifacts`
- Pipeline pattern: `Detect >> ExtractROIs >> Embed >> GalleryMatchNode(gallery=gallery)` — unchanged; `mata.run("recognize", ...)` wraps the single-image case
- `_run_recognize()` internal helper in `api.py` — handles image loading, adapter dispatch, gallery search, and result assembly

### Notes

- `mata.run("recognize", ...)` is the single-image convenience form; for per-ROI recognition in graphs, use `GalleryMatchNode` directly
- Zero regressions; all 5279+ pre-existing tests pass

### Tests

- `tests/test_matches_artifact.py` — 39 new tests for `Matches` and `MatchEntry` artifacts
- `tests/test_recognize_api.py` — 34 new tests for `mata.run("recognize", ...)` API
- `tests/test_cli_recognize.py` — 18 new tests for `mata recognize` CLI subcommand
- Total: **5201 + 78 new tests = 5279+ passing**

---

## [1.9.4] - 2026-03-21

### Added

**Notebook Integration — JupyterLab / Jupyter rich display for all result types**

- `_repr_html_()` on `VisionResult` — HTML table with label / score / bbox / track ID columns; optional base64-embedded image overlay when `meta["input_path"]` is set; truncates to 20 rows for results with >100 instances
- `_repr_html_()` on `ClassifyResult` — horizontal SVG bar chart (top-5) + score table
- `_repr_png_()` on `DepthResult` — magma-colormap PNG rendered via matplotlib
- `_repr_html_()` on `OCRResult` — text region table (text / score / bbox / label)
- `_repr_html_()` on `BarcodeResult` — decoded barcode table (data / type / score / bbox)
- `_repr_html_()` on `Embeddings` artifact — shape / dtype / normalized / instance ID summary
- `mata.show(result, image=None, **kwargs)` — explicit display utility; calls `IPython.display.display()` with HTML or PNG; falls back to `IPython.display.display(result)` for unknown types
- New `src/mata/notebook.py` module — all `render_*()` functions; all imports lazy-guarded; never breaks `import mata` without Jupyter installed
- `[notebook]` optional dependency group: `pip install datamata[notebook]` (installs `ipython>=7.0`, `matplotlib>=3.5.0`)
- `[all]` extras group now includes `notebook`
- Example notebooks: `examples/notebooks/01_detection.ipynb` through `06_vlm_query.ipynb`
- `.gitattributes` with `*.ipynb filter=nbstripout` to strip cell outputs on commit

### Notes

- All notebook display is fully optional — `import mata` works without IPython or matplotlib
- All user content is HTML-escaped (XSS-safe)
- `frozen=True` dataclasses unaffected — only methods added, no field mutations
- 50+ new tests in `tests/test_notebook.py`

### Tests

- New tests in `test_universal_loader.py`
- 5082 pre-existing tests pass with zero regressions (11 skipped)

---

## [1.9.3] - 2026-03-19

### Added

**Barcode & QR Code Detection — `barcode` task**

- `mata.run("barcode", image, model="pyzbar")` and `mata.load("barcode", "pyzbar")` — new first-class barcode/QR decoding task
- `BarcodeRegion` — frozen dataclass for a single decoded symbol: `data`, `type`, `bbox` (xyxy), `score`, `raw_bytes`
- `BarcodeResult` — frozen dataclass aggregating all decoded barcodes; supports `to_json()`, `from_json()`, `to_dict()`, `save()`, `filter_by_type()`
- `PyzbarAdapter` — primary barcode engine wrapping `pyzbar` (libzbar); MIT license; ~2 ms decode; supports 12+ symbologies (QR_CODE, EAN_13, EAN_8, UPC_A, UPC_E, CODE_128, CODE_39, CODE_93, ITF, CODABAR, DATA_MATRIX, PDF_417, AZTEC)
- `ZxingAdapter` — secondary barcode engine wrapping `zxing-cpp`; Apache 2.0; broader symbology support including Aztec and MaxiCode; 4-corner position → xyxy bbox
- Both adapters use lazy imports — zero overhead when the `barcode` task is unused
- `ModelType.PYZBAR` and `ModelType.ZXING` enum entries for explicit engine selection
- `BarcodeData` graph artifact — frozen, wraps `BarcodeResult` for typed graph wiring; `from_barcode_result()` factory; `instance_ids` correlation for ROI pipelines; `to_dict()` / `to_json()` serialization
- `BarcodeEntry` — frozen dataclass for a single barcode within the graph artifact
- `Barcode` graph node — accepts `Image` or `ROIs` inputs, produces `BarcodeData`; records `latency_ms` and `num_barcodes` metrics; supports `Detect >> ExtractROIs >> Barcode` composition
- VLM tool schema: `schema_for_task("barcode")` returns a valid `ToolSchema`; VLM agents with `tools=["barcode"]` can invoke barcode reading
- Optional dependency groups: `pip install datamata[barcode]` (pyzbar), `pip install datamata[barcode-zxing]` (zxing-cpp), `pip install datamata[barcode-all]`
- `BarcodeData` and `BarcodeEntry` exported from `mata.core.artifacts`
- `Barcode` exported from `mata.nodes`
- `PyzbarAdapter` and `ZxingAdapter` exported from `mata.adapters`
- 123 new tests: `test_barcode_adapter.py` (58), `test_barcode_node.py` (40), `test_barcode_integration.py` (25)
- `examples/barcode/basic_scan.py` — one-shot barcode scan example
- `examples/barcode/README.md` — barcode examples overview

**Multi-VLM Model Support**

- `HuggingFaceVLMAdapter` now accepts a `dtype` constructor kwarg (default `"auto"`); pass `dtype="bfloat16"` for MedGemma, LFM2.5-VL, and other models that require an explicit torch dtype
- `HuggingFaceVLMAdapter` now accepts a `trust_remote_code` constructor kwarg (default `False`); required for Florence-2, Phi-3.5 Vision, InternVL2, Moondream2, and other community models with custom code
- `_scale_bbox_from_vlm()` generalized with an auto-detection heuristic: coordinates `< 2.0` → `[0, 1]` normalized; `< 1500` → Qwen3-VL ~1000-unit space; otherwise raw pixel passthrough — eliminates the hardcoded Qwen3-VL assumption and enables correct bbox scaling for all VLM families
- `_is_vlm_model()` expanded with detection patterns for `medgemma`, `lfm.*vl`, `smolvlm`, `moondream2`
- 9 VLM model families now supported via the unified `AutoModelForImageTextToText` adapter: Qwen3-VL, MedGemma, LFM2.5-VL, SmolVLM, Florence-2, PaliGemma 2, Phi-3.5 Vision, LLaVA-NeXT, Moondream2 — see `docs/VLM_MODEL_SUPPORT.md` for the full compatibility table
- `mata.load("vlm", "google/medgemma-1.5-4b-it", dtype="bfloat16")` and `mata.load("vlm", "florence-community/Florence-2-large", trust_remote_code=True)` now work end-to-end
- Tool prompt module updated to reflect multi-model compatibility; comments are no longer Qwen3-VL-specific
- `bm_test/vlm/test_multi_vlm_smoke.py` — standalone GPU integration smoke test covering 7 model families (Qwen3-VL, MedGemma, LFM2.5-VL, SmolVLM, Moondream2, InternVL2, LLaVA-1.5)
- 20 new tests in `tests/test_vlm_adapter.py`: `TestCoordinateScalingHeuristic` (5), `TestVLMDtypeKwarg` (3), `TestVLMTrustRemoteCodeKwarg` (3), `TestVLMExpandedModelDetection` (6), `TestVLMLoaderKwargsPassthrough` (3)
- `docs/VLM_MODEL_SUPPORT.md` — new multi-model support guide with full compatibility matrix, dtype/trust_remote_code requirements, and model-specific usage examples

---

## [1.9.2] - 2026-03-19

### Added

**PyPI Distribution Rename**

- PyPI distribution name changed from `mata` to `datamata`; users now run `pip install datamata` to install
- `import mata` is unchanged — all existing code continues to work without modification
- Follows the PIL/Pillow precedent: distribution name and import name differ intentionally

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

**Feature Embedding — `embed` task (first-class public task)**

- `Embeddings` artifact — frozen dataclass `(N, D)` float32 embedding vectors for graph pipelines; auto-generates `instance_ids`, reshapes 1-D input to `(1, D)`, validates dimensionality; importable from `mata.core.artifacts`
- `EmbedAdapter` — public embedding adapter conforming to the `Embedder` protocol; wraps any `ReIDAdapter` subclass; accepts `Image` or `ROIs` artifacts and returns `np.ndarray`; `isinstance(adapter, Embedder)` returns `True`
- `mata.load("embed", "model-id")` — first-class task registration in `UniversalLoader`; HuggingFace model IDs create `HuggingFaceReIDAdapter` wrapped in `EmbedAdapter`; local `.onnx` files create `ONNXReIDAdapter` wrapped in `EmbedAdapter`; config alias support via `.mata/models.yaml`
- `mata.run("embed", image, model="model-id")` — one-liner embedding extraction; returns `np.ndarray` directly; accepts file paths, PIL Images, and numpy arrays
- `Embed` graph node — consumes `ROIs` artifact, produces `Embeddings` artifact; `normalize=True` by default; propagates `instance_ids` from source ROIs; records `num_embeddings` and `embedding_dim` metrics; handles empty ROIs gracefully
- `EmbedAdapter` exported from `mata.adapters`
- `Embeddings` exported from `mata.core.artifacts`
- `Embed` exported from `mata.nodes`
- 96 new tests: `test_embeddings_artifact.py` (25), `test_embed_adapter.py` (22), `test_embed_api.py` (25), `test_embed_node.py` (24)
- `examples/inference/embedding.py` — whole-image and graph pipeline embedding examples

**Graph Video Pipeline — `Graph.run()` callback, ReID node, AnnotateRT node**

- `VideoProcessor.process_video()` now accepts optional `callback(result, frame_num, frame_bgr)` parameter — fires after each frame's graph execution with the raw BGR frame; `None` default preserves existing behavior
- `Graph.run()` forwards `callback` to `process_video()` for video file sources — previously only supported for stream/webcam
- `CrossMatch` + `CrossMatches` artifact — frozen dataclass carrying cross-camera re-identification results through the graph; `to_dict()` / `from_dict()` round-trip; similarity validation in `[0.0, 1.0]`; importable from `mata.core.artifacts`
- `ReID` graph node — publishes tracked embeddings to Valkey via `ReIDBridge` and queries for cross-camera matches; inputs: `Tracks` + `Embeddings`, output: `CrossMatches`; empty inputs yield empty artifact gracefully; records `num_tracks_published` and `num_cross_matches` metrics
- `AnnotateRT` graph node — stateful real-time OpenCV annotation; draws boxes, labels, scores, track IDs, trajectory trails, camera labels, and cross-camera highlights (yellow double border); persists trail history across frames; `reset()` clears state
- `visualization_cv2` module — `track_color()`, `draw_boxes()`, `draw_trails()`, `draw_camera_label()` helper functions; lazy `cv2` import; duck-typed for any object with `bbox`/`score`/`label` attributes
- `_build_capability_map()` — added `"Embed": "embed"` and `"ReID": "reid"` entries; `_infer_capability()` updated with `"reid"` and `"embed"` heuristics
- `ReID` and `AnnotateRT` exported from `mata.nodes`
- `CrossMatch` and `CrossMatches` exported from `mata.core.artifacts`
- 166 new tests: `test_cross_matches.py` (29), `test_reid_node.py` (43), `test_annotate_rt.py` (50), `test_graph_video_pipeline_integration.py` (32), callback tests in `test_temporal.py` (+8) and `test_graph_run_video.py` (+4)

**Documentation**

- `docs/VALKEY_GUIDE.md` — full integration guide covering installation, basic usage, graph pipeline integration, YAML configuration, streaming patterns, Pub/Sub architecture, security (TLS, `password_env`, SSRF prevention, key sanitization), performance tuning (serializer choice, TTL strategies, connection pooling, async patterns), and top-5 troubleshooting issues
- `docs/GRAPH_API_REFERENCE.md` — new "Storage Nodes" section with full parameter tables for `ValkeyStore` and `ValkeyLoad`; new sections for `ReID` node, `AnnotateRT` node, `CrossMatches` artifact; updated `Graph.run()` callback documentation
- `docs/GRAPH_COOKBOOK.md` — 4 new recipes: single-camera tracking + annotation (Recipe 26), cross-camera ReID pipeline (Recipe 27), camera agent pipeline (Recipe 28), custom callback patterns (Recipe 29)
- `README.md` — Valkey added to Key Features list and Optional Dependencies table; ReID tracking section added with single-camera and cross-camera usage examples
- `QUICKSTART.md` — new "Valkey / Redis Result Storage" section with annotated code examples
- `QUICK_REFERENCE.md` — new "Valkey/Redis Storage Quick Reference (v1.9)" section with cheatsheet
- `docs/VALIDATION_GUIDE.md` — ReID tracking validation notes added

### Fixed

- Torchvision detection adapter: incorrect 0-based COCO label mapping replaced with proper 1-based `_get_torchvision_coco_labels()` matching torchvision's `__background__` + 80-class COCO convention
- Torchvision detection adapter: removed incorrect ImageNet mean/std normalization from preprocessing — torchvision detection models expect float32 tensors in `[0, 1]` range only
- Example scripts: added download instructions for COCO test image (`000000039769.jpg`); updated torchvision example to use `fasterrcnn_resnet50_fpn_v2`

### Changed

- `Graph.run()` callback signature for video files is now `(result, frame_num, frame_bgr)` — adds raw BGR frame as third argument; stream/webcam callbacks retain existing `(result, frame_num)` signature
- `_build_capability_map()` extended with `"Embed"` and `"ReID"` entries
- `mata.nodes.__all__` extended with `ValkeyStore` and `ValkeyLoad`
- `mata.core.exporters.__init__` extended with `export_valkey`, `load_valkey`, `publish_valkey`
- `mata.track()` signature extended with `reid_model`, `with_reid`, `reid_bridge` kwargs (backward-compatible defaults)
- `TrackingAdapter.__init__()` extended with `reid_encoder`, `reid_bridge` kwargs (both default to `None`; zero overhead when unused)
- `BOTSORT.get_dists()` appearance-distance branch now reachable when `encoder` is set via `reid_encoder`
- ByteTrack vs BotSort ReID comparison table in `README.md` updated to reflect v1.9.2 BotSort support
- `UniversalLoader._load_from_torchvision()` now supports `task="track"` by wrapping the detection adapter with `TrackingAdapter`

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
