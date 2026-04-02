# Documentation Audit: `.github/copilot-instructions.md`

- **Doc:** `.github/copilot-instructions.md`
- **Date:** 2026-04-02
- **Auditor:** GitHub Copilot (Claude Opus 4.6)
- **Total claims checked:** 42
- **Verified:** 30 | **Incorrect:** 8 | **Missing:** 0 | **Outdated:** 3 | **Misleading:** 1

---

## Critical Issues (INCORRECT)

### [DISC-001] ClassifyResult field name: `classifications` → `predictions`

- **Location:** Result Type Patterns / ClassifyResult section
- **Doc says:**
  ```python
  @dataclass
  class ClassifyResult:
      classifications: list[Classification]
  ```
- **Reality:** The actual field is `predictions`, not `classifications`:
  ```python
  @dataclass(frozen=True)
  class ClassifyResult:
      predictions: list[Classification]
  ```
- **Evidence:** `src/mata/core/types.py` line 1790
- **Impact:** Any AI agent or contributor following the doc will reference `result.classifications` which will raise `AttributeError`. This is a P0 issue that blocks users.

### [DISC-002] ClassifyResult `meta` field default

- **Location:** Result Type Patterns / ClassifyResult section
- **Doc says:**
  ```python
  meta: dict[str, Any] | None = None
  ```
- **Reality:** The doc correctly shows `dict[str, Any] | None = None` in the ClassifyResult snippet. _(Verified — this is correct.)_

_Note: Re-checked — the doc does show `None`. VERIFIED, not a discrepancy._

### [DISC-003] ClassifyResult `top1` return type

- **Location:** Result Type Patterns / ClassifyResult section
- **Doc says:**
  ```python
  @property
  def top1(self) -> Classification: ...
  ```
- **Reality:** The actual return type is `Classification | None`:
  ```python
  @property
  def top1(self) -> Classification | None:
      """Return the highest-confidence prediction, or None if empty."""
      return self.predictions[0] if self.predictions else None
  ```
- **Evidence:** `src/mata/core/types.py` line 1899
- **Impact:** Users won't know that `top1` can return `None` and may get `AttributeError` when accessing attributes on it without checking.

### [DISC-004] DepthResult primary field name and `save()` signature

- **Location:** Result Type Patterns / DepthResult section
- **Doc says:**
  ```python
  @dataclass
  class DepthResult:
      depth_map: np.ndarray  # (H, W) float array
      ...
      def save(self, output_path: str, colormap: str = "magma"): ...
  ```
- **Reality:**
  - The primary field is `depth`, not `depth_map`. (`depth_map` is a read-only property that returns `normalized` if available, otherwise `depth`.)
  - `save()` does NOT have a `colormap` parameter. Actual signature:
    ```python
    def save(self, output_path: str | Path, image: ... = None, format: str | None = None, **kwargs: Any) -> None:
    ```
- **Evidence:** `src/mata/core/types.py` lines 600 (field) and 660 (save)
- **Impact:** Users will try `DepthResult(depth_map=arr)` which fails, and `result.save("out.png", colormap="magma")` which silently passes through `**kwargs` but may not do what's expected.

### [DISC-005] EmbedResult `labels` field default

- **Location:** Result Type Patterns / EmbedResult section
- **Doc says:**
  ```python
  labels: list[str] = field(default_factory=list)
  ```
- **Reality:** The actual default is `None`:
  ```python
  labels: list[str] | None = None
  ```
- **Evidence:** `src/mata/core/types.py` line 999
- **Impact:** Code that does `for label in result.labels:` without checking for `None` will crash when using the default.

### [DISC-006] UniversalLoader `_detect_source_type` — 5-strategy claim is incomplete

- **Location:** UniversalLoader Detection Chain section
- **Doc says:** "5-strategy detection order" with strategies:
  1. None → "default"
  2. Config alias → "config_alias"
  3. Local file → "local_file"
  4. Contains '/' → "huggingface"
  5. Otherwise → "config_alias"
- **Reality:** The actual implementation has **7+ strategies** in this order:
  1. None → "default"
  2. Config alias → "config_alias"
  3. Local file (exists) → "local_file"
  4. Known extension (.pt/.pth/.onnx/.bin/.trt/.engine) → "local_file"
  5. Starts with "torchvision/" → "torchvision"
  6. External OCR engines (easyocr/paddleocr/tesseract) → "external_engine"
  7. External barcode engines (pyzbar/zxing) → "external_engine"
  8. Contains '/' → "huggingface"
  9. Fallback → "config_alias"
- **Evidence:** `src/mata/core/model_loader.py` lines 336-365
- **Impact:** Contributors adding new detection strategies may insert at wrong priority. Missing strategies (torchvision, external engines) are not documented.

### [DISC-007] `adapters/base/` described as a directory

- **Location:** Testing New Task Adapters section
- **Doc says:** "Inherit from appropriate base adapter in `adapters/base/`"
- **Reality:** There is no `adapters/base/` directory. The base adapter is a single file: `src/mata/adapters/base.py`
- **Evidence:** Directory listing of `src/mata/adapters/`
- **Impact:** Contributors looking for base classes in a directory will be confused. Minor, since the file is easy to find.

### [DISC-008] `load()` function described with `source` parameter

- **Location:** UniversalLoader Detection Chain section
- **Doc says:**
  ```python
  def _detect_source_type(self, task: str, source: Optional[str]) -> tuple[str, str]:
  ```
  and also refers to `load(self, task, source)`.
- **Reality:** The public `mata.load()` in `api.py` uses `model` as the parameter name:
  ```python
  def load(task: str, model: str | None = None, ...) -> Any:
  ```
  However, the internal `UniversalLoader.load()` does use `source`:
  ```python
  def load(self, task: str, source: str | None = None, ...) -> Any:
  ```
- **Evidence:** `src/mata/api.py` line 43 vs `src/mata/core/model_loader.py` line 55
- **Impact:** Low — the doc shows the internal API accurately, and the public API uses `model` as a positional arg. But the code section showing `load(self, task, source)` could mislead users into thinking it's the public API signature.

---

## Moderate Issues (OUTDATED / MISLEADING)

### [DISC-009] Version claim for EmbedResult: "v1.9.6"

- **Location:** Result Type Patterns / EmbedResult section header
- **Doc says:** `EmbedResult (🆕 v1.9.6):`
- **Reality:** The current `__version__` is `1.9.7`. The EmbedResult section shows v1.9.6 but earlier in the doc it says "v1.9.2b2 Beta Release 2 Embed" for the embed feature introduction. The version labeling is inconsistent.
- **Evidence:** `src/mata/__init__.py` line 26: `__version__ = "1.9.7"`
- **Impact:** Cosmetic — version annotations don't affect functionality, but consistency matters.

### [DISC-010] Architecture overview says "As of v1.9.5"

- **Location:** First paragraph of Architecture Overview
- **Doc says:** "As of v1.9.5, it features..."
- **Reality:** Current version is `1.9.7`. The overview should reference the current version or avoid pinning.
- **Evidence:** `src/mata/__init__.py` line 26
- **Impact:** Cosmetic — gives impression the doc hasn't been updated.

### [DISC-011] `load()` docstring mentions incomplete task list

- **Location:** Load function docstring in copilot-instructions
- **Doc says:** `task: Task type ("detect", "segment", "classify", "depth", "track")`
- **Reality:** The `run()` function in `api.py` lists these tasks:
  `"detect", "segment", "classify", "depth", "vlm", "ocr", "barcode", "embed", "recognize"`
  The `load()` call also supports `"vlm"`, `"ocr"`, `"barcode"`, `"embed"` tasks.
- **Evidence:** `src/mata/api.py` line 118 (run function docstring)
- **Impact:** Contributors may not realize all supported tasks; especially `vlm`, `ocr`, `barcode`, and `embed`.

### [DISC-012] ClassifyResult shown as non-frozen dataclass

- **Location:** Result Type Patterns / ClassifyResult section
- **Doc says:**
  ```python
  @dataclass
  class ClassifyResult:
  ```
- **Reality:** It's `@dataclass(frozen=True)`:
  ```python
  @dataclass(frozen=True)
  class ClassifyResult:
  ```
- **Evidence:** `src/mata/core/types.py` line 1781
- **Impact:** Minor — but could mislead contributors into thinking they can mutate ClassifyResult fields.

---

## Verified Claims (Correct)

| #   | Claim                                                                                                          | Status                                              |
| --- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 1   | `mata.load("detect", "facebook/detr-resnet-50")` pattern                                                       | ✅ VERIFIED                                         |
| 2   | `mata.load("classify", "./model.onnx")` pattern                                                                | ✅ VERIFIED                                         |
| 3   | `mata.load("track", "...", tracker="botsort")`                                                                 | ✅ VERIFIED                                         |
| 4   | `mata.load("embed", "openai/clip-vit-base-patch32")`                                                           | ✅ VERIFIED                                         |
| 5   | `mata.load("barcode", "pyzbar")` / `"zxing"`                                                                   | ✅ VERIFIED                                         |
| 6   | `mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")`                                                                | ✅ VERIFIED                                         |
| 7   | `mata.run("recognize", ...)` with `gallery=` kwarg                                                             | ✅ VERIFIED                                         |
| 8   | `mata.track()` with `reid_model=` and `reid_bridge=`                                                           | ✅ VERIFIED                                         |
| 9   | `from mata.trackers import ReIDBridge`                                                                         | ✅ VERIFIED                                         |
| 10  | VisionResult has `instances`, `meta`, `to_json`, `to_dict`, `save`, `get_instances`, `get_stuff`               | ✅ VERIFIED                                         |
| 11  | VisionResult `meta: dict[str, Any] = field(default_factory=dict)`                                              | ✅ VERIFIED                                         |
| 12  | `DetectResult = VisionResult` / `SegmentResult = VisionResult`                                                 | ✅ VERIFIED                                         |
| 13  | EmbedResult is `@dataclass(frozen=True)`                                                                       | ✅ VERIFIED                                         |
| 14  | EmbedResult has `embedding` property, `dim` property                                                           | ✅ VERIFIED                                         |
| 15  | EmbedResult has `to_json`, `to_dict`, `save`, `from_dict`, `from_json`                                         | ✅ VERIFIED                                         |
| 16  | `from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError`                                   | ✅ VERIFIED (both classes exist)                    |
| 17  | Gallery accepts `similarity_thresh` kwarg                                                                      | ✅ VERIFIED                                         |
| 18  | Gallery has `search()` and `add()` methods                                                                     | ✅ VERIFIED                                         |
| 19  | CLI has `recognize` subcommand                                                                                 | ✅ VERIFIED                                         |
| 20  | `EarlyExit`, `While`, `Graph.add(condition=...)` exist                                                         | ✅ VERIFIED                                         |
| 21  | `from mata.nodes import VLMQuery`                                                                              | ✅ VERIFIED                                         |
| 22  | VLMQuery, VLMDetect, VLMDescribe node classes exist                                                            | ✅ VERIFIED                                         |
| 23  | GalleryMatchNode exists in `src/mata/nodes/`                                                                   | ✅ VERIFIED                                         |
| 24  | `from mata.core.logging import get_logger`                                                                     | ✅ VERIFIED                                         |
| 25  | `agent_loop.py`, `tool_schema.py`, `tool_registry.py`, `tool_prompts.py`, `parsers.py`, `image_tools.py` exist | ✅ VERIFIED                                         |
| 26  | Bboxes always xyxy format                                                                                      | ✅ VERIFIED (per types.py comment)                  |
| 27  | VLM `dtype` and `trust_remote_code` kwargs                                                                     | ✅ VERIFIED (via HuggingFace VLM adapter)           |
| 28  | Instance has `embedding`, `track_id` fields                                                                    | ✅ VERIFIED                                         |
| 29  | `mata.track()` function exists in `api.py`                                                                     | ✅ VERIFIED                                         |
| 30  | Config precedence order documented                                                                             | ✅ VERIFIED (consistent with model_loader.py logic) |
