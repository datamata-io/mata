# Notebook Integration — Implementation Tasks

> **Project**: MATA — JupyterLab / Notebook Rich Display  
> **Feature**: Add `_repr_html_()` / `_repr_png_()` to all result types for interactive notebook display, plus a `show()` utility and starter notebooks  
> **Timeline**: 1.5 weeks (Estimated)  
> **Team**: 1 developer  
> **Status**: Planning Phase

## Progress Summary

### Completed ✅

_None._

### In Progress 🔄

_None._

### Pending ⏳

- **Phase A**: Core rendering module (`src/mata/notebook.py`)
- **Phase B**: `_repr_html_()` / `_repr_png_()` on result types
- **Phase C**: Public API (`show()`) and dependency wiring
- **Phase D**: Example notebooks
- **Phase E**: Testing
- **Phase F**: Documentation & Release

### Metrics

- **Tests**: Target 50+ new tests (rendering functions, repr methods, graceful degradation, truncation)
- **Code Coverage**: Maintain >80% across new code; all mocked (no real Jupyter in CI)
- **Regression**: 4307+ existing tests must pass with zero regressions
- **New Files**: 9 files (1 rendering module, 1 test file, ~6 example notebooks, 1 .gitattributes entry)
- **Modified Files**: 4 files (`types.py`, `artifacts/embeddings.py`, `__init__.py`, `pyproject.toml`)

---

## Issue / Bottleneck Summary

| ID  | Severity | Location                          | Description                                                                                                                                                             |
| --- | -------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| N-1 | Medium   | `src/mata/core/types.py`          | Result dataclasses are `frozen=True`; `_repr_html_()` is a method not a field — compatible, but must not introduce mutable state                                        |
| N-2 | Medium   | `src/mata/notebook.py`            | Image overlay requires source image; `VisionResult.meta` may not contain `input_path` — must gracefully fall back to table-only rendering                               |
| N-3 | Low      | `src/mata/notebook.py`            | JupyterLab has both dark and light themes — HTML must use neutral CSS or respect `prefers-color-scheme`                                                                 |
| N-4 | Low      | `pyproject.toml`                  | `matplotlib` is already an optional dep in `[eval]` and `[viz]` groups — `[notebook]` group should not duplicate but can declare `datamata[viz]`                        |
| N-5 | Info     | `examples/notebooks/`             | `.ipynb` files contain cell outputs that bloat Git; need `.gitattributes` or nbstripout to strip outputs before commit                                                  |
| N-6 | Info     | `src/mata/core/types.py` L261-941 | Multiple active TASK files also modify `types.py` (TASK_BARCODE, TASK_EMBED, TASK_GGUF) — notebook changes are additive (new methods, no field changes), low merge risk |

---

## Architectural Decisions

| Decision                                | Choice                                                           | Rationale                                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Display protocol                        | `_repr_html_()` + `_repr_png_()` on result dataclasses           | Standard IPython/Jupyter rich display protocol; auto-renders in cells without extra imports — same pattern as pandas/PIL |
| Rendering centralization                | New `src/mata/notebook.py` module with `render_*()` functions    | Keeps HTML/image generation out of `types.py`; result types delegate via lazy import; single place to maintain styling   |
| Image encoding                          | Base64-encoded PNG inline in HTML via `<img src="data:...">`     | No filesystem state; works in JupyterLab, VS Code notebook, Colab, and exported HTML                                     |
| Overlay generation                      | Reuse existing `export_image()` from `core/exporters`            | No duplication; proven PIL-based rendering with bbox/mask support                                                        |
| Graceful degradation                    | `_repr_html_()` catches `ImportError` and falls back to `repr()` | MATA must remain fully functional without IPython installed; no hard dependency                                          |
| HTML table truncation                   | First 20 rows + "…and N more" for results with >100 instances    | Prevents browser lag on large detection results                                                                          |
| No ipywidgets in v1                     | Only static HTML + images, no interactive widgets                | Keeps scope minimal; widgets can be added in a follow-up release                                                         |
| `show()` explicit utility               | `mata.show(result, image=..., **kwargs)` convenience function    | Power-user escape hatch for controlling visualization opts; thin wrapper around `IPython.display.display()`              |
| Optional dependency group: `[notebook]` | `["ipython>=7.0", "matplotlib>=3.5.0"]`                          | IPython for `display()` in `show()`; matplotlib already available via `[viz]` but explicit here for standalone installs  |

---

## Conflict Analysis

| Area                              | Risk | Mitigation                                                                                                  |
| --------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------- |
| `types.py` — active TASK files    | Low  | Notebook adds new methods at end of each class; other tasks add fields/enums — non-overlapping code regions |
| `__init__.py` — export list       | Low  | Single line addition (`show`); append to API section of `__all__`                                           |
| `pyproject.toml` — extras section | Low  | New `[notebook]` group is additive; no conflict with `[viz]`, `[eval]`, or GGUF extras                      |
| `visualization.py` — reuse only   | None | No modifications; only imports existing functions                                                           |

---

## Task Assignment Guide

### 🔴 Critical Path (dependency order)

```
A1 (notebook.py render functions) → B1 (VisionResult repr) → B2 (ClassifyResult repr) → B3 (DepthResult repr) → C1 (show utility) → E1 (unit tests) → E2 (regression)
```

### 🟡 Parallel Work (independent after A1)

- **B1**, **B2**, **B3**, **B4**, **B5**, **B6** — all repr methods are independent of each other (all depend on A1)
- **C2** (`pyproject.toml` extras) — independent of all B tasks
- **D1** (example notebooks) — can start once B tasks are done, parallel with C1

### 🟢 Post-Integration (after core merges)

- **E1**, **E2** — testing after all code changes
- **F1**, **F2** — documentation and CHANGELOG

---

## Phase A: Core Rendering Module

### Task A1: Create `src/mata/notebook.py` 🔴

**Priority**: Critical  
**Estimated time**: 6 hours  
**Dependencies**: None  
**Status**: ⏳ Pending

**Description**: Create the central notebook rendering module that all result types will delegate to. Contains functions that produce raw HTML strings or PNG bytes for each result type. Reuses existing visualization infrastructure.

**Files to create**:

- `src/mata/notebook.py` — New module (~250 lines)

**Changes required**:

```python
"""Rich display rendering for Jupyter Notebook / JupyterLab.

Provides render functions that produce HTML strings or PNG bytes
for MATA result types. These are called by _repr_html_() / _repr_png_()
on result dataclasses.

All IPython/matplotlib imports are guarded — this module is importable
without Jupyter installed (returns None on ImportError).
"""
from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mata.core.types import (
        BarcodeResult, ClassifyResult, DepthResult, OCRResult, VisionResult,
    )
    from mata.core.artifacts.embeddings import Embeddings


_MAX_TABLE_ROWS = 20       # Truncate large tables
_TRUNCATION_THRESHOLD = 100  # Show "...and N more" above this


def render_vision_html(result: VisionResult) -> str | None:
    """Render VisionResult as HTML table + optional inline image overlay."""
    ...

def render_classify_html(result: ClassifyResult) -> str | None:
    """Render ClassifyResult as top-5 bar chart + score table."""
    ...

def render_depth_png(result: DepthResult) -> bytes | None:
    """Render DepthResult as colormap PNG bytes."""
    ...

def render_ocr_html(result: OCRResult) -> str | None:
    """Render OCRResult as text region table."""
    ...

def render_barcode_html(result: BarcodeResult) -> str | None:
    """Render BarcodeResult as decoded barcode table."""
    ...

def render_embeddings_html(result: Embeddings) -> str | None:
    """Render Embeddings artifact as summary table."""
    ...

def show(result, image=None, **kwargs) -> None:
    """Explicit display utility — renders result in current notebook cell."""
    ...
```

Key implementation details:

1. **`render_vision_html()`**: Build HTML `<table>` from `result.instances` (label, score, bbox, track_id columns). If `result.meta.get("input_path")` exists and the file is readable, call `export_image()` from `mata.core.exporters` into a `BytesIO` buffer, base64-encode, embed as `<img src="data:image/png;base64,...">`. Truncate table at `_MAX_TABLE_ROWS` if `len(result.instances) > _TRUNCATION_THRESHOLD`.

2. **`render_classify_html()`**: Build HTML with a horizontal bar SVG for top-5 predictions + a `<table>` with label/score columns.

3. **`render_depth_png()`**: Use `matplotlib` to render `result.depth_map` with `magma` colormap into a PNG buffer. Return raw bytes for `_repr_png_()`.

4. **`render_ocr_html()`**: Build HTML `<table>` from `result.regions` (text, score, bbox columns).

5. **`render_barcode_html()`**: Build HTML `<table>` from `result.barcodes` (data, type, score, bbox columns).

6. **`render_embeddings_html()`**: Summary HTML: shape `(N, D)`, dtype, normalized flag, first 5 instance IDs.

7. **`show()`**: Imports `IPython.display` and calls `display(HTML(html))` or `display(Image(data=png_bytes))`. Passes `**kwargs` to the underlying `render_*` function.

8. **CSS styling**: Minimal inline styles. Use `border-collapse: collapse`, neutral `#f5f5f5` / `#333` colors, `font-family: monospace` for numeric columns. Avoid theme-specific colors.

9. **Error guard**: Every `render_*` function wraps in `try/except Exception` and returns `None` on failure (never break a notebook cell with rendering errors).

**Constraints**:

- No hard dependency on IPython, matplotlib, or any package not in core deps
- All imports behind `try/except ImportError`
- Must be importable and callable in plain Python (returns `None` when deps missing)
- No mutable global state

**Acceptance Criteria**:

- ✅ `render_vision_html()` returns valid HTML string with `<table>` for a VisionResult with 3 instances
- ✅ `render_vision_html()` returns HTML with embedded base64 `<img>` when source image is available
- ✅ `render_vision_html()` truncates to 20 rows when given 200 instances
- ✅ `render_classify_html()` returns HTML with SVG bar chart for top-5 predictions
- ✅ `render_depth_png()` returns PNG bytes (magic bytes `\x89PNG`) for a valid DepthResult
- ✅ `render_ocr_html()` returns HTML table for OCRResult with 5 text regions
- ✅ `render_barcode_html()` returns HTML table for BarcodeResult with 3 barcodes
- ✅ `render_embeddings_html()` returns HTML summary for (10, 512) embedding matrix
- ✅ All functions return `None` when matplotlib/PIL is not installed
- ✅ `show()` calls `IPython.display.display()` when IPython is available
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

## Phase B: Rich Repr Methods on Result Types

### Task B1: Add `_repr_html_()` to VisionResult 🟡

**Priority**: High  
**Estimated time**: 2 hours  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_html_()` method to `VisionResult` that delegates to `notebook.render_vision_html()`. The method catches `ImportError` so MATA works without notebook dependencies.

**Files to modify**:

- `src/mata/core/types.py` — Add method after line ~580 (end of `VisionResult.save()`)

**Changes required**:

```python
# Add at end of VisionResult class body (after save() method, before DepthResult)

def _repr_html_(self) -> str | None:
    """Rich HTML display for Jupyter notebooks."""
    try:
        from mata.notebook import render_vision_html
        return render_vision_html(self)
    except Exception:
        return None
```

**Constraints**:

- `frozen=True` — method is compatible (no field mutation)
- Lazy import only — no module-level import of `mata.notebook`
- Return `None` on any exception (Jupyter falls back to `__repr__`)

**Acceptance Criteria**:

- ✅ `VisionResult(...)._repr_html_()` returns HTML string when `mata.notebook` is available
- ✅ `VisionResult(...)._repr_html_()` returns `None` when `mata.notebook` import fails
- ✅ Existing VisionResult tests pass unchanged (no field changes)
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task B2: Add `_repr_html_()` to ClassifyResult 🟡

**Priority**: High  
**Estimated time**: 1 hour  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_html_()` to `ClassifyResult` delegating to `render_classify_html()`.

**Files to modify**:

- `src/mata/core/types.py` — Add method after line ~1774 (end of `ClassifyResult.filter_by_score()`)

**Changes required**:

```python
# Add at end of ClassifyResult class body (after filter_by_score(), before Track)

def _repr_html_(self) -> str | None:
    """Rich HTML display for Jupyter notebooks."""
    try:
        from mata.notebook import render_classify_html
        return render_classify_html(self)
    except Exception:
        return None
```

**Acceptance Criteria**:

- ✅ `ClassifyResult(...)._repr_html_()` returns HTML with bar chart when available
- ✅ Returns `None` on import failure
- ✅ Existing ClassifyResult tests pass unchanged
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task B3: Add `_repr_png_()` to DepthResult 🟡

**Priority**: High  
**Estimated time**: 1 hour  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_png_()` to `DepthResult` delegating to `render_depth_png()`. Uses PNG rather than HTML because depth maps are best rendered as colormapped images.

**Files to modify**:

- `src/mata/core/types.py` — Add method after line ~691 (end of `DepthResult.save()`)

**Changes required**:

```python
# Add at end of DepthResult class body (after save(), before TextRegion)

def _repr_png_(self) -> bytes | None:
    """Rich PNG display for Jupyter notebooks (colormap visualization)."""
    try:
        from mata.notebook import render_depth_png
        return render_depth_png(self)
    except Exception:
        return None
```

**Acceptance Criteria**:

- ✅ `DepthResult(...)._repr_png_()` returns PNG bytes when matplotlib available
- ✅ Returns `None` on import failure
- ✅ Existing DepthResult tests pass unchanged
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task B4: Add `_repr_html_()` to OCRResult 🟡

**Priority**: Medium  
**Estimated time**: 1 hour  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_html_()` to `OCRResult` delegating to `render_ocr_html()`.

**Files to modify**:

- `src/mata/core/types.py` — Add method after line ~836 (end of `OCRResult.save()`)

**Changes required**:

```python
# Add at end of OCRResult class body (after save(), before BarcodeRegion)

def _repr_html_(self) -> str | None:
    """Rich HTML display for Jupyter notebooks."""
    try:
        from mata.notebook import render_ocr_html
        return render_ocr_html(self)
    except Exception:
        return None
```

**Acceptance Criteria**:

- ✅ `OCRResult(...)._repr_html_()` returns HTML table for text regions
- ✅ Returns `None` on import failure
- ✅ Existing OCRResult tests pass unchanged
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task B5: Add `_repr_html_()` to BarcodeResult 🟡

**Priority**: Medium  
**Estimated time**: 1 hour  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_html_()` to `BarcodeResult` delegating to `render_barcode_html()`.

**Files to modify**:

- `src/mata/core/types.py` — Add method after line ~941 (end of `BarcodeResult.save()`)

**Changes required**:

```python
# Add at end of BarcodeResult class body (after save(), before ModelType)

def _repr_html_(self) -> str | None:
    """Rich HTML display for Jupyter notebooks."""
    try:
        from mata.notebook import render_barcode_html
        return render_barcode_html(self)
    except Exception:
        return None
```

**Acceptance Criteria**:

- ✅ `BarcodeResult(...)._repr_html_()` returns HTML table for decoded barcodes
- ✅ Returns `None` on import failure
- ✅ Existing BarcodeResult tests pass unchanged
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task B6: Add `_repr_html_()` to Embeddings 🟡

**Priority**: Medium  
**Estimated time**: 1 hour  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Add `_repr_html_()` to the `Embeddings` artifact class.

**Files to modify**:

- `src/mata/core/artifacts/embeddings.py` — Add method after line ~96 (end of `from_dict()`)

**Changes required**:

```python
# Add at end of Embeddings class body (after from_dict())

def _repr_html_(self) -> str | None:
    """Rich HTML display for Jupyter notebooks."""
    try:
        from mata.notebook import render_embeddings_html
        return render_embeddings_html(self)
    except Exception:
        return None
```

**Acceptance Criteria**:

- ✅ `Embeddings(...)._repr_html_()` returns HTML summary with shape, dtype, normalized
- ✅ Returns `None` on import failure
- ✅ Existing Embeddings tests pass unchanged
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

## Phase C: Public API & Dependencies

### Task C1: Add `show()` to public API 🟡

**Priority**: High  
**Estimated time**: 2 hours  
**Dependencies**: Task A1  
**Status**: ⏳ Pending

**Description**: Export `show()` from `mata` namespace as a lazy import, mirroring the pattern used for `visualize_segmentation` in `src/mata/__init__.py` L56-68.

**Files to modify**:

- `src/mata/__init__.py` — Add lazy import block + `__all__` entry

**Changes required**:

After the existing visualization lazy import block (line ~68), add:

```python
# Notebook display (lazy import to avoid hard dependency)
try:
    from .notebook import show

    _NOTEBOOK_AVAILABLE = True
except ImportError:
    _NOTEBOOK_AVAILABLE = False

    def show(*args, **kwargs):
        raise ImportError("Notebook display requires IPython. Install with: pip install datamata[notebook]")
```

Add `"show"` to `__all__` list in the API section (after `"verbose"`, around line 81).

**Acceptance Criteria**:

- ✅ `import mata; mata.show` is callable
- ✅ `mata.show(result)` calls `IPython.display.display()` when IPython available
- ✅ `mata.show(result)` raises `ImportError` with install instructions when IPython missing
- ✅ `"show"` appears in `mata.__all__`
- ✅ `import mata` does not fail when IPython is not installed
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

### Task C2: Add `[notebook]` optional dependency group 🟡

**Priority**: Medium  
**Estimated time**: 1 hour  
**Dependencies**: None  
**Status**: ⏳ Pending

**Description**: Add a `[notebook]` extras group to `pyproject.toml` and include it in the `[all]` group.

**Files to modify**:

- `pyproject.toml` — Add `notebook` group after `barcode-all` section (~line 115), update `all` group (~line 137)

**Changes required**:

```toml
# Notebook / JupyterLab rich display
notebook = [
    "ipython>=7.0",
    "matplotlib>=3.5.0",
]
```

Update the `all` group:

```toml
all = [
    "datamata[onnx,classification,eval,viz,segmentation,ocr,notebook]",
]
```

**Acceptance Criteria**:

- ✅ `pip install datamata[notebook]` installs IPython and matplotlib
- ✅ `pip install datamata[all]` includes notebook deps
- ✅ `pip install datamata` (plain) does not pull IPython
- ✅ Test requirements: see Task E1

**After Implementation**: [To be filled after completion]

---

## Phase D: Example Notebooks

### Task D1: Create starter notebooks 🟢

**Priority**: Low  
**Estimated time**: 6 hours  
**Dependencies**: Tasks B1-B6, C1  
**Status**: ⏳ Pending

**Description**: Create a set of example Jupyter notebooks demonstrating MATA's notebook integration for each major task. Each notebook should be minimal (5-15 cells), runnable, and demonstrate the auto-display feature.

**Files to create**:

- `examples/notebooks/01_detection.ipynb` — Load detector, run on sample image, auto-display VisionResult
- `examples/notebooks/02_classification.ipynb` — Classify images, display top-5 bar chart
- `examples/notebooks/03_segmentation.ipynb` — Instance + panoptic segmentation display
- `examples/notebooks/04_depth_estimation.ipynb` — Depth map with magma colormap display
- `examples/notebooks/05_tracking.ipynb` — Frame-by-frame tracking with track IDs
- `examples/notebooks/06_vlm_query.ipynb` — VLM Q&A with image + text display
- `examples/notebooks/README.md` — Index with descriptions and prerequisites

**Changes required**:

Each notebook follows this structure:

```
Cell 1 (Markdown): Title + brief description
Cell 2 (Code): pip install datamata[notebook] (commented out, for reference)
Cell 3 (Code): import mata; print(mata.__version__)
Cell 4 (Code): Load model + run inference
Cell 5 (Code): result  # auto-display
Cell 6 (Code): mata.show(result, image="...", show_masks=True)  # explicit display
Cell 7 (Markdown): Notes on customization
```

**Constraints**:

- Notebooks should NOT contain cell outputs (strip before commit)
- Use small/fast models for quick execution (DETR-ResNet-50, CLIP-ViT-B/32)
- Reference actual images in `data/assets/` or download from URLs
- Add `.gitattributes` entry: `*.ipynb filter=strip-notebook-output`

**Acceptance Criteria**:

- ✅ Each notebook runs without errors in JupyterLab
- ✅ Auto-display renders inline HTML/images in notebook cells
- ✅ `mata.show()` explicit call works
- ✅ No cell outputs committed to Git
- ✅ `examples/notebooks/README.md` lists all notebooks with descriptions

**After Implementation**: [To be filled after completion]

---

### Task D2: Add `.gitattributes` for notebook output stripping 🟢

**Priority**: Low  
**Estimated time**: 1 hour  
**Dependencies**: None  
**Status**: ⏳ Pending

**Description**: Configure Git to strip cell outputs from `.ipynb` files to keep the repository clean.

**Files to modify/create**:

- `.gitattributes` — Add filter rule for `*.ipynb`

**Changes required**:

Option A (if nbstripout is installed):

```
*.ipynb filter=nbstripout
```

Option B (simpler, just mark as binary to avoid noisy diffs):

```
*.ipynb diff=jupyternotebook linguist-language=JSON
```

Recommend Option A with a note in `CONTRIBUTING.md` to install `nbstripout`:

```bash
pip install nbstripout
nbstripout --install
```

**Acceptance Criteria**:

- ✅ `.gitattributes` file exists with notebook filter rule
- ✅ Committing a notebook with cell outputs strips the outputs (with nbstripout)
- ✅ Documented in `CONTRIBUTING.md`

**After Implementation**: [To be filled after completion]

---

## Phase E: Testing

### Task E1: Unit tests for notebook rendering 🔴

**Priority**: Critical  
**Estimated time**: 6 hours  
**Dependencies**: Tasks A1, B1-B6, C1, C2  
**Status**: ⏳ Pending

**Description**: Comprehensive test suite for all notebook rendering functions and repr methods. Mock IPython where needed; test both with and without optional deps.

**Files to create**:

- `tests/test_notebook.py` — New test file (~400 lines, 50+ tests)

**Changes required**:

Test groups:

1. **`render_vision_html()` tests (8+ tests)**
   - Empty VisionResult → valid HTML (empty table)
   - VisionResult with 3 instances → table with 3 rows
   - VisionResult with 200 instances → truncated to 20 rows + "…and 180 more"
   - VisionResult with `meta["input_path"]` pointing to real image → HTML with `<img>` tag
   - VisionResult without input_path → table-only HTML (no `<img>`)
   - HTML contains expected columns: Label, Score, BBox
   - VisionResult with track_ids → Track ID column present
   - VisionResult with text (VLM) → text displayed in HTML

2. **`render_classify_html()` tests (5+ tests)**
   - ClassifyResult with 5 predictions → bar chart SVG + table
   - ClassifyResult with 1 prediction → single bar
   - Empty ClassifyResult → valid HTML (empty state)
   - Scores correctly formatted (2 decimal places)
   - Labels correctly escaped (no XSS via label names)

3. **`render_depth_png()` tests (4+ tests)**
   - DepthResult with (100, 100) array → PNG bytes (starts with `\x89PNG`)
   - DepthResult with normalized array → uses normalized
   - Returns `None` when matplotlib not installed (mock import failure)
   - PNG dimensions match input array

4. **`render_ocr_html()` tests (4+ tests)**
   - OCRResult with 5 regions → table with 5 rows
   - Empty OCRResult → valid HTML
   - Text content correctly HTML-escaped
   - Columns: Text, Score, BBox

5. **`render_barcode_html()` tests (4+ tests)**
   - BarcodeResult with 3 barcodes → table with 3 rows
   - Empty BarcodeResult → valid HTML
   - Data content correctly HTML-escaped
   - Columns: Data, Type, Score, BBox

6. **`render_embeddings_html()` tests (4+ tests)**
   - Embeddings (10, 512) → summary with shape, dim, normalized
   - Embeddings with instance_ids → shows first 5 IDs
   - Empty embeddings → valid HTML

7. **`_repr_html_()` / `_repr_png_()` integration tests (6+ tests)**
   - `VisionResult(...)._repr_html_()` returns string
   - `ClassifyResult(...)._repr_html_()` returns string
   - `DepthResult(...)._repr_png_()` returns bytes
   - `OCRResult(...)._repr_html_()` returns string
   - `BarcodeResult(...)._repr_html_()` returns string
   - `Embeddings(...)._repr_html_()` returns string

8. **Graceful degradation tests (5+ tests)**
   - Mock `mata.notebook` import failure → `_repr_html_()` returns `None`
   - Mock matplotlib import failure → `render_depth_png()` returns `None`
   - Mock PIL import failure → overlay not rendered, table-only fallback
   - `show()` without IPython → raises `ImportError` with install message
   - `show()` with IPython → calls `IPython.display.display()`

9. **Security tests (3+ tests)**
   - Label names with `<script>` tags are HTML-escaped
   - Barcode data with HTML entities is escaped
   - OCR text with special chars is escaped

10. **`show()` utility tests (4+ tests)**
    - `show(VisionResult)` calls display with HTML
    - `show(DepthResult)` calls display with Image
    - `show(result, image="path")` passes image to render function
    - `show(result, show_masks=True)` passes kwargs through

**Constraints**:

- All tests must work without Jupyter/IPython installed (use mocks)
- No real model inference — construct result objects manually
- Use `data/assets/` images for overlay tests (or tiny synthetic images)

**Acceptance Criteria**:

- ✅ 50+ tests, all passing
- ✅ >90% coverage of `src/mata/notebook.py`
- ✅ Tests validate HTML structure (contains `<table>`, `<tr>`, `<td>`)
- ✅ Tests validate XSS protection (HTML-escaped user content)
- ✅ Tests validate graceful degradation (no crashes without deps)

**After Implementation**: [To be filled after completion]

---

### Task E2: Full regression suite verification 🔴

**Priority**: Critical  
**Estimated time**: 2 hours  
**Dependencies**: Tasks A1-C2, E1  
**Status**: ⏳ Pending

**Description**: Run the complete MATA test suite to verify zero regressions from notebook integration changes.

**Files to verify**:

- Run: `pytest tests/ -v`

**Changes required**:

None — this is a verification task.

**Acceptance Criteria**:

- ✅ All 4307+ existing tests pass
- ✅ New `test_notebook.py` tests pass (50+ tests)
- ✅ No new deprecation warnings introduced
- ✅ `import mata` startup time not measurably affected (lazy imports)

**After Implementation**: [To be filled after completion]

---

## Phase F: Documentation & Release

### Task F1: Update documentation 🟢

**Priority**: Low  
**Estimated time**: 3 hours  
**Dependencies**: Tasks A1-E2  
**Status**: ⏳ Pending

**Description**: Document the notebook integration feature in relevant documentation files.

**Files to modify**:

- `README.md` — Add "Notebook Support" section with quick example
- `QUICKSTART.md` — Add notebook usage example
- `INSTALLATION.md` — Document `pip install datamata[notebook]`
- `CHANGELOG.md` — Add v1.9.4 entry for notebook integration

**Changes required**:

README section:

````markdown
### Notebook Support

MATA results display automatically in Jupyter notebooks:

```python
import mata
result = mata.run("detect", "image.jpg", model="facebook/detr-resnet-50")
result  # Rich HTML display with image overlay
```
````

Install notebook support: `pip install datamata[notebook]`

````

CHANGELOG entry:
```markdown
## [1.9.4] - YYYY-MM-DD

### Added
- Notebook integration: `_repr_html_()` / `_repr_png_()` on all result types
- `mata.show()` explicit display utility for notebooks
- `[notebook]` optional dependency group
- Example notebooks for detection, classification, segmentation, depth, tracking, VLM
````

**Acceptance Criteria**:

- ✅ README has notebook section with code example
- ✅ INSTALLATION.md lists `[notebook]` extra
- ✅ CHANGELOG.md has entry for this feature
- ✅ All documentation code examples are syntactically correct

**After Implementation**: [To be filled after completion]

---

### Task F2: Update CONTRIBUTING.md with notebook guidelines 🟢

**Priority**: Low  
**Estimated time**: 1 hour  
**Dependencies**: Task D2  
**Status**: ⏳ Pending

**Description**: Add notebook contribution guidelines (nbstripout, no committed outputs, testing).

**Files to modify**:

- `CONTRIBUTING.md` — Add "Notebooks" section

**Changes required**:

```markdown
### Notebooks

Example notebooks live in `examples/notebooks/`. When contributing:

1. Install nbstripout: `pip install nbstripout && nbstripout --install`
2. Never commit cell outputs — they are stripped automatically
3. Use small/fast models for quick execution
4. Test notebooks manually in JupyterLab before submitting
```

**Acceptance Criteria**:

- ✅ CONTRIBUTING.md has notebook section
- ✅ nbstripout install instructions included

**After Implementation**: [To be filled after completion]

---

## Testing Checklist

### Unit Tests

- ⏳ All `render_*()` functions tested with valid inputs
- ⏳ All `render_*()` functions tested with empty/edge-case inputs
- ⏳ All `_repr_html_()` / `_repr_png_()` methods tested
- ⏳ Graceful degradation tested (missing deps)
- ⏳ HTML output sanitization tested (XSS prevention)
- ⏳ Table truncation tested (>100 instances)
- ⏳ Code coverage >90% for `notebook.py`

### Integration Tests

- ⏳ `mata.show()` end-to-end with mock IPython
- ⏳ `VisionResult._repr_html_()` with real image overlay
- ⏳ `DepthResult._repr_png_()` with real depth array

### Validation Tests

- ⏳ Full regression suite (4307+ tests)
- ⏳ `import mata` with and without `[notebook]` installed
- ⏳ `pip install datamata[notebook]` installs correct deps

### Manual Tests

- ⏳ Example notebooks run in JupyterLab
- ⏳ Auto-display works in VS Code notebook
- ⏳ Auto-display works in Google Colab
- ⏳ `mata.show()` renders correctly

---

## Definition of Done

A task is considered **DONE** when:

1. ⏳ **Code Complete**: All code written and committed
2. ⏳ **Tests Pass**: Unit tests written and passing
3. ⏳ **Code Review**: Reviewed by at least one other developer
4. ⏳ **Documentation**: Docstrings and comments added
5. ⏳ **Integration**: Works with dependent components
6. ⏳ **CI Passing**: All CI checks pass (including 4307+ existing tests)
7. ⏳ **No Hard Dependencies**: `import mata` works without notebook extras
8. ⏳ **XSS Safe**: All user content HTML-escaped in rendered output
