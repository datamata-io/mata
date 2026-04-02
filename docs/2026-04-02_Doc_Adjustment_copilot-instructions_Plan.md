# Adjustment Plan: `.github/copilot-instructions.md`

- **Audit Report:** `docs/2026-04-02_Doc_Audit_copilot-instructions.md`
- **Date:** 2026-04-02
- **Total fixes:** 10
- **P0 (blocks users/agents):** 4 | **P1 (misleading):** 4 | **P2 (cosmetic):** 2

---

## P0 Fixes — Blocks Users / Agents

### FIX-001: ClassifyResult field `classifications` → `predictions`

**Discrepancy:** DISC-001

**Before:**

```python
@dataclass
class ClassifyResult:
    classifications: list[Classification]
    meta: dict[str, Any] | None = None

    @property
    def top1(self) -> Classification: ...
    @property
    def top5(self) -> list[Classification]: ...

    def to_json(self) -> str: ...
    def save(self, output_path: str, **kwargs): ...
```

**After:**

```python
@dataclass(frozen=True)
class ClassifyResult:
    predictions: list[Classification]
    meta: dict[str, Any] | None = None

    @property
    def top1(self) -> Classification | None: ...
    @property
    def top5(self) -> list[Classification]: ...

    def to_json(self) -> str: ...
    def save(self, output_path: str, **kwargs): ...
```

**Changes:**

1. Field name: `classifications` → `predictions`
2. `@dataclass` → `@dataclass(frozen=True)`
3. `top1` return type: `Classification` → `Classification | None`

---

### FIX-002: DepthResult field name and save() signature

**Discrepancy:** DISC-004

**Before:**

```python
@dataclass
class DepthResult:
    depth_map: np.ndarray  # (H, W) float array
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, output_path: str, colormap: str = "magma"): ...
```

**After:**

```python
@dataclass(frozen=True)
class DepthResult:
    depth: np.ndarray           # (H, W) raw depth float array
    normalized: np.ndarray | None = None  # Optional [0, 1] normalized
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def depth_map(self) -> np.ndarray: ...  # Returns normalized if available, else depth

    def save(self, output_path: str | Path, image: ... = None,
             format: str | None = None, **kwargs): ...
```

**Changes:**

1. Primary field: `depth_map` → `depth` (with `depth_map` as property)
2. Added `normalized` field
3. `@dataclass` → `@dataclass(frozen=True)`
4. `save()` signature: removed `colormap`, added `image`, `format`, `**kwargs`

---

### FIX-003: EmbedResult `labels` field default

**Discrepancy:** DISC-005

**Before:**

```python
labels: list[str] = field(default_factory=list)
```

**After:**

```python
labels: list[str] | None = None
```

---

### FIX-004: `adapters/base/` → `adapters/base.py`

**Discrepancy:** DISC-007

**Before (in "Testing New Task Adapters"):**

```
1. Inherit from appropriate base adapter in `adapters/base/`
```

**After:**

```
1. Inherit from appropriate base adapter in `adapters/base.py`
```

---

## P1 Fixes — Misleading

### FIX-005: UniversalLoader detection chain — update to match actual strategies

**Discrepancy:** DISC-006

**Before:**

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

**After:**

```python
def _detect_source_type(self, task: str, source: Optional[str]) -> tuple[str, str]:
    """
    1. None → "default" (registry.get_default())
    2. Config alias (registry.has_alias()) → "config_alias"
    3. Local file (os.path.exists()) → "local_file"
    4. Known extension (.pt/.pth/.onnx/.bin/.trt/.engine) → "local_file"
    5. Starts with "torchvision/" → "torchvision"
    6. External OCR engine (easyocr/paddleocr/tesseract) → "external_engine"
    7. External barcode engine (pyzbar/zxing) → "external_engine"
    8. Contains '/' → "huggingface" (org/model pattern)
    9. Otherwise → "config_alias" (will raise ModelNotFoundError)
    """
```

---

### FIX-006: `load()` task list — add missing tasks

**Discrepancy:** DISC-011

**Before (in load() docstring section):**

```
task: Task type ("detect", "segment", "classify", "depth", "track")
```

**After:**

```
task: Task type ("detect", "segment", "classify", "depth", "track",
                 "vlm", "ocr", "barcode", "embed")
```

_Note: "recognize" is handled by `run()` only, not `load()`._

---

### FIX-007: Clarify `load()` parameter naming

**Discrepancy:** DISC-008

No doc change required — the section accurately describes the _internal_ `UniversalLoader._detect_source_type` which does use `source`. The public `mata.load()` maps `model` → `source` internally. Consider adding a note:

**After (add note):**

```
**Note:** The public API `mata.load(task, model=...)` passes `model` as `source` to the UniversalLoader internally.
```

---

### FIX-008: ClassifyResult `top1` return type

**Discrepancy:** DISC-003

_(Covered in FIX-001 above)_

---

## P2 Fixes — Cosmetic

### FIX-009: Version pinning "As of v1.9.5"

**Discrepancy:** DISC-010

**Before:**

```
As of v1.9.5, it features...
```

**After:**

```
It features...
```

_Rationale: Avoid pinning to a version that quickly becomes stale._

---

### FIX-010: EmbedResult version label "v1.9.6"

**Discrepancy:** DISC-009

**Before:**

```
**EmbedResult (🆕 v1.9.6):**
```

**After:**

```
**EmbedResult (🆕 v1.9.2b2):**
```

_Rationale: The embed feature was introduced in v1.9.2b2 per the Universal Loading section above. The "v1.9.6" label is inconsistent with the rest of the doc._

---

## Application Order

Apply in this order to avoid conflicts:

1. **FIX-001** — ClassifyResult (P0)
2. **FIX-002** — DepthResult (P0)
3. **FIX-003** — EmbedResult labels (P0)
4. **FIX-004** — adapters/base/ path (P0)
5. **FIX-005** — Detection chain (P1)
6. **FIX-006** — Task list (P1)
7. **FIX-009** — Version pinning (P2)
8. **FIX-010** — EmbedResult version label (P2)
