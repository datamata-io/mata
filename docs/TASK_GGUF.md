# GGUF Model Loading — Implementation Tasks

> **Project**: MATA — GGUF / llama-cpp-python Runtime Support
> **Feature**: Load quantized `.gguf` files for `vlm`, `embed`, and `classify` tasks via `llama-cpp-python`
> **Timeline**: 2 weeks (Estimated)
> **Team**: 1 developer
> **Status**: Planning Phase

## Progress Summary

### Completed ✅

_None._

### In Progress 🔄

_None._

### Pending ⏳

- **Phase A**: Foundation — `ModelType.GGUF` enum + optional dependency
- **Phase B**: Adapters — `LlamaCppBaseAdapter`, `LlamaCppVLMAdapter`, `LlamaCppEmbedAdapter`, `LlamaCppClassifyAdapter`
- **Phase C**: Loader Integration — extension detection, `_load_from_file()` dispatch, explicit-type routing, export
- **Phase F**: Testing — unit tests per adapter, loader integration tests, regression
- **Phase G**: Documentation & Examples — install guide, examples, CHANGELOG

### Metrics

- **Tests**: Target 80+ new tests (base adapter, 3 task adapters, loader integration)
- **Code Coverage**: Maintain >80% across new code; all mocked (no real GGUF models in CI)
- **Regression**: 4307+ existing tests must pass with zero regressions
- **New Files**: 7 files (1 base adapter, 3 task adapters, 3 test files)
- **Modified Files**: 5 files (`types.py`, `model_loader.py`, `adapters/__init__.py`, `pyproject.toml`, `tests/test_universal_loader.py`)

---

## Issue / Bottleneck Summary

| ID  | Severity | Location                                      | Description                                                                                                                                                                    |
| --- | -------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| G-1 | High     | `llama-cpp-python` (external)                 | GPU build requires CUDA + cmake at install time; CPU-only build is the safe default (`n_gpu_layers=0`)                                                                         |
| G-2 | High     | `src/mata/adapters/llamacpp_vlm_adapter.py`   | LLaVA-style multimodal GGUFs require a separate `mmproj` (projector) file; newer models (Qwen2-VL GGUF) bundle vision — adapter must handle both patterns                      |
| G-3 | Medium   | `src/mata/adapters/llamacpp_embed_adapter.py` | `llama-cpp-python` embedding mode only works if model was compiled with `embedding=True`; not all GGUF files support this — clear `UnsupportedModelError` required             |
| G-4 | Medium   | `src/mata/core/model_loader.py` L370-378      | `.gguf` extension is not in the extension whitelist for the "looks like local file" detection — must be added alongside `.pt`, `.onnx`, etc.                                   |
| G-5 | Low      | `pyproject.toml`                              | `llama-cpp-python` has platform-specific wheels (Windows/macOS/Linux); pip install instructions differ for CUDA vs CPU                                                         |
| G-6 | Low      | `src/mata/core/model_loader.py` L876          | `UnsupportedModelError` in `_load_from_file()` lists `.gguf` as unsupported; error message must be updated once support lands                                                  |
| G-7 | Info     | `docs/plans/TASK_VLM_EXPANSION.md`            | VLM expansion is in-progress and also modifies `model_loader.py` (L637-653, HF path) and `huggingface_vlm_adapter.py` — no overlap with GGUF paths, but coordinate merge order |

---

## Architectural Decisions

| Decision                       | Choice                                                                               | Rationale                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Runtime backend                | `llama-cpp-python` (local process)                                                   | Most mature Python binding for llama.cpp; supports multimodal via `LlavaLogitProcessor`; no HTTP server needed                   |
| Base class                     | Extends `BaseAdapter` directly (not `PyTorchBaseAdapter`)                            | GGUF runtime has no PyTorch dependency; same isolation design as `ONNXBaseAdapter`                                               |
| Supported tasks                | `vlm`, `embed`, `classify` only                                                      | No mainstream GGUF ecosystem for `detect`, `segment`, `depth`, or `track`; adding unsupported-task error is cleaner than silence |
| Embed adapter wiring           | `LlamaCppEmbedAdapter` implements `ReIDAdapter` interface, wrapped by `EmbedAdapter` | Reuses `EmbedAdapter`'s L2 normalization and `(N, D)` array contract; no new result type needed                                  |
| Classify adapter approach      | Cosine similarity between image embedding and text prompt embeddings                 | Matches CLIP zero-shot pattern from `HuggingFaceCLIPAdapter`; requires `embedding=True` GGUF model; returns `ClassifyResult`     |
| GPU default                    | `n_gpu_layers=0` (CPU-only)                                                          | Avoids CUDA build requirement surprises; opt-in via `n_gpu_layers=-1`                                                            |
| Optional dependency group name | `gguf` (CPU) + `gguf-gpu` (CUDA offload note)                                        | Mirrors `onnx` / `onnx-gpu` naming convention                                                                                    |
| No new result types            | VLM → `VisionResult`, embed → `ndarray`, classify → `ClassifyResult`                 | All target tasks already have result contracts; no new `to_json()` / `save()` work needed                                        |
| VLMWrapper compatibility       | `LlamaCppVLMAdapter` is wrapped by existing `VLMWrapper` unchanged                   | `VLMWrapper` wraps any adapter with `predict()` — no changes to graph system required                                            |

---

## Conflict Analysis

| Area                                           | Risk | Mitigation                                                                                                                                                                                                                 |
| ---------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_loader.py` (GGUF vs TASK_VLM_EXPANSION) | Low  | VLM expansion only modifies `_load_from_huggingface()` (L637-653); GGUF changes are in `_load_from_file()` (L772+), `_detect_source_type()` (L375), and `_load_with_explicit_type()` (L328+) — non-overlapping line ranges |
| `types.py` `ModelType` enum                    | Low  | Additive only — insert `GGUF = "gguf"` after `ZXING = "zxing"` at L1028; no existing values change                                                                                                                         |
| `adapters/__init__.py`                         | Low  | Additive export; no existing exports affected                                                                                                                                                                              |
| `pyproject.toml`                               | Low  | New `[gguf]` optional-dependency block; no existing blocks modified                                                                                                                                                        |
| `tests/test_universal_loader.py`               | Low  | Additive new test functions; existing tests unaffected                                                                                                                                                                     |

---

## Task Assignment Guide

### 🔴 Critical Path (Must complete in order)

```
A1 (ModelType.GGUF enum)
  → B1 (LlamaCppBaseAdapter)
    → B2 (LlamaCppVLMAdapter)  ┐
    → B3 (LlamaCppEmbedAdapter) ├─ parallel once B1 done
    → B4 (LlamaCppClassifyAdapter) ┘
      → C1 (_detect_source_type + _load_from_file dispatch)
        → C2 (_load_with_explicit_type + _validate_adapter_kwargs)
          → F1 (base adapter unit tests)
          → F2 (VLM adapter unit tests)
          → F3 (embed + classify unit tests)
          → F4 (loader integration tests in test_universal_loader.py)
            → F5 (full regression suite)
```

### 🟡 Parallel Work (Can work simultaneously)

- **A2** (pyproject.toml dep) — trivial, parallel with A1 and B1
- **B2**, **B3**, **B4** — independent adapters, parallel once **B1** is done
- **C3** (`adapters/__init__.py` exports) — parallel with **C1**/**C2**
- **F1**, **F2**, **F3**, **F4** — parallel once respective adapter code exists
- **G1**–**G3** — documentation parallel with **F1**–**F4**

### 🟢 Post-Integration (After core components)

- **F5** (full regression suite) — only after all adapters and loader are merged
- **G3** (CHANGELOG) — last, after F5 confirms final test counts

---

## Phase A: Foundation

### Task A1: Add ModelType.GGUF to types.py 🔴

**Priority**: Critical
**Estimated time**: 2 hours
**Dependencies**: None
**Status**: ⏳ Pending

**Description**: Add the `GGUF = "gguf"` enum value to `ModelType` in `src/mata/core/types.py` and update the docstring to document its valid kwargs. This is the unblocking prerequisite for all loader and adapter work.

**Files to modify**:

- `src/mata/core/types.py` — Insert `GGUF` enum value after `ZXING` at line 1028; update class docstring

**Changes required**:

In the `ModelType` enum class docstring, add entry after `ZXING`:

```python
# GGUF quantized model file (.gguf) — llama-cpp-python runtime
# Source: Local file path to GGUF file
# Valid kwargs: model_path, n_gpu_layers, n_ctx, mmproj (VLM), text_prompts (classify)
# Requires: llama-cpp-python installed (pip install datamata[gguf])
```

Add enum value after `ZXING = "zxing"` (currently line 1028):

```python
# GGUF quantized model file (.gguf)
GGUF = "gguf"
```

**Constraints**:

- Do not modify any existing enum values — additive only
- The `normalize()` classmethod handles all string values via `cls(normalized)` — no changes needed there

**Acceptance Criteria**:

- ✅ `ModelType.GGUF` exists and equals string `"gguf"`
- ✅ `ModelType.normalize("gguf")` returns `ModelType.GGUF`
- ✅ `ModelType.normalize(ModelType.GGUF)` returns `ModelType.GGUF`
- ✅ All existing `ModelType` values (ONNX, PYZBAR, ZXING, etc.) remain unchanged
- ✅ Test: see Task F4

**After Implementation**: [To be filled after completion]

---

### Task A2: Add gguf optional dependency to pyproject.toml 🟡

**Priority**: High
**Estimated time**: 1 hour
**Dependencies**: None
**Status**: ⏳ Pending

**Description**: Add `llama-cpp-python>=0.3.0` as a new optional dependency group `gguf` in `pyproject.toml`, following the `barcode` / `onnx` group patterns. No existing groups are modified.

**Files to modify**:

- `pyproject.toml` — Add new optional dependency groups after `barcode-all` block (line ~120)

**Changes required**:

Insert after the `barcode-all` block (after line ~120):

```toml
# GGUF quantized model files — llama-cpp-python runtime (CPU build)
# For GPU offloading: install llama-cpp-python with CUDA support manually
# See: https://llama-cpp-python.readthedocs.io/en/latest/#installation
gguf = [
    "llama-cpp-python>=0.3.0",
]
```

Also add `datamata[gguf]` to the `dev` extras group comment listing (documentation only, not the dep itself — gguf is too large for default dev setup).

**Constraints**:

- `llama-cpp-python` GPU builds require CUDA at compile time — the `gguf` group installs CPU-only by default; document this
- Do NOT add to the `all` extras group (GPU dependency, too large for default)

**Acceptance Criteria**:

- ✅ `pip install datamata[gguf]` installs `llama-cpp-python>=0.3.0`
- ✅ `gguf` group does not appear in `all` extras
- ✅ All other optional groups remain unchanged

**After Implementation**: [To be filled after completion]

---

## Phase B: Adapters

### Task B1: Create LlamaCppBaseAdapter 🔴

**Priority**: Critical
**Estimated time**: 4 hours
**Dependencies**: Task A1
**Status**: ⏳ Pending

**Description**: Create the base adapter class for all llama-cpp-python adapters in a new file `src/mata/adapters/llamacpp_base.py`. Extends `BaseAdapter` directly (no PyTorch dependency). Follows the `ONNXBaseAdapter` pattern at `src/mata/adapters/onnx_base.py` for lazy import, device management, session creation, and error handling.

**Files to modify**:

- `src/mata/adapters/llamacpp_base.py` — Create new file

**Changes required**:

```python
"""llama-cpp-python base adapter for MATA framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.exceptions import UnsupportedModelError
from mata.core.logging import get_logger
from .base import BaseAdapter

logger = get_logger(__name__)

_llama_cpp = None
LLAMA_CPP_AVAILABLE = None


def _ensure_llama_cpp():
    """Lazy import for llama-cpp-python. Raises ImportError with install hint."""
    global _llama_cpp, LLAMA_CPP_AVAILABLE
    if _llama_cpp is None:
        try:
            import llama_cpp
            _llama_cpp = llama_cpp
            LLAMA_CPP_AVAILABLE = True
            logger.debug(f"llama-cpp-python loaded successfully")
        except ImportError:
            LLAMA_CPP_AVAILABLE = False
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install datamata[gguf]  "
                "For GPU offloading see: https://llama-cpp-python.readthedocs.io/en/latest/#installation"
            )
    return _llama_cpp


class LlamaCppBaseAdapter(BaseAdapter):
    """Base adapter for llama-cpp-python GGUF models.

    Attributes:
        llama_cpp: llama_cpp module (lazily loaded)
        model_path: Absolute path to the .gguf file
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU-only, -1 = all)
        n_ctx: Context window size in tokens
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        verbose: bool = False,
        threshold: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(threshold=threshold)
        self.llama_cpp = _ensure_llama_cpp()

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}. "
                f"Download a .gguf file from HuggingFace Hub or another source."
            )
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Expected .gguf file, got: {path.suffix}")

        self.model_path = str(path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose

    def _create_llm(self, **extra_kwargs) -> Any:
        """Create a llama_cpp.Llama instance. Subclasses call with task-specific kwargs."""
        return self.llama_cpp.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
            **extra_kwargs,
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "backend": "llama-cpp-python",
        }
```

**Constraints**:

- No `torch` or `transformers` imports — this must be a pure `llama-cpp-python` path
- `n_gpu_layers=0` is the safe default (CPU-only, avoids CUDA build dependency)

**Acceptance Criteria**:

- ✅ `LlamaCppBaseAdapter` exists in `src/mata/adapters/llamacpp_base.py`
- ✅ Constructor raises `FileNotFoundError` when `.gguf` path does not exist
- ✅ Constructor raises `ValueError` for non-`.gguf` extension
- ✅ `_ensure_llama_cpp()` raises `ImportError` with `pip install datamata[gguf]` hint when not installed
- ✅ `_create_llm()` passes `n_gpu_layers`, `n_ctx`, `verbose` to `llama_cpp.Llama`
- ✅ No torch import anywhere in this file
- ✅ Test: see Task F1

**After Implementation**: [To be filled after completion]

---

### Task B2: Create LlamaCppVLMAdapter 🟡

**Priority**: High
**Estimated time**: 5 hours
**Dependencies**: Task B1
**Status**: ⏳ Pending

**Description**: Create `src/mata/adapters/llamacpp_vlm_adapter.py` implementing the VLM `predict()` contract. Supports two multimodal patterns: (1) LLaVA-style with a separate `mmproj` file; (2) self-contained models like Qwen2-VL GGUF. Returns `VisionResult` with `.text` populated, matching `HuggingFaceVLMAdapter` output. The adapter is wrapped by the existing `VLMWrapper` from `src/mata/adapters/wrappers/vlm_wrapper.py` in the loader — no changes to `VLMWrapper` needed.

**Files to modify**:

- `src/mata/adapters/llamacpp_vlm_adapter.py` — Create new file

**Changes required**:

```python
"""llama-cpp-python VLM adapter for MATA framework."""

from __future__ import annotations

from typing import Any

from mata.core.types import VisionResult
from mata.core.logging import get_logger
from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppVLMAdapter(LlamaCppBaseAdapter):
    """VLM adapter for GGUF files via llama-cpp-python.

    Supports LLaVA-style multimodal (mmproj) and self-contained
    multimodal GGUF files.

    Args:
        model_path: Path to the .gguf VLM file
        mmproj: Optional path to the multimodal projector .gguf file
                (required for LLaVA-v1.5/1.6; not needed for Qwen2-VL GGUF)
        n_gpu_layers: Layers to offload to GPU (0 = CPU-only, -1 = all)
        n_ctx: Context window size in tokens
        max_tokens: Default max tokens to generate
    """

    task = "vlm"
    name = "llamacpp_vlm"

    def __init__(
        self,
        model_path: str,
        mmproj: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        max_tokens: int = 512,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, **kwargs)
        self.mmproj = mmproj
        self.max_tokens = max_tokens

        # Build Llama instance — add mmproj for multimodal if provided
        extra = {}
        if mmproj:
            extra["chat_handler"] = self.llama_cpp.llava_chat_handler.LlavaLogitsProcessor(
                clip_model_path=mmproj, verbose=self.verbose
            )
        self._llm = self._create_llm(**extra)

    def predict(
        self,
        image: Any,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> VisionResult:
        """Generate text response for an image + text prompt.

        Args:
            image: PIL Image, file path, or numpy array
            prompt: Text prompt/question (required)
            max_new_tokens: Override default max_tokens

        Returns:
            VisionResult with .text = generated response
        """
        from mata.core.exceptions import InvalidInputError

        if not prompt:
            raise InvalidInputError("prompt is required for VLM predict()")

        pil_image, image_path = self._load_image(image)

        import base64
        import io

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        n_tokens = max_new_tokens or self.max_tokens
        response = self._llm.create_chat_completion(messages=messages, max_tokens=n_tokens)
        text = response["choices"][0]["message"]["content"]

        return VisionResult(
            instances=[],
            text=text,
            prompt=prompt,
            meta={
                "model_path": self.model_path,
                "backend": "llama-cpp-python",
                "n_gpu_layers": self.n_gpu_layers,
                "image_path": image_path,
            },
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "mmproj": self.mmproj, "max_tokens": self.max_tokens})
        return d
```

**Constraints**:

- The `llama_cpp.llava_chat_handler.LlavaLogitsProcessor` API changed across `llama-cpp-python` versions; import must be guarded and tested against `>=0.3.0`
- Return type must be `VisionResult` (not `str`) for `VLMWrapper` and graph compatibility
- `_load_image()` is inherited from `BaseAdapter` — handles PIL, path, numpy

**Acceptance Criteria**:

- ✅ `predict(image, prompt="...")` returns `VisionResult` with non-empty `.text`
- ✅ `predict(image, prompt=None)` raises `InvalidInputError`
- ✅ `mmproj=None` loads without LlavaLogitsProcessor (self-contained model path)
- ✅ `mmproj="path/to/projector.gguf"` wires `LlavaLogitsProcessor` into `_create_llm()`
- ✅ `VLMWrapper(LlamaCppVLMAdapter(...))` instantiates without error (duck-type compatible)
- ✅ Test: see Task F2

**After Implementation**: [To be filled after completion]

---

### Task B3: Create LlamaCppEmbedAdapter 🟡

**Priority**: High
**Estimated time**: 4 hours
**Dependencies**: Task B1
**Status**: ⏳ Pending

**Description**: Create `src/mata/adapters/llamacpp_embed_adapter.py` implementing the `ReIDAdapter` interface so it can be wrapped by the existing `EmbedAdapter`. Uses `llama_cpp.Llama(embedding=True)` mode for CLIP-GGUF models. Returns L2-normalized `(N, D)` float32 embeddings — matching `ONNXReIDAdapter` and `HuggingFaceReIDAdapter` output.

**Files to modify**:

- `src/mata/adapters/llamacpp_embed_adapter.py` — Create new file

**Changes required**:

```python
"""llama-cpp-python embed adapter for MATA framework."""

from __future__ import annotations

import numpy as np
from typing import Any

from mata.core.exceptions import UnsupportedModelError
from mata.core.logging import get_logger
from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppEmbedAdapter(LlamaCppBaseAdapter):
    """Embedding adapter for GGUF files via llama-cpp-python.

    Uses llama_cpp.Llama(embedding=True) mode — works with CLIP GGUF
    and other embedding-capable GGUF models.

    Implements the ReIDAdapter duck-type interface so it can be wrapped
    by EmbedAdapter for the public embed task API.
    """

    task = "embed"
    name = "llamacpp_embed"

    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 512, **kwargs: Any):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, **kwargs)
        self._embedding_dim: int | None = None
        try:
            self._llm = self._create_llm(embedding=True)
        except Exception as e:
            raise UnsupportedModelError(
                f"GGUF model '{model_path}' does not support embedding mode. "
                f"Ensure the model is a CLIP or embedding-capable GGUF file. "
                f"Original error: {e}"
            ) from e

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized embeddings for a batch of image crops.

        Args:
            crops: List of (H, W, 3) uint8 numpy arrays

        Returns:
            (N, D) float32 L2-normalized embedding array
        """
        from PIL import Image

        embeddings = []
        for crop in crops:
            pil = Image.fromarray(crop)
            # llama_cpp embed() accepts image bytes or token lists depending on version
            embedding = self._llm.embed(pil)
            embeddings.append(np.array(embedding, dtype=np.float32))

        result = np.stack(embeddings, axis=0)  # (N, D)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result = result / norms

        self._embedding_dim = result.shape[1]
        return result

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "embedding_dim": self._embedding_dim})
        return d
```

**Constraints**:

- Must implement `predict(crops: list[np.ndarray]) -> np.ndarray` + `embedding_dim` property to satisfy `EmbedAdapter`'s duck-type expectations (see `src/mata/adapters/embed_adapter.py`)
- Do not subclass `ReIDAdapter` directly — it inherits `PyTorchBaseAdapter` which imports torch

**Acceptance Criteria**:

- ✅ `LlamaCppEmbedAdapter` can be passed to `EmbedAdapter(encoder=...)` without error
- ✅ `predict([crop1, crop2])` returns `(2, D)` float32 L2-normalized array
- ✅ `embedding_dim` is `None` before first call, integer `D` after first call
- ✅ Constructor raises `UnsupportedModelError` with helpful message when model doesn't support `embedding=True`
- ✅ Test: see Task F3

**After Implementation**: [To be filled after completion]

---

### Task B4: Create LlamaCppClassifyAdapter 🟡

**Priority**: High
**Estimated time**: 4 hours
**Dependencies**: Task B1
**Status**: ⏳ Pending

**Description**: Create `src/mata/adapters/llamacpp_classify_adapter.py` for zero-shot classification via CLIP-GGUF cosine similarity. Matches the pattern of `HuggingFaceCLIPAdapter` at `src/mata/adapters/clip_adapter.py`. Returns `ClassifyResult` sorted by score descending.

**Files to modify**:

- `src/mata/adapters/llamacpp_classify_adapter.py` — Create new file

**Changes required**:

```python
"""llama-cpp-python classify adapter for MATA framework."""

from __future__ import annotations

import numpy as np
from typing import Any

from mata.core.types import ClassifyResult, Classification
from mata.core.exceptions import InvalidInputError
from mata.core.logging import get_logger
from .llamacpp_base import LlamaCppBaseAdapter

logger = get_logger(__name__)


class LlamaCppClassifyAdapter(LlamaCppBaseAdapter):
    """Zero-shot classification adapter for CLIP GGUF files.

    Computes cosine similarity between image embedding and
    text prompt embeddings. Requires a CLIP-capable GGUF model
    (embedding=True mode).

    Args:
        model_path: Path to CLIP GGUF file
        text_prompts: Class labels for zero-shot classification
    """

    task = "classify"
    name = "llamacpp_classify"

    def __init__(
        self,
        model_path: str,
        text_prompts: list[str] | None = None,
        n_gpu_layers: int = 0,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=512, **kwargs)
        self.text_prompts = text_prompts or []
        self._llm = self._create_llm(embedding=True)

    def predict(self, image: Any, text_prompts: list[str] | None = None, **kwargs: Any) -> ClassifyResult:
        """Classify image via cosine similarity against text prompts.

        Args:
            image: PIL Image, file path, or numpy array
            text_prompts: Override constructor text_prompts for this call

        Returns:
            ClassifyResult with classifications sorted by score descending
        """
        prompts = text_prompts or self.text_prompts
        if not prompts:
            raise InvalidInputError(
                "text_prompts required for GGUF classify. "
                "Pass at load time: mata.load('classify', 'model.gguf', text_prompts=[...]) "
                "or at run time: mata.run('classify', 'image.jpg', text_prompts=[...])"
            )

        pil_image, _ = self._load_image(image)
        from PIL import Image as PILImage

        img_emb = np.array(self._llm.embed(pil_image), dtype=np.float32)
        img_emb /= np.linalg.norm(img_emb) + 1e-8

        classifications = []
        for label in prompts:
            text_emb = np.array(self._llm.embed(label), dtype=np.float32)
            text_emb /= np.linalg.norm(text_emb) + 1e-8
            score = float(np.dot(img_emb, text_emb))
            classifications.append(Classification(label=label, score=max(0.0, score)))

        classifications.sort(key=lambda c: c.score, reverse=True)
        return ClassifyResult(
            classifications=classifications,
            meta={"model_path": self.model_path, "backend": "llama-cpp-python"},
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"task": self.task, "text_prompts": self.text_prompts})
        return d
```

**Constraints**:

- If `text_prompts` is empty both at constructor and call time, raise `InvalidInputError` — do not silently return empty result
- Cosine similarity may return negative values for dissimilar pairs; clamp to `max(0.0, score)` for consistency with `ClassifyResult` score convention `[0.0, 1.0]`

**Acceptance Criteria**:

- ✅ `predict(image, text_prompts=["cat","dog"])` returns `ClassifyResult` with 2 `Classification` items
- ✅ `top1` is the highest-scoring label
- ✅ `predict(image)` with no prompts anywhere raises `InvalidInputError` with install-hint message
- ✅ Scores are in `[0.0, 1.0]` (clamped, not raw cosine)
- ✅ Test: see Task F3

**After Implementation**: [To be filled after completion]

---

## Phase C: Loader Integration

### Task C1: Add .gguf detection and dispatch in model_loader.py 🔴

**Priority**: Critical
**Estimated time**: 3 hours
**Dependencies**: Tasks A1, B1, B2, B3, B4
**Status**: ⏳ Pending

**Description**: Wire `.gguf` files into the UniversalLoader in `src/mata/core/model_loader.py`. Two changes required: (1) add `.gguf` to the extension-based local-file detection so paths like `"model.gguf"` are recognized before checking HuggingFace slash-detection; (2) add a `.gguf` dispatch branch inside `_load_from_file()` that routes to the three new adapters by task.

**Files to modify**:

- `src/mata/core/model_loader.py` — Two edits: `_detect_source_type()` ~L375 and `_load_from_file()` ~L872

**Changes required**:

**Edit 1** — `_detect_source_type()`, the extension whitelist check (currently `[".pt", ".pth", ".onnx", ".bin", ".trt", ".engine"]` around L375):

```python
# Before:
if path.suffix.lower() in [".pt", ".pth", ".onnx", ".bin", ".trt", ".engine"]:

# After:
if path.suffix.lower() in [".pt", ".pth", ".onnx", ".bin", ".trt", ".engine", ".gguf"]:
```

**Edit 2** — `_load_from_file()`, insert new `elif` branch for `.gguf` before the final `else: raise UnsupportedModelError` (before line 874):

```python
elif extension in [".gguf"]:
    if task == "vlm":
        from mata.adapters.llamacpp_vlm_adapter import LlamaCppVLMAdapter
        from mata.adapters.wrappers.vlm_wrapper import VLMWrapper

        adapter = LlamaCppVLMAdapter(model_path=file_path, **kwargs)
        return VLMWrapper(adapter)
    elif task == "embed":
        from mata.adapters.llamacpp_embed_adapter import LlamaCppEmbedAdapter
        from mata.adapters.embed_adapter import EmbedAdapter

        encoder = LlamaCppEmbedAdapter(model_path=file_path, **kwargs)
        return EmbedAdapter(encoder=encoder)
    elif task == "classify":
        from mata.adapters.llamacpp_classify_adapter import LlamaCppClassifyAdapter

        return LlamaCppClassifyAdapter(model_path=file_path, **kwargs)
    else:
        raise UnsupportedModelError(
            f"GGUF models are not supported for task '{task}'. "
            f"Supported tasks: vlm, embed, classify. "
            f"For detect/segment, use an ONNX or HuggingFace model instead."
        )
```

**Edit 3** — Update the final `raise UnsupportedModelError` message at line 876 to include `.gguf`:

```python
# Before:
f"Unsupported file extension: {extension}. " f"Supported: .onnx, .pth, .pt, .bin, .trt, .engine"

# After:
f"Unsupported file extension: {extension}. " f"Supported: .onnx, .pth, .pt, .bin, .trt, .engine, .gguf"
```

**Constraints**:

- The `.gguf` branch in `_load_from_file()` must come before the final `else: raise` block
- The `track` task early-return at line 784 only routes to `_load_from_file("detect", ...)` internally — no `.gguf` track support is needed or expected

**Acceptance Criteria**:

- ✅ `UniversalLoader()._detect_source_type("vlm", "model.gguf")` returns `("local_file", "model.gguf")`
- ✅ `UniversalLoader()._detect_source_type("vlm", "/abs/path/model.gguf")` returns `("local_file", "/abs/path/model.gguf")`
- ✅ `mata.load("vlm", "model.gguf")` (mocked) dispatches to `LlamaCppVLMAdapter`
- ✅ `mata.load("detect", "model.gguf")` raises `UnsupportedModelError` with helpful message listing supported tasks
- ✅ Final error message includes `.gguf` in list of supported extensions
- ✅ Test: see Task F4

**After Implementation**: [To be filled after completion]

---

### Task C2: Add ModelType.GGUF routing in \_load_with_explicit_type and \_validate_adapter_kwargs 🟡

**Priority**: High
**Estimated time**: 2 hours
**Dependencies**: Tasks A1, C1
**Status**: ⏳ Pending

**Description**: Add `ModelType.GGUF` handling to two methods in `src/mata/core/model_loader.py`: (1) `_load_with_explicit_type()` so users can pass `model_type=ModelType.GGUF` to bypass auto-detection; (2) `_validate_adapter_kwargs()` to document valid GGUF kwargs. Also update the `ModelType` docstring in `model_loader.py` (lines 73-79) to mention GGUF.

**Files to modify**:

- `src/mata/core/model_loader.py` — Two edits: `_load_with_explicit_type()` at ~L328 and `_validate_adapter_kwargs()` at ~L179

**Changes required**:

**Edit 1** — `_load_with_explicit_type()`, insert after `elif model_type == ModelType.ZXING:` block (after line ~L332), before the final `else: raise`:

```python
elif model_type == ModelType.GGUF:
    if not source or not self._is_local_file(source):
        raise ModelNotFoundError(
            f"Valid GGUF file required when model_type=GGUF. Got: {source}"
        )
    return self._load_from_file(task, source, **kwargs)
```

**Edit 2** — `_validate_adapter_kwargs()`, add `ModelType.GGUF` to the `ADAPTER_KWARGS` dict (after `ModelType.TENSORRT` entry, ~L180):

```python
ModelType.GGUF: {"model_path", "n_gpu_layers", "n_ctx", "mmproj", "text_prompts", "max_tokens", "verbose"},
```

**Constraints**:

- `_load_with_explicit_type()` for GGUF simply delegates to `_load_from_file()` which already handles task routing — no duplication

**Acceptance Criteria**:

- ✅ `mata.load("vlm", "model.gguf", model_type=ModelType.GGUF)` routes correctly
- ✅ `mata.load("vlm", None, model_type=ModelType.GGUF)` raises `ModelNotFoundError`
- ✅ `mata.load("vlm", "model.gguf", model_type=ModelType.GGUF, extra_unknown=True)` emits warning listing valid GGUF kwargs
- ✅ Test: see Task F4

**After Implementation**: [To be filled after completion]

---

### Task C3: Export adapters from adapters/**init**.py 🟡

**Priority**: Low
**Estimated time**: 1 hour
**Dependencies**: Tasks B1, B2, B3, B4
**Status**: ⏳ Pending

**Description**: Export the four new adapter classes from `src/mata/adapters/__init__.py` so they are importable as `from mata.adapters import LlamaCppVLMAdapter` etc. Follow the existing export pattern.

**Files to modify**:

- `src/mata/adapters/__init__.py` — Add 4 export lines and update `__all__`

**Changes required**:

Locate where `EmbedAdapter` and OCR/Barcode adapters are exported (search for `from mata.adapters.embed_adapter`) and add after the same block:

```python
# GGUF / llama-cpp-python adapters (v1.9.4+)
from mata.adapters.llamacpp_base import LlamaCppBaseAdapter
from mata.adapters.llamacpp_vlm_adapter import LlamaCppVLMAdapter
from mata.adapters.llamacpp_embed_adapter import LlamaCppEmbedAdapter
from mata.adapters.llamacpp_classify_adapter import LlamaCppClassifyAdapter
```

Add the four names to `__all__`.

**Constraints**:

- Imports are always present (not guarded by `try/except`) — same as ONNX adapters; the `ImportError` is raised lazily inside `_ensure_llama_cpp()` only when the adapter is actually instantiated

**Acceptance Criteria**:

- ✅ `from mata.adapters import LlamaCppVLMAdapter` works
- ✅ All four adapter names appear in `mata.adapters.__all__`
- ✅ Import does NOT raise `ImportError` when `llama-cpp-python` is not installed (lazy import)

**After Implementation**: [To be filled after completion]

---

## Phase F: Testing

### Task F1: Unit tests for LlamaCppBaseAdapter 🟡

**Priority**: High
**Estimated time**: 4 hours
**Dependencies**: Task B1
**Status**: ⏳ Pending

**Description**: Create `tests/test_llamacpp_base.py` with unit tests for `LlamaCppBaseAdapter` and `_ensure_llama_cpp()`. All tests use `unittest.mock` — no real GGUF models required.

**Files to modify**:

- `tests/test_llamacpp_base.py` — Create new file (~25 tests)

**Changes required**:

Test coverage targets:

```
_ensure_llama_cpp():
  - Returns llama_cpp module when available
  - Raises ImportError with "pip install datamata[gguf]" hint when not installed
  - Caches module after first import (global _llama_cpp is reused)

LlamaCppBaseAdapter.__init__():
  - Accepts valid .gguf path (mocked file existence) → no error
  - Raises FileNotFoundError for nonexistent path
  - Raises ValueError for .onnx extension (wrong file type)
  - Passes n_gpu_layers, n_ctx, verbose to _create_llm()
  - Default n_gpu_layers=0 (CPU-only)

_create_llm():
  - Calls llama_cpp.Llama with correct kwargs
  - Forwards extra_kwargs (e.g., embedding=True)

info():
  - Returns dict with model_path, n_gpu_layers, n_ctx, backend="llama-cpp-python"
```

**Constraints**:

- Mock `llama_cpp` module using `unittest.mock.patch` — do not require it to be installed in test environment
- Use `tmp_path` pytest fixture for temporary `.gguf` file creation (empty file is fine for path validation)

**Acceptance Criteria**:

- ✅ All ~25 tests pass with `pytest tests/test_llamacpp_base.py -v`
- ✅ No `ImportError` for `llama_cpp` during test collection
- ✅ `FileNotFoundError` test verified with a nonexistent path
- ✅ `ValueError` test verified with a `.onnx` extension path

**After Implementation**: [To be filled after completion]

---

### Task F2: Unit tests for LlamaCppVLMAdapter 🟡

**Priority**: High
**Estimated time**: 4 hours
**Dependencies**: Tasks B2, F1
**Status**: ⏳ Pending

**Description**: Create `tests/test_llamacpp_vlm_adapter.py` with unit tests for `LlamaCppVLMAdapter`. Mocks `llama_cpp.Llama` and `create_chat_completion`. Also tests `VLMWrapper(LlamaCppVLMAdapter(...))` integration.

**Files to modify**:

- `tests/test_llamacpp_vlm_adapter.py` — Create new file (~25 tests)

**Changes required**:

Test coverage targets:

```
LlamaCppVLMAdapter.__init__():
  - Without mmproj: Llama created without chat_handler kwarg
  - With mmproj: Llama created with LlavaLogitsProcessor wired in
  - Inherits FileNotFoundError from base for missing model_path

predict():
  - Returns VisionResult with .text from chat completion response
  - .text is the "content" field from choices[0].message
  - Raises InvalidInputError when prompt=None
  - Raises InvalidInputError when prompt=""
  - max_new_tokens kwarg overrides constructor default
  - image loading works for file path (uses _load_image → PIL)
  - meta dict contains model_path and backend="llama-cpp-python"

VLMWrapper integration:
  - VLMWrapper(LlamaCppVLMAdapter(...)) instantiates without TypeError
  - wrapper.query(image, "prompt") delegates to adapter.predict()
```

**Acceptance Criteria**:

- ✅ All ~25 tests pass with `pytest tests/test_llamacpp_vlm_adapter.py -v`
- ✅ `InvalidInputError` for missing prompt is verified
- ✅ `VLMWrapper` duck-type compatibility verified without full VLMWrapper instantiation of HF models

**After Implementation**: [To be filled after completion]

---

### Task F3: Unit tests for LlamaCppEmbedAdapter and LlamaCppClassifyAdapter 🟡

**Priority**: High
**Estimated time**: 5 hours
**Dependencies**: Tasks B3, B4, F1
**Status**: ⏳ Pending

**Description**: Create `tests/test_llamacpp_embed_classify.py` with unit tests for both `LlamaCppEmbedAdapter` and `LlamaCppClassifyAdapter`. Mocks `llama_cpp.Llama.embed()`.

**Files to modify**:

- `tests/test_llamacpp_embed_classify.py` — Create new file (~30 tests)

**Changes required**:

Test coverage targets for `LlamaCppEmbedAdapter`:

```
__init__():
  - Llama session created with embedding=True
  - UnsupportedModelError raised when Llama raises on embedding=True

predict(crops):
  - Returns (N, D) float32 L2-normalized array for N crops
  - L2 norms of output rows are all ~1.0 (within 1e-5)
  - embedding_dim is None before first call; set to D after first call
  - Single crop returns (1, D) array
  - Empty crops list returns (0, D)? or raises? (document choice)

EmbedAdapter wrapping:
  - EmbedAdapter(encoder=LlamaCppEmbedAdapter(...)) does not error
```

Test coverage targets for `LlamaCppClassifyAdapter`:

```
__init__():
  - Stores text_prompts from constructor

predict():
  - Returns ClassifyResult with len(classifications) == len(text_prompts)
  - top1 is highest-scoring label
  - All scores in [0.0, 1.0]
  - Raises InvalidInputError when no text_prompts at constructor or call time
  - text_prompts at call time override constructor prompts
  - meta contains model_path and backend

Cosine similarity:
  - Test with known orthogonal embeddings → score ≈ 0.0
  - Test with known identical embeddings → score ≈ 1.0
```

**Acceptance Criteria**:

- ✅ All ~30 tests pass with `pytest tests/test_llamacpp_embed_classify.py -v`
- ✅ L2 normalization verified numerically
- ✅ `InvalidInputError` for missing text_prompts is verified for both constructor-empty and call-time-empty cases

**After Implementation**: [To be filled after completion]

---

### Task F4: Loader integration tests for .gguf detection 🟡

**Priority**: High
**Estimated time**: 3 hours
**Dependencies**: Tasks C1, C2, A1
**Status**: ⏳ Pending

**Description**: Add `.gguf` detection and dispatch tests to the existing `tests/test_universal_loader.py`. Also add `ModelType.GGUF` explicit-type tests.

**Files to modify**:

- `tests/test_universal_loader.py` — Add new test functions (no existing tests modified)

**Changes required**:

New test functions to add:

```python
# Extension detection
def test_detect_source_type_gguf_returns_local_file():
    # patch _is_local_file to return True for .gguf path
def test_detect_source_type_gguf_extension_without_file_on_disk():
    # Still returns "local_file" based on extension alone (like .onnx behavior)

# _load_from_file dispatch
def test_load_from_file_gguf_vlm_dispatches_to_llamacpp_vlm_adapter():
def test_load_from_file_gguf_embed_dispatches_to_llamacpp_embed_adapter():
def test_load_from_file_gguf_classify_dispatches_to_llamacpp_classify_adapter():
def test_load_from_file_gguf_detect_raises_unsupported_error():
def test_load_from_file_gguf_segment_raises_unsupported_error():
def test_load_from_file_gguf_error_message_lists_supported_tasks():

# ModelType.GGUF explicit routing
def test_load_with_explicit_type_gguf_routes_to_load_from_file():
def test_load_with_explicit_type_gguf_none_source_raises_model_not_found():

# _validate_adapter_kwargs
def test_validate_adapter_kwargs_gguf_unknown_kwarg_emits_warning():
def test_validate_adapter_kwargs_gguf_valid_kwargs_no_warning():

# ModelType enum
def test_model_type_gguf_value_is_gguf_string():
def test_model_type_normalize_gguf_string():
```

**Constraints**:

- All new tests mock the adapter constructors — they should not require `llama-cpp-python` to be installed
- Patch `LlamaCppVLMAdapter.__init__` and `LlamaCppEmbedAdapter.__init__` to avoid real GGUF loading

**Acceptance Criteria**:

- ✅ All new tests pass with `pytest tests/test_universal_loader.py -v`
- ✅ Zero existing tests in `test_universal_loader.py` fail
- ✅ 14+ new test functions added

**After Implementation**: [To be filled after completion]

---

### Task F5: Full Regression Suite Verification 🟢

**Priority**: Medium
**Estimated time**: 1 hour
**Dependencies**: Tasks F1, F2, F3, F4
**Status**: ⏳ Pending

**Description**: Run the complete test suite to verify zero regressions. The only valid outcome is all pre-existing tests passing plus all new GGUF tests passing.

**Files to modify**:

- None — verification only

**Changes required**:

```bash
pytest tests/ -v --tb=short 2>&1 | tee test_results_gguf.txt
```

Check:

1. All 4307+ pre-existing tests pass
2. All new `tests/test_llamacpp_*.py` tests pass
3. All new tests in `tests/test_universal_loader.py` pass
4. No `DeprecationWarning` from new code

**Acceptance Criteria**:

- ✅ Pre-existing test count unchanged (no deletions or renames)
- ✅ Zero pre-existing test failures
- ✅ All new tests (80+ target) pass
- ✅ No new `DeprecationWarning` from MATA code (third-party warnings are acceptable)

**After Implementation**: [To be filled after completion]

---

## Phase G: Documentation & Examples

### Task G1: Update INSTALLATION.md with GGUF install instructions 🟡

**Priority**: Low
**Estimated time**: 1 hour
**Dependencies**: Task A2
**Status**: ⏳ Pending

**Description**: Add a `GGUF Models` section to `INSTALLATION.md` explaining the `pip install datamata[gguf]` command, GPU variant, and platform notes.

**Files to modify**:

- `INSTALLATION.md` — Add new section after the Barcode section

**Changes required**:

````markdown
## GGUF Models (llama-cpp-python)

For loading `.gguf` quantized models (VLM, embed, classify):

```bash
# CPU-only (default — no CUDA required)
pip install datamata[gguf]

# GPU offloading (requires CUDA toolkit + cmake; see llama-cpp-python docs)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Platform notes:**

- **Windows**: CPU build works with the default pip wheel. GPU build requires Visual Studio Build Tools.
- **macOS**: Apple Silicon Metal acceleration is enabled automatically when using the default wheel.
- **Linux**: CPU build is straightforward. GPU build requires `nvidia-cuda-toolkit`.
````

**Acceptance Criteria**:

- ✅ CPU install command documented
- ✅ GPU install note with link/reference
- ✅ Platform notes for Windows, macOS, Linux

**After Implementation**: [To be filled after completion]

---

### Task G2: Add GGUF usage examples 🟡

**Priority**: Low
**Estimated time**: 2 hours
**Dependencies**: Tasks B2, B3, B4, C1
**Status**: ⏳ Pending

**Description**: Create two minimal usage examples under `examples/`. Follow the style of existing examples such as `examples/vlm/basic_vlm.py` and `examples/embed/`.

**Files to modify**:

- `examples/vlm/gguf_vlm.py` — Create new file
- `examples/embed/gguf_embed.py` — Create new file

**Changes required**:

`examples/vlm/gguf_vlm.py` — demonstrate `mata.load("vlm", "model.gguf")` with a docstring explaining how to download a GGUF VLM from HuggingFace Hub.

`examples/embed/gguf_embed.py` — demonstrate `mata.load("embed", "clip.gguf")` for feature extraction.

Both files must include:

- A top-level docstring explaining which GGUF model to download and from where
- A `# pip install datamata[gguf]` comment
- Guard: `if not Path("model.gguf").exists(): sys.exit("Download a .gguf file first")`

**Acceptance Criteria**:

- ✅ Both example files are syntactically valid (`python -m py_compile examples/vlm/gguf_vlm.py`)
- ✅ Each example has a clear download instruction in the docstring
- ✅ Neither example hard-codes an absolute path

**After Implementation**: [To be filled after completion]

---

### Task G3: Update CHANGELOG.md 🟢

**Priority**: Low
**Estimated time**: 1 hour
**Dependencies**: Task F5
**Status**: ⏳ Pending

**Description**: Add a `v1.9.4` entry to `CHANGELOG.md` documenting the GGUF feature. Use actual final test counts from Task F5.

**Files to modify**:

- `CHANGELOG.md` — Prepend new version entry

**Changes required**:

```markdown
## [v1.9.4] — 2026-XX-XX

### Added

- `mata.load("vlm", "model.gguf")` — Load quantized GGUF VLMs via `llama-cpp-python`
- `mata.load("embed", "clip.gguf")` — Embedding extraction from CLIP GGUF files
- `mata.load("classify", "clip.gguf", text_prompts=[...])` — Zero-shot classification via CLIP GGUF
- `ModelType.GGUF` added to loader explicit-type dispatch
- `LlamaCppBaseAdapter`, `LlamaCppVLMAdapter`, `LlamaCppEmbedAdapter`, `LlamaCppClassifyAdapter`
- Optional dependency: `pip install datamata[gguf]` (llama-cpp-python)
- GPU offloading via `n_gpu_layers=-1`; CPU default `n_gpu_layers=0`
- LLaVA-style models: `mata.load("vlm", "llava.gguf", mmproj="projector.gguf")`

### Tests

- [N]+ new tests, [TOTAL]+ total, all passing
```

**Acceptance Criteria**:

- ✅ Entry uses real test counts from F5 output
- ✅ Release date is filled in (not XX-XX)
- ✅ All four new public API patterns are documented

**After Implementation**: [To be filled after completion]

---

## Testing Checklist

### Unit Tests

- ⬜ `LlamaCppBaseAdapter` — file validation, lazy import, session creation (Task F1)
- ⬜ `LlamaCppVLMAdapter` — predict(), mmproj wiring, VLMWrapper duck-type (Task F2)
- ⬜ `LlamaCppEmbedAdapter` — embedding extraction, L2 normalization, EmbedAdapter wrapping (Task F3)
- ⬜ `LlamaCppClassifyAdapter` — cosine similarity, ClassifyResult construction, empty-prompts error (Task F3)
- ⬜ `_ensure_llama_cpp()` — import success, import failure with hint (Task F1)

### Integration Tests

- ⬜ `_detect_source_type()` recognizes `.gguf` as `"local_file"` (Task F4)
- ⬜ `_load_from_file()` dispatches `.gguf` to correct adapter per task (Task F4)
- ⬜ `_load_with_explicit_type()` with `ModelType.GGUF` routes correctly (Task F4)
- ⬜ Unsupported task + `.gguf` raises `UnsupportedModelError` with actionable message (Task F4)

### Regression Tests

- ⬜ Full suite: `pytest tests/ -v` — 4307+ existing tests all pass (Task F5)
- ⬜ No new `DeprecationWarning` from MATA code (Task F5)

### Manual Tests (Optional — requires llama-cpp-python + .gguf file)

- ⬜ `mata.load("vlm", "path/to/qwen2-vl-q4.gguf")` loads successfully
- ⬜ `result.text` is a non-empty string after predict()
- ⬜ `mata.load("vlm", "nonexistent.gguf")` raises `FileNotFoundError`
- ⬜ `mata.load("detect", "model.gguf")` raises `UnsupportedModelError`
- ⬜ `mata.load("embed", "clip-q8.gguf")` returns `EmbedAdapter` instance

---

## Definition of Done

- ✅ All 5 phases (A–C, F, G) tasks are marked Completed
- ✅ 80+ new tests, all passing
- ✅ Zero regressions in full suite (4307+ existing tests)
- ✅ `ModelType.GGUF` in types.py
- ✅ `.gguf` detected as `"local_file"` by `_detect_source_type()`
- ✅ `_load_from_file()` routes `vlm`, `embed`, `classify` `.gguf` to correct adapters
- ✅ `_load_with_explicit_type()` handles `ModelType.GGUF`
- ✅ `pyproject.toml` has `gguf = ["llama-cpp-python>=0.3.0"]`
- ✅ Install instructions in `INSTALLATION.md`
- ✅ Examples in `examples/vlm/gguf_vlm.py` and `examples/embed/gguf_embed.py`
- ✅ CHANGELOG.md entry with real test counts
- ✅ No torch import in any `llamacpp_*.py` file
- ✅ Lazy import pattern: `llama_cpp` only imported on first adapter instantiation
