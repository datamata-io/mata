# VLM Model Support (v1.9.3)

> **MATA** — Vision-Language Model compatibility reference for v1.9.3.  
> All models are loaded via `AutoModelForImageTextToText` (HuggingFace Transformers ≥ 5.0.0).

---

## Quick Start

```python
import mata

# Describe an image (default Qwen3-VL)
result = mata.run("vlm", "image.jpg",
                  model="Qwen/Qwen3-VL-2B-Instruct",
                  prompt="Describe this image.")
print(result.text)

# Medical imaging with MedGemma
vlm = mata.load("vlm", "google/medgemma-1.5-4b-it", dtype="bfloat16")
result = vlm.predict("xray.jpg", prompt="Describe the findings in this chest X-ray.")
print(result.text)

# Multilingual VQA with LLaVA-NeXT
vlm = mata.load("vlm", "llava-hf/llava-v1.6-mistral-7b-hf")
result = vlm.predict("scene.jpg", prompt="Describe this image.")
print(result.text)
```

---

## Supported Models

| Model              | HF ID                                 | Size    | Use Case                  | `dtype` Override | `trust_remote_code` | Status                                                 |
| ------------------ | ------------------------------------- | ------- | ------------------------- | ---------------- | ------------------- | ------------------------------------------------------ |
| **Qwen3-VL**       | `Qwen/Qwen3-VL-2B-Instruct`           | 2B / 7B | General VQA, grounding    | `"auto"`         | No                  | ✅ Shipped                                             |
| **MedGemma**       | `google/medgemma-1.5-4b-it`           | 4B      | Medical imaging           | `"bfloat16"`     | No                  | ✅ Tested                                              |
| **LFM2.5-VL**      | `LiquidAI/LFM2.5-VL-1.6B`             | 1.6B    | Lightweight general       | `"bfloat16"`     | No                  | ✅ Tested                                              |
| **SmolVLM**        | `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M    | Ultra-light edge / mobile | `"auto"`         | No                  | ✅ Tested                                              |
| **Florence-2**     | `florence-community/Florence-2-large` | 0.77B   | Grounding, captioning     | `"auto"`         | No                  | ✅ Tested (community model)                            |
| **PaliGemma 2**    | `google/paligemma2-3b-pt-224`         | 3B      | Document understanding    | `"bfloat16"`     | No                  | ✅ Tested (gated — HF login required)                  |
| **Phi-3.5 Vision** | `microsoft/Phi-3.5-vision-instruct`   | 4.2B    | Code, diagrams, charts    | `"auto"`         | Yes                 | ❌ Deferred (requires FlashAttention2 — Linux only)    |
| **InternVL2**      | `OpenGVLab/InternVL2-1B`              | 1B      | Multilingual VQA          | `"auto"`         | Yes                 | ❌ Not supported (Accelerate meta-device incompatible) |
| **LLaVA-1.5-7B**   | `llava-hf/llava-1.5-7b-hf`            | 7B      | High-quality VQA          | `"auto"`         | No                  | ✅ Tested                                              |
| **Moondream2**     | `vikhyatk/moondream2`                 | 1.9B    | Tiny / fast inference     | `"auto"`         | Yes                 | ❌ Not supported                                       |
| **DeepSeek OCR**   | `deepseek-ai/DeepSeek-OCR-2`          | —       | Document OCR              | —                | Yes                 | ❌ Deferred (requires FlashAttention2 — Linux only)    |

**Legend** — `dtype` Override column shows the recommended value for reliable loading:

- `"auto"` — Transformers selects the optimal dtype automatically (safe default)
- `"bfloat16"` — Required for models that use BFloat16 weights (see model notes below)

---

## Model-Specific Notes

### Qwen3-VL (General VQA / Grounding)

The default development and testing model. Supports grounding (bounding-box output), multi-image
comparison, and structured JSON with `output_mode="detect"`. No special flags required.

```python
# Standard loading
vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

# Grounding with structured JSON + auto-promotion
result = vlm.predict(
    "scene.jpg",
    prompt="Detect all cats with bounding boxes in JSON format.",
    output_mode="detect",
    auto_promote=True,
)
print(len(result.instances))  # Instance objects with xyxy bboxes
```

**Available sizes:** 2B (dev/testing), 7B (production quality)  
**Coordinate output:** ~1000-unit space — MATA scales automatically to pixel coords.

---

### MedGemma (Medical Imaging)

Google's medical imaging specialist. Trained on de-identified medical data (radiology, pathology,
ophthalmology). Requires `dtype="bfloat16"` for correct weight loading.

```python
vlm = mata.load("vlm", "google/medgemma-1.5-4b-it", dtype="bfloat16")

# Radiology
result = vlm.predict("chest_xray.jpg",
                     prompt="Describe findings in this chest X-ray.")

# Pathology
result = vlm.predict("pathology_slide.jpg",
                     prompt="Identify cell types visible in this histology slide.")
```

**Model ID:** `google/medgemma-1.5-4b-it`  
**Size:** 4B parameters  
**Required kwargs:** `dtype="bfloat16"`  
**Use cases:** Radiology report drafting, pathology slide analysis, ophthalmology screening  
**Limitations:** Not a medical device. Output should be reviewed by qualified clinicians.

---

### LFM2.5-VL (Lightweight)

LiquidAI's compact multimodal model optimised for fast inference. Good balance between size and
capability. Also requires `dtype="bfloat16"`.

```python
vlm = mata.load("vlm", "LiquidAI/LFM2.5-VL-1.6B", dtype="bfloat16")
result = vlm.predict("scene.jpg", prompt="What is in this image?")
```

**Model ID:** `LiquidAI/LFM2.5-VL-1.6B`  
**Size:** 1.6B parameters  
**Required kwargs:** `dtype="bfloat16"`  
**Use cases:** Resource-constrained servers, batch processing pipelines

---

### SmolVLM (Edge / Mobile)

HuggingFace's ultra-lightweight VLM optimised for on-device inference. At 256M parameters it can
run on CPU and small-memory devices.

```python
vlm = mata.load("vlm", "HuggingFaceTB/SmolVLM-256M-Instruct")
result = vlm.predict("image.jpg", prompt="Briefly describe this image.")

# Larger variant for better accuracy
vlm = mata.load("vlm", "HuggingFaceTB/SmolVLM-500M-Instruct")
```

**Model IDs:** `HuggingFaceTB/SmolVLM-256M-Instruct`, `HuggingFaceTB/SmolVLM-500M-Instruct`  
**Size:** 256M / 500M parameters  
**Required kwargs:** None  
**Use cases:** Edge devices, mobile deployment, CPU-only inference, high-throughput pipelines

---

### Florence-2 (Grounding / Captioning)

Microsoft's vision foundation model excelling at dense captioning, object detection, OCR, and
region-level understanding. The original `microsoft/Florence-2-large` repo uses custom code that
is incompatible with Transformers ≥ 5.0. MATA automatically redirects it to the official
`florence-community` port, which uses native Transformers classes and requires no
`trust_remote_code`.

```python
# Recommended: load community model directly (no trust_remote_code needed)
vlm = mata.load("vlm", "florence-community/Florence-2-large")
result = vlm.predict("scene.jpg", prompt="Describe in detail.")

# Legacy ID also works — MATA silently redirects to the community model
vlm = mata.load("vlm", "microsoft/Florence-2-large", trust_remote_code=True)

# Smaller variant
vlm = mata.load("vlm", "florence-community/Florence-2-base")
```

**Model IDs:** `florence-community/Florence-2-base`, `florence-community/Florence-2-large`  
**Size:** 0.23B / 0.77B parameters  
**Required kwargs:** None (community model); `trust_remote_code=True` accepted but ignored for the legacy IDs  
**Use cases:** Dense captioning, OCR, grounded object detection, document understanding  
**Note:** Uses an encoder-decoder architecture. MATA strips input-token trimming automatically
for encoder-decoder models so decoded output is complete.  
**Compatibility redirect:** `microsoft/Florence-2-*` → `florence-community/Florence-2-*` (Transformers ≥ 5.0).

---

### PaliGemma 2 (Document Understanding) | Gated Repository

Google's PaLI-Gemma 2 model family, strong at document understanding, fine-grained recognition,
and chart/table parsing. Tested and working in MATA v1.9.3.

```python
vlm = mata.load("vlm", "google/paligemma2-3b-pt-224", dtype="bfloat16")
result = vlm.predict("document.jpg", prompt="Extract all text from this document.")

# Higher resolution variant for dense documents
vlm = mata.load("vlm", "google/paligemma2-3b-pt-448", dtype="bfloat16")
```

**Model IDs:** `google/paligemma2-3b-pt-224`, `google/paligemma2-3b-pt-448`, `google/paligemma2-10b-pt-448`  
**Size:** 3B / 10B parameters  
**Required kwargs:** None  
**Recommended kwargs:** `dtype="bfloat16"` (reduces VRAM)  
**Use cases:** Document OCR, table extraction, chart reading, fine-grained image recognition  
**Note:** PaliGemma models are gated behind a licence agreement on HuggingFace.
Run `huggingface-cli login` and accept the licence before first use.

---

### Phi-3.5 Vision (Code / Diagrams) — Deferred

> ❌ **Phi-3.5 Vision is deferred from MATA v1.9.3 on Windows.**  
> Loading `microsoft/Phi-3.5-vision-instruct` requires **FlashAttention2** (`flash-attn ≥ 1.0.3`)
> for meta-device support under Accelerate. Pre-built wheels for FlashAttention2 are not
> available on Windows. The model will be validated and enabled in a future release after
> Linux/WSL2 testing is complete.

Microsoft's Phi-3.5 Vision is optimised for reasoning about code screenshots, architecture
diagrams, charts, and structured document content.

```python
# Linux / WSL2 only — requires: pip install flash-attn --no-build-isolation
vlm = mata.load("vlm", "microsoft/Phi-3.5-vision-instruct",
                trust_remote_code=True, dtype="bfloat16")
result = vlm.predict("diagram.png",
                     prompt="Explain this architecture diagram.")
```

**Model ID:** `microsoft/Phi-3.5-vision-instruct`  
**Size:** 4.2B parameters  
**Required kwargs:** `trust_remote_code=True`  
**Recommended kwargs:** `dtype="bfloat16"` (reduces VRAM, maintains quality)  
**Use cases:** Code screenshot analysis, UML/architecture diagrams, chart understanding, slide decks  
**Status:** Deferred to a future release pending Linux validation and a Windows-compatible
FlashAttention2 build.

---

### InternVL2 (Not Supported)

> ❌ **InternVL2 is not supported in MATA v1.9.3.**  
> The model's custom `__init__` calls `Tensor.item()` during construction, which is
> incompatible with Accelerate's meta-device dispatch (`device_map`). Loading without
> `device_map` also fails because neither `AutoModel` nor `AutoModelForCausalLM` can
> instantiate its config class. Use `llava-hf/llava-v1.6-mistral-7b-hf` or
> `Qwen/Qwen3-VL-2B-Instruct` as multilingual/high-quality alternatives.

---

### LLaVA-NeXT (High-Quality VQA)

LLaVA is a widely used open-source VLM architecture. The `-hf` variants on HuggingFace are
natively supported without `trust_remote_code`.

```python
vlm = mata.load("vlm", "llava-hf/llava-v1.6-mistral-7b-hf")
result = vlm.predict("scene.jpg", prompt="Describe this image in detail.")

# Larger Qwen-2-based variant
vlm = mata.load("vlm", "llava-hf/llava-onevision-qwen2-7b-ov-hf")
```

**Model IDs:** `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`,
`llava-hf/llava-onevision-qwen2-7b-ov-hf`  
**Size:** 7B+ parameters  
**Required kwargs:** None  
**Use cases:** High-quality open-ended VQA, detailed image descriptions, academic benchmarks  
**Note:** 7B models require at least 16 GB VRAM for `bfloat16` or 8 GB with 4-bit quantisation.

---

### Moondream2 (Tiny / Fast)

A 1.9B parameter model designed for minimal resource use while maintaining usable accuracy.
Requires `trust_remote_code=True` for its custom vision encoder.

```python
vlm = mata.load("vlm", "vikhyatk/moondream2",
                trust_remote_code=True,
                revision="2025-01-09")
result = vlm.predict("image.jpg", prompt="Describe this image.")
```

**Model ID:** `vikhyatk/moondream2`  
**Size:** 1.9B parameters  
**Required kwargs:** `trust_remote_code=True`  
**Recommended kwargs:** `revision="2025-01-09"` (pin to a known-good snapshot)  
**Use cases:** Embedded systems, CPU inference, rapid prototyping, low-latency applications

---

## Known Limitations

### DeepSeek OCR (Deferred)

`deepseek-ai/DeepSeek-OCR-2` is deferred from v1.9.3. The model depends on **FlashAttention2**
(`flash-attn`), which has no stable pre-built wheel for Windows. Attempting to load it on Windows
raises an `ImportError`.

**Workaround (Linux/WSL2):**

```bash
pip install flash-attn --no-build-isolation
```

```python
vlm = mata.load("vlm", "deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)
```

**Status:** Deferred to a future release pending a Windows-compatible FlashAttention2 build or an
alternative loading path that avoids the dependency.

---

## Configuration Examples (YAML)

Store frequently used VLMs in `.mata/models.yaml` (project-local) or `~/.mata/models.yaml`
(user-global) to use short alias names instead of full HuggingFace IDs.

```yaml
models:
  vlm:
    # Qwen3-VL — general purpose (no special flags)
    qwen3-vl:
      source: "Qwen/Qwen3-VL-2B-Instruct"
      device: "cuda"
      max_new_tokens: 512

    # MedGemma — medical imaging
    medgemma:
      source: "google/medgemma-1.5-4b-it"
      dtype: "bfloat16"
      device: "cuda"
      max_new_tokens: 300

    # LFM2.5-VL — lightweight
    lfm2:
      source: "LiquidAI/LFM2.5-VL-1.6B"
      dtype: "bfloat16"
      device: "cuda"

    # SmolVLM — edge/CPU
    smolvlm:
      source: "HuggingFaceTB/SmolVLM-256M-Instruct"
      device: "cpu"

    # Florence-2 — grounding/captioning (community model, no trust_remote_code)
    florence2:
      source: "florence-community/Florence-2-large"
      device: "cuda"

    # PaliGemma 2 — document understanding (gated: huggingface-cli login required)
    paligemma2:
      source: "google/paligemma2-3b-pt-224"
      dtype: "bfloat16"
      device: "cuda"

    # Phi-3.5 Vision — deferred (requires FlashAttention2, Linux only)
    # phi35-vision:
    #   source: "microsoft/Phi-3.5-vision-instruct"
    #   trust_remote_code: true
    #   dtype: "bfloat16"
    #   device: "cuda"

    # LLaVA-NeXT — high-quality VQA
    llava:
      source: "llava-hf/llava-v1.6-mistral-7b-hf"
      device: "cuda"
      max_new_tokens: 512

    # Moondream2 — tiny/fast
    moondream:
      source: "vikhyatk/moondream2"
      trust_remote_code: true
      device: "cpu"
```

Using an alias:

```python
vlm = mata.load("vlm", "medgemma")           # uses alias
vlm = mata.load("vlm", "smolvlm")            # uses alias
result = mata.run("vlm", "img.jpg",
                  model="llava",             # uses alias
                  prompt="Describe this.")
```

---

## API Reference

### `mata.load("vlm", model_id, **kwargs)`

```python
vlm = mata.load(
    "vlm",
    "google/medgemma-1.5-4b-it",   # HuggingFace ID or config alias
    device="cuda",                  # "auto" | "cpu" | "cuda" | "cuda:0"
    dtype="bfloat16",               # "auto" | "float16" | "bfloat16" | "float32"
    trust_remote_code=False,        # Required for InternVL, Phi-Vision, etc.
    max_new_tokens=512,
    system_prompt=None,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
)
```

### `vlm.predict(image, prompt, **kwargs)`

```python
result = vlm.predict(
    "image.jpg",                    # path | PIL.Image | np.ndarray
    prompt="Describe this image.",  # required
    system_prompt=None,             # overrides constructor default
    max_new_tokens=300,             # overrides constructor default
    temperature=0.5,
    output_mode=None,               # None | "json" | "detect" | "classify" | "describe"
    auto_promote=False,             # promote JSON bboxes to Instance objects
    images=["img2.jpg"],            # additional images for multi-image queries
)

print(result.text)                  # raw text response
print(result.entities)              # parsed entities (if output_mode set)
print(result.instances)             # Instance objects (if auto_promote=True)
print(result.meta["model_id"])      # "google/medgemma-1.5-4b-it"
print(result.meta["tokens_generated"])
```

### `mata.run("vlm", image, model=..., **kwargs)` (one-shot)

```python
result = mata.run(
    "vlm",
    "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="What is in this image?",
    dtype="auto",
    trust_remote_code=False,
    max_new_tokens=200,
)
```

---

## FAQ

### Q: Which model should I use for development and testing?

**`Qwen/Qwen3-VL-2B-Instruct`** — 2B parameters, no special flags, good general accuracy.
It is the default recommendation for local dev work.

For CPU-only environments, try **`HuggingFaceTB/SmolVLM-256M-Instruct`** (256M, no flags).

---

### Q: I get an OOM error loading a 7B model. What can I do?

1. **Use a smaller model**: Qwen3-VL 2B, LFM2.5-VL 1.6B, InternVL2 1B, SmolVLM 256M.
2. **Use `dtype="float16"` or `dtype="bfloat16"`**: Halves VRAM compared to float32.
3. **Use `device_map="auto"`** (passed via `device="auto"` — already the default): Splits layers
   across all available GPUs and CPU.
4. **Quantise**: Use `bitsandbytes` 4-bit quantisation (not yet a first-class MATA kwarg but can
   be passed via `from_pretrained` after loading the adapter manually).

---

### Q: What is `trust_remote_code` and when is it safe to enable?

`trust_remote_code=True` allows the HuggingFace library to download and execute Python code
bundled with the model repository. It is required for models with custom architectures not yet
merged into the official Transformers library (e.g., InternVL2, Phi-3.5, Moondream2, Florence-2).

**Only set `trust_remote_code=True` for models from trusted publishers** (Microsoft, Google,
HuggingFaceTB, OpenGVLab, vikhyatk). Never set it for unknown or unverified repositories.

---

### Q: FlashAttention warning / error on Windows

FlashAttention2 is not supported on Windows without a custom build. Models that require it
(e.g., DeepSeek OCR) are deferred. For all other VLMs, MATA does not require FlashAttention.

If you see a warning about `flash_attn` for a non-deferred model, it is safe to ignore — the
model will fall back to standard attention automatically.

---

### Q: `dtype="bfloat16"` vs `dtype="float16"` — which should I use?

- **`bfloat16`** — Better numerical stability (wider exponent range). Preferred for training and
  for models explicitly described as BFloat16 (MedGemma, LFM2.5). Requires relatively modern
  CUDA hardware (Ampere+ for native BF16 support).
- **`float16`** — More widely supported (Maxwell+ GPUs). Slightly less stable for very deep
  networks but fine for inference.
- **`"auto"`** — Let Transformers decide (safe default for most models).

---

### Q: Does MATA support API-based VLMs (GPT-4o, Gemini, Claude)?

Not in v1.9.3. The current VLM adapter is designed for locally-loaded HuggingFace models that use
`AutoModelForImageTextToText`. API-based VLMs are planned for a future release (v2.0 target).

---

### Q: Multi-image queries — how do I pass more than one image?

```python
result = vlm.predict(
    "main.jpg",
    images=["ref1.jpg", "ref2.jpg"],    # additional images
    prompt="Compare these three images.",
)
```

Or images-only (no primary):

```python
result = vlm.predict(
    images=["before.jpg", "after.jpg"],
    prompt="What has changed between these two images?",
)
```

---

### Q: Structured JSON output and bounding-box promotion

Use `output_mode="detect"` to request a detection-style JSON response, then `auto_promote=True`
to convert any bounding boxes in the JSON to first-class `Instance` objects:

```python
result = vlm.predict(
    "scene.jpg",
    prompt="Detect all objects and return bounding boxes in JSON.",
    output_mode="detect",
    auto_promote=True,
)
for inst in result.instances:
    print(inst.label, inst.bbox)    # xyxy pixel coords
```

Note: Bounding-box output quality depends heavily on the model. Qwen3-VL has the best grounding
support. Other models may return rough or non-standard coordinate formats — MATA's coordinate
scaling heuristic handles [0,1] normalized, ~1000-unit, and raw pixel coords automatically.

---

_Last updated: 2026-03-20 — MATA v1.9.3_
