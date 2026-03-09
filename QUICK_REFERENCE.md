# MATA Quick Reference — v1.5 to v1.9

## 📋 Table of Contents

| Section | Version |
|---------|─────────|
| [Universal Loader](#-what-changed) | v1.5 |
| [Configuration](#-configuration-file) | v1.5 |
| [Segmentation & SAM](#-segmentation-quick-reference) | v1.5.1 |
| [Save / Export](#-result-saving-quick-reference) | v1.5 |
| [CLIP Zero-Shot](#-zero-shot-classification-quick-reference) | v1.5.2 |
| [Vision-Language Models (VLM)](#-vision-language-models-quick-reference) | v1.5.3 |
| [Graph System](#️-graph-system-quick-reference-v16) | v1.6 |
| [VLM Agent Mode](#-vlm-agent-mode-quick-reference-v17) | v1.7 |
| [Object Tracking](#-object-tracking-quick-reference-v18) | v1.8 |
| [OCR / Text Extraction](#-ocr--text-extraction-quick-reference-v19) | v1.9 |
| [Evaluation](#-evaluation-quick-reference-v18) | v1.8 |
| [Valkey/Redis Storage](#-valkeyredis-storage-quick-reference-v19) | v1.9 |

---

## 🎯 What Changed

**Before (v1.4):**

```python
import mata
detector = mata.load("detect", "rtdetr")  # Plugin name only
```

**After (v1.5):**

```python
import mata

# Load from HuggingFace (recommended)
detector = mata.load("detect", "facebook/detr-resnet-50")

# Load from local file
detector = mata.load("detect", "./models/rtdetr.onnx")

# Load using config alias
detector = mata.load("detect", "fast-model")

# Load legacy plugin (backward compat)
detector = mata.load("detect", "rtdetr")  # Still works!
```

---

## 📁 Configuration File

Create `~/.mata/models.yaml` or `.mata/models.yaml`:

```yaml
models:
  detect:
    # Alias for HuggingFace model
    rtdetr-r18:
      source: "facebook/detr-resnet-50"
      threshold: 0.5

    # Alias for local ONNX model
    fast-model:
      source: "./models/rtdetr_optimized.onnx"
      threshold: 0.3

    # Alias for PyTorch checkpoint
    custom-model:
      source: "./checkpoints/rtdetr_finetuned.pth"
      config: "./configs/rtdetr.yml"
      architecture: "rtdetr"
```

**Usage:**

```python
detector = mata.load("detect", "rtdetr-r18")
```

---

## 🔄 Model Source Detection Chain

When you call `mata.load("detect", "source")`, MATA checks in order:

1. **Config alias** - Is "source" in `models.yaml`?
2. **Local file** - Does file exist on disk?
3. **HuggingFace ID** - Does "source" contain `/`?
4. **Legacy plugin** - Is "source" a registered plugin?
5. **Default** - Use task's default model

---

## 📦 Supported Formats

### HuggingFace Models

```python
# RT-DETR variants
mata.load("detect", "facebook/detr-resnet-50")
mata.load("detect", "PekingU/rtdetr_v2_r50vd")
mata.load("detect", "PekingU/rtdetr_v2_r101vd")

# DINO
mata.load("detect", "IDEA-Research/dino-resnet-50")

# Conditional DETR
mata.load("detect", "microsoft/conditional-detr-resnet-50")

# Depth estimation (Depth Anything)
mata.load("depth", "LiheYoung/depth-anything-small-hf")
mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf")
```

### Torchvision CNN Models (Apache 2.0)

```python
# RetinaNet (fast, single-stage)
mata.load("detect", "torchvision/retinanet_resnet50_fpn")

# Faster R-CNN (accurate, two-stage)
mata.load("detect", "torchvision/fasterrcnn_resnet50_fpn_v2")

# SSDLite (mobile-optimized)
mata.load("detect", "torchvision/ssdlite320_mobilenet_v3_large")
```

### Local Files

```python
# ONNX Runtime
mata.load("detect", "./models/model.onnx")

# PyTorch (requires config file)
mata.load("detect", "./checkpoints/model.pth",
          config="./configs/config.yml")

# TensorRT (future)
mata.load("detect", "./engines/model.engine")
```

---

## ⚙️ Runtime Registration

Register models programmatically:

```python
import mata

# Register an alias at runtime
mata.register_model("detect", "production-model", {
    "source": "s3://models/rtdetr_prod.onnx",
    "threshold": 0.7
})

# Use the alias
detector = mata.load("detect", "production-model")
```

---

## 🚨 Migration Checklist

### If you're upgrading from v1.4:

- [ ] **Option 1:** No changes needed - old code still works with deprecation warnings
- [ ] **Option 2:** Update to model IDs/paths for new features:

  ```python
  # Old
  detector = mata.load("detect", "rtdetr")

  # New (recommended)
  detector = mata.load("detect", "facebook/detr-resnet-50")
  ```

- [ ] **Option 3:** Create config file for aliases
- [ ] Update code to use `.instances` instead of `.detections` (removed in v1.5)

---

## 📚 Documentation

- **Examples:** [examples/](examples/)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Graph System:** [GRAPH_API_REFERENCE.md](docs/GRAPH_API_REFERENCE.md)
- **VLM Agent:** [VLM_TOOL_CALLING_SUMMARY.md](docs/VLM_TOOL_CALLING_SUMMARY.md)
- **Tracking:** [TASK_TRACKING.md](docs/TASK_TRACKING.md)

---

## ✅ Testing

**4047+ tests across all subsystems:**

```bash
pytest tests/ -v
```

**Test coverage by subsystem:**

- Universal Loader: 17/17 ✅
- Classification (CLIP + HF + ONNX): 81+ ✅
- Segmentation (Mask2Former + SAM + SAM3): 49+ ✅
- Depth Anything: 10+ ✅
- Object Tracking (vendored ByteTrack + BotSort): 687 ✅
- VLM tool-calling agent system: 336 ✅
- Graph system (DAG execution + nodes): 200+ ✅
- Evaluation metrics: 100+ ✅

---

## �️ Graph System Quick Reference (v1.6)

### `mata.infer()` — Multi-task DAG execution

```python
import mata
from mata.nodes import Detect, Filter, Fuse

detector = mata.load("detect", "facebook/detr-resnet-50")

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Filter(src="dets", score_gt=0.3, out="filtered"),
        Fuse(dets="filtered", out="final"),
    ],
    providers={"detector": detector},
)
for inst in result.final.instances:
    print(f"{inst.label_name}: {inst.score:.2f} at {inst.bbox}")
```

### Common Node Types

| Node        | Description                  | Key params                           |
| ----------- | ---------------------------- | ------------------------------------ |
| `Detect`    | Run object detection         | `using`, `out`                       |
| `Classify`  | Run classification           | `using`, `out`                       |
| `Segment`   | Run segmentation             | `using`, `out`                       |
| `Depth`     | Run depth estimation         | `using`, `out`                       |
| `VLMQuery`  | Run VLM with prompt          | `using`, `prompt`, `out`             |
| `VLMDetect` | VLM → structured detections  | `using`, `prompt`, `out`             |
| `Filter`    | Filter by score / label      | `src`, `score_gt`, `label_in`, `out` |
| `TopK`      | Keep top-K detections        | `k`, `src`, `out`                    |
| `Merge`     | Merge multiple inputs        | `srcs`, `out`                        |
| `Fuse`      | Collect results into channel | `dets`, `out`                        |
| `Annotate`  | Draw boxes/masks on image    | `dets`, `out`                        |
| `Crop`      | Extract ROIs per detection   | `dets`, `image`, `out`               |
| `Track`     | Add multi-object tracking    | `using`, `dets`, `out`               |
| `If`        | Conditional branch           | `condition`, `then`, `out`           |

### Parallel Execution

```python
from mata.core.graph import Graph, ParallelScheduler
from mata.nodes import Detect, Classify, Depth, Fuse

result = mata.infer(
    image="photo.jpg",
    graph=[
        Detect(using="detector", out="dets"),
        Classify(using="classifier", out="class"),   # ← parallel stage
        Depth(using="depth_model", out="depth"),      # ← parallel stage
        Fuse(dets="dets", out="final"),
    ],
    providers={
        "detector":    mata.load("detect", "facebook/detr-resnet-50"),
        "classifier":  mata.load("classify", "openai/clip-vit-base-patch32",
                                 text_prompts=["indoor", "outdoor"]),
        "depth_model": mata.load("depth", "depth-anything/Depth-Anything-V2-Small-hf"),
    },
    scheduler=ParallelScheduler(),
)
print(result.dets)   # VisionResult
print(result.class_)  # ClassifyResult
print(result.depth)  # DepthResult
```

### Fluent Graph Builder

```python
from mata.core.graph import Graph
from mata.nodes import Detect, Filter, TopK, Fuse

graph = (Graph("pipeline")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", score_gt=0.3, out="filtered"))
    .then(TopK(k=5, src="filtered", out="top5"))
    .then(Fuse(dets="top5", out="final"))
)

result = mata.infer("photo.jpg", graph, providers={"detector": detector})
```

### DSL Graph Construction

```python
from mata.dsl import dsl_graph

graph = dsl_graph("""
detect using=detector out=dets
filter src=dets score_gt=0.3 out=filtered
fuse dets=filtered out=final
""")

result = mata.infer("photo.jpg", graph, providers={"detector": detector})
```

### Industry Presets

```python
from mata.presets import (
    surveillance_basic,          # detect persons + tracking
    retail_analytics,            # detect products + classify
    medical_analysis,            # segment + VLM description
    traffic_monitoring,          # detect vehicles + tracking
    crowd_monitoring_botsort,    # detect persons + BotSort
)

result = mata.infer(
    image="scene.jpg",
    **crowd_monitoring_botsort(),
    providers={"detector": mata.load("detect", "facebook/detr-resnet-50"), ...},
)
```

**Documentation:** [GRAPH_API_REFERENCE.md](docs/GRAPH_API_REFERENCE.md) · [GRAPH_COOKBOOK.md](docs/GRAPH_COOKBOOK.md) · [examples/graph/](examples/graph/)

---

## 💡 Examples

### Basic Detection

```python
import mata
from PIL import Image

# Load detector
detector = mata.load("detect", "facebook/detr-resnet-50")

# Run detection
image = Image.open("image.jpg")
result = detector.predict(image, threshold=0.5)

# Access results
for det in result.detections:
    print(f"{det.label}: {det.score:.2f} at {det.bbox}")
```

### Using Config File

```python
# Create .mata/models.yaml
# models:
#   detect:
#     my-detector:
#       source: "facebook/detr-resnet-50"
#       threshold: 0.6

import mata

# Load using alias
detector = mata.load("detect", "my-detector")
result = detector.predict("image.jpg")  # Uses threshold=0.6 from config
```

### One-Shot Inference

```python
import mata

# No need to instantiate detector
result = mata.run("detect", "image.jpg",
                  model="facebook/detr-resnet-50",
                  threshold=0.5)
```

---

## 🎭 Segmentation Quick Reference

### Traditional Segmentation (Instance/Panoptic/Semantic)

```python
import mata

# Instance segmentation (COCO objects)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/mask2former-swin-tiny-coco-instance",
    threshold=0.5
)

# Panoptic segmentation (instances + stuff)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/mask2former-swin-tiny-coco-panoptic",
    segment_mode="panoptic"
)

# Semantic segmentation
result = mata.run(
    "segment",
    "image.jpg",
    model="nvidia/segformer-b0-finetuned-ade-512-512"
)

# Filter results
instances = result.get_instances()  # Countable objects
stuff = result.get_stuff()          # Background regions
high_conf = result.filter_by_score(0.8)
```

### Zero-Shot Segmentation (SAM)

**New in v1.5.1** - Segment any object with prompts:

#### Original SAM (Visual Prompts)

```python
import mata

# Point prompt (click on object)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/sam-vit-base",
    point_prompts=[(320, 240, 1)],  # (x, y, foreground=1)
    threshold=0.8
)

# Box prompt (region of interest)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/sam-vit-base",
    box_prompts=[(50, 50, 400, 400)]  # (x1, y1, x2, y2)
)

# Combined prompts (foreground + background)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/sam-vit-base",
    point_prompts=[(320, 240, 1), (100, 100, 0)],  # Include + exclude
    box_prompts=[(50, 50, 450, 450)],
    threshold=0.85
)

# Get best mask (SAM returns 3 masks per prompt)
best_mask = result.masks[0]  # Sorted by IoU score
print(f"IoU: {best_mask.score:.3f}, Area: {best_mask.area} pixels")
```

#### SAM3 (Text Prompts) 🆕

**Zero-shot concept segmentation** with natural language:

```python
import mata

# Text prompt (find all instances of concept)
result = mata.run(
    "segment",
    "image.jpg",
    model="facebook/sam3",
    text_prompts="cat",
    threshold=0.5
)

print(f"Found {len(result.masks)} cats")

# Multiple concepts
result = mata.run(
    "segment",
    "street.jpg",
    model="facebook/sam3",
    text_prompts=["person", "car", "bicycle"]
)

# Text + negative box (exclude specific region)
result = mata.run(
    "segment",
    "kitchen.jpg",
    model="facebook/sam3",
    text_prompts="handle",
    box_prompts=[(40, 183, 318, 204)],  # Oven area to exclude
    box_labels=[0],  # 0 = negative box
    threshold=0.6
)

# Supports 270K+ concepts: objects, parts, attributes
# Examples: "ear", "wheel", "laptop", "red car", "open door"
```

**SAM3 Authentication (Gated Model):**

SAM3 requires HuggingFace authentication. Three options:

```python
# Option 1: Login once (recommended)
# In terminal: huggingface-cli login
result = mata.run("segment", "image.jpg", model="facebook/sam3",
                  text_prompts="cat", token=True)

# Option 2: Pass token explicitly
result = mata.run("segment", "image.jpg", model="facebook/sam3",
                  text_prompts="cat", token="hf_...")

# Option 3: Environment variable
# export HF_TOKEN=hf_...
result = mata.run("segment", "image.jpg", model="facebook/sam3",
                  text_prompts="cat", token=True)
```

**Requirements:** `transformers>=4.46.0` for SAM3 support

````

**SAM Models**:

- `facebook/sam-vit-base` - Fast (recommended)
- `facebook/sam-vit-large` - Better quality
- `facebook/sam-vit-huge` - Best quality

**SAM Config Example**:

```yaml
# .mata/models.yaml
models:
  segment:
    sam:
      source: "facebook/sam-vit-base"
      threshold: 0.8
      use_rle: true # Compact mask format
```

---

## 💾 Result Saving Quick Reference

### Save to Multiple Formats

```python
import mata

# Run any task
result = mata.run("detect", "image.jpg", threshold=0.5)

# Save JSON (structured data)
result.save("detections.json")

# Save CSV (spreadsheet)
result.save("detections.csv")

# Save image overlay (visual)
result.save("overlay.png")

# Extract detection crops
result.save("crops.png", crop_dir="my_crops")
```

### Format Auto-Detection

Extension determines format:
- `.json` → JSON serialization
- `.csv` → Tabular export (spreadsheet-compatible)
- `.png`, `.jpg` → Image overlay with bboxes/masks
- Any extension + `crop_dir=...` → Individual detection crops

### Customization Options

```python
# Image overlay options
result.save(
    "overlay.png",
    show_boxes=True,
    show_labels=True,
    show_scores=True,
    alpha=0.5  # Mask transparency
)

# Crop extraction with padding
result.save(
    "crops.png",
    crop_dir="objects",
    padding=10  # 10px around each crop
)

# Compact JSON (no indentation)
result.save("compact.json", indent=None)
```

### Override Image Source

When using PIL Image or numpy array inputs:

```python
from PIL import Image

# PIL input (no path stored)
pil_img = Image.open("test.jpg")
result = mata.run("detect", pil_img)

# Provide image explicitly for overlay/crops
result.save("overlay.png", image="test.jpg")
```

### Access Stored Path

```python
result = mata.run("detect", "image.jpg")
print(result.get_input_path())  # "image.jpg"

# None if PIL/numpy input
result2 = mata.run("detect", pil_img)
print(result2.get_input_path())  # None
```

### Classification Charts

```python
result = mata.run("classify", "image.jpg")

# Bar chart (requires matplotlib)
result.save("predictions.png", top_k=5)

# CSV format
result.save("predictions.csv")  # rank, label, label_name, score
```

---

## 🎨 Zero-Shot Classification Quick Reference

### CLIP (Text-Prompted Classification) 🆕

**New in v1.5.2** - Classify images using natural language without training data:

```python
import mata

# Basic usage (3 lines)
result = mata.run(
    "classify",
    "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog", "bird"]
)
print(result.top(1))  # Most likely label

# Custom template (context-aware)
result = mata.run(
    "classify",
    "product.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["phone", "laptop", "tablet"],
    template="a photo of a {}, a type of electronic device"
)

# Template ensemble (better accuracy, +2-5%)
result = mata.run(
    "classify",
    "image.jpg",
    model="openai/clip-vit-large-patch14",
    text_prompts=["cat", "dog"],
    template="ensemble"  # Uses 6 templates
)

# Threshold + top-k filtering
result = mata.run(
    "classify",
    "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog", "bird", "horse"],
    threshold=0.1,  # Only show ≥10% probability
    top_k=3         # Max 3 results
)

# Raw scores vs softmax
result_raw = mata.run(
    "classify", "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog"],
    use_softmax=False  # Raw similarity scores
)

result_prob = mata.run(
    "classify", "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog"],
    use_softmax=True   # Calibrated probabilities (default)
)

# Batch classification (same prompts)
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = [
    mata.run(
        "classify", img,
        model="openai/clip-vit-base-patch32",
        text_prompts=["indoor", "outdoor"]
    )
    for img in images
]
```

**Supported CLIP Models:**
- `openai/clip-vit-base-patch32` - Fast (recommended)
- `openai/clip-vit-base-patch16` - Better quality
- `openai/clip-vit-large-patch14` - Best quality

**Template Shortcuts:**
- `"basic"` - Single template: `"a photo of a {}"`
- `"ensemble"` - 6 templates (default, +2-5% accuracy)
- `"detailed"` - 18 templates (+5-8% accuracy)
- Custom string: Your own template with `{}`
- Custom list: Multiple templates for ensemble

**When to Use CLIP:**
- Open-vocabulary classification (any categories)
- No training data available
- Rapid prototyping (instant model, no download)
- Domain adaptation (use domain-specific templates)

**Config Example:**

```yaml
# .mata/models.yaml
models:
  classify:
    clip-ensemble:
      source: "openai/clip-vit-large-patch14"
      text_prompts: ["cat", "dog", "bird"]
      template: "ensemble"
      threshold: 0.15
      top_k: 5
      use_softmax: true
```

**Documentation:** [CLIP_QUICK_START.md](docs/CLIP_QUICK_START.md)

### Segmentation Overlays

```python
result = mata.run("segment", "image.jpg", model="facebook/sam-vit-base")

# Mask overlay
result.save(
    "masks.png",
    show_masks=True,
    show_boxes=True,
    alpha=0.5
)

# CSV with mask metadata
result.save("segments.csv")  # includes has_mask, area
```

### Complete Example

```python
import mata

# Run detection
result = mata.run(
    "detect",
    "street.jpg",
    model="facebook/detr-resnet-50",
    threshold=0.5
)

# Export all formats
result.save("detections.json")       # For ML pipeline
result.save("detections.csv")        # For spreadsheet
result.save("overlay.png")           # For presentation
result.save("objects.png", crop_dir="objects")  # For dataset creation

# Verify stored path
print(f"Processed: {result.get_input_path()}")
```

**Documentation:** See [GRAPH_API_REFERENCE.md](docs/GRAPH_API_REFERENCE.md) for complete save/export API reference

---

## 🤖 Vision-Language Models Quick Reference

### VLM (Image Understanding & VQA) 🆕

**New in v1.5.3** - Use vision-language models for image captioning, visual question answering, and visual understanding:

```python
import mata

# Basic image description
result = mata.run(
    "vlm",
    "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Describe this image in detail."
)
print(result.text)  # Natural language description

# Visual question answering (VQA)
result = mata.run(
    "vlm",
    "street.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="How many cars are visible in this image?"
)
print(f"Answer: {result.text}")

# Domain-specific analysis with system prompt
result = mata.run(
    "vlm",
    "product.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Describe any defects or quality issues you see.",
    system_prompt="You are a manufacturing quality inspector. Be precise and technical."
)
print(result.text)

# Control generation parameters
result = mata.run(
    "vlm",
    "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="What is the main subject?",
    max_new_tokens=100,   # Limit response length
    temperature=0.5,      # Lower = more focused (0.0-1.0)
    top_p=0.9,           # Nucleus sampling threshold
    top_k=50             # Top-k sampling
)

# Load-then-predict pattern (batch processing)
vlm = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in images:
    result = vlm.predict(img, prompt="What is in this image?")
    print(f"{img}: {result.text}")

# Access result metadata
result = mata.run(
    "vlm",
    "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Describe this image."
)

print(f"Prompt: {result.prompt}")
print(f"Response: {result.text}")
print(f"Tokens generated: {result.meta['tokens_generated']}")
print(f"Model: {result.meta['model_id']}")
print(f"Device: {result.meta['device']}")
```

### Structured Output (v1.5.4+) 🆕

**Extract structured data from VLM responses:**

```python
import mata

# Object detection via VLM
result = mata.run(
    "vlm",
    "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="List all objects you can identify in this image.",
    output_mode="detect"  # Request structured JSON output
)

# Access parsed entities
for entity in result.entities:
    print(f"{entity.label}: {entity.score:.2f}")
    print(f"  Attributes: {entity.attributes}")

# Raw text always available (fallback if JSON parsing fails)
print(f"Raw response: {result.text}")

# Promote entities to instances (for downstream use in graph pipelines)
for entity in result.entities:
    # After spatial grounding (e.g., from GroundingDINO/SAM)
    instance = entity.promote(bbox=(x1, y1, x2, y2))
    # Now instance has both semantic + spatial data
```

**Available output modes:**
- `"json"` - Generic JSON object
- `"detect"` - Object detection format: `[{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2] (optional)}]`
- `"classify"` - Classification format: `[{"label": str, "confidence": float}]`
- `"describe"` - Description format: `{"description": str, "objects": [...], "scene": str}`

**How it works:**
1. VLM adapter injects a JSON schema instruction into the system prompt
2. Model generates response (may or may not be valid JSON)
3. Parser extracts JSON from text (handles markdown fences, embedded JSON)
4. Entities are created from parsed JSON with flexible key mapping
5. If parsing fails: graceful degradation (entities=[], raw text preserved)

**Entity dataclass:**
```python
@dataclass(frozen=True)
class Entity:
    label: str              # Required: object/concept name
    score: float = 1.0      # Confidence (default 1.0 if not provided)
    attributes: dict = {}   # Additional fields from VLM output

    def promote(self, bbox=None, mask=None) -> Instance:
        """Convert semantic Entity to spatial Instance."""
```

**Auto-Promotion (v1.5.4+):** 🆕

When VLMs output bboxes directly (like Qwen3-VL grounding mode), use `auto_promote=True` to convert entities with spatial data to `Instance` objects automatically. This enables direct comparison with spatial detection models.

```python
# Without auto_promote: bbox stored in Entity.attributes
result = mata.run("vlm", "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Detect objects with bboxes in JSON format.",
    output_mode="detect")
for entity in result.entities:
    print(f"{entity.label}: bbox={entity.attributes.get('bbox')}")
# Output: cat: bbox=[10, 20, 100, 150]

# With auto_promote=True: bbox → Instance objects
result = mata.run("vlm", "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Detect all objects and provide bounding boxes as [x1, y1, x2, y2] coordinates.",
    output_mode="detect",
    auto_promote=True)  # Entities with bbox/mask → Instance
for instance in result.instances:  # Now in instances, not entities!
    print(f"{instance.label_name}: {instance.bbox} ({instance.score:.2f})")
# Output: cat: (10, 20, 100, 150) (0.95)

# Note: VLMs need explicit prompting for bbox output. The output_mode="detect"
# injects a schema hint, but you should also be specific in your prompt.

# Compare VLM spatial output with traditional detector
vlm_result = mata.run("vlm", "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Detect all objects with bounding boxes.",
    output_mode="detect",
    auto_promote=True)

detector_result = mata.run("detect", "image.jpg",
    model="facebook/detr-resnet-50")

print(f"VLM: {len(vlm_result.instances)} instances")
print(f"DETR: {len(detector_result.instances)} instances")
# Both use the same Instance format for downstream processing!
```

**Supported bbox formats:** `[x1, y1, x2, y2]` (xyxy), `[x, y, w, h]` (xywh - first 4 values used)
**Supported mask formats:** RLE dict, binary array, polygon coordinates
**Fallback behavior:** If bbox/mask validation fails, returns Entity with spatial data in attributes

### Multi-Image Support (v1.5.4+) 🆕

**Send multiple images in a single query:**

```python
import mata

# Compare two images
result = mata.run(
    "vlm",
    "before.jpg",  # Primary image (first positional arg)
    model="Qwen/Qwen3-VL-2B-Instruct",
    images=["after.jpg"],  # Additional images
    prompt="What changed between these images?"
)
print(result.text)
print(f"Images analyzed: {result.meta['image_count']}")

# Analyze multiple images (no primary)
result = mata.run(
    "vlm",
    images=["img1.jpg", "img2.jpg", "img3.jpg"],
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="Summarize all images. What's the common theme?"
)

# Combine with structured output
result = mata.run(
    "vlm",
    "product_front.jpg",
    images=["product_back.jpg", "product_side.jpg"],
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="List all visible text and logos.",
    output_mode="detect"
)
for entity in result.entities:
    print(f"Found: {entity.label}")
```

**Multi-image metadata:**
```python
result.meta["image_count"]     # Total images processed
result.meta["image_paths"]     # List of all image paths
result.meta["image_path"]      # First image (backward compat)
```

**Available VLM Models:**
- `Qwen/Qwen3-VL-2B-Instruct` - Fast, chat-capable (recommended for dev, ~4GB GPU RAM)

**VLM-Specific Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | **Required** | Text prompt/question for the model |
| `system_prompt` | `str` | `None` | System-level instruction to configure model behavior |
| `max_new_tokens` | `int` | `512` | Maximum number of tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0 = deterministic, 1.0 = creative) |
| `top_p` | `float` | `0.8` | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | `int` | `20` | Top-k sampling (limits token pool) |
| `output_mode` | `str` | `None` | Structured output mode: `"json"`, `"detect"`, `"classify"`, `"describe"` (v1.5.4+) |
| `images` | `list` | `None` | Additional images for multi-image queries (v1.5.4+) |
| `auto_promote` | `bool` | `False` | Auto-promote entities with bbox/mask to Instance objects (v1.5.4+) |

**When to Use VLM:**
- Image captioning and description
- Visual question answering (VQA)
- Scene understanding and analysis
- Visual reasoning tasks
- Domain-specific image analysis (with system prompts)
- Interactive visual assistants

**Result Format:**

```python
result = mata.run("vlm", "image.jpg",
    model="Qwen/Qwen3-VL-2B-Instruct",
    prompt="What do you see?")

# VisionResult with text field
assert isinstance(result, VisionResult)
assert result.text is not None  # Generated text
assert result.prompt == "What do you see?"  # Original prompt
assert result.instances == []  # Empty (raw text mode)

# Metadata
assert "model_id" in result.meta
assert "tokens_generated" in result.meta
assert "device" in result.meta
```

**Config Example:**

```yaml
# .mata/models.yaml
models:
  vlm:
    qwen-vlm:
      source: "Qwen/Qwen3-VL-2B-Instruct"
      max_new_tokens: 256
      temperature: 0.7
      system_prompt: "You are a helpful visual assistant."

    inspector:
      source: "Qwen/Qwen3-VL-2B-Instruct"
      max_new_tokens: 512
      temperature: 0.3  # More focused for technical analysis
      system_prompt: "You are a quality control inspector. Be precise and technical."
```

**Documentation:** [VLM_TOOL_CALLING_SUMMARY.md](docs/VLM_TOOL_CALLING_SUMMARY.md), [basic_vlm.py](examples/vlm/basic_vlm.py)

---

## 🤖 VLM Agent Mode Quick Reference (v1.7)

VLM nodes can operate in **agent mode** where they iteratively call specialized tools to gather information before producing a final answer.

### Basic Usage

```python
import mata
from mata.nodes import VLMQuery

# Standard mode (no tools) — unchanged
result = mata.infer(
    image="photo.jpg",
    graph=[VLMQuery(using="qwen", prompt="Describe this image.", out="desc")],
    providers={"qwen": mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")},
)

# Agent mode — VLM calls tools iteratively
result = mata.infer(
    image="photo.jpg",
    graph=[
        VLMQuery(
            using="qwen",
            prompt="What vehicle types are in this image and how many?",
            tools=["detector", "classifier"],  # ← enables agent mode
            max_iterations=5,
            on_error="retry",  # "retry" (default) | "skip" | "fail"
            out="analysis",
        )
    ],
    providers={
        "qwen":       mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct"),
        "detector":   mata.load("detect", "facebook/detr-resnet-50"),
        "classifier": mata.load("classify", "openai/clip-vit-base-patch32",
                                 text_prompts=["car", "truck", "bus", "motorcycle"]),
    },
)
print(result.analysis.text)  # Final synthesized answer
```

### Built-in Tools (always available)

```python
# zoom — magnify a region for closer inspection
# crop — extract a sub-region of the image

result = mata.infer(
    image="crowd.jpg",
    graph=[
        VLMQuery(
            using="qwen",
            prompt="How many people are in the top-left corner?",
            tools=["zoom", "crop", "detector"],  # built-ins + provider tools
            max_iterations=8,
            out="answer",
        )
    ],
    providers={
        "qwen":     mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct"),
        "detector": mata.load("detect", "facebook/detr-resnet-50"),
    },
)
```

### Agent Loop Execution Flow

```
VLM receives prompt + auto-generated tool descriptions in system prompt
    ↓
VLM outputs tool call (fenced block / XML / raw JSON)
    ↓                                 ← repeats up to max_iterations
ToolRegistry dispatches to provider adapter or built-in image function
    ↓
Tool results appended as conversation history
    ↓
VLM outputs final answer (no tool calls) → agent exits
    ↓
AgentResult → Detections artifact (accumulated across all tool calls)
```

### VLM Node Types in Agent Mode

| Node | Output artifact | Agent-capable |
|------|----------------|---------------|
| `VLMQuery` | Text + accumulated detections | ✅ |
| `VLMDetect` | Structured `Detections` | ✅ |
| `VLMDescribe` | Narrative text description | ✅ |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | `list[str]` | `None` | Provider keys or `"zoom"`/`"crop"`. `None` = standard mode. |
| `max_iterations` | `int` | `5` | Safety cap on agent loop iterations |
| `on_error` | `str` | `"retry"` | Error recovery: `"retry"`, `"skip"`, `"fail"` |

**Documentation:** [VLM_TOOL_CALLING_SUMMARY.md](docs/VLM_TOOL_CALLING_SUMMARY.md) · [examples/](examples/)

---

## 🎯 Object Tracking Quick Reference (v1.8)

### `mata.track()` — One-liner video/stream tracking

```python
import mata

# Video file — returns list[VisionResult] (one per frame)
results = mata.track(
    "video.mp4",
    model="facebook/detr-resnet-50",
    tracker="botsort",        # "botsort" (default) or "bytetrack"
    conf=0.3,                 # detection confidence threshold
    iou=0.7,                  # NMS IoU threshold
    classes=[0, 2],           # optional: filter to class indices
    max_frames=500,           # optional: stop after N frames
    save=True,                # save annotated video to runs/track/exp1/
    show_track_ids=True,      # render #ID labels with per-track colors
    show_trails=True,         # draw trajectory polylines (last 30 frames)
)
for frame_idx, result in enumerate(results):
    for inst in result.instances:
        print(f"Frame {frame_idx}: Track #{inst.track_id} "
              f"{inst.label_name} ({inst.score:.2f})")

# Stream mode — generator for constant memory usage
for result in mata.track("rtsp://camera/stream",
                          model="facebook/detr-resnet-50",
                          stream=True):
    tracks = [i for i in result.instances if i.track_id is not None]
    print(f"Active: {len(tracks)}")

# Single image — list with one VisionResult
results = mata.track("image.jpg", model="facebook/detr-resnet-50")
```

### `mata.load("track", ...)` — Persistent per-frame tracking

```python
import cv2
import mata

# Load a combined detect+track adapter
tracker = mata.load(
    "track",
    "facebook/detr-resnet-50",
    tracker="botsort",    # or "bytetrack"
    frame_rate=30,
)

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # persist=True: keep Kalman state across calls (YOLO pattern)
    result = tracker.update(frame, persist=True)

    for inst in result.instances:
        print(f"Track #{inst.track_id}: {inst.label_name} @ {inst.bbox}")

cap.release()
tracker.reset()  # clear all tracker state
```

### Supported Source Types

| Source | Example | Notes |
|--------|---------|-------|
| Video file | `"video.mp4"` | `.mp4` `.avi` `.mkv` `.mov` `.wmv` |
| RTSP stream | `"rtsp://..."` | Live camera feed |
| HTTP stream | `"http://..."` or `"https://..."` | IP cameras |
| Webcam | `0` (integer) | Device index |
| Image directory | `"frames/"` | Sorted alphabetically |
| Single image | `"frame.jpg"` | 1-frame result list |
| numpy array | `np.ndarray` (H×W×3 BGR) | Direct frame |
| PIL Image | `Image.open(...)` | Direct frame |

### Tracker Selection

```python
# BotSort (default, recommended)
# - Sparse optical flow for camera-motion compensation
# - Better for panning/tilting cameras
mata.track("video.mp4", tracker="botsort", ...)

# ByteTrack (faster, no camera-motion compensation)
# - Two-stage IoU association (high-conf → low-conf detections)
# - Best when camera is static
mata.track("video.mp4", tracker="bytetrack", ...)
```

### Tracker Configuration (YAML)

```yaml
# .mata/models.yaml
models:
  track:
    highway-cam:
      source: "facebook/detr-resnet-50"
      tracker: botsort
      frame_rate: 30
      tracker_config:
        track_high_thresh: 0.6   # high-conf detection threshold
        track_low_thresh: 0.1    # low-conf (second-stage) threshold
        new_track_thresh: 0.7    # minimum score to start new track
        track_buffer: 60         # frames to keep lost tracks (2s @ 30fps)
        match_thresh: 0.8        # maximum IoU distance for matching
        gmc_method: sparseOptFlow  # BotSort: global motion compensation
```

```python
tracker = mata.load("track", "highway-cam")
# equivalent to:
# mata.load("track", "facebook/detr-resnet-50",
#           tracker="botsort", frame_rate=30,
#           tracker_config={...})
```

### Visualization Options

```python
# Track ID labels + per-track deterministic colors
results = mata.track("video.mp4", show_track_ids=True, ...)

# Trajectory trail polylines (last N frames per track)
results = mata.track("video.mp4", show_trails=True,
                     trail_length=50, ...)

# VisionResult.save() also supports both options
result.save("frame.png", show_track_ids=True)
```

### Export Formats

```python
import mata
from mata.core.exporters import export_tracks_csv, export_tracking_json

# Collect results
results = mata.track("video.mp4", model="facebook/detr-resnet-50")

# MOT-compatible CSV (frame_id, track_id, label, score, x1,y1,x2,y2, area)
export_tracks_csv(results, "tracks.csv")

# JSON with frame structure + metadata
export_tracking_json(results, "tracks.json")
# Output shape:
# { "frames": [{"frame_id": 0, "instances": [...]}, ...],
#   "meta":   {"num_frames": 120, "unique_tracks": 5, "tracker": "botsort"} }
```

### Graph Node Integration

```python
from mata.nodes import Detect, Filter, Track, Annotate
from mata.nodes.track import ByteTrackWrapper, BotSortWrapper

graph = [
    Detect(using="detr", out="dets"),
    Filter(src="dets", labels=["person", "car"], out="filtered"),
    Track(using="botsort", dets="filtered", out="tracks"),
    Annotate(using="drawer", dets="tracks",
             show_track_ids=True, out="annotated"),
]

result = mata.infer(
    graph=graph,
    video="video.mp4",
    providers={
        "detr":    mata.load("detect", "facebook/detr-resnet-50"),
        "botsort": BotSortWrapper(),
        "drawer":  ...,
    },
)
```

### Pre-built Presets

```python
from mata.presets import (
    crowd_monitoring,          # ByteTrack
    crowd_monitoring_botsort,  # BotSort
    traffic_tracking,          # ByteTrack
    traffic_tracking_botsort,  # BotSort
)

result = mata.infer("video.mp4", crowd_monitoring_botsort(), providers={...})
```

**Test suites:** `tests/test_trackers/` (354) · `tests/test_tracking_adapter.py` (73) · `tests/test_track_api.py` (62) · `tests/test_tracking_visualization.py` (103) · `tests/test_video_io.py` (56) — **687 tracking tests total**

**Documentation:** [TASK_TRACKING.md](docs/TASK_TRACKING.md) · [examples/track/](examples/track/)

---

## � OCR / Text Extraction Quick Reference (v1.9)

### Load backends

```python
import mata

mata.load("ocr", "easyocr")                                   # External engine (80+ languages)
mata.load("ocr", "paddleocr", lang="en")                      # PaddleOCR
mata.load("ocr", "paddleocr", lang="zh", use_gpu=False)       # PaddleOCR (CPU)
mata.load("ocr", "tesseract", lang="eng")                     # Tesseract (system binary required)
mata.load("ocr", "tesseract", lang="eng+fra")                 # multiple languages
mata.load("ocr", "microsoft/trocr-base-handwritten")          # HuggingFace TrOCR
mata.load("ocr", "microsoft/trocr-large-printed")             # HuggingFace TrOCR (large)
mata.load("ocr", "stepfun-ai/GOT-OCR-2.0-hf")                # HuggingFace GOT-OCR2
mata.load("ocr", "my-alias")                                  # config alias
```

### `mata.run()` — one-liner OCR

```python
result = mata.run("ocr", "document.jpg", model="easyocr")

print(result.full_text)                        # all text joined by newlines
for region in result.regions:
    print(f"{region.text!r}  "
          f"conf={region.score:.2f}  "
          f"bbox={region.bbox}")              # bbox is None for HF models

# Filter by confidence
high_conf = result.filter_by_score(0.85)
```

### Export formats

```python
result.save("output.txt")    # plain concatenated text
result.save("output.csv")    # CSV: text, score, x1, y1, x2, y2
result.save("output.json")   # structured JSON (roundtrippable)
result.save("overlay.png")   # image with bounding-box overlays
result.save("overlay.jpg")   # same, JPEG output

# or use to_json() / to_dict() directly
json_str = result.to_json(indent=2)
data     = result.to_dict()
```

### Backend comparison

| Backend | Alias | Bbox | Confidence | GPU | Languages |
|---|---|---|---|---|---|
| EasyOCR | `"easyocr"` | ✅ xyxy | ✅ float | ✅ `gpu=True` | 80+ |
| PaddleOCR | `"paddleocr"` | ✅ xyxy | ✅ float | ✅ `use_gpu=True` | 80+ |
| Tesseract | `"tesseract"` | ✅ xyxy | ✅ normalized | ❌ | 100+ |
| TrOCR | HF model ID | ❌ (whole-image) | ❌ (1.0 placeholder) | ✅ `device=` | EN (printed/handwritten) |
| GOT-OCR2 | HF model ID | ❌ (whole-image) | ❌ (1.0 placeholder) | ✅ `device=` | Multi |

> **TrOCR note:** designed for pre-cropped single text-line images.
> Use GOT-OCR2 or an external engine for full-page documents.

### Config aliases (`.mata/models.yaml`)

```yaml
models:
  ocr:
    default-ocr:
      source: "easyocr"           # "easyocr" | "paddleocr" | "tesseract" | HF ID
      lang: "en"
    doc-ocr:
      source: "stepfun-ai/GOT-OCR-2.0-hf"
      device: "cuda"
    handwriting:
      source: "microsoft/trocr-base-handwritten"
      device: "cpu"
```

```python
adapter = mata.load("ocr", "doc-ocr")   # resolves via config
```

### Graph node — `OCR`

```python
from mata.nodes import OCR

# Whole-image OCR
node = OCR(using="ocr_engine", out="text")

# ROI-mode: processes each crop individually, sets instance_ids
node = OCR(using="ocr_engine", src="rois", out="ocr_result")
```

### Pipeline: Detect → Extract ROIs → OCR → Fuse

```python
from mata.nodes import Detect, Filter, ExtractROIs, OCR, Fuse
from mata.core.graph import Graph

graph = (
    Graph("sign_reader")
    .then(Detect(using="detector", out="dets"))
    .then(Filter(src="dets", label_in=["sign", "license_plate"], out="filtered"))
    .then(ExtractROIs(src_dets="filtered", out="rois"))
    .then(OCR(using="ocr_engine", src="rois", out="ocr_result"))
    .then(Fuse(out="final", dets="filtered", ocr="ocr_result"))
)

result = mata.infer(graph, image="street.jpg", providers={
    "detector":   mata.load("detect", "facebook/detr-resnet-50"),
    "ocr_engine": mata.load("ocr",    "easyocr"),
})
for item in result["final"].items:
    print(f"{item.label}: {item.ocr_text!r}")
```

When `OCR` receives a `ROIs` artifact, `OCRText.instance_ids` aligns one-to-one
with the source ROI IDs so `Fuse` can correlate detections with their text.

### VLM tool-calling integration

```python
from mata.nodes import VLMQuery

node = VLMQuery(
    using="vlm",
    prompt="Extract all visible text and summarize.",
    tools=["ocr", "zoom"],   # OCR exposed as an agent tool
    max_iterations=4,
)
result = mata.infer(graph, image="form.jpg", providers={
    "vlm": mata.load("vlm", "qwen3-vl"),
    "ocr": mata.load("ocr", "easyocr"),
})
```

**Documentation:** [OCR Architecture Summary](docs/OCR_IMPLEMENTATION_SUMMARY.md)

---

## �📊 Evaluation Quick Reference (v1.8)

### `mata.val()` — YOLO-style validation

```python
import mata

# Evaluate a detection model
metrics = mata.val(
    "detect",
    model="facebook/detr-resnet-50",
    data="data/coco/coco.yaml",   # dataset YAML (images + annotations)
    conf=0.001,                   # confidence threshold
    iou=0.50,                     # IoU threshold for TP/FP matching
    verbose=True,                 # print per-class table
    plots=True,                   # save PR/F1/confusion-matrix charts
    save_dir="runs/val/exp1/",
)
print(f"mAP@50:    {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

# Evaluate classification
metrics = mata.val(
    "classify",
    model="openai/clip-vit-base-patch32",
    data="data/imagenet/imagenet.yaml",
    verbose=True,
)
print(f"Top-1 accuracy: {metrics.top1:.4f}")
print(f"Top-5 accuracy: {metrics.top5:.4f}")

# Standalone mode — evaluate pre-run predictions
metrics = mata.val(
    "detect",
    predictions=my_predictions,         # list of VisionResult
    ground_truth="data/coco/val.json",  # COCO annotation JSON
    conf=0.25,
    iou=0.50,
)
```

### Metrics by Task

| Task | Metric class | Key properties |
|------|-------------|----------------|
| `detect` | `DetMetrics` | `box.map`, `box.map50`, `box.map75`, `box.mp`, `box.mr`, `box.maps` |
| `segment` | `SegmentMetrics` | `box.map50`, `box.map`, `seg.map50`, `seg.map` |
| `classify` | `ClassifyMetrics` | `top1`, `top5`, `fitness` |
| `depth` | `DepthMetrics` | `abs_rel`, `sq_rel`, `rmse`, `delta_1`, `delta_2`, `delta_3` |

### `mata.val()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | _required_ | `"detect"`, `"segment"`, `"classify"`, `"depth"` |
| `model` | `str \| adapter` | `None` | Model ID, path, alias, or pre-loaded adapter |
| `data` | `str \| dict` | `None` | Dataset YAML path or config dict |
| `predictions` | `list` | `None` | Pre-run predictions (standalone mode) |
| `ground_truth` | `str \| list` | `None` | COCO JSON path (standalone mode) |
| `conf` | `float` | `0.001` | Confidence threshold for filtering |
| `iou` | `float` | `0.50` | IoU threshold for TP/FP matching |
| `verbose` | `bool` | `True` | Print per-class metrics table |
| `plots` | `bool` | `False` | Save PR curve, F1 curve, confusion matrix |
| `save_dir` | `str` | `""` | Output directory for plots |
| `split` | `str` | `"val"` | Dataset split: `"val"`, `"test"`, `"train"` |

**Documentation:** [Validation Guide](docs/VALIDATION_GUIDE.md)

---

## 🗄️ Valkey/Redis Storage Quick Reference (v1.9)

### Installation

```bash
pip install mata[valkey]   # valkey-py (recommended)
pip install mata[redis]    # redis-py (alternative)
```

### `result.save()` — URI scheme

```python
# Any result type supports valkey:// and redis:// URIs directly in save()
result.save("valkey://localhost:6379/my_key")          # basic
result.save("valkey://localhost:6379/0/my_key")        # with DB number
result.save("redis://localhost:6379/my_key")           # redis-py fallback
result.save("valkey://localhost:6379/my_key", ttl=300) # with TTL (seconds)
```

### Low-level exporter

```python
from mata.core.exporters import export_valkey, load_valkey, publish_valkey

# Store
export_valkey(result, url="valkey://localhost:6379", key="my_key", ttl=3600)

# Load
loaded = load_valkey(url="valkey://localhost:6379", key="my_key")

# Load with explicit result type (skip auto-detection)
loaded = load_valkey(url="valkey://localhost:6379", key="my_key",
                     result_type="vision")  # or "classify", "depth", "ocr"

# Pub/Sub publish (fire-and-forget)
n_receivers = publish_valkey(result, url="valkey://localhost:6379",
                              channel="detections:stream")
```

### URI formats

| Format | Example |
| ------ | ------- |
| Basic | `valkey://localhost:6379/key` |
| With DB | `valkey://localhost:6379/0/key` |
| With auth | `valkey://user:pass@host:6379/0/key` |
| Redis | `redis://localhost:6379/key` |
| Redis TLS | `rediss://host:6379/key` |

### Graph nodes: `ValkeyStore` / `ValkeyLoad`

```python
from mata.nodes import ValkeyStore, ValkeyLoad

# Sink node — store artifact and pass it through unchanged
ValkeyStore(
    src="filtered",                       # artifact name in graph context
    url="valkey://localhost:6379",
    key="pipeline:{node}:{timestamp}",    # {node} and {timestamp} placeholders
    ttl=3600,                             # optional TTL in seconds
    serializer="json",                    # "json" (default) or "msgpack"
    out="filtered",                       # optional override for output name
)

# Source node — load artifact as graph entry point
ValkeyLoad(
    url="valkey://localhost:6379",
    key="upstream:detections:latest",
    result_type="auto",                   # or "vision", "classify", "depth", "ocr"
    out="dets",
)
```

### Auto-detection of result type

| Key in stored data | Detected type | Output artifact |
| ------------------ | ------------- | --------------- |
| `instances` | `vision` | `Detections` |
| `predictions` | `classify` | `Classifications` |
| `depth` | `depth` | `DepthMap` |
| `regions` | `ocr` | _(raw dict)_ |

### Named connections (YAML config)

```yaml
# .mata/models.yaml
storage:
  valkey:
    default:
      url: "valkey://localhost:6379"
      db: 0
      ttl: 3600
    production:
      url: "valkey://prod-cluster:6379"
      password_env: "VALKEY_PASSWORD"  # read from env, never stored in plaintext
      tls: true
```

```python
from mata.core.model_registry import ModelRegistry

registry = ModelRegistry()
conn = registry.get_valkey_connection("production")
# → {"url": "valkey://...", "password": "<from env>", "tls": True}
```

**Documentation:** [Graph API Reference — Storage Nodes](docs/GRAPH_API_REFERENCE.md#storage-nodes)

---

**Version:** 1.9.0
**Date:** March 9, 2026
**Status:** ✅ Production Ready
````
