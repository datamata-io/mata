# Zero-Shot Object Detection Guide

**MATA v1.5+** | Text-Prompt-Based Detection | No Training Required

---

## Overview

Zero-shot object detection enables detecting arbitrary objects using text descriptions instead of predefined class lists. MATA supports state-of-the-art zero-shot detection models through a unified interface.

**Key Features:**

- ✅ Text-prompt-based detection (describe what you want to find)
- ✅ No training or fine-tuning required
- ✅ Support for multiple architectures (GroundingDINO, OWL-ViT v1/v2)
- ✅ Batch processing for multiple images
- ✅ Multi-modal pipeline with SAM for instance segmentation
- ✅ Unified `VisionResult` format with `Instance` objects

---

## Quick Start

### Basic Zero-Shot Detection

```python
import mata
from PIL import Image

# Load zero-shot detector
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

# Load image
image = Image.open("image.jpg")

# Detect with text prompts
result = detector.predict(image, text_prompts="cat . dog . person")

# Access results
for instance in result.instances:
    print(f"{instance.label_name}: {instance.score:.2f} at {instance.bbox}")
```

### GroundingDINO→SAM Pipeline

Combine zero-shot detection with zero-shot segmentation:

```python
# Load pipeline
pipeline = mata.load(
    "pipeline",
    detector_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base"
)

# Get both bboxes and masks
result = pipeline.predict(image, text_prompts="car . building")

# Access multi-modal instances
for instance in result.instances:
    print(f"{instance.label_name}:")
    print(f"  BBox: {instance.bbox}")
    print(f"  Mask: {'✓' if instance.mask else '✗'}")
    print(f"  Area: {instance.area}")
```

---

## Zero-Shot Classification vs Detection

**New in v1.5.2:** MATA now supports zero-shot **classification** with CLIP in addition to zero-shot **detection**. Choose the right task for your needs:

### When to Use Classification (CLIP)

**Use Case:** Categorize the entire image into one or more classes without spatial localization.

```python
# Classification: "What is this image?"
result = mata.run(
    "classify",
    "product.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["phone", "laptop", "tablet", "camera"]
)
# Output: ClassifyResult with scores for each category
# Example: [("laptop", 0.85), ("tablet", 0.10), ("camera", 0.03), ("phone", 0.02)]
```

**Strengths:**

- ✅ Fast inference (~0.1-0.3s)
- ✅ Calibrated probabilities (softmax scores)
- ✅ Excellent for global image understanding
- ✅ Template customization for domain adaptation
- ✅ No spatial output (lower computational cost)

**Limitations:**

- ❌ No bounding boxes (can't locate objects)
- ❌ No instance separation (multiple objects = single aggregate score)
- ❌ Single global prediction per image

### When to Use Detection (GroundingDINO/OWL-ViT)

**Use Case:** Find and locate specific objects with bounding boxes.

```python
# Detection: "Where are the cats and dogs?"
result = mata.run(
    "detect",
    "pets.jpg",
    model="IDEA-Research/grounding-dino-tiny",
    text_prompts="cat . dog"
)
# Output: VisionResult with bbox for each detected instance
# Example: [
#   Instance(label="cat", bbox=(100,50,300,400), score=0.92),
#   Instance(label="cat", bbox=(450,80,600,350), score=0.87),
#   Instance(label="dog", bbox=(200,200,500,600), score=0.95)
# ]
```

**Strengths:**

- ✅ Spatial localization (bounding boxes)
- ✅ Instance-level detection (multiple objects of same class)
- ✅ Works with pipeline → SAM for segmentation masks
- ✅ Precise object counting

**Limitations:**

- ❌ Slower inference (~0.5-5s depending on model)
- ❌ Requires text prompts in specific format (space-dot for GroundingDINO)
- ❌ Higher computational cost

### Decision Matrix

| Scenario                               | Recommended Task    | Model                               |
| -------------------------------------- | ------------------- | ----------------------------------- |
| "Is this a cat or dog?"                | **Classification**  | `openai/clip-vit-base-patch32`      |
| "Find all cats in the image"           | **Detection**       | `IDEA-Research/grounding-dino-tiny` |
| "Categorize product photos"            | **Classification**  | `openai/clip-vit-large-patch14`     |
| "Count people in a crowd"              | **Detection**       | `google/owlv2-base-patch16`         |
| "Scene understanding (indoor/outdoor)" | **Classification**  | `openai/clip-vit-base-patch32`      |
| "Locate safety violations"             | **Detection**       | `IDEA-Research/grounding-dino-tiny` |
| "Content moderation (NSFW/SFW)"        | **Classification**  | `openai/clip-vit-base-patch32`      |
| "Extract objects for dataset creation" | **Detection** + SAM | Pipeline                            |

### Prompt Format Differences

**Classification (CLIP):** Simple list, any natural language

```python
text_prompts=["cat", "dog", "bird"]
text_prompts=["a photo of a cat", "a dog running", "bird in flight"]
```

**Detection (GroundingDINO):** Space-dot separated, noun phrases

```python
text_prompts="cat . dog . bird"
text_prompts="running cat . sleeping dog . flying bird"
```

**Detection (OWL-ViT):** List or space-dot, noun phrases

```python
text_prompts=["cat", "dog", "bird"]
text_prompts="cat . dog . bird"
```

### Combining Both Tasks

For comprehensive analysis:

```python
import mata

# Step 1: Classify scene
scene_result = mata.run(
    "classify", "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["indoor", "outdoor"]
)
print(f"Scene: {scene_result.top(1)[0].label_name}")  # "outdoor"

# Step 2: Detect specific objects
if scene_result.top(1)[0].label_name == "outdoor":
    detect_result = mata.run(
        "detect", "image.jpg",
        model="IDEA-Research/grounding-dino-tiny",
        text_prompts="tree . car . person . building"
    )
    print(f"Found {len(detect_result.instances)} outdoor objects")
```

**Documentation:**

- Classification: [CLIP_QUICK_START.md](CLIP_QUICK_START.md)
- Detection: This guide (ZEROSHOT_DETECTION_GUIDE.md)

---

## Supported Models

### 1. GroundingDINO (IDEA-Research)

**Best for:** State-of-the-art accuracy, complex scenes

```python
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
```

**Specifications:**

- Architecture: DINO + Text-Visual Fusion
- Performance: Highest accuracy, slower inference
- Prompt Format: Space-dot separated (e.g., `"cat . dog . person"`)
- Input Resolution: 800×1333 (flexible)
- Strengths: Complex scenes, small objects, rare categories

**Available Models:**

- `IDEA-Research/grounding-dino-tiny` - 48M params (recommended)
- `IDEA-Research/grounding-dino-base` - 230M params

**When to Use:**

- Need highest detection accuracy
- Detecting small or distant objects
- Complex scenes with many object types
- Rare or unusual object categories
- Can afford longer inference time (~2-5s)

### 2. OWL-ViT v2 (Google)

**Best for:** Fast inference, good accuracy tradeoff

```python
detector = mata.load("detect", "google/owlv2-base-patch16")
```

**Specifications:**

- Architecture: Vision Transformer + CLIP-style text encoder
- Performance: Fast inference, good accuracy
- Prompt Format: List (e.g., `["cat", "dog", "person"]`) or space-dot
- Input Resolution: 840×840
- Strengths: Speed, versatility, general-purpose

**Available Models:**

- `google/owlv2-base-patch16` - 93M params (recommended)
- `google/owlv2-base-patch16-ensemble` - Ensemble model
- `google/owlv2-large-patch14` - 1.1B params (best accuracy)

**When to Use:**

- Need fast inference (<1s)
- General-purpose object detection
- Real-time or interactive applications
- Balanced speed/accuracy requirements

### 3. OWL-ViT v1 (Google)

**Best for:** Compatibility, baseline comparisons

```python
detector = mata.load("detect", "google/owlvit-base-patch32")
```

**Specifications:**

- Architecture: Original OWL-ViT (2022)
- Performance: Fast, lower accuracy than v2
- Prompt Format: List or space-dot
- Input Resolution: 768×768
- Strengths: Lightweight, proven baseline

**Available Models:**

- `google/owlvit-base-patch32` - 93M params
- `google/owlvit-base-patch16` - Higher resolution
- `google/owlvit-large-patch14` - Larger model

**When to Use:**

- Baseline comparisons with v2
- Compatibility with older code
- Resource-constrained environments

---

## Model Comparison

| Model                  | Params | Inference Speed | Accuracy   | Best For                                |
| ---------------------- | ------ | --------------- | ---------- | --------------------------------------- |
| **GroundingDINO-tiny** | 48M    | Slow (~2-5s)    | ⭐⭐⭐⭐⭐ | Complex scenes, small objects           |
| **OWL-ViT v2 base**    | 93M    | Fast (~0.5-1s)  | ⭐⭐⭐⭐   | General-purpose, speed-accuracy balance |
| **OWL-ViT v2 large**   | 1.1B   | Medium (~1-2s)  | ⭐⭐⭐⭐⭐ | Highest accuracy, more resources        |
| **OWL-ViT v1 base**    | 93M    | Fast (~0.5-1s)  | ⭐⭐⭐     | Baseline, compatibility                 |

**Benchmark (COCO-style evaluation):**

- GroundingDINO: ~52 mAP (zero-shot)
- OWL-ViT v2: ~45 mAP (zero-shot)
- OWL-ViT v1: ~38 mAP (zero-shot)

---

## Text Prompt Engineering

### Prompt Formats

**Space-dot separated** (GroundingDINO preferred):

```python
text_prompts = "cat . dog . person . car"
```

**List format** (OWL-ViT preferred):

```python
text_prompts = ["cat", "dog", "person", "car"]
```

**Conversion:** Both formats work with all models (MATA handles conversion).

### Effective Prompt Writing

**✅ Good Prompts:**

```python
# Specific, singular nouns
"dog . cat . bird"

# Add descriptors for specificity
"red car . blue truck . yellow bus"

# Use common object names
"person . vehicle . animal"

# Include variations
"chair . armchair . sofa . bench"
```

**❌ Poor Prompts:**

```python
# Plural forms (less accurate)
"dogs . cats . birds"

# Too generic (ambiguous)
"object . thing . item"

# Too long/complex (worse accuracy)
"a person wearing a red shirt standing next to a blue car"

# Misspellings (won't work)
"dag . kat . berd"
```

### Advanced Prompt Techniques

**1. Hierarchical Prompts:**

```python
# Broad to specific
text_prompts = "vehicle . car . sedan . sports car"
```

**2. Contextual Prompts:**

```python
# Scene-specific
text_prompts = "kitchen . refrigerator . stove . microwave"
```

**3. Attribute-Based:**

```python
# Color/size/state
text_prompts = "red apple . green apple . rotten apple"
```

**4. Empty Prompt (Detect All):**

```python
# Let model find all salient objects
text_prompts = ""  # or leave empty
```

### Prompt Best Practices

| Scenario                | Prompt Strategy            | Example                           |
| ----------------------- | -------------------------- | --------------------------------- |
| **General detection**   | Common singular nouns      | `"person . car . building"`       |
| **Specific objects**    | Add descriptive attributes | `"red car . wooden chair"`        |
| **Rare categories**     | Use precise terminology    | `"corgi . beagle . poodle"`       |
| **Scene understanding** | Context-aware prompts      | `"street . sidewalk . crosswalk"` |
| **Fine-grained**        | Include variations         | `"sedan . SUV . pickup truck"`    |

---

## Configuration & Parameters

### Detection Threshold

Control confidence filtering:

```python
# Load with default threshold (0.3)
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

# Custom threshold at load time
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny", threshold=0.5)

# Override at runtime
result = detector.predict(image, text_prompts="cat", threshold=0.7)
```

**Threshold Guidelines:**

- `0.2-0.3`: Permissive (more false positives, better recall)
- `0.3-0.5`: Balanced (default range)
- `0.5-0.7`: Strict (fewer false positives, may miss objects)
- `>0.7`: Very strict (high precision, low recall)

### Device Selection

```python
# Auto-select (CUDA if available)
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

# Force CPU
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny", device="cpu")

# Specific GPU
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny", device="cuda:1")
```

### Batch Processing

Process multiple images efficiently:

```python
images = [Image.open(f"image_{i}.jpg") for i in range(10)]

# Batch inference
results = detector.predict(images, text_prompts="cat . dog")

# Access individual results
for i, result in enumerate(results):
    print(f"Image {i}: {len(result.instances)} objects")
```

**Batch Size Recommendations:**

- GroundingDINO: 1-4 images (memory intensive)
- OWL-ViT: 4-16 images (more efficient)
- Adjust based on GPU memory

---

## Working with Results

### VisionResult Format

All zero-shot detectors return `VisionResult`:

```python
result = detector.predict(image, text_prompts="cat . dog")

# Result structure
print(result.task)        # "detect"
print(len(result.instances))  # Number of detections
print(result.meta)        # Metadata dict
```

### Instance Objects

Each detection is an `Instance`:

```python
for instance in result.instances:
    # Basic attributes
    print(instance.label)         # Class ID (int)
    print(instance.label_name)    # Class name (str)
    print(instance.score)         # Confidence (float, 0-1)

    # Bounding box (always present for detection)
    x1, y1, x2, y2 = instance.bbox  # xyxy format (absolute pixels)

    # Mask (None for pure detection, present for pipeline)
    if instance.mask is not None:
        binary_mask = instance.mask.to_binary()  # numpy array
        print(instance.area)                      # Mask area in pixels

    # Metadata
    print(instance.meta)  # Additional info dict
```

### Filtering & Sorting

```python
# Filter by score
high_conf = [inst for inst in result.instances if inst.score >= 0.7]

# Filter by label
cats_only = [inst for inst in result.instances if inst.label_name == "cat"]

# Sort by confidence
sorted_instances = sorted(result.instances, key=lambda x: x.score, reverse=True)

# Get top-k detections
top_5 = sorted_instances[:5]
```

### Serialization

```python
# Convert to JSON
json_str = result.to_json()

# Convert to dict
data = result.to_dict()

# Save to file
import json
with open("result.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## Visualization

### Using Built-in Visualizer

```python
from mata.visualization import visualize_segmentation

# Visualize detection (bbox-only)
vis = visualize_segmentation(result, image, backend="pil")
vis.show()

# Save visualization
visualize_segmentation(
    result,
    image,
    alpha=0.5,
    show_boxes=True,
    show_labels=True,
    show_scores=True,
    output_path="output.jpg"
)
```

### Custom Visualization

```python
from PIL import ImageDraw

def draw_detections(image, result):
    draw = ImageDraw.Draw(image)

    for instance in result.instances:
        # Draw bbox
        x1, y1, x2, y2 = instance.bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Draw label
        label = f"{instance.label_name}: {instance.score:.2f}"
        draw.text((x1, y1 - 15), label, fill='red')

    return image

vis_image = draw_detections(image.copy(), result)
vis_image.show()
```

---

## GroundingDINO→SAM Pipeline

### Pipeline Overview

Combine zero-shot detection with zero-shot segmentation for precise instance masks:

```
Text Prompts → GroundingDINO (bboxes) → SAM3 (masks) → VisionResult (bbox+mask)
```

### Basic Pipeline Usage

```python
# Load pipeline
pipeline = mata.load(
    "pipeline",
    detector_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base"
)

# Run pipeline
result = pipeline.predict(image, text_prompts="car . person")

# Access instances with both bbox and mask
for instance in result.instances:
    print(f"{instance.label_name}:")
    print(f"  BBox: {instance.bbox}")
    print(f"  Has Mask: {instance.mask is not None}")
    if instance.mask:
        print(f"  Mask Area: {instance.area} pixels")
```

### Pipeline Configuration

```python
pipeline = mata.load(
    "pipeline",
    detector_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base",
    detection_threshold=0.3,      # Filter weak detections
    segmentation_threshold=0.5,   # Filter low-quality masks
    device="cuda"                 # GPU device
)
```

### Pipeline Thresholds

**Detection Threshold:** Applied to GroundingDINO output

- Controls which bboxes are passed to SAM
- Higher = fewer objects segmented (faster)
- Lower = more objects segmented (slower)

**Segmentation Threshold:** Applied to SAM masks

- Controls mask quality filtering
- Higher = only high-quality masks kept
- Lower = more masks (some may be noisy)

### Pipeline Workflow

```python
# Step-by-step breakdown
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
segmenter = mata.load("segment", "facebook/sam-vit-base")

# 1. Detect objects with text prompts
detect_result = detector.predict(image, text_prompts="car . person")

# 2. For each detection, segment with SAM (conceptual)
for instance in detect_result.instances:
    bbox = instance.bbox
    # SAM segments using bbox as prompt
    # (Pipeline does this automatically)
```

### Pipeline Benefits

- ✅ **Precision:** Pixel-accurate masks instead of just bboxes
- ✅ **Zero-shot:** No training for either detection or segmentation
- ✅ **Flexible:** Any text prompt works
- ✅ **Unified:** Single `VisionResult` with multi-modal instances

---

## Performance Optimization

### Memory Management

```python
# Clear CUDA cache between runs
import torch
torch.cuda.empty_cache()

# Use smaller models for limited memory
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")  # 48M params

# Process images in smaller batches
batch_size = 4
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    results = detector.predict(batch, text_prompts="cat")
```

### Inference Speed

**Optimize for Speed:**

```python
# Use OWL-ViT v2 for faster inference
detector = mata.load("detect", "google/owlv2-base-patch16")

# Higher threshold = fewer detections = faster postprocessing
detector = mata.load("detect", "...", threshold=0.5)

# Smaller input images (resize before inference)
from PIL import Image
image = Image.open("large_image.jpg").resize((800, 600))
```

**Benchmark Results (single image, GPU):**

- GroundingDINO-tiny: ~2-3s
- OWL-ViT v2 base: ~0.5-1s
- OWL-ViT v1 base: ~0.4-0.8s

### Model Caching

Models are cached after first load:

```python
# First call: downloads and loads model (~10-30s)
detector1 = mata.load("detect", "IDEA-Research/grounding-dino-tiny")

# Second call: instant (cached)
detector2 = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
```

Cache location: `~/.cache/huggingface/hub/`

---

## Common Use Cases

### 1. Custom Object Detection

Detect objects not in standard datasets:

```python
# Detect specific products
detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
result = detector.predict(image, text_prompts="coca-cola bottle . pepsi bottle")

# Detect specific vehicles
result = detector.predict(image, text_prompts="tesla model 3 . ford f-150")
```

### 2. Scene Understanding

Analyze scene composition:

```python
# Indoor scene
result = detector.predict(
    image,
    text_prompts="sofa . table . chair . television . lamp"
)

# Outdoor scene
result = detector.predict(
    image,
    text_prompts="tree . building . road . sidewalk . car . person"
)
```

### 3. Object Counting

Count specific objects:

```python
result = detector.predict(image, text_prompts="person")
person_count = len(result.instances)
print(f"Detected {person_count} people")

# Count by category
result = detector.predict(image, text_prompts="car . truck . bus")
vehicles_by_type = {}
for instance in result.instances:
    label = instance.label_name
    vehicles_by_type[label] = vehicles_by_type.get(label, 0) + 1
```

### 4. Content Moderation

Detect inappropriate content:

```python
result = detector.predict(
    image,
    text_prompts="weapon . knife . gun . explicit content"
)

if len(result.instances) > 0:
    print("⚠ Potentially inappropriate content detected")
```

### 5. Inventory Management

Track products on shelves:

```python
result = detector.predict(
    shelf_image,
    text_prompts="cereal box . milk carton . bread loaf . egg carton"
)

inventory = {}
for instance in result.instances:
    product = instance.label_name
    inventory[product] = inventory.get(product, 0) + 1

print("Shelf inventory:", inventory)
```

---

## Troubleshooting

### Low Detection Accuracy

**Problem:** Model misses obvious objects or produces many false positives.

**Solutions:**

1. **Adjust threshold:**

   ```python
   # Lower for more detections (higher recall)
   result = detector.predict(image, text_prompts="cat", threshold=0.2)

   # Higher for fewer false positives (higher precision)
   result = detector.predict(image, text_prompts="cat", threshold=0.6)
   ```

2. **Improve prompts:**

   ```python
   # Bad: too generic
   text_prompts = "object"

   # Good: specific
   text_prompts = "cat . dog . person"
   ```

3. **Try different models:**
   ```python
   # GroundingDINO often better for rare objects
   detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
   ```

### Memory Errors

**Problem:** `CUDA out of memory` or system crashes.

**Solutions:**

1. **Use smaller models:**

   ```python
   detector = mata.load("detect", "IDEA-Research/grounding-dino-tiny")
   ```

2. **Reduce batch size:**

   ```python
   # Instead of batch=16
   results = detector.predict(images, text_prompts="cat")

   # Use batch=4
   results = []
   for i in range(0, len(images), 4):
       batch_results = detector.predict(images[i:i+4], text_prompts="cat")
       results.extend(batch_results)
   ```

3. **Use CPU:**
   ```python
   detector = mata.load("detect", "...", device="cpu")
   ```

### Slow Inference

**Problem:** Detection takes too long (>5s per image).

**Solutions:**

1. **Switch to OWL-ViT v2:**

   ```python
   detector = mata.load("detect", "google/owlv2-base-patch16")
   ```

2. **Resize images:**

   ```python
   image = Image.open("large.jpg").resize((800, 600))
   ```

3. **Use GPU:**
   ```python
   detector = mata.load("detect", "...", device="cuda")
   ```

### Empty Results

**Problem:** Model returns no detections.

**Solutions:**

1. **Lower threshold:**

   ```python
   result = detector.predict(image, text_prompts="cat", threshold=0.1)
   ```

2. **Verify prompts:**

   ```python
   # Ensure prompts match objects in image
   text_prompts = "car"  # not "cars"
   ```

3. **Check image:**
   ```python
   # Ensure image loaded correctly
   print(image.size, image.mode)
   ```

---

## API Reference

### Loading Models

```python
detector = mata.load(
    task="detect",
    model_id="IDEA-Research/grounding-dino-tiny",
    threshold=0.3,           # Confidence threshold (0-1)
    device="cuda",           # Device: "cuda", "cpu", "cuda:0", etc.
    cache_dir=None,          # Custom cache directory
)
```

### Prediction

```python
result = detector.predict(
    image,                   # PIL Image, numpy array, or path
    text_prompts,            # Space-dot string or list
    threshold=None,          # Override default threshold
)
```

**Returns:** `VisionResult` with `instances` list.

### Pipeline Loading

```python
pipeline = mata.load(
    task="pipeline",
    detector_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base",
    detection_threshold=0.3,
    segmentation_threshold=0.5,
    device="cuda",
)
```

### Pipeline Prediction

```python
result = pipeline.predict(
    image,
    text_prompts,
    detection_threshold=None,    # Override detector threshold
    segmentation_threshold=None, # Override SAM threshold
)
```

**Returns:** `VisionResult` with `instances` containing both `bbox` and `mask`.

---

## Examples

See full working examples in `examples/detect/` and `examples/segment/`:

- **`zeroshot_detection.py`** - Zero-shot detection examples
- **`grounding_sam_pipeline.py`** - Pipeline examples

Run with:

```bash
python examples/detect/zeroshot_detection.py
python examples/segment/grounding_sam_pipeline.py
```

---

## Further Reading

- [MATA Quick Reference](../QUICK_REFERENCE.md)
- [API Documentation](../README.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [VisionResult Specification](../docs/UNIVERSAL_RESULT_REFACTOR_PLAN.md)

**Research Papers:**

- [GroundingDINO](https://arxiv.org/abs/2303.05499) - Grounding DINO: Marrying DINO with Grounded Pre-Training
- [OWL-ViT v2](https://arxiv.org/abs/2306.09683) - Scaling Open-Vocabulary Object Detection
- [SAM](https://arxiv.org/abs/2304.02643) - Segment Anything Model

---

**Version:** 1.5.0  
**Last Updated:** February 4, 2026  
**Status:** Complete
