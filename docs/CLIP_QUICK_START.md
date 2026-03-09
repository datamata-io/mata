# CLIP Zero-Shot Image Classification - Quick Start Guide

**Version:** 1.5.0  
**Date:** February 5, 2026  
**Status:** ✅ Production Ready

## Overview

MATA now supports **zero-shot image classification** using CLIP (Contrastive Language-Image Pre-training). Unlike traditional classifiers that require predefined training classes, CLIP allows you to define categories at **runtime via text prompts**, enabling flexible classification without retraining.

### Key Features

- 🎯 **Zero-shot classification** - Define categories dynamically with text
- 🎨 **Template customization** - Single templates or ensemble averaging
- ⚙️ **Flexible filtering** - Combined threshold and top-k selection
- 📊 **Softmax calibration** - Probabilities or raw similarity scores
- 🚀 **Model flexibility** - Any HuggingFace CLIP-compatible model

---

## Quick Start (3 Lines)

```python
import mata

# Load CLIP classifier
classifier = mata.load("classify", "openai/clip-vit-base-patch32")

# Classify with custom categories
result = classifier.predict("cat.jpg", text_prompts=["cat", "dog", "bird"])

print(result.predictions[0].label_name)  # "cat"
```

---

## Installation

```bash
pip install datamata transformers torch
```

**Requirements:**

- Python 3.8+
- transformers >= 4.30.0
- torch >= 2.0.0

---

## Supported Models

### Tested Models (Recommended)

| Model ID                                | Size  | Speed | Accuracy  | Use Case                  |
| --------------------------------------- | ----- | ----- | --------- | ------------------------- |
| `openai/clip-vit-base-patch32`          | 150MB | Fast  | Good      | General purpose, balanced |
| `openai/clip-vit-large-patch14`         | 900MB | Slow  | Excellent | High accuracy required    |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 150MB | Fast  | Very Good | Open-source alternative   |

### Compatibility

✅ **Any HuggingFace model** compatible with `CLIPProcessor` and `CLIPModel` classes  
✅ All OpenAI CLIP variants (base-32, base-16, large-14, etc.)  
✅ LAION OpenCLIP models  
✅ Multilingual CLIP variants  
⚠️ **Untested models** may work but aren't officially supported

**Model Detection:** Models with `'clip'` in the model ID are automatically routed to the CLIP adapter.

---

## Usage Examples

### Example 1: Basic Classification

```python
import mata

# Load classifier
classifier = mata.load("classify", "openai/clip-vit-base-patch32")

# Define custom categories
text_prompts = ["cat", "dog", "bird", "fish", "horse"]

# Classify image
result = classifier.predict("image.jpg", text_prompts=text_prompts)

# Print results
for pred in result.predictions:
    print(f"{pred.label_name}: {pred.score:.3f}")

# Output:
# cat: 0.856
# dog: 0.102
# bird: 0.028
# fish: 0.012
# horse: 0.002
```

### Example 2: Custom Text Template

Templates control how prompts are formatted. Default: `"a photo of a {}"`.

```python
# Load with custom template
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template="this is a {}"
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog"])

# Behind the scenes:
# Prompts become: ["this is a cat", "this is a dog"]
```

**Why templates matter:**

- CLIP is sensitive to phrasing
- Templates improve zero-shot accuracy
- Different templates work better for different domains

### Example 3: Template Ensemble (Improved Accuracy)

Average predictions across multiple templates for **robust classification**.

```python
# Use predefined ensemble (6 templates)
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template="ensemble"  # Shortcut
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog"])

# Ensemble templates used:
# - "a photo of a {}"
# - "a picture of a {}"
# - "an image of a {}"
# - "a rendering of a {}"
# - "a cropped photo of a {}"
# - "a good photo of a {}"
```

**Custom ensemble:**

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template=["a photo of a {}", "a picture of a {}", "an image of a {}"]
)
```

### Example 4: Threshold Filtering

Filter predictions below a confidence threshold.

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    threshold=0.1  # Only return predictions >= 10%
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog", "bird", "fish", "horse"])

# Only high-confidence predictions returned
for pred in result.predictions:
    print(f"{pred.label_name}: {pred.score:.3f}")  # All >= 0.1
```

### Example 5: Top-K Selection

Return only the top-k most confident predictions.

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    top_k=3  # Return max 3 predictions
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog", "bird", "fish", "horse"])

print(len(result.predictions))  # 3 (top 3 only)
```

### Example 6: Combined Threshold + Top-K

Apply threshold first, then select top-k.

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    threshold=0.05,  # Filter low confidence
    top_k=3          # Return max 3 after filtering
)

result = classifier.predict("image.jpg", text_prompts=[...10 categories...])
# Returns at most 3 predictions, all with score >= 0.05
```

### Example 7: Softmax vs Raw Similarities

**Softmax (default):** Converts scores to probabilities (sum = 1.0)

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    use_softmax=True  # Default
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog", "bird"])

total = sum(p.score for p in result.predictions)
print(total)  # ~1.0 (probabilities)
```

**Raw similarities:** Return CLIP's native similarity scores (unbounded)

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    use_softmax=False
)

result = classifier.predict("image.jpg", text_prompts=["cat", "dog"], use_softmax=False)

# Scores show relative similarity, not probabilities
# Useful for comparing across different prompt sets
```

### Example 8: Runtime Parameter Override

Override instance settings per prediction.

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    top_k=5,
    threshold=0.0
)

# Override at inference time
result = classifier.predict(
    "image.jpg",
    text_prompts=["cat", "dog", "bird"],
    top_k=2,          # Override: return max 2
    threshold=0.2,    # Override: filter < 0.2
    use_softmax=False # Override: raw scores
)
```

---

## Template Engineering

### Predefined Template Sets

MATA provides 3 predefined template shortcuts:

```python
from mata.adapters.clip_adapter import TEMPLATE_SETS

# Basic (1 template)
TEMPLATE_SETS["basic"] = [
    "a photo of a {}"
]

# Ensemble (6 templates) - Recommended
TEMPLATE_SETS["ensemble"] = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a rendering of a {}",
    "a cropped photo of a {}",
    "a good photo of a {}"
]

# Detailed (18 templates) - Maximum robustness
TEMPLATE_SETS["detailed"] = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a black and white photo of a {}",
    # ... 15 more variations
]
```

**Usage:**

```python
classifier = mata.load("classify", "openai/clip-vit-base-patch32", template="ensemble")
```

### Template Best Practices

| Scenario                    | Recommended Template          |
| --------------------------- | ----------------------------- |
| General images              | `"a photo of a {}"` (default) |
| Product images              | `"a product photo of a {}"`   |
| Scenes/landscapes           | `"a landscape photo of a {}"` |
| Objects on white background | `"an image of a {}"`          |
| Artwork/drawings            | `"a drawing of a {}"`         |
| **High accuracy needed**    | `template="ensemble"`         |

### Custom Domain Templates

```python
# Example: Food classification
food_templates = [
    "a photo of a {}",
    "a delicious {}",
    "a plate of {}",
    "a serving of {}"
]

classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template=food_templates
)
```

---

## Similarity Scoring Guide

### When to use Softmax (default)

✅ **Use softmax when:**

- Comparing scores across categories for one image
- Interpreting scores as probabilities (confidence %)
- Need scores to sum to 1.0
- Standard classification use case

```python
result = classifier.predict("cat.jpg", text_prompts=["cat", "dog", "bird"])
# Scores: [0.85, 0.10, 0.05] → sum = 1.0
```

### When to use Raw Similarities

✅ **Use raw similarities when:**

- Comparing same image across different prompt sets
- Absolute similarity scores matter
- Debugging CLIP behavior
- Advanced use cases

```python
result = classifier.predict("cat.jpg", text_prompts=["cat", "dog"], use_softmax=False)
# Scores: [24.3, 18.7] → CLIP's native similarity
```

---

## Filtering Strategies

### Strategy 1: Top-K Only (Simple)

**Use case:** Always return N predictions

```python
classifier = mata.load("classify", "openai/clip-vit-base-patch32", top_k=3)
# Always returns 3 predictions (unless fewer classes)
```

### Strategy 2: Threshold Only (Confidence-based)

**Use case:** Only confident predictions, variable count

```python
classifier = mata.load("classify", "openai/clip-vit-base-patch32", threshold=0.1)
# Returns 0 to N predictions, all >= 0.1
```

### Strategy 3: Combined (Recommended)

**Use case:** Confident predictions, bounded count

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    threshold=0.05,  # Min confidence: 5%
    top_k=5          # Max predictions: 5
)
# Returns 0 to 5 predictions, all >= 0.05
```

**Order of operations:**

1. Apply threshold filtering
2. Sort by score (descending)
3. Select top-k

---

## Batch Processing

**Note:** CLIP adapter processes one image at a time (same prompts for all images).

```python
import mata

classifier = mata.load("classify", "openai/clip-vit-base-patch32")

images = ["cat.jpg", "dog.jpg", "bird.jpg"]
text_prompts = ["cat", "dog", "bird", "fish"]

results = []
for image_path in images:
    result = classifier.predict(image_path, text_prompts=text_prompts)
    results.append(result)

    # Print top prediction for each image
    print(f"{image_path}: {result.predictions[0].label_name}")
```

**Future:** Batch optimization with text embedding caching planned for Phase 2.

---

## When to Use CLIP vs Standard Classification

### Use CLIP (Zero-Shot) When:

✅ Categories change frequently  
✅ Need custom taxonomy not in pretrained models  
✅ Rapid prototyping without training data  
✅ Flexible, runtime-defined categories  
✅ Open-vocabulary classification

**Examples:**

- E-commerce: categorize products with custom labels
- Content moderation: detect specific themes/concepts
- Research: explore arbitrary semantic categories

### Use Standard Classification When:

✅ Fixed, well-defined categories (ImageNet, etc.)  
✅ Maximum accuracy on known classes  
✅ Faster inference (no text encoding)  
✅ Categories align with pretrained models

**Examples:**

- ImageNet-1k classification
- Specific domain with fixed classes (medical, satellite, etc.)

---

## Performance Considerations

### Inference Speed

| Model                  | Image Size | Speed (CPU) | Speed (GPU) |
| ---------------------- | ---------- | ----------- | ----------- |
| clip-vit-base-patch32  | 224×224    | ~200ms      | ~20ms       |
| clip-vit-large-patch14 | 224×224    | ~600ms      | ~50ms       |

**Factors affecting speed:**

- Number of text prompts (linear scaling)
- Template ensemble size (N templates = N× text encoding)
- Image resolution (preprocessing)
- Device (CPU vs GPU)

### Accuracy Tips

1. **Use template ensembles** for +2-5% accuracy boost
2. **Tune templates** to your domain (products, scenes, etc.)
3. **Larger models** (large-patch14) for +3-7% accuracy
4. **More prompts** doesn't always help (be specific)
5. **Threshold tuning** to filter noise

---

## Common Use Cases

### Use Case 1: Product Categorization

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template="a product photo of a {}"
)

categories = ["shirt", "pants", "shoes", "dress", "jacket", "accessories"]
result = classifier.predict("product.jpg", text_prompts=categories, top_k=1)

print(f"Product category: {result.predictions[0].label_name}")
```

### Use Case 2: Scene Classification

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    template="ensemble"
)

scenes = ["beach", "mountain", "city", "forest", "desert", "ocean", "countryside"]
result = classifier.predict("photo.jpg", text_prompts=scenes, top_k=3)

print("Top 3 scenes:")
for pred in result.predictions:
    print(f"  - {pred.label_name}: {pred.score*100:.1f}%")
```

### Use Case 3: Content Filtering

```python
classifier = mata.load(
    "classify",
    "openai/clip-vit-base-patch32",
    threshold=0.3  # High confidence required
)

content_types = ["safe content", "unsafe content", "questionable content"]
result = classifier.predict("user_upload.jpg", text_prompts=content_types)

if result.predictions and result.predictions[0].label_name == "unsafe content":
    print("⚠️ Content flagged for review")
```

---

## Troubleshooting

### Issue: Low confidence scores

**Solutions:**

- ✅ Use template ensemble (`template="ensemble"`)
- ✅ Try domain-specific templates
- ✅ Use larger model (clip-vit-large-patch14)
- ✅ Refine text prompts (be more specific)

### Issue: Wrong predictions

**Solutions:**

- ✅ Check text prompts are clear and unambiguous
- ✅ Use more specific prompts (e.g., "golden retriever" vs "dog")
- ✅ Try different templates
- ✅ Increase prompt set size to include close alternatives

### Issue: Slow inference

**Solutions:**

- ✅ Use GPU (`device="cuda"`)
- ✅ Reduce number of text prompts
- ✅ Use single template instead of ensemble
- ✅ Use base model instead of large model

### Issue: Import errors

```python
# Error: transformers not installed
pip install transformers torch

# Error: Model not found
# Verify model ID is correct and accessible on HuggingFace
```

---

## API Reference Summary

### Loading

```python
mata.load(
    task="classify",
    model="openai/clip-vit-base-patch32",  # Any CLIP-compatible model
    device="auto",                          # "cuda", "cpu", or "auto"
    top_k=5,                               # Max predictions to return
    threshold=0.0,                         # Min confidence threshold
    template="a photo of a {}",            # String, list, or shortcut
    use_softmax=True                       # Apply softmax to scores
)
```

### Prediction

```python
classifier.predict(
    image,                    # str path, PIL Image, or numpy array
    text_prompts,            # List of categories or comma-separated string
    top_k=None,              # Override instance top_k
    threshold=None,          # Override instance threshold
    use_softmax=None         # Override instance use_softmax
)
```

### Result Structure

```python
ClassifyResult(
    predictions=[
        Classification(
            label=0,
            score=0.856,
            label_name="cat"
        ),
        ...
    ],
    meta={
        "model_id": "openai/clip-vit-base-patch32",
        "num_classes": 3,
        "template_type": "single" or "ensemble",
        "num_templates": 1,
        "use_softmax": True,
        "threshold": 0.0,
        "top_k": 5
    }
)
```

---

## Next Steps

- 📖 See [examples/classify/clip_zeroshot.py](../../examples/classify/clip_zeroshot.py) for complete examples
- 🔧 Read [CLIP_IMPLEMENTATION_COMPLETE.md](CLIP_IMPLEMENTATION_COMPLETE.md) for technical details
- 🚀 Explore [ZEROSHOT_DETECTION_GUIDE.md](ZEROSHOT_DETECTION_GUIDE.md) for zero-shot detection
- 📚 Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for all MATA tasks

---

## Contributing

Found a CLIP model that works well? Have template recommendations? Open an issue or PR!

**Tested models list:** Help us grow the compatibility matrix by reporting your results.

---

**Last Updated:** February 5, 2026  
**MATA Version:** 1.5.0+  
**License:** MIT
