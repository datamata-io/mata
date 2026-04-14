---
title: "Recognition"
description: "Build gallery-based recognition flows using embeddings, reference sets, and top-k matching."
sidebar_position: 4
---

# Recognition Guide

> **v1.9.6** — Embedding extraction, gallery-based identity matching, and graph recognition pipelines.

---

## Overview

MATA's recognition system combines feature embeddings with nearest-neighbour search to resolve the question _"Who or what is this?"_ without retraining. The pipeline has three stages:

1. **Embed** — extract a compact vector representation of each image region
2. **Gallery** — an in-memory store of labeled reference embeddings
3. **Match** — cosine similarity search against the gallery to return ranked identity labels

This works for any recognition problem: person re-identification, product recognition, vehicle matching, face ID, logo detection, and more.

---

## Quick Start

```python
import mata
import numpy as np
from mata.recognition import Gallery

# 1. Build a gallery from reference images
gallery = Gallery(similarity_thresh=0.5)
for name, img_path in [("alice", "alice.jpg"), ("bob", "bob.jpg")]:
    result = mata.run("embed", img_path, model="openai/clip-vit-base-patch32")
    gallery.add(name, result.embedding)

# 2. Match a query image
query = mata.run("embed", "query.jpg", model="openai/clip-vit-base-patch32")
matches = gallery.search(query.embedding, top_k=1)
print(f"Identity: {matches[0].label} ({matches[0].similarity:.2%})")
```

Or use the one-liner convenience API:

```python
result = mata.run("recognize", "query.jpg",
                  gallery=gallery,
                  model="openai/clip-vit-base-patch32",
                  top_k=3)
for m in result.matches:
    print(f"  {m.label}: {m.similarity:.2%}")
```

---

## Embedding Extraction

### `mata.run("embed", ...)`

```python
# Single image → EmbedResult
result = mata.run("embed", "image.jpg", model="openai/clip-vit-base-patch32")

print(result.embeddings.shape)   # (1, 512)
print(result.embedding.shape)    # (512,) — convenience: first row
print(result.dim)                 # 512
```

### Batch crops

```python
crops = [np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8) for _ in range(5)]
result = mata.run("embed", crops, model="openai/clip-vit-base-patch32")
print(result.embeddings.shape)   # (5, 512)
```

### Video / text embeddings (v1.9.6+)

```python
# Video clip embedding (X-CLIP) — pass pre-extracted frames
frames = [frame1, frame2, frame3]
emb = mata.run("embed", frames,
               model="microsoft/xclip-base-patch32")

# Video file embedding (auto-extracts frames)
emb = mata.run("embed", "video.mp4",
               model="microsoft/xclip-base-patch32")

# Text query embedding — input must be None
emb = mata.run("embed", None,
               model="microsoft/xclip-base-patch32",
               text="person running")

# Mixed-modal: image + text → single joint embedding (Qwen3-VL-Embedding only)
# The text is NOT a VLM prompt — it is co-embedded with the image.
# The model attends to both modalities and produces one vector that
# captures the image content steered by the text context.
emb = mata.run("embed", "photo.jpg",
               model="Qwen/Qwen3-VL-Embedding-2B",
               text="a red truck on a highway", dtype="bfloat16")
```

> **Mixed-modal vs VLM:** `mata.run("embed", image, text=...)` produces a
> **vector** (no text generation). The `text=` parameter is semantic context
> that steers the embedding, not an instruction to describe the image.
> For text generation, use `mata.run("vlm", image, prompt="...")` instead.

### Pre-loaded adapter

```python
embedder = mata.load("embed", "openai/clip-vit-base-patch32")

# Reuse across many images — avoids reloading the model
for img_path in image_paths:
    result = embedder.predict(img_path)
    embeddings.append(result.embedding)
```

### ONNX embedder (CPU-efficient)

```python
embedder = mata.load("embed", "./osnet_x0_25.onnx")
result = embedder.predict("person.jpg")
```

---

## EmbedResult

`EmbedResult` is a frozen dataclass returned by all embed operations.

```python
from mata import EmbedResult

result: EmbedResult = mata.run("embed", "image.jpg", model="openai/clip-vit-base-patch32")

result.embeddings     # np.ndarray, shape (N, D), float32, L2-normalised
result.embedding      # np.ndarray, shape (D,) — first embedding (convenience)
result.dim            # int — embedding dimensionality D
result.meta           # dict — optional metadata (model name, etc.)
```

**Serialization:**

```python
result.to_json()                              # JSON string
result.to_dict()                              # dict with list embeddings
result.save("embeddings.json")               # save as JSON
result.save("embeddings.npz")               # save as compressed numpy
result2 = EmbedResult.from_json(json_str)    # round-trip
```

---

## Gallery

`Gallery` is an in-memory embedding store backed by numpy. It supports galleries up to ~50,000 entries with brute-force cosine search (no external dependencies).

### Building a gallery

```python
from mata.recognition import Gallery

gallery = Gallery(similarity_thresh=0.5)

# Add one entry at a time
gallery.add("alice", alice_embedding)   # returns insertion index int
gallery.add("alice", alice_embedding2)  # multiple embeddings per label OK

# Add many entries at once
gallery.add_many(
    labels=["bob", "carol", "dave"],
    embeddings=embeddings_matrix,        # (3, D) float32
)
```

### Searching

```python
matches = gallery.search(query_embedding, top_k=3, threshold=0.6)
# Returns list[GalleryMatch], sorted by similarity descending

for m in matches:
    print(f"  {m.label}: {m.similarity:.3f} (index={m.index})")
```

```python
# Batch search
queries = np.stack([emb1, emb2, emb3])  # (3, D)
batch_matches = gallery.search_batch(queries, top_k=1)
# Returns list[list[GalleryMatch]] — one list per query
```

### Removing entries

```python
n_removed = gallery.remove("alice")  # removes all entries for label "alice"
```

### Properties

```python
len(gallery)               # total number of stored embeddings
gallery.size               # same as len()
gallery.labels             # list of all label strings (with duplicates)
gallery.unique_labels      # sorted list of distinct labels
```

### Persistence

```python
# Save
gallery.save("gallery.npz")          # compressed npz, no pickle

# Load
gallery2 = Gallery.load("gallery.npz")

# JSON serialization
d = gallery.to_dict()
gallery3 = Gallery.from_dict(d)
```

### Serialization formats

```python
# NPZ (recommended — compact, allow_pickle=False for security)
gallery.save("gallery.npz")

# JSON (human-readable, larger)
json_str = gallery.to_json()
gallery4 = Gallery.from_json(json_str)
```

---

## One-Liner Recognition — `mata.run("recognize", ...)`

For single-image recognition without building a graph:

```python
result = mata.run(
    "recognize", "query.jpg",
    gallery=gallery,                         # required: pre-populated Gallery
    model="openai/clip-vit-base-patch32",    # embed model
    top_k=3,                                 # max matches to return
    threshold=0.5,                           # optional: override gallery default
)

# result is a Matches artifact
for m in result.matches:
    print(f"  {m.label}: {m.similarity:.3f}")
```

---

## Graph-Based Recognition Pipeline

For recognition within a multi-step pipeline (e.g., detect persons, then re-identify each one):

```python
import mata
from mata.recognition import Gallery
from mata.nodes import Detect, ExtractROIs, Embed, GalleryMatchNode

# Pre-populate gallery
gallery = Gallery.load("persons.npz")

result = mata.infer(
    image="crowd.jpg",
    graph=[
        Detect(using="detector", out="detections"),
        ExtractROIs(dets="detections", out="crops"),
        Embed(using="embedder", src="crops", out="embeddings"),
        GalleryMatchNode(gallery=gallery, src="embeddings", out="matches"),
    ],
    providers={
        "detector": mata.load("detect", "facebook/detr-resnet-50"),
        "embedder": mata.load("embed", "openai/clip-vit-base-patch32"),
    }
)

matches = result["matches"]   # Matches artifact
for m in matches.entries:
    print(f"  Instance {m.instance_id}: {m.label} ({m.similarity:.2%})")
```

### Node contracts

| Node               | Input                      | Output                |
| ------------------ | -------------------------- | --------------------- |
| `Embed`            | `ROIs` or `Image` artifact | `Embeddings` artifact |
| `GalleryMatchNode` | `Embeddings` artifact      | `Matches` artifact    |

### Graph node parameters

**`Embed` node:**

```python
Embed(
    using="embedder",     # provider key
    src="crops",          # input artifact key (default: "rois")
    out="embeddings",     # output artifact key (default: "embeddings")
    normalize=True,       # L2-normalize output (default: True)
)
```

**`GalleryMatchNode`:**

```python
GalleryMatchNode(
    gallery=gallery,           # Gallery instance (required)
    src="embeddings",          # input artifact key (default: "embeddings")
    out="matches",             # output artifact key (default: "matches")
    top_k=1,                   # max matches per embedding
    threshold=None,            # override gallery default similarity threshold
)
```

---

## Common Patterns

### Person Re-Identification

```python
# Build gallery from enrollment images
gallery = Gallery(similarity_thresh=0.6)
for person_id, image_path in enrollment_data:
    result = mata.run("embed", image_path, model="openai/clip-vit-base-patch32")
    gallery.add(person_id, result.embedding)
gallery.save("persons.npz")

# Query from video frame crops
for crop in detected_person_crops:
    result = mata.run("recognize", crop,
                      gallery=gallery,
                      model="openai/clip-vit-base-patch32")
    if result.matches:
        print(f"Identified: {result.matches[0].label}")
    else:
        print("Unknown person")
```

### Product Recognition

```python
from mata.recognition import Gallery

# One-time setup: build gallery from product catalog
gallery = Gallery(similarity_thresh=0.7)
for product_id, product_image in catalog.items():
    result = mata.run("embed", product_image, model="openai/clip-vit-base-patch32")
    gallery.add(product_id, result.embedding)
gallery.save("product_catalog.npz")

# Runtime: identify product in scene
detected = mata.run("detect", "shelf.jpg", model="facebook/detr-resnet-50")
for det in detected.instances:
    crop = det.crop("shelf.jpg")
    result = mata.run("recognize", crop,
                      gallery=gallery,
                      model="openai/clip-vit-base-patch32",
                      threshold=0.65)
    if result.matches:
        print(f"Product: {result.matches[0].label} ({result.matches[0].similarity:.1%})")
```

### Vehicle Re-ID with Cross-Camera Tracking

```python
from mata.recognition import Gallery
from mata.trackers import ReIDBridge

# Per-camera gallery + cross-camera bridge
gallery = Gallery.load("vehicles.npz")
bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-entrance")

results = mata.track(
    "rtsp://entrance/stream",
    model="facebook/detr-resnet-50",
    tracker="botsort",
    reid_model="openai/clip-vit-base-patch32",  # auto-enables BotSort ReID
    reid_bridge=bridge,
    stream=True,
)
for result in results:
    for inst in result.instances:
        if inst.track_id is not None:
            print(f"Track {inst.track_id} at {inst.bbox}")
```

### Similarity Search on a Corpus

```python
import numpy as np
from mata.recognition import Gallery

# Index all images
gallery = Gallery()
for path in image_paths:
    result = mata.run("embed", path, model="openai/clip-vit-base-patch32")
    gallery.add(path, result.embedding)   # use path as label

# Search top-5 similar images
query = mata.run("embed", "query.jpg", model="openai/clip-vit-base-patch32")
matches = gallery.search(query.embedding, top_k=5, threshold=0.0)
for m in matches:
    print(f"  {m.label}: {m.similarity:.3f}")
```

---

## Recommended Models

| Use Case                 | Model                           | Runtime | Dim |
| ------------------------ | ------------------------------- | ------- | --- |
| General-purpose (images) | `openai/clip-vit-base-patch32`  | PyTorch | 512 |
| General-purpose (ONNX)   | `./osnet_x0_25.onnx`            | ONNX    | 512 |
| High-accuracy            | `openai/clip-vit-large-patch14` | PyTorch | 768 |
| Video semantic search    | `microsoft/xclip-base-patch32`  | PyTorch | 512 |
| Lightweight edge         | `./osnet_x0_25_msmt17.onnx`     | ONNX    | 512 |

> **X-CLIP / transformers compatibility note:** In `transformers >= 5.2`, `XCLIPProcessor.__call__` dispatches inputs by iterating `get_attributes()`, which returns `['image_processor', 'tokenizer']`. The `videos=` keyword is routed to `video_processor` — not in that list — so it is **silently dropped** and `pixel_values` is never populated, causing `AttributeError: 'NoneType' object has no attribute 'shape'` inside the model forward pass.
>
> **Workaround (used internally by MATA's `XCLIPAdapter`):** pass a flat list of PIL `Image` objects via `images=` instead of `videos=`. The `image_processor` slot is backed by `VideoMAEImageProcessor`, which correctly produces `pixel_values` of shape `(1, n_frames, C, H, W)` when given a list of frames:
>
> ```python
> from transformers import XCLIPProcessor
> from PIL import Image as PILImage
>
> processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
>
> # ❌ Broken in transformers >= 5.2 — videos= is silently dropped
> inputs = processor(videos=pil_frames, return_tensors="pt")
>
> # ✅ Correct workaround — split text and video into two calls, then merge
> text_inputs = processor(text=["eating spaghetti"], return_tensors="pt", padding=True)
> video_inputs = processor(images=pil_frames, return_tensors="pt")
> inputs = {**text_inputs, **video_inputs}
> # inputs["pixel_values"].shape → (1, 8, 3, 224, 224) ✓
> ```
>
> MATA's `XCLIPAdapter` handles this automatically — if you use `mata.load("embed", "microsoft/xclip-base-patch32")` you are unaffected.

---

## Performance Tips

### Batch crops for throughput

```python
# ✅ Good — batch crops before embedding
crops = [det.crop(image) for det in detections]
result = mata.run("embed", crops, model="openai/clip-vit-base-patch32")
# result.embeddings shape: (N, 512)

# ❌ Slow — one forward pass per crop
for det in detections:
    result = mata.run("embed", det.crop(image), model="openai/clip-vit-base-patch32")
```

### Pre-load the adapter

```python
# ✅ Load once, predict many
embedder = mata.load("embed", "openai/clip-vit-base-patch32")
for frame in video_frames:
    result = embedder.predict(frame)
```

### Gallery size limits

- Brute-force numpy cosine search is efficient up to ~50,000 entries
- Beyond 50k entries, consider migrating to FAISS (drop-in: see below)

### FAISS migration path

When your gallery grows beyond ~50,000 entries:

```python
import faiss
import numpy as np

# Export gallery embeddings
d = gallery.to_dict()
matrix = np.array(d["embeddings"], dtype=np.float32)    # (N, D)
labels = d["labels"]

# Build FAISS index
index = faiss.IndexFlatIP(matrix.shape[1])   # Inner product = cosine (L2-normed)
index.add(matrix)

# Search
query = result.embedding.reshape(1, -1).astype(np.float32)
D, I = index.search(query, k=5)
for dist, idx in zip(D[0], I[0]):
    print(f"  {labels[idx]}: {dist:.3f}")
```

---

## Relationship to Existing Features

### vs. CLIP Zero-Shot Classification

|                                | CLIP Zero-Shot          | Gallery Matching           |
| ------------------------------ | ----------------------- | -------------------------- |
| Requires labels at query time? | ✅ (text prompts)       | ❌ enrolled in gallery     |
| Closed-set categories?         | ❌ open vocabulary      | ✅ gallery labels          |
| Threshold-based rejection?     | ❌                      | ✅ via `similarity_thresh` |
| Re-ID across sessions?         | ❌                      | ✅ via gallery.npz         |
| Use case                       | "Is this a cat or dog?" | "Is this Alice or Bob?"    |

```python
# Zero-shot (no gallery needed)
result = mata.run("classify", "image.jpg",
    model="openai/clip-vit-base-patch32",
    text_prompts=["cat", "dog", "bird"])

# Gallery-based (enrollment required)
result = mata.run("recognize", "image.jpg",
    gallery=gallery,
    model="openai/clip-vit-base-patch32")
```

### vs. Tracking ReID

Tracking ReID (v1.9.2) maintains cross-frame identity _within a video_ using the same embed backbone. Supplying `reid_model=...` on a BotSort tracker is enough to activate the appearance-matching path. Gallery recognition operates _across sessions_ against a persistent `.npz` store.

You can combine both: use tracking for intra-video identity and gallery lookup for cross-video identity resolution.

```python
# This works: tracking ReID + gallery lookup are complementary
results = mata.track("video.mp4",
    model="facebook/detr-resnet-50",
    reid_model="openai/clip-vit-base-patch32")

for result in results:
    for inst in result.instances:
        if inst.embedding is not None:
            matches = gallery.search(inst.embedding, top_k=1)
            if matches:
                print(f"Track {inst.track_id} = {matches[0].label}")
```

---

## CLI

```bash
# Recognize identity in an image
mata recognize query.jpg \
    --gallery persons.npz \
    --model openai/clip-vit-base-patch32 \
    --top-k 3 \
    --threshold 0.5

# JSON output
mata recognize query.jpg --gallery persons.npz --json
```

---

## API Reference

| Symbol                       | Module                | Description                                           |
| ---------------------------- | --------------------- | ----------------------------------------------------- |
| `mata.run("embed", ...)`     | `mata`                | Extract embeddings (returns `EmbedResult`)            |
| `mata.run("recognize", ...)` | `mata`                | Embed + gallery search (returns `Matches`)            |
| `mata.load("embed", ...)`    | `mata`                | Load `EmbedAdapter` for reuse                         |
| `EmbedResult`                | `mata`                | Frozen dataclass: `.embeddings`, `.embedding`, `.dim` |
| `Gallery`                    | `mata.recognition`    | Embedding store with cosine search                    |
| `GalleryMatch`               | `mata.recognition`    | Single match: `.label`, `.similarity`, `.index`       |
| `Embed`                      | `mata.nodes`          | Graph node: ROIs/Image → Embeddings                   |
| `GalleryMatchNode`           | `mata.nodes`          | Graph node: Embeddings → Matches                      |
| `Embeddings`                 | `mata.core.artifacts` | Artifact: `(N, D)` vectors + instance IDs             |
| `Matches`                    | `mata.core.artifacts` | Artifact: match entries + labels                      |

---

## See Also

- [CLIP Quick Start](./CLIP_QUICK_START.md) — zero-shot classification with CLIP
- [Tracking Guide](./TRACKING_GUIDE.md) — multi-object tracking with ReID
- [Graph Cookbook](./GRAPH_COOKBOOK.md) — pipeline composition recipes
- [Valkey Guide](./VALKEY_GUIDE.md) — cross-camera embedding sharing
- [Graph API Reference](./GRAPH_API_REFERENCE.md) — full node reference
