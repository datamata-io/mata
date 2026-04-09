---
title: "Barcode"
description: "Detect and decode barcodes in MATA pipelines with guidance for common formats and image quality issues."
sidebar_position: 8
---

# MATA Barcode & QR Code Guide

> **Version**: 1.9.3 | **Last Updated**: April 3, 2026

MATA's `barcode` task decodes QR codes, linear barcodes, and 2D symbologies from images. Two backends are supported — **pyzbar** (based on libzbar) and **zxing** (based on zxing-cpp) — through a unified adapter interface.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Backend Comparison](#backend-comparison)
4. [Symbology Reference](#symbology-reference)
5. [API Reference](#api-reference)
6. [BarcodeResult & BarcodeRegion](#barcoderesult--barcoderegion)
7. [Graph Integration](#graph-integration)
8. [Export & Saving](#export--saving)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### pyzbar (recommended for most use cases)

```bash
pip install pyzbar

# Linux: install libzbar shared library
sudo apt-get install libzbar0          # Debian/Ubuntu
sudo yum install zbar                  # RHEL/CentOS

# macOS
brew install zbar
```

### zxing-cpp (broader format coverage)

```bash
pip install zxing-cpp
```

Verify:

```python
import mata
print(mata.load("barcode", "pyzbar"))   # PyzbarAdapter
print(mata.load("barcode", "zxing"))    # ZxingAdapter
```

---

## Quick Start

### One-Liner API

```python
import mata

# Decode with pyzbar (default)
result = mata.run("barcode", "image.jpg", model="pyzbar")

# Decode with zxing-cpp
result = mata.run("barcode", "image.jpg", model="zxing")

# Inspect results
for region in result.barcodes:
    print(region.data)      # decoded string
    print(region.type)      # "QR_CODE", "EAN_13", etc.
    print(region.bbox)      # (x1, y1, x2, y2) bounding box in pixels
```

### Persistent Adapter

```python
reader = mata.load("barcode", "pyzbar")
result = reader.predict(image)   # accepts NumPy array or PIL Image

# Process multiple images efficiently
for frame in video_frames:
    result = reader.predict(frame)
    if result.barcodes:
        print(f"Found {len(result.barcodes)} codes")
```

### Filter by Symbology

```python
# pyzbar: filter to QR codes and EAN-13 only
reader = mata.load("barcode", "pyzbar", symbols=["QR_CODE", "EAN_13"])

# zxing: filter to DataMatrix and Aztec
reader = mata.load("barcode", "zxing", formats=["DataMatrix", "Aztec"])
```

---

## Backend Comparison

| Feature                  | pyzbar            | zxing-cpp                      |
| ------------------------ | ----------------- | ------------------------------ |
| **Underlying library**   | libzbar           | zxing-cpp                      |
| **Linear codes**         | ✅ Full           | ✅ Full                        |
| **QR Code**              | ✅                | ✅                             |
| **Data Matrix**          | ❌                | ✅                             |
| **Aztec**                | ❌                | ✅                             |
| **MaxiCode**             | ❌                | ✅                             |
| **PDF417**               | ✅                | ✅                             |
| **Polygon corners**      | ❌ (bbox only)    | ❌ (bbox only)                 |
| **Multi-code per image** | ✅                | ✅                             |
| **System dependency**    | `libzbar0`        | None (pure Python wheel)       |
| **Best for**             | General QR/linear | 2D symbologies, no system deps |

**Recommendation:** Use pyzbar for QR codes and retail barcodes. Use zxing-cpp for Data Matrix, Aztec, or when a native library dependency is undesirable.

---

## Symbology Reference

### pyzbar Symbols

| Symbol Name   | Description                          |
| ------------- | ------------------------------------ |
| `QR_CODE`     | QR Code (all versions)               |
| `EAN_13`      | European Article Number 13           |
| `EAN_8`       | European Article Number 8            |
| `UPC_A`       | Universal Product Code A             |
| `UPC_E`       | Universal Product Code E             |
| `CODE_128`    | Code 128 (high-density alphanumeric) |
| `CODE_39`     | Code 39 (alphanumeric + special)     |
| `CODE_93`     | Code 93 (compact alphanumeric)       |
| `ITF`         | Interleaved 2-of-5                   |
| `DATABAR`     | GS1 DataBar (RSS-14)                 |
| `DATABAR_EXP` | GS1 DataBar Expanded                 |
| `CODABAR`     | Codabar (medical/library)            |
| `PDF_417`     | PDF417 (2D stacked linear)           |

### zxing-cpp Formats

Superset of pyzbar plus:

| Format        | Description                   |
| ------------- | ----------------------------- |
| `DataMatrix`  | Data Matrix (2D matrix code)  |
| `Aztec`       | Aztec Code (concentric rings) |
| `MaxiCode`    | MaxiCode (UPS logistics)      |
| `PDF417`      | PDF417                        |
| `MicroQRCode` | Micro QR Code                 |
| `RMQRCode`    | Rectangular Micro QR Code     |

---

## API Reference

### `mata.load("barcode", backend, **kwargs)`

| Parameter | Type                | Description                                    |
| --------- | ------------------- | ---------------------------------------------- |
| `backend` | `str`               | `"pyzbar"` or `"zxing"`                        |
| `symbols` | `list[str] \| None` | _pyzbar only_ — restrict to listed symbologies |
| `formats` | `list[str] \| None` | _zxing only_ — restrict to listed format names |

Returns a `PyzbarAdapter` or `ZxingAdapter` instance.

### `mata.run("barcode", input, model=..., **kwargs)`

| Parameter | Type  | Description                                                    |
| --------- | ----- | -------------------------------------------------------------- |
| `input`   | `str` | Path to image file                                             |
| `model`   | `str` | Backend: `"pyzbar"` or `"zxing"`; defaults to registry default |

Returns a `BarcodeResult`.

### `adapter.predict(image)`

| Parameter | Type                      | Description                     |
| --------- | ------------------------- | ------------------------------- |
| `image`   | `np.ndarray \| PIL.Image` | Input image (any colour format) |

Returns a `BarcodeResult`.

---

## BarcodeResult & BarcodeRegion

### `BarcodeResult`

Frozen result container, importable from `mata.core.types` (or the short form `from mata.core import BarcodeResult`).

| Attribute  | Type                  | Description                                        |
| ---------- | --------------------- | -------------------------------------------------- |
| `barcodes` | `list[BarcodeRegion]` | All detected and decoded barcode regions           |
| `meta`     | `dict`                | Provenance metadata (model name, image size, etc.) |

**Methods:**

| Method       | Returns | Description                       |
| ------------ | ------- | --------------------------------- |
| `to_dict()`  | `dict`  | Serialize to JSON-compatible dict |
| `to_json()`  | `str`   | Serialize to JSON string          |
| `save(path)` | `None`  | Save as `.json` or `.csv` file    |

### `BarcodeRegion`

Frozen dataclass for a single detected barcode.

| Attribute   | Type                                        | Description                                                                                       |
| ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `data`      | `str`                                       | Decoded text / payload                                                                            |
| `type`      | `str`                                       | Symbology name (e.g. `"QR_CODE"`, `"EAN_13"`)                                                     |
| `bbox`      | `tuple[float, float, float, float] \| None` | Bounding box in xyxy pixel coords `(x1, y1, x2, y2)`. `None` if decoder provides no spatial info. |
| `score`     | `float`                                     | Confidence score in [0.0, 1.0]. Algorithmic decoders always return 1.0.                           |
| `raw_bytes` | `bytes \| None`                             | Raw undecoded bytes (useful for binary payloads)                                                  |

---

## Graph Integration

### BarcodeNode

Use the `Barcode` node to decode barcodes within a `Graph` pipeline:

```python
import mata
from mata.nodes import Barcode
from mata.core.graph import Graph

graph = (
    Graph("barcode_pipeline")
    .then(Barcode(using="pyzbar", out="codes"))
)

result = graph.run(image="package.jpg")
for entry in result["codes"].entries:
    print(f"{entry.type}: {entry.data}")
```

### Detect → Crop → Barcode Pipeline

Pair with detection to focus on regions of interest before decoding:

```python
from mata.nodes import Detect, ExtractROIs, Barcode
from mata.core.graph import Graph

graph = (
    Graph("label_scanner")
    .then(Detect(using="detector", classes=["label", "package"]))
    .then(ExtractROIs(src="detections", out="rois"))
    .then(Barcode(using="pyzbar", src="rois", out="codes"))
)

result = graph.run(
    image="warehouse_shelf.jpg",
    providers={"detector": mata.load("detect", "facebook/detr-resnet-50")},
)
print(result["codes"])
```

### BarcodeData Artifact

The graph-layer artifact wrapping a `BarcodeResult`:

```python
from mata.core.artifacts import BarcodeData

bd = result["codes"]            # BarcodeData
for entry in bd.entries:
    print(entry.data)           # decoded payload
    print(entry.type)           # symbology name
    print(entry.confidence)     # score in [0.0, 1.0]
    print(entry.bbox)           # (x1, y1, x2, y2) or None
```

---

## Export & Saving

```python
result = mata.run("barcode", "image.jpg", model="pyzbar")

# Save as JSON
result.save("output/codes.json")

# Get dict for downstream processing
data = result.to_dict()
print(data["barcodes"][0]["data"])

# Iterate and print all codes
for region in result.barcodes:
    print(f"[{region.type}] {region.data}")
    print(f"  bbox: {region.bbox}")
```

---

## Troubleshooting

### `ImportError: Unable to find zbar shared library`

pyzbar requires `libzbar0` at runtime. Install the system package:

```bash
# Ubuntu / Debian
sudo apt-get install libzbar0

# macOS
brew install zbar

# Docker
RUN apt-get install -y libzbar0
```

### `ImportError: No module named 'zxing_cpp'`

```bash
pip install zxing-cpp
```

### No barcodes detected

1. Ensure the image is sharp and well-lit.
2. Increase image resolution — barcodes smaller than ~30 px on the short axis are unreliable.
3. Try the alternate backend (`pyzbar` ↔ `zxing`).
4. Use the Detect → Crop → Barcode pipeline to isolate regions of interest before decoding.

### Binary payloads

Some QR codes encode raw bytes rather than UTF-8 text. Access the raw payload:

```python
for region in result.barcodes:
    if region.raw_bytes:
        payload = region.raw_bytes   # bytes
```

---

**Version:** 1.9.3
**Date:** April 3, 2026
**Status:** ✅ Production Ready
