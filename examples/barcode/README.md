# Barcode & QR Code Examples

> Scan barcodes and QR codes from images using MATA's `barcode` task. _(Added in v1.9.3)_

## Quick Start

```python
import mata

result = mata.run("barcode", "image_with_barcode.jpg", model="pyzbar")
for bc in result.barcodes:
    print(f"[{bc.type}] {bc.data}")
```

## Installation

```bash
# pyzbar engine (default, MIT license)
pip install datamata[barcode]

# zxing-cpp engine (Apache 2.0, broader symbology support)
pip install datamata[barcode-zxing]

# both engines
pip install datamata[barcode-all]
```

## Examples

| File                             | What it shows                                                     |
| -------------------------------- | ----------------------------------------------------------------- |
| [`basic_scan.py`](basic_scan.py) | One-shot scan, load/reuse, engine switching, export, ROI pipeline |

## Running the Examples

```bash
# Run with the default sample image
python examples/barcode/basic_scan.py

# Run with your own image
python examples/barcode/basic_scan.py path/to/image.jpg
```

## Supported Engines

| Engine   | Install                   | License    | Symbologies                                                                                          |
| -------- | ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| `pyzbar` | `datamata[barcode]`       | MIT        | QR_CODE, EAN_13, EAN_8, UPC_A, UPC_E, CODE_128, CODE_39, CODE_93, ITF, CODABAR, DATA_MATRIX, PDF_417 |
| `zxing`  | `datamata[barcode-zxing]` | Apache 2.0 | All pyzbar types + Aztec, MaxiCode, and more                                                         |

## Supported Tasks

### One-Shot Scan

```python
result = mata.run("barcode", "image.jpg", model="pyzbar")
print(f"Found {len(result.barcodes)} barcode(s)")
```

### Load Once, Predict Many

```python
adapter = mata.load("barcode", "pyzbar")
for img_path in image_list:
    result = adapter.predict(img_path)
```

### Filter by Symbology

```python
result = mata.run("barcode", "image.jpg", model="pyzbar")

qr_codes  = result.filter_by_type("QR_CODE")
ean_codes = result.filter_by_type("EAN_13", "EAN_8")
```

### Export

```python
result.save("output/barcodes.json")   # JSON
result.save("output/barcodes.csv")    # CSV (data, type, score, bbox)
json_str = result.to_json(indent=2)   # JSON string
```

### ROI Pipeline (Graph API)

Detect objects first, then scan barcode crops — correlates barcode data with detected instances:

```python
from mata.nodes import Detect, ExtractROIs, Barcode, Fuse

graph = (
    Detect(model="facebook/detr-resnet-50", threshold=0.5)
    >> ExtractROIs()
    >> Barcode(provider_name="pyzbar")
    >> Fuse()
)

result = mata.infer(graph, image="image.jpg")
```

## Result Structure

```python
result.barcodes          # list[BarcodeRegion]

bc = result.barcodes[0]
bc.data        # str  — decoded payload (URL, number, text, etc.)
bc.type        # str  — symbology ("QR_CODE", "EAN_13", "CODE_128", ...)
bc.bbox        # tuple[float, float, float, float] | None — xyxy pixel coords
bc.score       # float — confidence (1.0 for algorithmic decoders)
bc.raw_bytes   # bytes | None — raw binary payload (binary QR codes)
```

## See Also

- [VLM & OCR examples](../vlm/) — text extraction with EasyOCR, PaddleOCR, Tesseract
- [Graph examples](../graph/) — multi-node pipelines including barcode correlation
- [MATA docs](../../docs/) — full documentation
