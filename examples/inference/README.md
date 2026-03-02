# Inference Examples

General inference examples for MATA. Task-specific examples are organized in sibling directories:

- [`../classify/`](../classify/) — Image classification examples
- [`../detect/`](../detect/) — Object detection examples
- [`../segment/`](../segment/) — Segmentation examples
- [`../depth/`](../depth/) — Depth estimation examples
- [`../track/`](../track/) — Object tracking examples
- [`../vlm/`](../vlm/) — Vision-language model examples

## Quick Start

```python
import mata

# Detection
model = mata.load("detect", "facebook/detr-resnet-50")
result = model.predict("image.jpg")

# Classification
model = mata.load("classify", "google/vit-base-patch16-224")
result = model.predict("image.jpg")

# Segmentation
model = mata.load("segment", "facebook/mask2former-swin-small-coco-instance")
result = model.predict("image.jpg")
```

See the [QUICKSTART.md](../../QUICKSTART.md) for a full walkthrough.
