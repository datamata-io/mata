# Supported Models

MATA supports any model loadable via HuggingFace Transformers, ONNX Runtime, TorchScript, or Torchvision. Below are the tested and recommended models for each task.

**Universal loading** — all models use the same API:

```python
import mata

adapter = mata.load("<task>", "<HuggingFace ID or file path>")
result  = mata.run("<task>", "image.jpg", model="<HuggingFace ID or file path>")
```

## Object Detection

### Transformer Models (HuggingFace)

| Model                | HuggingFace ID                         | Runtime            | mAP (COCO) | Speed (RTX 3080) | License    |
| -------------------- | -------------------------------------- | ------------------ | ---------- | ---------------- | ---------- |
| **RT-DETR v2 R18**   | `PekingU/rtdetr_v2_r18vd`              | PyTorch ✅ ONNX ✅ | 46.5       | ~50 FPS          | Apache 2.0 |
| **DETR ResNet-50**   | `facebook/detr-resnet-50`              | PyTorch ✅ ONNX ✅ | 42.0       | ~20 FPS          | Apache 2.0 |
| **Conditional DETR** | `microsoft/conditional-detr-resnet-50` | PyTorch ✅         | —          | —                | Apache 2.0 |
| **GroundingDINO**    | `IDEA-Research/grounding-dino-tiny`    | PyTorch ✅         | —          | —                | Apache 2.0 |
| **OWL-ViT v2**       | `google/owlv2-base-patch16`            | PyTorch ✅         | —          | —                | Apache 2.0 |

> GroundingDINO and OWL-ViT v2 are **zero-shot** detectors — they accept text prompts at runtime and do not require a fixed class list.

### CNN Models (Torchvision — Apache 2.0)

| Model               | Torchvision ID                           | mAP (COCO) | Speed (RTX 3080) |
| ------------------- | ---------------------------------------- | ---------- | ---------------- |
| **RetinaNet**       | `torchvision/retinanet_resnet50_fpn`     | 39.8       | ~40 FPS          |
| **Faster R-CNN V2** | `torchvision/fasterrcnn_resnet50_fpn_v2` | 42.2       | ~25 FPS          |
| **FCOS**            | `torchvision/fcos_resnet50_fpn`          | —          | ~30 FPS          |
| **SSD**             | `torchvision/ssd300_vgg16`               | —          | ~60 FPS          |

## Image Classification

| Model            | HuggingFace ID                           | Runtime                           | Description              |
| ---------------- | ---------------------------------------- | --------------------------------- | ------------------------ |
| **ResNet**       | `microsoft/resnet-50`                    | PyTorch ✅ ONNX ✅ TorchScript ✅ | Classic CNN              |
| **ViT**          | `google/vit-base-patch16-224`            | PyTorch ✅ ONNX ✅ TorchScript ✅ | Vision transformer       |
| **ConvNeXt**     | `facebook/convnext-base-224`             | PyTorch ✅ ONNX ✅                | Modern CNN               |
| **EfficientNet** | `google/efficientnet-b0`                 | PyTorch ✅ ONNX ✅                | Efficient scaling        |
| **Swin**         | `microsoft/swin-base-patch4-window7-224` | PyTorch ✅                        | Hierarchical transformer |
| **CLIP** (zero)  | `openai/clip-vit-base-patch32`           | PyTorch ✅                        | Zero-shot classification |

## Instance & Panoptic Segmentation

| Model           | HuggingFace ID                                 | Mode         | Description                 |
| --------------- | ---------------------------------------------- | ------------ | --------------------------- |
| **Mask2Former** | `facebook/mask2former-swin-tiny-coco-instance` | Instance     | High-quality instance masks |
| **Mask2Former** | `facebook/mask2former-swin-tiny-coco-panoptic` | Panoptic     | Instance + stuff regions    |
| **MaskFormer**  | `facebook/maskformer-swin-tiny-ade`            | Instance     | Unified segmentation        |
| **SAM** (zero)  | `facebook/sam-vit-base`                        | Prompt-based | Point/box prompts           |
| **SAM3** (zero) | `facebook/sam3`                                | Text prompts | 270K+ concepts              |

**SAM model variants:**

- `facebook/sam-vit-base` — Fast, good quality (recommended for prototyping)
- `facebook/sam-vit-large` — Slower, better quality
- `facebook/sam-vit-huge` — Slowest, best quality

> ⚠️ **Transformers v5.5.0 note:** `DetrForSegmentation` was refactored in transformers v5.5.0 (PR #41549). Validate panoptic segmentation with `pytest tests/test_segment_adapter.py -v -k detr` before bumping the transformers pin to `>=5.5.0`.

## Depth Estimation

| Model                 | HuggingFace ID                              | Description                      |
| --------------------- | ------------------------------------------- | -------------------------------- |
| **Depth Anything V2** | `depth-anything/Depth-Anything-V2-Small-hf` | Fast, good quality (recommended) |
| **Depth Anything V2** | `depth-anything/Depth-Anything-V2-Base-hf`  | Balanced speed/quality           |
| **Depth Anything V1** | `LiheYoung/depth-anything-small-hf`         | Original version                 |

## Vision-Language Models

| Model           | HuggingFace ID                        | Required kwargs            | Use Case               |
| --------------- | ------------------------------------- | -------------------------- | ---------------------- |
| **Qwen3-VL 2B** | `Qwen/Qwen3-VL-2B-Instruct`           | —                          | General VQA (dev)      |
| **MedGemma**    | `google/medgemma-1.5-4b-it`           | `dtype="bfloat16"`         | Medical imaging        |
| **LFM2.5-VL**   | `LiquidAI/LFM2.5-VL-1.6B`             | `dtype="bfloat16"`         | Lightweight general    |
| **SmolVLM**     | `HuggingFaceTB/SmolVLM-256M-Instruct` | —                          | Edge / mobile          |
| **Florence-2**  | `florence-community/Florence-2-large` | `trust_remote_code=True` *(optional)* | Grounding/captioning   |
| **PaliGemma 2** | `google/paligemma2-3b-pt-224`         | `dtype="bfloat16"` (gated) | Document understanding |
| **LLaVA-NeXT**  | `llava-hf/llava-v1.6-mistral-7b-hf`   | —                          | High-quality VQA       |
| **Moondream2**     | `vikhyatk/moondream2`                   | `trust_remote_code=True`                              | Tiny / fast                                                           |
| **Phi-3.5 Vision** | `microsoft/Phi-3.5-vision-instruct`     | `dtype="bfloat16"`, `trust_remote_code=True`          | Code / diagrams — ❌ Deferred (FlashAttention2 — Linux only)           |
| **InternVL2**      | `OpenGVLab/InternVL2-1B`                | `trust_remote_code=True`                              | ❌ Not supported (Accelerate meta-device incompatible)                 |

See [VLM Model Support](VLM_MODEL_SUPPORT.md) for the full compatibility table.

## OCR / Text Extraction

| Engine           | Model ID                           | Description                                                  |
| ---------------- | ---------------------------------- | ------------------------------------------------------------ |
| **EasyOCR**      | `easyocr`                          | 80+ languages, polygon bboxes                                |
| **PaddleOCR**    | `paddleocr`                        | Strong on non-Latin scripts — GPU ❌ unstable, CPU ✅ OK      |
| **Tesseract**    | `tesseract`                        | Classic open-source engine — ❌ unstable on Windows           |
| **GOT-OCR2**     | `stepfun-ai/GOT-OCR-2.0-hf`        | ⚠️ Known hallucinations — avoid until further notice         |
| **TrOCR**        | `microsoft/trocr-base-handwritten` | Single text-line crops                                       |
| **DeepSeek OCR** | `deepseek-ai/DeepSeek-OCR-2`       | ❌ Deferred — requires FlashAttention2 (Linux only)           |

## Embedding / ReID

| Model                        | Model ID                             | Output Dim | Description                              |
| ---------------------------- | ------------------------------------ | ---------- | ---------------------------------------- |
| **CLIP ViT-B/32**            | `openai/clip-vit-base-patch32`       | 512        | General-purpose                          |
| **OSNet**                    | `./osnet_x0_25.onnx`                 | 256        | Person ReID (ONNX)                       |
| **DINOv2**                   | `facebook/dinov2-small`              | 384        | Self-supervised features                 |
| **X-CLIP**                   | `microsoft/xclip-base-patch32`       | 512        | Video-language encoder                   |
| **Qwen3-VL-Embedding (2B)**  | `Qwen/Qwen3-VL-Embedding-2B`         | up to 4096 | Multimodal: text / image / video         |
| **Qwen3-VL-Embedding (8B)**  | `Qwen/Qwen3-VL-Embedding-8B`         | up to 4096 | Multimodal: text / image / video (large) |

## Barcode / QR

| Engine     | Model ID | Symbologies                                                       |
| ---------- | -------- | ----------------------------------------------------------------- |
| **pyzbar** | `pyzbar` | QR, EAN-13/8, UPC-A/E, Code 128/39/93, ITF, Codabar, PDF417, etc. |
| **zxing**  | `zxing`  | All pyzbar + Aztec, MaxiCode, RSS-14, broader coverage            |

## Runtime Compatibility

| Runtime          | Detection | Classification | Segmentation | Depth | VLM | OCR | Embed | Barcode |
| ---------------- | --------- | -------------- | ------------ | ----- | --- | --- | ----- | ------- |
| PyTorch          | ✅        | ✅             | ✅           | ✅    | ✅  | ✅  | ✅    | —       |
| ONNX Runtime     | ✅        | ✅             | —            | —     | —   | —   | ✅    | —       |
| TorchScript      | ✅        | ✅             | —            | —     | —   | —   | —     | —       |
| Torchvision      | ✅        | —              | —            | —     | —   | —   | —     | —       |
| Native (libzbar) | —         | —              | —            | —     | —   | —   | —     | ✅      |

**All models support:** single-image inference, PIL/path/numpy input, configurable thresholds, GPU and CPU execution, consistent output formats (xyxy bbox, RLE masks).
