#!/usr/bin/env bash
# mata run -- one-shot inference across all supported tasks.
#
# Run from: examples/cli/
#   bash 2_run_examples.sh
#
# GPU-intensive examples (VLM, large models) are commented out;
# uncomment to use on a CUDA machine.

set -euo pipefail

IMAGE="../images/000000039769.jpg"
BARCODE_IMAGE="../images/sample_qr.png"

# =====================================================================
# Detection
# =====================================================================

# Standard HuggingFace detection model
mata run detect "$IMAGE" --model facebook/detr-resnet-50

# With a confidence threshold
mata run detect "$IMAGE" --model facebook/detr-resnet-50 --conf 0.5

# Save annotated output image to a custom directory
mata run detect "$IMAGE" \
    --model facebook/detr-resnet-50 \
    --save --save-dir runs/detect/

# Emit results as JSON to stdout
mata run detect "$IMAGE" --model facebook/detr-resnet-50 --json

# Zero-shot detection with text prompts (requires GroundingDINO weights)
# mata run detect "$IMAGE" \
#     --model IDEA-Research/grounding-dino-tiny \
#     --text "cat . dog . person"

# Via a config alias defined in .mata/models.yaml
# mata run detect "$IMAGE" --model my-detector

# =====================================================================
# Classification
# =====================================================================

# Top-5 CLIP predictions (CLIP requires --text labels for zero-shot classification)
mata run classify "$IMAGE" --model openai/clip-vit-base-patch32 --text "cat,dog,bird,car,person"

# Zero-shot with custom labels (comma-separated, no spaces)
mata run classify "$IMAGE" \
    --model openai/clip-vit-base-patch32 \
    --text "cat,dog,bird,car,person"

# JSON output
mata run classify "$IMAGE" --model openai/clip-vit-base-patch32 --text "cat,dog,bird,car,person" --json

# =====================================================================
# Segmentation
# =====================================================================

# Instance segmentation, save overlay image
mata run segment "$IMAGE" --model facebook/detr-resnet-50 --save

# =====================================================================
# Depth Estimation
# =====================================================================

# Estimate depth -- prints shape and value range
mata run depth "$IMAGE" --model depth-anything/Depth-Anything-V2-Small-hf

# Save a colorized (magma) depth map PNG
mata run depth "$IMAGE" \
    --model depth-anything/Depth-Anything-V2-Small-hf \
    --save --save-dir runs/depth/

# =====================================================================
# Feature Embedding
# =====================================================================

# Extract a CLIP embedding (prints shape and dtype)
mata run embed "$IMAGE" --model openai/clip-vit-base-patch32

# JSON array output -- useful for downstream scripts or similarity search
mata run embed "$IMAGE" --model openai/clip-vit-base-patch32 --json

# ONNX embedding model from a local file
# mata run embed "$IMAGE" --model ./osnet_x0_25.onnx --json

# =====================================================================
# Barcode and QR Code
# =====================================================================

# Scan with pyzbar (pip install datamata[barcode])
mata run barcode "$BARCODE_IMAGE" --model pyzbar

# Scan with zxing-cpp (pip install datamata[barcode-zxing])
# mata run barcode "$BARCODE_IMAGE" --model zxing

# JSON output -- includes type, data, and bounding box
mata run barcode "$BARCODE_IMAGE" --model pyzbar --json

# =====================================================================
# VLM (Vision-Language Model)
# =====================================================================
# Note: VLM models are large. First run downloads weights from HuggingFace.
# A GPU is recommended for acceptable performance.

# Image description
# mata run vlm "$IMAGE" \
#     --model Qwen/Qwen3-VL-2B-Instruct \
#     --prompt "Describe what you see in this image."

# Visual question answering
# mata run vlm "$IMAGE" \
#     --model Qwen/Qwen3-VL-2B-Instruct \
#     --prompt "How many objects are visible?"

# Medical imaging model (requires bfloat16 GPU)
# mata run vlm "$IMAGE" \
#     --model google/medgemma-1.5-4b-it \
#     --prompt "Describe any notable features in this image."
