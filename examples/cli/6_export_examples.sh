#!/usr/bin/env bash
# mata export -- export a model to a portable format.
#
# Run from: examples/cli/
#   bash 6_export_examples.sh
#
# NOTE: mata export is a STUB in v1.9.5.
# Full ONNX and TorchScript export support is planned for v2.0.
# Running these commands prints a summary of the pending export;
# no model files are written until v2.0.

set -euo pipefail

# -----------------------------------------------
# Export to ONNX (v2.0 planned)
# -----------------------------------------------

# Export a HuggingFace detection model to ONNX
mata export detect facebook/detr-resnet-50 --format onnx

# Export a classification model to ONNX
mata export classify openai/clip-vit-base-patch32 --format onnx

# -----------------------------------------------
# Export to TorchScript (v2.0 planned)
# -----------------------------------------------

mata export detect facebook/detr-resnet-50 --format torchscript

# -----------------------------------------------
# Quantization options (v2.0 planned)
# -----------------------------------------------

# INT8 -- smallest model size, fastest CPU inference
mata export detect ./model.pt --format onnx --quantize int8

# FP16 -- half-precision, GPU-optimized
mata export detect ./model.pt --format onnx --quantize fp16

# -----------------------------------------------
# Custom output path (v2.0 planned)
# -----------------------------------------------

mata export detect facebook/detr-resnet-50 \
    --format onnx \
    --output ./exported_models/detr_r50.onnx
