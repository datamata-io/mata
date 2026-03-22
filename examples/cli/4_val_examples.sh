#!/usr/bin/env bash
# mata val -- evaluate a model against a labelled dataset.
#
# Run from: examples/cli/
#   bash 4_val_examples.sh
#
# Prerequisites:
#   pip install datamata
#   Dataset config: ../configs/coco.yaml
#
# IMPORTANT: Evaluation requires the dataset images and annotations to be
# present at the paths declared in the config YAML files.
# Edit the configs under ../configs/ to point to your local dataset copies.

set -euo pipefail

COCO_CFG="../configs/coco.yaml"
IMAGENET_CFG="../configs/imagenet.yaml"
DIODE_CFG="../configs/diode.yaml"
MODEL="facebook/detr-resnet-50"

# -----------------------------------------------
# Detection evaluation
# -----------------------------------------------

# Compute mAP on the validation split
mata val detect \
    --model "$MODEL" \
    --data "$COCO_CFG"

# Custom confidence and IoU thresholds
mata val detect \
    --model "$MODEL" \
    --data "$COCO_CFG" \
    --conf 0.01 --iou 0.5

# Save PR / F1 / confusion-matrix plots to a custom directory
mata val detect \
    --model "$MODEL" \
    --data "$COCO_CFG" \
    --plots --save-dir runs/val/detect/

# Output metrics as JSON (suitable for CI assertions)
mata val detect \
    --model "$MODEL" \
    --data "$COCO_CFG" \
    --json

# -----------------------------------------------
# Classification evaluation
# -----------------------------------------------

mata val classify \
    --model openai/clip-vit-base-patch32 \
    --data "$IMAGENET_CFG" \
    --json

# -----------------------------------------------
# Segmentation evaluation
# -----------------------------------------------

mata val segment \
    --model "$MODEL" \
    --data "$COCO_CFG" \
    --json

# -----------------------------------------------
# Depth evaluation
# -----------------------------------------------

mata val depth \
    --model depth-anything/Depth-Anything-V2-Small-hf \
    --data "$DIODE_CFG"

# -----------------------------------------------
# Custom dataset split
# -----------------------------------------------

# Evaluate on the test split instead of val
mata val detect \
    --model "$MODEL" \
    --data "$COCO_CFG" \
    --split test
